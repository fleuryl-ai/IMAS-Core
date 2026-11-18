// @file  test_engine_write_read_nested_dynamic_aos.cpp
// @brief PanzerDB nested dynamic AoS test: writes a static A(3) with dynamic B AoS
//        holding "signal" and "signal_2", then reads slices using local time indices.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: Nested Dynamic Array of Structures (AoS) ===\n\n";
    const std::string filename = "test_nested_dynamic_aos.panzer";

    // ===================================================================
    // 1. Write a complex structure
    //    A (static, size=3)
    //    - A[0]/B (dynamic, timebase="time")
    //      - 3 slices of "signal" and "signal_2"
    //    - A[1]/B (dynamic, timebase="time")
    //      - 5 slices of "signal" and "signal_2"
    //    - A[2]/B (dynamic, timebase="time")
    //      - 2 slices of "signal" and "signal_2"
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Writing the nested structure...\n";
        db.beginArray("A", 3);

        // --- Writing into A[0] ---
        db.beginArray("B", "time"); // Marks B as dynamic
        std::vector<double> data_A0_s1 = {100.0, 101.0, 102.0};
        std::vector<double> data_A0_s2 = {150.0, 151.0, 152.0};
        db.writeDataSlices("signal", {1}, data_A0_s1.data(), 3, "");
        db.writeDataSlices("signal_2", {1}, data_A0_s2.data(), 3, "");
        db.endArray(); // B
        db.incrementArrayIndex(); // Moves to A[1]

        // --- Writing into A[1] ---
        db.beginArray("B", "time"); // Marks B as dynamic
        std::vector<double> data_A1_s1 = {200.0, 201.0, 202.0, 203.0, 204.0};
        std::vector<double> data_A1_s2 = {250.0, 251.0, 252.0, 253.0, 254.0};
        db.writeDataSlices("signal", {1}, data_A1_s1.data(), 5, "");
        db.writeDataSlices("signal_2", {1}, data_A1_s2.data(), 5, "");
        db.endArray(); // B
        db.incrementArrayIndex(); // Moves to A[2]

        // --- Writing into A[2] ---
        db.beginArray("B", "time"); // Marks B as dynamic
        std::vector<double> data_A2_s1 = {300.0, 301.0};
        std::vector<double> data_A2_s2 = {350.0, 351.0};
        db.writeDataSlices("signal", {1}, data_A2_s1.data(), 2, "");
        db.writeDataSlices("signal_2", {1}, data_A2_s2.data(), 2, "");
        db.endArray(); // B
        db.incrementArrayIndex(); // Increments A (even though we are at the end)

        db.endArray(); // A
        db.close();
        std::cout << "File closed.\n\n";
    }

    // ===================================================================
    // 2. Reopen and verify
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        std::cout << "Verifying the structure...\n";

        // The global time is no longer a single, simple concept.
        // We validate the data based on the time_index local to each dynamic AoS.

        // Reading the 4th slice (index 3) of "signal" in A[1]
        // The time_index is local to the A/1/B AoS.
        // The 4th slice of 'signal' in A[1]/B has a local time_index of 3.
        {
            uint64_t static_indices[] = {1}; // A[1]
            int64_t time_index = 3; // The local index of the 4th slice is 3.
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/1/B/signal";
            int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0 && "The slice read must succeed.");
            
            // A scalar signal read at a given time t is a scalar (ndim=0) or a size-1 array (ndim=1, shape=[1])
            bool is_scalar = (ndim_out == 0) || (ndim_out == 1 && shape_out[0] == 1);
            assert(is_scalar && "The read signal must be a scalar (ndim=0 or shape=[1]).");

            assert(std::abs(data_out[0] - 203.0) < 1e-9 && "The value of the slice must be 203.0.");
            std::cout << "  [OK] Data of A[1]/B[slice_3]/signal: " << data_out[0] << " (expected: 203.0)\n";
            delete[] data_out;
        }

        // Reading the 1st slice (index 0) of "signal_2" in A[2]
        // The time_index is local to the A/2/B AoS.
        // CORRECTION: With the correct alignment logic, 'signal' and 'signal_2' are aligned.
        // So 'signal_2' starts at index 0, just like 'signal'.
        {
            uint64_t static_indices[] = {2}; // A[2]
            int64_t time_index = 0; // Alignment: signal_2 starts at 0
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/2/B/signal_2";
            int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0 && "The slice read must succeed.");
            
            // A scalar signal read at a given time t is a scalar (ndim=0) or a size-1 array (ndim=1, shape=[1])
            bool is_scalar = (ndim_out == 0) || (ndim_out == 1 && shape_out[0] == 1);
            assert(is_scalar && "The read signal must be a scalar (ndim=0 or shape=[1]).");

            assert(std::abs(data_out[0] - 350.0) < 1e-9 && "The value of the slice must be 350.0.");
            std::cout << "  [OK] Data of A[2]/B[slice_0]/signal_2: " << data_out[0] << " (expected: 350.0)\n";
            delete[] data_out;
        }
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}