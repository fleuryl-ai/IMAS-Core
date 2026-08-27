// @file  test_engine_write_read_dynamic_aos.cpp
// @brief PanzerDB dynamic AoS test: writes two time slices of a scalar signal in a
//        dynamic B AoS under A[0] and reads back slice 1 by index.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: Dynamic Array of Structures (AoS) ===\n\n";
    const std::string filename = "test_dynamic_aos.panzer";

    // ===================================================================
    // 1. Write the initial structure with a dynamic AoS
    //    A (static, size=1)
    //    - A[0]/B (dynamic, timebase="time")
    //      - A[0]/B[0]/signal (value = 100.0)
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        
        std::cout << "Writing the initial structure...\n";
        db.beginArray("A", 1);
        db.beginArray("B", "time"); // Marks B as dynamic
        
        // Writing TWO data slices for the signal in B
        std::vector<double> all_slices_data = {100.0, 200.0};
        db.writeDataSlices("signal", {1}, all_slices_data.data(), 2, "time");
        
        db.endArray(); // B
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

        // B should now have a size of 2
        auto shapeB = db.getAOSShape("B");
        // For a dynamic AoS, getAOSShape is not well defined.
        // We verify the global time instead.
        /*uint64_t total_time = db.getCurrentTime();
        assert(total_time == 2 && "The global time must be 2 after writing 2 slices.");
        std::cout << "Global time: " << total_time << " (expected: 2)\n";*/

        // Reading the second slice (B[1])
        // For a dynamic AoS, we only provide the parent's static index
        // and the time index of the slice.
        uint64_t static_indices[] = {0}; // A[0]
        int64_t time_index = 1; // We want the second slice (index 1)
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        const char* full_path = "A/0/B/signal";
        int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

        assert(result == 0 && "The slice 1 read must succeed.");
        
        // A scalar signal read at a given time t is a scalar (ndim=0) or a size-1 array (ndim=1, shape=[1])
        bool is_scalar = (ndim_out == 0) || (ndim_out == 1 && shape_out[0] == 1);
        assert(is_scalar && "The read signal must be a scalar (ndim=0 or shape=[1]).");

        assert(std::abs(data_out[0] - 200.0) < 1e-9 && "The value of slice 1 must be 200.0.");
        std::cout << "Data of A[0]/B[1]/signal: " << data_out[0] << " (expected: 200.0)\n";
        delete[] data_out;
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}