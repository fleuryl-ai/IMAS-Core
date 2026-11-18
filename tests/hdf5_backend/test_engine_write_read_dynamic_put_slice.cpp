// @file  test_engine_write_read_dynamic_put_slice.cpp
// @brief PanzerDB dynamic data test: writes static and dynamic signals at root and in
//        nested AoS (A, B), validates them, then appends a slice and validates again.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    std::cout << "=== TEST PanzerDB: Dynamic Data Put() then Slice() ===\n\n";
    const std::string filename = "test_dynamic_put_slice.panzer";

    // Cleanup before the test
    if (fs::exists(filename)) {
        fs::remove(filename);
    }

    // ===================================================================
    // 1. Initial write of dynamic data (simulates a put())
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        std::cout << "Step 1: Initial write of 2 slices with writeDataSlices()...\n";

        // Writing a 1D static signal
        std::vector<double> static_signal_data = {42.0, 43.0, 44.0, 45.0};
        db.writeData("static_signal", {4}, static_signal_data.data(), static_signal_data.size());

        db.beginArray("A", 1);

        // Writing a 1D static signal IN AoS A
        std::vector<double> static_in_aos_data = {101.0, 102.0, 103.0};
        db.writeData("static_in_aos", {3}, static_in_aos_data.data(), static_in_aos_data.size());

        // Writing an AoS B inside A
        db.beginArray("B", 2);
        for (int i = 0; i < 2; ++i) {
            // Writing a 2D static signal in B
            std::vector<double> matrix_data(2 * 3); // 2x3 matrix
            for(size_t j = 0; j < matrix_data.size(); ++j) matrix_data[j] = 1000.0 + i * 100.0 + j;
            db.writeData("matrix", {2, 3}, matrix_data.data(), matrix_data.size());
            db.incrementArrayIndex();
        }

        db.endArray(); // B

        double val0 = 10.0;
        db.writeDataSlices("signal", {1}, &val0, 1, "time");
        std::cout << "  Wrote signal[0] = " << val0 << "\n";

        double val1 = 20.0;
        db.writeDataSlices("signal", {1}, &val1, 1, "time");
        std::cout << "  Wrote signal[1]: " << val1 << "\n";

        db.endArray();
        db.close();
        std::cout << "PanzerDB closed after the initial write.\n\n";
    }

    // ===================================================================
    // 2. Validation after the initial write
    // ===================================================================
    {
        PanzerDB db_read(filename, PanzerDB::OpenMode::READ);
        std::cout << "Step 2: Validation after the initial write...\n";

        /*uint64_t total_time = db_read.getCurrentTime();
        assert(total_time == 2 && "The global time must be 2 after 2 writes.");
        std::cout << "  Global time read: " << total_time << " (expected: 2)\n";*/

        uint64_t static_indices[] = {0};
        int64_t time_index = 0;
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        
        // Validation of the static signal
        // CORRECTION: To read a value at the root, n_static_levels must be 0.
        // time_index must be -1 for static data.
        const char* full_path_static_signal = "static_signal";
        int result = db_read.readDataByIndex(full_path_static_signal, -1, &ndim_out, shape_out, &data_out);
        assert(result == 0 && ndim_out == 1 && shape_out[0] == 4 && std::abs(data_out[0] - 42.0) < 1e-9 && std::abs(data_out[3] - 45.0) < 1e-9 && "The static signal must be correct.");
        std::cout << "  Read static_signal[0]: " << data_out[0] << ", static_signal[3]: " << data_out[3] << " (expected: 42.0, 45.0)\n";
        delete[] data_out;

        // Validation of the static signal IN AoS A[0]
        const char* full_path_static_in_aos = "A/0/static_in_aos";
        result = db_read.readDataByIndex(full_path_static_in_aos, -1, &ndim_out, shape_out, &data_out);
        assert(result == 0 && "The read of the static signal in the AoS must succeed.");
        assert(ndim_out == 1 && shape_out[0] == 3 && "The shape of the static signal in the AoS is incorrect.");
        assert(std::abs(data_out[0] - 101.0) < 1e-9 && std::abs(data_out[2] - 103.0) < 1e-9 && "The data of the static signal in the AoS are incorrect.");
        std::cout << "  Read static_in_aos[0]: " << data_out[0] << ", static_in_aos[2]: " << data_out[2] << " (expected: 101.0, 103.0)\n";
        delete[] data_out;

        // Validation of the 2D static signal in A[0]/B[1]
        uint64_t nested_static_indices[] = {0, 1}; // A[0], B[1]
        const char* full_path_matrix = "A/0/B/1/matrix";
        result = db_read.readDataByIndex(full_path_matrix, -1, &ndim_out, shape_out, &data_out);
        assert(result == 0 && "The read of the 2D signal in the nested AoS must succeed.");
        assert(ndim_out == 2 && "The 2D signal dimension must be 2.");
        assert(shape_out[0] == 2 && shape_out[1] == 3 && "The 2D signal shape is incorrect.");
        // Expected value for A[0]/B[1]: 1000.0 + 1*100.0 + j
        assert(std::abs(data_out[0] - 1100.0) < 1e-9 && "First value of the matrix incorrect.");
        assert(std::abs(data_out[5] - 1105.0) < 1e-9 && "Last value of the matrix incorrect.");
        std::cout << "  Read matrix[0][0] in A[0]/B[1]: " << data_out[0] << " (expected: 1100.0)\n";
        delete[] data_out;
        //const char* full_path_matrix_last = "A/0/B/1/matrix";
        //result = db_read.readDataByIndex(full_path_matrix_last, -1, &ndim_out, shape_out, &data_out);
        // Validation of the dynamic signal, slice 0
        time_index = 0;
        const char* full_path_signal_0 = "A/0/signal";
        result = db_read.readDataByIndex(full_path_signal_0, time_index, &ndim_out, shape_out, &data_out);
        assert(result == 0 && std::abs(data_out[0] - 10.0) < 1e-9 && "Slice 0 must be 10.0");
        std::cout << "  Read signal[0]: " << data_out[0] << " (expected: 10.0)\n";
        delete[] data_out;

        // Validation of the dynamic signal, slice 1
        time_index = 1;
        const char* full_path_signal_1 = "A/0/signal";
        result = db_read.readDataByIndex(full_path_signal_1, time_index, &ndim_out, shape_out, &data_out);
        assert(result == 0 && std::abs(data_out[0] - 20.0) < 1e-9 && "Slice 1 must be 20.0");
        std::cout << "  Read signal[1]: " << data_out[0] << " (expected: 20.0)\n";
        delete[] data_out;

        db_read.close();
        std::cout << "Initial validation passed.\n\n";
    }

    // ===================================================================
    // 3. Appending new slices in APPEND mode
    // ===================================================================
    {
        // KEY CHANGE: Use OpenMode::APPEND instead of OpenMode::WRITE
        PanzerDB db_append(filename, PanzerDB::OpenMode::APPEND, true);
        std::cout << "Step 3: Appending an extra slice with writeDataSlices()...\n";

        db_append.beginArray("A", 1);

        double val2 = 30.0;
        db_append.writeDataSlices("signal", {1}, &val2, 1, "time");
        std::cout << "  Wrote signal[2] = " << val2 << "\n";

        db_append.endArray();
        db_append.close();
        std::cout << "PanzerDB closed after appending the slices.\n\n";
    }

    // ===================================================================
    // 4. Validation after appending the slices
    // ===================================================================
    {
        PanzerDB db_final_read(filename, PanzerDB::OpenMode::READ);
        std::cout << "Step 4: Validation after appending the slices...\n";

        /*uint64_t total_time = db_final_read.getCurrentTime();
        assert(total_time == 3 && "The global time must be 3 after 3 writes.");
        std::cout << "  Global time read: " << total_time << " (expected: 3)\n";*/

        uint64_t static_indices[] = {0};
        int64_t time_index = 2;
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        const char* full_path_signal = "A/0/signal";
        int result = db_final_read.readDataByIndex(full_path_signal, time_index, &ndim_out, shape_out, &data_out);
        assert(result == 0 && std::abs(data_out[0] - 30.0) < 1e-9 && "Slice 2 must be 30.0");
        std::cout << "  Read signal[2]: " << data_out[0] << " (expected: 30.0)\n";
        delete[] data_out;

        std::cout << "Final validation passed.\n\n";
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}