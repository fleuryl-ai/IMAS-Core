// @file  test_engine_readdata_by_index_ultradeep.cpp
// @brief PanzerDB readDataByIndex test: writes an ultra-deep A/B/C/D/E structure
//        with many small tensors in E[0] and reads one of them back by index.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>

int main() {
    std::cout << "=== TEST PanzerDB: readDataByIndex (ultra deep & complex) ===\n\n";
    const std::string filename = "test_read_by_index_ultra_deep.panzer";

    // ===================================================================
    // 1. Write a complex structure
    //    A (size=1)
    //    - A[0]/B (size=2)
    //      - A[0]/B[0]/C (size=1, but empty)
    //      - A[0]/B[1]/C (size=2)
    //          - A[0]/B[1]/C[0]/D (size=1)
    //              - A[0]/B[1]/C[0]/D[0]/E (size=1)
    //                  - tensor_1 (shape 5)
    //                  - tensor_2 (shape 12)
    //                  ... (10 tensors in total)
    //          - A[0]/B[1]/C[1]/D (size=1)
    //              - A[0]/B[1]/C[1]/D[0]/E (size=1)
    //                  - tensor_11 (shape 20)
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Writing the ultra-complex structure...\n";
        db.beginArray("A", 1);
        db.beginArray("B", 2); // B has a size of 2

        // --- Iterating over B ---
        // A[0]/B[0]
        db.beginArray("C", 1);
        db.endArray(); // C
        db.incrementArrayIndex(); // Moves to B[1]

        // A[0]/B[1]
        db.beginArray("C", 2);
        // --- Iterating over C ---
        // A[0]/B[1]/C[0]
        db.beginArray("D", 1);
        db.beginArray("E", 1);
        for (int i = 1; i <= 10; ++i) {
            std::vector<double> data(i);
            std::iota(data.begin(), data.end(), 100.0 + i);
            db.writeData("tensor_" + std::to_string(i), {(size_t)i}, data.data(), data.size());
            // No incrementArrayIndex here since we write several signals in the same E[0] instance
        }
        db.endArray(); // E
        db.endArray(); // D
        db.incrementArrayIndex(); // Moves to C[1]

        // A[0]/B[1]/C[1]
        db.beginArray("D", 1);
        db.beginArray("E", 1);
        std::vector<double> data_11(20);
        std::iota(data_11.begin(), data_11.end(), 200.0);
        db.writeData("tensor_11", {20}, data_11.data(), data_11.size());
        db.endArray(); // E
        db.endArray(); // D
        db.incrementArrayIndex(); // Increments C (even though we are at the end)

        db.endArray(); // C
        db.incrementArrayIndex(); // Increments B (even though we are at the end)

        db.endArray(); // B
        db.incrementArrayIndex(); // Increments A (even though we are at the end)

        db.endArray(); // A

        db.close();
        std::cout << "File closed.\n\n";
    }

    // ===================================================================
    // 2. Reopen and read a few signals
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        // --- Validation of "tensor_7" in A[0]/B[1]/C[0]/D[0]/E[0] ---
        std::cout << "Reading 'tensor_7' in A[0]/B[1]/C[0]/D[0]/E[0]...\n";
        {
            uint64_t static_indices[] = {0, 1, 0, 0, 0}; // A[0], B[1], C[0], D[0], E[0]
            int64_t time_index = -1;
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/0/B/1/C/0/D/0/E/0/tensor_7";
            int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0);
            assert(ndim_out == 1);
            assert(shape_out[0] == 7);
            assert(std::abs(data_out[0] - 107.0) < 1e-9);
            assert(std::abs(data_out[6] - 113.0) < 1e-9);
            std::cout << "Read 'tensor_7' data successfully.\n";
            delete[] data_out;
        }
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}