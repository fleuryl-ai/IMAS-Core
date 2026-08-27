// @file  test_engine_readdata_by_index_verydeep.cpp
// @brief PanzerDB readDataByIndex test: writes a deep A[0]/B[1]/C[0]/D[0]/tensor
//        structure (skipping B[0]) with a 2x3 tensor and reads it back by index.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>

int main() {
    std::cout << "=== TEST PanzerDB: readDataByIndex (very deep) ===\n\n";
    const std::string filename = "test_read_by_index_very_deep.panzer";

    // ===================================================================
    // 1. Write an A[0]/B[1]/C[0]/D[0]/tensor structure
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Writing A[0]/B[1]/C[0]/D[0]/tensor...\n";
        db.beginArray("A", 1);
        db.beginArray("B", 2); // B has a size of 2

        // We "skip" the B[0] index by incrementing manually
        db.incrementArrayIndex(); // Moves to B[1]

        // We now write into B[1]
        db.beginArray("C", 1);
        db.beginArray("D", 1);

        std::vector<double> data_to_write(2 * 3); // 2D 2x3 tensor
        std::iota(data_to_write.begin(), data_to_write.end(), 300.0);
        db.writeData("tensor", {2, 3}, data_to_write.data(), data_to_write.size());

        db.endArray(); // D
        db.endArray(); // C

        db.incrementArrayIndex(); // Increments B (even though we are at the end)
        db.endArray(); // B
        db.incrementArrayIndex(); // Increments A (even though we are at the end)
        db.endArray(); // A

        db.close();
        std::cout << "File closed.\n\n";
    }

    // ===================================================================
    // 2. Reopen and read with readDataByIndex
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        std::cout << "Reading with readDataByIndex...\n";

        uint64_t static_indices[] = {0, 1, 0, 0}; // A[0], B[1], C[0], D[0]
        int64_t time_index = -1; // Static
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        const char* full_path = "A/0/B/1/C/0/D/0/tensor";
        int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

        assert(result == 0);
        assert(ndim_out == 2);
        assert(shape_out[0] == 2);
        assert(shape_out[1] == 3);

        std::cout << "Read data: ";
        for (uint64_t i = 0; i < (shape_out[0] * shape_out[1]); ++i) {
            std::cout << data_out[i] << " ";
            assert(std::abs(data_out[i] - (300.0 + i)) < 1e-9);
        }
        std::cout << "\n";

        delete[] data_out;
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}