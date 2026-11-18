// @file  test_engine_readdata_by_index.cpp
// @brief PanzerDB readDataByIndex smoke test: writes a simple A[0]/B[0]/eeg
//        structure with a 5-element signal and reads it back by index.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: readDataByIndex ===\n\n";
    const std::string filename = "test_read_by_index.panzer";

    // ===================================================================
    // 1. Write a simple A[0]/B[0]/eeg structure
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Writing A[0]/B[0]/eeg...\n";
        db.beginArray("A", 1);
        db.beginArray("B", 1);

        std::vector<double> data_to_write = {101.0, 102.0, 103.0, 104.0, 105.0};
        db.writeData("eeg", {5}, data_to_write.data(), data_to_write.size());

        db.endArray(); // B
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

        uint64_t static_indices[] = {0, 0}; // A[0], B[0]
        int64_t time_index = -1; // Static
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        const char* full_path = "A/0/B/0/eeg";
        int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

        assert(result == 0);
        assert(ndim_out == 1);
        assert(shape_out[0] == 5);

        std::cout << "Read data: ";
        for (uint64_t i = 0; i < shape_out[0]; ++i) {
            std::cout << data_out[i] << " ";
            assert(std::abs(data_out[i] - (101.0 + i)) < 1e-9);
        }
        std::cout << "\n";

        delete[] data_out; // Remember to free the memory
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}