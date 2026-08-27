// @file  test_engine_readdata_by_index_2signals.cpp
// @brief PanzerDB readDataByIndex test: writes two static signals, "flux" under
//        A/B/C and "field" directly under A, and reads both back by index.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: readDataByIndex (deep, 2 signals) ===\n\n";
    const std::string filename = "test_read_by_index_deep_2_signals.panzer";

    // ===================================================================
    // 1. Write the A/B/C/flux and A/field structure
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Writing A/B/C/flux and A/field...\n";
        db.beginArray("A", 1);

        // Writing the "field" signal directly under A
        std::vector<double> field_data = {55.5, 66.6};
        db.writeData("field", {2}, field_data.data(), field_data.size());

        db.beginArray("B", 1);
        db.beginArray("C", 1);

        std::vector<double> flux_data = {201.5, 202.5, 203.5};
        db.writeData("flux", {3}, flux_data.data(), flux_data.size());

        db.incrementArrayIndex(); // Increments C
        db.endArray(); // C
        db.incrementArrayIndex(); // Increments B
        db.endArray(); // B
        db.incrementArrayIndex(); // Increments A
        db.endArray(); // A

        db.close();
        std::cout << "File closed.\n\n";
    }

    // ===================================================================
    // 2. Reopen and read the two signals
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        // --- Validation of "flux" ---
        std::cout << "Reading 'flux' with readDataByIndex...\n";
        {
            uint64_t static_indices[] = {0, 0, 0}; // A[0], B[0], C[0]
            int64_t time_index = -1;
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/0/B/0/C/0/flux";
            int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0);
            assert(ndim_out == 1);
            assert(shape_out[0] == 3);

            std::cout << "Read 'flux' data: ";
            for (uint64_t i = 0; i < shape_out[0]; ++i) {
                std::cout << data_out[i] << " ";
                assert(std::abs(data_out[i] - (201.5 + i)) < 1e-9);
            }
            std::cout << "\n";
            delete[] data_out;
        }

        // --- Validation of "field" ---
        std::cout << "\nReading 'field' with readDataByIndex...\n";
        {
            uint64_t static_indices[] = {0}; // A[0]
            int64_t time_index = -1;
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/0/field";
            int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);
            assert(result == 0);
            assert(ndim_out == 1 && shape_out[0] == 2);
            assert(std::abs(data_out[0] - 55.5) < 1e-9 && std::abs(data_out[1] - 66.6) < 1e-9);
            std::cout << "Read 'field' data: " << data_out[0] << ", " << data_out[1] << "\n";
            delete[] data_out;
        }
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}