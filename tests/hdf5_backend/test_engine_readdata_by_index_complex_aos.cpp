// @file  test_engine_readdata_by_index_complex_aos.cpp
// @brief PanzerDB readDataByIndex test: writes a complex A/B/C structure with
//        different B sizes per element, and reads "flux" and "field" by index.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: readDataByIndex (complex AoS) ===\n\n";
    const std::string filename = "test_read_by_index_complex_aos.panzer";

    // ===================================================================
    // 1. Write a complex structure
    //    A (size=2)
    //    - A[0]/field
    //    - A[0]/B (size=2)
    //      - A[0]/B[0]/C (size=1) / flux
    //      - A[0]/B[1]/C (size=1) / flux
    //    - A[1]/field
    //    - A[1]/B (size=3)
    //      - A[1]/B[0]/C (size=1) / flux
    //      ...
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Writing the complex structure...\n";
        db.beginArray("A", 2);
        // --- Writing into A[0] ---
        std::vector<double> field_data_0 = {10.0, 11.0};
        db.writeData("field", {2}, field_data_0.data(), field_data_0.size());
        db.beginArray("B", 2);
        for (int b = 0; b < 2; ++b) {
            db.beginArray("C", 1);
            std::vector<double> flux_data_0x = {1000.0 + (double)b};
            db.writeData("flux", {1}, flux_data_0x.data(), flux_data_0x.size());
            db.endArray(); // C
            db.incrementArrayIndex(); // Increments B's index
        }
        db.endArray(); // B
        db.incrementArrayIndex(); // Increments A's index

        // --- Writing into A[1] ---
        std::vector<double> field_data_1 = {20.0, 21.0, 22.0};
        db.writeData("field", {3}, field_data_1.data(), field_data_1.size());
        db.beginArray("B", 3);
        for (int b = 0; b < 3; ++b) {
            db.beginArray("C", 1);
            std::vector<double> flux_data_1x = {2000.0 + (double)b};
            db.writeData("flux", {1}, flux_data_1x.data(), flux_data_1x.size());
            db.endArray(); // C
            db.incrementArrayIndex(); // Increments B's index
        }
        db.endArray(); // B
        db.incrementArrayIndex(); // Increments A's index
        
        db.endArray(); // A        
        db.close();
        std::cout << "File closed.\n\n";
    }

    // ===================================================================
    // 2. Reopen and read the signals
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        // --- Validation of "flux" in A[1]/B[2]/C[0] ---
        std::cout << "Reading 'flux' in A[1]/B[2]/C[0]...\n";
        {
            uint64_t static_indices[] = {1, 2, 0}; // A[1], B[2], C[0]
            int64_t time_index = -1;
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/1/B/2/C/0/flux";
            int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0);
            assert(ndim_out == 1);
            assert(shape_out[0] == 1);
            assert(std::abs(data_out[0] - 2002.0) < 1e-9);
            std::cout << "Read 'flux' data: " << data_out[0] << " (expected: 2002.0)\n";
            delete[] data_out;
        }

        // --- Validation of "field" in A[0] ---
        std::cout << "\nReading 'field' in A[0]...\n";
        {
            uint64_t static_indices[] = {0}; // A[0]
            int64_t time_index = -1;
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/0/field";
            int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0);
            assert(ndim_out == 1);
            assert(shape_out[0] == 2);
            assert(std::abs(data_out[0] - 10.0) < 1e-9);
            assert(std::abs(data_out[1] - 11.0) < 1e-9);
            std::cout << "Read 'field' data: " << data_out[0] << ", " << data_out[1] << " (expected: 10.0, 11.0)\n";
            delete[] data_out;
        }
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}