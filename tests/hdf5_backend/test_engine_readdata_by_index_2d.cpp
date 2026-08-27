// @file  test_engine_readdata_by_index_2d.cpp
// @brief PanzerDB readDataByIndex test: writes a 4x5 2D signal inside A[1] of an
//        A[3]/B[2] structure and reads it back by index, checking shape and values.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>

int main() {
    std::cout << "=== TEST PanzerDB: readDataByIndex (2D signal) ===\n\n";
    const std::string filename = "test_read_2d_signal.panzer";

    // ===================================================================
    // 1. Write an A[3]/B[2] structure with a 2D signal in A[1]
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Writing the structure...\n";
        db.beginArray("A", 3);
        for (int i = 0; i < 3; ++i) {
            // We write a 2D signal only in element A[1]
            if (i == 1) {
                std::vector<double> data_to_write(4 * 5); // 4x5 tensor
                std::iota(data_to_write.begin(), data_to_write.end(), 500.0);
                db.writeData("signal_2d", {4, 5}, data_to_write.data(), data_to_write.size());
            }

            // Each element of A contains a size-2 AoS B, but with no direct data.
            db.beginArray("B", 2);
            for (int j = 0; j < 2; ++j) {
                // Intentionally empty, we just increment B's index
                //db.incrementArrayIndex();
            }
            db.endArray(); // B

            db.incrementArrayIndex(); // Moves to the next A element
        }
        db.endArray(); // A

        db.close();
        std::cout << "File closed.\n\n";
    }

    // ===================================================================
    // 2. Reopen and read the 2D signal
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        std::cout << "Verifying the AoS sizes...\n";
        auto shapeA = db.getAOSShape("A");
        assert(shapeA.size() == 1 && "The shape of A must contain a single element.");
        assert(shapeA[0] == 3 && "The size of A must be 3.");
        std::cout << "Shape of A: { " << shapeA[0] << " } (expected: { 3 })\n";

        auto shapeB = db.getAOSShape("A/0/B");
        assert(shapeB[0] == 2 && "The size of B must be 2.");
        auto shape1B = db.getAOSShape("A/1/B");
        assert(shape1B[0] == 2 && "The size of B must be 2.");
        auto shape2B = db.getAOSShape("A/2/B");
        assert(shape2B[0] == 2 && "The size of B must be 2.");

        std::cout << "Reading signal_2d in A[1]...\n";

        uint64_t static_indices[] = {1}; // A[1]
        int64_t time_index = -1; // Static
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        const char* full_path = "A/1/signal_2d";
        int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

        assert(result == 0 && "The 2D signal read must succeed.");
        assert(ndim_out == 2 && "The signal dimension must be 2.");
        assert(shape_out[0] == 4 && "The first dimension must be 4.");
        assert(shape_out[1] == 5 && "The second dimension must be 5.");

        std::cout << "Data read successfully. Verifying the values...\n";
        for (uint64_t i = 0; i < (shape_out[0] * shape_out[1]); ++i) {
            assert(std::abs(data_out[i] - (500.0 + i)) < 1e-9);
        }
        std::cout << "Toutes les valeurs sont correctes.\n";

        delete[] data_out;
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}