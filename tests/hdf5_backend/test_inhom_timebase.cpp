// test_inhom_timebase.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

// Colors for debug
#define RESET ""
#define BOLD ""
#define GREEN ""
#define RED ""

const std::string URI = "imas:hdf5?path=./test_db_inhom_timebase";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Inhomogeneous Timebase ===\n" << RESET;

        // 1. Writing
        {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Set homogeneous_time = 0
            int homogeneous_time = 0;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            int size_A = 3;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);

            for (int i = 0; i < size_A; ++i) {
                int size_B = 3;
                ArraystructContext ctxB(&ctxA, "B", "");
                backend.beginArraystructAction(&ctxB, &size_B);

                for (int j = 0; j < size_B; ++j) {
                    // Define timebase for B[j]
                    std::vector<double> time_values;
                    if (j == 0) {
                        for (int k = 0; k < 10; ++k) time_values.push_back((double)k);
                    } else if (j == 1) {
                        for (int k = 0; k < 10; ++k) time_values.push_back((double)(10 + k));
                    } else if (j == 2) {
                        for (int k = 0; k < 5; ++k) time_values.push_back((double)(20 + k));
                    }

                    int time_dim = 1;
                    int time_size[] = {(int)time_values.size()};
                    
                    // Write timebase (as static array within B[j])
                    backend.writeData(&ctxB, "time", "", time_values.data(), alconst::double_data, time_dim, time_size);

                    // Write dynamic signal (associated with time)
                    std::vector<double> dyn_sig(time_values.size());
                    for(size_t k=0; k<time_values.size(); ++k) dyn_sig[k] = time_values[k] * 10.0;
                    
                    backend.writeData(&ctxB, "dyn_sig", "time", dyn_sig.data(), alconst::double_data, time_dim, time_size);

                    int size_C = 1;
                    ArraystructContext ctxC(&ctxB, "C", "");
                    backend.beginArraystructAction(&ctxC, &size_C);
                    // Just empty C
                    backend.endAction(&ctxC);

                    if (j < size_B - 1) ctxB.nextIndex(1);
                }
                backend.endAction(&ctxB);
                if (i < size_A - 1) ctxA.nextIndex(1);
            }
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // 2. Reading
        {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", READ_OP);
            backend.beginAction(&opCtx);

            int size_A = 0;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);
            assert(size_A == 3);

            for (int i = 0; i < size_A; ++i) {
                int size_B = 0;
                ArraystructContext ctxB(&ctxA, "B", "");
                backend.beginArraystructAction(&ctxB, &size_B);
                assert(size_B == 3);

                for (int j = 0; j < size_B; ++j) {
                    void* data_ptr = nullptr;
                    int datatype = alconst::double_data;
                    int dim = 0;
                    int size[H5S_MAX_RANK];

                    // Read dyn_sig
                    // Note: In READ_OP (Global), we read the full array.
                    // Since we wrote it as a single block (via writeData with timebase), it should come back as such.
                    backend.readData(&ctxB, "dyn_sig", "time", &data_ptr, &datatype, &dim, size);
                    
                    int expected_size = (j == 2) ? 5 : 10;
                    assert(dim == 1);
                    assert(size[0] == expected_size);
                    
                    double* sig_vals = (double*)data_ptr;
                    double start_val = (j == 0) ? 0.0 : ((j == 1) ? 10.0 : 20.0);
                    for(int k=0; k<expected_size; ++k) {
                        assert(std::abs(sig_vals[k] - (start_val + k) * 10.0) < 1e-9);
                    }
                    free(data_ptr);

                    if (j < size_B - 1) ctxB.nextIndex(1);
                }
                backend.endAction(&ctxB);
                if (i < size_A - 1) ctxA.nextIndex(1);
            }
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Reading completed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}