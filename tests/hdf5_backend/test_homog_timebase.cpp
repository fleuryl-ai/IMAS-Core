// test_homog_timebase.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_homog_timebase";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Homogeneous Timebase ===\n" << RESET;

        // 1. Writing
        {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Set homogeneous_time = 1 (Homogeneous)
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Write root timebase 'time' (10 points)
            // Note: Even for the time array itself, we often pass "time" as timebase in AL conventions for homogeneous cases,
            // or it can be treated as a special signal.
            std::vector<double> time_values(10);
            for (int k = 0; k < 10; ++k) time_values[k] = (double)k * 0.1;
            int time_dim = 1;
            int time_size[] = {10};
            backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, time_dim, time_size);

            int size_A = 3;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);

            for (int i = 0; i < size_A; ++i) {
                // Static signal in A (no timebase)
                double static_val_A = 100.0 + i;
                backend.writeData(&ctxA, "static_sig_A", "", &static_val_A, alconst::double_data, 0, nullptr);

                int size_B = 3;
                ArraystructContext ctxB(&ctxA, "B", "");
                backend.beginArraystructAction(&ctxB, &size_B);

                for (int j = 0; j < size_B; ++j) {
                    // Static signal in B (no timebase)
                    double static_val_B = 200.0 + i * 10 + j;
                    backend.writeData(&ctxB, "static_sig_B", "", &static_val_B, alconst::double_data, 0, nullptr);

                    // Dynamic signal in B (10 points) associated with root 'time'
                    std::vector<double> dyn_sig(10);
                    for(int k=0; k<10; ++k) dyn_sig[k] = (double)(i * 100 + j * 10 + k);
                    
                    // We pass "time" as timebase, which refers to the root time array since we are in homogeneous mode
                    backend.writeData(&ctxB, "dyn_sig", "time", dyn_sig.data(), alconst::double_data, time_dim, time_size);

                    int size_C = 1;
                    ArraystructContext ctxC(&ctxB, "C", "");
                    backend.beginArraystructAction(&ctxC, &size_C);
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
                // Read static signal A
                void* data_ptr = nullptr;
                int datatype = alconst::double_data;
                int dim = 0;
                int size[H5S_MAX_RANK];
                
                backend.readData(&ctxA, "static_sig_A", "", &data_ptr, &datatype, &dim, size);
                assert(dim == 0); // scalar
                assert(std::abs(((double*)data_ptr)[0] - (100.0 + i)) < 1e-9);
                free(data_ptr);

                int size_B = 0;
                ArraystructContext ctxB(&ctxA, "B", "");
                backend.beginArraystructAction(&ctxB, &size_B);
                assert(size_B == 3);

                for (int j = 0; j < size_B; ++j) {
                    // Read static signal B
                    backend.readData(&ctxB, "static_sig_B", "", &data_ptr, &datatype, &dim, size);
                    assert(dim == 0);
                    assert(std::abs(((double*)data_ptr)[0] - (200.0 + i * 10 + j)) < 1e-9);
                    free(data_ptr);

                    // Read dyn_sig
                    backend.readData(&ctxB, "dyn_sig", "time", &data_ptr, &datatype, &dim, size);
                    
                    assert(dim == 1);
                    assert(size[0] == 10);
                    
                    double* sig_vals = (double*)data_ptr;
                    for(int k=0; k<10; ++k) {
                        assert(std::abs(sig_vals[k] - (double)(i * 100 + j * 10 + k)) < 1e-9);
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

        // 3. Writing Slices (Append)
        std::cout << BOLD << "\n=== 3. Writing Slices (Append) ===\n" << RESET;
        for (int k = 0; k < 5; ++k) {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double time_val = 1.0 + k * 0.1; // 1.0, 1.1, 1.2, 1.3, 1.4
            
            double time_undef = alconst::undefined_time;
            int interpmode_undef = alconst::undefined_interp;
            OperationContext opCtx(&dataEntryCtx, "test_ids", WRITE_OP, alconst::slice_op, time_undef, interpmode_undef);
            
            backend.beginAction(&opCtx);

            // Write time scalar for this slice
            backend.writeData(&opCtx, "time", "time", &time_val, alconst::double_data, 0, nullptr);

            int size_A = 3;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);

            for (int i = 0; i < size_A; ++i) {
                int size_B = 3;
                ArraystructContext ctxB(&ctxA, "B", "");
                backend.beginArraystructAction(&ctxB, &size_B);

                for (int j = 0; j < size_B; ++j) {
                    // dyn_sig value for this slice
                    // Formula used in step 1: (double)(i * 100 + j * 10 + k) where k was 0..9
                    // We extend this pattern. k here is 0..4 (relative to append), so global index is 10+k.
                    double val = (double)(i * 100 + j * 10 + (10 + k));
                    
                    backend.writeData(&ctxB, "dyn_sig", "time", &val, alconst::double_data, 0, nullptr);

                    if (j < size_B - 1) ctxB.nextIndex(1);
                }
                backend.endAction(&ctxB);
                if (i < size_A - 1) ctxA.nextIndex(1);
            }
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Slice " << k << " (t=" << time_val << ") written.\n" << RESET;
        }

        // 4. Reading Slices (Validation)
        std::cout << BOLD << "\n=== 4. Reading Slices (Validation) ===\n" << RESET;
        {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            // Validate at t=1.25 (between 1.2 and 1.3, indices 12 and 13)
            double target_time = 1.25;
            int interpmode = alconst::linear_interp;
            OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP, alconst::slice_op, target_time, interpmode);
            
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
                    printf("Reading dyn_sig at A[%d]/B[%d]...\n", i, j);
                    backend.readData(&ctxB, "dyn_sig", "time", &data_ptr, &datatype, &dim, size);
                    
                    assert(dim == 0); // Scalar slice
                    double val = *(double*)data_ptr;
                    // Expected: Interpolation between (i*100 + j*10 + 12) and (i*100 + j*10 + 13) at 1.25 (midpoint)
                    double expected = (double)(i * 100 + j * 10) + 12.5;
                    
                    if (std::abs(val - expected) > 1e-9) {
                        std::cerr << RED << "Mismatch at A[" << i << "]/B[" << j << "]: expected " << expected << ", got " << val << RESET << std::endl;
                        return 1;
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
            std::cout << GREEN << "[OK] Slice validation at t=" << target_time << " passed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}