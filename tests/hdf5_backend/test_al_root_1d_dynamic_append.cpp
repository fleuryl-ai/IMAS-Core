// @file  test_al_root_1d_dynamic_append.cpp
// @brief Tests appending slices of a root-level dynamic 1D signal: first slice
//        written, second slice appended, then both slice reads validated.
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
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_root_1d_dynamic_append";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Root 1D Dynamic Signal Append ===\n" << RESET;

        const int vec_size = 1;

        // 1. Phase 1: Writing first slice (t=0.0)
        {
            std::cout << "\n--- Phase 1: Writing first slice (t=0.0) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Write time slice 0
            double time_val = 0.0;
            int time_dim = 1;
            int time_size[] = {1};
            backend.writeData(&opCtx, "time", "time", &time_val, alconst::double_data, time_dim, time_size);

            // Write dynamic 1D signal slice 0
            std::vector<double> sig_val(vec_size);
            for(int i=0; i<vec_size; ++i) sig_val[i] = 100.0 + i;
            
            // Dimension of the slice is 1D (vector of size 5)
            // In AL, for a time slice of a 1D signal, we pass dim=1 and size={5}
            int sig_dim = 1;
            int sig_size[] = {vec_size};
            backend.writeData(&opCtx, "dyn_1d_sig", "time", sig_val.data(), alconst::double_data, sig_dim, sig_size);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Phase 1 completed.\n" << RESET;
        }

        // 2. Phase 2: Appending second slice (t=1.0)
        {
            std::cout << "\n--- Phase 2: Appending second slice (t=1.0) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double time_undef = alconst::undefined_time;
            int interpmode_undef = alconst::undefined_interp;
            // Using SLICE_OP for appending
            OperationContext opCtx(&dataEntryCtx, "test_ids", WRITE_OP, alconst::slice_op, time_undef, interpmode_undef);
            backend.beginAction(&opCtx);

            // Write time slice 1
            double time_val = 1.0;
            int time_dim = 1;
            int time_size[] = {1};
            backend.writeData(&opCtx, "time", "time", &time_val, alconst::double_data, time_dim, time_size);

            // Write dynamic 1D signal slice 1
            std::vector<double> sig_val(vec_size);
            for(int i=0; i<vec_size; ++i) sig_val[i] = 200.0 + i;
            
            int sig_dim = 1;
            int sig_size[] = {vec_size};
            backend.writeData(&opCtx, "dyn_1d_sig", "time", sig_val.data(), alconst::double_data, sig_dim, sig_size);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Phase 2 completed.\n" << RESET;
        }

        // 3. Phase 3: Validation
        {
            std::cout << "\n--- Phase 3: Validation ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            // Validate Slice 0 (t=0.0)
            {
                OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP, alconst::slice_op, 0.0, alconst::closest_interp);
                backend.beginAction(&opCtx);

                void* data_ptr = nullptr;
                int datatype = alconst::double_data;
                int dim = 0;
                int size[H5S_MAX_RANK];
                backend.readData(&opCtx, "dyn_1d_sig", "time", &data_ptr, &datatype, &dim, size);
                
                assert(dim == 1);
                assert(size[0] == vec_size);
                double* vals = (double*)data_ptr;
                for(int i=0; i<vec_size; ++i) {
                    assert(std::abs(vals[i] - (100.0 + i)) < 1e-9);
                }
                std::cout << "  [OK] Slice t=0.0 valid.\n";
                free(data_ptr);
                backend.endAction(&opCtx);
            }

            // Validate Slice 1 (t=1.0)
            {
                OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP, alconst::slice_op, 1.0, alconst::closest_interp);
                backend.beginAction(&opCtx);

                void* data_ptr = nullptr;
                int datatype = alconst::double_data;
                int dim = 0;
                int size[H5S_MAX_RANK];
                backend.readData(&opCtx, "dyn_1d_sig", "time", &data_ptr, &datatype, &dim, size);
                
                assert(dim == 1);
                assert(size[0] == vec_size);
                double* vals = (double*)data_ptr;
                for(int i=0; i<vec_size; ++i) {
                    assert(std::abs(vals[i] - (200.0 + i)) < 1e-9);
                }
                std::cout << "  [OK] Slice t=1.0 valid.\n";
                free(data_ptr);
                backend.endAction(&opCtx);
            }

            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Validation completed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}