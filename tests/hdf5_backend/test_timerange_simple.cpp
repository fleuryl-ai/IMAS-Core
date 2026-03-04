// tests/hdf5_backend/test_timerange_simple.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_timerange_simple";

int main() {
    try {
        std::cout << BOLD << "\n=== Test TimeRange Read (Simple, No Resampling) ===\n" << RESET;

        const int time_dim_size = 20;

        // 1. Writing
        {
            std::cout << "\n--- Phase 1: Writing Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Set homogeneous_time = 1
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Write root timebase 'time'
            std::vector<double> time_values(time_dim_size);
            for (int k = 0; k < time_dim_size; ++k) {
                time_values[k] = (double)k * 0.1;
            }
            int time_dim[] = {time_dim_size};
            backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, 1, time_dim);

            // Write a dynamic signal at the root
            std::vector<double> dyn_signal_values(time_dim_size);
            for (int k = 0; k < time_dim_size; ++k) {
                dyn_signal_values[k] = 100.0 + k;
            }
            int dyn_sig_dim[] = {time_dim_size};
            backend.writeData(&opCtx, "dyn_signal", "time", dyn_signal_values.data(), alconst::double_data, 1, dyn_sig_dim);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // 2. Reading (Time Range without resampling)
        {
            std::cout << "\n--- Phase 2: Reading Time Range [0.5, 1.0] ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double tmin = 0.5;
            double tmax = 1.0;
            std::vector<double> dtime; // Empty for no resampling

            // OperationContext for TimeRange read
            OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP, alconst::timerange_op, tmin, tmax, dtime, alconst::closest_interp);
            backend.beginAction(&opCtx);

            void* data_ptr = nullptr;
            int datatype = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            backend.readData(&opCtx, "dyn_signal", "time", &data_ptr, &datatype, &dim, size);

            // Validation
            int expected_size = 6; // time points 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (indices 5 to 10)
            assert(dim == 1);
            printf("--> Read data size: %d (expected %d)\n", size[0], expected_size);
            assert(size[0] == expected_size);
            
            double* sig_vals = (double*)data_ptr;
            std::cout << "  Data read back: ";
            for(int k=0; k<expected_size; ++k) {
                double expected_val = 100.0 + (5 + k); // values for indices 5 to 10
                std::cout << sig_vals[k] << " ";
                assert(std::abs(sig_vals[k] - expected_val) < 1e-9);
            }
            std::cout << "\n";

            free(data_ptr);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Time range read validation passed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}