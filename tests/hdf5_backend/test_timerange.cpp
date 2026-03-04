// tests/hdf5_backend/test_timerange.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <iomanip> // For std::setprecision

namespace fs = std::filesystem;

// Colors for debug
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_timerange";
const std::string IDS_NAME = "magnetics";

int main() {
    try {
        std::cout << BOLD << "\n=== Test TimeRange Read (from Matlab test parameters) ===\n" << RESET;

        const int time_dim_size = 10;

        // 1. Writing Data (similar to matlab test setup)
        {
            std::cout << "\n--- Phase 1: Writing Data ---\n";
            // Clean up previous test run
            fs::remove_all("./test_db_timerange_matlab");

            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, IDS_NAME, "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Set homogeneous_time = 1
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Write root timebase 'time' from 0.1 to 1.0
            std::vector<double> time_values(time_dim_size);
            for (int k = 0; k < time_dim_size; ++k) {
                time_values[k] = (double)(k + 1) * 0.1;
            }
            int time_dim[] = {time_dim_size};
            backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, 1, time_dim);

            // Write a dynamic signal where value = 100 + time * 10
            std::vector<double> dyn_signal_values(time_dim_size);
            for (int k = 0; k < time_dim_size; ++k) {
                dyn_signal_values[k] = 100.0 + time_values[k] * 10.0;
            }
            int dyn_sig_dim[] = {time_dim_size};
            backend.writeData(&opCtx, "dyn_signal", "time", dyn_signal_values.data(), alconst::double_data, 1, dyn_sig_dim);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // 2. Reading Test Case 1: Time Range without resampling
        {
            std::cout << "\n--- Phase 2: Reading Time Range [0.3, 0.8] (no resampling) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double tmin = 0.3;
            double tmax = 0.8;
            std::vector<double> dtime; // Empty for no resampling

            OperationContext opCtx(&dataEntryCtx, IDS_NAME, READ_OP, alconst::timerange_op, tmin, tmax, dtime, alconst::closest_interp);
            backend.beginAction(&opCtx);

            void* data_ptr = nullptr;
            int datatype = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            backend.readData(&opCtx, "dyn_signal", "time", &data_ptr, &datatype, &dim, size);

            // Validation
            int expected_size = 6; // time points 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
            assert(dim == 1);
            printf("--> Read data size: %d (expected %d)\n", size[0], expected_size);
            assert(size[0] == expected_size);
            
            double* sig_vals = (double*)data_ptr;
            std::cout << "  Data read back: ";
            for(int k=0; k<expected_size; ++k) {
                double expected_val = 100.0 + (0.3 + k * 0.1) * 10.0; // 103, 104, ... 108
                std::cout << sig_vals[k] << " ";
                assert(std::abs(sig_vals[k] - expected_val) < 1e-9);
            }
            std::cout << "\n";

            free(data_ptr);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Test Case 1 (no resampling) validation passed.\n" << RESET;
        }

        // 3. Reading Test Case 2: Time Range with resampling (step)
        {
            std::cout << "\n--- Phase 3: Reading Time Range [0.3, 0.8] (resampling with step=0.05) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double tmin = 0.3;
            double tmax = 0.8;
            std::vector<double> dtime = {0.05}; // Resampling step

            OperationContext opCtx(&dataEntryCtx, IDS_NAME, READ_OP, alconst::timerange_op, tmin, tmax, dtime, alconst::linear_interp);
            backend.beginAction(&opCtx);

            // Read signal and time to validate
            void* sig_ptr = nullptr;
            int sig_size[H5S_MAX_RANK], sig_dim=0, sig_datatype=alconst::double_data;
            backend.readData(&opCtx, "dyn_signal", "time", &sig_ptr, &sig_datatype, &sig_dim, sig_size);

            void* time_ptr = nullptr;
            int time_size[H5S_MAX_RANK], time_dim=0, time_datatype=alconst::double_data;
            backend.readData(&opCtx, "time", "time", &time_ptr, &time_datatype, &time_dim, time_size);

            // Validation
            int expected_size = 11; // (0.8 - 0.3) / 0.05 + 1
            assert(sig_dim == 1 && time_dim == 1);
            printf("--> Read data size: %d (expected %d)\n", sig_size[0], expected_size);
            assert(sig_size[0] == expected_size && time_size[0] == expected_size);
            
            double* sig_vals = (double*)sig_ptr;
            double* time_vals = (double*)time_ptr;
            
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Resampled time and data:\n";
            for(int k=0; k<expected_size; ++k) {
                double expected_time = 0.3 + k * 0.05;
                double expected_val = 100.0 + expected_time * 10.0;
                assert(std::abs(time_vals[k] - expected_time) < 1e-9);
                assert(std::abs(sig_vals[k] - expected_val) < 1e-9);
            }
            std::cout << std::defaultfloat;

            free(sig_ptr);
            free(time_ptr);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Test Case 2 (resampling with step) validation passed.\n" << RESET;
        }

        // 4. Reading Test Case 3: Time Range with resampling (user vector)
        {
            std::cout << "\n--- Phase 4: Reading Time Range [0.0, 1.0] (resampling with user vector) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double tmin = 0.0;
            double tmax = 1.0;
            std::vector<double> dtime = {0.32, 0.34, 0.37}; // User-defined time points

            OperationContext opCtx(&dataEntryCtx, IDS_NAME, READ_OP, alconst::timerange_op, tmin, tmax, dtime, alconst::linear_interp);
            backend.beginAction(&opCtx);

            // Read signal and time to validate
            void* sig_ptr = nullptr;
            int sig_size[H5S_MAX_RANK], sig_dim=0, sig_datatype=alconst::double_data;
            backend.readData(&opCtx, "dyn_signal", "time", &sig_ptr, &sig_datatype, &sig_dim, sig_size);

            void* time_ptr = nullptr;
            int time_size[H5S_MAX_RANK], time_dim=0, time_datatype=alconst::double_data;
            backend.readData(&opCtx, "time", "time", &time_ptr, &time_datatype, &time_dim, time_size);

            // Validation
            int expected_size = 3;
            assert(sig_dim == 1 && time_dim == 1);
            printf("--> Read data size: %d (expected %d)\n", sig_size[0], expected_size);
            assert(sig_size[0] == expected_size && time_size[0] == expected_size);
            
            double* sig_vals = (double*)sig_ptr;
            double* time_vals = (double*)time_ptr;
            
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Resampled time and data:\n";
            for(int k=0; k<expected_size; ++k) {
                double expected_time = dtime[k];
                double expected_val = 100.0 + expected_time * 10.0;
                assert(std::abs(time_vals[k] - expected_time) < 1e-9);
                assert(std::abs(sig_vals[k] - expected_val) < 1e-9);
            }
            std::cout << std::defaultfloat;

            free(sig_ptr);
            free(time_ptr);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Test Case 3 (resampling with user vector) validation passed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    std::cout << BOLD << GREEN << "\n✓ All timerange tests passed!\n" << RESET;
    return 0;
}