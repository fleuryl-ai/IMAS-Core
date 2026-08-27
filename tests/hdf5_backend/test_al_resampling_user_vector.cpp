// tests/hdf5_backend/test_magnetics_resampling_vector.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_magnetics_resampling_vector";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Magnetics Resampling with Vector (getSample equivalent) ===\n" << RESET;

        const int time_dim_size = 10;
        const int flux_loop_size = 1;

        // 1. Writing
        {
            std::cout << "\n--- Phase 1: Writing Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "magnetics", "", WRITE_OP);
            backend.beginAction(&opCtx);

            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            std::vector<double> time_values(time_dim_size);
            for (int k = 0; k < time_dim_size; ++k) {
                time_values[k] = 1.0 + k; // 1.0, 2.0, ..., 10.0
            }
            int time_dim[] = {time_dim_size};
            backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, 1, time_dim);

            // --- Flux Loop AoS ---
            ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
            backend.beginArraystructAction(&fluxLoopCtx, (int*)&flux_loop_size);

            // flux_loop(0).flux.data
            std::vector<double> flux_data(time_dim_size);
            for (int k = 0; k < time_dim_size; ++k) {
                flux_data[k] = 5.0 + k; // 5.0, 6.0, ..., 14.0
            }
            int flux_dim[] = {time_dim_size};
            backend.writeData(&fluxLoopCtx, "flux/data", "time", flux_data.data(), alconst::double_data, 1, flux_dim);

            backend.endAction(&fluxLoopCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // 2. Reading (Time Range with resampling vector)
        {
            std::cout << "\n--- Phase 2: Reading Time Range with explicit dtime vector ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            // Explicit time points for resampling: {0.5, 1.0, 1.5, 1.7, 11.0}
            std::vector<double> dtime = {0.5, 1.0, 1.5, 1.7, 11.0};
            // tmin and tmax are ignored when dtime has multiple elements
            double tmin = 0.0; 
            double tmax = 0.0;

            // Using linear interpolation to match the test expectation (5.5 for 1.5)
            OperationContext opCtx(&dataEntryCtx, "magnetics", READ_OP, alconst::timerange_op, tmin, tmax, dtime, alconst::linear_interp);
            backend.beginAction(&opCtx);

            // Check time array first (should match dtime)
            void* time_ptr = nullptr;
            int time_datatype = alconst::double_data;
            int time_dim_out = 0;
            int time_size_out[H5S_MAX_RANK];
            backend.readData(&opCtx, "time", "time", &time_ptr, &time_datatype, &time_dim_out, time_size_out);
            
            assert(time_dim_out == 1);
            assert(time_size_out[0] == 5);
            double* read_times = (double*)time_ptr;
            for(size_t i=0; i<dtime.size(); ++i) {
                assert(std::abs(read_times[i] - dtime[i]) < 1e-9);
            }
            free(time_ptr);
            std::cout << "  [OK] Time vector read correctly.\n";

            // Check flux_loop data
            int read_aos_size = 0;
            ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
            backend.beginArraystructAction(&fluxLoopCtx, &read_aos_size);
            assert(read_aos_size == 1);

            void* data_ptr = nullptr;
            int datatype = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            backend.readData(&fluxLoopCtx, "flux/data", "time", &data_ptr, &datatype, &dim, size);
            
            assert(dim == 1);
            assert(size[0] == 5);
            
            double* vals = (double*)data_ptr;
            std::vector<double> expected = {5.0, 5.0, 5.5, 5.7, 14.0};
            
            for(int j=0; j<5; ++j) {
                std::cout << "    flux/data[" << j << "] (t=" << dtime[j] << ") = " << vals[j] << " (expected " << expected[j] << ")\n";
                if (std::abs(vals[j] - expected[j]) > 1e-9) {
                    std::cerr << RED << "Mismatch at index " << j << RESET << std::endl;
                    return 1;
                }
            }
            free(data_ptr);

            backend.endAction(&fluxLoopCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Time range read with resampling vector validation passed.\n" << RESET;
        }

        // 3. Reading (Time Range with single dtime element and tmin/tmax)
        {
            std::cout << "\n--- Phase 3: Reading Time Range with single dtime element (resampling step) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            std::vector<double> dtime = {0.2};
            double tmin = 2.0;
            double tmax = 3.0;

            // Using linear interpolation
            OperationContext opCtx(&dataEntryCtx, "magnetics", READ_OP, alconst::timerange_op, tmin, tmax, dtime, alconst::linear_interp);
            backend.beginAction(&opCtx);

            // Check time array first
            void* time_ptr = nullptr;
            int time_datatype = alconst::double_data;
            int time_dim_out = 0;
            int time_size_out[H5S_MAX_RANK];
            backend.readData(&opCtx, "time", "time", &time_ptr, &time_datatype, &time_dim_out, time_size_out);
            
            // Expected size: (3.0 - 2.0)/0.2 + 1 = 6
            int expected_size = 6;
            assert(time_dim_out == 1);
            assert(time_size_out[0] == expected_size);
            
            double* read_times = (double*)time_ptr;
            for(int i=0; i<expected_size; ++i) {
                double expected_time = tmin + i * dtime[0];
                if (std::abs(read_times[i] - expected_time) > 1e-9) {
                     std::cerr << RED << "Time mismatch at index " << i << ": expected " << expected_time << ", got " << read_times[i] << RESET << std::endl;
                     return 1;
                }
            }
            free(time_ptr);
            std::cout << "  [OK] Time vector read correctly.\n";

            // Check flux_loop data
            int read_aos_size = 0;
            ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
            backend.beginArraystructAction(&fluxLoopCtx, &read_aos_size);
            assert(read_aos_size == 1);

            void* data_ptr = nullptr;
            int datatype = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            backend.readData(&fluxLoopCtx, "flux/data", "time", &data_ptr, &datatype, &dim, size);
            
            assert(dim == 1);
            assert(size[0] == expected_size);
            
            double* vals = (double*)data_ptr;
            
            for(int j=0; j<expected_size; ++j) {
                double current_time = tmin + j * dtime[0];
                // Original data: flux = time + 4.0
                // Written: time 1.0 -> flux 5.0, time 2.0 -> flux 6.0
                double expected_val = current_time + 4.0;
                
                if (std::abs(vals[j] - expected_val) > 1e-9) {
                    std::cerr << RED << "Mismatch at index " << j << ": expected " << expected_val << ", got " << vals[j] << RESET << std::endl;
                    return 1;
                }
            }
            free(data_ptr);

            backend.endAction(&fluxLoopCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Time range read with single dtime validation passed.\n" << RESET;
        }

        // 4. Reading (Time Range with single dtime element and tmin/tmax, more samples)
        {
            std::cout << "\n--- Phase 4: Reading Time Range with single dtime element (more samples) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            std::vector<double> dtime = {0.2};
            double tmin = 2.0;
            double tmax = 5.0;

            // Using linear interpolation
            OperationContext opCtx(&dataEntryCtx, "magnetics", READ_OP, alconst::timerange_op, tmin, tmax, dtime, alconst::linear_interp);
            backend.beginAction(&opCtx);

            // Check time array first
            void* time_ptr = nullptr;
            int time_datatype = alconst::double_data;
            int time_dim_out = 0;
            int time_size_out[H5S_MAX_RANK];
            backend.readData(&opCtx, "time", "time", &time_ptr, &time_datatype, &time_dim_out, time_size_out);
            
            // Expected size: (5.0 - 2.0)/0.2 + 1 = 16
            int expected_size = 16;
            assert(time_dim_out == 1);
            assert(time_size_out[0] == expected_size);
            
            double* read_times = (double*)time_ptr;
            for(int i=0; i<expected_size; ++i) {
                double expected_time = tmin + i * dtime[0];
                if (std::abs(read_times[i] - expected_time) > 1e-9) {
                     std::cerr << RED << "Time mismatch at index " << i << ": expected " << expected_time << ", got " << read_times[i] << RESET << std::endl;
                     return 1;
                }
            }
            free(time_ptr);
            std::cout << "  [OK] Time vector read correctly.\n";

            // Check flux_loop data
            int read_aos_size = 0;
            ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
            backend.beginArraystructAction(&fluxLoopCtx, &read_aos_size);
            assert(read_aos_size == 1);

            void* data_ptr = nullptr;
            int datatype = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            backend.readData(&fluxLoopCtx, "flux/data", "time", &data_ptr, &datatype, &dim, size);
            
            assert(dim == 1);
            assert(size[0] == expected_size);
            
            double* vals = (double*)data_ptr;
            
            for(int j=0; j<expected_size; ++j) {
                double current_time = tmin + j * dtime[0];
                double expected_val = current_time + 4.0;
                
                if (std::abs(vals[j] - expected_val) > 1e-9) {
                    std::cerr << RED << "Mismatch at index " << j << ": expected " << expected_val << ", got " << vals[j] << RESET << std::endl;
                    return 1;
                }
            }
            free(data_ptr);

            backend.endAction(&fluxLoopCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Time range read with more samples validation passed.\n" << RESET;
        }

        // 5. Reading (Time Range with single dtime element, tmin/tmax, closest interpolation)
        {
            std::cout << "\n--- Phase 5: Reading Time Range with single dtime element (closest interp) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            std::vector<double> dtime = {0.2};
            double tmin = 2.0;
            double tmax = 3.0;

            // Using closest interpolation
            OperationContext opCtx(&dataEntryCtx, "magnetics", READ_OP, alconst::timerange_op, tmin, tmax, dtime, alconst::closest_interp);
            backend.beginAction(&opCtx);

            // Check time array first
            void* time_ptr = nullptr;
            int time_datatype = alconst::double_data;
            int time_dim_out = 0;
            int time_size_out[H5S_MAX_RANK];
            backend.readData(&opCtx, "time", "time", &time_ptr, &time_datatype, &time_dim_out, time_size_out);
            
            // Expected size: (3.0 - 2.0)/0.2 + 1 = 6
            int expected_size = 6;
            assert(time_dim_out == 1);
            assert(time_size_out[0] == expected_size);
            
            double* read_times = (double*)time_ptr;
            for(int i=0; i<expected_size; ++i) {
                double expected_time = tmin + i * dtime[0];
                if (std::abs(read_times[i] - expected_time) > 1e-9) {
                     std::cerr << RED << "Time mismatch at index " << i << ": expected " << expected_time << ", got " << read_times[i] << RESET << std::endl;
                     return 1;
                }
            }
            free(time_ptr);
            std::cout << "  [OK] Time vector read correctly.\n";

            // Check flux_loop data
            int read_aos_size = 0;
            ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
            backend.beginArraystructAction(&fluxLoopCtx, &read_aos_size);
            assert(read_aos_size == 1);

            void* data_ptr = nullptr;
            int datatype = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            backend.readData(&fluxLoopCtx, "flux/data", "time", &data_ptr, &datatype, &dim, size);
            
            assert(dim == 1);
            assert(size[0] == expected_size);
            
            double* vals = (double*)data_ptr;
            
            for(int j=0; j<expected_size; ++j) {
                double current_time = tmin + j * dtime[0];
                // Original data: flux = time + 4.0
                // Closest interpolation means we pick the value at the closest integer time point
                // 2.0 -> 2.0, 2.2 -> 2.0, 2.4 -> 2.0, 2.6 -> 3.0, 2.8 -> 3.0, 3.0 -> 3.0
                double closest_time = std::round(current_time);
                double expected_val = closest_time + 4.0;
                
                if (std::abs(vals[j] - expected_val) > 1e-9) {
                    std::cerr << RED << "Mismatch at index " << j << " (t=" << current_time << "): expected " << expected_val << ", got " << vals[j] << RESET << std::endl;
                    return 1;
                }
            }
            free(data_ptr);

            backend.endAction(&fluxLoopCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Time range read with closest interp validation passed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}
