// tests/hdf5_backend/test_timerange_nested_dynamic_aos.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_timerange_nested_dynamic_aos";

int main() {
    try {
        std::cout << BOLD << "\n=== Test TimeRange Read with Nested Dynamic AoS ===\n" << RESET;

        const int time_dim_size = 20;
        const int aos_size = 2;

        // 1. Writing
        {
            std::cout << "\n--- Phase 1: Writing Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            std::vector<double> time_values(time_dim_size);
            for (int k = 0; k < time_dim_size; ++k) {
                time_values[k] = (double)k * 0.1;
            }
            int time_dim[] = {time_dim_size};
            backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, 1, time_dim);

            // --- Start of Static AoS ---
            ArraystructContext staticAosCtx(&opCtx, "static_aos", "");
            backend.beginArraystructAction(&staticAosCtx, (int*)&aos_size);

            for (int i = 0; i < aos_size; ++i) {
                double static_val = 500.0 + i;
                backend.writeData(&staticAosCtx, "static_signal", "", &static_val, alconst::double_data, 0, nullptr);

                // Dynamic signal in static AoS (using root time)
                std::vector<double> dyn_signal_values(time_dim_size);
                for (int k = 0; k < time_dim_size; ++k) {
                    dyn_signal_values[k] = 1000.0 + (i * 100) + k;
                }
                int dyn_sig_dim[] = {time_dim_size};
                backend.writeData(&staticAosCtx, "dynamic_signal", "time", dyn_signal_values.data(), alconst::double_data, 1, dyn_sig_dim);

                // --- Start of Nested Dynamic AoS ---
                // It has the same number of elements as the timebase to match "homogeneous" behavior locally
                int dyn_aos_size = time_dim_size; 
                ArraystructContext dynAosCtx(&staticAosCtx, "dynamic_aos", "time");
                backend.beginArraystructAction(&dynAosCtx, &dyn_aos_size);

                for (int t = 0; t < dyn_aos_size; ++t) {
                    double current_time = time_values[t];
                    backend.writeData(&dynAosCtx, "time", "", &current_time, alconst::double_data, 0, nullptr);
                    
                    double sig_val = 2000.0 + (i * 100) + t;
                    backend.writeData(&dynAosCtx, "signal_in_dyn_aos", "time", &sig_val, alconst::double_data, 0, nullptr);

                    if (t < dyn_aos_size - 1) dynAosCtx.nextIndex(1);
                }
                backend.endAction(&dynAosCtx);
                // --- End of Nested Dynamic AoS ---

                if (i < aos_size - 1) {
                    staticAosCtx.nextIndex(1);
                }
            }
            backend.endAction(&staticAosCtx);
            // --- End of Static AoS ---

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

            OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP, alconst::timerange_op, tmin, tmax, dtime, alconst::closest_interp);
            backend.beginAction(&opCtx);

            int read_aos_size = 0;
            ArraystructContext staticAosCtx(&opCtx, "static_aos", "");
            backend.beginArraystructAction(&staticAosCtx, &read_aos_size);
            assert(read_aos_size == aos_size);

            for (int i = 0; i < read_aos_size; ++i) {
                std::cout << "  Validating static_aos[" << i << "]...\n";
                
                // Nested Dynamic AoS
                int read_dyn_aos_size = 0;
                ArraystructContext dynAosCtx(&staticAosCtx, "dynamic_aos", "time");
                backend.beginArraystructAction(&dynAosCtx, &read_dyn_aos_size);
                
                std::cout << "    dynamic_aos size: " << read_dyn_aos_size << " (expected 6)\n";
                assert(read_dyn_aos_size == 6); // Indices 5 to 10 (0.5 to 1.0)

                for (int t = 0; t < read_dyn_aos_size; ++t) {
                    void* data_ptr = nullptr;
                    int datatype = alconst::double_data;
                    int dim = 0;
                    int size[H5S_MAX_RANK];

                    backend.readData(&dynAosCtx, "signal_in_dyn_aos", "time", &data_ptr, &datatype, &dim, size);
                    
                    assert(dim == 0);
                    double val = *(double*)data_ptr;
                    
                    // indices 5 to 10. t goes 0..5. global index = 5 + t.
                    double expected = 2000.0 + (i * 100) + (5 + t);
                    if (std::abs(val - expected) > 1e-9) {
                         std::cerr << RED << "Mismatch at static_aos[" << i << "]/dynamic_aos[" << t << "]: expected " << expected << ", got " << val << RESET << std::endl;
                         return 1;
                    }
                    free(data_ptr);

                    if (t < read_dyn_aos_size - 1) dynAosCtx.nextIndex(1);
                }
                backend.endAction(&dynAosCtx);

                if (i < read_aos_size - 1) staticAosCtx.nextIndex(1);
            }
            backend.endAction(&staticAosCtx);

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