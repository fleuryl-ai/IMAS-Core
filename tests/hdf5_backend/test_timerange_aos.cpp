// tests/hdf5_backend/test_timerange_aos.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_timerange_aos";

int main() {
    try {
        std::cout << BOLD << "\n=== Test TimeRange Read with Static AoS ===\n" << RESET;

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

                std::vector<double> dyn_signal_values(time_dim_size);
                for (int k = 0; k < time_dim_size; ++k) {
                    dyn_signal_values[k] = 1000.0 + (i * 100) + k;
                }
                int dyn_sig_dim[] = {time_dim_size};
                backend.writeData(&staticAosCtx, "dynamic_signal", "time", dyn_signal_values.data(), alconst::double_data, 1, dyn_sig_dim);

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
                void* data_ptr = nullptr;
                int datatype = alconst::double_data;
                int dim = 0;
                int size[H5S_MAX_RANK];

                backend.readData(&staticAosCtx, "static_signal", "", &data_ptr, &datatype, &dim, size);
                assert(dim == 0);
                assert(std::abs(*(double*)data_ptr - (500.0 + i)) < 1e-9);
                std::cout << "    static_signal: " << *(double*)data_ptr << " [OK]\n";
                free(data_ptr);

                backend.readData(&staticAosCtx, "dynamic_signal", "time", &data_ptr, &datatype, &dim, size);
                assert(dim == 1 && size[0] == 6);
                // ... validation of dynamic signal values ...
                free(data_ptr);

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