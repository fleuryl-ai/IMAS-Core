// @file  test_al_profiles_1d_dynamic_append_slice_large.cpp
// @brief Profiling test for profiles_1d dynamic AoS with large data: initial global
//        write of signals, strings and nested ion/element AoS, then slice appends.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <cstring>

namespace fs = std::filesystem;

// Colors for debug
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_profiles_1d_dynamic_append_slice_large";

int main() {
    try {
        std::cout << BOLD << "\n=== Test profiles_1d Dynamic AoS PROFILING (Large Data) ===\n" << RESET;

        const int initial_time_steps = 1000;
        const int spatial_size = 200;

        // 1. Writing Initial Data (Phase 1)
        {
            std::cout << "\n--- Phase 1: Writing Initial Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            auto version = backend.getVersion(&dataEntryCtx);
            if (version == std::make_pair(1,0)) {
                backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
                return 0;
            } 

            OperationContext opCtx(&dataEntryCtx, "core_profiles", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Homogeneous time
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Time array
            std::vector<double> time_values(initial_time_steps);
            for (int i = 0; i < initial_time_steps; ++i) {
                time_values[i] = (double)i * 0.1;
            }
            int time_dim = 1;
            int time_size[] = {initial_time_steps};
            backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, time_dim, time_size);

            // profiles_1d Dynamic AoS
            ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
            backend.beginArraystructAction(&profilesCtx, (int*)&initial_time_steps);

            for (int t = 0; t < initial_time_steps; ++t) {
                // signal_1d
                std::vector<double> val(spatial_size);
                for(int x=0; x<spatial_size; ++x) {
                    val[x] = 100.0 + t * 10.0 + x;
                }
                int sig_dim = 1;
                int sig_size[] = {spatial_size};
                backend.writeData(&profilesCtx, "signal_1d", "time", val.data(), alconst::double_data, sig_dim, sig_size);
                
                // string_0d
                std::string str_val = "String_" + std::to_string(t);
                int str_dim = 1;
                int str_size_arr[] = {(int)str_val.length()};
                backend.writeData(&profilesCtx, "string_0d", "time", (void*)str_val.c_str(), alconst::char_data, str_dim, str_size_arr);

                // string_1d
                int str_1d_len = 32;
                int str_1d_dim = 2;
                int str_1d_size_arr[] = {spatial_size, str_1d_len};
                std::vector<char> str_1d_buf(spatial_size * str_1d_len, 0);
                for(int x=0; x<spatial_size; ++x) {
                    std::string s = "S_" + std::to_string(t) + "_" + std::to_string(x);
                    strncpy(str_1d_buf.data() + x * str_1d_len, s.c_str(), str_1d_len - 1);
                }
                backend.writeData(&profilesCtx, "string_1d", "time", str_1d_buf.data(), alconst::char_data, str_1d_dim, str_1d_size_arr);

                // matrix_2d (New signal for volume)
                int mat_width = 5;
                int mat_dim = 2;
                int mat_size_arr[] = {spatial_size, mat_width};
                std::vector<double> mat_val(spatial_size * mat_width);
                for(int x=0; x<spatial_size * mat_width; ++x) {
                    mat_val[x] = (double)x * 0.01 + t;
                }
                backend.writeData(&profilesCtx, "matrix_2d", "time", mat_val.data(), alconst::double_data, mat_dim, mat_size_arr);

                // ion
                int ion_size = 5;
                ArraystructContext ionCtx(&profilesCtx, "ion", "");
                backend.beginArraystructAction(&ionCtx, &ion_size);
                
                for (int i = 0; i < ion_size; ++i) {
                    double z_ion_val = 1.0 + t + i;
                    backend.writeData(&ionCtx, "z_ion", "", &z_ion_val, alconst::double_data, 0, nullptr);

                    // element
                    int element_size = 5;
                    ArraystructContext elementCtx(&ionCtx, "element", "");
                    backend.beginArraystructAction(&elementCtx, &element_size);
                    for (int j = 0; j < element_size; ++j) {
                        double a_dyn_val = 2.0 + t + i + j;
                        backend.writeData(&elementCtx, "a_dyn", "", &a_dyn_val, alconst::double_data, 0, nullptr);
                        double b_dyn_val = 3.0 + t + i + j;
                        backend.writeData(&elementCtx, "b_dyn", "", &b_dyn_val, alconst::double_data, 0, nullptr);
                        if (j < element_size - 1) elementCtx.nextIndex(1);
                    }
                    backend.endAction(&elementCtx);
                    if (i < ion_size - 1) ionCtx.nextIndex(1);
                }
                backend.endAction(&ionCtx);

                if (t < initial_time_steps - 1) profilesCtx.nextIndex(1);
            }

            backend.endAction(&profilesCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Phase 1 completed.\n" << RESET;
        }

                // 2. Appending Slices (Phase 2)
        {
            std::cout << "\n--- Phase 2: Appending Slices ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;

            std::vector<double> append_times = {0.3, 0.4};
            
            for (double t_append : append_times) {
                std::cout << "Appending slice at t=" << t_append << "...\n";
                backend.openPulse(&dataEntryCtx, OPEN_PULSE);

                int interpmode = alconst::closest_interp;
                OperationContext opCtx(&dataEntryCtx, "core_profiles", WRITE_OP, alconst::slice_op, t_append, interpmode);
                backend.beginAction(&opCtx);

                // Write time
                double t_val = t_append;
                backend.writeData(&opCtx, "time", "time", &t_val, alconst::double_data, 0, nullptr);

                int t_idx = (int)(t_append * 10.0 + 0.5);

                // profiles_1d
                int p_size = 1;
                ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
                // Note: No need to call setIndex(t_idx) here; PanzerDB manages the append index in APPEND mode
                backend.beginArraystructAction(&profilesCtx, &p_size);

                // ion
                int ion_size = 5;
                ArraystructContext ionCtx(&profilesCtx, "ion", "");
                backend.beginArraystructAction(&ionCtx, &ion_size);
                
                for (int i = 0; i < ion_size; ++i) {
                    // element
                    int element_size = 5;
                    ArraystructContext elementCtx(&ionCtx, "element", "");
                    backend.beginArraystructAction(&elementCtx, &element_size);
                    for (int j = 0; j < element_size; ++j) {
                        double a_dyn_val = 2.0 + t_idx + i + j;
                        backend.writeData(&elementCtx, "a_dyn", "", &a_dyn_val, alconst::double_data, 0, nullptr);
                        if (j < element_size - 1) elementCtx.nextIndex(1);
                    }
                    backend.endAction(&elementCtx);
                    if (i < ion_size - 1) ionCtx.nextIndex(1);
                }
                backend.endAction(&ionCtx);

                backend.endAction(&profilesCtx);
                backend.endAction(&opCtx);
                backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            }
            std::cout << GREEN << "[OK] Phase 2 completed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}