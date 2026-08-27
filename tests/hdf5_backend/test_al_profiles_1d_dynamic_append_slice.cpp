// tests/hdf5_backend/test_profiles_1d_dynamic_append_slice.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_profiles_1d_dynamic_append_slice";

int main() {
    try {
        std::cout << BOLD << "\n=== Test profiles_1d Dynamic AoS Append Slice ===\n" << RESET;

        const int initial_time_steps = 3;
        const int spatial_size = 10;

        // 1. Writing Initial Data (Phase 1)
        {
            std::cout << "\n--- Phase 1: Writing Initial Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            auto version = backend.getVersion(&dataEntryCtx);
            //std::cout << "Backend version: " << version.first << "." << version.second << std::endl;

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

                // ion
                int ion_size = 3;
                ArraystructContext ionCtx(&profilesCtx, "ion", "");
                backend.beginArraystructAction(&ionCtx, &ion_size);
                
                for (int i = 0; i < ion_size; ++i) {
                    double z_ion_val = 1.0 + t + i;
                    backend.writeData(&ionCtx, "z_ion", "", &z_ion_val, alconst::double_data, 0, nullptr);

                    // element
                    int element_size = 2;
                    ArraystructContext elementCtx(&ionCtx, "element", "");
                    backend.beginArraystructAction(&elementCtx, &element_size);
                    for (int j = 0; j < element_size; ++j) {
                        double a_dyn_val = 2.0 + t + i + j;
                        backend.writeData(&elementCtx, "a_dyn", "", &a_dyn_val, alconst::double_data, 0, nullptr);
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
                // Note: Pas besoin de setIndex(t_idx) ici, PanzerDB gère l'index d'ajout en mode APPEND
                backend.beginArraystructAction(&profilesCtx, &p_size);

                // ion
                int ion_size = 3;
                ArraystructContext ionCtx(&profilesCtx, "ion", "");
                backend.beginArraystructAction(&ionCtx, &ion_size);
                
                for (int i = 0; i < ion_size; ++i) {
                    // element
                    int element_size = 2;
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

        // 3. Validation (Phase 3)
        {
            std::cout << "\n--- Phase 3: Validation of Appended Slices ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            std::vector<double> target_times = {0.3, 0.4};
            
            for (double target_time : target_times) {
                std::cout << "Validating slice at t=" << target_time << "...\n";
                int interpmode = alconst::closest_interp;
                OperationContext opCtx(&dataEntryCtx, "core_profiles", READ_OP, alconst::slice_op, target_time, interpmode);
                backend.beginAction(&opCtx);

                int t_idx = (int)(target_time * 10.0 + 0.5);

                ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
                int p_size = 0;
                backend.beginArraystructAction(&profilesCtx, &p_size);
                assert(p_size == 1);

                ArraystructContext ionCtx(&profilesCtx, "ion", "");
                int ion_size = 0;
                backend.beginArraystructAction(&ionCtx, &ion_size);
                assert(ion_size == 3);
                
                for (int i = 0; i < ion_size; ++i) {
                    ArraystructContext elementCtx(&ionCtx, "element", "");
                    int element_size = 0;
                    backend.beginArraystructAction(&elementCtx, &element_size);
                    assert(element_size == 2);
                    
                    for (int j = 0; j < element_size; ++j) {
                        void* data = nullptr;
                        int type = alconst::double_data;
                        int dim = 0;
                        int size[H5S_MAX_RANK];
                        
                        backend.readData(&elementCtx, "a_dyn", "", &data, &type, &dim, size);
                        if (!data) {
                            std::cerr << RED << "Read failed (data is null) for a_dyn at t=" << target_time << " i=" << i << " j=" << j << RESET << std::endl;
                            return 1;
                        }
                        assert(dim == 0);
                        double a_val = *(double*)data;
                        double expected_a = 2.0 + t_idx + i + j;
                        if (std::abs(a_val - expected_a) > 1e-9) {
                            std::cerr << RED << "Mismatch for a_dyn at t=" << target_time << " i=" << i << " j=" << j << ": expected " << expected_a << ", got " << a_val << RESET << std::endl;
                            return 1;
                        }
                        free(data);
                        if (j < element_size - 1) elementCtx.nextIndex(1);
                    }
                    backend.endAction(&elementCtx);
                    
                    if (i < ion_size - 1) ionCtx.nextIndex(1);
                }
                backend.endAction(&ionCtx);
                backend.endAction(&profilesCtx);
                backend.endAction(&opCtx);
            }
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Phase 3 completed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}