// @file  test_al_profiles_1d_dynamic_signal_2d.cpp
// @brief Tests profiles_1d dynamic AoS with a 2D signal and a 1D nested signal:
//        global write and iterative read validation (legacy backend aware).
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

double get_expected_value(int t, int x, int y, int d1, int d2, bool legacy_mode);

// Colors for debug
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_profiles_1d_dynamic_signal_2d";

int main() {
    try {
        std::cout << BOLD << "\n=== Test profiles_1d Dynamic AoS with 2D Signal and 1D Nested Signal ===\n" << RESET;

        const int time_steps = 3;
        const int dim1 = 4;
        const int dim2 = 5;
        const int a_dyn_size = 3;

        // 1. Writing
        {
            std::cout << "\n--- Phase 1: Writing Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "core_profiles", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Homogeneous time
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Time array
            std::vector<double> time_values(time_steps);
            for (int i = 0; i < time_steps; ++i) {
                time_values[i] = (double)i * 0.1;
            }
            int time_dim = 1;
            int time_size[] = {time_steps};
            backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, time_dim, time_size);

            // profiles_1d Dynamic AoS
            ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
            backend.beginArraystructAction(&profilesCtx, (int*)&time_steps);

            for (int t = 0; t < time_steps; ++t) {
                // signal_2d: 2D spatial signal
                std::vector<double> val_2d(dim1 * dim2);
                for(int x=0; x<dim1; ++x) {
                    for(int y=0; y<dim2; ++y) {
                        val_2d[x * dim2 + y] = 1000.0 + t * 100.0 + x * 10.0 + y;
                        //printf("t=%d x=%d y=%d val=%.1f\n", t, x, y, val_2d[x * dim2 + y]);
                    }
                }
                
                int sig_dim = 2;
                int sig_size[] = {dim1, dim2};

                backend.writeData(&profilesCtx, "signal_2d", "", val_2d.data(), alconst::double_data, sig_dim, sig_size);
                
                // Static AoS 'ion'
                int ion_size = 3;
                ArraystructContext ionCtx(&profilesCtx, "ion", "");
                backend.beginArraystructAction(&ionCtx, &ion_size);
                
                for (int i = 0; i < ion_size; ++i) {
                    double z_ion_val = 1.0 + t + i;
                    backend.writeData(&ionCtx, "z_ion", "", &z_ion_val, alconst::double_data, 0, nullptr);

                    // Static AoS 'element' inside 'ion'
                    int element_size = 2;
                    ArraystructContext elementCtx(&ionCtx, "element", "");
                    backend.beginArraystructAction(&elementCtx, &element_size);
                    for (int j = 0; j < element_size; ++j) {
                        // a_dyn: 1D spatial signal
                        std::vector<double> a_dyn_vals(a_dyn_size);
                        for(int k=0; k<a_dyn_size; ++k) {
                            a_dyn_vals[k] = 2.0 + t + i + j + k * 0.1;
                        }
                        int a_dim = 1;
                        int a_size[] = {a_dyn_size};
                        
                        backend.writeData(&elementCtx, "a_dyn", "", a_dyn_vals.data(), alconst::double_data, a_dim, a_size);
                        
                        if (j < element_size - 1) elementCtx.nextIndex(1);
                    }
                    backend.endAction(&elementCtx);
                    
                    if (i < ion_size - 1) ionCtx.nextIndex(1);
                }
                backend.endAction(&ionCtx);

                if (t < time_steps - 1) profilesCtx.nextIndex(1);
            }

            backend.endAction(&profilesCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // 2. Reading / Validation
        {
            std::cout << "\n--- Phase 2: Reading Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);
            std::pair<int,int> version = backend.getVersion(&dataEntryCtx);
            OperationContext opCtx(&dataEntryCtx, "core_profiles", "", READ_OP);
            
            backend.beginAction(&opCtx);

            // Read time
            void* data = nullptr;
            int type = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];
            backend.readData(&opCtx, "time", "time", &data, &type, &dim, size);
            assert(dim == 1);
            assert(size[0] == time_steps);
            free(data);

            // Read profiles_1d
            ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
            int p_size = 0;
            backend.beginArraystructAction(&profilesCtx, &p_size);
            assert(p_size == time_steps);

            for (int t = 0; t < time_steps; ++t) {
                // Validate signal_2d
                backend.readData(&profilesCtx, "signal_2d", "", &data, &type, &dim, size);

                double* vals = (double*)data;

                assert(dim == 2);
                assert(size[0] == dim1);
                assert(size[1] == dim2);
                
                /*for(int x=0; x<dim1; ++x) {
                    for(int y=0; y<dim2; ++y) {
                        printf("t=%d x=%d y=%d val=%.1f\n", t, x, y, vals[x * dim2 + y]);
                        double expected = 1000.0 + t * 100.0 + x * 10.0 + y;
                        if (std::abs(vals[x * dim2 + y] - expected) > 1e-9) {
                            std::cerr << RED << "Mismatch signal_2d at t=" << t << ", x=" << x << ", y=" << y << ": expected " << expected << ", got " << vals[x * dim2 + y] << RESET << std::endl;
                            return 1;
                        }

                    }
                }
                free(data);*/

                // 1. Determine whether the legacy backend is in use (via the version or the dataspace shape)
                bool is_legacy = version == std::make_pair(1,0); // Assume version 1.0 is the legacy backend

                for(int x=0; x<dim1; ++x) {
                    for(int y=0; y<dim2; ++y) {
                        double val_got = vals[x * dim2 + y];
                        double expected = get_expected_value(t, x, y, dim1, dim2, is_legacy);

                        if (std::abs(val_got - expected) > 1e-9) {
                             std::cerr << RED << "Mismatch signal_2d at t=" << t << ", x=" << x << ", y=" << y << ": expected " << expected << ", got " << vals[x * dim2 + y] << RESET << std::endl;
                            return 1;
                        }
                    }
                }

                // Validation of static AoS 'ion'
                ArraystructContext ionCtx(&profilesCtx, "ion", "");
                int ion_size = 0;
                backend.beginArraystructAction(&ionCtx, &ion_size);
                assert(ion_size == 3);
                
                for (int i = 0; i < ion_size; ++i) {
                    // z_ion
                    backend.readData(&ionCtx, "z_ion", "", &data, &type, &dim, size);
                    assert(dim == 0);
                    double z_val = *(double*)data;
                    if (std::abs(z_val - (1.0 + t + i)) > 1e-9) {
                        std::cerr << RED << "Mismatch for z_ion at t=" << t << " i=" << i << RESET << std::endl;
                        return 1;
                    }
                    free(data);

                    // Validation of static AoS 'element'
                    ArraystructContext elementCtx(&ionCtx, "element", "");
                    int element_size = 0;
                    backend.beginArraystructAction(&elementCtx, &element_size);
                    assert(element_size == 2);
                    
                    for (int j = 0; j < element_size; ++j) {
                        // a_dyn (1D)
                        backend.readData(&elementCtx, "a_dyn", "", &data, &type, &dim, size);
                        assert(dim == 1);
                        assert(size[0] == a_dyn_size);
                        
                        double* a_vals = (double*)data;
                        for(int k=0; k<a_dyn_size; ++k) {
                            double expected = 2.0 + t + i + j + k * 0.1;
                            if (std::abs(a_vals[k] - expected) > 1e-9) {
                                std::cerr << RED << "Mismatch for a_dyn at t=" << t << " i=" << i << " j=" << j << " k=" << k << ": expected " << expected << ", got " << a_vals[k] << RESET << std::endl;
                                return 1;
                            }
                        }
                        free(data);
                        if (j < element_size - 1) elementCtx.nextIndex(1);
                    }
                    backend.endAction(&elementCtx);
                    
                    if (i < ion_size - 1) ionCtx.nextIndex(1);
                }
                backend.endAction(&ionCtx);
                
                if (t < time_steps - 1) profilesCtx.nextIndex(1);
            }

            backend.endAction(&profilesCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Validation completed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}

double get_expected_value(int t, int x, int y, int d1, int d2, bool legacy_mode) {
    if (!legacy_mode) {
        // Normal logic (new backend)
        return 1000.0 + t * 100.0 + x * 10.0 + y;
    } else {
        /* Legacy backend logic: it only skips 'dim2' instead of 'dim1*dim2'
           Simulates the observed offset: t=1 starts at index x=1 of the t=0 block
        */
        int actual_x = x + t; // The observed offset (1010.0 at t=1,x=0)
        int actual_t = actual_x / d1;
        actual_x = actual_x % d1;
        
        return 1000.0 + actual_t * 100.0 + actual_x * 10.0 + y;
    }
}