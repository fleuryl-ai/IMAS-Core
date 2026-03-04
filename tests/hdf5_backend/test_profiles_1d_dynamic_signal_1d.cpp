// tests/hdf5_backend/test_profiles_1d_dynamic_signal_1d.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_profiles_1d_dynamic_signal_1d";

int main() {
    try {

        std::cout << BOLD << "\n=== Test profiles_1d Dynamic AoS with 1D Signals (Homogeneous Time) ===\n" << RESET;

        const int time_steps = 3;
        const int spatial_size = 10;

        // 1. Writing
        {
            std::cout << "\n--- Phase 1: Writing Data ---\n";
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
            std::vector<double> time_values(time_steps);
            for (int i = 0; i < time_steps; ++i) {
                time_values[i] = (double)i * 0.1;
            }
            int time_dim = 1;
            int time_size[] = {time_steps};
            backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, time_dim, time_size);

            // profiles_1d Dynamic AoS
            // profiles_1d est un tableau de structures indexé par le temps.
            ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
            backend.beginArraystructAction(&profilesCtx, (int*)&time_steps);

            for (int t = 0; t < time_steps; ++t) {
                // Signal inside profiles_1d. Let's call it "signal_1d".
                // C'est un tableau 1D spatial (taille 10) à chaque pas de temps.
                // Pour écrire une slice temporelle d'un tableau 1D, on doit passer dim=2 et size={spatial, 1}
                // pour indiquer qu'on écrit 1 slice de taille spatiale donnée.
                
                std::vector<double> val(spatial_size);
                for(int x=0; x<spatial_size; ++x) {
                    val[x] = 100.0 + t * 10.0 + x;
                }
                int sig_dim = 1;
                int sig_size[] = {spatial_size};
                // On écrit la valeur pour l'instant t courant
                backend.writeData(&profilesCtx, "signal_1d", "", val.data(), alconst::double_data, sig_dim, sig_size);

                // Ajout d'un deuxième signal 1D spatial ("signal_1d_bis")
                std::vector<double> val_bis(spatial_size);
                for(int x=0; x<spatial_size; ++x) {
                    val_bis[x] = 500.0 + t * 20.0 + x * 2.0;
                }
                backend.writeData(&profilesCtx, "signal_1d_bis", "", val_bis.data(), alconst::double_data, sig_dim, sig_size);
                
                // Ajout signal 0D char_data 'string_0d'
                std::string str_val = "String_" + std::to_string(t);
                int str_dim = 1;
                int str_size_arr[] = {(int)str_val.length()};
                backend.writeData(&profilesCtx, "string_0d", "", (void*)str_val.c_str(), alconst::char_data, str_dim, str_size_arr);

                // Ajout signal 1D char_data 'string_1d'
                int str_1d_len = 32;
                int str_1d_dim = 2;
                int str_1d_size_arr[] = {spatial_size, str_1d_len};
                std::vector<char> str_1d_buf(spatial_size * str_1d_len, 0);
                for(int x=0; x<spatial_size; ++x) {
                    std::string s = "S_" + std::to_string(t) + "_" + std::to_string(x);
                    strncpy(str_1d_buf.data() + x * str_1d_len, s.c_str(), str_1d_len - 1);
                }
                backend.writeData(&profilesCtx, "string_1d", "", str_1d_buf.data(), alconst::char_data, str_1d_dim, str_1d_size_arr);

                // Ajout AOS statique 'ion' avec donnée 0D 'z_ion'
                int ion_size = 3;
                ArraystructContext ionCtx(&profilesCtx, "ion", "");
                backend.beginArraystructAction(&ionCtx, &ion_size);
                
                for (int i = 0; i < ion_size; ++i) {
                    double z_ion_val = 1.0 + t + i;
                    backend.writeData(&ionCtx, "z_ion", "", &z_ion_val, alconst::double_data, 0, nullptr);

                    // Ajout AOS statique 'element' dans 'ion' avec donnée 0D 'a_dyn'
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
            double* t_vals = (double*)data;
            for(int i=0; i<time_steps; ++i) {
                assert(std::abs(t_vals[i] - (i * 0.1)) < 1e-9);
            }
            free(data);

            // Read profiles_1d
            ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
            int p_size = 0;
            backend.beginArraystructAction(&profilesCtx, &p_size);
            assert(p_size == time_steps);

            for (int t = 0; t < time_steps; ++t) {
                backend.readData(&profilesCtx, "signal_1d", "", &data, &type, &dim, size);
                
                // En lecture globale itérative sur un AoS, on lit l'élément courant (la slice).
                // Ici, c'est un tableau 1D spatial.
                //printf("dim=%d size[0]=%d\n", dim, size[0]);
                assert(dim == 1);
                assert(size[0] == spatial_size);
                double* vals = (double*)data;
                for(int x=0; x<spatial_size; ++x) {
                    double expected = 100.0 + t * 10.0 + x;
                    if (std::abs(vals[x] - expected) > 1e-9) {
                        std::cerr << RED << "Mismatch at t=" << t << ", x=" << x << ": expected " << expected << ", got " << vals[x] << RESET << std::endl;
                        return 1;
                    }
                }
                free(data);

                // Validate signal_1d_bis
                backend.readData(&profilesCtx, "signal_1d_bis", "", &data, &type, &dim, size);
                assert(dim == 1);
                assert(size[0] == spatial_size);
                
                double* vals_bis = (double*)data;
                for(int x=0; x<spatial_size; ++x) {
                    double expected = 500.0 + t * 20.0 + x * 2.0;
                    if (std::abs(vals_bis[x] - expected) > 1e-9) {
                        std::cerr << RED << "Mismatch for signal_1d_bis at t=" << t << ", x=" << x << ": expected " << expected << ", got " << vals_bis[x] << RESET << std::endl;
                        return 1;
                    }
                }
                free(data);

                // Validation string_0d
                void* str_data = nullptr;
                int str_type = alconst::char_data;
                int str_dim = 0;
                int str_size_arr[H5S_MAX_RANK];
                backend.readData(&profilesCtx, "string_0d", "", &str_data, &str_type, &str_dim, str_size_arr);
                assert(str_dim == 1);
                std::string read_str((char*)str_data);
                std::string expected_str = "String_" + std::to_string(t);
                if (read_str != expected_str) {
                    std::cerr << RED << "Mismatch for string_0d at t=" << t << ": expected " << expected_str << ", got " << read_str << RESET << std::endl;
                    return 1;
                }
                delete[] (char*)str_data; // GlobalReadStrategy uses new for char data

                // Validation string_1d
                void* str_1d_data = nullptr;
                int str_1d_type = alconst::char_data;
                int str_1d_rdim = 0;
                int str_1d_rsize[H5S_MAX_RANK];
                backend.readData(&profilesCtx, "string_1d", "", &str_1d_data, &str_1d_type, &str_1d_rdim, str_1d_rsize);
                assert(str_1d_rdim == 2);
                assert(str_1d_rsize[0] == spatial_size);
                int r_len = str_1d_rsize[1];
                char* r_buf = (char*)str_1d_data;
                for(int x=0; x<spatial_size; ++x) {
                    std::string read_s(r_buf + x * r_len);
                    std::string expected_s = "S_" + std::to_string(t) + "_" + std::to_string(x);
                    if (read_s != expected_s) {
                         std::cerr << RED << "Mismatch string_1d t=" << t << " x=" << x << " exp=" << expected_s << " got=" << read_s << RESET << std::endl;
                         return 1;
                    }
                }
                delete[] (char*)str_1d_data;

                // Validation AOS statique 'ion'
                ArraystructContext ionCtx(&profilesCtx, "ion", "");
                int ion_size = 0;
                backend.beginArraystructAction(&ionCtx, &ion_size);
                assert(ion_size == 3);
                
                for (int i = 0; i < ion_size; ++i) {
                    void* z_data = nullptr;
                    backend.readData(&ionCtx, "z_ion", "", &z_data, &type, &dim, size);
                    assert(dim == 0);
                    double z_val = *(double*)z_data;
                    if (std::abs(z_val - (1.0 + t + i)) > 1e-9) {
                        std::cerr << RED << "Mismatch for z_ion at t=" << t << " i=" << i << ": expected " << (1.0 + t + i) << ", got " << z_val << RESET << std::endl;
                        return 1;
                    }
                    free(z_data);

                    // Validation AOS statique 'element'
                    ArraystructContext elementCtx(&ionCtx, "element", "");
                    int element_size = 0;
                    backend.beginArraystructAction(&elementCtx, &element_size);
                    assert(element_size == 2);
                    
                    for (int j = 0; j < element_size; ++j) {
                        void* a_data = nullptr;
                        backend.readData(&elementCtx, "a_dyn", "", &a_data, &type, &dim, size);
                        assert(dim == 0);
                        double a_val = *(double*)a_data;
                        if (std::abs(a_val - (2.0 + t + i + j)) > 1e-9) {
                            std::cerr << RED << "Mismatch for a_dyn at t=" << t << " i=" << i << " j=" << j << ": expected " << (2.0 + t + i + j) << ", got " << a_val << RESET << std::endl;
                            return 1;
                        }
                        free(a_data);
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

        // 3. Reading Slices
        {
            std::cout << "\n--- Phase 3: Reading Slices ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            std::vector<double> target_times = {0.0, 0.2};

            for (double target_time : target_times) {
                std::cout << "Reading slice at t=" << target_time << "...\n";
                int interpmode = alconst::closest_interp;
                OperationContext opCtx(&dataEntryCtx, "core_profiles", READ_OP, alconst::slice_op, target_time, interpmode);
                backend.beginAction(&opCtx);

                int t_idx = (int)(target_time * 10.0 + 0.5);

                ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
                int p_size = 0;
                backend.beginArraystructAction(&profilesCtx, &p_size);
                assert(p_size == 1);

                void* data = nullptr;
                int type = alconst::double_data;
                int dim = 0;
                int size[H5S_MAX_RANK];
                backend.readData(&profilesCtx, "signal_1d", "", &data, &type, &dim, size);
                printf("dim=%d size[0]=%d\n", dim, size[0]);
                assert(dim == 1);
                assert(size[0] == spatial_size);
                
                double* vals = (double*)data;
                for(int x=0; x<spatial_size; ++x) {
                    double expected = 100.0 + t_idx * 10.0 + x;
                    if (std::abs(vals[x] - expected) > 1e-9) {
                        std::cerr << RED << "Mismatch for signal_1d at t=" << target_time << ", x=" << x << ": expected " << expected << ", got " << vals[x] << RESET << std::endl;
                        return 1;
                    }
                }
                free(data);

                // Validate signal_1d_bis (Slice)
                backend.readData(&profilesCtx, "signal_1d_bis", "", &data, &type, &dim, size);
                assert(dim == 1);
                assert(size[0] == spatial_size);
                
                double* vals_bis = (double*)data;
                for(int x=0; x<spatial_size; ++x) {
                    double expected = 500.0 + t_idx * 20.0 + x * 2.0;
                    if (std::abs(vals_bis[x] - expected) > 1e-9) {
                        std::cerr << RED << "Mismatch for signal_1d_bis at t=" << target_time << ", x=" << x << ": expected " << expected << ", got " << vals_bis[x] << RESET << std::endl;
                        return 1;
                    }
                }
                free(data);

                // Validation string_0d (Slice)
                void* str_data = nullptr;
                int str_type = alconst::char_data;
                int str_dim = 0;
                int str_size_arr[H5S_MAX_RANK];
                backend.readData(&profilesCtx, "string_0d", "", &str_data, &str_type, &str_dim, str_size_arr);
                assert(str_dim == 1);
                std::string read_str_slice((char*)str_data);
                std::string expected_str_slice = "String_" + std::to_string(t_idx);
                if (read_str_slice != expected_str_slice) {
                    std::cerr << RED << "Mismatch for string_0d at t=" << target_time << ": expected " << expected_str_slice << ", got " << read_str_slice << RESET << std::endl;
                    return 1;
                }
                free(str_data); // SliceReadStrategy uses malloc via panzerdb

                // Validation string_1d (Slice)
                void* str_1d_data = nullptr;
                int str_1d_type = alconst::char_data;
                int str_1d_rdim = 0;
                int str_1d_rsize[H5S_MAX_RANK];
                backend.readData(&profilesCtx, "string_1d", "", &str_1d_data, &str_1d_type, &str_1d_rdim, str_1d_rsize);
                assert(str_1d_rdim == 2);
                assert(str_1d_rsize[0] == spatial_size);
                int r_len_slice = str_1d_rsize[1];
                char* r_buf_slice = (char*)str_1d_data;
                for(int x=0; x<spatial_size; ++x) {
                    std::string read_s(r_buf_slice + x * r_len_slice);
                    std::string expected_s = "S_" + std::to_string(t_idx) + "_" + std::to_string(x);
                    if (read_s != expected_s) {
                         std::cerr << RED << "Mismatch string_1d t=" << target_time << " x=" << x << " exp=" << expected_s << " got=" << read_s << RESET << std::endl;
                         return 1;
                    }
                }
                free(str_1d_data);

                ArraystructContext ionCtx(&profilesCtx, "ion", "");
                int ion_size = 0;
                backend.beginArraystructAction(&ionCtx, &ion_size);
                assert(ion_size == 3);
                
                for (int i = 0; i < ion_size; ++i) {
                    backend.readData(&ionCtx, "z_ion", "", &data, &type, &dim, size);
                    assert(dim == 0);
                    double z_val = *(double*)data;
                    double expected_z = 1.0 + t_idx + i;
                    if (std::abs(z_val - expected_z) > 1e-9) {
                        std::cerr << RED << "Mismatch for z_ion at t=" << target_time << " i=" << i << ": expected " << expected_z << ", got " << z_val << RESET << std::endl;
                        return 1;
                    }
                    free(data);

                    ArraystructContext elementCtx(&ionCtx, "element", "");
                    int element_size = 0;
                    backend.beginArraystructAction(&elementCtx, &element_size);
                    assert(element_size == 2);
                    
                    for (int j = 0; j < element_size; ++j) {
                        backend.readData(&elementCtx, "a_dyn", "", &data, &type, &dim, size);
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
            std::cout << GREEN << "[OK] Slice Reading completed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}
