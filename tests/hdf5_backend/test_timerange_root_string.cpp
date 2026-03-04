// tests/hdf5_backend/test_timerange_root_string.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_timerange_root_string";

int main() {
    try {
        std::cout << BOLD << "\n=== Test TimeRange Read with Root Static String ===\n" << RESET;

        const char* str_value = "Static_Root_String";
        const int time_dim_size = 20;
        const int new_aos_size = 2;
        const double dbl_scalar_base = 123.45;
        const char* infra_name_val = "MyInfrastructure";

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

            // Write static string at root
            int str_dim = 1;
            int str_size[] = {(int)strlen(str_value)};
            backend.writeData(&opCtx, "static_string", "", (void*)str_value, alconst::char_data, str_dim, str_size);

            // --- Ajout d'un nouvel AoS statique ---
            ArraystructContext newAosCtx(&opCtx, "new_static_aos", "");
            backend.beginArraystructAction(&newAosCtx, (int*)&new_aos_size);

            for (int i = 0; i < new_aos_size; ++i) {
                // Écriture d'un scalaire double
                double dbl_val = dbl_scalar_base + i;
                backend.writeData(&newAosCtx, "double_scalar", "", &dbl_val, alconst::double_data, 0, nullptr);

                // Écriture d'une chaîne de caractères
                std::string aos_str_val = "String_in_AOS_" + std::to_string(i);
                int aos_str_dim = 1;
                int aos_str_size[] = {(int)aos_str_val.length()};
                backend.writeData(&newAosCtx, "string_in_aos", "", (void*)aos_str_val.c_str(), alconst::char_data, aos_str_dim, aos_str_size);

                if (i < new_aos_size - 1) {
                    newAosCtx.nextIndex(1);
                }
            }
            backend.endAction(&newAosCtx);

            // Écriture du champ statique ids_properties/plugins/infrastructure_get/name
            int infra_dim = 1;
            int infra_size[] = {(int)strlen(infra_name_val)};
            backend.writeData(&opCtx, "ids_properties/plugins/infrastructure_get/name", "", (void*)infra_name_val, alconst::char_data, infra_dim, infra_size);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // 2. Reading (Time Range)
        {
            std::cout << "\n--- Phase 2: Reading Time Range ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double tmin = 0.5;
            double tmax = 1.0;
            std::vector<double> dtime; // Empty for no resampling

            OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP, alconst::timerange_op, tmin, tmax, dtime, alconst::closest_interp);
            backend.beginAction(&opCtx);

            void* data_ptr = nullptr;
            int datatype = alconst::char_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            backend.readData(&opCtx, "static_string", "", &data_ptr, &datatype, &dim, size);
            
            // For a static scalar string, readData returns dim=1, size[0]=length
            std::cout << "    Read dim: " << dim << ", size[0]: " << size[0] << "\n";
            assert(dim == 1);
            assert(size[0] == strlen(str_value));

            char* char_data = (char*)data_ptr;
            std::string val(char_data); // char_data is null-terminated by backend
            std::cout << "    Value: " << val << " (expected " << str_value << ")\n";
            assert(val == str_value);
            
            delete[] char_data;

            // --- Validation du nouvel AoS statique ---
            int read_aos_size = 0;
            ArraystructContext newAosCtx(&opCtx, "new_static_aos", "");
            backend.beginArraystructAction(&newAosCtx, &read_aos_size);
            assert(read_aos_size == new_aos_size);

            for (int i = 0; i < read_aos_size; ++i) {
                std::cout << "  Validating new_static_aos[" << i << "]...\n";
                void* aos_data_ptr = nullptr;
                int aos_datatype;
                int aos_dim;
                int aos_size_arr[H5S_MAX_RANK];

                // Valider le scalaire double
                aos_datatype = alconst::double_data;
                backend.readData(&newAosCtx, "double_scalar", "", &aos_data_ptr, &aos_datatype, &aos_dim, aos_size_arr);
                assert(aos_dim == 0);
                assert(std::abs(*(double*)aos_data_ptr - (dbl_scalar_base + i)) < 1e-9);
                std::cout << "    double_scalar: " << *(double*)aos_data_ptr << " [OK]\n";
                free(aos_data_ptr);

                // Valider la chaîne de caractères
                aos_datatype = alconst::char_data;
                backend.readData(&newAosCtx, "string_in_aos", "", &aos_data_ptr, &aos_datatype, &aos_dim, aos_size_arr);
                assert(aos_dim == 1);
                std::string expected_str = "String_in_AOS_" + std::to_string(i);
                assert(aos_size_arr[0] == expected_str.length());
                std::string read_str((char*)aos_data_ptr);
                assert(read_str == expected_str);
                std::cout << "    string_in_aos: " << read_str << " [OK]\n";
                delete[] (char*)aos_data_ptr;

                if (i < read_aos_size - 1) newAosCtx.nextIndex(1);
            }
            backend.endAction(&newAosCtx);

            // Validation du champ statique ids_properties/plugins/infrastructure_get/name
            void* infra_ptr = nullptr;
            int infra_datatype = alconst::char_data;
            int infra_dim = 0;
            int infra_size_arr[H5S_MAX_RANK];
            backend.readData(&opCtx, "ids_properties/plugins/infrastructure_get/name", "", &infra_ptr, &infra_datatype, &infra_dim, infra_size_arr);
            assert(infra_dim == 1);
            assert(infra_size_arr[0] == strlen(infra_name_val));
            std::string read_infra((char*)infra_ptr);
            assert(read_infra == infra_name_val);
            std::cout << "    ids_properties/plugins/infrastructure_get/name: " << read_infra << " [OK]\n";
            delete[] (char*)infra_ptr;

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