// tests/hdf5_backend/test_timerange_aos_string_list.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_timerange_aos_string_list";

int main() {
    try {
        std::cout << BOLD << "\n=== Test TimeRange Read with Static AoS containing String List ===\n" << RESET;

        const int aos_size = 1;
        const int str_list_size = 3;
        const int max_len = 10;
        // Création d'un buffer plat pour simuler le comportement standard de l'AL
        char str_values_flat[3][10];
        memset(str_values_flat, 0, 3*10);
        strcpy(str_values_flat[0], "Static_1");
        strcpy(str_values_flat[1], "Static_2");
        strcpy(str_values_flat[2], "Static_3");

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

            // --- Start of Static AoS ---
            ArraystructContext staticAosCtx(&opCtx, "static_aos", "");
            backend.beginArraystructAction(&staticAosCtx, (int*)&aos_size);

            for (int i = 0; i < aos_size; ++i) {
                // Write static string list
                int str_dim = 2;
                int str_size[] = {str_list_size, max_len}; // {count, max_len}
                backend.writeData(&staticAosCtx, "static_string_list", "", (void*)str_values_flat, alconst::char_data, str_dim, str_size);

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

            int read_aos_size = 0;
            ArraystructContext staticAosCtx(&opCtx, "static_aos", "");
            backend.beginArraystructAction(&staticAosCtx, &read_aos_size);
            assert(read_aos_size == aos_size);

            for (int i = 0; i < read_aos_size; ++i) {
                std::cout << "  Validating static_aos[" << i << "]...\n";
                void* data_ptr = nullptr;
                int datatype = alconst::char_data;
                int dim = 0;
                int size[H5S_MAX_RANK];

                backend.readData(&staticAosCtx, "static_string_list", "", &data_ptr, &datatype, &dim, size);
                
                // The global read for a static array of strings returns a flat char* buffer.
                // The AL convention for this is dim=2, with size={max_len, count}.
                
                std::cout << "    Read dim: " << dim << ", size[0]: " << size[0] << ", size[1]: " << size[1] << "\n";
                assert(dim == 2);
                assert(size[0] == str_list_size); // size[0] is count

                char* char_data = (char*)data_ptr;
                int max_len_read = size[1]; // size[1] is max_len

                for(int k=0; k<str_list_size; ++k) {
                    std::string val(char_data + k * max_len_read);
                    std::cout << "    [" << k << "] " << val << " (expected " << str_values_flat[k] << ")\n";
                    assert(val == str_values_flat[k]);
                }
                delete[] char_data;

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