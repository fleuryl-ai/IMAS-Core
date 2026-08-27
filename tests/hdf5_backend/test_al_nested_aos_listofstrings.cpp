// @file  test_al_nested_aos_listofstrings.cpp
// @brief Tests a nested static AoS (coordinate_system > coordinate) holding a
//        list of strings per element, written and read back.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <cstring> // For strncpy

namespace fs = std::filesystem;

// Colors for debug
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_nested_aos_char_data";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Nested Static AoS with List of Strings ===\n" << RESET;

        const int coordinate_system_size = 1; // Parent AoS
        const int coordinate_size = 2;        // Child AoS
        const int string_list_count = 3;      // Number of strings in the list
        const int max_string_len = 32;        // Max length for each string

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

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Homogeneous time (required by some logic even if not used for static)
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Static AoS 'coordinate_system'
            ArraystructContext coordinateSystemCtx(&opCtx, "coordinate_system", "");
            backend.beginArraystructAction(&coordinateSystemCtx, (int*)&coordinate_system_size);

            for (int cs_idx = 0; cs_idx < coordinate_system_size; ++cs_idx) {
                // Static AoS 'coordinate'
                ArraystructContext coordinateCtx(&coordinateSystemCtx, "coordinate", "");
                backend.beginArraystructAction(&coordinateCtx, (int*)&coordinate_size);

                for (int c_idx = 0; c_idx < coordinate_size; ++c_idx) {
                    // List of 3 strings (alconst::char_data, dim=2)
                    int char_data_dim = 2;
                    int char_data_size_arr[] = {string_list_count, max_string_len};
                    std::vector<char> char_data_buf(string_list_count * max_string_len, 0); // Initialize with nulls

                    for(int s_idx = 0; s_idx < string_list_count; ++s_idx) {
                        std::string s = "CS_" + std::to_string(cs_idx) + "_C_" + std::to_string(c_idx) + "_Str_" + std::to_string(s_idx);
                        // Ensure null-termination and prevent buffer overflow
                        strncpy(char_data_buf.data() + s_idx * max_string_len, s.c_str(), max_string_len - 1);
                    }
                    backend.writeData(&coordinateCtx, "my_string_list", "", char_data_buf.data(), alconst::char_data, char_data_dim, char_data_size_arr);
                    
                    if (c_idx < coordinate_size - 1) coordinateCtx.nextIndex(1);
                }
                backend.endAction(&coordinateCtx);

                if (cs_idx < coordinate_system_size - 1) coordinateSystemCtx.nextIndex(1);
            }
            backend.endAction(&coordinateSystemCtx);

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

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", READ_OP);
            backend.beginAction(&opCtx);

            // Read 'coordinate_system'
            ArraystructContext coordinateSystemCtx(&opCtx, "coordinate_system", "");
            int cs_read_size = 0;
            backend.beginArraystructAction(&coordinateSystemCtx, &cs_read_size);
            assert(cs_read_size == coordinate_system_size);

            for (int cs_idx = 0; cs_idx < cs_read_size; ++cs_idx) {
                // Read 'coordinate'
                ArraystructContext coordinateCtx(&coordinateSystemCtx, "coordinate", "");
                int c_read_size = 0;
                backend.beginArraystructAction(&coordinateCtx, &c_read_size);
                assert(c_read_size == coordinate_size);

                for (int c_idx = 0; c_idx < c_read_size; ++c_idx) {
                    void* data = nullptr;
                    int type = alconst::char_data;
                    int dim = 0;
                    int size[H5S_MAX_RANK];

                    backend.readData(&coordinateCtx, "my_string_list", "", &data, &type, &dim, size);
                    
                    assert(dim == 2); // Should be a list of strings
                    assert(size[0] == string_list_count); // Number of strings
                    
                    char* char_data_ptr = (char*)data;
                    int read_max_len = size[1]; // Max length of strings read back

                    for(int s_idx = 0; s_idx < string_list_count; ++s_idx) {
                        std::string read_s(char_data_ptr + s_idx * read_max_len);
                        std::string expected_s = "CS_" + std::to_string(cs_idx) + "_C_" + std::to_string(c_idx) + "_Str_" + std::to_string(s_idx);
                        
                        if (read_s != expected_s) {
                            std::cerr << RED << "Mismatch for my_string_list at CS[" << cs_idx << "]/C[" << c_idx << "]/Str[" << s_idx << "]: expected '" << expected_s << "', got '" << read_s << "'" << RESET << std::endl;
                            return 1;
                        }
                    }
                    free(data); // Free memory allocated by readData

                    if (c_idx < c_read_size - 1) coordinateCtx.nextIndex(1);
                }
                backend.endAction(&coordinateCtx);

                if (cs_idx < cs_read_size - 1) coordinateSystemCtx.nextIndex(1);
            }
            backend.endAction(&coordinateSystemCtx);

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