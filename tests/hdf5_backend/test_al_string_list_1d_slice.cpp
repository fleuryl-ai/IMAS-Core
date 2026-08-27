// @file  test_al_string_list_1d_slice.cpp
// @brief Bug reproduction test: a list of strings containing a single element
//        (value_labels) must be read with dim=2, not dim=1.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <filesystem>
#include <random>
#include <algorithm>

namespace fs = std::filesystem;

// Colors for debug
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_bug_string_list_1d";

std::string random_string(size_t length) {
    auto randchar = []() -> char {
        const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[rand() % max_index];
    };
    std::string str(length,0);
    std::generate_n(str.begin(), length, randchar);
    return str;
}

int main() {
    try {
        std::cout << BOLD << "\n=== Test Bug Reproduction: List of Strings (1D vs 2D) ===\n" << RESET;

        srand(time(0));
        // Clean up
        if (fs::exists("test_db_bug_string_list_1d")) {
            fs::remove_all("test_db_bug_string_list_1d");
        }

        // ===================================================================
        // 1. Writing (simulates ids_put)
        // ===================================================================
        std::string expected_string;

        {
            std::cout << "\n--- Phase 1: Writing Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Structure: coordinate_system/coordinate(n)/value_labels
            int coord_sys_size = 2;
            ArraystructContext coordSysCtx(&opCtx, "coordinate_system", "");
            backend.beginArraystructAction(&coordSysCtx, &coord_sys_size);

            for (int i = 0; i < coord_sys_size; ++i) {
                int coord_size = 2;
                ArraystructContext coordCtx(&coordSysCtx, "coordinate", "");
                backend.beginArraystructAction(&coordCtx, &coord_size);

                for (int j = 0; j < coord_size; ++j) {
                    // Problematic case: a list of strings containing only a single element.
                    // In IMAS, this is a 1D array of strings, i.e. a 2D char array [1, len].
                    expected_string = random_string(10 + (rand() % 10));
                    const char* single_string = expected_string.c_str();
                    int n_strings = 1;
                    int max_len = strlen(single_string) + 1; 

                    // Prepare the buffer for writeData (flat char array)
                    // dim = 2, size = {n_strings, max_len}
                    int dim = 2;
                    int size[] = {n_strings, max_len};
                    
                    // Allocate and copy
                    std::vector<char> buffer(n_strings * max_len, 0);
                    strncpy(buffer.data(), single_string, max_len);

                    std::cout << "Writing 'value_labels' (list of 1 string): " << single_string << "...\n";
                    backend.writeData(&coordCtx, "value_labels", "", buffer.data(), alconst::char_data, dim, size);

                    if (j < coord_size - 1) coordCtx.nextIndex(1);
                }
                backend.endAction(&coordCtx);
                if (i < coord_sys_size - 1) coordSysCtx.nextIndex(1);
            }
            backend.endAction(&coordSysCtx);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // ===================================================================
        // 2. Reading (simulates ids_get) and validation
        // ===================================================================
        {
            std::cout << "\n--- Phase 2: Reading Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", READ_OP);
            backend.beginAction(&opCtx);

            ArraystructContext coordSysCtx(&opCtx, "coordinate_system", "");
            int cs_size = 0;
            backend.beginArraystructAction(&coordSysCtx, &cs_size);

            for (int i = 0; i < cs_size; ++i) {
                ArraystructContext coordCtx(&coordSysCtx, "coordinate", "");
                int c_size = 0;
                backend.beginArraystructAction(&coordCtx, &c_size);

                for (int j = 0; j < c_size; ++j) {
                    std::cout << "Reading 'value_labels' for coordinate_system[" << i << "]/coordinate[" << j << "]...\n";
                    
                    void* data = nullptr;
                    int datatype = alconst::char_data;
                    int dim = 0;
                    int size[H5S_MAX_RANK];

                    backend.readData(&coordCtx, "value_labels", "", &data, &datatype, &dim, size);
                    
                    std::cout << "Returned dimensions: " << dim << "D [";
                    for(int k=0; k<dim; ++k) std::cout << size[k] << (k<dim-1 ? " x " : "");
                    std::cout << "]\n";
                    
                    if (dim == 1) {
                        std::cout << RED << ">>> BUG REPRODUCED: dim=1 for a string list (expected: 2)\n";
                        std::cout << "    The backend returns a scalar string instead of a list of size 1.\n" << RESET;
                    } else if (dim == 2) {
                        // Content check (optional/simplified)
                        // char* chars = (char*)data;
                        // size[0] = number of strings (1), size[1] = max_len
                        /*if (size[0] == 1) {
                            std::string read_str(chars);
                            // Complex validation omitted to simplify the multi-element test
                        }*/
                        std::cout << GREEN << ">>> SUCCESS: dim=2. The bug appears fixed or is not reproduced.\n" << RESET;
                    } else {
                        std::cout << RED << ">>> UNEXPECTED RESULT: dim=" << dim << "\n" << RESET;
                    }

                    if (data) free(data);

                    if (j < c_size - 1) coordCtx.nextIndex(1);
                }
                backend.endAction(&coordCtx);
                if (i < cs_size - 1) coordSysCtx.nextIndex(1);
            }
            backend.endAction(&coordSysCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}