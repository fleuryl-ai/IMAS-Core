// tests/hdf5_backend/test_bug_string_list_1d.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

// Colors for debug
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_bug_string_list_1d";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Bug Reproduction: List of Strings (1D vs 2D) ===\n" << RESET;

        // Clean up
        if (fs::exists("test_db_bug_string_list_1d")) {
            fs::remove_all("test_db_bug_string_list_1d");
        }

        // ===================================================================
        // 1. Écriture (simule ids_put)
        // ===================================================================
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
            int coord_sys_size = 1;
            ArraystructContext coordSysCtx(&opCtx, "coordinate_system", "");
            backend.beginArraystructAction(&coordSysCtx, &coord_sys_size);

            for (int i = 0; i < coord_sys_size; ++i) {
                int coord_size = 1;
                ArraystructContext coordCtx(&coordSysCtx, "coordinate", "");
                backend.beginArraystructAction(&coordCtx, &coord_size);

                for (int j = 0; j < coord_size; ++j) {
                    // Cas qui pose problème : Une liste de strings contenant UN SEUL élément.
                    // Dans IMAS, c'est un tableau 1D de strings, donc 2D de chars [1, len].
                    const char* single_string = "SingleLabel";
                    int n_strings = 1;
                    int max_len = strlen(single_string) + 1; 

                    // Preparation du buffer pour writeData (flat char array)
                    // dim = 2, size = {n_strings, max_len}
                    int dim = 2;
                    int size[] = {n_strings, max_len};
                    
                    // Allocation et copie
                    std::vector<char> buffer(n_strings * max_len, 0);
                    strncpy(buffer.data(), single_string, max_len);

                    std::cout << "Écriture de 'value_labels' (liste de 1 string)...\n";
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
        // 2. Lecture (simule ids_get) et Vérification
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
            
            ArraystructContext coordCtx(&coordSysCtx, "coordinate", "");
            int c_size = 0;
            backend.beginArraystructAction(&coordCtx, &c_size);

            std::cout << "Lecture de 'value_labels'...\n";
            
            void* data = nullptr;
            int datatype = alconst::char_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            backend.readData(&coordCtx, "value_labels", "", &data, &datatype, &dim, size);
            
            std::cout << "Dimensions retournées: " << dim << "D [";
            for(int k=0; k<dim; ++k) std::cout << size[k] << (k<dim-1 ? " x " : "");
            std::cout << "]\n";
            
            if (dim == 1) {
                std::cout << RED << ">>> BUG REPRODUIT: dim=1 pour une liste de strings (attendu: 2)\n";
                std::cout << "    Le backend retourne un scalaire string au lieu d'une liste de taille 1.\n" << RESET;
            } else if (dim == 2) {
                std::cout << GREEN << ">>> SUCCÈS: dim=2. Le bug semble corrigé ou non reproduit.\n" << RESET;
            } else {
                std::cout << RED << ">>> RÉSULTAT INATTENDU: dim=" << dim << "\n" << RESET;
            }

            if (data) free(data);

            backend.endAction(&coordCtx);
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