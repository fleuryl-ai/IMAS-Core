// tests/hdf5_backend/test_profiles_1d_dynamic.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_profiles_1d_dynamic";

int main() {
    try {
        std::cout << BOLD << "\n=== Test profiles_1d Dynamic AoS (Homogeneous Time) ===\n" << RESET;

        const int time_steps = 10;

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
            // profiles_1d est un tableau de structures indexé par le temps.
            ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
            backend.beginArraystructAction(&profilesCtx, (int*)&time_steps);

            for (int t = 0; t < time_steps; ++t) {
                // Signal inside profiles_1d. Let's call it "electrons/density_thermal".
                // C'est un scalaire à chaque pas de temps (donc dimension 0 pour la slice).
                double val = 100.0 + t * 1.5;
                
                // On écrit la valeur pour l'instant t courant
                backend.writeData(&profilesCtx, "electrons/density_thermal", "time", &val, alconst::double_data, 0, nullptr);
                
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
                backend.readData(&profilesCtx, "electrons/density_thermal", "time", &data, &type, &dim, size);
                
                // En lecture globale itérative sur un AoS, on lit l'élément courant.
                // Ici, c'est un scalaire (la valeur du signal à l'instant t).
                assert(dim == 0);
                double val = *(double*)data;
                double expected = 100.0 + t * 1.5;
                if (std::abs(val - expected) > 1e-9) {
                    std::cerr << RED << "Mismatch at t=" << t << ": expected " << expected << ", got " << val << RESET << std::endl;
                    return 1;
                }
                free(data);
                
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
