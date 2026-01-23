// test_equilibrium_deep_nested_aos_fixed.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

namespace fs = std::filesystem;

const std::string URI = "imas:hdf5?path=./test_db_test_equilibrium_deep";

// ==============================================================
// CONFIGURATION DES TAILLES (à modifier ici)
// ==============================================================
int TIME_SIZE = 3;    // time_slice
int PROFILE_SIZE = 1; // profiles_1d
int GRID_SIZE = 1;    // grid
int SEGMENT_SIZE = 1; // segment
int VALUES_SIZE = 10; // values (fixe)

int main() {
  try {
    std::cout << "=== DEBUT Test equilibrium : AOS imbriqués + tailles fixes + "
                 "debug ===\n";
    std::cout << "[CONFIG] time_slice=" << TIME_SIZE
              << " | profiles_1d=" << PROFILE_SIZE << " | grid=" << GRID_SIZE
              << " | segment=" << SEGMENT_SIZE << " | values=" << VALUES_SIZE
              << "\n\n";

    // ==============================================================
    // 1. Contextes
    // ==============================================================
    DataEntryContext dataEntryCtx(URI);
    std::cout << "[DEBUG] DataEntryContext créé\n";

    // ==============================================================
    // 2. Backend + Ouverture
    // ==============================================================
    HDF5Backend backend;
    std::cout << "[DEBUG] Avant openPulse\n";
    backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);
    std::cout << "[DEBUG] openPulse OK\n";

    OperationContext opCtx(&dataEntryCtx, "equilibrium", "", WRITE_OP);
    std::cout << "[DEBUG] Avant beginAction(opCtx)\n";
    backend.beginAction(&opCtx);
    std::cout << "[DEBUG] beginAction(opCtx) OK\n";

    // ==============================================================
    // 3. homogeneous_time = 1
    // ==============================================================
    int homogeneous_time = 1;
    std::cout << "[DEBUG] Écriture ids_properties/homogeneous_time\n";
    backend.writeData(&opCtx, "ids_properties/homogeneous_time", "",
                      &homogeneous_time, alconst::integer_data, 0, nullptr);
    std::cout << "[DEBUG] homogeneous_time écrit\n";

    // ==============================================================
    // 4. Structure imbriquée
    // ==============================================================
    ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
    std::cout << "[DEBUG] Avant beginArraystructAction(time_slice, size="
              << TIME_SIZE << ")\n";
    backend.beginArraystructAction(&timeSliceCtx, &TIME_SIZE);
    std::cout << "[DEBUG] beginArraystructAction(time_slice) OK\n";

    for (int t = 0; t < TIME_SIZE; ++t) {
      std::cout << "[DEBUG] === time_slice[" << t << "] ===\n";

      ArraystructContext profileCtx(&timeSliceCtx, "profiles_1d", "time");
      std::cout << "[DEBUG]   Avant beginArraystructAction(profiles_1d, size="
                << PROFILE_SIZE << ")\n";
      backend.beginArraystructAction(&profileCtx, &PROFILE_SIZE);
      std::cout << "[DEBUG]   beginArraystructAction(profiles_1d) OK\n";

      for (int p = 0; p < PROFILE_SIZE; ++p) {
        std::cout << "[DEBUG]     profile[" << p << "]\n";

        ArraystructContext gridCtx(&profileCtx, "grid", "");
        std::cout << "[DEBUG]     Avant beginArraystructAction(grid, size="
                  << GRID_SIZE << ")\n";
        backend.beginArraystructAction(&gridCtx, &GRID_SIZE);
        std::cout << "[DEBUG]     beginArraystructAction(grid) OK\n";

        for (int g = 0; g < GRID_SIZE; ++g) {
          std::cout << "[DEBUG]       grid[" << g << "]\n";

          ArraystructContext segmentCtx(&gridCtx, "segment", "");
          std::cout
              << "[DEBUG]       Avant beginArraystructAction(segment, size="
              << SEGMENT_SIZE << ")\n";
          backend.beginArraystructAction(&segmentCtx, &SEGMENT_SIZE);
          std::cout << "[DEBUG]       beginArraystructAction(segment) OK\n";

          for (int s = 0; s < SEGMENT_SIZE; ++s) {
            std::cout << "[DEBUG]         segment[" << s
                      << "]: écriture values[" << VALUES_SIZE << "]\n";

            std::vector<double> values(VALUES_SIZE);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(0.0, 1.0);

            // Dans la boucle :
            for (int i = 0; i < VALUES_SIZE; ++i) {
              values[i] = dis(gen); // bruit blanc → incompressible
            }

            /*for (int i = 0; i < VALUES_SIZE; ++i) {
                values[i] = 10000.0 +
                            t * 1000.0 +
                            p * 100.0 +
                            g * 10.0 +
                            s * 1.0 +
                            std::sin(i * 0.01) * 50.0;
            }*/

            int dim = 1;
            int shape[1] = {VALUES_SIZE};
            backend.writeData(&segmentCtx, "values", "", values.data(),
                              alconst::double_data, dim, shape);

            std::cout << "[DEBUG]         writeData(values) OK\n";
            segmentCtx.nextIndex(1);
          }

          std::cout << "[DEBUG]       Avant endAction(segment)\n";
          backend.endAction(&segmentCtx);
          std::cout << "[DEBUG]       endAction(segment) OK\n";

          gridCtx.nextIndex(1);
        }

        std::cout << "[DEBUG]     Avant endAction(grid)\n";
        backend.endAction(&gridCtx);
        std::cout << "[DEBUG]     endAction(grid) OK\n";
        profileCtx.nextIndex(1);
      }

      std::cout << "[DEBUG]   Avant endAction(profiles_1d)\n";
      backend.endAction(&profileCtx);
      std::cout << "[DEBUG]   endAction(profiles_1d) OK\n";
      timeSliceCtx.nextIndex(1);
    }

    std::cout << "[DEBUG] Avant endAction(time_slice)\n";
    backend.endAction(&timeSliceCtx);
    std::cout << "[DEBUG] endAction(time_slice) OK\n";

    std::cout << "[DEBUG] Avant endAction(opCtx)\n";
    backend.endAction(&opCtx);
    std::cout << "[DEBUG] endAction(opCtx) OK\n";

    std::cout << "Écriture terminée.\n\n";

    // ==============================================================
    // 5. Vérification
    // ==============================================================
    std::string ids_file =
        std::string("./test_db_test_equilibrium_deep") + "/equilibrium.h5";
    if (!fs::exists(ids_file)) {
      std::cerr << "ERREUR : Fichier non créé : " << ids_file << "\n";
      return 1;
    } else {
      std::cout << "[OK] Fichier créé : " << ids_file << "\n";
      auto file_size = fs::file_size(ids_file) / (1024.0 * 1024.0);
      std::cout << "[INFO] Taille : " << file_size << " Mo\n";
    }

    std::cout << "\n=== TEST RÉUSSI ===\n";
  } catch (const std::exception &e) {
    std::cerr << "Exception : " << e.what() << std::endl;
    return 1;
  }
  return 0;
}