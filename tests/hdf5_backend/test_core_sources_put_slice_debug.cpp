// test_core_sources_aos_fixed.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

const std::string URI = "imas:hdf5?path=./test_db_test_core_sources_debug";

int main() {
  try {
    std::cout
        << "=== Test core_sources : AOS imbriqués + putSlice (CORRIGÉ) ===\n";

    int s = 2, t = 3;
    std::vector<double> time(t), vect1DDouble(t);
    for (int i = 0; i < t; ++i) {
      time[i] = 1.0 * (i + 1);
      vect1DDouble[i] = time[i] * 10.0;
    }

    std::vector<std::unique_ptr<ArraystructContext>> contexts;

    DataEntryContext dataEntryCtx(URI);
    HDF5Backend backend;
    backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

    OperationContext opCtx1(&dataEntryCtx, "core_sources", "", WRITE_OP);
    backend.beginAction(&opCtx1);
    int homogeneous_time = 1;
    backend.writeData(&opCtx1, "ids_properties/homogeneous_time", "",
                      &homogeneous_time, alconst::integer_data, 0, nullptr);

    backend.endAction(&opCtx1);

    double undefined_time = alconst::undefined_time;
    int interpmode = alconst::undefined_interp;
    OperationContext opCtx(&dataEntryCtx, "core_sources", WRITE_OP,
                           alconst::slice_op, undefined_time, interpmode);
    backend.beginAction(&opCtx);

    auto *sourceCtx = new ArraystructContext(&opCtx, "source", "");
    contexts.push_back(std::unique_ptr<ArraystructContext>(sourceCtx));
    backend.beginArraystructAction(sourceCtx, &s);

    for (int i = 0; i < s; ++i) {

      auto *profilesCtx =
          new ArraystructContext(sourceCtx, "profiles_1d", "time");
      contexts.push_back(std::unique_ptr<ArraystructContext>(profilesCtx));
      int profiles_size = t;
      backend.beginArraystructAction(profilesCtx, &profiles_size);

      for (int j = 0; j < t; ++j) {

        fprintf(stderr, "Ecriture core_sources.source[%d].profiles_1d[%d]\n", i,
                j);

        // Debug print: show axis sizes before writing ion data
        fprintf(stderr, "[DEBUG] source axis size: %d\n", s);
        fprintf(stderr, "[DEBUG] profiles_1d axis size: %d\n", t);
        int p = 3;
        fprintf(stderr, "[DEBUG] ion axis size: %d\n", p);

        auto *ionCtx = new ArraystructContext(profilesCtx, "ion", "");
        contexts.push_back(std::unique_ptr<ArraystructContext>(ionCtx));
        backend.beginArraystructAction(ionCtx, &p);
        // Prepare explicit ion values (avoid reading time[k] when p > t)
        std::vector<double> ion_values(p);
        for (int k = 0; k < p; ++k)
          ion_values[k] = (double)(k + 1);
        for (int k = 0; k < p; ++k) {
          backend.writeData(ionCtx, "z_ion", "", &ion_values[k],
                            alconst::double_data, 0, nullptr);
          ionCtx->nextIndex(1);
        }
        backend.endAction(ionCtx);
        profilesCtx->nextIndex(1);
      }
      backend.endAction(profilesCtx);
      sourceCtx->nextIndex(1); // Correct : après source[i]
    }

    backend.endAction(sourceCtx);

    int size_time = 3;
    double time_value[3]  = {1.0, 2.0, 3.0};
    backend.writeData(&opCtx, "time", "time", &time_value , alconst::double_data, 1,
        &size_time); fprintf(stderr, "  Fin Ecriture vect_1d_double \n");
    backend.endAction(&opCtx);

    std::cout << "core_sources IDS saved\n";

    std::string ids_file = "./test_db_test_core_sources_debug/core_sources.h5";
    if (!fs::exists(ids_file)) {
      std::cerr << "ERREUR : Fichier non créé\n";
      return 1;
    }
    std::cout << "[OK] Fichier créé : " << ids_file << "\n";

    std::cout << "\n=== TEST RÉUSSI ===\n";
  } catch (const std::exception &e) {
    std::cerr << "Exception : " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
