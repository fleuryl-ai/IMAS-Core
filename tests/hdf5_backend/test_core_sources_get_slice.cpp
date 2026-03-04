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
        << "=== Test core_sources : AOS imbriqués + getSlice (CORRIGÉ) ===\n";

    //int s = 1, t = 2;
    /*std::vector<double> time(t), vect1DDouble(t);
    for (int i = 0; i < t; ++i) {
      time[i] = 1.0 * (i + 1);
      vect1DDouble[i] = time[i] * 10.0;
    }*/

    int s, t;


    std::vector<std::unique_ptr<ArraystructContext>> contexts;

    DataEntryContext dataEntryCtx(URI);
    HDF5Backend backend;
    backend.openPulse(&dataEntryCtx, OPEN_PULSE);

    /*double undefined_time = 2.0;
    int interpmode = alconst::undefined_interp;
    OperationContext opCtx(&dataEntryCtx, "core_sources", READ_OP,
                           alconst::slice_op, undefined_time, 1);*/
    OperationContext opCtx(&dataEntryCtx, "core_sources", "", READ_OP);
    backend.beginAction(&opCtx);

    auto *sourceCtx = new ArraystructContext(&opCtx, "source", "");
    contexts.push_back(std::unique_ptr<ArraystructContext>(sourceCtx));
    backend.beginArraystructAction(sourceCtx, &s);

    fprintf(stderr, "[DEBUG] source axis size: %d\n", s);

    for (int i = 0; i < s; ++i) {

      auto *profilesCtx = new ArraystructContext(sourceCtx, "profiles_1d", "");
      contexts.push_back(std::unique_ptr<ArraystructContext>(profilesCtx));
      int profiles_size;
      backend.beginArraystructAction(profilesCtx, &profiles_size);
      fprintf(stderr, "[DEBUG] profiles_1d axis size: %d\n", profiles_size);

      for (int j = 0; j < profiles_size; ++j) {

        fprintf(stderr, "Lecture core_sources.source[%d].profiles_1d[%d]\n", i,
                j);
        
        int p;
        
        /*int size_time[1] = {1};
        fprintf(stderr, "  Ecriture time = %f \n", time[j]);
        backend.writeData(&opCtx, "time", "", &time[j], alconst::double_data, 1,
        size_time); fprintf(stderr, "  Fin Ecriture vect_1d_double \n");*/

        auto *ionCtx = new ArraystructContext(profilesCtx, "ion", "");
        contexts.push_back(std::unique_ptr<ArraystructContext>(ionCtx));
        backend.beginArraystructAction(ionCtx, &p);
        fprintf(stderr, "[DEBUG] ion axis size: %d\n", p);

        // Prepare explicit ion values (avoid reading time[k] when p > t)
        //std::vector<double> ion_values;
        double *ion_values = nullptr;
        for (int k = 0; k < p; ++k) {
            int size;
            int dim = 0;
            int datatype = alconst::double_data;
            //readData(Context * ctx, std::string fieldname, std::string timebasename, void **data, int *datatype, int *dim, int *size)
            backend.readData(ionCtx, "z_ion", "", (void**) &ion_values, &datatype, &dim, &size);
            fprintf(stderr, "size = %d\n", size);
            //fprintf(stderr, "    z_ion[%d] = %f\n", k, ion_values[k]);
            ionCtx->nextIndex(1);
        }
        backend.endAction(ionCtx);
        profilesCtx->nextIndex(1);
      }
      backend.endAction(profilesCtx);
      sourceCtx->nextIndex(1); // Correct : après source[i]
    }

    backend.endAction(sourceCtx);
    backend.endAction(&opCtx);

    /*std::cout << "core_sources IDS saved\n";

    std::string ids_file = "./test_db_test_core_sources_debug/core_sources.h5";
    if (!fs::exists(ids_file)) {
      std::cerr << "ERREUR : Fichier non créé\n";
      return 1;
    }
    std::cout << "[OK] Fichier créé : " << ids_file << "\n";
*/
    std::cout << "\n=== TEST RÉUSSI ===\n";
  } catch (const std::exception &e) {
    std::cerr << "Exception : " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
