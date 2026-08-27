// @file  test_al_core_sources_put_slice.cpp
// @brief Put-slice regression test for core_sources with nested AoS
//        (source / profiles_1d / ion), verifying the written files exist.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

const std::string URI = "imas:hdf5?path=./test_db_test_core_sources";

int main() {
  try {
    std::cout
        << "=== Test core_sources: nested AoS + putSlice (FIXED) ===\n";

    int s = 5, t = 3;
    std::vector<double> time(t), vect1DDouble(t);
    for (int i = 0; i < t; ++i) {
      time[i] = 1.0 * i;
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
    // std::string comment = "This is a test IDS in C++";
    // backend.writeData(&opCtx, "ids_properties/comment", "", comment.c_str(),
    // alconst::char_data, 0, nullptr);
    backend.endAction(&opCtx1);

    double undefined_time = alconst::undefined_time;
    int interpmode = alconst::undefined_interp;
    OperationContext opCtx(&dataEntryCtx, "core_sources", WRITE_OP,
                           alconst::slice_op, undefined_time, interpmode);
    // std::cout << "Adresse de operation context    : " << &(opCtx) <<
    // std::endl;

    // OperationContext opCtx(&dataEntryCtx, "core_sources", "", WRITE_OP);
    backend.beginAction(&opCtx);

    bool first_slice = true;

    auto *sourceCtx = new ArraystructContext(&opCtx, "source", "");
    contexts.push_back(std::unique_ptr<ArraystructContext>(sourceCtx));
    backend.beginArraystructAction(sourceCtx, &s);

    for (int i = 0; i < s; ++i) {

      // ArraystructContext profilesCtx(&sourceCtx, "profiles_1d", "time");
      auto *profilesCtx =
          new ArraystructContext(sourceCtx, "profiles_1d", "time");
      contexts.push_back(std::unique_ptr<ArraystructContext>(profilesCtx));
      int profiles_size = t;
      backend.beginArraystructAction(profilesCtx, &profiles_size);

      for (int j = 0; j < t; ++j) {

        /*double nd_time = alconst::undefined_time;
        int interpmode = alconst::undefined_interp;
        OperationContext sliceCtx(&dataEntryCtx, "core_sources", WRITE_OP,
        alconst::slice_op, nd_time, interpmode);
        backend.beginAction(&sliceCtx);*/
        fprintf(stderr, "Writing core_sources.source[%d].profiles_1d[%d]\n", i,
                j);

        // int size_time[1] = {1};
        // backend.writeData(&opCtx, "time", "", &time[j], alconst::double_data,
        // 1,
        //                   size_time);
        //  backend.endAction(&sliceCtx);

        for (int k = 0; k < j; ++k) {
          int size_rho_thor_norm[1] = {t};
          backend.writeData(profilesCtx, "grid/rho_thor_norm", "",
                            &vect1DDouble[0], alconst::double_data, 1,
                            size_rho_thor_norm);
        }

        // profiles_1d.time
        // backend.writeData(profilesCtx, "time", "", &time[j],
        // alconst::double_data, 0, nullptr);

        // ArraystructContext ionCtx(&ionCtx, "ion", "");
        auto *ionCtx = new ArraystructContext(profilesCtx, "ion", "");
        contexts.push_back(std::unique_ptr<ArraystructContext>(ionCtx));
        // ion loop
        backend.beginArraystructAction(ionCtx, &i);
        for (int k = 0; k < i; ++k) {
          // backend.writeData(ionCtx, "z_ion", "", &time[k],
          // alconst::double_data,
          //                   0, nullptr);

          // particles
          // ArraystructContext particlesCtx(&ionCtx, "particles", "");
          //   auto *particlesCtx = new ArraystructContext(ionCtx, "particles",
          //   ""); backend.beginArraystructAction(particlesCtx, &j);

          //   for (int l = 0; l < j; ++l) {
          //     double val = 2.0 * l + k;
          //     backend.writeData(particlesCtx, "particles", "", &val,
          //                       alconst::double_data, 0, nullptr);
          //     particlesCtx->nextIndex(1);
          //   }
          //   backend.endAction(particlesCtx);

          // state
          // ArraystructContext stateCtx(&ionCtx, "state", "");
          /*auto *stateCtx = new ArraystructContext(ionCtx, "state", "");
          int state_size = 10;
          backend.beginArraystructAction(stateCtx, &state_size);
          for (int l = 0; l < 10; ++l) {
            double z_min = l;
            backend.writeData(stateCtx, "z_min", "", &z_min,
                              alconst::double_data, 0, nullptr);
            stateCtx->nextIndex(1);
          }
          backend.endAction(stateCtx);*/

          ionCtx->nextIndex(1);
        }
        backend.endAction(ionCtx);
        profilesCtx->nextIndex(1);
      }
      backend.endAction(profilesCtx);
      sourceCtx->nextIndex(1); // Correct: after source[i]
    }

    backend.endAction(sourceCtx);
    backend.endAction(&opCtx);

    std::cout << "core_sources IDS saved\n";

    std::string ids_file = "./test_db_test_core_sources/core_sources.h5";
    if (!fs::exists(ids_file)) {
      std::cerr << "ERROR: file was not created\n";
      return 1;
    }
    std::cout << "[OK] File created: " << ids_file << "\n";

    std::cout << "\n=== TEST SUCCESSFUL ===\n";
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
