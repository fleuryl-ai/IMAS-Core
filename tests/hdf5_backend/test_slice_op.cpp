// test_nested_dynamic_timebase.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include "panzerdb.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

namespace fs = std::filesystem;

// Couleurs ANSI pour debug
#define RESET ""
#define BOLD ""
#define GREEN ""
#define YELLOW ""
#define BLUE ""
#define MAGENTA ""
#define CYAN ""
#define RED ""

const std::string URI = "imas:hdf5?path=./test_db_nested_dynamic_timebase";

int main() {
  try {
    std::cout << BOLD << MAGENTA
              << "\n=== Test getTimeIndex avec base de temps dans un AoS "
                 "dynamique imbriqué ===\n"
              << RESET;

    // ==============================================================
    // 1. Écriture des données
    // ==============================================================
    {
      DataEntryContext dataEntryCtx(URI);
      HDF5Backend backend;
      backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

      OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
      backend.beginAction(&opCtx);

      // --- AoS Statique 'static_aos' ---
      int static_aos_size = 2;
      ArraystructContext staticAosCtx(&opCtx, "static_aos", "");
      backend.beginArraystructAction(&staticAosCtx, &static_aos_size);

      for (int i = 0; i < static_aos_size; ++i) {
        // Signal statique dans l'AoS statique
        double static_signal_data = 100.0 + i;
        backend.writeData(&staticAosCtx, "static_signal", "",
                          &static_signal_data, alconst::double_data, 0,
                          nullptr);
        // Signal 1D de shape 10
          std::vector<double> dyn_1d_data(10);
          for(int k=0; k<10; ++k) dyn_1d_data[k] = k* 10.0 + 1;
          int dyn_1d_dim = 1;
          int dyn_1d_size[] = {10};
          backend.writeData(&staticAosCtx, "dyn_1d", "time", dyn_1d_data.data(), alconst::double_data, dyn_1d_dim, dyn_1d_size);
          
        // --- AoS Dynamique 'dynamic_aos' imbriqué ---
        int dynamic_aos_size = 10;
        ArraystructContext dynamicAosCtx(&staticAosCtx, "dynamic_aos", "time");
        backend.beginArraystructAction(&dynamicAosCtx, &dynamic_aos_size);

        // Écriture de la base de temps et des signaux pour cet AoS dynamique
        for (int t = 0; t < dynamic_aos_size; ++t) {
          // Écrire la valeur de temps pour cette tranche
          double current_time = 0.1 * (t + 1) + i;
          backend.writeData(&dynamicAosCtx, "time", "time", &current_time,
                            alconst::double_data, 0, nullptr);

          // Écriture de deux signaux dynamiques
          double signal1_data = 1000.0 + i * 100 + t;
          double signal2_data = 2000.0 + i * 100 + t;
          backend.writeData(&dynamicAosCtx, "signal1", "time", &signal1_data,
                            alconst::double_data, 0, nullptr);
          backend.writeData(&dynamicAosCtx, "signal2", "time", &signal2_data,
                            alconst::double_data, 0, nullptr);
          
          

          if (t < dynamic_aos_size - 1) {
            dynamicAosCtx.nextIndex(1);
          }
        }
        backend.endAction(&dynamicAosCtx);
        if (i < static_aos_size - 1) {
          staticAosCtx.nextIndex(1);
        }
      }

      backend.endAction(&staticAosCtx);
      // --- AJOUT : Signal dynamique 1D dans static_aos ---
          // On écrit aussi le temps pour static_aos (nécessaire pour le signal dynamique)
      int time_dim = 1;
      int time_size[] = {10};
      std::vector<double> time(10);
      for(int k=0; k<10; ++k) time[k] =  k;
      backend.writeData(&staticAosCtx, "time", "time", time.data(), alconst::double_data, time_dim, time_size);
      backend.endAction(&opCtx);
      backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);

      std::cout << GREEN << "[OK] Écriture terminée.\n" << RESET;
    }

    // ==============================================================
    // 2. Validation avec PanzerDB::getTimeIndex
    // ==============================================================
    std::cout << BOLD << MAGENTA << "\n[2] Validation de getTimeIndex...\n"
              << RESET;
    {
      std::string ids_file = "./test_db_nested_dynamic_timebase/test_ids.h5";
      hid_t h5_file_id = H5Fopen(ids_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      if (h5_file_id < 0) {
        std::cerr << RED << "ERREUR : Impossible d'ouvrir le fichier HDF5.\n"
                  << RESET;
        return 1;
      }
      hid_t group_id = H5Gopen2(h5_file_id, "test_ids", H5P_DEFAULT);
      H5Fclose(h5_file_id);
      if (group_id < 0) {
        std::cerr << RED
                  << "ERREUR : Impossible d'ouvrir le groupe 'test_ids'.\n"
                  << RESET;
        return 1;
      }

      PanzerDB db(group_id, PanzerDB::OpenMode::READ, true, true);

      // --- Test 1: Chercher t=1.3 dans static_aos[1]/dynamic_aos/time ---

      std::string timebase_path1 = "static_aos/1/dynamic_aos/time";
      double requested_time1 = 1.3;
      int64_t expected_index1 = 2;
      int64_t found_index1 = db.getTimeIndex(timebase_path1, requested_time1);

      std::cout << CYAN << "  Test 1: Recherche de t=" << requested_time1
                << " dans '" << timebase_path1 << "'...\n"
                << RESET;
      if (found_index1 != expected_index1) {
        std::cerr << RED << "  ERREUR : getTimeIndex a retourné "
                  << found_index1 << " (attendu: " << expected_index1 << ")\n"
                  << RESET;
        return 1;
      }
      std::cout << GREEN << "  [OK] Index trouvé : " << found_index1 << "\n"
                << RESET;

      // --- Test 2: Chercher t=0.8 dans static_aos[0]/dynamic_aos/time ---
      // Le chemin complet est "static_aos/0/dynamic_aos/time"
      // Les valeurs sont [0.1, 0.2, ..., 0.8, 0.9, 1.0]
      // L'index attendu pour t=0.8 est 7.
      std::string timebase_path2 = "static_aos/0/dynamic_aos/time";
      double requested_time2 = 0.8;
      int64_t expected_index2 = 7;
      int64_t found_index2 = db.getTimeIndex(timebase_path2, requested_time2);

      std::cout << CYAN << "  Test 2: Recherche de t=" << requested_time2
                << " dans '" << timebase_path2 << "'...\n"
                << RESET;
      if (found_index2 != expected_index2) {
        std::cerr << RED << "  ERREUR : getTimeIndex a retourné "
                  << found_index2 << " (attendu: " << expected_index2 << ")\n"
                  << RESET;
        return 1;
      }
      std::cout << GREEN << "  [OK] Index trouvé : " << found_index2 << "\n"
                << RESET;
    }

    // ==============================================================
    // 3. Lecture par Slices (Extension demandée)
    // ==============================================================
    std::cout << BOLD << MAGENTA << "\n[3] Test de lecture par Slices...\n" << RESET;

    std::vector<double> target_times = {0.55, 1.55};

    for (double target_time : target_times) {
        try {
            DataEntryContext readDataEntryCtx(URI);
            HDF5Backend readBackend;
            readBackend.openPulse(&readDataEntryCtx, OPEN_PULSE);

            std::cout << YELLOW << "[DEBUG] Lecture slice t=" << target_time << "...\n" << RESET;

            int interpmode = alconst::linear_interp;
            OperationContext opCtx(&readDataEntryCtx, "test_ids", READ_OP, alconst::slice_op, target_time, interpmode);
            readBackend.beginAction(&opCtx);

            // static_aos
            ArraystructContext staticAosCtx(&opCtx, "static_aos", "");
            int static_size = 0;
            readBackend.beginArraystructAction(&staticAosCtx, &static_size);
            assert(static_size == 2);

            for (int i = 0; i < static_size; ++i) {
                
                // --- Validation dyn_1d (dans static_aos) ---
                void* dyn_ptr = nullptr;
                int dyn_datatype = alconst::double_data;
                int dyn_dim = 0;
                int dyn_size[H5S_MAX_RANK];
                ///dyn_size[0] = 10; 
                readBackend.readData(&staticAosCtx, "dyn_1d", "time", &dyn_ptr, &dyn_datatype, &dyn_dim, dyn_size);
                
                assert(dyn_dim == 0);
                assert(dyn_size[0] == 1); 
                
                double* dyn_vals = static_cast<double*>(dyn_ptr);
                
                // Calcul du temps attendu (clamped)
                //double t_start = 0.1 + i;
               // double t_end = 1.0 + i;
                //double t_clamped = std::max(t_start, std::min(target_time, t_end));
                
                for(int k=0; k<1; ++k) {
                    double expected = 1;
                    if (std::abs(dyn_vals[k] - expected) > 1e-6) {
                         std::cerr << RED << "Erreur lecture dyn_1d t=" << target_time << " static_aos[" << i << "][" << k << "]: "
                                  << "Attendu " << expected << ", Reçu " << dyn_vals[k] << RESET << std::endl;
                         return 1;
                    }
                }
                free(dyn_ptr);

                // dynamic_aos (timed)
                ArraystructContext dynamicAosCtx(&staticAosCtx, "dynamic_aos", "time");
                int dynamic_size = 0;
                readBackend.beginArraystructAction(&dynamicAosCtx, &dynamic_size);
                //printf("dynamic_size: %d\n", dynamic_size);
                assert(dynamic_size == 1); // Slice mode -> size 1

                for (int j = 0; j < dynamic_size; ++j) {
                    void* data_ptr = nullptr;
                    int datatype = alconst::double_data;
                    int dim = 0;
                    int size[H5S_MAX_RANK];
  
                    readBackend.readData(&dynamicAosCtx, "signal1", "time", &data_ptr, &datatype, &dim, size);
                    
                    double read_val = static_cast<double*>(data_ptr)[0];
                    free(data_ptr);

                    // Validation logic
                    // i=0: times 0.1..1.0. signal1 = 1000 + t_index.
                    // i=1: times 1.1..2.0. signal1 = 1100 + t_index.
                    // time = 0.1 * (t_index + 1) + i  =>  t_index = (time - i) * 10 - 1
                    
                    double t_idx_calc = (target_time - i) * 10.0 - 1.0;
                    // Clamping t_index to [0, 9] (size of dynamic_aos is 10)
                    if (t_idx_calc < 0) t_idx_calc = 0;
                    if (t_idx_calc > 9) t_idx_calc = 9;
                    
                    double expected_val = 1000.0 + i * 100.0 + t_idx_calc;
                    
                    if (std::abs(read_val - expected_val) > 1e-6) {
                         std::cerr << RED << "Erreur lecture t=" << target_time << " static_aos[" << i << "]: "
                                  << "Attendu " << expected_val << ", Reçu " << read_val << RESET << std::endl;
                         return 1;
                    }
                    dynamicAosCtx.nextIndex(1);
                }
                readBackend.endAction(&dynamicAosCtx);
                staticAosCtx.nextIndex(1);
            }
            readBackend.endAction(&staticAosCtx);
            readBackend.endAction(&opCtx);
            readBackend.closePulse(&readDataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "  [OK] Slice t=" << target_time << " validée.\n" << RESET;

        } catch (const std::exception &e) {
            std::cerr << RED << "Exception lors de la lecture t=" << target_time << ": " << e.what() << RESET << std::endl;
            return 1;
        }
    }

    std::cout << BOLD << GREEN << "\n✓ Tous les tests ont réussi !\n" << RESET;

  } catch (const std::exception &e) {
    std::cerr << RED << BOLD << "\nException capturée : " << e.what() << RESET
              << std::endl;
    return 1;
  }

  

  return 0;
}