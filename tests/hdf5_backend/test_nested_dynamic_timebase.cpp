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

      // On force le mode non-homogène pour que chaque AOS puisse avoir sa propre base de temps
      //OperationContext opCtxIds(&dataEntryCtx, "test_ids", "", WRITE_OP);

      //backend.beginAction(&opCtxIds);
      int homogeneous_time = 1;
      backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);
      //backend.endAction(&opCt);

      //OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
      //backend.beginAction(&opCtx);
      int dyn_1d_time_size = 10;
        std::vector<double> time_data(dyn_1d_time_size);
        //std::vector<double> dyn_1d_data(dyn_1d_time_size);
        for(int t=0; t<dyn_1d_time_size; ++t) {
            time_data[t] = (double)t * 0.1; // ex: [0.0..0.9] pour i=0
            //dyn_1d_data[t] = time_data[t] * 10.0 + 6.5; // Donnée simple pour validation
        }
        int dim[] = {dyn_1d_time_size};
        backend.writeData(&opCtx, "time", "time", time_data.data(), alconst::double_data, 1, dim);

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

        // --- AoS Dynamique 'dynamic_aos' imbriqué ---
        int dynamic_aos_size = 10;
        ArraystructContext dynamicAosCtx(&staticAosCtx, "dynamic_aos", "time");
        backend.beginArraystructAction(&dynamicAosCtx, &dynamic_aos_size);

        // Écriture de la base de temps et des signaux pour cet AoS dynamique
        for (int t = 0; t < dynamic_aos_size; ++t) {
          // Écrire la valeur de temps pour cette tranche
          // Temps différents pour chaque instance de static_aos
          double current_time = 0.1 * (t);
          backend.writeData(&dynamicAosCtx, "time", "time", &current_time,
                            alconst::double_data, 0, nullptr);

          // Écriture de deux signaux dynamiques
          double signal1_data = 1000.0 + t*10;
          double signal2_data = 2000.0 + t*10;
          backend.writeData(&dynamicAosCtx, "signal1", "time", &signal1_data,
                            alconst::double_data, 0, nullptr);
          backend.writeData(&dynamicAosCtx, "signal2", "time", &signal2_data,
                            alconst::double_data, 0, nullptr);

          if (t < dynamic_aos_size - 1) {
            dynamicAosCtx.nextIndex(1);
          }
        }
        backend.endAction(&dynamicAosCtx);

        // --- AJOUT : Signal dynamique 1D 'dyn_1d' dans static_aos ---
        // Ce signal a sa propre base de temps, indépendante de celle de dynamic_aos
        //int dyn_1d_time_size = 10;
        //std::vector<double> time_data(dyn_1d_time_size);
        std::vector<double> dyn_1d_data(dyn_1d_time_size);
        for(int t=0; t<dyn_1d_time_size; ++t) {
            //time_data[t] = (double)t * 0.1 + i; // ex: [0.0..0.9] pour i=0
            dyn_1d_data[t] = time_data[t] * 10.0 + 6.5; // Donnée simple pour validation
        }
        //int dim[] = {dyn_1d_time_size};
        //backend.writeData(&staticAosCtx, "time", "time", time_data.data(), alconst::double_data, 1, dim);
        int size_dyn_1d[] = {dyn_1d_time_size};
        backend.writeData(&staticAosCtx, "dyn_1d", "time", dyn_1d_data.data(), alconst::double_data, 1, size_dyn_1d);

        if (i < static_aos_size - 1) {
          staticAosCtx.nextIndex(1);
        }
      }

      backend.endAction(&staticAosCtx);
      backend.endAction(&opCtx);
      backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);

      std::cout << GREEN << "[OK] Écriture terminée.\n" << RESET;
    }

    // ==============================================================
    // 2. Lecture par Slices 
    // ==============================================================
    std::cout << BOLD << MAGENTA << "\n[3] Test de lecture par Slices...\n" << RESET;

    std::vector<double> target_times = {0.55, 0.85};
    double expected_val[2] = {1055, 1085};  
    int k = 0;
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
                
                readBackend.readData(&staticAosCtx, "dyn_1d", "time", &dyn_ptr, &dyn_datatype, &dyn_dim, dyn_size);
                
                assert(dyn_dim == 1); // En mode slice, on lit un scalaire
                
                double read_val = static_cast<double*>(dyn_ptr)[0];
                // La donnée écrite est time*10 + 6.5, donc la valeur interpolée doit être target_time*10 + 6.5
                double expected_val1 = target_time * 10.0 + 6.5;

                if (std::abs(read_val - expected_val1 ) > 1e-6) {
                    std::cerr << RED << "Erreur lecture dyn_1d t=" << target_time << " static_aos[" << i << "]: Attendu " << expected_val << ", Reçu " << read_val << RESET << std::endl;
                    return 1;
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
                    
                    if (std::abs(read_val - expected_val[k] ) > 1e-6) {
                         std::cerr << RED << "Erreur lecture signal1 at t=" << target_time << " static_aos[" << i << "]: "
                                  << "Attendu " << expected_val[k]  << ", Reçu " << read_val << RESET << std::endl;
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
        k++;
    }

    std::cout << BOLD << GREEN << "\n✓ Tous les tests ont réussi !\n" << RESET;

  } catch (const std::exception &e) {
    std::cerr << RED << BOLD << "\nException capturée : " << e.what() << RESET
              << std::endl;
    return 1;
  }

  

  return 0;
}