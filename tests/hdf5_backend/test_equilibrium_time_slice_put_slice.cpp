// test_equilibrium_time_slice.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

const std::string URI = "imas:hdf5?path=./test_db_test_equilibrium";

int main() {

  try {
    std::cout << "=== Test equilibrium : time_slice[5] → tree/node[3] → r[10] "
                 "+ homogeneous_time = 1 ===\n";

    // ==============================================================
    // 1. Contextes
    // ==============================================================

    DataEntryContext dataEntryCtx(URI);

    // ==============================================================
    // 2. Backend
    // ==============================================================

    HDF5Backend backend;
    backend.openPulse(&dataEntryCtx, OPEN_PULSE);

    double time = alconst::undefined_time;
    int interpmode = alconst::undefined_interp;
    OperationContext opCtx(&dataEntryCtx, "equilibrium", WRITE_OP,
                           alconst::slice_op, time, interpmode);

    backend.beginAction(&opCtx);

    // --- time_slice (AOS dynamique) ---
    ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");

    // ==============================================================
    // Écriture de time_slice → tree/node → r
    // ==============================================================

    int time_size = 2;

    int node_size = 3;

    int dim_r = 1;
    int size_r[1] = {10};

    backend.beginArraystructAction(&timeSliceCtx, &time_size);
    for (int t = 0; t < time_size; ++t) {
      // --- tree/node (AOS statique, dépend du temps) ---
      ArraystructContext nodeCtx(&timeSliceCtx, "tree/node", "");
      backend.beginArraystructAction(&nodeCtx, &node_size);
      for (int n = 0; n < node_size; ++n) {
        double data[10];
        // For the put-slice test we append a single slice to an existing file.
        // Use values starting at 1500 so the appended (6th) slice is clearly
        // identifiable.
        for (int i = 0; i < 10; ++i) {
          data[i] = 1500.0 + n * 10.0 + i +
                    t * 100.0; // 1500..1509, 1510..1519, 1520..1529
        }

        backend.writeData(&nodeCtx, "r", "", data, alconst::double_data, dim_r,
                          size_r);
        nodeCtx.nextIndex(1);
      }
      backend.endAction(&nodeCtx);
      timeSliceCtx.nextIndex(1);
    }
    backend.endAction(&timeSliceCtx);
    backend.endAction(&opCtx);

    std::cout << "Écriture terminée.\n\n";

    // ==============================================================
    // 5. Vérification
    // ==============================================================

    std::string ids_file = std::string("./test_db_test_equilibrium") +
                           std::string("/equilibrium.h5");
    if (!fs::exists(ids_file)) {
      std::cerr << "ERREUR : Fichier non créé\n";
      return 1;
    } else
      std::cout << "[OK] Fichier créé : " << ids_file << "\n";

    // ==============================================================
    // 6. Validation du contenu HDF5 après put_slice
    // ==============================================================
    std::cout << "\n[6] Validation du contenu HDF5 après put_slice...\n";

    // Ouvrir le fichier HDF5
    hid_t file_id = H5Fopen(ids_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
      std::cerr << "ERREUR : Impossible d'ouvrir le fichier HDF5\n";
      return 1;
    }

    // Ouvrir le groupe equilibrium
    hid_t group_id = H5Gopen2(file_id, "equilibrium", H5P_DEFAULT);
    if (group_id < 0) {
      std::cerr << "ERREUR : Impossible d'ouvrir le groupe 'equilibrium'\n";
      H5Fclose(file_id);
      return 1;
    }

    // Vérifier time_slice&AOS_SHAPE (devrait être 6 maintenant)
    std::cout << "  Vérification de time_slice&AOS_SHAPE...\n";
    hid_t dataset_id =
        H5Dopen2(group_id, "time_slice[]&AOS_SHAPE", H5P_DEFAULT);
    if (dataset_id < 0) {
      std::cerr << "ERREUR : Dataset 'time_slice[]&AOS_SHAPE' non trouvé\n";
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }
    int time_slice_shape_read;
    H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            &time_slice_shape_read);
    H5Dclose(dataset_id);

    if (time_slice_shape_read != 7) {
      std::cerr << "ERREUR : time_slice&AOS_SHAPE = " << time_slice_shape_read
                << " (attendu: 7 après put_slice)\n";
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }
    std::cout << "  [OK] time_slice&AOS_SHAPE = " << time_slice_shape_read
              << " (5 originales + 2 ajoutées)\n";

    // Vérifier time_slice[].tree&node[]&AOS_SHAPE
    std::cout << "  Vérification de time_slice[].tree&node[]&AOS_SHAPE...\n";
    dataset_id =
        H5Dopen2(group_id, "time_slice[]&tree&node[]&AOS_SHAPE", H5P_DEFAULT);
    if (dataset_id < 0) {
      std::cerr << "ERREUR : Dataset 'time_slice[]&tree&node[]&AOS_SHAPE' non "
                   "trouvé\n";
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    // Lire les dimensions du dataset AOS_SHAPE
    hid_t dataspace_id = H5Dget_space(dataset_id);
    hsize_t dims_aos[1];
    H5Sget_simple_extent_dims(dataspace_id, dims_aos, NULL);
    H5Sclose(dataspace_id);

    if (dims_aos[0] != 7) {
      std::cerr << "ERREUR : node&AOS_SHAPE devrait avoir 7 éléments (un par "
                   "time_slice)\n";
      H5Dclose(dataset_id);
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    int node_shapes[7];
    H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            node_shapes);
    H5Dclose(dataset_id);

    for (int i = 0; i < 7; ++i) {
      if (node_shapes[i] != 3) {
        std::cerr << "ERREUR : time_slice[" << i
                  << "].tree&node&AOS_SHAPE = " << node_shapes[i]
                  << " (attendu: 3)\n";
        H5Gclose(group_id);
        H5Fclose(file_id);
        return 1;
      }
    }
    std::cout << "  [OK] Chaque time_slice a 3 nodes\n";

    // Vérifier time_slice[].tree&node[].r
    std::cout << "  Vérification de time_slice[].tree&node[].r...\n";
    dataset_id = H5Dopen2(group_id, "time_slice[]&tree&node[]&r", H5P_DEFAULT);
    if (dataset_id < 0) {
      std::cerr << "ERREUR : Dataset 'time_slice[]&tree&node[]&r' non trouvé\n";
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    // Lire les dimensions du dataset r
    dataspace_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(dataspace_id);
    hsize_t dims[3];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    H5Sclose(dataspace_id);

    if (ndims != 3 || dims[0] != 7 || dims[1] != 3 || dims[2] != 10) {
      std::cerr
          << "ERREUR : Dimensions incorrectes pour time_slice[].tree&node[].r\n"
          << "  Attendu: [7, 3, 10], Obtenu: [" << dims[0] << ", " << dims[1]
          << ", " << dims[2] << "]\n";
      H5Dclose(dataset_id);
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    // Lire les données
    double data_read[7][3][10];
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            data_read);
    H5Dclose(dataset_id);

    // Vérifier les valeurs
    bool data_ok = true;

    // Vérifier les 5 premières slices (valeurs originales)
    for (int t = 0; t < 5; ++t) {
      for (int n = 0; n < 3; ++n) {
        for (int i = 0; i < 10; ++i) {
          double expected = 1000.0 + t * 100.0 + n * 10.0 + i;
          if (std::abs(data_read[t][n][i] - expected) > 1e-6) {
            std::cerr << "ERREUR : time_slice[" << t << "].tree&node[" << n
                      << "].r[" << i << "] = " << data_read[t][n][i]
                      << " (attendu: " << expected << ")\n";
            data_ok = false;
          }
        }
      }
    }

    // Vérifier les slices ajoutées (index 5 et 6)
    for (int t = 0; t < 2; ++t) {
      for (int n = 0; n < 3; ++n) {
        for (int i = 0; i < 10; ++i) {
          double expected = 1500.0 + n * 10.0 + i + t * 100.0;
          if (std::abs(data_read[5 + t][n][i] - expected) > 1e-6) {
            std::cerr << "ERREUR : time_slice[" << (5 + t) << "].tree&node["
                      << n << "].r[" << i << "] = " << data_read[5 + t][n][i]
                      << " (attendu: " << expected << " - slice ajoutée)\n";
            data_ok = false;
          }
        }
      }
    }

    if (!data_ok) {
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    std::cout << "  [OK] Toutes les valeurs sont correctes (5 slices "
                 "originales + 1 ajoutée)\n";

    // Afficher quelques valeurs pour confirmation
    std::cout << "\n  Échantillon de valeurs lues du fichier HDF5:\n";
    std::cout
        << "    Slice originale [0]: time_slice[0].tree&node[0].r[0..2] = ["
        << data_read[0][0][0] << ", " << data_read[0][0][1] << ", "
        << data_read[0][0][2] << "]\n";
    std::cout
        << "    Slice originale [4]: time_slice[4].tree&node[2].r[7..9] = ["
        << data_read[4][2][7] << ", " << data_read[4][2][8] << ", "
        << data_read[4][2][9] << "]\n";
    std::cout
        << "    Slice ajoutée   [5]: time_slice[5].tree&node[0].r[0..2] = ["
        << data_read[5][0][0] << ", " << data_read[5][0][1] << ", "
        << data_read[5][0][2] << "]\n";
    std::cout
        << "    Slice ajoutée   [5]: time_slice[5].tree&node[2].r[7..9] = ["
        << data_read[5][2][7] << ", " << data_read[5][2][8] << ", "
        << data_read[5][2][9] << "]\n";

    // Fermer les ressources HDF5
    H5Gclose(group_id);
    H5Fclose(file_id);

    std::cout << "\n✓ Validation complète réussie !\n";
    std::cout
        << "✓ La slice a été correctement ajoutée au fichier existant !\n";
    std::cout << "\nTest réussi !\n";

  } catch (const std::exception &e) {
    std::cerr << "Exception : " << e.what() << std::endl;
    return 1;
  }

  return 0;
}