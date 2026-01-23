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
    backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

    OperationContext opCtx(&dataEntryCtx, "equilibrium", "", WRITE_OP);
    backend.beginAction(&opCtx);

    // --- time_slice (AOS dynamique) ---
    ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");

    // ==============================================================
    // 3. Écriture de ids_properties/homogeneous_time = 1 (PREMIER)
    // ==============================================================

    int value = 1;
    backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &value,
                      alconst::integer_data, 0, nullptr);

    // ==============================================================
    // 4. Écriture de time_slice → tree/node → r
    // ==============================================================

    int time_size = 5;

    int node_size = 3;

    int dim_r = 1;
    int size_r[1] = {10};

    backend.beginArraystructAction(&timeSliceCtx, &time_size);
    for (int t = 0; t < time_size; ++t) {
      // --- tree/node (AOS statique, dépend du temps) ---
      ArraystructContext nodeCtx(&timeSliceCtx, "tree/node", "time");
      backend.beginArraystructAction(&nodeCtx, &node_size);
      for (int n = 0; n < node_size; ++n) {
        double data[10];
        for (int i = 0; i < 10; ++i) {
          data[i] = 1000.0f + t * 100.0f + n * 10.0f + i;
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
    // 6. Validation du contenu HDF5
    // ==============================================================
    std::cout << "\n[6] Validation du contenu HDF5...\n";

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

    // Vérifier ids_properties/homogeneous_time
    std::cout << "  Vérification de ids_properties/homogeneous_time...\n";
    hid_t dataset_id =
        H5Dopen2(group_id, "ids_properties&homogeneous_time", H5P_DEFAULT);
    if (dataset_id < 0) {
      std::cerr
          << "ERREUR : Dataset 'ids_properties&homogeneous_time' non trouvé\n";
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }
    int homogeneous_time_read;
    H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            &homogeneous_time_read);
    H5Dclose(dataset_id);

    if (homogeneous_time_read != 1) {
      std::cerr << "ERREUR : homogeneous_time = " << homogeneous_time_read
                << " (attendu: 1)\n";
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }
    std::cout << "  [OK] homogeneous_time = " << homogeneous_time_read << "\n";

    // Vérifier time_slice&AOS_SHAPE
    std::cout << "  Vérification de time_slice&AOS_SHAPE...\n";
    dataset_id = H5Dopen2(group_id, "time_slice[]&AOS_SHAPE", H5P_DEFAULT);
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

    if (time_slice_shape_read != 5) {
      std::cerr << "ERREUR : time_slice&AOS_SHAPE = " << time_slice_shape_read
                << " (attendu: 5)\n";
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }
    std::cout << "  [OK] time_slice&AOS_SHAPE = " << time_slice_shape_read
              << "\n";

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

    if (dims_aos[0] != 5) {
      std::cerr << "ERREUR : node&AOS_SHAPE devrait avoir 5 éléments (un par "
                   "time_slice)\n";
      H5Dclose(dataset_id);
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    int node_shapes[5];
    H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            node_shapes);
    H5Dclose(dataset_id);

    for (int i = 0; i < 5; ++i) {
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

    if (ndims != 3 || dims[0] != 5 || dims[1] != 3 || dims[2] != 10) {
      std::cerr
          << "ERREUR : Dimensions incorrectes pour time_slice[].tree&node[].r\n"
          << "  Attendu: [5, 3, 10], Obtenu: [" << dims[0] << ", " << dims[1]
          << ", " << dims[2] << "]\n";
      H5Dclose(dataset_id);
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    // Lire les données
    double data_read[5][3][10];
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            data_read);
    H5Dclose(dataset_id);

    // Vérifier les valeurs
    bool data_ok = true;
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

    if (!data_ok) {
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    std::cout << "  [OK] Toutes les valeurs de time_slice[].tree&node[].r sont "
                 "correctes\n";

    // Afficher quelques valeurs lues pour confirmation
    std::cout << "\n  Échantillon de valeurs lues du fichier HDF5:\n";
    std::cout << "    time_slice[0].tree&node[0].r[0..2] = ["
              << data_read[0][0][0] << ", " << data_read[0][0][1] << ", "
              << data_read[0][0][2] << "]\n";
    std::cout << "    time_slice[2].tree&node[1].r[0..2] = ["
              << data_read[2][1][0] << ", " << data_read[2][1][1] << ", "
              << data_read[2][1][2] << "]\n";
    std::cout << "    time_slice[4].tree&node[2].r[7..9] = ["
              << data_read[4][2][7] << ", " << data_read[4][2][8] << ", "
              << data_read[4][2][9] << "]\n";

    // Fermer les ressources HDF5
    H5Gclose(group_id);
    H5Fclose(file_id);

    std::cout << "\n✓ Validation complète réussie !\n";
    std::cout << "\nTest réussi !\n";

  } catch (const std::exception &e) {
    std::cerr << "Exception : " << e.what() << std::endl;
    return 1;
  }

  return 0;
}