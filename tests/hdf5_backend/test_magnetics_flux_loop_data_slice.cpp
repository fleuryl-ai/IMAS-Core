// test_magnetics_flux_loop.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

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

const std::string URI = "imas:hdf5?path=./test_db_test_magnetics_flux_loop_slice";

int main() {

  try {
    std::cout << BOLD << MAGENTA
              << "\n=== Test magnetics : flux_loop[2].data[5] + "
                 "ids_properties.homogeneous_time = 1 ===\n"
              << RESET;

    // ==============================================================
    // 1. Contextes
    // ==============================================================
    std::cout << CYAN << "\n[1] Création du DataEntryContext avec URI : " << URI
              << RESET << std::endl;
    DataEntryContext dataEntryCtx(URI);
    std::cout << GREEN << "[OK] DataEntryContext créé.\n" << RESET;

    // ==============================================================
    // 2. Backend HDF5
    // ==============================================================
    std::cout << CYAN << "\n[2] Initialisation du backend HDF5...\n" << RESET;
    HDF5Backend backend;

    std::cout << GREEN << "[OK] Backend HDF5 initialisé.\n" << RESET;

    backend.openPulse(&dataEntryCtx, OPEN_PULSE);

    std::cout << YELLOW
              << "[DEBUG] Démarrage de l'opération WRITE sur 'magnetics'..."
              << RESET << std::endl;
    double time = alconst::undefined_time;
    int interpmode = alconst::undefined_interp;
    OperationContext opCtx(&dataEntryCtx, "magnetics", WRITE_OP,
                           alconst::slice_op, time, interpmode);

    backend.beginAction(&opCtx);
    std::cout << GREEN << "[OK] beginAction(magnetics) → OK\n" << RESET;

    // --- flux_loop (AOS statique, data dépend du temps) ---
    std::cout << YELLOW
              << "[DEBUG] Préparation du contexte Arraystruct pour 'flux_loop' "
                 "(time-dependent)\n"
              << RESET;
    ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");

    // ==============================================================
    // 3. Écriture de flux_loop (AOS de taille 2)
    // ==============================================================
    std::cout
        << CYAN
        << "\n[4] Écriture de flux_loop (AOS, size=2, data[5] par élément)\n"
        << RESET;
    int aos_size = 2;
    std::cout << YELLOW << "[DEBUG] beginArraystructAction(size=" << aos_size
              << ")\n"
              << RESET;
    backend.beginArraystructAction(&fluxLoopCtx, &aos_size);

    for (int i = 0; i < aos_size; ++i) {
      std::cout << BLUE << "\n  → Écriture de flux_loop[" << i
                << "].data (5 valeurs)\n"
                << RESET;

      int dim_data = 1;
      int size_data[1] = {1};
      double data[1] = {
          200.0f + i * 10
      };

      std::cout << YELLOW
                << "    writeData(\"data\", type=double, dim=1, size=[1])\n"
                << RESET;
      backend.writeData(&fluxLoopCtx, "data", "time", data,
                        alconst::double_data, dim_data, size_data);

      std::cout << GREEN << "    [OK] flux_loop[" << i << "].data écrit.\n"
                << RESET;

      if (i < aos_size - 1) {
        std::cout << YELLOW
                  << "    nextIndex(1) → passage à l'élément suivant\n"
                  << RESET;
        fluxLoopCtx.nextIndex(1);
      }
    }

    std::cout << YELLOW << "[DEBUG] endAction(flux_loop)\n" << RESET;
    backend.endAction(&fluxLoopCtx);
    std::cout << GREEN << "[OK] Arraystruct flux_loop fermé.\n" << RESET;

    std::cout << YELLOW << "[DEBUG] endAction(magnetics)\n" << RESET;
    backend.endAction(&opCtx);
    std::cout << GREEN << "[OK] Opération magnetics terminée.\n" << RESET;

    std::cout << BOLD << GREEN << "\nÉcriture terminée avec succès.\n" << RESET;

    // ==============================================================
    // 5. Vérification (optionnelle avec h5dump)
    // ==============================================================
    std::cout << CYAN << "\n[5] Vérification du fichier généré...\n" << RESET;
    std::string ids_file =
        std::string("./test_db_test_magnetics") + std::string("/magnetics.h5");
    if (!fs::exists(ids_file)) {
      std::cerr << RED << "ERREUR : Fichier non créé → " << ids_file << RESET
                << std::endl;
      return 1;
    }
    std::cout << GREEN << "[OK] Fichier créé : " << ids_file << "\n" << RESET;

    // ==============================================================
    // 6. Validation du contenu HDF5
    // ==============================================================
    std::cout << CYAN << "\n[6] Validation du contenu HDF5...\n" << RESET;

    // Ouvrir le fichier HDF5
    hid_t file_id = H5Fopen(ids_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
      std::cerr << RED << "ERREUR : Impossible d'ouvrir le fichier HDF5\n"
                << RESET;
      return 1;
    }

    // Ouvrir le groupe magnetics
    hid_t group_id = H5Gopen2(file_id, "magnetics", H5P_DEFAULT);
    if (group_id < 0) {
      std::cerr << RED << "ERREUR : Impossible d'ouvrir le groupe 'magnetics'\n"
                << RESET;
      H5Fclose(file_id);
      return 1;
    }

    // Vérifier flux_loop&AOS_SHAPE
    std::cout << YELLOW << "  Vérification de flux_loop&AOS_SHAPE...\n"
              << RESET;
    hid_t dataset_id = H5Dopen2(group_id, "flux_loop[]&AOS_SHAPE", H5P_DEFAULT);
    if (dataset_id < 0) {
      std::cerr << RED
                << "ERREUR : Dataset 'flux_loop[]&AOS_SHAPE' non trouvé\n"
                << RESET;
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }
    int aos_shape_read;
    H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            &aos_shape_read);
    H5Dclose(dataset_id);

    if (aos_shape_read != 2) {
      std::cerr << RED << "ERREUR : flux_loop&AOS_SHAPE = " << aos_shape_read
                << " (attendu: 2)\n"
                << RESET;
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }
    std::cout << GREEN << "  [OK] flux_loop&AOS_SHAPE = " << aos_shape_read
              << "\n"
              << RESET;

    // Vérifier flux_loop[].data
    std::cout << YELLOW << "  Vérification de flux_loop[].data...\n" << RESET;
    dataset_id = H5Dopen2(group_id, "flux_loop[]&data", H5P_DEFAULT);
    if (dataset_id < 0) {
      std::cerr << RED << "ERREUR : Dataset 'flux_loop[]&data' non trouvé\n"
                << RESET;
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    // Lire les dimensions du dataset
    hid_t dataspace_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(dataspace_id);
    hsize_t dims[3];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    H5Sclose(dataspace_id);

    // En mode slice avec time, on s'attend à [2, 6] (2 flux_loop, 5 initiales +
    // 1 slice)
    if (ndims != 2 || dims[0] != 2 || dims[1] != 6) {
      std::cerr << RED
                << "ERREUR : Dimensions incorrectes pour flux_loop[].data\n"
                << "  Attendu: [2, 6], Obtenu: [" << dims[0] << ", " << dims[1]
                << "]\n"
                << RESET;
      H5Dclose(dataset_id);
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    // Lire les données
    double data_read[2][6];
    H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            data_read);
    H5Dclose(dataset_id);

    // Vérifier les valeurs
    bool data_ok = true;
    for (int i = 0; i < 2; ++i) {
      // Vérifier les 5 premières valeurs (test initial)
      for (int j = 0; j < 5; ++j) {
        double expected = 100.0 + i * 10 + j;
        if (std::abs(data_read[i][j] - expected) > 1e-6) {
          std::cerr << RED << "ERREUR : flux_loop[" << i << "].data[" << j
                    << "] = " << data_read[i][j] << " (attendu: " << expected
                    << ")\n"
                    << RESET;
          data_ok = false;
        }
      }

      // Vérifier la 6ème valeur (slice ajoutée)
      double expected_slice = 200.0 + i * 10;
      if (std::abs(data_read[i][5] - expected_slice) > 1e-6) {
        std::cerr << RED << "ERREUR : flux_loop[" << i
                  << "].data[5] = " << data_read[i][5]
                  << " (attendu: " << expected_slice << ")\n"
                  << RESET;
        data_ok = false;
      }
    }

    if (!data_ok) {
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    std::cout << GREEN
              << "  [OK] Toutes les valeurs de flux_loop[].data sont correctes "
                 "(5 initiales + 1 slice)\n"
              << RESET;

    // Afficher les valeurs lues pour confirmation
    std::cout << CYAN << "\n  Valeurs lues du fichier HDF5:\n" << RESET;
    for (int i = 0; i < 2; ++i) {
      std::cout << YELLOW << "    flux_loop[" << i << "].data = [";
      for (int j = 0; j < 6; ++j) {
        std::cout << std::fixed << std::setprecision(1) << data_read[i][j];
        if (j < 5)
          std::cout << ", ";
      }
      std::cout << "]\n" << RESET;
    }

    // Fermer les ressources HDF5
    H5Gclose(group_id);
    H5Fclose(file_id);

    std::cout << BOLD << GREEN << "\n✓ Validation complète réussie !\n"
              << RESET;
    std::cout << BOLD << GREEN << "\nTest réussi !\n" << RESET;

  } catch (const std::exception &e) {
    std::cerr << RED << BOLD << "\nException capturée : " << e.what() << RESET
              << std::endl;
    return 1;
  }

  return 0;
}