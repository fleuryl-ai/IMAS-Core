// test_magnetics_flux_loop_position_data.cpp
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
const std::string URI = "imas:hdf5?path=./test_db_test_magnetics_flux_loop_position_data";
int main() {
  try {
    std::cout << BOLD << MAGENTA
              << "\n=== Test magnetics : flux_loop[2].position[2].data[5] + "
                 "ids_properties.homogeneous_time = 1 (2 AOS imbriqués) ===\n"
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
    std::cout << YELLOW << "[DEBUG] Appel à getVersion()..." << RESET
              << std::endl;
    // backend.getVersion(&dataEntryCtx);
    std::cout << GREEN << "[OK] Backend HDF5 initialisé.\n" << RESET;
    backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);
    std::cout << YELLOW
              << "[DEBUG] Démarrage de l'opération WRITE sur 'magnetics'..."
              << RESET << std::endl;
    OperationContext opCtx(&dataEntryCtx, "magnetics", "", WRITE_OP);
    backend.beginAction(&opCtx);
    std::cout << GREEN << "[OK] beginAction(magnetics) → OK\n" << RESET;
    // --- flux_loop (AOS externe, time-dependent) ---
    std::cout << YELLOW
              << "[DEBUG] Préparation du contexte Arraystruct pour 'flux_loop' "
                 "(time-dependent)\n"
              << RESET;
    ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "time");
    // ==============================================================
    // 3. Écriture de ids_properties/homogeneous_time = 1 (PREMIER)
    // ==============================================================
    std::cout
        << CYAN
        << "\n[3] Écriture de ids_properties/homogeneous_time = 1 (scalaire)\n"
        << RESET;
    int value = 1;
    std::cout
        << YELLOW
        << "[DEBUG] writeData(path=\"homogeneous_time/ids_properties\", value="
        << value << ", type=integer, dim=0)\n"
        << RESET;
    backend.writeData(&opCtx, "homogeneous_time/ids_properties", "", &value,
                      alconst::integer_data, 0, nullptr);
    std::cout << GREEN << "[OK] homogeneous_time écrit.\n" << RESET;
    // ==============================================================
    // 4. Écriture de flux_loop (AOS externe de taille 2)
    // ==============================================================
    std::cout << CYAN << "\n[4] Écriture de flux_loop (AOS externe, size=2)\n"
              << RESET;
    int aos_size_outer = 2;
    std::cout << YELLOW
              << "[DEBUG] beginArraystructAction(size=" << aos_size_outer
              << ")\n"
              << RESET;
    backend.beginArraystructAction(&fluxLoopCtx, &aos_size_outer);
    for (int i = 0; i < aos_size_outer; ++i) {
      std::cout << BLUE << "\n → flux_loop[" << i << "] (élément externe)\n"
                << RESET;
      // --- position (AOS interne, non time-dependent) ---
      std::cout << YELLOW
                << "   [DEBUG] Préparation du contexte Arraystruct pour "
                   "'position' (non time-dependent)\n"
                << RESET;
      ArraystructContext positionCtx(&fluxLoopCtx, "position", "time");
      std::cout << CYAN << "   Écriture de position (AOS interne, size=2)\n"
                << RESET;
      int aos_size_inner = 3;
      std::cout << YELLOW
                << "   [DEBUG] beginArraystructAction(size=" << aos_size_inner
                << ")\n"
                << RESET;
      backend.beginArraystructAction(&positionCtx, &aos_size_inner);
      for (int j = 0; j < aos_size_inner; ++j) {
        std::cout << BLUE << "     → Écriture de position[" << j
                  << "].data (5 valeurs)\n"
                  << RESET;
        int dim_data = 1;
        int size_data[1] = {5};
        double data[5] = {100.0 + i * 10 + j * 1, 101.0 + i * 10 + j * 1,
                          102.0 + i * 10 + j * 1, 103.0 + i * 10 + j * 1,
                          104.0 + i * 10 + j * 1};
        // Affichage des données
        std::cout << YELLOW << "     data = [";
        for (int k = 0; k < 5; ++k) {
          std::cout << std::fixed << std::setprecision(1) << data[k];
          if (k < 4)
            std::cout << ", ";
        }
        std::cout << "]\n" << RESET;
        std::cout << YELLOW
                  << "     writeData(\"data\", type=double, dim=1, size=[5])\n"
                  << RESET;
        backend.writeData(&positionCtx, "data", "time", data,
                          alconst::double_data, dim_data, size_data);
        std::cout << GREEN << "     [OK] position[" << j << "].data écrit.\n"
                  << RESET;
        if (j < aos_size_inner - 1) {
          std::cout
              << YELLOW
              << "     nextIndex(1) → passage à l'élément suivant (position)\n"
              << RESET;
          positionCtx.nextIndex(1);
        }
      }
      std::cout << YELLOW << "   [DEBUG] endAction(position)\n" << RESET;
      backend.endAction(&positionCtx);
      std::cout << GREEN << "   [OK] Arraystruct position fermé.\n" << RESET;
      if (i < aos_size_outer - 1) {
        std::cout << YELLOW
                  << " nextIndex(1) → passage à l'élément suivant (flux_loop)\n"
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
    std::string ids_file = std::string("./test_db_test_magnetics_nested") +
                           std::string("/magnetics.h5");
    if (!fs::exists(ids_file)) {
      std::cerr << RED << "ERREUR : Fichier non créé → " << ids_file << RESET
                << std::endl;
      return 1;
    }
    std::cout << GREEN << "[OK] Fichier créé : " << ids_file << "\n" << RESET;

    // ==============================================================
    // 6. Validation du contenu HDF5 avec PanzerDB
    // ==============================================================

    std::cout << CYAN << "\n[6] Validation du contenu HDF5...\n" << RESET;

    // Vérifier homogeneous_time/ids_properties
        // 1. Valider le scalaire homogeneous_time
    std::cout << YELLOW << "  Vérification de ids_properties&homogeneous_time...\n"
              << RESET;
    int status = -1;

    
  return 0;
}