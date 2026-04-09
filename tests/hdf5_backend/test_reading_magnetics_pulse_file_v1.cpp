// test_magnetics_flux_loop.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
#include "panzerdb.h"

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
#define BOLD_GREEN ""

const std::string URI = "imas:hdf5?path=/Imas_public/public/imasdb/west/3/54178/0";

int main() {
    
  // ==============================================================
  // 6. Re-lecture via HDF5Backend
  // ==============================================================
  std::cout << BOLD << MAGENTA << "\n[7] Lecture des données...\n" << RESET;
  std::pair<int,int> version;

  try {
      DataEntryContext readDataEntryCtx(URI);
      HDF5Backend readBackend;

      version = readBackend.getVersion(&readDataEntryCtx);
      std::cout << YELLOW << "  [DEBUG] Version du backend : " << version.first << "." << version.second << "\n" << RESET;

      std::cout << CYAN << "  Ouverture du pulse en mode lecture...\n" << RESET;
      readBackend.openPulse(&readDataEntryCtx, OPEN_PULSE);

      OperationContext readOpCtx(&readDataEntryCtx, "magnetics", "", READ_OP);
      readBackend.beginAction(&readOpCtx);
      std::cout << GREEN << "  [OK] beginAction(magnetics, READ_OP)\n" << RESET;

      // --- Validation de la base de temps ---
      std::cout << BLUE << "\n  → Lecture de la base de temps 'time'\n" << RESET;
      void* read_time_ptr = nullptr;
      int read_time_datatype = alconst::double_data;
      int read_time_dim = 0;
      int read_time_size[H5S_MAX_RANK] = {0};
      int time_found = readBackend.readData(&readOpCtx, "time", "time", &read_time_ptr, &read_time_datatype, &read_time_dim, read_time_size);

      if (!time_found) {
          std::cerr << RED << "ERREUR : Base de temps 'time' non trouvée.\n" << RESET;
          return 1;
      }
      readBackend.endAction(&readOpCtx);
      readBackend.closePulse(&readDataEntryCtx, OPEN_PULSE);

      std::string uri_write = "imas:hdf5?path=./db_write";
      DataEntryContext writeEntryCtx(uri_write);
      HDF5Backend writeBackend;
      writeBackend.openPulse(&writeEntryCtx, FORCE_CREATE_PULSE);
      
      OperationContext writeOpCtx(&writeEntryCtx, "magnetics", "", WRITE_OP);
      writeBackend.beginAction(&writeOpCtx);
      double stat_val = 999.0;
      writeBackend.writeData(&writeOpCtx, "stat/sig_stat0D", "", &stat_val, alconst::double_data, 0, nullptr);
      writeBackend.endAction(&writeOpCtx);
      writeBackend.closePulse(&readDataEntryCtx, FORCE_CREATE_PULSE);


  } catch (const std::exception &e) {
      std::cerr << RED << BOLD << "\nException capturée pendant le test de getTimeIndex : " << e.what() << RESET << std::endl;
      return 1;
  }

  return 0;
}