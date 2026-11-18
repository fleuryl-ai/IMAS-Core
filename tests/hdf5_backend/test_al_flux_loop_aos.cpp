// @file  test_al_flux_loop_aos.cpp
// @brief Write/read regression test for magnetics flux_loop (static AoS with
//        time-dependent data), plus a PanzerDB::getTimeIndex check.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
#include "panzerdb.h"

namespace fs = std::filesystem;

// ANSI colors for debug
#define RESET ""
#define BOLD ""
#define GREEN ""
#define YELLOW ""
#define BLUE ""
#define MAGENTA ""
#define CYAN ""
#define RED ""
#define BOLD_GREEN ""

const std::string URI = "imas:hdf5?path=./test_db_test_magnetics_flux_loop";

int main() {

  try {
    std::cout << BOLD << MAGENTA
              << "\n=== Test magnetics: flux_loop[2].data[5] + "
                 "ids_properties.homogeneous_time = 1 ===\n"
              << RESET;

    // ==============================================================
    // 1. Contexts
    // ==============================================================
    std::cout << CYAN << "\n[1] Creating the DataEntryContext with URI: " << URI
              << RESET << std::endl;
    DataEntryContext dataEntryCtx(URI);
    std::cout << GREEN << "[OK] DataEntryContext created.\n" << RESET;

    // ==============================================================
    // 2. Backend HDF5
    // ==============================================================
    std::cout << CYAN << "\n[2] Initializing the HDF5 backend...\n" << RESET;
    HDF5Backend backend;
    std::cout << YELLOW << "[DEBUG] Calling getVersion()..." << RESET
              << std::endl;
    // backend.getVersion(&dataEntryCtx);

    std::cout << GREEN << "[OK] HDF5 backend initialized.\n" << RESET;

    backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

    std::cout << YELLOW
              << "[DEBUG] Starting the WRITE operation on 'magnetics'..."
              << RESET << std::endl;
    OperationContext opCtx(&dataEntryCtx, "magnetics", "", WRITE_OP);
    backend.beginAction(&opCtx);
    std::cout << GREEN << "[OK] beginAction(magnetics) → OK\n" << RESET;

    // --- flux_loop (static AoS, time-dependent data) ---
    std::cout << YELLOW
              << "[DEBUG] Preparing the Arraystruct context for 'flux_loop' "
                 "(time-dependent)\n"
              << RESET;
    ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");

    // ==============================================================
    // 3. Writing ids_properties/homogeneous_time = 1 (FIRST)
    // ==============================================================
    std::cout
        << CYAN
        << "\n[3] Writing ids_properties/homogeneous_time = 1 (scalar)\n"
        << RESET;
    int value = 1;
    std::cout
        << YELLOW
        << "[DEBUG] writeData(path=\"homogeneous_time/ids_properties\", value="
        << value << ", type=integer, dim=0)\n"
        << RESET;
    backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &value,
                      alconst::integer_data, 0, nullptr);
    std::cout << GREEN << "[OK] homogeneous_time written.\n" << RESET;

    // ==============================================================
    // 4. Writing flux_loop (AoS of size 2)
    // ==============================================================
    std::cout
        << CYAN
        << "\n[4] Writing flux_loop (AoS, size=2, data[5] per element)\n"
        << RESET;
    int aos_size = 2;
    std::cout << YELLOW << "[DEBUG] beginArraystructAction(size=" << aos_size
              << ")\n"
              << RESET;
    backend.beginArraystructAction(&fluxLoopCtx, &aos_size);

    for (int i = 0; i < aos_size; ++i) {
      std::cout << BLUE << "\n  → Writing flux_loop[" << i
                 << "].data (5 values)\n"
                 << RESET;

      int dim_data = 1;
      int size_data[1] = {5};
      double data[5] = {100.0f + i * 10, 101.0f + i * 10, 102.0f + i * 10,
                        103.0f + i * 10, 104.0f + i * 10};

      // Display the data
      std::cout << YELLOW << "    data = [";
      for (int j = 0; j < 5; ++j) {
        std::cout << std::fixed << std::setprecision(1) << data[j];
        if (j < 4)
          std::cout << ", ";
      }
      std::cout << "]\n" << RESET;

      std::cout << YELLOW
                << "    writeData(\"data\", type=double, dim=1, size=[5])\n"
                << RESET;
       backend.writeData(&fluxLoopCtx, "data", "time", data,
                        alconst::double_data, dim_data, size_data);

      std::cout << GREEN << "    [OK] flux_loop[" << i << "].data written.\n"
                << RESET;

      if (i < aos_size - 1) {
        std::cout << YELLOW
                  << "    nextIndex(1) → advancing to the next element\n"
                  << RESET;
        fluxLoopCtx.nextIndex(1);
      }
    }

    std::cout << YELLOW << "[DEBUG] endAction(flux_loop)\n" << RESET;
    backend.endAction(&fluxLoopCtx);
    std::cout << GREEN << "[OK] Arraystruct flux_loop closed.\n" << RESET;

    // ==============================================================
    // 5. Writing the time base 'time'
    // ==============================================================
    std::cout << CYAN << "\n[5] Writing the time base 'time'...\n"
              << RESET;
    int dim_time = 1;
    int size_time[1] = {5};
    double time_data[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    std::cout << YELLOW
              << "    writeData(\"time\", type=double, dim=1, size=[5])\n"
              << RESET;
    backend.writeData(&opCtx, "time", "time", time_data, alconst::double_data,
                      dim_time, size_time);
    std::cout << GREEN << "[OK] Time base 'time' written.\n" << RESET;

    std::cout << YELLOW << "[DEBUG] endAction(magnetics)\n" << RESET;
    backend.endAction(&opCtx);
    std::cout << GREEN << "[OK] magnetics operation finished.\n" << RESET;

    backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);

    std::cout << BOLD << GREEN << "\nWrite completed successfully.\n" << RESET;

  } catch (const std::exception &e) {
    //H5Fclose(h5_file_id); // Ensure the file is closed in case of exception
    std::cerr << RED << BOLD << "\nException caught: " << e.what() << RESET
              << std::endl;
    return 1;
  }

  // ==============================================================
  // 6. Read-back via HDF5Backend
  // ==============================================================
  std::cout << BOLD << MAGENTA << "\n[7] Reading back the data...\n" << RESET;
  std::pair<int,int> version;

  try {
      DataEntryContext readDataEntryCtx(URI);
      HDF5Backend readBackend;

      version = readBackend.getVersion(&readDataEntryCtx);
      std::cout << YELLOW << "  [DEBUG] Backend version: " << version.first << "." << version.second << "\n" << RESET;

      std::cout << CYAN << "  Opening the pulse in read mode...\n" << RESET;
      readBackend.openPulse(&readDataEntryCtx, OPEN_PULSE);

      OperationContext readOpCtx(&readDataEntryCtx, "magnetics", "", READ_OP);
      readBackend.beginAction(&readOpCtx);
      std::cout << GREEN << "  [OK] beginAction(magnetics, READ_OP)\n" << RESET;

      // --- Validation of the time base ---
      std::cout << BLUE << "\n  → Reading the time base 'time'\n" << RESET;
      void* read_time_ptr = nullptr;
      int read_time_datatype = alconst::double_data;
      int read_time_dim = 0;
      int read_time_size[H5S_MAX_RANK] = {0};
      int time_found = readBackend.readData(&readOpCtx, "time", "time", &read_time_ptr, &read_time_datatype, &read_time_dim, read_time_size);

      if (!time_found) {
          std::cerr << RED << "ERROR: time base 'time' not found.\n" << RESET;
          return 1;
      }
     
      if ( read_time_dim != 1  || read_time_size[0] != 5) {
          std::cerr << RED << "ERROR: wrong dimensions for 'time'. Got: dim=" << read_time_dim << ", size[0]=" << read_time_size[0] << " (Expected: dim=1, size[0]=5)\n" << RESET;
          return 1;
      } else {
          double* read_time_values = static_cast<double*>(read_time_ptr);
          for (int j = 0; j < 5; ++j) {
              double expected = 0.1 * (j + 1);
              if (std::abs(read_time_values[j] - expected) > 1e-9) {
                  std::cerr << RED << "ERROR: wrong value for time[" << j << "]. Got: " << read_time_values[j] << ", Expected: " << expected << "\n" << RESET;
                  return 1;
              }
          }
      }
      delete[] static_cast<double*>(read_time_ptr);
      std::cout << GREEN << "  [OK] Time base 'time' validated.\n" << RESET;

      ArraystructContext readFluxLoopCtx(&readOpCtx, "flux_loop", "");
      int read_aos_size = 0;
      readBackend.beginArraystructAction(&readFluxLoopCtx, &read_aos_size);

      if (read_aos_size != 2) {
          std::cerr << RED << "ERROR: size read for flux_loop is " << read_aos_size << " (expected: 2)\n" << RESET;
          return 1;
      }
      std::cout << GREEN << "  [OK] Size read for flux_loop: " << read_aos_size << "\n" << RESET;

      bool all_read_ok = true;
      for (int i = 0; i < read_aos_size; ++i) {
          std::cout << BLUE << "\n  → Reading flux_loop[" << i << "].data\n" << RESET;

          void* read_data_ptr = nullptr;
          int read_datatype = alconst::double_data;
          int read_dim = 0;
          if (version == std::pair<int, int> (1,0)) {
              read_dim = 1; //for backend v1, dim is not set by the reader
          }
          int read_size[H5S_MAX_RANK] = {0};

          int data_found = readBackend.readData(&readFluxLoopCtx, "data", "time", &read_data_ptr, &read_datatype, &read_dim, read_size);

          if (!data_found) {
              std::cerr << RED << "ERROR: no data found for flux_loop[" << i << "].data\n" << RESET;
              all_read_ok = false;
              continue;
          }

          if (read_dim != 1 || read_size[0] != 5) {
              std::cerr << RED << "ERROR: wrong dimensions for flux_loop[" << i << "].data. Got: dim=" << read_dim << ", size[0]=" << read_size[0] << " (Expected: dim=1, size[0]=5)\n" << RESET;
              all_read_ok = false;
          } 
           else {
              double* read_values = static_cast<double*>(read_data_ptr);
              for (int j = 0; j < 5; ++j) {
                  double expected = 100.0 + i * 10 + j;
                  if (std::abs(read_values[j] - expected) > 1e-9) {
                      std::cerr << RED << "ERROR: wrong value for flux_loop[" << i << "].data[" << j << "]. Got: " << read_values[j] << ", Expected: " << expected << "\n" << RESET;
                      all_read_ok = false;
                  }
              }
          }

          delete[] static_cast<double*>(read_data_ptr); // Free the memory allocated by readData

          if (i < read_aos_size - 1) {
              readFluxLoopCtx.nextIndex(1);
          }
      }

      readBackend.endAction(&readFluxLoopCtx);
      readBackend.endAction(&readOpCtx);
      readBackend.closePulse(&readDataEntryCtx, OPEN_PULSE);

      if (all_read_ok) {
          std::cout << BOLD_GREEN << "\n✓ Read-back validation successful!\n" << RESET;
      } else {
          return 1;
      }

  } catch (const std::exception &e) {
      std::cerr << RED << BOLD << "\nException caught during read-back: " << e.what() << RESET << std::endl;
      return 1;
  }

  // ==============================================================
  // 7. Test de PanzerDB::getTimeIndex
  // ==============================================================

  if (version == std::pair<int, int> (1,0)) {
      //std::cout << YELLOW << "\n[DEBUG] Backend version is 1.0, skipping getTimeIndex test (not supported in v1.0)\n" << RESET;
      return 0;
  } 
  std::cout << BOLD << MAGENTA << "\n[8] Testing PanzerDB::getTimeIndex...\n" << RESET;
  try {
      std::string ids_file = "./test_db_test_magnetics_flux_loop/magnetics.h5";
      hid_t h5_file_id = H5Fopen(ids_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      if (h5_file_id < 0) {
          std::cerr << RED << "ERROR: unable to open the HDF5 file for the getTimeIndex test.\n" << RESET;
          return 1;
      }
      hid_t magnetics_group_id = H5Gopen2(h5_file_id, "magnetics", H5P_DEFAULT);
      H5Fclose(h5_file_id);
      if (magnetics_group_id < 0) {
          std::cerr << RED << "ERROR: unable to open the 'magnetics' group for the getTimeIndex test.\n" << RESET;
          return 1;
      }

      PanzerDB db(magnetics_group_id, PanzerDB::OpenMode::READ, true, true);
      double requested_time = 0.3;
      int64_t found_index = db.getTimeIndex("time", requested_time, alconst::closest_interp);

      if (found_index != 2) {
          std::cerr << RED << "ERROR: getTimeIndex(\"time\", " << requested_time << ") returned " << found_index << " (expected: 2)\n" << RESET;
          return 1;
      }
      std::cout << GREEN << "  [OK] getTimeIndex(\"time\", " << requested_time << ") returned the correct index: " << found_index << "\n" << RESET;
  } catch (const std::exception &e) {
      std::cerr << RED << BOLD << "\nException caught during the getTimeIndex test: " << e.what() << RESET << std::endl;
      return 1;
  }

  return 0;
}