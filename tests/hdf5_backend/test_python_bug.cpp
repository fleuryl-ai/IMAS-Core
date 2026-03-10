// tests/hdf5_backend/test_magnetics_repro.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

namespace fs = std::filesystem;

// ANSI Colors for debug output
#define RESET ""
#define BOLD ""
#define GREEN ""
#define YELLOW ""
#define BLUE ""
#define MAGENTA ""
#define CYAN ""
#define RED ""

const std::string URI = "imas:hdf5?path=/ZONE_TRAVAIL/LF218007/AL_repos/al-python/build/tests/testdb/test/3/9999/9999";

//const std::string URI = "imas:hdf5?path=/Imas_public/public/imasdb/west/3/57430/0";

int main() {

  try {
    std::cout << BOLD << MAGENTA
              << "\n=== Test Magnetics Repro: flux_loop[3].flux.data[10] (Homogeneous) ===\n"
              << RESET;

    
    // ==============================================================
    // 2. READ / VERIFICATION PHASE
    // ==============================================================
    std::cout << CYAN << "\n[2] Reading and Verifying data...\n" << RESET;

    DataEntryContext readCtx(URI);
    HDF5Backend readBackend;
    readBackend.openPulse(&readCtx, OPEN_PULSE);

    //OperationContext readOpCtx(&readCtx, "barometry", "", READ_OP);
    int interpmode = alconst::closest_interp;

    OperationContext readOpCtx(&readCtx, "barometry", READ_OP, alconst::slice_op, 0.0, interpmode);
    readBackend.beginAction(&readOpCtx);

    OperationContext readOpCtx2(&readCtx, "barometry", "", READ_OP);
    readBackend.beginAction(&readOpCtx2);

    // Verify homogeneous_time
    void* data_ptr = nullptr;
    int datatype = alconst::integer_data;
    int dim = 0;
    int size[H5S_MAX_RANK];
    readBackend.readData(&readOpCtx2, "ids_properties&homogeneous_time", "", &data_ptr, &datatype, &dim, size);
    printf("  Read homogeneous_time: %d\n", data_ptr ? *(int*)data_ptr : -1);
    readBackend.endAction(&readOpCtx2);

    // Read comment
    data_ptr = nullptr;
    datatype = alconst::char_data;
    dim = 1;
    //size[H5S_MAX_RANK] = {0};

    readBackend.readData(&readOpCtx, "ids_properties&version_put&data_dictionary", "", &data_ptr, &datatype, &dim, size);
    printf("  Read data_dictionary: '%s'\n", (char*)data_ptr);

    data_ptr = nullptr;
    datatype = alconst::char_data;
    dim = 1;

    readBackend.readData(&readOpCtx, "ids_properties/comment", "", &data_ptr, &datatype, &dim, size);
    printf("  Read comment: '%s'\n", (char*)data_ptr);

    readBackend.endAction(&readOpCtx);
    readBackend.closePulse(&readCtx, OPEN_PULSE);

  } catch (const std::exception &e) {
    std::cerr << RED << BOLD << "\nException caught: " << e.what() << RESET << std::endl;
    return 1;
  }

  return 0;
}
