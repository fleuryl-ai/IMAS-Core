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

const std::string URI = "imas:hdf5?path=./test_db_test_magnetics_repro";

int main() {

  try {
    std::cout << BOLD << MAGENTA
              << "\n=== Test Magnetics Repro: flux_loop[3].flux.data[10] (Homogeneous) ===\n"
              << RESET;

    // Parameters from the user scenario
    int dynamicsize = 10;
    int staticsize = 3;

    // ==============================================================
    // 1. WRITE PHASE
    // ==============================================================
    std::cout << CYAN << "\n[1] Writing data...\n" << RESET;
    
    DataEntryContext dataEntryCtx(URI);
    HDF5Backend backend;
    
    // Equivalent to: ids.open(uri, FORCE_CREATE_PULSE);
    backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

    // Equivalent to: ids._magnetics.put() (Start)
    OperationContext opCtx(&dataEntryCtx, "magnetics", "", WRITE_OP);
    backend.beginAction(&opCtx);

    // Equivalent to: ids._magnetics.ids_properties.homogeneous_time = 1;
    int homogeneous_time = 1;
    backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);
    std::cout << GREEN << "  [OK] ids_properties.homogeneous_time = 1\n" << RESET;

    // Equivalent to: ids._magnetics.time loop
    std::vector<double> time_data(dynamicsize);
    for (int i = 0; i < dynamicsize; i++) {
        time_data[i] = 0.1 * i;
    }
    int time_dim = 1;
    int time_size_arr[] = {dynamicsize};
    // Note: For homogeneous time, we write to "time" with timebase "time" (self-reference or convention)
    backend.writeData(&opCtx, "time", "time", time_data.data(), alconst::double_data, time_dim, time_size_arr);
    std::cout << GREEN << "  [OK] time array written (size " << dynamicsize << ")\n" << RESET;

    // Equivalent to: ids._magnetics.flux_loop.resize(staticsize);
    ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
    backend.beginArraystructAction(&fluxLoopCtx, &staticsize);

    for (int j = 0; j < staticsize; j++) {
        std::cout << BLUE << "    Processing flux_loop[" << j << "]...\n" << RESET;
        
        // Equivalent to: ids._magnetics.flux_loop(j).flux.data loop
        std::vector<double> flux_data(dynamicsize);
        for (int i = 0; i < dynamicsize; i++) {
            flux_data[i] = j * 100.0 + i;
        }

        // Writing flux/data
        // Note: The path is relative to flux_loop element. 
        // High-level: flux_loop(j).flux.data -> Low-level path: "flux/data"
        backend.writeData(&fluxLoopCtx, "flux/data", "time", flux_data.data(), alconst::double_data, time_dim, time_size_arr);
        
        if (j < staticsize - 1) {
            fluxLoopCtx.nextIndex(1);
        }
    }
    backend.endAction(&fluxLoopCtx);

    // Equivalent to: ids._magnetics.put() (End)
    backend.endAction(&opCtx);
    
    // Equivalent to: ids.close();
    backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
    std::cout << GREEN << "[OK] Write operation completed.\n" << RESET;


    // ==============================================================
    // 2. READ / VERIFICATION PHASE
    // ==============================================================
    std::cout << CYAN << "\n[2] Reading and Verifying data...\n" << RESET;

    DataEntryContext readCtx(URI);
    HDF5Backend readBackend;
    readBackend.openPulse(&readCtx, OPEN_PULSE);

    OperationContext readOpCtx(&readCtx, "magnetics", "", READ_OP);
    readBackend.beginAction(&readOpCtx);

    // Verify homogeneous_time
    void* data_ptr = nullptr;
    int datatype = alconst::integer_data;
    int dim = 0;
    int size[H5S_MAX_RANK];
    readBackend.readData(&readOpCtx, "ids_properties/homogeneous_time", "", &data_ptr, &datatype, &dim, size);
    if (data_ptr && *(int*)data_ptr == 1) {
        std::cout << GREEN << "  [OK] Verified homogeneous_time = 1\n" << RESET;
    } else {
        std::cerr << RED << "  [FAIL] homogeneous_time mismatch\n" << RESET;
        return 1;
    }
    if(data_ptr) free(data_ptr);

    // Verify time array
    datatype = alconst::double_data;
    readBackend.readData(&readOpCtx, "time", "time", &data_ptr, &datatype, &dim, size);
    if (dim == 1 && size[0] == dynamicsize) {
        double* vals = (double*)data_ptr;
        bool ok = true;
        for(int i=0; i<dynamicsize; ++i) {
            if (std::abs(vals[i] - (0.1 * i)) > 1e-9) ok = false;
        }
        if (ok) std::cout << GREEN << "  [OK] Verified time array\n" << RESET;
        else {
             std::cerr << RED << "  [FAIL] Time array values mismatch\n" << RESET;
             return 1;
        }
    } else {
        std::cerr << RED << "  [FAIL] Time array dimension mismatch\n" << RESET;
        return 1;
    }
    if(data_ptr) free(data_ptr);

    // Verify flux_loop data
    ArraystructContext readFluxCtx(&readOpCtx, "flux_loop", "");
    int read_size = 0;
    readBackend.beginArraystructAction(&readFluxCtx, &read_size);
    
    if (read_size != staticsize) {
        std::cerr << RED << "  [FAIL] flux_loop size mismatch. Expected " << staticsize << ", got " << read_size << "\n" << RESET;
        return 1;
    }

    for (int j = 0; j < read_size; ++j) {
        datatype = alconst::double_data;
        readBackend.readData(&readFluxCtx, "flux/data", "time", &data_ptr, &datatype, &dim, size);
        
        if (dim == 1 && size[0] == dynamicsize) {
            double* vals = (double*)data_ptr;
            bool ok = true;
            for(int i=0; i<dynamicsize; ++i) {
                double expected = j * 100.0 + i;
                if (std::abs(vals[i] - expected) > 1e-9) {
                    ok = false;
                    std::cerr << RED << "    Mismatch at flux_loop[" << j << "].flux.data[" << i << "]: " 
                              << vals[i] << " != " << expected << "\n" << RESET;
                }
            }
            if (ok) std::cout << GREEN << "    [OK] Verified flux_loop[" << j << "].flux.data\n" << RESET;
            else return 1;
        } else {
            std::cerr << RED << "    [FAIL] flux_loop[" << j << "].flux.data dimension mismatch\n" << RESET;
            return 1;
        }
        if(data_ptr) free(data_ptr);

        if (j < read_size - 1) readFluxCtx.nextIndex(1);
    }
    readBackend.endAction(&readFluxCtx);
    readBackend.endAction(&readOpCtx);
    readBackend.closePulse(&readCtx, OPEN_PULSE);

    std::cout << BOLD << GREEN << "\n✓ Test passed successfully!\n" << RESET;

  } catch (const std::exception &e) {
    std::cerr << RED << BOLD << "\nException caught: " << e.what() << RESET << std::endl;
    return 1;
  }

  return 0;
}
