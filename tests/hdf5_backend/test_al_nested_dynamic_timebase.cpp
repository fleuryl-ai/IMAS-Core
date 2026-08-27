// @file  test_al_nested_dynamic_timebase.cpp
// @brief Tests slice reading (getTimeIndex) with a timebase inside a nested
//        dynamic AoS: write, then read slices by time.
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

// ANSI colors for debug
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
              << "\n=== Test getTimeIndex with a timebase inside a "
                  "nested dynamic AoS ===\n"
              << RESET;

    // ==============================================================
    // 1. Writing data
    // ==============================================================
    {
      DataEntryContext dataEntryCtx(URI);
      HDF5Backend backend;
      backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

      OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
      backend.beginAction(&opCtx);

      // Force non-homogeneous mode so that each AoS can have its own timebase
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
            time_data[t] = (double)t * 0.1; // e.g. [0.0..0.9] for i=0
            //dyn_1d_data[t] = time_data[t] * 10.0 + 6.5; // Simple data for validation
        }
        int dim[] = {dyn_1d_time_size};
        backend.writeData(&opCtx, "time", "time", time_data.data(), alconst::double_data, 1, dim);

      // --- Static AoS 'static_aos' ---
      int static_aos_size = 2;
      ArraystructContext staticAosCtx(&opCtx, "static_aos", "");
      backend.beginArraystructAction(&staticAosCtx, &static_aos_size);

      for (int i = 0; i < static_aos_size; ++i) {
        // Static signal in the static AoS
        double static_signal_data = 100.0 + i;
        backend.writeData(&staticAosCtx, "static_signal", "",
                          &static_signal_data, alconst::double_data, 0,
                          nullptr);

        // --- Nested dynamic AoS 'dynamic_aos' ---
        int dynamic_aos_size = 10;
        ArraystructContext dynamicAosCtx(&staticAosCtx, "dynamic_aos", "time");
        backend.beginArraystructAction(&dynamicAosCtx, &dynamic_aos_size);

        // Write the timebase and the signals for this dynamic AoS
        for (int t = 0; t < dynamic_aos_size; ++t) {
          // Write the time value for this slice
          // Different times for each instance of static_aos
          double current_time = 0.1 * (t);
          backend.writeData(&dynamicAosCtx, "time", "time", &current_time,
                            alconst::double_data, 0, nullptr);

          // Write two dynamic signals
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

        // --- ADDED: 1D dynamic signal 'dyn_1d' inside static_aos ---
        // This signal has its own timebase, independent of the one from dynamic_aos
        //int dyn_1d_time_size = 10;
        //std::vector<double> time_data(dyn_1d_time_size);
        std::vector<double> dyn_1d_data(dyn_1d_time_size);
        for(int t=0; t<dyn_1d_time_size; ++t) {
            //time_data[t] = (double)t * 0.1 + i; // e.g. [0.0..0.9] for i=0
            dyn_1d_data[t] = time_data[t] * 10.0 + 6.5; // Simple data for validation
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

      std::cout << GREEN << "[OK] Write complete.\n" << RESET;
    }

    // ==============================================================
    // 2. Reading by Slices
    // ==============================================================
    std::cout << BOLD << MAGENTA << "\n[3] Slice reading test...\n" << RESET;

    std::vector<double> target_times = {0.55, 0.85};
    double expected_val[2] = {1055, 1085};  
    int k = 0;
    for (double target_time : target_times) {
        
        try {
            DataEntryContext readDataEntryCtx(URI);
            HDF5Backend readBackend;
            readBackend.openPulse(&readDataEntryCtx, OPEN_PULSE);

            std::cout << YELLOW << "[DEBUG] Reading slice t=" << target_time << "...\n" << RESET;

            int interpmode = alconst::linear_interp;
            OperationContext opCtx(&readDataEntryCtx, "test_ids", READ_OP, alconst::slice_op, target_time, interpmode);
            readBackend.beginAction(&opCtx);

            // static_aos
            ArraystructContext staticAosCtx(&opCtx, "static_aos", "");
            int static_size = 0;
            readBackend.beginArraystructAction(&staticAosCtx, &static_size);
            assert(static_size == 2);

            for (int i = 0; i < static_size; ++i) {
                
                // --- Validate dyn_1d (in static_aos) ---
                void* dyn_ptr = nullptr;
                int dyn_datatype = alconst::double_data;
                int dyn_dim = 0;
                int dyn_size[H5S_MAX_RANK];
                
                readBackend.readData(&staticAosCtx, "dyn_1d", "time", &dyn_ptr, &dyn_datatype, &dyn_dim, dyn_size);
                
                assert(dyn_dim == 1); // In slice mode, we read a scalar
                
                double read_val = static_cast<double*>(dyn_ptr)[0];
                // The written data is time*10 + 6.5, so the interpolated value must be target_time*10 + 6.5
                double expected_val1 = target_time * 10.0 + 6.5;

                if (std::abs(read_val - expected_val1 ) > 1e-6) {
                    std::cerr << RED << "Error reading dyn_1d t=" << target_time << " static_aos[" << i << "]: expected " << expected_val << ", got " << read_val << RESET << std::endl;
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
                          std::cerr << RED << "Error reading signal1 at t=" << target_time << " static_aos[" << i << "]: "
                                   << "expected " << expected_val[k]  << ", got " << read_val << RESET << std::endl;
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
            std::cout << GREEN << "  [OK] Slice t=" << target_time << " validated.\n" << RESET;

        } catch (const std::exception &e) {
            std::cerr << RED << "Exception while reading t=" << target_time << ": " << e.what() << RESET << std::endl;
            return 1;
        }
        k++;
    }

    std::cout << BOLD << GREEN << "\n✓ All tests passed!\n" << RESET;

  } catch (const std::exception &e) {
    std::cerr << RED << BOLD << "\nException caught: " << e.what() << RESET
              << std::endl;
    return 1;
  }

  

  return 0;
}