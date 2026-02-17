// tests/hdf5_backend/test_bug_b_field_na_slice.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Colors for debug
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_b_field_na_slice_bug";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Bug Reproduction: Slice of b_field_non_axisymmetric (Homogeneous=0) ===\n" << RESET;

        // Cleanup
        if (fs::exists("test_db_b_field_na_slice_bug")) {
            fs::remove_all("test_db_b_field_na_slice_bug");
        }

        const int TIME_STEPS = 3;
        const int FIELD_MAP_SIZE = 1;
        const int R_SIZE = 5;

        // ===================================================================
        // 1. Writing Phase (simulates ids_put)
        // ===================================================================
        {
            std::cout << "\n--- Phase 1: Writing Data (Global Write) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "b_field_non_axisymmetric", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Set homogeneous_time = 0 (Inhomogeneous / Time inside structure)
            int homogeneous_time = 0;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Dynamic AoS: time_slice
            ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
            int ts_size = TIME_STEPS;
            backend.beginArraystructAction(&timeSliceCtx, &ts_size);

            for (int t = 0; t < TIME_STEPS; ++t) {
                // Write time for this slice (scalar time per slice)
                double current_time = t * 0.1;
                backend.writeData(&timeSliceCtx, "time", "", &current_time, alconst::double_data, 0, nullptr);

                // Static AoS: field_map
                ArraystructContext fieldMapCtx(&timeSliceCtx, "field_map", "");
                int fm_size = FIELD_MAP_SIZE;
                backend.beginArraystructAction(&fieldMapCtx, &fm_size);

                for (int f = 0; f < FIELD_MAP_SIZE; ++f) {
                    // Write r (1D signal) inside grid structure
                    // Path: time_slice/field_map/grid/r
                    std::vector<double> r_data(R_SIZE);
                    for(int i=0; i<R_SIZE; ++i) r_data[i] = t * 100.0 + f * 10.0 + i;
                    
                    int r_dim = 1;
                    int r_size[] = {R_SIZE};
                    // Note: "grid" is a structure, so we write "grid/r" directly from field_map context
                    backend.writeData(&fieldMapCtx, "grid/r", "", r_data.data(), alconst::double_data, r_dim, r_size);

                    if (f < FIELD_MAP_SIZE - 1) fieldMapCtx.nextIndex(1);
                }
                backend.endAction(&fieldMapCtx);

                if (t < TIME_STEPS - 1) timeSliceCtx.nextIndex(1);
            }
            backend.endAction(&timeSliceCtx);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // ===================================================================
        // 2. Reading Phase (simulates ids_get_slice) and Verification
        // ===================================================================
        {
            std::cout << "\n--- Phase 2: Reading Slice and checking for bug ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double target_time = 0.1; // Corresponds to index 1

            std::cout << "Attempting to read slice at t=" << target_time << "...\n";
            OperationContext opCtx(&dataEntryCtx, "b_field_non_axisymmetric", READ_OP, alconst::slice_op, target_time, alconst::closest_interp);
            backend.beginAction(&opCtx);

            // Navigate to the data
            ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
            int ts_size_read = 0;
            backend.beginArraystructAction(&timeSliceCtx, &ts_size_read);
            assert(ts_size_read == 1); // Slice mode: size is 1

            ArraystructContext fieldMapCtx(&timeSliceCtx, "field_map", "");
            int fm_size_read = 0;
            backend.beginArraystructAction(&fieldMapCtx, &fm_size_read);
            assert(fm_size_read == FIELD_MAP_SIZE);

            // This call should throw the exception if the bug is present
            void* data = nullptr;
            int datatype = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];
            
            backend.readData(&fieldMapCtx, "grid/r", "", &data, &datatype, &dim, size);
            
            std::cout << GREEN << ">>> TEST PASSED: readData did not throw.\n" << RESET;
            if (data) free(data);

            backend.endAction(&fieldMapCtx);
            backend.endAction(&timeSliceCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }

    } catch (const std::exception& e) {
        std::string error_msg = e.what();
        if (error_msg.find("Unexpected time vector with size 0") != std::string::npos) {
            std::cout << RED << ">>> BUG REPRODUCED SUCCESSFULLY <<<\n";
            std::cout << "    Caught expected exception: " << error_msg << RESET << std::endl;
            return 0; // Test succeeds by reproducing the bug
        }
        std::cerr << RED << "Caught an unexpected exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 1; // If no exception is caught, the bug is not reproduced.
}