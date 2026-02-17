// tests/hdf5_backend/test_bug_dynamic_aos_nested_static.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_dynamic_aos_nested_static_bug";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Bug Reproduction: Slice of dynamic AoS with nested static AoS ===\n" << RESET;

        // Cleanup
        if (fs::exists("test_db_dynamic_aos_nested_static_bug")) {
            fs::remove_all("test_db_dynamic_aos_nested_static_bug");
        }

        const int TIME_SLICE_SIZE = 2;
        const int CONTROL_SURFACE_SIZE = 3; // From the bug report: expected 3
        const int EXPECTED_BUG_SIZE = 2;    // From the bug report: got 2

        // ===================================================================
        // 1. Écriture (simule ids_put of b_field_non_axisymmetric)
        // ===================================================================
        {
            std::cout << "\n--- Phase 1: Writing Data (Global Write) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "b_field_non_axisymmetric", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Set homogeneous_time = 1
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Write global time vector
            std::vector<double> times(TIME_SLICE_SIZE);
            for(int t=0; t<TIME_SLICE_SIZE; ++t) times[t] = t * 0.1;
            int time_dim = 1;
            int time_size[] = {TIME_SLICE_SIZE};
            backend.writeData(&opCtx, "time", "time", (void*)times.data(), alconst::double_data, time_dim, time_size);

            // Dynamic AoS: time_slice
            ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
            int ts_size = TIME_SLICE_SIZE;
            backend.beginArraystructAction(&timeSliceCtx, &ts_size);

            for (int t = 0; t < TIME_SLICE_SIZE; ++t) {
                // Static AoS nested inside: control_surface
                ArraystructContext controlSurfaceCtx(&timeSliceCtx, "control_surface", "");
                int cs_size = CONTROL_SURFACE_SIZE;
                backend.beginArraystructAction(&controlSurfaceCtx, &cs_size);

                for (int c = 0; c < CONTROL_SURFACE_SIZE; ++c) {
                    // Write some dummy data
                    double r_val = 10.0 * t + c;
                    backend.writeData(&controlSurfaceCtx, "r", "", &r_val, alconst::double_data, 0, nullptr);
                    if (c < CONTROL_SURFACE_SIZE - 1) controlSurfaceCtx.nextIndex(1);
                }
                backend.endAction(&controlSurfaceCtx);

                if (t < TIME_SLICE_SIZE - 1) timeSliceCtx.nextIndex(1);
            }
            backend.endAction(&timeSliceCtx);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // ===================================================================
        // 2. Lecture (simule ids_get_slice) et Vérification
        // ===================================================================
        {
            std::cout << "\n--- Phase 2: Reading Slice and checking for bug ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double target_time = 0.1; // Corresponds to index 1 in our time vector
            int target_time_idx = 1;

            std::cout << "Attempting to read slice at t=" << target_time << "...\n";
            OperationContext opCtx(&dataEntryCtx, "b_field_non_axisymmetric", READ_OP, alconst::slice_op, target_time, alconst::closest_interp);
            backend.beginAction(&opCtx);

            // Dynamic AoS: time_slice
            ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
            int ts_size_read = 0;
            backend.beginArraystructAction(&timeSliceCtx, &ts_size_read);
            assert(ts_size_read == 1); // In slice mode, the dynamic AoS has size 1

            // Static AoS: control_surface
            ArraystructContext controlSurfaceCtx(&timeSliceCtx, "control_surface", "");
            int cs_size_read = 0;
            backend.beginArraystructAction(&controlSurfaceCtx, &cs_size_read);
            
            std::cout << "Size of nested static AoS 'control_surface' read from slice: " << cs_size_read << "\n";
            std::cout << "Expected correct size: " << CONTROL_SURFACE_SIZE << "\n";
            std::cout << "Expected bug size: " << EXPECTED_BUG_SIZE << "\n";

            if (cs_size_read == EXPECTED_BUG_SIZE) {
                std::cout << RED << ">>> BUG REPRODUCED SUCCESSFULLY <<<\n";
                std::cout << "    Read " << cs_size_read << " elements for 'control_surface', but expected " << CONTROL_SURFACE_SIZE << ".\n" << RESET;
                backend.endAction(&controlSurfaceCtx);
                backend.endAction(&timeSliceCtx);
                backend.endAction(&opCtx);
                backend.closePulse(&dataEntryCtx, OPEN_PULSE);
                return 0; // Test succeeds by reproducing the bug
            } else if (cs_size_read == CONTROL_SURFACE_SIZE) {
                 std::cout << GREEN << ">>> TEST PASSED: Size is correct (" << cs_size_read << "). The bug is not reproduced.\n" << RESET;
            } else {
                std::cerr << RED << ">>> UNEXPECTED RESULT: Read size is " << cs_size_read << ". Neither correct nor the expected bug size.\n" << RESET;
                return 1;
            }

            backend.endAction(&controlSurfaceCtx);
            backend.endAction(&timeSliceCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Caught an unexpected exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 1; // If bug is not reproduced, test fails.
}