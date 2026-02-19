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

        const int TIME_SLICE_SIZE = 3;
        const int CONTROL_SURFACE_SIZE = 3; // From the bug report: expected 3
        const int BUG_TRIGGER_SIZE = 2;    // Size at t=0 to trigger the bug

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

            

            // Dynamic AoS: time_slice
            ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
            int ts_size = TIME_SLICE_SIZE;
            backend.beginArraystructAction(&timeSliceCtx, &ts_size);

            for (int t = 0; t < TIME_SLICE_SIZE; ++t) {
                double current_time = t * 0.1;
                backend.writeData(&timeSliceCtx, "time", "", &current_time, alconst::double_data, 0, nullptr);

                // Add 'r' in time_slice (1D spatial field with 2 elements)
                const int R_SPATIAL_SIZE = 2;
                std::vector<double> r_values(R_SPATIAL_SIZE);
                for(int i=0; i<R_SPATIAL_SIZE; ++i) r_values[i] = (double)(t * 10 + i);
                int r_dim = 1;
                int r_size[] = {R_SPATIAL_SIZE};
                backend.writeData(&timeSliceCtx, "r", "", r_values.data(), alconst::double_data, r_dim, r_size);

                // Static AoS nested inside: control_surface
                ArraystructContext controlSurfaceCtx(&timeSliceCtx, "control_surface", "");
                
                // VARYING SIZE: t=0 -> 2 elements, t=1 -> 3 elements
                // This forces the bug to appear if the reader incorrectly looks at t=0 structure
                int cs_size = (t == 0) ? BUG_TRIGGER_SIZE : CONTROL_SURFACE_SIZE;
                
                backend.beginArraystructAction(&controlSurfaceCtx, &cs_size);

                for (int c = 0; c < cs_size; ++c) {
                    // Write some dummy data
                    const int R_CS_SIZE = 3;
                    std::vector<double> r_cs_values(R_CS_SIZE);
                    for(int k=0; k<R_CS_SIZE; ++k) r_cs_values[k] = (double)(t * 100 + c * 10 + k);
                    int r_cs_dim = 1;
                    int r_cs_size[] = {R_CS_SIZE};
                    backend.writeData(&controlSurfaceCtx, "r", "", r_cs_values.data(), alconst::double_data, r_cs_dim, r_cs_size);
                    if (c < cs_size - 1) controlSurfaceCtx.nextIndex(1);
                }
                backend.endAction(&controlSurfaceCtx);

                if (t < TIME_SLICE_SIZE - 1) timeSliceCtx.nextIndex(1);
            }
            backend.endAction(&timeSliceCtx);

            // Add 'code_library/description' (static AoS with 4 elements)
            int code_lib_size = 4;
            ArraystructContext codeLibCtx(&opCtx, "code_library", "");
            backend.beginArraystructAction(&codeLibCtx, &code_lib_size);
            
            for (int i = 0; i < code_lib_size; ++i) {
                std::string desc = "Lib " + std::to_string(i);
                int desc_dim = 1;
                int desc_size[] = {(int)desc.length()};
                backend.writeData(&codeLibCtx, "description", "", (void*)desc.c_str(), alconst::char_data, desc_dim, desc_size);
                
                if (i < code_lib_size - 1) codeLibCtx.nextIndex(1);
            }
            backend.endAction(&codeLibCtx);

            // Write global time vector
            std::vector<double> times(TIME_SLICE_SIZE);
            for(int t=0; t<TIME_SLICE_SIZE; ++t) times[t] = t * 0.1;
            int time_dim = 1;
            int time_size[] = {TIME_SLICE_SIZE};
            backend.writeData(&opCtx, "time", "time", (void*)times.data(), alconst::double_data, time_dim, time_size);

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
            
            for (int t_idx = 0; t_idx < TIME_SLICE_SIZE; ++t_idx) {
                double target_time = t_idx * 0.1;
                int expected_size = (t_idx == 0) ? BUG_TRIGGER_SIZE : CONTROL_SURFACE_SIZE;

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
                std::cout << "Expected correct size: " << expected_size << "\n";

                if (cs_size_read != expected_size) {
                    std::cout << RED << ">>> BUG DETECTED or UNEXPECTED SIZE at t=" << target_time << " <<<\n";
                    std::cout << "    Read " << cs_size_read << " elements for 'control_surface', but expected " << expected_size << ".\n" << RESET;
                    backend.endAction(&controlSurfaceCtx);
                    backend.endAction(&timeSliceCtx);
                    backend.endAction(&opCtx);
                    backend.closePulse(&dataEntryCtx, OPEN_PULSE);
                    return 1;
                } else {
                     std::cout << GREEN << ">>> TEST PASSED for t=" << target_time << ": Size is correct (" << cs_size_read << ").\n" << RESET;
                }

                backend.endAction(&controlSurfaceCtx);
                backend.endAction(&timeSliceCtx);
                backend.endAction(&opCtx);
            }
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Caught an unexpected exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0; // If bug is not reproduced, test OK.
}