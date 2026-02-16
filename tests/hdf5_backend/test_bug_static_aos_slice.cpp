// tests/hdf5_backend/test_bug_static_aos_slice.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_static_aos_slice_bug";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Bug Reproduction: Slice of dynamic signal in static AoS ===\n" << RESET;

        // Cleanup
        if (fs::exists("test_db_static_aos_slice_bug")) {
            fs::remove_all("test_db_static_aos_slice_bug");
        }

        const int flux_loop_size = 2;
        const int n_time_steps = 3;
        const std::vector<double> times = {0.1, 0.2, 0.3};

        // ===================================================================
        // 1. Écriture (simule ids_put of magnetics)
        // ===================================================================
        {
            std::cout << "\n--- Phase 1: Writing Data (Global Write) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "magnetics", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Set homogeneous_time = 0 because time vectors are nested
            int homogeneous_time = 0;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Static AoS: flux_loop
            ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
            int fl_size = flux_loop_size;
            backend.beginArraystructAction(&fluxLoopCtx, &fl_size);

            for (int i = 0; i < flux_loop_size; ++i) {
                // Inside each flux_loop element, we have a dynamic signal 'flux/data'
                // with its own time vector 'flux/time'

                // Write flux/time
                int time_dim = 1;
                int time_size[] = {n_time_steps};
                backend.writeData(&fluxLoopCtx, "time", "time", (void*)times.data(), alconst::double_data, time_dim, time_size);

                // Write flux/data
                std::vector<double> flux_data;
                flux_data.reserve(n_time_steps);
                for(int t=0; t<n_time_steps; ++t) {
                    flux_data.push_back(10.0 * i + t); // e.g., for i=0: {0,1,2}, for i=1: {10,11,12}
                }
                int data_dim = 1;
                int data_size[] = {n_time_steps};
                backend.writeData(&fluxLoopCtx, "flux/data", "time", (void*)flux_data.data(), alconst::double_data, data_dim, data_size);

                if (i < flux_loop_size - 1) fluxLoopCtx.nextIndex(1);
            }
            backend.endAction(&fluxLoopCtx);

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

            double target_time = 0.2; // Corresponds to index 1 in our time vector

            std::cout << "Attempting to read slice at t=" << target_time << "...\n";
            OperationContext opCtx(&dataEntryCtx, "magnetics", READ_OP, alconst::slice_op, target_time, alconst::closest_interp);
            backend.beginAction(&opCtx);

            ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
            int fl_size_read = 0;
            backend.beginArraystructAction(&fluxLoopCtx, &fl_size_read);
            assert(fl_size_read == flux_loop_size);

            std::cout << "Reading from flux_loop[0]...\n";

            void* data = nullptr;
            int type = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            // This is the call that fails in MATLAB. The backend must find 'flux_loop/0/time'.
            backend.readData(&fluxLoopCtx, "flux/data", "time", &data, &type, &dim, size);

            std::cout << GREEN << ">>> TEST PASSED: readData did not throw an exception.\n" << RESET;
            std::cout << "    This means the bug is likely fixed or this test doesn't reproduce it.\n";

            free(data);

            backend.endAction(&fluxLoopCtx);
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