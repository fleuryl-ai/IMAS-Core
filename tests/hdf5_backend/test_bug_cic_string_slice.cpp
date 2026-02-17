// tests/hdf5_backend/test_bug_cic_string_slice.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Colors for debug
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_cic_string_slice_bug";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Bug Reproduction: Slice of list-of-strings(1) in nested AoS ===\n" << RESET;

        // Cleanup
        if (fs::exists("test_db_cic_string_slice_bug")) {
            fs::remove_all("test_db_cic_string_slice_bug");
        }

        const int CHANGE_SIZE = 1;
        const int TIME_STEPS = 3;
        const std::string TEST_STRING = "TestSourceString";

        // ===================================================================
        // 1. Writing Phase (simulates ids_put of core_instant_changes)
        // ===================================================================
        {
            std::cout << "\n--- Phase 1: Writing Data (Global Write) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "core_instant_changes", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Set homogeneous_time = 1
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Write global time vector
            std::vector<double> times(TIME_STEPS);
            for(int t=0; t<TIME_STEPS; ++t) times[t] = t * 0.1;
            int time_dim = 1;
            int time_size[] = {TIME_STEPS};
            backend.writeData(&opCtx, "time", "time", (void*)times.data(), alconst::double_data, time_dim, time_size);

            // Static AoS: change
            ArraystructContext changeCtx(&opCtx, "change", "");
            int change_size = CHANGE_SIZE;
            backend.beginArraystructAction(&changeCtx, &change_size);

            // Dynamic AoS nested inside: profiles_1d
            ArraystructContext profilesCtx(&changeCtx, "profiles_1d", "time");
            int profiles_size = TIME_STEPS;
            backend.beginArraystructAction(&profilesCtx, &profiles_size);

            for (int t = 0; t < TIME_STEPS; ++t) {
                // The problematic field: a list of strings with only one element.
                const char* single_string = TEST_STRING.c_str();
                int n_strings = 1;
                int max_len = strlen(single_string) + 1;

                // Prepare buffer for writeData (flat char array)
                // dim = 2, size = {n_strings, max_len}
                int dim = 2;
                int size[] = {n_strings, max_len};
                
                std::vector<char> buffer(n_strings * max_len, 0);
                strncpy(buffer.data(), single_string, max_len);

                // Path is relative to the current context (profilesCtx)
                backend.writeData(&profilesCtx, "electrons/temperature_fit/source", "time", buffer.data(), alconst::char_data, dim, size);

                if (t < TIME_STEPS - 1) profilesCtx.nextIndex(1);
            }
            backend.endAction(&profilesCtx);
            backend.endAction(&changeCtx);

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
            OperationContext opCtx(&dataEntryCtx, "core_instant_changes", READ_OP, alconst::slice_op, target_time, alconst::closest_interp);
            backend.beginAction(&opCtx);

            // Navigate to the data
            ArraystructContext changeCtx(&opCtx, "change", "");
            int change_size_read = 0;
            backend.beginArraystructAction(&changeCtx, &change_size_read);
            assert(change_size_read == CHANGE_SIZE);

            ArraystructContext profilesCtx(&changeCtx, "profiles_1d", "time");
            int profiles_size_read = 0;
            backend.beginArraystructAction(&profilesCtx, &profiles_size_read);
            assert(profiles_size_read == 1); // Slice mode

            // This call should throw the exception if the bug is present
            void* data = nullptr;
            int datatype = alconst::char_data;
            int dim = 0;
            int size[H5S_MAX_RANK];
            backend.readData(&profilesCtx, "electrons/temperature_fit/source", "time", &data, &datatype, &dim, size);

            std::cout << "Read dimension: " << dim << "\n";
            
            if (dim == 1) {
                std::cout << RED << ">>> BUG REPRODUCED: Got dim=1 for a list of strings (expected 2).\n";
                std::cout << "    The backend returned a scalar string instead of a list of size 1.\n" << RESET;
                return 0; // Success in reproducing the bug
            } else {
                std::cout << GREEN << ">>> TEST PASSED: Got dim=" << dim << " (expected 2).\n" << RESET;
            }

            backend.endAction(&profilesCtx);
            backend.endAction(&changeCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }

    } catch (const std::exception& e) {
        std::string error_msg = e.what();
        if (error_msg.find("Wrong dimension of Data returned by backend") != std::string::npos) {
            std::cout << RED << ">>> BUG REPRODUCED SUCCESSFULLY <<<\n";
            std::cout << "    Caught expected exception: " << error_msg << RESET << std::endl;
            return 0; // Test succeeds by reproducing the bug
        }
        std::cerr << RED << "Caught an unexpected exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0; // If no exception is caught, the test is passed.
}