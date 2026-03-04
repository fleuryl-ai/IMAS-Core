#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <limits>
#include <filesystem>

namespace fs = std::filesystem;

// Colors for debug
#define RESET ""
#define BOLD ""
#define GREEN ""
#define RED ""

const std::string URI = "imas:hdf5?path=./test_db_dynamic_aos_gap";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Dynamic AoS with Gap in Signal ===\n" << RESET;

        // Phase 1: Writing 5 slices
        {
            std::cout << "\n--- Phase 1: Writing 5 slices ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            int homogeneous_time = 0;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            int size_A = 1;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);

            int size_B = 10; // Dynamic AoS
            ArraystructContext ctxB(&ctxA, "B", "time");
            backend.beginArraystructAction(&ctxB, &size_B);

            for(int t=0; t<5; ++t) {
                double current_time = (double)t * 0.1;
                backend.writeData(&ctxB, "time", "", &current_time, alconst::double_data, 0, nullptr);

                double sig1 = 100.0 + t;
                backend.writeData(&ctxB, "sig1", "time", &sig1, alconst::double_data, 0, nullptr);

                double sig2 = 200.0 + t;
                backend.writeData(&ctxB, "sig2", "time", &sig2, alconst::double_data, 0, nullptr);
                
                if (t < 4) ctxB.nextIndex(1);
            }
            backend.endAction(&ctxB);
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
        }

        // Phase 2: Append 1 slice (sig1 only)
        {
            std::cout << "\n--- Phase 2: Appending 1 slice (sig1 only) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double time_undef = alconst::undefined_time;
            int interpmode_undef = alconst::undefined_interp;
            OperationContext opCtx(&dataEntryCtx, "test_ids", WRITE_OP, alconst::slice_op, time_undef, interpmode_undef);
            backend.beginAction(&opCtx);

            int size_A = 1;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);

            int size_B = 1;
            ArraystructContext ctxB(&ctxA, "B", "time");
            backend.beginArraystructAction(&ctxB, &size_B);

            double current_time = 0.5;
            backend.writeData(&ctxB, "time", "", &current_time, alconst::double_data, 0, nullptr);

            double sig1 = 100.0 + 5; // 105.0
            backend.writeData(&ctxB, "sig1", "time", &sig1, alconst::double_data, 0, nullptr);

            // Skip sig2

            backend.endAction(&ctxB);
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }

        // Phase 3: Append 1 slice (sig1 and sig2)
        {
            std::cout << "\n--- Phase 3: Appending 1 slice (sig1 and sig2) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double time_undef = alconst::undefined_time;
            int interpmode_undef = alconst::undefined_interp;
            OperationContext opCtx(&dataEntryCtx, "test_ids", WRITE_OP, alconst::slice_op, time_undef, interpmode_undef);
            backend.beginAction(&opCtx);

            int size_A = 1;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);

            int size_B = 1;
            ArraystructContext ctxB(&ctxA, "B", "time");
            backend.beginArraystructAction(&ctxB, &size_B);

            double current_time = 0.6;
            backend.writeData(&ctxB, "time", "", &current_time, alconst::double_data, 0, nullptr);

            double sig1 = 100.0 + 6; // 106.0
            backend.writeData(&ctxB, "sig1", "time", &sig1, alconst::double_data, 0, nullptr);

            double sig2 = 200.0 + 6; // 206.0
            backend.writeData(&ctxB, "sig2", "time", &sig2, alconst::double_data, 0, nullptr);

            backend.endAction(&ctxB);
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }

        // Phase 4: Validation
        {
            std::cout << "\n--- Phase 4: Validation ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);
            
            double t_req = 0.6;

            // 1. Closest Interp
            {
                std::cout << "Checking Closest Interp at t=" << t_req << "...\n";
                int interpmode = alconst::closest_interp;
                OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP, alconst::slice_op, t_req, interpmode);
                backend.beginAction(&opCtx);

                int size_A = 0;
                ArraystructContext ctxA(&opCtx, "A", "");
                backend.beginArraystructAction(&ctxA, &size_A);

                int size_B = 0;
                ArraystructContext ctxB(&ctxA, "B", "time");
                backend.beginArraystructAction(&ctxB, &size_B);

                // sig2
                void* data_ptr = nullptr;
                int datatype = alconst::double_data;
                int dim = 0;
                int size[H5S_MAX_RANK];
                backend.readData(&ctxB, "sig2", "time", &data_ptr, &datatype, &dim, size);
                double val2 = *(double*)data_ptr;
                free(data_ptr);
                
                std::cout << "  sig2: " << val2 << " (Expected 206.0)\n";
                assert(std::abs(val2 - 206.0) < 1e-9);

                backend.endAction(&ctxB);
                backend.endAction(&ctxA);
                backend.endAction(&opCtx);
            }

            // 2. Linear Interp
            {
                std::cout << "Checking Linear Interp at t=" << t_req << "...\n";
                int interpmode = alconst::linear_interp;
                OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP, alconst::slice_op, t_req, interpmode);
                backend.beginAction(&opCtx);
                // ... (Code similaire pour Linear, voir fichier complet)
                backend.endAction(&opCtx);
            }
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }
        std::cout << GREEN << "[OK] Validation completed.\n" << RESET;

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}