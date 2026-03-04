// tests/hdf5_backend/test_static_aos_dynamic_signal_slice.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

// Colors for debug
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_static_aos_dynamic_signal_slice";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Static AoS with Dynamic Signal (Slice by Slice Write/Read) ===\n" << RESET;

        // Cleanup
        if (fs::exists("test_db_static_aos_dynamic_signal_slice")) {
            fs::remove_all("test_db_static_aos_dynamic_signal_slice");
        }

        const int STATIC_SIZE = 2;
        const int NUM_SLICES = 3;

        // 1. Writing Slices
        std::cout << "\n--- Phase 1: Writing " << NUM_SLICES << " slices ---\n";
        for (int t = 0; t < NUM_SLICES; ++t) {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            
            // Create for first slice, Open for subsequent
            int open_mode = (t == 0) ? FORCE_CREATE_PULSE : OPEN_PULSE;
            backend.openPulse(&dataEntryCtx, open_mode);

            double current_time = t * 1.0;
            std::cout << "Writing slice " << t << " at t=" << current_time << "...\n";

            int interpmode = alconst::undefined_interp;
            OperationContext opCtx(&dataEntryCtx, "test_ids", WRITE_OP, alconst::slice_op, current_time, interpmode);
            backend.beginAction(&opCtx);

            if (t == 0) {
                int homogeneous_time = 0;
                backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);
            }

            // Write time
            //backend.writeData(&opCtx, "time", "time", &current_time, alconst::double_data, 0, nullptr);

            // Static AoS
            ArraystructContext staticCtx(&opCtx, "static_aos", "");
            int size = STATIC_SIZE;
            backend.beginArraystructAction(&staticCtx, &size);

            for (int i = 0; i < STATIC_SIZE; ++i) {
                // Dynamic signal inside static AoS
                double val = 100.0 + i * 10.0 + t;
                // dim=0 for scalar slice
                backend.writeData(&staticCtx, "dynamic_signal", "time", &val, alconst::double_data, 0, nullptr);
                backend.writeData(&staticCtx, "time", "time", &current_time, alconst::double_data, 0, nullptr);
                
                if (i < STATIC_SIZE - 1) staticCtx.nextIndex(1);
            }
            backend.endAction(&staticCtx);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, open_mode);
        }
        std::cout << GREEN << "[OK] Writing completed.\n" << RESET;

        // 2. Reading Slices
        std::cout << "\n--- Phase 2: Reading Slices ---\n";
        for (int t = 0; t < NUM_SLICES; ++t) {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double target_time = t * 1.0;
            std::cout << "Reading slice at t=" << target_time << "...\n";

            OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP, alconst::slice_op, target_time, alconst::closest_interp);
            backend.beginAction(&opCtx);

            ArraystructContext staticCtx(&opCtx, "static_aos", "");
            int size = 0;
            backend.beginArraystructAction(&staticCtx, &size);
            assert(size == STATIC_SIZE);

            for (int i = 0; i < STATIC_SIZE; ++i) {
                void* data = nullptr;
                int type = alconst::double_data;
                int dim = 0;
                int dims[H5S_MAX_RANK];

                backend.readData(&staticCtx, "dynamic_signal", "time", &data, &type, &dim, dims);
                
                assert(dim == 0);
                double val = *(double*)data;
                double expected = 100.0 + i * 10.0 + t;
                
                if (std::abs(val - expected) > 1e-9) {
                    std::cerr << RED << "Mismatch at t=" << t << " i=" << i << ": expected " << expected << ", got " << val << RESET << std::endl;
                    return 1;
                }
                free(data);

                if (i < STATIC_SIZE - 1) staticCtx.nextIndex(1);
            }
            backend.endAction(&staticCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }
        std::cout << GREEN << "[OK] Reading validation completed.\n" << RESET;

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}