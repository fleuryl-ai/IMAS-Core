// tests/hdf5_backend/test_time_slice_nested_x_point.cpp
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

const std::string URI = "imas:hdf5?path=./test_db_time_slice_nested_x_point";

int main() {
    try {
        std::cout << BOLD << "\n=== Test time_slice Dynamic AoS with Nested constraints/x_point ===\n" << RESET;

        const int x_point_size = 2;

        // 1. Phase 1: Write initial slice (Global Write of 1 element)
        {
            std::cout << "\n--- Phase 1: Writing Initial Slice (t=0.0) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            auto version = backend.getVersion(&dataEntryCtx);

            if (version == std::make_pair(1,0)) {
                backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
                return 0;
            } 
            

            OperationContext opCtx(&dataEntryCtx, "core_profiles", "", WRITE_OP);
            backend.beginAction(&opCtx);

            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Time array for initial slice
            double time_val = 0.0;
            int time_dim = 1;
            int time_size[] = {1};
            backend.writeData(&opCtx, "time", "time", &time_val, alconst::double_data, time_dim, time_size);

            // time_slice Dynamic AoS
            ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
            int ts_size = 1;
            backend.beginArraystructAction(&timeSliceCtx, &ts_size);

            // constraints/x_point AoS inside time_slice
            ArraystructContext xPointCtx(&timeSliceCtx, "constraints/x_point", "");
            int xp_sz = x_point_size;
            backend.beginArraystructAction(&xPointCtx, &xp_sz);
            for(int i=0; i<x_point_size; ++i) {
                double sigma_val = 0.5 + i * 0.1;
                backend.writeData(&xPointCtx, "sigma", "", &sigma_val, alconst::double_data, 0, nullptr);
                if (i < x_point_size - 1) xPointCtx.nextIndex(1);
            }
            backend.endAction(&xPointCtx);
            backend.endAction(&timeSliceCtx);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Phase 1 completed.\n" << RESET;
        }

        // 2. Phase 2: Append 2 slices in separate operation contexts
        std::vector<double> append_times = {0.1, 0.2};

        for (size_t i = 0; i < append_times.size(); ++i) {
            std::cout << "\n--- Phase 2: Appending Slice " << i << " (t=" << append_times[i] << ") ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double t_append = append_times[i];
            int t_idx = i + 1; // 1, 2
            int interpmode = alconst::closest_interp; 
            // Use SLICE_OP for appending
            OperationContext opCtx(&dataEntryCtx, "core_profiles", WRITE_OP, alconst::slice_op, t_append, interpmode);
            backend.beginAction(&opCtx);

            // Write time
            backend.writeData(&opCtx, "time", "time", &t_append, alconst::double_data, 0, nullptr);

            // time_slice
            ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
            int ts_size = 1;
            backend.beginArraystructAction(&timeSliceCtx, &ts_size);

            // constraints/x_point AoS inside time_slice
            ArraystructContext xPointCtx(&timeSliceCtx, "constraints/x_point", "");
            int xp_sz = x_point_size;
            backend.beginArraystructAction(&xPointCtx, &xp_sz);
            for(int k=0; k<x_point_size; ++k) {
                double sigma_val = 0.5 + k * 0.1 + t_idx * 0.01;
                backend.writeData(&xPointCtx, "sigma", "", &sigma_val, alconst::double_data, 0, nullptr);
                if (k < x_point_size - 1) xPointCtx.nextIndex(1);
            }
            backend.endAction(&xPointCtx);
            backend.endAction(&timeSliceCtx);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Slice " << i << " appended.\n" << RESET;
        }

        // 3. Phase 3: Read back slices
        std::cout << "\n--- Phase 3: Validation ---\n";
        
        std::vector<double> all_times = {0.0, 0.1, 0.2};
        for (size_t i = 0; i < all_times.size(); ++i) {
             DataEntryContext dataEntryCtx(URI);
             HDF5Backend backend;
             backend.openPulse(&dataEntryCtx, OPEN_PULSE);
             
             double target_time = all_times[i];
             int t_idx = i;
             std::cout << "Validating slice at t=" << target_time << "...\n";
             OperationContext opCtx(&dataEntryCtx, "core_profiles", READ_OP, alconst::slice_op, target_time, alconst::closest_interp);
             backend.beginAction(&opCtx);
             
             // time_slice
             ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
             int ts_size = 0;
             backend.beginArraystructAction(&timeSliceCtx, &ts_size);
             assert(ts_size == 1);

             // constraints/x_point inside time_slice
             ArraystructContext xPointCtx(&timeSliceCtx, "constraints/x_point", "");
             int xp_sz = 0;
             backend.beginArraystructAction(&xPointCtx, &xp_sz);
             assert(xp_sz == x_point_size);
             for(int k=0; k<x_point_size; ++k) {
                 void* data = nullptr;
                 int type = alconst::double_data;
                 int dim = 0;
                 int size[H5S_MAX_RANK];
                 backend.readData(&xPointCtx, "sigma", "", &data, &type, &dim, size);
                 assert(dim == 0);
                 if (std::abs(*(double*)data - (0.5 + k * 0.1 + t_idx * 0.01)) > 1e-9) {
                     std::cerr << RED << "Mismatch time_slice/constraints/x_point[" << k << "]/sigma t=" << target_time << ": expected " << (0.5 + k * 0.1 + t_idx * 0.01) << ", got " << *(double*)data << RESET << std::endl;
                     return 1;
                 }
                 free(data);
                 if (k < x_point_size - 1) xPointCtx.nextIndex(1);
             }
             backend.endAction(&xPointCtx);
             backend.endAction(&timeSliceCtx);

             backend.endAction(&opCtx);
             backend.closePulse(&dataEntryCtx, OPEN_PULSE);
             std::cout << GREEN << "  [OK] Slice t=" << target_time << " validated.\n" << RESET;
        }
        
        std::cout << GREEN << "[OK] All validations passed.\n" << RESET;

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}