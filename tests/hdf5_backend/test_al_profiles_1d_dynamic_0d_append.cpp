// @file  test_al_profiles_1d_dynamic_0d_append.cpp
// @brief Tests appending slices to a profiles_1d dynamic AoS with 0D/1D
//        signals, nested ion, time_slice and constraints, then validation.
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

const std::string URI = "imas:hdf5?path=./test_db_profiles_1d_dynamic_0d_append";

int main() {
    try {
        std::cout << BOLD << "\n=== Test profiles_1d Dynamic AoS with 0D/1D Signal Append + Root Static + Nested Ion + Nested Constraints ===\n" << RESET;

        const int spatial_1d_size = 5;
        const int ts_1d_size = 3;
        const int ion_size = 2;
        const int x_point_size = 2;

        // 1. Phase 1: Write initial slice (Global Write of 1 element)
        {
            std::cout << "\n--- Phase 1: Writing Initial Slice (t=0.0) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            auto version = backend.getVersion(&dataEntryCtx);
            //std::cout << "Backend version: " << version.first << "." << version.second << std::endl;

            if (version == std::make_pair(1,0)) {
                backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
                return 0;
            } 
            

            OperationContext opCtx(&dataEntryCtx, "core_profiles", "", WRITE_OP);
            backend.beginAction(&opCtx);

            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Static root signal
            double stat_val = 999.0;
            backend.writeData(&opCtx, "stat/sig_stat0D", "", &stat_val, alconst::double_data, 0, nullptr);

            // Time array for initial slice
            double time_val = 0.0;
            int time_dim = 1;
            int time_size[] = {1};
            backend.writeData(&opCtx, "time", "time", &time_val, alconst::double_data, time_dim, time_size);

            // profiles_1d Dynamic AoS
            ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
            int p_size = 1;
            backend.beginArraystructAction(&profilesCtx, &p_size);

            // 0D signal inside profiles_1d
            double val = 100.0;
            // dim=0 for scalar.
            backend.writeData(&profilesCtx, "signal_0d", "", &val, alconst::double_data, 0, nullptr);

            // 1D signal inside profiles_1d
            std::vector<double> sig1d_val(spatial_1d_size);
            for(int x=0; x<spatial_1d_size; ++x) sig1d_val[x] = 1000.0 + 0 * 100.0 + x;
            int sig1d_dim = 1;
            int sig1d_size[] = {spatial_1d_size};
            backend.writeData(&profilesCtx, "signal_1d", "", sig1d_val.data(), alconst::double_data, sig1d_dim, sig1d_size);

            // ion AoS inside profiles_1d
            ArraystructContext ionCtx(&profilesCtx, "ion", "");
            int ion_sz = ion_size;
            backend.beginArraystructAction(&ionCtx, &ion_sz);
            for(int i=0; i<ion_size; ++i) {
                std::vector<double> ion_sig_val(spatial_1d_size);
                for(int x=0; x<spatial_1d_size; ++x) ion_sig_val[x] = 2000.0 + 0 * 100.0 + i * 10.0 + x;
                backend.writeData(&ionCtx, "signal_1d", "", ion_sig_val.data(), alconst::double_data, sig1d_dim, sig1d_size);
                if (i < ion_size - 1) ionCtx.nextIndex(1);
            }
            backend.endAction(&ionCtx);

            backend.endAction(&profilesCtx);

            // time_slice Dynamic AoS
            ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
            int ts_size = 1;
            backend.beginArraystructAction(&timeSliceCtx, &ts_size);
            // ts_sig_0d
            double ts_0d_val = 50.0;
            backend.writeData(&timeSliceCtx, "ts_sig_0d", "", &ts_0d_val, alconst::double_data, 0, nullptr);
            // ts_sig_1d
            std::vector<double> ts_1d_val(ts_1d_size);
            for(int x=0; x<ts_1d_size; ++x) ts_1d_val[x] = 5000.0 + 0 * 100.0 + x;
            int ts_1d_dim = 1;
            int ts_1d_size_arr[] = {ts_1d_size};
            backend.writeData(&timeSliceCtx, "ts_sig_1d", "", ts_1d_val.data(), alconst::double_data, ts_1d_dim, ts_1d_size_arr);

            // ion AoS inside time_slice
            ArraystructContext tsIonCtx(&timeSliceCtx, "ion", "");
            int ts_ion_sz = ion_size;
            backend.beginArraystructAction(&tsIonCtx, &ts_ion_sz);
            for(int i=0; i<ion_size; ++i) {
                int species_val = 10 + i;
                backend.writeData(&tsIonCtx, "species", "", &species_val, alconst::integer_data, 0, nullptr);
                if (i < ion_size - 1) tsIonCtx.nextIndex(1);
            }
            backend.endAction(&tsIonCtx);

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
        std::vector<double> append_values = {200.0, 300.0};

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

            printf("-->Appending slice at time %.1f...\n", t_append);

            // Write time
            backend.writeData(&opCtx, "time", "time", &t_append, alconst::double_data, 0, nullptr);

            printf("Appending slice at time %.1f...\n", t_append);

            // profiles_1d
            int p_size = 1;
            ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
            backend.beginArraystructAction(&profilesCtx, &p_size);

            // 0D signal
            double val = append_values[i];
            backend.writeData(&profilesCtx, "signal_0d", "", &val, alconst::double_data, 0, nullptr);

            // 1D signal
            std::vector<double> sig1d_val(spatial_1d_size);
            for(int x=0; x<spatial_1d_size; ++x) sig1d_val[x] = 1000.0 + t_idx * 100.0 + x;
            int sig1d_dim = 1;
            int sig1d_size[] = {spatial_1d_size};
            backend.writeData(&profilesCtx, "signal_1d", "", sig1d_val.data(), alconst::double_data, sig1d_dim, sig1d_size);

            printf("Appended slice %zu at time %.1f with signal_0d=%.1f and signal_1d=[%.1f, %.1f, %.1f, %.1f, %.1f]\n", 
                   i, t_append, val, sig1d_val[0], sig1d_val[1], sig1d_val[2], sig1d_val[3], sig1d_val[4]);

            // ion AoS inside profiles_1d
            ArraystructContext ionCtx(&profilesCtx, "ion", "");
            int ion_sz = ion_size;
            backend.beginArraystructAction(&ionCtx, &ion_sz);
            for(int k=0; k<ion_size; ++k) {
                std::vector<double> ion_sig_val(spatial_1d_size);
                for(int x=0; x<spatial_1d_size; ++x) ion_sig_val[x] = 2000.0 + t_idx * 100.0 + k * 10.0 + x;
                backend.writeData(&ionCtx, "signal_1d", "", ion_sig_val.data(), alconst::double_data, sig1d_dim, sig1d_size);
                if (k < ion_size - 1) ionCtx.nextIndex(1);
            }
            backend.endAction(&ionCtx);

            backend.endAction(&profilesCtx);

            // time_slice
            ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
            int ts_size = 1;
            backend.beginArraystructAction(&timeSliceCtx, &ts_size);
            // ts_sig_0d
            double ts_0d_val = 50.0 + t_idx * 50.0;
            backend.writeData(&timeSliceCtx, "ts_sig_0d", "", &ts_0d_val, alconst::double_data, 0, nullptr);
            // ts_sig_1d
            std::vector<double> ts_1d_val(ts_1d_size);
            for(int x=0; x<ts_1d_size; ++x) ts_1d_val[x] = 5000.0 + t_idx * 100.0 + x;
            int ts_1d_dim = 1;
            int ts_1d_size_arr[] = {ts_1d_size};
            backend.writeData(&timeSliceCtx, "ts_sig_1d", "", ts_1d_val.data(), alconst::double_data, ts_1d_dim, ts_1d_size_arr);

            // ion AoS inside time_slice
            ArraystructContext tsIonCtx(&timeSliceCtx, "ion", "");
            int ts_ion_sz = ion_size;
            backend.beginArraystructAction(&tsIonCtx, &ts_ion_sz);
            for(int k=0; k<ion_size; ++k) {
                int species_val = 10 + k + t_idx * 10;
                backend.writeData(&tsIonCtx, "species", "", &species_val, alconst::integer_data, 0, nullptr);
                if (k < ion_size - 1) tsIonCtx.nextIndex(1);
            }
            backend.endAction(&tsIonCtx);

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
        
        // Validate initial slice (t=0.0)
        {
             DataEntryContext dataEntryCtx(URI);
             HDF5Backend backend;
             backend.openPulse(&dataEntryCtx, OPEN_PULSE);
             
             double target_time = 0.0;
             int t_idx = 0;
             std::cout << "Validating slice at t=" << target_time << "...\n";
             OperationContext opCtx(&dataEntryCtx, "core_profiles", READ_OP, alconst::slice_op, target_time, alconst::closest_interp);
             backend.beginAction(&opCtx);
             
             // Validate stat/sig_stat0D
             void* stat_data = nullptr;
             int stat_type = alconst::double_data;
             int stat_dim = 0;
             int stat_size[H5S_MAX_RANK];
             backend.readData(&opCtx, "stat/sig_stat0D", "", &stat_data, &stat_type, &stat_dim, stat_size);
             assert(stat_dim == 0);
             if (std::abs(*(double*)stat_data - 999.0) > 1e-9) {
                 std::cerr << RED << "Mismatch stat/sig_stat0D: expected 999.0, got " << *(double*)stat_data << RESET << std::endl;
                 return 1;
             }
             free(stat_data);

             ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
             int p_size = 0;
             backend.beginArraystructAction(&profilesCtx, &p_size);
             assert(p_size == 1);
             
             void* data = nullptr;
             int type = alconst::double_data;
             int dim = 0;
             int size[H5S_MAX_RANK];
             
             backend.readData(&profilesCtx, "signal_0d", "", &data, &type, &dim, size);
             assert(dim == 0);
             double val = *(double*)data;
             if (std::abs(val - 100.0) > 1e-9) {
                 std::cerr << RED << "Mismatch at t=0.0: expected 100.0, got " << val << RESET << std::endl;
                 return 1;
             }
             free(data);

             // signal_1d
             backend.readData(&profilesCtx, "signal_1d", "", &data, &type, &dim, size);
             assert(dim == 1);
             assert(size[0] == spatial_1d_size);
             double* sig1d_vals = (double*)data;
             for(int x=0; x<spatial_1d_size; ++x) {
                 double expected = 1000.0 + t_idx * 100.0 + x;
                 if (std::abs(sig1d_vals[x] - expected) > 1e-9) {
                     std::cerr << RED << "Mismatch signal_1d t=0 x=" << x << ": expected " << expected << ", got " << sig1d_vals[x] << RESET << std::endl;
                     return 1;
                 }
             }
             free(data);

             // ion inside profiles_1d
             ArraystructContext ionCtx(&profilesCtx, "ion", "");
             int ion_sz = 0;
             backend.beginArraystructAction(&ionCtx, &ion_sz);
             assert(ion_sz == ion_size);
             for(int k=0; k<ion_size; ++k) {
                 backend.readData(&ionCtx, "signal_1d", "", &data, &type, &dim, size);
                 assert(dim == 1);
                 assert(size[0] == spatial_1d_size);
                 double* ion_vals = (double*)data;
                 for(int x=0; x<spatial_1d_size; ++x) {
                     double expected = 2000.0 + t_idx * 100.0 + k * 10.0 + x;
                     if (std::abs(ion_vals[x] - expected) > 1e-9) {
                         std::cerr << RED << "Mismatch profiles_1d/ion[" << k << "]/signal_1d t=0 x=" << x << ": expected " << expected << ", got " << ion_vals[x] << RESET << std::endl;
                         return 1;
                     }
                 }
                 free(data);
                 if (k < ion_size - 1) ionCtx.nextIndex(1);
             }
             backend.endAction(&ionCtx);
             
             backend.endAction(&profilesCtx);

             // time_slice
             ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
             int ts_size = 0;
             backend.beginArraystructAction(&timeSliceCtx, &ts_size);
             assert(ts_size == 1);
             // ts_sig_0d
             backend.readData(&timeSliceCtx, "ts_sig_0d", "", &data, &type, &dim, size);
             assert(dim == 0);
             if (std::abs(*(double*)data - 50.0) > 1e-9) {
                 std::cerr << RED << "Mismatch ts_sig_0d t=0: expected 50.0, got " << *(double*)data << RESET << std::endl;
                 return 1;
             }
             free(data);
             // ts_sig_1d
             backend.readData(&timeSliceCtx, "ts_sig_1d", "", &data, &type, &dim, size);
             assert(dim == 1);
             assert(size[0] == ts_1d_size);
             double* ts1d_vals = (double*)data;
             for(int x=0; x<ts_1d_size; ++x) {
                 double expected = 5000.0 + t_idx * 100.0 + x;
                 if (std::abs(ts1d_vals[x] - expected) > 1e-9) {
                     std::cerr << RED << "Mismatch ts_sig_1d t=0 x=" << x << ": expected " << expected << ", got " << ts1d_vals[x] << RESET << std::endl;
                     return 1;
                 }
             }
             free(data);

             // ion inside time_slice
             ArraystructContext tsIonCtx(&timeSliceCtx, "ion", "");
             int ts_ion_sz = 0;
             backend.beginArraystructAction(&tsIonCtx, &ts_ion_sz);
             assert(ts_ion_sz == ion_size);
             for(int k=0; k<ion_size; ++k) {
                 backend.readData(&tsIonCtx, "species", "", &data, &type, &dim, size);
                 assert(dim == 0);
                 if (*(int*)data != (10 + k + t_idx * 10)) {
                     std::cerr << RED << "Mismatch time_slice/ion[" << k << "]/species t=0: expected " << (10 + k + t_idx * 10) << ", got " << *(int*)data << RESET << std::endl;
                     return 1;
                 }
                 free(data);
                 if (k < ion_size - 1) tsIonCtx.nextIndex(1);
             }
             backend.endAction(&tsIonCtx);

             // constraints/x_point inside time_slice
             ArraystructContext xPointCtx(&timeSliceCtx, "constraints/x_point", "");
             int xp_sz = 0;
             backend.beginArraystructAction(&xPointCtx, &xp_sz);
             assert(xp_sz == x_point_size);
             for(int k=0; k<x_point_size; ++k) {
                 backend.readData(&xPointCtx, "sigma", "", &data, &type, &dim, size);
                 assert(dim == 0);
                 if (std::abs(*(double*)data - (0.5 + k * 0.1 + t_idx * 0.01)) > 1e-9) {
                     std::cerr << RED << "Mismatch time_slice/constraints/x_point[" << k << "]/sigma t=0: expected " << (0.5 + k * 0.1 + t_idx * 0.01) << ", got " << *(double*)data << RESET << std::endl;
                     return 1;
                 }
                 free(data);
                 if (k < x_point_size - 1) xPointCtx.nextIndex(1);
             }
             backend.endAction(&xPointCtx);
             backend.endAction(&timeSliceCtx);

             backend.endAction(&opCtx);
             backend.closePulse(&dataEntryCtx, OPEN_PULSE);
             std::cout << GREEN << "  [OK] Slice t=0.0 validated.\n" << RESET;
        }

        // Validate appended slices
        for (size_t i = 0; i < append_times.size(); ++i) {
             DataEntryContext dataEntryCtx(URI);
             HDF5Backend backend;
             backend.openPulse(&dataEntryCtx, OPEN_PULSE);
             
             double target_time = append_times[i];
             int t_idx = i + 1;
             std::cout << "Validating slice at t=" << target_time << "...\n";
             OperationContext opCtx(&dataEntryCtx, "core_profiles", READ_OP, alconst::slice_op, target_time, alconst::closest_interp);
             backend.beginAction(&opCtx);
             
             ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
             int p_size = 0;
             backend.beginArraystructAction(&profilesCtx, &p_size);
             assert(p_size == 1);
             
             void* data = nullptr;
             int type = alconst::double_data;
             int dim = 0;
             int size[H5S_MAX_RANK];
             
             backend.readData(&profilesCtx, "signal_0d", "", &data, &type, &dim, size);
             assert(dim == 0);
             double val = *(double*)data;
             if (std::abs(val - append_values[i]) > 1e-9) {
                 std::cerr << RED << "Mismatch at t=" << target_time << ": expected " << append_values[i] << ", got " << val << RESET << std::endl;
                 return 1;
             }
             free(data);

             // signal_1d
             backend.readData(&profilesCtx, "signal_1d", "", &data, &type, &dim, size);
             assert(dim == 1);
             assert(size[0] == spatial_1d_size);
             double* sig1d_vals = (double*)data;
             for(int x=0; x<spatial_1d_size; ++x) {
                 double expected = 1000.0 + t_idx * 100.0 + x;
                 if (std::abs(sig1d_vals[x] - expected) > 1e-9) {
                     std::cerr << RED << "Mismatch signal_1d t=" << target_time << " x=" << x << ": expected " << expected << ", got " << sig1d_vals[x] << RESET << std::endl;
                     return 1;
                 }
             }
             free(data);

             // ion inside profiles_1d
             ArraystructContext ionCtx(&profilesCtx, "ion", "");
             int ion_sz = 0;
             backend.beginArraystructAction(&ionCtx, &ion_sz);
             assert(ion_sz == ion_size);
             for(int k=0; k<ion_size; ++k) {
                 backend.readData(&ionCtx, "signal_1d", "", &data, &type, &dim, size);
                 assert(dim == 1);
                 assert(size[0] == spatial_1d_size);
                 double* ion_vals = (double*)data;
                 for(int x=0; x<spatial_1d_size; ++x) {
                     double expected = 2000.0 + t_idx * 100.0 + k * 10.0 + x;
                     if (std::abs(ion_vals[x] - expected) > 1e-9) {
                         std::cerr << RED << "Mismatch profiles_1d/ion[" << k << "]/signal_1d t=" << target_time << " x=" << x << ": expected " << expected << ", got " << ion_vals[x] << RESET << std::endl;
                         return 1;
                     }
                 }
                 free(data);
                 if (k < ion_size - 1) ionCtx.nextIndex(1);
             }
             backend.endAction(&ionCtx);
             
             backend.endAction(&profilesCtx);

             // time_slice
             ArraystructContext timeSliceCtx(&opCtx, "time_slice", "time");
             int ts_size = 0;
             backend.beginArraystructAction(&timeSliceCtx, &ts_size);
             assert(ts_size == 1);
             // ts_sig_0d
             backend.readData(&timeSliceCtx, "ts_sig_0d", "", &data, &type, &dim, size);
             assert(dim == 0);
             double expected_ts0d = 50.0 + t_idx * 50.0;
             if (std::abs(*(double*)data - expected_ts0d) > 1e-9) {
                 std::cerr << RED << "Mismatch ts_sig_0d t=" << target_time << ": expected " << expected_ts0d << ", got " << *(double*)data << RESET << std::endl;
                 return 1;
             }
             free(data);
             // ts_sig_1d
             backend.readData(&timeSliceCtx, "ts_sig_1d", "", &data, &type, &dim, size);
             assert(dim == 1);
             assert(size[0] == ts_1d_size);
             double* ts1d_vals = (double*)data;
             for(int x=0; x<ts_1d_size; ++x) {
                 double expected = 5000.0 + t_idx * 100.0 + x;
                 if (std::abs(ts1d_vals[x] - expected) > 1e-9) {
                     std::cerr << RED << "Mismatch ts_sig_1d t=" << target_time << " x=" << x << ": expected " << expected << ", got " << ts1d_vals[x] << RESET << std::endl;
                     return 1;
                 }
             }
             free(data);

             // ion inside time_slice
             ArraystructContext tsIonCtx(&timeSliceCtx, "ion", "");
             int ts_ion_sz = 0;
             backend.beginArraystructAction(&tsIonCtx, &ts_ion_sz);
             assert(ts_ion_sz == ion_size);
             for(int k=0; k<ion_size; ++k) {
                 backend.readData(&tsIonCtx, "species", "", &data, &type, &dim, size);
                 assert(dim == 0);
                 if (*(int*)data != (10 + k + t_idx * 10)) {
                     std::cerr << RED << "Mismatch time_slice/ion[" << k << "]/species t=" << target_time << ": expected " << (10 + k + t_idx * 10) << ", got " << *(int*)data << RESET << std::endl;
                     return 1;
                 }
                 free(data);
                 if (k < ion_size - 1) tsIonCtx.nextIndex(1);
             }
             backend.endAction(&tsIonCtx);

             // constraints/x_point inside time_slice
             ArraystructContext xPointCtx(&timeSliceCtx, "constraints/x_point", "");
             int xp_sz = 0;
             backend.beginArraystructAction(&xPointCtx, &xp_sz);
             assert(xp_sz == x_point_size);
             for(int k=0; k<x_point_size; ++k) {
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