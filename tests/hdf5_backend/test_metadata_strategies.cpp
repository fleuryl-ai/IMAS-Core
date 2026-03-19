// tests/hdf5_backend/test_metadata_strategies.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <cstring>

namespace fs = std::filesystem;

#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_metadata_strategies";

// Helper to write metadata (scalar string)
void write_meta(HDF5Backend& backend, OperationContext* ctx, const std::string& path, const std::string& value) {
    const char* c_str = value.c_str();
    int dim = 1;
    int size[] = {(int)value.length()};
    backend.writeData(ctx, path, "", (void*)c_str, alconst::char_data, dim, size);
}

// Helper to verify metadata
void verify_meta(HDF5Backend& backend, OperationContext* ctx, const std::string& path, const std::string& expected) {
    void* data = nullptr;
    int type = alconst::char_data;
    int dim = 0;
    int size[H5S_MAX_RANK];
    
    backend.readData(ctx, path, "", &data, &type, &dim, size);
    
    if (data == nullptr) {
        std::cerr << RED << "[FAIL] Metadata " << path << " not found (null)." << RESET << std::endl;
        throw std::runtime_error("Metadata missing");
    }
    
    std::string val((char*)data);
    if (val != expected) {
        std::cerr << RED << "[FAIL] Metadata " << path << ": expected '" << expected << "', got '" << val << "'" << RESET << std::endl;
        free(data);
        throw std::runtime_error("Metadata mismatch");
    }
    
    std::cout << "    [Meta] " << path << " = '" << val << "' [OK]" << std::endl;
    free(data);
}

int main() {
    try {
        std::cout << BOLD << "\n=== Test Metadata Write/Read (Global, Slice, TimeRange) ===\n" << RESET;

        const int time_steps = 10;
        const int spatial_size = 5;

        // =================================================================================
        // PHASE 1: WRITING DATA & METADATA
        // =================================================================================
        {
            std::cout << "\n--- Phase 1: Writing Data & Metadata ---\n";
            if (fs::exists("test_db_metadata_strategies")) {
                fs::remove_all("test_db_metadata_strategies");
            }

            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            // 1.1 GLOBAL WRITE (Static Data)
            {
                std::cout << "  -> Writing Static Data (Global)...\n";
                OperationContext opCtx(&dataEntryCtx, "core_profiles", "", WRITE_OP);
                backend.beginAction(&opCtx);

                int homogeneous_time = 1;
                backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

                // Static 0D: global_quantities/ip
                double ip_val = 1.5e6;
                backend.writeData(&opCtx, "global_quantities/ip", "", &ip_val, alconst::double_data, 0, nullptr);
                write_meta(backend, &opCtx, "global_quantities/ip@units", "A");
                write_meta(backend, &opCtx, "global_quantities/ip@description", "Plasma current");

                // Static 1D: static_1d_signal
                std::vector<double> rho(spatial_size);
                for(int i=0; i<spatial_size; ++i) rho[i] = (double)i / (spatial_size-1);
                int rho_dim = 1; 
                int rho_size[] = {spatial_size};
                backend.writeData(&opCtx, "static_1d_signal", "", rho.data(), alconst::double_data, rho_dim, rho_size);
                write_meta(backend, &opCtx, "static_1d_signal@units", "m");
                write_meta(backend, &opCtx, "static_1d_signal@coordinate1", "space");

                ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
                std::vector<double> te_val(spatial_size);
                double t = 0;
                for(int x=0; x<spatial_size; ++x) te_val[x] = 1000.0 + t*10.0 + x;
                int te_dim = 1; int te_size[] = {spatial_size};
                backend.writeData(&profilesCtx, "t_e", "", te_val.data(), alconst::double_data, te_dim, te_size);
                // Write Metadata for Dynamic Signals (Once)
                write_meta(backend, &opCtx, "profiles_1d/t_e@units", "eV");
                write_meta(backend, &opCtx, "profiles_1d/t_e@description", "Electron Temperature");
                write_meta(backend, &opCtx, "profiles_1d/t_e@type", "dynamic");

                backend.endAction(&opCtx);
            }

            // 1.2 SLICE WRITE (Dynamic Data)
            {
                std::cout << "  -> Writing Dynamic Data (Slices)...\n";

                double time = alconst::undefined_time;
                int interpmode = alconst::undefined_interp;
                OperationContext opCtx(&dataEntryCtx, "core_profiles", WRITE_OP,
                           alconst::slice_op, time, interpmode);

                backend.beginAction(&opCtx);
                
                // Write time array (global)
                std::vector<double> time_values(time_steps);
                for(int i=0; i<time_steps; ++i) time_values[i] = i * 0.1;
                int t_dim = 1; int t_size[] = {time_steps};
                backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, t_dim, t_size);

                ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
                int p_size = time_steps;
                backend.beginArraystructAction(&profilesCtx, &p_size);

                for(int t=1; t<time_steps; ++t) {
                    // t_e (1D dynamic)
                    std::vector<double> te_val(spatial_size);
                    for(int x=0; x<spatial_size; ++x) te_val[x] = 1000.0 + t*10.0 + x;
                    int te_dim = 1; int te_size[] = {spatial_size};
                    backend.writeData(&profilesCtx, "t_e", "", te_val.data(), alconst::double_data, te_dim, te_size);

                    if(t < time_steps - 1) profilesCtx.nextIndex(1);
                }
                backend.endAction(&profilesCtx);
                backend.endAction(&opCtx);
            }

            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Phase 1 Completed.\n" << RESET;
        }

        // =================================================================================
        // PHASE 2: READING (GLOBAL STRATEGY)
        // =================================================================================
        {
            std::cout << "\n--- Phase 2: Reading (Global Strategy) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            OperationContext opCtx(&dataEntryCtx, "core_profiles", "", READ_OP); // Defaults to GLOBAL_OP
            backend.beginAction(&opCtx);

            verify_meta(backend, &opCtx, "global_quantities/ip@units", "A");
            verify_meta(backend, &opCtx, "global_quantities/ip@description", "Plasma current");
            verify_meta(backend, &opCtx, "static_1d_signal@units", "m");
            verify_meta(backend, &opCtx, "static_1d_signal@coordinate1", "space");
            verify_meta(backend, &opCtx, "profiles_1d/t_e@units", "eV");
            verify_meta(backend, &opCtx, "profiles_1d/t_e@description", "Electron Temperature");

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Phase 2 Completed.\n" << RESET;
        }

        // =================================================================================
        // PHASE 3: READING (SLICE STRATEGY)
        // =================================================================================
        {
            std::cout << "\n--- Phase 3: Reading (Slice Strategy) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double target_time = 0.5; // t_index = 5
            int interpmode = alconst::closest_interp;
            OperationContext opCtx(&dataEntryCtx, "core_profiles", READ_OP, alconst::slice_op, target_time, interpmode);
            backend.beginAction(&opCtx);

            // Verify metadata is accessible even in slice mode (as static data)
            verify_meta(backend, &opCtx, "profiles_1d/t_e@units", "eV");
            verify_meta(backend, &opCtx, "profiles_1d/t_e@type", "dynamic");

            // Read Data Slice
            ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
            int p_size = 0;
            backend.beginArraystructAction(&profilesCtx, &p_size); // Slicing -> size 1
            
            void* data = nullptr;
            int type = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];
            backend.readData(&profilesCtx, "t_e", "", &data, &type, &dim, size);
            
            double* vals = (double*)data;
            assert(dim == 1);
            assert(size[0] == spatial_size);
            assert(std::abs(vals[0] - 1050.0) < 1e-9); // t=0.5 -> index 5
            std::cout << "    [Data] t_e slice at 0.5 verified.\n";
            free(data);

            backend.endAction(&profilesCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Phase 3 Completed.\n" << RESET;
        }

        // =================================================================================
        // PHASE 4: READING (TIMERANGE STRATEGY)
        // =================================================================================
        {
            std::cout << "\n--- Phase 4: Reading (TimeRange Strategy) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double tmin = 0.2;
            double tmax = 0.4;
            std::vector<double> dtime; // No resampling
            OperationContext opCtx(&dataEntryCtx, "core_profiles", READ_OP, alconst::timerange_op, tmin, tmax, dtime, alconst::closest_interp);
            backend.beginAction(&opCtx);

            verify_meta(backend, &opCtx, "profiles_1d/t_e@units", "eV");
            
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Phase 4 Completed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    std::cout << BOLD << GREEN << "\n✓ All metadata tests passed!\n" << RESET;
    return 0;
}