// tests/hdf5_backend/test_magnetics_repro_slice.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// ANSI Colors
#define RESET ""
#define BOLD ""
#define GREEN ""
#define RED ""
#define CYAN ""
#define MAGENTA ""

const std::string URI = "imas:hdf5?path=./test_db_test_magnetics_slice";

int main() {
    try {
        std::cout << BOLD << MAGENTA << "\n=== Test Magnetics Repro Slice (Full Get + Get Slice) ===\n" << RESET;

        int dynamicsize = 10;
        int staticsize = 3;

        // ==============================================================
        // 1. SETUP: WRITE DATA (Création du Pulse)
        // ==============================================================
        {
            std::cout << CYAN << "[1] Setting up data (Write)...\n" << RESET;
            DataEntryContext ctx(URI);
            HDF5Backend backend;
            backend.openPulse(&ctx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&ctx, "magnetics", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // ids._magnetics.ids_properties.homogeneous_time = 1
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            int dim_1[] = {dynamicsize};

            // ids._magnetics.flux_loop
            ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
            backend.beginArraystructAction(&fluxLoopCtx, &staticsize);

            for (int j = 0; j < staticsize; j++) {
                std::vector<double> flux_data(dynamicsize);
                // Data pattern: j*100 + i. 
                // Note: i corresponds to time 0.1*i. So value approx j*100 + time*10.
                for (int i = 0; i < dynamicsize; i++) flux_data[i] = j * 100.0 + i;
                
                backend.writeData(&fluxLoopCtx, "flux/data", "time", flux_data.data(), alconst::double_data, 1, dim_1);
                if (j < staticsize - 1) fluxLoopCtx.nextIndex(1);
            }
            backend.endAction(&fluxLoopCtx);

            // ids._magnetics.time
            std::vector<double> time_data(dynamicsize);
            for (int i = 0; i < dynamicsize; i++) time_data[i] = 0.1 * i; // 0.0, 0.1, ..., 0.9
            
            backend.writeData(&opCtx, "time", "time", time_data.data(), alconst::double_data, 1, dim_1);

            backend.endAction(&opCtx);
            backend.closePulse(&ctx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "  [OK] Data written.\n" << RESET;
        }

        // ==============================================================
        // 2. SCENARIO: GET FULL (ids._magnetics.get())
        // ==============================================================
        {
            std::cout << CYAN << "\n[2] Getting full magnetics (ids._magnetics.get())...\n" << RESET;
            DataEntryContext ctx(URI);
            HDF5Backend backend;
            backend.openPulse(&ctx, OPEN_PULSE);

            OperationContext opCtx(&ctx, "magnetics", "", READ_OP);
            backend.beginAction(&opCtx);

            // Check time size
            void* data = nullptr;
            int type = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];
            backend.readData(&opCtx, "time", "time", &data, &type, &dim, size);
            
            int slices = (dim == 1) ? size[0] : 0;
            std::cout << "nb of time slices = " << slices << std::endl;
            if (slices != dynamicsize) throw std::runtime_error("Time slices mismatch");
            if (data) free(data);

            // Check flux_loop size
            ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
            int fl = 0;
            backend.beginArraystructAction(&fluxLoopCtx, &fl);
            std::cout << "nb of flux_loop elements = " << fl << std::endl;
            if (fl != staticsize) throw std::runtime_error("Flux loop size mismatch");

            for (int j = 0; j < fl; j++) {
                backend.readData(&fluxLoopCtx, "flux/data", "time", &data, &type, &dim, size);
                int d = (dim == 1) ? size[0] : 0;
                std::cout << "nb of data slices for flux_loop(" << j << ") = " << d << std::endl;
                
                if (d != dynamicsize) throw std::runtime_error("Flux data size mismatch");
                
                // Verification sommaire des données
                double* vals = (double*)data;
                // for (int i=0; i<d; i++) std::cout << "data(" << i << ") = " << vals[i] << std::endl;
                
                if (data) free(data);
                if (j < fl - 1) fluxLoopCtx.nextIndex(1);
            }
            backend.endAction(&fluxLoopCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&ctx, OPEN_PULSE);
        }

        // ==============================================================
        // 3. SCENARIO: GET SLICE (ids._magnetics.getSlice(time, interp))
        // ==============================================================
        {
            double target_time = 0.21;
            std::cout << CYAN << "\n[3] Get single slice of magnetics (t=" << target_time << ")\n" << RESET;
            
            DataEntryContext ctx(URI);
            HDF5Backend backend;
            backend.openPulse(&ctx, OPEN_PULSE);

            int interpmode = alconst::linear_interp; // Equivalent to interp=2 usually in UAL/AL mapping if linear
            OperationContext opCtx(&ctx, "magnetics", READ_OP, alconst::slice_op, target_time, interpmode);
            backend.beginAction(&opCtx);

            // flux_loop(0).flux.data(0)
            ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
            int fl = 0;
            backend.beginArraystructAction(&fluxLoopCtx, &fl);
            
            // On lit flux/data pour le premier élément (j=0)
            void* data = nullptr;
            int type = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];
            
            backend.readData(&fluxLoopCtx, "flux/data", "time", &data, &type, &dim, size);
            
            // En mode slice, la dimension temporelle disparait -> scalaire (dim 0)
            if (dim != 1) throw std::runtime_error("Slice read should return 1 (no dim change expected upon slicing)");
            
            double val = *(double*)data;
            std::cout << "flux_loop(0).flux.data(0) = " << val << std::endl;
            
            // Calcul de la valeur attendue (Interpolation Linéaire)
            // Time array: 0.0, 0.1, 0.2, 0.3 ...
            // Target: 0.21
            // Indices: 2 (0.2) et 3 (0.3)
            // Data pour j=0: 0.0, 1.0, 2.0, 3.0 ... (valeur = index)
            // Val(0.2) = 2.0
            // Val(0.3) = 3.0
            // Interp: 2.0 + (3.0 - 2.0) * (0.21 - 0.20) / (0.30 - 0.20)
            //       = 2.0 + 1.0 * 0.01 / 0.1
            //       = 2.1
            double expected = 2.1;
            
            if (std::abs(val - expected) > 1e-5) {
                std::cerr << RED << "Mismatch! Expected " << expected << ", got " << val << RESET << std::endl;
                if (data) free(data);
                return 1;
            } else {
                std::cout << GREEN << "  [OK] Value matches expected linear interpolation." << RESET << std::endl;
            }
            
            if (data) free(data);

            backend.endAction(&fluxLoopCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&ctx, OPEN_PULSE);
        }
        
        std::cout << BOLD << GREEN << "\n✓ Test passed successfully!\n" << RESET;

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}
