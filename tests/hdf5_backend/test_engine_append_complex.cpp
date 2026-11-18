// @file  test_engine_append_complex.cpp
// @brief PanzerDB complex static/dynamic structure test: writes A/B/C and E/F/G (with
//        dynamic AoS G), then appends new time slices in APPEND mode and validates alignment.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    std::cout << "=== TEST PanzerDB: Complex Static/Dynamic Structures with APPEND ===\n\n";
    const std::string filename = "test_complex_append.panzer";

    if (fs::exists(filename)) {
        fs::remove(filename);
    }

    // ===================================================================
    // Phase 1: Initial write
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        std::cout << "--- Phase 1: Initial write ---\n";

        // ... (Write code unchanged) ...
        std::cout << "Writing the A/B/C hierarchy...\n";
        db.beginArray("A", 2);
        for (int a = 0; a < 2; ++a) {
            db.writeData("a_scalar", {}, new double(20.0 + a), 1);
            db.writeData("a_sig", {1}, new double(10.0 + a), 1);
            db.beginArray("B", 3);
            for (int b = 0; b < 3; ++b) {
                db.writeData("b_sig", {1}, new double(100.0 + a * 10 + b), 1);
                db.beginArray("C", 1);
                db.writeData("c_scalar", {}, new double(2000.0 + a * 100 + b * 10), 1);
                db.writeData("c_sig", {1}, new double(1000.0 + a * 100 + b * 10), 1);
                db.endArray();
                db.incrementArrayIndex();
            }
            db.endArray();
            db.incrementArrayIndex();
        }
        db.endArray();

        std::cout << "Writing the E/F/G hierarchy...\n";
        db.beginArray("E", 2);
        for (int e = 0; e < 2; ++e) {
            db.writeData("e_scalar", {}, new double(70.0 + e), 1);
            db.writeData("e_sig", {1}, new double(50.0 + e), 1);
            db.beginArray("F", 1);
            db.writeData("f_scalar", {}, new double(700.0 + e), 1);
            db.writeData("f_sig", {1}, new double(500.0 + e), 1);

            // Standalone dynamic
            /*std::vector<double> f_sig_dynamique_slice(3*2);
            for(size_t i=0; i<f_sig_dynamique_slice.size(); ++i) f_sig_dynamique_slice[i] = 2500.0 + e*1000 + i;
            db.writeDataSlices("f_sig_dynamique", {3, 3}, f_sig_dynamique_slice.data(), 1, "time");*/

            std::vector<double> f_sig_dynamique_slice(3 * 3); 
            for(size_t i=0; i<f_sig_dynamique_slice.size(); ++i) {
                f_sig_dynamique_slice[i] = 2500.0 + e*1000 + i;
            }
            db.writeDataSlices("f_sig_dynamique", {3, 3}, f_sig_dynamique_slice.data(), 1, "time");

            db.beginArray("G", "time"); // G is dynamic
            
            // Dynamic scalar
            double g_scalar_val = 7000.0 + e*1000;
            db.writeDataSlices("g_scalar_dynamic", {}, &g_scalar_val, 1, "time");
            
            // Current
            /*std::vector<double> current_slice(2*2);
            for(size_t i=0; i<current_slice.size(); ++i) current_slice[i] = 9000.0 + e*1000 + i;
            db.writeDataSlices("current", {2,2}, current_slice.data(), 1, "time");*/
            std::vector<double> current_slice(2*2);
            for(size_t i=0; i<current_slice.size(); ++i) current_slice[i] = 9000.0 + e*1000 + i;

            // CORRECTION HERE: Pass {1, 2, 2} instead of {2, 2}
            // This means: "1 time dimension, then spatially 2x2"
            db.writeDataSlices("current", {2, 2}, current_slice.data(), 1, "time");

            db.endArray(); // G
            db.endArray(); // F
            db.incrementArrayIndex();
        }
        db.endArray(); // E

        db.close();
        std::cout << "Phase 1 complete. File closed.\n\n";
    }

    // ===================================================================
    // Phase 2: Validation
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        std::cout << "--- Phase 2: Validation of the initial write ---\n";

        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;

        // Validate A/B/C
        uint64_t abc_indices[] = {1, 2, 0}; 
        const char* full_path = "A/1/B/2/C/0/c_sig";
        int result = db.readDataByIndex(full_path, -1, &ndim_out, shape_out, &data_out);
        assert(result == 0);
        std::cout << "  [OK] A[1]/B[2]/C[0]/c_sig = " << data_out[0] << "\n";
        delete[] data_out;

        // Validate f_sig_dynamique (TI=0 since it is the 1st instance)
        uint64_t f_sig_dynamique_indices[] = {1, 0}; 
        /*result = db.readDataByIndex("E", f_sig_dynamique_indices, 2, "f_sig_dynamique", 0, &ndim_out, shape_out, &data_out);
        assert(result == 0);
        assert(std::abs(data_out[4] - 3504.0) < 1e-9);
        std::cout << "  [OK] E[1]/F[0]/f_sig_dynamique[1][1] = " << data_out[4] << "\n";
        delete[] data_out;*/
        const char* full_path_f_sig = "E/1/F/0/f_sig_dynamique";
        result = db.readDataByIndex(full_path_f_sig, 0, &ndim_out, shape_out, &data_out);
        assert(result == 0);

        // Verification
        // With the {1, 3, 3} fix, we get a [3, 3] slice.
        // data_out[4] is now valid (index 4 out of 9).
        assert(std::abs(data_out[4] - 3504.0) < 1e-9); 

        std::cout << "  [OK] E[1]/F[0]/f_sig_dynamique[1][1] = " << data_out[4] << "\n";
        delete[] data_out;

        // Validate E/F/G/current
        // MAJOR CORRECTION HERE:
        // 1. Indices: only {1, 0} (E[1], F[0]). G is dynamic, no static index.
        // 2. TI: 0 (With the alignment fix, current and g_scalar_dynamic are both at TI=0)
        
        uint64_t efg_indices[] = {1, 0}; // <-- CORRECTION: Size 2, not 3
        const char* full_path_current = "E/1/F/0/G/0/current";
        result = db.readDataByIndex(full_path_current, 0, &ndim_out, shape_out, &data_out);

        assert(result == 0 && "Reading E[1]/F[0]/G/current");
        assert(ndim_out == 2 && shape_out[0] == 2 && shape_out[1] == 2);
        assert(std::abs(data_out[3] - (9000.0 + 1*1000 + 3)) < 1e-9);
        std::cout << "  [OK] E[1]/F[0]/G/current[1][1] = " << data_out[3] << "\n";
        delete[] data_out;

        db.close();
        std::cout << "Phase 2 complete.\n\n";
    }

    // ===================================================================
    // Phase 3: APPEND
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::APPEND, true);
        std::cout << "--- Phase 3: Appending data in APPEND mode ---\n";

        db.beginArray("E", 2);
        db.setCurrentArrayIndex(1); // Goes to E[1]
        db.beginArray("F", 1);
        db.beginArray("G", "time");

        // Append 10 slices
        std::vector<double> current_slices(10 * 2 * 2);
        for(size_t i=0; i<current_slices.size(); ++i) current_slices[i] = 5000.0 + i;
        db.writeDataSlices("current", {2, 2}, current_slices.data(), 10, "time");
        
        db.endArray(); // G
        db.endArray(); // F
        db.endArray(); // E

        db.close();
        std::cout << "Phase 3 complete.\n\n";
    }

    // ===================================================================
    // Phase 4: Validation finale
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        std::cout << "--- Phase 4: Validation finale ---\n";
        
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;

        // Validate the new slice
        // Indices: {1, 0} (E, F)
        // TI: Start(1) + Offset(4) = 5 (Since Phase 1 ended at TI=0, append starts at TI=1)
        uint64_t efg_indices_append[] = {1, 0}; // <-- CORRECTION: Size 2
        const char* full_path_append = "E/1/F/0/G/0/current";
        int result = db.readDataByIndex(full_path_append, 5, &ndim_out, shape_out, &data_out);

        assert(result == 0);
        assert(std::abs(data_out[2] - 5018.0) < 1e-9); // 5018 is in the slice at offset 4 (index 4 of the batch)
        std::cout << "  [OK] New slice E[1]/F[0]/G/current (TI=5) = " << data_out[2] << "\n";
        delete[] data_out;

        db.close();
        std::cout << "Phase 4 complete.\n\n";
    }
    
    // ... The rest (Phase 5 and 6) follows the same logic:
    // Never put a static index for G in the index arrays. ...

    // ===================================================================
    // Phase 5: Append 3 slices everywhere
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::APPEND, true);
        std::cout << "--- Phase 5: Appending 3 slices ---\n";

        // E[0]/F[0]
        db.beginArray("E", 2);
        db.setCurrentArrayIndex(0);
        db.beginArray("F", 1);
        std::vector<double> f_sig_dyn_new_slices(3 * 3 * 3);
        for(size_t i=0; i<f_sig_dyn_new_slices.size(); ++i) f_sig_dyn_new_slices[i] = 4000.0 + i;
        db.writeDataSlices("f_sig_dynamique", {3, 3}, f_sig_dyn_new_slices.data(), 3, "time");
        db.endArray();
        db.endArray();

        // E[1]/F[0]/G
        db.beginArray("E", 2);
        db.setCurrentArrayIndex(1);
        db.beginArray("F", 1);
        db.beginArray("G", "time");
        std::vector<double> current_new_slices(3 * 2 * 2);
        for(size_t i=0; i<current_new_slices.size(); ++i) current_new_slices[i] = 8000.0 + i;
        db.writeDataSlices("current", {2, 2}, current_new_slices.data(), 3, "time");
        db.endArray();
        db.endArray();
        db.endArray();

        db.close();
        std::cout << "Phase 5 complete.\n\n";
    }

    // ===================================================================
    // Phase 6
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        std::cout << "--- Phase 6: Validation finale ---\n";
        
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;

        // Re-valider Current Original (TI=0)
        uint64_t efg_indices[] = {1, 0}; // E, F
        const char* full_path = "E/1/F/0/G/current";
        int result = db.readDataByIndex(full_path, 0, &ndim_out, shape_out, &data_out);
        assert(result == 0);
        assert(std::abs(data_out[3] - 10003.0) < 1e-9);
        std::cout << "  [OK] Re-valid Original Current\n";
        delete[] data_out;

        // Re-valider Current Phase 3 (TI=5)
        const char* full_path_append = "E/1/F/0/G/current";
        result = db.readDataByIndex(full_path_append, 5, &ndim_out, shape_out, &data_out);
        assert(result == 0);
        assert(std::abs(data_out[2] - 5018.0) < 1e-9);
        std::cout << "  [OK] Re-valid Append 1 Current\n";
        delete[] data_out;

        // Validate f_sig_dynamique Phase 5
        // E[0]/F[0], TI start = 1. Offset = 1. Target TI = 2.
        uint64_t e0f0_indices[] = {0, 0};
        const char* full_path_f_sig = "E/0/F/0/f_sig_dynamique";
        result = db.readDataByIndex(full_path_f_sig, 2, &ndim_out, shape_out, &data_out);
        assert(result == 0);
        assert(std::abs(data_out[1] - 4010.0) < 1e-9);
        std::cout << "  [OK] New f_sig_dynamique slice\n";
        delete[] data_out;

        // Validate g_scalar_dynamic (TI=0)
        // E[0]/F[0], G is implicit since it is dynamic
        const char* full_path_g_scalar = "E/0/F/0/G/0/g_scalar_dynamic";
        result = db.readDataByIndex(full_path_g_scalar, 0, &ndim_out, shape_out, &data_out);
        assert(result == 0);
        assert(std::abs(data_out[0] - 7000.0) < 1e-9);
        std::cout << "  [OK] g_scalar_dynamic\n";
        delete[] data_out;
        
        db.close();
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}