// @file  test_engine_write_read_string_complex.cpp
// @brief PanzerDB strings + complex test: writes static strings and a dynamic complex
//        scalar through an A/B/C hierarchy (dynamic AoS C), then validates them on read.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <complex>
#include <filesystem>

namespace fs = std::filesystem;

#include "fixtures/panzerdb_helper.h"

int main() {
    std::cout << "=== TEST PanzerDB: Strings and Complex Data ===\n\n";
    const std::string filename = "test_string_complex.panzer";

    // Cleanup before the test
    if (fs::exists(filename)) {
        fs::remove(filename);
    }

    // ===================================================================
    // Phase 1: Writing
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        std::cout << "--- Phase 1: Writing strings and complexes ---\n";

        // --- Static A/B hierarchy ---
        db.beginArray("A", 2);
        for (int a = 0; a < 2; ++a) {
            // String scalar in A
            const char* a_str_val = (a == 0) ? "String_in_A0" : "String_in_A1";
            db.writeData("a_string", {}, &a_str_val, 1);

            db.beginArray("B", 1);
            
            // String scalar in B
            const char* b_str_val = "String_in_B";
            db.writeData("b_string", {}, &b_str_val, 1);

            // Dynamic AoS C inside B
            db.beginArray("C", "time");
            // Dynamic complex scalar in C
            std::complex<double> c_dyn_complex_val(1.1 + a, 2.2 + a);
            db.writeDataSlices("c_dyn_complex", {}, &c_dyn_complex_val, 1, "");
            db.endArray(); // C
            
            db.incrementArrayIndex(); // B (useless here since size=1, but good practice)
            db.endArray(); // B
            db.incrementArrayIndex(); // A
        }
        db.endArray(); // A

        db.close();
        std::cout << "Phase 1 complete. File closed.\n\n";
    }

    // ===================================================================
    // Phase 2: Validation
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        std::cout << "--- Phase 2: Validation of the data ---\n";

        std::vector<PanzerDB::Leaf> leaves = db.getLeaves();
        
        // Validate A[1]/a_string
        const PanzerDB::Leaf* leaf_a1_str = find_leaf(leaves, "A/1/a_string");
        assert(leaf_a1_str != nullptr && "Leaf A/1/a_string not found.");
        std::string read_str;
        db.readTensor(*leaf_a1_str, &read_str);
        assert(read_str == "String_in_A1" && "A[1]/a_string value incorrect.");
        std::cout << "  [OK] A[1]/a_string = '" << read_str << "'\n";

        // Validate A[0]/B[0]/b_string
        const PanzerDB::Leaf* leaf_a0b0_str = find_leaf(leaves, "A/0/B/0/b_string");
        assert(leaf_a0b0_str != nullptr && "Leaf A/0/B/0/b_string not found.");
        db.readTensor(*leaf_a0b0_str, &read_str);
        assert(read_str == "String_in_B" && "A[0]/B/0/b_string value incorrect.");
        std::cout << "  [OK] A[0]/B/0/b_string = '" << read_str << "'\n";

        // ---------------- CORRECTION HERE ----------------
        // Validate A[1]/B[0]/C[0]/c_dyn_complex
        // This is a NEW C instance (parent is A/1/B/0 instead of A/0/B/0).
        // The Time Index starts again at 0.
        
        const PanzerDB::Leaf* leaf_a1b0c0_dyn_complex = find_leaf(leaves, "A/1/B/0/C/c_dyn_complex", 0); // <-- CORRECTION: 0 instead of 1
        assert(leaf_a1b0c0_dyn_complex != nullptr && "Leaf A/1/B/0/C/c_dyn_complex not found.");
        
        std::complex<double> read_complex;
        db.readTensor(*leaf_a1b0c0_dyn_complex, &read_complex);
        std::complex<double> expected_complex(1.1 + 1, 2.2 + 1); // a=1
        
        assert(std::abs(read_complex.real() - expected_complex.real()) < 1e-9 && "Real part incorrect.");
        assert(std::abs(read_complex.imag() - expected_complex.imag()) < 1e-9 && "Imaginary part incorrect.");
        
        std::cout << "  [OK] A[1]/B[0]/C[0]/c_dyn_complex = (" << read_complex.real() << ", " << read_complex.imag() << "i)"
                  << " (expected: (" << expected_complex.real() << ", " << expected_complex.imag() << "i))\n";

        db.close();
        std::cout << "Phase 2 complete.\n\n";
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}