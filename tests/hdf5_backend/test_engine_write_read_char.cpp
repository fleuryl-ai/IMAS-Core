// @file  test_engine_write_read_char.cpp
// @brief PanzerDB char_data (string) test: writes static and dynamic strings through
//        an A/B hierarchy with a dynamic AoS C, then validates all of them on read.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

#include "fixtures/panzerdb_helper.h"

int main() {
    std::cout << "=== TEST PanzerDB: char_data (Strings) ===\n\n";
    const std::string filename = "test_char_data.panzer";

    if (fs::exists(filename)) {
        fs::remove(filename);
    }

    // ===================================================================
    // Phase 1: Writing
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        std::cout << "--- Phase 1: Writing strings ---\n";

        db.beginArray("A", 2);
        for (int a = 0; a < 2; ++a) {
            const char* a_str_val = (a == 0) ? "Hello_A0" : "World_A1";
            db.writeData("a_string", {}, &a_str_val, 1);

            db.beginArray("B", 2);
            for (int b = 0; b < 2; ++b) {
                const char* b_str_val = (b == 0) ? "Nested_B0" : "Nested_B1";
                db.writeData("b_string", {}, &b_str_val, 1);

                // Dynamic AoS C inside B
                // Note: A new C instance is created for each (a,b) pair
                // The time counter restarts at 0 for each new instance.
                db.beginArray("C", "time");
                const char* c_dyn_str_val = (a == 0 && b == 0) ? "Dynamic_C00" : "Dynamic_C_other";
                db.writeDataSlices("c_dyn_string", {}, &c_dyn_str_val, 1, "");
                db.endArray(); // C
                
                db.incrementArrayIndex(); // Moves to B[1]
            }
            db.endArray(); // B
            db.incrementArrayIndex(); // Moves to A[1]
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
        std::cout << "--- Phase 2: Validation of the strings ---\n";

        std::vector<PanzerDB::Leaf> leaves = db.getLeaves();
        std::string read_str;
        
        // Validate A[0]/a_string
        const PanzerDB::Leaf* leaf_a0_str = find_leaf(leaves, "A/0/a_string");
        assert(leaf_a0_str != nullptr && "Leaf A/0/a_string not found.");
        db.readTensor(*leaf_a0_str, &read_str);
        assert(read_str == "Hello_A0" && "A[0]/a_string value incorrect.");
        std::cout << "  [OK] A[0]/a_string = '" << read_str << "'\n";

        // Validate A[1]/a_string
        const PanzerDB::Leaf* leaf_a1_str = find_leaf(leaves, "A/1/a_string");
        assert(leaf_a1_str != nullptr && "Leaf A/1/a_string not found.");
        db.readTensor(*leaf_a1_str, &read_str);
        assert(read_str == "World_A1" && "A[1]/a_string value incorrect.");
        std::cout << "  [OK] A[1]/a_string = '" << read_str << "'\n";

        // Validate A[0]/B[1]/b_string
        const PanzerDB::Leaf* leaf_a0b1_str = find_leaf(leaves, "A/0/B/1/b_string");
        assert(leaf_a0b1_str != nullptr && "Leaf A/0/B/1/b_string not found.");
        db.readTensor(*leaf_a0b1_str, &read_str);
        assert(read_str == "Nested_B1" && "A[0]/B[1]/b_string value incorrect.");
        std::cout << "  [OK] A[0]/B[1]/b_string = '" << read_str << "'\n";

        // ---------------- CORRECTION HERE ----------------
        // Each C instance is unique (different parent path), so its time starts at 0.

        // Validate A[0]/B[0]/C[0]/c_dyn_string (time_index 0)
        const PanzerDB::Leaf* leaf_a0b0c0_dyn_str = find_leaf(leaves, "A/0/B/0/C/c_dyn_string", 0);
        assert(leaf_a0b0c0_dyn_str != nullptr && "Leaf A/0/B/0/C/c_dyn_string not found.");
        db.readTensor(*leaf_a0b0c0_dyn_str, &read_str);
        assert(read_str == "Dynamic_C00" && "A[0]/B[0]/C[0]/c_dyn_string value incorrect.");
        std::cout << "  [OK] A[0]/B[0]/C[0]/c_dyn_string = '" << read_str << "'\n";

        // Validate A[1]/B[1]/C[0]/c_dyn_string (time_index must be 0 as well!)
        // This is the first time slice for THIS instance (A/1/B[1]/C).
        const PanzerDB::Leaf* leaf_a1b1c0_dyn_str = find_leaf(leaves, "A/1/B/1/C/c_dyn_string", 0); // <-- CORRECTION: 0 instead of 1
        assert(leaf_a1b1c0_dyn_str != nullptr && "Leaf A/1/B[1]/C/c_dyn_string not found.");
        db.readTensor(*leaf_a1b1c0_dyn_str, &read_str);
        assert(read_str == "Dynamic_C_other" && "A[1]/B[1]/C[0]/c_dyn_string value incorrect.");
        std::cout << "  [OK] A[1]/B[1]/C[0]/c_dyn_string = '" << read_str << "'\n";

        db.close();
        std::cout << "Phase 2 complete.\n\n";
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}