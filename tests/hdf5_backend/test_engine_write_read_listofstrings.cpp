// @file  test_engine_write_read_listofstrings.cpp
// @brief PanzerDB list-of-strings test: writes a static list of strings in A[0] and a
//        dynamic list of strings in A[0]/B[0]/C, then validates both on read.
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
    std::cout << "=== TEST PanzerDB: List of Strings ===\n\n";
    const std::string filename = "test_list_of_strings.panzer";

    // Cleanup before the test
    if (fs::exists(filename)) {
        fs::remove(filename);
    }

    // ===================================================================
    // Phase 1: Writing lists of strings
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        std::cout << "--- Phase 1: Writing lists of strings ---\n";

        // --- Static A/B hierarchy ---
        db.beginArray("A", 1);
        
        // Static list of strings in A
        const char* a_str_list[] = {"Static_1", "Static_2", "Static_3"};
        db.writeData("a_string_list", {3}, &a_str_list[0], 3);

        db.beginArray("B", 1);
        
        // Dynamic AoS C inside B
        db.beginArray("C", "time");
        
        // Dynamic list of strings in C
        const char* c_dyn_str_list[] = {"Dynamic_A", "Dynamic_B"};
        db.writeDataSlices("c_dyn_string_list", {2}, c_dyn_str_list, 1, "time");
        
        db.endArray(); // C
        db.endArray(); // B
        db.endArray(); // A

        db.close();
        std::cout << "Phase 1 complete. File closed.\n\n";
    }

    // ===================================================================
    // Phase 2: Validation of the written data
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        std::cout << "--- Phase 2: Validation of the data ---\n";

        std::vector<PanzerDB::Leaf> leaves = db.getLeaves();
        
        // Validate A[0]/a_string_list
        const PanzerDB::Leaf* leaf_a_str_list = find_leaf(leaves, "A/0/a_string_list");
        assert(leaf_a_str_list != nullptr && "Leaf A/0/a_string_list not found.");
        assert(leaf_a_str_list->count == 3 && "The count of a_string_list must be 3.");
        
        std::vector<std::string> read_str_list(3);
        db.readTensor(*leaf_a_str_list, read_str_list.data());
        
        assert(read_str_list[0] == "Static_1" && "Value [0] of a_string_list incorrect.");
        assert(read_str_list[1] == "Static_2" && "Value [1] of a_string_list incorrect.");
        assert(read_str_list[2] == "Static_3" && "Value [2] of a_string_list incorrect.");
        std::cout << "  [OK] A[0]/a_string_list validated.\n";

        // Validate A[0]/B[0]/C[0]/c_dyn_string_list (time_index 0)
        const PanzerDB::Leaf* leaf_c_dyn_str_list = find_leaf(leaves, "A/0/B/0/C/c_dyn_string_list", 0);
        assert(leaf_c_dyn_str_list != nullptr && "Leaf A/0/B/0/C/c_dyn_string_list not found.");
        assert(leaf_c_dyn_str_list->count == 2 && "The count of c_dyn_string_list must be 2.");

        read_str_list.resize(2);
        db.readTensor(*leaf_c_dyn_str_list, read_str_list.data());

        assert(read_str_list[0] == "Dynamic_A" && "Value [0] of c_dyn_string_list incorrect.");
        assert(read_str_list[1] == "Dynamic_B" && "Value [1] of c_dyn_string_list incorrect.");
        std::cout << "  [OK] A[0]/B[0]/C/c_dyn_string_list validated.\n";

        db.close();
        std::cout << "Phase 2 complete.\n\n";
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}