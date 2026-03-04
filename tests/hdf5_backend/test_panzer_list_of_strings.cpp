// test_panzer_list_of_strings.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <complex>
#include <filesystem>

namespace fs = std::filesystem;

// Helper pour trouver une Leaf par son chemin et son time_index
const PanzerDB::Leaf* find_leaf(const std::vector<PanzerDB::Leaf>& leaves, const std::string& path, int64_t time_index = -1) {
   for (const auto& leaf : leaves) {
       if (leaf.path == path && (time_index == -1 || leaf.time_index == static_cast<uint64_t>(time_index))) {
           return &leaf;
       }
   }
   return nullptr;
}

int main() {
    std::cout << "=== TEST PanzerDB: List of Strings ===\n\n";
    const std::string filename = "test_list_of_strings.panzer";

    // Nettoyage avant le test
    if (fs::exists(filename)) {
        fs::remove(filename);
    }

    // ===================================================================
    // Phase 1: Écriture de listes de chaînes
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        std::cout << "--- Phase 1: Écriture de listes de chaînes ---\n";

        // --- Hiérarchie A/B (statique) ---
        db.beginArray("A", 1);
        
        // Liste de strings statique dans A
        const char* a_str_list[] = {"Static_1", "Static_2", "Static_3"};
        db.writeData("a_string_list", {3}, &a_str_list[0], 3);

        db.beginArray("B", 1);
        
        // AoS dynamique C dans B
        db.beginArray("C", "time");
        
        // Liste de strings dynamique dans C
        const char* c_dyn_str_list[] = {"Dynamic_A", "Dynamic_B"};
        db.writeDataSlices("c_dyn_string_list", {2}, c_dyn_str_list, 1, "time");
        
        db.endArray(); // C
        db.endArray(); // B
        db.endArray(); // A

        db.close();
        std::cout << "Phase 1 terminée. Fichier fermé.\n\n";
    }

    // ===================================================================
    // Phase 2: Validation des données écrites
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        std::cout << "--- Phase 2: Validation des données ---\n";

        std::vector<PanzerDB::Leaf> leaves = db.getLeaves();
        
        // Valider A[0]/a_string_list
        const PanzerDB::Leaf* leaf_a_str_list = find_leaf(leaves, "A/0/a_string_list");
        assert(leaf_a_str_list != nullptr && "Leaf A/0/a_string_list non trouvée.");
        assert(leaf_a_str_list->count == 3 && "Le count de a_string_list doit être 3.");
        
        std::vector<std::string> read_str_list(3);
        db.readTensor(*leaf_a_str_list, read_str_list.data());
        
        assert(read_str_list[0] == "Static_1" && "Valeur [0] de a_string_list incorrecte.");
        assert(read_str_list[1] == "Static_2" && "Valeur [1] de a_string_list incorrecte.");
        assert(read_str_list[2] == "Static_3" && "Valeur [2] de a_string_list incorrecte.");
        std::cout << "  [OK] A[0]/a_string_list validée.\n";

        // Valider A[0]/B[0]/C[0]/c_dyn_string_list (time_index 0)
        const PanzerDB::Leaf* leaf_c_dyn_str_list = find_leaf(leaves, "A/0/B/0/C/c_dyn_string_list", 0);
        assert(leaf_c_dyn_str_list != nullptr && "Leaf A/0/B/0/C/c_dyn_string_list non trouvée.");
        assert(leaf_c_dyn_str_list->count == 2 && "Le count de c_dyn_string_list doit être 2.");

        read_str_list.resize(2);
        db.readTensor(*leaf_c_dyn_str_list, read_str_list.data());

        assert(read_str_list[0] == "Dynamic_A" && "Valeur [0] de c_dyn_string_list incorrecte.");
        assert(read_str_list[1] == "Dynamic_B" && "Valeur [1] de c_dyn_string_list incorrecte.");
        std::cout << "  [OK] A[0]/B/0/C/c_dyn_string_list validée.\n";

        db.close();
        std::cout << "Phase 2 terminée.\n\n";
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}