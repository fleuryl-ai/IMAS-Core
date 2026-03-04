// test_panzer_char_data.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
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
    std::cout << "=== TEST PanzerDB: char_data (Strings) ===\n\n";
    const std::string filename = "test_char_data.panzer";

    if (fs::exists(filename)) {
        fs::remove(filename);
    }

    // ===================================================================
    // Phase 1: Écriture
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        std::cout << "--- Phase 1: Écriture de chaînes de caractères ---\n";

        db.beginArray("A", 2);
        for (int a = 0; a < 2; ++a) {
            const char* a_str_val = (a == 0) ? "Hello_A0" : "World_A1";
            db.writeData("a_string", {}, &a_str_val, 1);

            db.beginArray("B", 2);
            for (int b = 0; b < 2; ++b) {
                const char* b_str_val = (b == 0) ? "Nested_B0" : "Nested_B1";
                db.writeData("b_string", {}, &b_str_val, 1);

                // AoS dynamique C dans B
                // Note: Une nouvelle instance de C est créée pour chaque paire (a,b)
                // Le compteur de temps repart à 0 pour chaque nouvelle instance.
                db.beginArray("C", "time");
                const char* c_dyn_str_val = (a == 0 && b == 0) ? "Dynamic_C00" : "Dynamic_C_other";
                db.writeDataSlices("c_dyn_string", {}, &c_dyn_str_val, 1, "");
                db.endArray(); // C
                
                db.incrementArrayIndex(); // Passe à B[1]
            }
            db.endArray(); // B
            db.incrementArrayIndex(); // Passe à A[1]
        }
        db.endArray(); // A

        db.close();
        std::cout << "Phase 1 terminée. Fichier fermé.\n\n";
    }

    // ===================================================================
    // Phase 2: Validation
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        std::cout << "--- Phase 2: Validation des chaînes de caractères ---\n";

        std::vector<PanzerDB::Leaf> leaves = db.getLeaves();
        std::string read_str;
        
        // Valider A[0]/a_string
        const PanzerDB::Leaf* leaf_a0_str = find_leaf(leaves, "A/0/a_string");
        assert(leaf_a0_str != nullptr && "Leaf A/0/a_string non trouvée.");
        db.readTensor(*leaf_a0_str, &read_str);
        assert(read_str == "Hello_A0" && "Valeur A[0]/a_string incorrecte.");
        std::cout << "  [OK] A[0]/a_string = '" << read_str << "'\n";

        // Valider A[1]/a_string
        const PanzerDB::Leaf* leaf_a1_str = find_leaf(leaves, "A/1/a_string");
        assert(leaf_a1_str != nullptr && "Leaf A/1/a_string non trouvée.");
        db.readTensor(*leaf_a1_str, &read_str);
        assert(read_str == "World_A1" && "Valeur A[1]/a_string incorrecte.");
        std::cout << "  [OK] A[1]/a_string = '" << read_str << "'\n";

        // Valider A[0]/B[1]/b_string
        const PanzerDB::Leaf* leaf_a0b1_str = find_leaf(leaves, "A/0/B/1/b_string");
        assert(leaf_a0b1_str != nullptr && "Leaf A/0/B/1/b_string non trouvée.");
        db.readTensor(*leaf_a0b1_str, &read_str);
        assert(read_str == "Nested_B1" && "Valeur A[0]/B[1]/b_string incorrecte.");
        std::cout << "  [OK] A[0]/B[1]/b_string = '" << read_str << "'\n";

        // ---------------- CORRECTION ICI ----------------
        // Chaque instance de C est unique (chemin parent différent), donc son temps commence à 0.

        // Valider A[0]/B[0]/C[0]/c_dyn_string (time_index 0)
        const PanzerDB::Leaf* leaf_a0b0c0_dyn_str = find_leaf(leaves, "A/0/B/0/C/c_dyn_string", 0);
        assert(leaf_a0b0c0_dyn_str != nullptr && "Leaf A/0/B/0/C/c_dyn_string non trouvée.");
        db.readTensor(*leaf_a0b0c0_dyn_str, &read_str);
        assert(read_str == "Dynamic_C00" && "Valeur A[0]/B[0]/C[0]/c_dyn_string incorrecte.");
        std::cout << "  [OK] A[0]/B[0]/C[0]/c_dyn_string = '" << read_str << "'\n";

        // Valider A[1]/B[1]/C[0]/c_dyn_string (time_index doit être 0 aussi !)
        // C'est la première tranche temporelle pour CETTE instance (A/1/B/1/C).
        const PanzerDB::Leaf* leaf_a1b1c0_dyn_str = find_leaf(leaves, "A/1/B/1/C/c_dyn_string", 0); // <-- CORRECTION : 0 au lieu de 1
        assert(leaf_a1b1c0_dyn_str != nullptr && "Leaf A/1/B/1/C/c_dyn_string non trouvée.");
        db.readTensor(*leaf_a1b1c0_dyn_str, &read_str);
        assert(read_str == "Dynamic_C_other" && "Valeur A[1]/B[1]/C[0]/c_dyn_string incorrecte.");
        std::cout << "  [OK] A[1]/B[1]/C[0]/c_dyn_string = '" << read_str << "'\n";

        db.close();
        std::cout << "Phase 2 terminée.\n\n";
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}