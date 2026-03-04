// test_panzer_string_complex.cpp
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
    std::cout << "=== TEST PanzerDB: Strings and Complex Data ===\n\n";
    const std::string filename = "test_string_complex.panzer";

    // Nettoyage avant le test
    if (fs::exists(filename)) {
        fs::remove(filename);
    }

    // ===================================================================
    // Phase 1: Écriture
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        std::cout << "--- Phase 1: Écriture de chaînes et de complexes ---\n";

        // --- Hiérarchie A/B (statique) ---
        db.beginArray("A", 2);
        for (int a = 0; a < 2; ++a) {
            // Scalaire string dans A
            const char* a_str_val = (a == 0) ? "String_in_A0" : "String_in_A1";
            db.writeData("a_string", {}, &a_str_val, 1);

            db.beginArray("B", 1);
            
            // Scalaire string dans B
            const char* b_str_val = "String_in_B";
            db.writeData("b_string", {}, &b_str_val, 1);

            // AoS dynamique C dans B
            db.beginArray("C", "time");
            // Scalaire complex dynamique dans C
            std::complex<double> c_dyn_complex_val(1.1 + a, 2.2 + a);
            db.writeDataSlices("c_dyn_complex", {}, &c_dyn_complex_val, 1, "");
            db.endArray(); // C
            
            db.incrementArrayIndex(); // B (inutile ici car size=1, mais bonne pratique)
            db.endArray(); // B
            db.incrementArrayIndex(); // A
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
        std::cout << "--- Phase 2: Validation des données ---\n";

        std::vector<PanzerDB::Leaf> leaves = db.getLeaves();
        
        // Valider A[1]/a_string
        const PanzerDB::Leaf* leaf_a1_str = find_leaf(leaves, "A/1/a_string");
        assert(leaf_a1_str != nullptr && "Leaf A/1/a_string non trouvée.");
        std::string read_str;
        db.readTensor(*leaf_a1_str, &read_str);
        assert(read_str == "String_in_A1" && "Valeur A[1]/a_string incorrecte.");
        std::cout << "  [OK] A[1]/a_string = '" << read_str << "'\n";

        // Valider A[0]/B[0]/b_string
        const PanzerDB::Leaf* leaf_a0b0_str = find_leaf(leaves, "A/0/B/0/b_string");
        assert(leaf_a0b0_str != nullptr && "Leaf A/0/B/0/b_string non trouvée.");
        db.readTensor(*leaf_a0b0_str, &read_str);
        assert(read_str == "String_in_B" && "Valeur A/0/B/0/b_string incorrecte.");
        std::cout << "  [OK] A[0]/B/0/b_string = '" << read_str << "'\n";

        // ---------------- CORRECTION ICI ----------------
        // Valider A[1]/B[0]/C[0]/c_dyn_complex
        // C'est une NOUVELLE instance de C (car parent est A/1/B/0 au lieu de A/0/B/0).
        // Le Time Index recommence à 0.
        
        const PanzerDB::Leaf* leaf_a1b0c0_dyn_complex = find_leaf(leaves, "A/1/B/0/C/c_dyn_complex", 0); // <-- CORRECTION: 0 au lieu de 1
        assert(leaf_a1b0c0_dyn_complex != nullptr && "Leaf A/1/B/0/C/c_dyn_complex non trouvée.");
        
        std::complex<double> read_complex;
        db.readTensor(*leaf_a1b0c0_dyn_complex, &read_complex);
        std::complex<double> expected_complex(1.1 + 1, 2.2 + 1); // a=1
        
        assert(std::abs(read_complex.real() - expected_complex.real()) < 1e-9 && "Partie réelle incorrecte.");
        assert(std::abs(read_complex.imag() - expected_complex.imag()) < 1e-9 && "Partie imaginaire incorrecte.");
        
        std::cout << "  [OK] A[1]/B[0]/C[0]/c_dyn_complex = (" << read_complex.real() << ", " << read_complex.imag() << "i)"
                  << " (attendu: (" << expected_complex.real() << ", " << expected_complex.imag() << "i))\n";

        db.close();
        std::cout << "Phase 2 terminée.\n\n";
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}