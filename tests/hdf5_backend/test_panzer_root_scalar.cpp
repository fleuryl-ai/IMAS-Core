// test_panzer_root_scalar.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: Root Scalar Write/Read ===\n\n";
    const std::string filename = "test_root_scalar.panzer";

    // ===================================================================
    // 1. Écriture d'un scalaire à la racine
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Écriture d'un scalaire 'temperature' à la racine...\n";
        double temp_value = 98.6;
        // On appelle writeData directement, sans beginArray
        db.writeData("temperature", {}, &temp_value, 1);

        std::cout << "Écriture d'un second scalaire 'pression' à la racine...\n";
        double pressure_value = 1013.25;
        db.writeData("pression", {}, &pressure_value, 1);

        db.close();
        std::cout << "Fichier fermé.\n\n";
    }

    // ===================================================================
    // 2. Réouverture et lecture du scalaire
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        /*std::cout << "Lecture du scalaire 'temperature'...\n";
        auto leaves = db.getLeaves();

        assert(leaves.size() == 2 && "Il devrait y avoir exactement deux feuilles.");

        const PanzerDB::Leaf* temperature_leaf = nullptr;
        for (const auto& leaf : leaves) {
            if (leaf.path == "temperature") {
                temperature_leaf = &leaf;
                break;
            }
        }

        assert(temperature_leaf != nullptr && "La feuille 'temperature' doit être trouvée.");

        assert(temperature_leaf->parent_path == "" && "Le parent du scalaire à la racine doit être vide.");
        assert(temperature_leaf->shape.empty() && "La shape d'un scalaire doit être vide.");
        assert(temperature_leaf->count == 1 && "Le count d'un scalaire doit être 1.");

        std::vector<double> data = db.readTensor(*temperature_leaf);
        assert(data.size() == 1 && "Le tenseur lu doit contenir un seul élément.");
        assert(std::abs(data[0] - 98.6) < 1e-9 && "La valeur du scalaire doit être 98.6.");

        std::cout << "Scalaire lu avec succès: " << data[0] << " (attendu: 98.6)\n";*/

        std::cout << "Lecture du scalaire 'temperature'...\n";
        int status = -1;
        double temp = db.readScalar<double>("temperature", &status);
        assert(status == 0);
        assert(std::abs(temp - 98.6) < 1e-9);
        std::cout << "Scalaire 'temperature' lu avec succès: " << temp << "\n";

        std::cout << "Lecture du scalaire 'pression'...\n";
        double pressure = db.readScalar<double>("pression", &status);
        assert(status == 0);
        assert(std::abs(pressure - 1013.25) < 1e-9);
        std::cout << "Scalaire 'pression' lu avec succès: " << pressure << "\n";
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}