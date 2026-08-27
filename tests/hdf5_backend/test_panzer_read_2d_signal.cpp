// test_panzer_read_2d_signal.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>

int main() {
    std::cout << "=== TEST PanzerDB: readDataByIndex (2D signal) ===\n\n";
    const std::string filename = "test_read_2d_signal.panzer";

    // ===================================================================
    // 1. Écriture d'une structure A[3]/B[2] avec un signal 2D dans A[1]
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Écriture de la structure...\n";
        db.beginArray("A", 3);
        for (int i = 0; i < 3; ++i) {
            // On écrit un signal 2D uniquement dans l'élément A[1]
            if (i == 1) {
                std::vector<double> data_to_write(4 * 5); // Tenseur 4x5
                std::iota(data_to_write.begin(), data_to_write.end(), 500.0);
                db.writeData("signal_2d", {4, 5}, data_to_write.data(), data_to_write.size());
            }

            // Chaque élément de A contient un AoS B de taille 2, mais sans données directes.
            db.beginArray("B", 2);
            for (int j = 0; j < 2; ++j) {
                // Intentionnellement vide, on incrémente juste l'index de B
                //db.incrementArrayIndex();
            }
            db.endArray(); // B

            db.incrementArrayIndex(); // Passe à l'élément A suivant
        }
        db.endArray(); // A

        db.close();
        std::cout << "Fichier fermé.\n\n";
    }

    // ===================================================================
    // 2. Réouverture et lecture du signal 2D
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        std::cout << "Vérification des tailles des AoS...\n";
        auto shapeA = db.getAOSShape("A");
        assert(shapeA.size() == 1 && "La shape de A doit contenir un seul élément.");
        assert(shapeA[0] == 3 && "La taille de A doit être 3.");
        std::cout << "Shape de A: { " << shapeA[0] << " } (attendu: { 3 })\n";

        auto shapeB = db.getAOSShape("A/0/B");
        assert(shapeB[0] == 2 && "La taille de B doit être 2.");
        auto shape1B = db.getAOSShape("A/1/B");
        assert(shape1B[0] == 2 && "La taille de B doit être 2.");
        auto shape2B = db.getAOSShape("A/2/B");
        assert(shape2B[0] == 2 && "La taille de B doit être 2.");

        std::cout << "Lecture du signal_2d dans A[1]...\n";

        uint64_t static_indices[] = {1}; // A[1]
        int64_t time_index = -1; // Statique
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        const char* full_path = "A/1/signal_2d";
        int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

        assert(result == 0 && "La lecture du signal 2D doit réussir.");
        assert(ndim_out == 2 && "La dimension du signal doit être 2.");
        assert(shape_out[0] == 4 && "La première dimension doit être 4.");
        assert(shape_out[1] == 5 && "La deuxième dimension doit être 5.");

        std::cout << "Données lues avec succès. Vérification des valeurs...\n";
        for (uint64_t i = 0; i < (shape_out[0] * shape_out[1]); ++i) {
            assert(std::abs(data_out[i] - (500.0 + i)) < 1e-9);
        }
        std::cout << "Toutes les valeurs sont correctes.\n";

        delete[] data_out;
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}