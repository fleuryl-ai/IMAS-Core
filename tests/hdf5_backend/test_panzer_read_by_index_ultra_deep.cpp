// test_panzer_read_by_index_ultra_deep.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>

int main() {
    std::cout << "=== TEST PanzerDB: pz_readData_by_index (ultra deep & complex) ===\n\n";
    const std::string filename = "test_read_by_index_ultra_deep.panzer";

    // ===================================================================
    // 1. Écriture d'une structure complexe
    //    A (size=1)
    //    - A[0]/B (size=2)
    //      - A[0]/B[0]/C (size=1, mais vide)
    //      - A[0]/B[1]/C (size=2)
    //          - A[0]/B[1]/C[0]/D (size=1)
    //              - A[0]/B[1]/C[0]/D[0]/E (size=1)
    //                  - tensor_1 (shape 5)
    //                  - tensor_2 (shape 12)
    //                  ... (10 tenseurs au total)
    //          - A[0]/B[1]/C[1]/D (size=1)
    //              - A[0]/B[1]/C[1]/D[0]/E (size=1)
    //                  - tensor_11 (shape 20)
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Écriture de la structure ultra complexe...\n";
        db.beginArray("A", 1);
        db.beginArray("B", 2); // B a une taille de 2

        // --- Itération sur B ---
        // A[0]/B[0]
        db.beginArray("C", 1);
        db.endArray(); // C
        db.incrementArrayIndex(); // Passe à B[1]

        // A[0]/B[1]
        db.beginArray("C", 2);
        // --- Itération sur C ---
        // A[0]/B[1]/C[0]
        db.beginArray("D", 1);
        db.beginArray("E", 1);
        for (int i = 1; i <= 10; ++i) {
            std::vector<double> data(i);
            std::iota(data.begin(), data.end(), 100.0 + i);
            db.writeData("tensor_" + std::to_string(i), {(size_t)i}, data.data(), data.size());
            // Pas de incrementArrayIndex ici car on écrit plusieurs signaux dans la même instance E[0]
        }
        db.endArray(); // E
        db.endArray(); // D
        db.incrementArrayIndex(); // Passe à C[1]

        // A[0]/B[1]/C[1]
        db.beginArray("D", 1);
        db.beginArray("E", 1);
        std::vector<double> data_11(20);
        std::iota(data_11.begin(), data_11.end(), 200.0);
        db.writeData("tensor_11", {20}, data_11.data(), data_11.size());
        db.endArray(); // E
        db.endArray(); // D
        db.incrementArrayIndex(); // Incrémente C (même si on est à la fin)

        db.endArray(); // C
        db.incrementArrayIndex(); // Incrémente B (même si on est à la fin)

        db.endArray(); // B
        db.incrementArrayIndex(); // Incrémente A (même si on est à la fin)

        db.endArray(); // A

        db.close();
        std::cout << "Fichier fermé.\n\n";
    }

    // ===================================================================
    // 2. Réouverture et lecture de quelques signaux
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        // --- Validation de "tensor_7" dans A[0]/B[1]/C[0]/D[0]/E[0] ---
        std::cout << "Lecture de 'tensor_7' dans A[0]/B[1]/C[0]/D[0]/E[0]...\n";
        {
            uint64_t static_indices[] = {0, 1, 0, 0, 0}; // A[0], B[1], C[0], D[0], E[0]
            int64_t time_index = -1;
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/0/B/1/C/0/D/0/E/0/tensor_7";
            int result = db.pz_readData_by_index(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0);
            assert(ndim_out == 1);
            assert(shape_out[0] == 7);
            assert(std::abs(data_out[0] - 107.0) < 1e-9);
            assert(std::abs(data_out[6] - 113.0) < 1e-9);
            std::cout << "Données 'tensor_7' lues avec succès.\n";
            delete[] data_out;
        }
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}