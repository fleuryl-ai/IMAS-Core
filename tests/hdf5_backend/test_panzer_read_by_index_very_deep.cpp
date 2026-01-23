// test_panzer_read_by_index_very_deep.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>

int main() {
    std::cout << "=== TEST PanzerDB: pz_readData_by_index (very deep) ===\n\n";
    const std::string filename = "test_read_by_index_very_deep.panzer";

    // ===================================================================
    // 1. Écriture d'une structure A[0]/B[1]/C[0]/D[0]/tensor
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Écriture de A[0]/B[1]/C[0]/D[0]/tensor...\n";
        db.beginArray("A", 1);
        db.beginArray("B", 2); // B a une taille de 2

        // On "saute" l'index B[0] en incrémentant manuellement
        db.incrementArrayIndex(); // Passe à B[1]

        // On écrit maintenant dans B[1]
        db.beginArray("C", 1);
        db.beginArray("D", 1);

        std::vector<double> data_to_write(2 * 3); // Tenseur 2D de 2x3
        std::iota(data_to_write.begin(), data_to_write.end(), 300.0);
        db.writeData("tensor", {2, 3}, data_to_write.data(), data_to_write.size());

        db.endArray(); // D
        db.endArray(); // C

        db.incrementArrayIndex(); // Incrémente B (même si on est à la fin)
        db.endArray(); // B
        db.incrementArrayIndex(); // Incrémente A (même si on est à la fin)
        db.endArray(); // A

        db.close();
        std::cout << "Fichier fermé.\n\n";
    }

    // ===================================================================
    // 2. Réouverture et lecture avec pz_readData_by_index
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        std::cout << "Lecture avec pz_readData_by_index...\n";

        uint64_t static_indices[] = {0, 1, 0, 0}; // A[0], B[1], C[0], D[0]
        int64_t time_index = -1; // Statique
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        const char* full_path = "A/0/B/1/C/0/D/0/tensor";
        int result = db.pz_readData_by_index(full_path, time_index, &ndim_out, shape_out, &data_out);

        assert(result == 0);
        assert(ndim_out == 2);
        assert(shape_out[0] == 2);
        assert(shape_out[1] == 3);

        std::cout << "Données lues: ";
        for (uint64_t i = 0; i < (shape_out[0] * shape_out[1]); ++i) {
            std::cout << data_out[i] << " ";
            assert(std::abs(data_out[i] - (300.0 + i)) < 1e-9);
        }
        std::cout << "\n";

        delete[] data_out;
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}