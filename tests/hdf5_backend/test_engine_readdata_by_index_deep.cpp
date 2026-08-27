// test_panzer_read_by_index_deep.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: readDataByIndex (deep) ===\n\n";
    const std::string filename = "test_read_by_index_deep.panzer";

    // ===================================================================
    // 1. Écriture d'une structure A[0]/B[0]/C[0]/flux
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Écriture de A[0]/B[0]/C[0]/flux...\n";
        db.beginArray("A", 1);
        db.beginArray("B", 1);
        db.beginArray("C", 1);

        std::vector<double> data_to_write = {201.5, 202.5, 203.5};
        db.writeDataSlices("flux", {3}, data_to_write.data(), 1, "time");

        db.endArray(); // C
        db.endArray(); // B
        db.endArray(); // A

        db.close();
        std::cout << "Fichier fermé.\n\n";
    }

    // ===================================================================
    // 2. Réouverture et lecture avec readDataByIndex
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        std::cout << "Lecture avec readDataByIndex...\n";

        uint64_t static_indices[] = {0, 0, 0}; // A[0], B[0], C[0]
        int64_t time_index = -1; // Statique
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        const char* full_path = "A/0/B/0/C/0/flux";
        int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

        assert(result == 0);
        assert(ndim_out == 1);
        assert(shape_out[0] == 3);

        std::cout << "Données lues: ";
        for (uint64_t i = 0; i < shape_out[0]; ++i) {
            std::cout << data_out[i] << " ";
            assert(std::abs(data_out[i] - (201.5 + i)) < 1e-9);
        }
        std::cout << "\n";

        delete[] data_out; // N'oubliez pas de libérer la mémoire
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}