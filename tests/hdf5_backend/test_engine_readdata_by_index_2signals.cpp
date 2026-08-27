// test_panzer_read_by_index_deep_2_signals.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: readDataByIndex (deep, 2 signals) ===\n\n";
    const std::string filename = "test_read_by_index_deep_2_signals.panzer";

    // ===================================================================
    // 1. Écriture d'une structure A/B/C/flux et A/field
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Écriture de A/B/C/flux et A/field...\n";
        db.beginArray("A", 1);

        // Écriture du signal "field" directement sous A
        std::vector<double> field_data = {55.5, 66.6};
        db.writeData("field", {2}, field_data.data(), field_data.size());

        db.beginArray("B", 1);
        db.beginArray("C", 1);

        std::vector<double> flux_data = {201.5, 202.5, 203.5};
        db.writeData("flux", {3}, flux_data.data(), flux_data.size());

        db.incrementArrayIndex(); // Incrémente C
        db.endArray(); // C
        db.incrementArrayIndex(); // Incrémente B
        db.endArray(); // B
        db.incrementArrayIndex(); // Incrémente A
        db.endArray(); // A

        db.close();
        std::cout << "Fichier fermé.\n\n";
    }

    // ===================================================================
    // 2. Réouverture et lecture des deux signaux
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        // --- Validation de "flux" ---
        std::cout << "Lecture de 'flux' avec readDataByIndex...\n";
        {
            uint64_t static_indices[] = {0, 0, 0}; // A[0], B[0], C[0]
            int64_t time_index = -1;
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/0/B/0/C/0/flux";
            int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0);
            assert(ndim_out == 1);
            assert(shape_out[0] == 3);

            std::cout << "Données 'flux' lues: ";
            for (uint64_t i = 0; i < shape_out[0]; ++i) {
                std::cout << data_out[i] << " ";
                assert(std::abs(data_out[i] - (201.5 + i)) < 1e-9);
            }
            std::cout << "\n";
            delete[] data_out;
        }

        // --- Validation de "field" ---
        std::cout << "\nLecture de 'field' avec readDataByIndex...\n";
        {
            uint64_t static_indices[] = {0}; // A[0]
            int64_t time_index = -1;
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/0/field";
            int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);
            assert(result == 0);
            assert(ndim_out == 1 && shape_out[0] == 2);
            assert(std::abs(data_out[0] - 55.5) < 1e-9 && std::abs(data_out[1] - 66.6) < 1e-9);
            std::cout << "Données 'field' lues: " << data_out[0] << ", " << data_out[1] << "\n";
            delete[] data_out;
        }
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}