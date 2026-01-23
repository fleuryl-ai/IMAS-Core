// test_panzer_read_by_index_complex_aos.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: pz_readData_by_index (complex AoS) ===\n\n";
    const std::string filename = "test_read_by_index_complex_aos.panzer";

    // ===================================================================
    // 1. Écriture d'une structure complexe
    //    A (size=2)
    //    - A[0]/field
    //    - A[0]/B (size=2)
    //      - A[0]/B[0]/C (size=1) / flux
    //      - A[0]/B[1]/C (size=1) / flux
    //    - A[1]/field
    //    - A[1]/B (size=3)
    //      - A[1]/B[0]/C (size=1) / flux
    //      ...
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Écriture de la structure complexe...\n";
        db.beginArray("A", 2);
        // --- Écriture dans A[0] ---
        std::vector<double> field_data_0 = {10.0, 11.0};
        db.writeData("field", {2}, field_data_0.data(), field_data_0.size());
        db.beginArray("B", 2);
        for (int b = 0; b < 2; ++b) {
            db.beginArray("C", 1);
            std::vector<double> flux_data_0x = {1000.0 + (double)b};
            db.writeData("flux", {1}, flux_data_0x.data(), flux_data_0x.size());
            db.endArray(); // C
            db.incrementArrayIndex(); // Incrémente l'index de B
        }
        db.endArray(); // B
        db.incrementArrayIndex(); // Incrémente l'index de A

        // --- Écriture dans A[1] ---
        std::vector<double> field_data_1 = {20.0, 21.0, 22.0};
        db.writeData("field", {3}, field_data_1.data(), field_data_1.size());
        db.beginArray("B", 3);
        for (int b = 0; b < 3; ++b) {
            db.beginArray("C", 1);
            std::vector<double> flux_data_1x = {2000.0 + (double)b};
            db.writeData("flux", {1}, flux_data_1x.data(), flux_data_1x.size());
            db.endArray(); // C
            db.incrementArrayIndex(); // Incrémente l'index de B
        }
        db.endArray(); // B
        db.incrementArrayIndex(); // Incrémente l'index de A

        db.endArray(); // A        
        db.close();
        std::cout << "Fichier fermé.\n\n";
    }

    // ===================================================================
    // 2. Réouverture et lecture des signaux
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        // --- Validation de "flux" dans A[1]/B[2]/C[0] ---
        std::cout << "Lecture de 'flux' dans A[1]/B[2]/C[0]...\n";
        {
            uint64_t static_indices[] = {1, 2, 0}; // A[1], B[2], C[0]
            int64_t time_index = -1;
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/1/B/2/C/0/flux";
            int result = db.pz_readData_by_index(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0);
            assert(ndim_out == 1);
            assert(shape_out[0] == 1);
            assert(std::abs(data_out[0] - 2002.0) < 1e-9);
            std::cout << "Données 'flux' lues: " << data_out[0] << " (attendu: 2002.0)\n";
            delete[] data_out;
        }

        // --- Validation de "field" dans A[0] ---
        std::cout << "\nLecture de 'field' dans A[0]...\n";
        {
            uint64_t static_indices[] = {0}; // A[0]
            int64_t time_index = -1;
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/0/field";
            int result = db.pz_readData_by_index(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0);
            assert(ndim_out == 1);
            assert(shape_out[0] == 2);
            assert(std::abs(data_out[0] - 10.0) < 1e-9);
            assert(std::abs(data_out[1] - 11.0) < 1e-9);
            std::cout << "Données 'field' lues: " << data_out[0] << ", " << data_out[1] << " (attendu: 10.0, 11.0)\n";
            delete[] data_out;
        }
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}