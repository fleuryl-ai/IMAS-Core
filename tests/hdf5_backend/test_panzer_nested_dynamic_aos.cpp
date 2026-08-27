// test_panzer_nested_dynamic_aos.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: Nested Dynamic Array of Structures (AoS) ===\n\n";
    const std::string filename = "test_nested_dynamic_aos.panzer";

    // ===================================================================
    // 1. Écriture d'une structure complexe
    //    A (statique, size=2)
    //    - A[0]/B (dynamique, timebase="time")
    //      - 3 slices de "signal" et "signal_2"
    //    - A[1]/B (dynamique, timebase="time")
    //      - 5 slices de "signal" et "signal_2"
    //    - A[2]/B (dynamique, timebase="time")
    //      - 2 slices de "signal" et "signal_2"
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Écriture de la structure imbriquée...\n";
        db.beginArray("A", 3);

        // --- Écriture dans A[0] ---
        db.beginArray("B", "time"); // Marque B comme dynamique
        std::vector<double> data_A0_s1 = {100.0, 101.0, 102.0};
        std::vector<double> data_A0_s2 = {150.0, 151.0, 152.0};
        db.writeDataSlices("signal", {1}, data_A0_s1.data(), 3, "");
        db.writeDataSlices("signal_2", {1}, data_A0_s2.data(), 3, "");
        db.endArray(); // B
        db.incrementArrayIndex(); // Passe à A[1]

        // --- Écriture dans A[1] ---
        db.beginArray("B", "time"); // Marque B comme dynamique
        std::vector<double> data_A1_s1 = {200.0, 201.0, 202.0, 203.0, 204.0};
        std::vector<double> data_A1_s2 = {250.0, 251.0, 252.0, 253.0, 254.0};
        db.writeDataSlices("signal", {1}, data_A1_s1.data(), 5, "");
        db.writeDataSlices("signal_2", {1}, data_A1_s2.data(), 5, "");
        db.endArray(); // B
        db.incrementArrayIndex(); // Passe à A[2]

        // --- Écriture dans A[2] ---
        db.beginArray("B", "time"); // Marque B comme dynamique
        std::vector<double> data_A2_s1 = {300.0, 301.0};
        std::vector<double> data_A2_s2 = {350.0, 351.0};
        db.writeDataSlices("signal", {1}, data_A2_s1.data(), 2, "");
        db.writeDataSlices("signal_2", {1}, data_A2_s2.data(), 2, "");
        db.endArray(); // B
        db.incrementArrayIndex(); // Incrémente A (même si on est à la fin)

        db.endArray(); // A
        db.close();
        std::cout << "Fichier fermé.\n\n";
    }

    // ===================================================================
    // 2. Réouverture et vérification
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        std::cout << "Vérification de la structure...\n";

        // Le temps global n'est plus un concept unique et simple.
        // On valide les données en se basant sur les time_index locaux à chaque AoS dynamique.

        // Lecture de la 4ème slice (index 3) de "signal" dans A[1]
        // Le time_index est local à l'AoS A/1/B.
        // La 4ème slice de 'signal' dans A[1]/B a un time_index local de 3.
        {
            uint64_t static_indices[] = {1}; // A[1]
            int64_t time_index = 3; // L'index local de la 4ème slice est 3.
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/1/B/signal";
            int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0 && "La lecture de la slice doit réussir.");
            
            // Un signal scalaire lu à un instant t est un scalaire (ndim=0) ou un tableau de taille 1 (ndim=1, shape=[1])
            bool is_scalar = (ndim_out == 0) || (ndim_out == 1 && shape_out[0] == 1);
            assert(is_scalar && "Le signal lu doit être un scalaire (ndim=0 ou shape=[1]).");

            assert(std::abs(data_out[0] - 203.0) < 1e-9 && "La valeur de la slice doit être 203.0.");
            std::cout << "  [OK] Données de A[1]/B[slice_3]/signal: " << data_out[0] << " (attendu: 203.0)\n";
            delete[] data_out;
        }

        // Lecture de la 1ère slice (index 0) de "signal_2" dans A[2]
        // Le time_index est local à l'AoS A/2/B.
        // CORRECTION: Avec la logique d'alignement correcte, 'signal' et 'signal_2' sont alignés.
        // Donc 'signal_2' commence à l'index 0, tout comme 'signal'.
        {
            uint64_t static_indices[] = {2}; // A[2]
            int64_t time_index = 0; // Alignement : signal_2 commence à 0
            uint64_t ndim_out = 0;
            uint64_t shape_out[6] = {0};
            double* data_out = nullptr;
            const char* full_path = "A/2/B/signal_2";
            int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

            assert(result == 0 && "La lecture de la slice doit réussir.");
            
            // Un signal scalaire lu à un instant t est un scalaire (ndim=0) ou un tableau de taille 1 (ndim=1, shape=[1])
            bool is_scalar = (ndim_out == 0) || (ndim_out == 1 && shape_out[0] == 1);
            assert(is_scalar && "Le signal lu doit être un scalaire (ndim=0 ou shape=[1]).");

            assert(std::abs(data_out[0] - 350.0) < 1e-9 && "La valeur de la slice doit être 350.0.");
            std::cout << "  [OK] Données de A[2]/B[slice_0]/signal_2: " << data_out[0] << " (attendu: 350.0)\n";
            delete[] data_out;
        }
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}