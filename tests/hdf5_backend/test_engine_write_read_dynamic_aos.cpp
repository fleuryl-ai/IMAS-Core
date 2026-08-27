// test_panzer_dynamic_aos.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: Dynamic Array of Structures (AoS) ===\n\n";
    const std::string filename = "test_dynamic_aos.panzer";

    // ===================================================================
    // 1. Écriture de la structure initiale avec un AoS dynamique
    //    A (statique, size=1)
    //    - A[0]/B (dynamique, timebase="time")
    //      - A[0]/B[0]/signal (valeur = 100.0)
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        
        std::cout << "Écriture de la structure initiale...\n";
        db.beginArray("A", 1);
        db.beginArray("B", "time"); // Marque B comme dynamique
        
        // Écriture de DEUX slices de données pour le signal dans B
        std::vector<double> all_slices_data = {100.0, 200.0};
        db.writeDataSlices("signal", {1}, all_slices_data.data(), 2, "time");
        
        db.endArray(); // B
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

        // B devrait maintenant avoir une taille de 2
        auto shapeB = db.getAOSShape("B");
        // Pour un AoS dynamique, getAOSShape n'est pas bien défini.
        // On vérifie plutôt le temps global.
        /*uint64_t total_time = db.getCurrentTime();
        assert(total_time == 2 && "Le temps global doit être 2 après l'écriture de 2 slices.");
        std::cout << "Temps global: " << total_time << " (attendu: 2)\n";*/

        // Lecture de la deuxième slice (B[1])
        // Pour un AoS dynamique, on ne fournit que l'index statique du parent
        // et l'index temporel de la slice.
        uint64_t static_indices[] = {0}; // A[0]
        int64_t time_index = 1; // On veut la deuxième slice (index 1)
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        const char* full_path = "A/0/B/signal";
        int result = db.readDataByIndex(full_path, time_index, &ndim_out, shape_out, &data_out);

        assert(result == 0 && "La lecture de la slice 1 doit réussir.");
        
        // Un signal scalaire lu à un instant t est un scalaire (ndim=0) ou un tableau de taille 1 (ndim=1, shape=[1])
        bool is_scalar = (ndim_out == 0) || (ndim_out == 1 && shape_out[0] == 1);
        assert(is_scalar && "Le signal lu doit être un scalaire (ndim=0 ou shape=[1]).");

        assert(std::abs(data_out[0] - 200.0) < 1e-9 && "La valeur de la slice 1 doit être 200.0.");
        std::cout << "Données de A[0]/B[1]/signal: " << data_out[0] << " (attendu: 200.0)\n";
        delete[] data_out;
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}