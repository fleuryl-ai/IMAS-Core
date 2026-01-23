// test_panzer_dynamic_data_put_and_slice.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    std::cout << "=== TEST PanzerDB: Dynamic Data Put() then Slice() ===\n\n";
    const std::string filename = "test_dynamic_put_slice.panzer";

    // Nettoyage avant le test
    if (fs::exists(filename)) {
        fs::remove(filename);
    }

    // ===================================================================
    // 1. Écriture initiale de données dynamiques (simule un put())
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        std::cout << "Étape 1: Écriture initiale de 2 slices avec writeDataSlices()...\n";

        // Écriture d'un signal 1D statique
        std::vector<double> static_signal_data = {42.0, 43.0, 44.0, 45.0};
        db.writeData("static_signal", {4}, static_signal_data.data(), static_signal_data.size());

        db.beginArray("A", 1);

        // Écriture d'un signal 1D statique DANS l'AoS A
        std::vector<double> static_in_aos_data = {101.0, 102.0, 103.0};
        db.writeData("static_in_aos", {3}, static_in_aos_data.data(), static_in_aos_data.size());

        // Écriture d'un AoS B dans A
        db.beginArray("B", 2);
        for (int i = 0; i < 2; ++i) {
            // Écriture d'un signal 2D statique dans B
            std::vector<double> matrix_data(2 * 3); // Matrice 2x3
            for(size_t j = 0; j < matrix_data.size(); ++j) matrix_data[j] = 1000.0 + i * 100.0 + j;
            db.writeData("matrix", {2, 3}, matrix_data.data(), matrix_data.size());
            db.incrementArrayIndex();
        }

        db.endArray(); // B

        double val0 = 10.0;
        db.writeDataSlices("signal", {1}, &val0, 1, "time");
        std::cout << "  Écrit signal[0] = " << val0 << "\n";

        double val1 = 20.0;
        db.writeDataSlices("signal", {1}, &val1, 1, "time");
        std::cout << "  Écrit signal[1]: " << val1 << "\n";

        db.endArray();
        db.close();
        std::cout << "PanzerDB fermé après l'écriture initiale.\n\n";
    }

    // ===================================================================
    // 2. Validation après l'écriture initiale
    // ===================================================================
    {
        PanzerDB db_read(filename, PanzerDB::OpenMode::READ);
        std::cout << "Étape 2: Validation après l'écriture initiale...\n";

        /*uint64_t total_time = db_read.getCurrentTime();
        assert(total_time == 2 && "Le temps global doit être 2 après 2 écritures.");
        std::cout << "  Temps global lu: " << total_time << " (attendu: 2)\n";*/

        uint64_t static_indices[] = {0};
        int64_t time_index = 0;
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        
        // Validation du signal statique
        // CORRECTION: Pour lire une donnée à la racine, n_static_levels doit être 0.
        // time_index doit être -1 pour les données statiques.
        const char* full_path_static_signal = "static_signal";
        int result = db_read.pz_readData_by_index(full_path_static_signal, -1, &ndim_out, shape_out, &data_out);
        assert(result == 0 && ndim_out == 1 && shape_out[0] == 4 && std::abs(data_out[0] - 42.0) < 1e-9 && std::abs(data_out[3] - 45.0) < 1e-9 && "Le signal statique doit être correct.");
        std::cout << "  Lu static_signal[0]: " << data_out[0] << ", static_signal[3]: " << data_out[3] << " (attendu: 42.0, 45.0)\n";
        delete[] data_out;

        // Validation du signal statique DANS l'AoS A[0]
        const char* full_path_static_in_aos = "A/0/static_in_aos";
        result = db_read.pz_readData_by_index(full_path_static_in_aos, -1, &ndim_out, shape_out, &data_out);
        assert(result == 0 && "La lecture du signal statique dans l'AoS doit réussir.");
        assert(ndim_out == 1 && shape_out[0] == 3 && "La shape du signal statique dans l'AoS est incorrecte.");
        assert(std::abs(data_out[0] - 101.0) < 1e-9 && std::abs(data_out[2] - 103.0) < 1e-9 && "Les données du signal statique dans l'AoS sont incorrectes.");
        std::cout << "  Lu static_in_aos[0]: " << data_out[0] << ", static_in_aos[2]: " << data_out[2] << " (attendu: 101.0, 103.0)\n";
        delete[] data_out;

        // Validation du signal 2D statique dans A[0]/B[1]
        uint64_t nested_static_indices[] = {0, 1}; // A[0], B[1]
        const char* full_path_matrix = "A/0/B/1/matrix";
        result = db_read.pz_readData_by_index(full_path_matrix, -1, &ndim_out, shape_out, &data_out);
        assert(result == 0 && "La lecture du signal 2D dans l'AoS imbriqué doit réussir.");
        assert(ndim_out == 2 && "La dimension du signal 2D doit être 2.");
        assert(shape_out[0] == 2 && shape_out[1] == 3 && "La shape du signal 2D est incorrecte.");
        // Valeur attendue pour A[0]/B[1] : 1000.0 + 1*100.0 + j
        assert(std::abs(data_out[0] - 1100.0) < 1e-9 && "Première valeur de la matrice incorrecte.");
        assert(std::abs(data_out[5] - 1105.0) < 1e-9 && "Dernière valeur de la matrice incorrecte.");
        std::cout << "  Lu matrix[0][0] dans A[0]/B[1]: " << data_out[0] << " (attendu: 1100.0)\n";
        delete[] data_out;
        //const char* full_path_matrix_last = "A/0/B/1/matrix";
        //result = db_read.pz_readData_by_index(full_path_matrix_last, -1, &ndim_out, shape_out, &data_out);
        // Validation du signal dynamique, slice 0
        time_index = 0;
        const char* full_path_signal_0 = "A/0/signal";
        result = db_read.pz_readData_by_index(full_path_signal_0, time_index, &ndim_out, shape_out, &data_out);
        assert(result == 0 && std::abs(data_out[0] - 10.0) < 1e-9 && "Slice 0 doit être 10.0");
        std::cout << "  Lu signal[0]: " << data_out[0] << " (attendu: 10.0)\n";
        delete[] data_out;

        // Validation du signal dynamique, slice 1
        time_index = 1;
        const char* full_path_signal_1 = "A/0/signal";
        result = db_read.pz_readData_by_index(full_path_signal_1, time_index, &ndim_out, shape_out, &data_out);
        assert(result == 0 && std::abs(data_out[0] - 20.0) < 1e-9 && "Slice 1 doit être 20.0");
        std::cout << "  Lu signal[1]: " << data_out[0] << " (attendu: 20.0)\n";
        delete[] data_out;

        db_read.close();
        std::cout << "Validation initiale réussie.\n\n";
    }

    // ===================================================================
    // 3. Ajout de nouvelles slices en mode APPEND
    // ===================================================================
    {
        // CHANGEMENT CLÉ: Utiliser OpenMode::APPEND au lieu de OpenMode::WRITE
        PanzerDB db_append(filename, PanzerDB::OpenMode::APPEND, true);
        std::cout << "Étape 3: Ajout d'une slice supplémentaire avec writeDataSlices()...\n";

        db_append.beginArray("A", 1);

        double val2 = 30.0;
        db_append.writeDataSlices("signal", {1}, &val2, 1, "time");
        std::cout << "  Écrit signal[2] = " << val2 << "\n";

        db_append.endArray();
        db_append.close();
        std::cout << "PanzerDB fermé après l'ajout de slices.\n\n";
    }

    // ===================================================================
    // 4. Validation après l'ajout de slices
    // ===================================================================
    {
        PanzerDB db_final_read(filename, PanzerDB::OpenMode::READ);
        std::cout << "Étape 4: Validation après l'ajout de slices...\n";

        /*uint64_t total_time = db_final_read.getCurrentTime();
        assert(total_time == 3 && "Le temps global doit être 3 après 3 écritures.");
        std::cout << "  Temps global lu: " << total_time << " (attendu: 3)\n";*/

        uint64_t static_indices[] = {0};
        int64_t time_index = 2;
        uint64_t ndim_out = 0;
        uint64_t shape_out[6] = {0};
        double* data_out = nullptr;
        const char* full_path_signal = "A/0/signal";
        int result = db_final_read.pz_readData_by_index(full_path_signal, time_index, &ndim_out, shape_out, &data_out);
        assert(result == 0 && std::abs(data_out[0] - 30.0) < 1e-9 && "Slice 2 doit être 30.0");
        std::cout << "  Lu signal[2]: " << data_out[0] << " (attendu: 30.0)\n";
        delete[] data_out;

        std::cout << "Validation finale réussie.\n\n";
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}