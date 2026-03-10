// test_light.cpp  Test complet de la version légère
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=== TEST VERSION LÉGÈRE PANZERDB (sans index_names) ===\n\n";

    // ===================================================================
    // 1. Écriture : AOS + nœuds vides + slices temporelles
    // ===================================================================
    {
        PanzerDB db("light_test.panzer", PanzerDB::OpenMode::WRITE, true);

        std::cout << "Écriture AOS avec nœuds vides...\n";
        db.beginArray("A", 3);
        for (int a = 0; a < 3; ++a) {
            // La taille de B dépend de l'index de A
            size_t size_b = 5 + a;
            db.beginArray("B", size_b);

            for (size_t b = 0; b < size_b; ++b) {
                // Condition pour laisser A[0]/B[2] vide
                if (a == 0 && b == 2) {
                    // Ne rien écrire, juste incrémenter pour passer à l'élément suivant de B
                } else {
                    std::vector<double> data(10, 100.0 + a * 10 + b);
                    db.writeData<double>("eeg", {10}, data.data(), data.size());
                }
                // Incrémenter l'index de B à chaque itération
                if (b < size_b -1) {
                    db.incrementArrayIndex();
                }
            }
            db.endArray(); // Fin de B pour l'instance courante de A

            // Incrémenter l'index de A pour passer à l'élément suivant
            if (a < 2) {
                db.incrementArrayIndex();
            }
        }

        db.endArray();

        std::cout << "Ajout de 500 slices temporelles (data dynamique)...\n";
        size_t big_size = 64 * 1000 * 500;
        // OPTIMISATION: Allocation directe et remplissage par index (plus rapide que push_back)
        std::vector<double> big(big_size);
        for (size_t i = 0; i < big_size; ++i) big[i] = 1000.0 + i * 0.001;
        
        db.writeDataSlices("stream", {64, 1000}, big.data(), 500, "time");

        db.close();
        std::cout << "Fichier fermé.\n\n";
    }

    // ===================================================================
    // 2. Réouverture et lecture
    // ===================================================================
    {
        PanzerDB db("light_test.panzer", PanzerDB::OpenMode::READ);

        std::cout << "Vérification AOS shapes...\n";
        auto shapeA = db.getAOSShape("A");
        //auto shapeB = db.getAOSShape("B");
        std::cout << "A: "; for (auto x : shapeA) std::cout << x << ' '; std::cout << "\n";
        //std::cout << "B: "; for (auto x : shapeB) std::cout << x << ' '; std::cout << "\n";

        assert((shapeA == std::vector<size_t>{3}));

        auto shapeB_in_A0 = db.getAOSShape("A/0/B");
        auto shapeB_in_A1 = db.getAOSShape("A/1/B");
        auto shapeB_in_A2 = db.getAOSShape("A/2/B");

        std::cout << "B dans A[0]: ";
        for (auto x : shapeB_in_A0) std::cout << x << ' ';
        std::cout << "\n";

        std::cout << "B dans A[1]: ";
        for (auto x : shapeB_in_A1) std::cout << x << ' ';
        std::cout << "\n";

        std::cout << "B dans A[2]: ";
        for (auto x : shapeB_in_A2) std::cout << x << ' ';
        std::cout << "\n";

        assert(shapeB_in_A0[0] == 5);
        assert(shapeB_in_A1[0] == 6);
        assert(shapeB_in_A2[0] == 7);

        std::cout << "Lecture A[1].B[3].eeg (devrait être 100 + 1*10 + 3 = 113)...\n";
        const auto& all_leaves = db.getLeaves();
        
        // On cherche le chemin exact "A/1/B/eeg" et on compte les occurrences
        std::vector<const PanzerDB::Leaf*> a1b_eeg_leaves;
        a1b_eeg_leaves.reserve(all_leaves.size());
        std::string target_parent_path = "A/1/B";
        for (const auto& leaf : all_leaves) {
            // CORRECTION: On cherche un parent_path qui COMMENCE par target_parent_path
            if (leaf.flags == 0 && leaf.parent_path.rfind(target_parent_path, 0) == 0 && leaf.path.rfind("/eeg") != std::string::npos) {
                a1b_eeg_leaves.push_back(&leaf);
            }
        }
        
        std::cout << "Trouvé " << a1b_eeg_leaves.size() << " feuilles 'eeg' dans " << target_parent_path << "\n";
        
        if (a1b_eeg_leaves.size() > 3) { // A[1].B[3] est le 4ème élément (index 3)
            const auto& leaf_to_read = *a1b_eeg_leaves[3];
            std::vector<double> tensor(leaf_to_read.count);
            db.readTensor<double>(leaf_to_read, tensor.data());
            std::cout << "A[1].B[3].eeg[0] = " << tensor.front() << "\n";
            assert(std::abs(tensor.front() - 113.0) < 1e-9);
        } else {
            std::cout << "ERREUR: Pas assez de feuilles A/1/B/eeg (attendu au moins 4, trouvé " 
                      << a1b_eeg_leaves.size() << ")\n";
            return 1;
        }

        /*std::cout << "getSlice(374.6) sur stream...\n";
        std::vector<double> slice;
        db.getSlice(374.6, slice);
        std::cout << "Valeur[0] ≈ " << slice[0] << "\n";
        assert(std::abs(slice[0] - (1000.0 + 374 * 64000 * 0.001)) < 1e-3);

        std::cout << "Dernière slice (t=499)...\n";
        db.getSlice(499.0, slice);
        std::cout << "Valeur[0] ≈ " << slice[0] << "\n";*/

        db.close();
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS  VERSION LÉGÈRE PARFAITE !\n";
    return 0;
}