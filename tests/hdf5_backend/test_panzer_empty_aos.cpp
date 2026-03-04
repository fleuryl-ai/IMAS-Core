// test_panzer_empty_aos.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: getAOSShape on an AoS with no data leaves ===\n\n";
    const std::string filename = "test_empty_aos.panzer";

    // ===================================================================
    // 1. Écriture d'un AoS "A" de taille 5.
    //    Chaque élément de A est parcouru, mais aucune donnée n'est écrite
    //    avec writeData().
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Écriture d'un AoS 'A' de taille 5 sans aucune donnée...\n";
        db.beginArray("A", 5);
        for (int i = 0; i < 5; ++i) {
            // On crée une structure enfant "B" pour chaque élément de "A",
            // mais on n'appelle jamais db.writeData().
            db.beginArray("B", 1);
            // Pas d'écriture, donc pas d'incrément automatique de l'index de B.
            db.endArray(); // B
            db.incrementArrayIndex(); // On passe manuellement à l'élément A suivant.
        }
        db.endArray(); // A

        db.close();
        std::cout << "Fichier fermé.\n\n";
    }

    // ===================================================================
    // 2. Réouverture et vérification de la taille de A
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        std::cout << "Vérification de la taille de l'AoS 'A'...\n";
        auto shapeA = db.getAOSShape("A");

        assert(shapeA.size() == 1 && "La shape de A doit contenir un seul élément.");
        assert(shapeA[0] == 5 && "La taille de A doit être 5.");
        std::cout << "Shape de A: { " << shapeA[0] << " } (attendu: { 5 })\n";

        auto shapeB = db.getAOSShape("A/0/B");
        printf("shapeB[0]=%d\n", shapeB[0]);
        assert(shapeB[0] == 1 && "La taille de B doit être 1.");    

        auto shape1B = db.getAOSShape("A/1/B");
        assert(shape1B[0] == 1 && "La taille de B doit être 1.");

        /*auto shapeB = db.getAOSShape("B");
        std::vector<size_t> expected_shapeB = {1, 1, 1, 1, 1};
        assert(shapeB == expected_shapeB && "La shape de B doit être {1, 1, 1, 1, 1}.");
        std::cout << "Shape de B: { ";
        for(size_t s : shapeB) { std::cout << s << " "; }
        std::cout << "} (attendu: { 1 1 1 1 1 })\n";*/

        /*size_t aos_size = db.getCurrentTotalSize("B");
        printf("aos_size=%d\n ", aos_size);
        assert(aos_size == 5 && "La taille totale de B doit être 5.");*/

    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}