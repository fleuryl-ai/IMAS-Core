// test_panzer_empty_aos.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: getAOSShape on an AoS with no data leaves ===\n\n";
    const std::string filename = "test_empty_aos.panzer";

    // ===================================================================
    // 1. Write an AoS "A" of size 5.
    //    Each element of A is iterated, but no data is written
    //    with writeData().
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Writing an AoS 'A' of size 5 with no data...\n";
        db.beginArray("A", 5);
        for (int i = 0; i < 5; ++i) {
            // A child structure "B" is created for each element of "A",
            // but db.writeData() is never called.
            db.beginArray("B", 1);
            // No write, so no automatic increment of B's index.
            db.endArray(); // B
            db.incrementArrayIndex(); // Manually move to the next A element.
        }
        db.endArray(); // A

        db.close();
        std::cout << "File closed.\n\n";
    }

    // ===================================================================
    // 2. Reopen and verify the size of A
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        std::cout << "Verifying the size of AoS 'A'...\n";
        auto shapeA = db.getAOSShape("A");

        assert(shapeA.size() == 1 && "The shape of A must contain a single element.");
        assert(shapeA[0] == 5 && "The size of A must be 5.");
        std::cout << "Shape of A: { " << shapeA[0] << " } (expected: { 5 })\n";

        auto shapeB = db.getAOSShape("A/0/B");
        printf("shapeB[0]=%d\n", shapeB[0]);
        assert(shapeB[0] == 1 && "La taille de B doit être 1.");    

        auto shape1B = db.getAOSShape("A/1/B");
        assert(shape1B[0] == 1 && "La taille de B doit être 1.");

    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}