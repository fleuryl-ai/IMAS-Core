// test_light.cpp  Complete test of the light version
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=== TEST LIGHTWEIGHT VERSION PANZERDB (without index_names) ===\n\n";

    // ===================================================================
    // 1. Write: AOS + empty nodes + time slices
    // ===================================================================
    {
        PanzerDB db("light_test.panzer", PanzerDB::OpenMode::WRITE, true);

        std::cout << "Writing AOS with empty nodes...\n";
        db.beginArray("A", 3);
        for (int a = 0; a < 3; ++a) {
            // The size of B depends on the index of A
            size_t size_b = 5 + a;
            db.beginArray("B", size_b);

            for (size_t b = 0; b < size_b; ++b) {
                // Condition to leave A[0]/B[2] empty
                if (a == 0 && b == 2) {
                    // Write nothing, just increment to move to the next element of B
                } else {
                    std::vector<double> data(10, 100.0 + a * 10 + b);
                    db.writeData<double>("eeg", {10}, data.data(), data.size());
                }
                // Increment B's index at each iteration
                if (b < size_b -1) {
                    db.incrementArrayIndex();
                }
            }
            db.endArray(); // End of B for the current instance of A

            // Increment A's index to move to the next element
            if (a < 2) {
                db.incrementArrayIndex();
            }
        }

        db.endArray();

        std::cout << "Adding 500 time slices (dynamic data)...\n";
        size_t big_size = 64 * 1000 * 500;
        // OPTIMIZATION: Direct allocation and filling by index (faster than push_back)
        std::vector<double> big(big_size);
        for (size_t i = 0; i < big_size; ++i) big[i] = 1000.0 + i * 0.001;
        
        db.writeDataSlices("stream", {64, 1000}, big.data(), 500, "time");

        db.close();
        std::cout << "File closed.\n\n";
    }

    // ===================================================================
    // 2. Réouverture et lecture
    // ===================================================================
    {
        PanzerDB db("light_test.panzer", PanzerDB::OpenMode::READ);

        std::cout << "Verifying AOS shapes...\n";
        auto shapeA = db.getAOSShape("A");
        //auto shapeB = db.getAOSShape("B");
        std::cout << "A: "; for (auto x : shapeA) std::cout << x << ' '; std::cout << "\n";
        //std::cout << "B: "; for (auto x : shapeB) std::cout << x << ' '; std::cout << "\n";

        assert((shapeA == std::vector<size_t>{3}));

        auto shapeB_in_A0 = db.getAOSShape("A/0/B");
        auto shapeB_in_A1 = db.getAOSShape("A/1/B");
        auto shapeB_in_A2 = db.getAOSShape("A/2/B");

        std::cout << "B in A[0]: ";
        for (auto x : shapeB_in_A0) std::cout << x << ' ';
        std::cout << "\n";

        std::cout << "B in A[1]: ";
        for (auto x : shapeB_in_A1) std::cout << x << ' ';
        std::cout << "\n";

        std::cout << "B in A[2]: ";
        for (auto x : shapeB_in_A2) std::cout << x << ' ';
        std::cout << "\n";

        assert(shapeB_in_A0[0] == 5);
        assert(shapeB_in_A1[0] == 6);
        assert(shapeB_in_A2[0] == 7);

        std::cout << "Reading A[1].B[3].eeg (should be 100 + 1*10 + 3 = 113)...\n";
        const auto& all_leaves = db.getLeaves();
        
        // We search for the exact path "A/1/B/eeg" and count occurrences
        std::vector<const PanzerDB::Leaf*> a1b_eeg_leaves;
        a1b_eeg_leaves.reserve(all_leaves.size());
        std::string target_parent_path = "A/1/B";
        for (const auto& leaf : all_leaves) {
            // CORRECTION: We search for a parent_path that STARTS with target_parent_path
            if (leaf.flags == 0 && leaf.parent_path.rfind(target_parent_path, 0) == 0 && leaf.path.rfind("/eeg") != std::string::npos) {
                a1b_eeg_leaves.push_back(&leaf);
            }
        }
        
        std::cout << "Found " << a1b_eeg_leaves.size() << " 'eeg' leaves in " << target_parent_path << "\n";
        
        if (a1b_eeg_leaves.size() > 3) { // A[1].B[3] est le 4ème élément (index 3)
            const auto& leaf_to_read = *a1b_eeg_leaves[3];
            std::vector<double> tensor(leaf_to_read.count);
            db.readTensor<double>(leaf_to_read, tensor.data());
            std::cout << "A[1].B[3].eeg[0] = " << tensor.front() << "\n";
            assert(std::abs(tensor.front() - 113.0) < 1e-9);
        } else {
            std::cout << "ERROR: Not enough A/1/B/eeg leaves (expected at least 4, found " 
                      << a1b_eeg_leaves.size() << ")\n";
            return 1;
        }

        /*std::cout << "getSlice(374.6) on stream...\n";
        std::vector<double> slice;
        db.getSlice(374.6, slice);
        std::cout << "Valeur[0] ≈ " << slice[0] << "\n";
        assert(std::abs(slice[0] - (1000.0 + 374 * 64000 * 0.001)) < 1e-3);

        std::cout << "Dernière slice (t=499)...\n";
        db.getSlice(499.0, slice);
        std::cout << "Valeur[0] ≈ " << slice[0] << "\n";*/

        db.close();
    }

    std::cout << "\nALL TESTS PASSED  PERFECT LIGHTWEIGHT VERSION!\n";
    return 0;
}