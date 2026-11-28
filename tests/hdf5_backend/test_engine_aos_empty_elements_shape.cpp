// @file  test_engine_aos_empty_elements_shape.cpp
// @brief getAOSShape must return the DECLARED element count of an AoS, not the
//        number of elements that happen to carry a value.
//
// Measured defect (pulse 65782 fidelity campaign, 2.0 read vs 1.0 read): an AoS
// whose elements are not all filled is read with fewer elements from the 2.0
// file than from the 1.0 file — the 2.0 file kept every element, but the reader
// derived the size from the data leaves:
//   - calorimetry group occ 0         : 19 elements stored  -> 1 read
//   - core_profiles profiles_2d occ 2 : 86080 stored         -> 0 read
//   - equilibrium grids_ggd occ 2     : 1 element stored     -> 0 read
//
// Reproduction (kept small):
//   1. STATIC AoS "group", 19 elements, only element 0 filled. The writer
//      records the declared size on the meta-row and the leaf of element 0;
//      the read must see 19 elements and the value of element 0.
//   2. DYNAMIC AoS "profiles_2d", 5 elements iterated, static children with
//      no value anywhere. The read must see 5 elements and the declared size
//      (2) of the empty static children.
//
// Before the fix: case 1 reads 1 (max filled index + 1, the declared size is
// only consulted when NO element leaf exists at all); case 2 reads 0 (no data
// leaf -> no time range -> "size is 0"). Both assert the declared size now.

#include "panzerdb.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <filesystem>

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "[FAIL] " << msg << std::endl;                      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

int main() {
    const std::string f1 = "test_aos_partial_static.panzer";
    const std::string f2 = "test_aos_empty_dynamic.panzer";
    std::filesystem::remove(f1);
    std::filesystem::remove(f2);

    const int N_STATIC = 19;   // like calorimetry group (occ 0)
    const int N_DYN    = 5;    // stand-in for the 86080 profiles_2d elements
    const int N_ION    = 2;    // declared size of the empty static child

    std::cout << "=== TEST PanzerDB: getAOSShape with EMPTY AoS ELEMENTS ===\n\n";

    // ===================================================================
    // Case 1 — STATIC AoS partially filled
    // ===================================================================
    {
        std::cout << "--- Writing a static AoS 'group' of " << N_STATIC
                  << " elements, only element 0 filled ---\n";
        PanzerDB db(f1, PanzerDB::OpenMode::WRITE, true);
        db.beginArray("group", N_STATIC);
        for (int i = 0; i < N_STATIC; ++i) {
            if (i == 0) {
                const double weight = 42.0;
                db.writeData("weight", {1}, &weight, 1);
            }
            if (i < N_STATIC - 1) db.incrementArrayIndex();
        }
        db.endArray();
        db.close();
    }
    {
        std::cout << "--- Reading it back: size must be " << N_STATIC << " ---\n";
        PanzerDB db(f1, PanzerDB::OpenMode::READ);

        auto shape = db.getAOSShape("group");
        std::cout << "group size = " << (shape.empty() ? 0 : shape[0])
                  << " (expected " << N_STATIC << ")\n";
        CHECK(shape.size() == 1 && shape[0] == N_STATIC,
              "static AoS with one filled element must read with its declared size ("
              "the empty elements are part of the pulse, the file stores them)");

        // The filled element is still there, still readable.
        uint64_t ndim = 0; uint64_t out_shape[6] = {0}; double* v = nullptr;
        CHECK(db.readDataByIndex("group/0/weight", -1, &ndim, out_shape, &v) == 0 && v,
              "group/0/weight must be readable");
        CHECK(std::abs(*v - 42.0) < 1e-12, "group/0/weight value corrupted");
        free(v);
    }

    // ===================================================================
    // Case 2 — DYNAMIC AoS iterated, but with no value anywhere
    // ===================================================================
    {
        std::cout << "\n--- Writing a dynamic AoS 'profiles_2d' of " << N_DYN
                  << " elements, empty static children, no value ---\n";
        PanzerDB db(f2, PanzerDB::OpenMode::WRITE, true);
        db.beginArray("profiles_2d", "time");
        for (int t = 0; t < N_DYN; ++t) {
            db.beginArray("ion", N_ION);
            db.beginArray("element", 1);
            db.endArray();   // element
            db.endArray();   // ion
            if (t < N_DYN - 1) db.incrementArrayIndex();
        }
        db.endArray();   // profiles_2d
        db.close();
    }
    {
        std::cout << "--- Reading it back: size must be " << N_DYN << " ---\n";
        PanzerDB db(f2, PanzerDB::OpenMode::READ);

        auto shape = db.getAOSShape("profiles_2d");
        std::cout << "profiles_2d size = " << (shape.empty() ? 0 : shape[0])
                  << " (expected " << N_DYN << ")\n";
        CHECK(shape.size() == 1 && shape[0] == N_DYN,
              "dynamic AoS whose elements carry no value must read with its iterated size "
              "(the file stores the instances; size may not collapse to 0)");

        auto ion = db.getAOSShape("profiles_2d/0/ion");
        std::cout << "profiles_2d/0/ion size = " << (ion.empty() ? 0 : ion[0])
                  << " (expected " << N_ION << ")\n";
        CHECK(ion.size() == 1 && ion[0] == N_ION,
              "the declared size of an empty static child must survive the read");
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}
