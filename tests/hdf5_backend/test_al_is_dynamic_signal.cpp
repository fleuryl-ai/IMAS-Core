// @file  test_al_is_dynamic_signal.cpp
// @brief Regression test for PanzerDB::isDynamicSignal / imas::direct_access::isDynamicSignal.
//
// Covers the five classifications:
//   (1) signal under a dynamic AoS              -> DYNAMIC
//   (2) own time axis (bulk scalar)             -> DYNAMIC
//   (3) own time axis (bulk multi-dim spatial)  -> DYNAMIC
//   (4a) static scalar (no time)                -> STATIC
//   (4b) static 1D array (no time)              -> STATIC   (1D spatial is NOT the time axis)
//   (5) nonexistent path                        -> false (not found, treated as static)
#include "panzerdb.h"
#include "direct_access_api.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string DB = "test_db_is_dynamic_signal";

int main() {
    try {
        std::cout << BOLD
                  << "\n=== Test PanzerDB::isDynamicSignal ===\n"
                  << RESET;

        if (std::filesystem::exists(DB)) std::filesystem::remove_all(DB);

        // ----------------------------------------------------------------
        // 1. Write fixtures
        // ----------------------------------------------------------------
        {
            PanzerDB db(DB, PanzerDB::OpenMode::WRITE);
            int32_t ht = 1;
            db.writeData("ids_properties/homogeneous_time", {}, &ht, 1);

            // (1) dynamic AoS "loops", scalar per slice (time axis comes from the AoS)
            db.beginArray("loops", "time");
            for (int t = 0; t < 2; ++t) {
                db.setCurrentArrayIndex(t);
                double time_val = t * 0.1;
                db.writeData("time", {}, &time_val, 1);
                double r = 2.0;
                db.writeData("r", {}, &r, 1);
            }
            db.endArray();

            std::vector<double> flux(5);
            for (int i = 0; i < 5; ++i) flux[i] = i * 1.0;
            db.writeDataSlices("flux", {}, flux.data(), 5, "time");

            // (3) 2D spatial [3,4] sampled over 2 time steps (bulk), base_shape {3,4}
            std::vector<double> field(3 * 4 * 2);  // [dim0, dim1, time]
            for (int i = 0; i < (int)field.size(); ++i) field[i] = i * 0.5;
            db.writeDataSlices("field", {3, 4}, field.data(), 2, "time");

            // (4a) static scalar
            double id = 42.0;
            db.writeData("id", {}, &id, 1);

            // (4b) static 1D array (not time-varying)
            std::vector<double> vec(5, 3.0);
            db.writeData("vec", {5}, vec.data(), 5);

            db.close();
        }

        // ----------------------------------------------------------------
        // 2. Queries (PanzerDB direct, same as imas::direct_access::isDynamicSignal)
        // ----------------------------------------------------------------
        int nfail = 0;
        auto check = [&nfail](bool cond, const std::string& what) {
            std::cout << (cond ? GREEN : RED) << "  " << (cond ? "OK  " : "FAIL")
                      << "  " << what << "\n" << RESET;
            if (!cond) ++nfail;
        };

        {
            PanzerDB q(DB, PanzerDB::OpenMode::READ);

            bool dynamic_loops_r = q.isDynamicSignal("loops/0/r");
            check(dynamic_loops_r == true,
                  "loops/0/r            under dynamic AoS          -> DYNAMIC");
            check(q.isDynamicSignal("loops/1/r") == true,
                  "loops/1/r            under dynamic AoS          -> DYNAMIC");
            check(q.isDynamicSignal("flux") == true,
                  "flux                 bulk scalar over 5 steps   -> DYNAMIC");
            check(q.isDynamicSignal("field") == true,
                  "field                bulk 2D[3,4] over 2 steps  -> DYNAMIC");
            check(q.isDynamicSignal("id") == false,
                  "id                   static scalar              -> STATIC");
            check(q.isDynamicSignal("vec") == false,
                  "vec                  static 1D array            -> STATIC");
            check(q.isDynamicSignal("no/such/path") == false,
                  "no/such/path         unknown                    -> false");

            // And through the imas::direct_access wrapper (same result).
            check(imas::direct_access::isDynamicSignal(DB, "flux") == true,
                  "direct_access::isDynamicSignal(DB, \"flux\")    -> DYNAMIC");
            check(imas::direct_access::isDynamicSignal(DB, "vec") == false,
                  "direct_access::isDynamicSignal(DB, \"vec\")     -> STATIC");

            // Cross-reference against isDynamicAOS (should be true for loops only).
            check(q.isDynamicAOS("loops") == true,
                  "isDynamicAOS(\"loops\")                          -> true");
            check(q.isDynamicAOS("loops/0/r") == false,
                  "isDynamicAOS(\"loops/0/r\")                      -> false (leaf, not AoS)");
        }

        if (std::filesystem::exists(DB)) std::filesystem::remove_all(DB);

        if (nfail == 0) {
            std::cout << GREEN << BOLD << "\n[OK] All isDynamicSignal assertions passed.\n"
                      << RESET;
        } else {
            std::cerr << RED << BOLD << "\n[FAIL] " << nfail << " assertion(s) failed.\n"
                      << RESET;
        }
        return nfail;
    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << "\n" << RESET;
        if (std::filesystem::exists(DB)) std::filesystem::remove_all(DB);
        return 1;
    }
}
