// @file  test_engine_time_context.cpp
// @brief Pins the table restoreTimeContext() rebuilds when a PanzerDB file is opened
//        (aos_time_counters) — the table getDynamicAOSSize and getAOSShape read from
//        on the READ side, and the anchor the APPEND writer continues from.
//
// It exists because that loop is a hot path (it walks every index row at each open)
// and was rewritten to drop per-row allocations: the values it produces must not move.
// The two things it pins:
//   * a per-slice signal contributes (time_index + number of slices in the row) to its
//     OWN path and to its innermost dynamic AoS;
//   * a bulk leaf (one row holding the whole array) counts as ONE time step.
#include "panzerdb.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(const char* what, uint64_t got, uint64_t want) {
    if (got != want) {
        std::cout << "  FAIL: " << what << " = " << got << ", expected " << want << "\n";
        ++failures;
    }
}

int main() {
    const std::string filename = "test_engine_time_context.panzer";
    std::filesystem::remove(filename);
    std::cout << "=== TEST restoreTimeContext: counters rebuilt at open ===\n";

    // A (static, 2)
    //   A[0]/B (dynamic): signal = 3 slices, signal_2 = 5 slices
    //   A[1]/B (dynamic): signal = 2 slices
    // root: time = one bulk row of 7 values
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        db.beginArray("A", 2);

        db.beginArray("B", "time");
        std::vector<double> s1 = {100.0, 101.0, 102.0};
        std::vector<double> s2 = {150.0, 151.0, 152.0, 153.0, 154.0};
        db.writeDataSlices("signal", {1}, s1.data(), 3, "");
        db.writeDataSlices("signal_2", {1}, s2.data(), 5, "");
        db.endArray();
        db.incrementArrayIndex();

        db.beginArray("B", "time");
        std::vector<double> s3 = {200.0, 201.0};
        db.writeDataSlices("signal", {1}, s3.data(), 2, "");
        db.endArray();

        db.endArray(); // A

        std::vector<double> tv = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
        db.writeDataSlices("time", {7}, tv.data(), 1, "");
        db.close();
    }

    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        check("READ A/0/B size", db.getDynamicAOSSize("A/0/B"), 5);
        check("READ A/1/B size", db.getDynamicAOSSize("A/1/B"), 2);
        // Per-path counters: the row itself, not just its AoS.
        check("READ A/0/B/signal counter", db.getDynamicAOSSize("A/0/B/signal"), 3);
        check("READ A/0/B/signal_2 counter", db.getDynamicAOSSize("A/0/B/signal_2"), 5);
        // Bulk row (7 values in a single row) = one time step.
        check("READ time counter", db.getDynamicAOSSize("time"), 1);
        auto shape = db.getAOSShape("A/0/B");
        if (shape.empty() || shape.front() != 5) {
            std::cout << "  FAIL: getAOSShape(A/0/B) front = "
                      << (shape.empty() ? 0 : shape.front()) << ", expected 5\n";
            ++failures;
        }
    }

    // APPEND must see the same table, and continue from it: two more slices land at
    // time indices 5 and 6, taking A/0/B to 7.
    {
        PanzerDB db(filename, PanzerDB::OpenMode::APPEND);
        check("APPEND A/0/B size", db.getDynamicAOSSize("A/0/B"), 5);
        db.beginArray("A", 2);
        db.beginArray("B", "time");
        std::vector<double> more = {300.0, 301.0};
        db.writeDataSlices("signal", {1}, more.data(), 2, "");
        db.endArray();
        db.endArray();
        db.close();
    }
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        check("A/0/B size after append", db.getDynamicAOSSize("A/0/B"), 7);
        uint64_t ndim = 0, shape[6] = {0};
        double* out = nullptr;
        const int rc = db.readDataByIndex("A/0/B/signal", 6, &ndim, shape, &out);
        if (rc != 0 || !out || out[0] != 301.0) {
            std::cout << "  FAIL: appended slice 6 not readable at its own index (rc=" << rc
                      << ", value=" << (out ? out[0] : 0.0) << ", expected 301)\n";
            ++failures;
        }
        delete[] out;
    }

    std::filesystem::remove(filename);
    if (failures) { std::cout << failures << " FAILURE(S)\n"; return 1; }
    std::cout << "ALL PASSED\n";
    return 0;
}
