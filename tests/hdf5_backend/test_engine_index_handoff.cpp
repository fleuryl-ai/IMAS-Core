// @file  test_engine_index_handoff.cpp
// @brief Pins the index handoff between PanzerDB engines opened on one file: the decoded
//        /index outlives the engine that read it, and the next engine on that file resumes
//        from it instead of re-reading the whole table (see PanzerDB::adoptIndexHandoff).
//
// Reading the tables is what dominates engine opening, and a writer doing one SLICE_OP per
// time step builds one engine per step, so that cost was paid once per step. Reusing the
// decode is only correct while it is still the head of the on-disk table, so what this test
// pins is exactly the cases where that assumption could fail:
//   * rows written by an earlier engine stay visible AND readable after the state was
//     parked and taken back — including paths interned before the handoff, whose text now
//     lives in a block the current engine did not read;
//   * reopening without appending must not lose or duplicate rows;
//   * a WRITE (truncate) open must not bring rows back that its own delete removed;
//   * a state parked for one file must never be adopted by another file.
//
// Reuse is observed through leaf_cache_generation(): it is bumped by every cache rebuild,
// so an engine that took the parked state over reports at least the previous engine's
// value, while an engine that decodes the table itself starts again at 1.
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

// Generation reported by the last engine that decoded an index. An engine that appended
// rows rebuilt the cache, so if it resumed from a parked state its generation is strictly
// above the previous engine's; one that appended nothing leaves it unchanged. Either way it
// never drops: an engine that decodes the table itself starts again at 1.
static uint64_t last_generation = 0;

static void check_chain_grows(const char* what, const PanzerDB& db) {
    const uint64_t gen = db.leaf_cache_generation();
    if (gen <= last_generation) {
        std::cout << "  FAIL: " << what << ": generation " << gen
                  << " does not continue the chain (last was " << last_generation << ")\n";
        ++failures;
    }
    last_generation = gen;
}

static void check_chain_kept(const char* what, const PanzerDB& db) {
    const uint64_t gen = db.leaf_cache_generation();
    if (gen < last_generation) {
        std::cout << "  FAIL: " << what << ": generation " << gen
                  << " restarts the chain (last was " << last_generation << ")\n";
        ++failures;
    }
    last_generation = gen;
}

static void check_fresh(const char* what, const PanzerDB& db) {
    check(what, db.leaf_cache_generation(), 1);
    last_generation = 1;
}

/// Reads one time slice of a one-value signal and checks the value it holds.
static void check_slice(PanzerDB& db, const std::string& path, int index, double want) {
    uint64_t ndim = 0, shape[6] = {0};
    double* out = nullptr;
    const int rc = db.readDataByIndex(path.c_str(), index, &ndim, shape, &out);
    if (rc != 0 || !out) {
        std::cout << "  FAIL: slice " << index << " of " << path << " not readable (rc=" << rc << ")\n";
        ++failures;
        return;
    }
    if (out[0] != want) {
        std::cout << "  FAIL: " << path << "[" << index << "] = " << out[0]
                  << ", expected " << want << "\n";
        ++failures;
    }
    delete[] out;
}

/// One engine per time step, like a SLICE_OP writer: check what it inherited, append one
/// slice of A/0/B/signal holding `value`, close.
static void append_slice(const std::string& filename, double value, uint64_t size_at_open) {
    PanzerDB db(filename, PanzerDB::OpenMode::APPEND);
    check_chain_grows("appending engine", db);
    check("A/0/B size seen at open", (uint64_t)db.getDynamicAOSSize("A/0/B"), size_at_open);
    db.beginArray("A", 1);
    db.beginArray("B", "time");
    std::vector<double> v = {value};
    db.writeDataSlices("signal", {1}, v.data(), 1, "");
    db.endArray();
    db.endArray();
    db.close();
}

int main() {
    const std::string f1 = "test_engine_index_handoff.panzer";
    const std::string f2 = "test_engine_index_handoff_b.panzer";
    std::filesystem::remove(f1);
    std::filesystem::remove(f2);
    std::cout << "=== TEST index handoff between engines on one file ===\n";

    // A(1) / A/0/B (dynamic AoS, "time") / signal: one slice per engine, so every slice
    // below is written by an engine reusing what the previous one decoded.
    {
        PanzerDB db(f1, PanzerDB::OpenMode::WRITE, true);
        db.beginArray("A", 1);
        db.beginArray("B", "time");
        std::vector<double> v = {100.0};
        db.writeDataSlices("signal", {1}, v.data(), 1, "");
        db.endArray();
        db.endArray();
        db.close();
    }
    for (int step = 1; step <= 4; ++step) {
        append_slice(f1, 100.0 + step, (uint64_t)step);
    }

    {
        PanzerDB db(f1, PanzerDB::OpenMode::READ);
        check_chain_kept("reading engine", db);
        check("A/0/B size after 5 engines", (uint64_t)db.getDynamicAOSSize("A/0/B"), 5);
        // Slices 0..2 were written by earlier engines: their paths were interned before
        // this engine read anything, so reading them proves the older path blocks stayed
        // reachable.
        for (int i = 0; i < 5; ++i) check_slice(db, "A/0/B/signal", i, 100.0 + i);
    }

    // Reopen without writing: the parked decode already covers the whole table.
    {
        PanzerDB db(f1, PanzerDB::OpenMode::READ);
        check_chain_kept("no-append reopen", db);
        check("A/0/B size after a no-op reopen", (uint64_t)db.getDynamicAOSSize("A/0/B"), 5);
        check_slice(db, "A/0/B/signal", 4, 104.0);
        check_slice(db, "A/0/B/signal", 0, 100.0);
    }

    // A truncate deletes /index: rows written before it must not come back, whether or
    // not this engine parked anything on its way out.
    {
        PanzerDB db(f1, PanzerDB::OpenMode::WRITE, true);
        std::vector<double> v = {7.0};
        db.writeDataSlices("other", {1}, v.data(), 1, "");
        db.close();
    }
    {
        PanzerDB db(f1, PanzerDB::OpenMode::READ);
        check("signal truncated away", (uint64_t)db.getDynamicAOSSize("A/0/B/signal"), 0);
        check("signal written after truncate", (uint64_t)db.getDynamicAOSSize("other"), 1);
        uint64_t ndim = 0, shape[6] = {0};
        double* out = nullptr;
        const int rc = db.readDataByIndex("A/0/B/signal", 0, &ndim, shape, &out);
        if (rc == 0 && out) {
            std::cout << "  FAIL: signal removed by the truncate is still readable\n";
            ++failures;
        }
        delete[] out;
    }

    // A parked state belongs to one file: another file must decode its own table.
    {
        PanzerDB db(f2, PanzerDB::OpenMode::WRITE, true);
        db.beginArray("A", 1);
        db.beginArray("B", "time");
        std::vector<double> v = {900.0};
        db.writeDataSlices("signal", {1}, v.data(), 1, "");
        db.endArray();
        db.endArray();
        db.close();
    }
    {
        PanzerDB db(f1, PanzerDB::OpenMode::READ);
        check("f1 decoded", (uint64_t)db.getDynamicAOSSize("other"), 1);
    }
    {
        PanzerDB db(f2, PanzerDB::OpenMode::READ);
        check_fresh("other file does not inherit", db);
        check_slice(db, "A/0/B/signal", 0, 900.0);
    }

    std::filesystem::remove(f1);
    std::filesystem::remove(f2);
    if (failures) { std::cout << failures << " FAILURE(S)\n"; return 1; }
    std::cout << "ALL PASSED\n";
    return 0;
}
