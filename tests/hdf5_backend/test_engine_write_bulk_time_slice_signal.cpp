// @file  test_engine_write_bulk_time_slice_signal.cpp
// @brief Regression test for the "bulk timebase + per-slice signal" write pattern.
//        Verifies that a signal written slice-by-slice AFTER a bulk timebase
//        (n_slices=N in one call for the time field) is placed at time_index 0..N-1,
//        not shifted onto time_index N-1..2N-2. Covers both numeric and string
//        signals (the user's reported case is the string path).
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    const std::string filename = "test_bulk_time_slice_signal.pane";
    if (fs::exists(filename)) fs::remove(filename);

    const int N = 5;
    std::vector<double> times(N); for (int k = 0; k < N; ++k) times[k] = 0.1 * k;
    std::vector<double> sig(N);   for (int k = 0; k < N; ++k) sig[k]   = 100.0 + k;
    std::vector<std::string> str(N); for (int k = 0; k < N; ++k) str[k] = "str_val_" + std::to_string(k);

    // ------------------------------------------------------------------
    // Phase 1: BULK time write (1 call, N slices), then PER-SLICE signals
    // (numeric + string), each N calls of n_slices=1.
    // ------------------------------------------------------------------
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        db.beginArray("A", 1);
        db.beginArray("B", "time");       // dynamic AOS, timebase field = "time"

        // BULK: all N time points in ONE call.
        db.writeDataSlices("time", {1}, times.data(), N, "");

        // PER-SLICE numeric signal.
        for (int k = 0; k < N; ++k)
            db.writeDataSlices("signal", {1}, &sig[k], 1, "time");

        // PER-SLICE string signal (1 scalar string per call).
        for (int k = 0; k < N; ++k) {
            const char* s = str[k].c_str();
            db.writeDataSlices("str_signal", {1}, &s, 1, "time");
        }

        db.endArray();   // B
        db.endArray();   // A
        db.close();
    }

    // ------------------------------------------------------------------
    // Phase 2: read back every slice by time_index k and check the value.
    // Each of time_index 0..N-1 must exist and hold the correctly-written value.
    // (Before the fix, the numeric and string signals were shifted to
    //  time_index N-1..2N-2 and reads at 0..N-2 would fail / return wrong data.)
    // ------------------------------------------------------------------
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();

        for (int k = 0; k < N; ++k) {
            // numeric signal
            const PanzerDB::Leaf* leaf_sig = nullptr;
            for (const auto& l : leaves) {
                if (std::string(l.path) == "A/0/B/signal" && l.time_index == (uint64_t)k) { leaf_sig = &l; break; }
            }
            assert(leaf_sig && "numeric signal leaf at time_index k NOT FOUND (was it shifted?)");
            double val = 0;
            db.readTensor(*leaf_sig, &val);
            assert(std::abs(val - (100.0 + k)) < 1e-9 && "numeric signal value at time_index k wrong");

            // string signal
            const PanzerDB::Leaf* leaf_str = nullptr;
            for (const auto& l : leaves) {
                if (std::string(l.path) == "A/0/B/str_signal" && l.time_index == (uint64_t)k) { leaf_str = &l; break; }
            }
            assert(leaf_str && "string signal leaf at time_index k NOT FOUND (was it shifted?)");
            std::string sval;
            db.readTensor(*leaf_str, &sval);
            assert(sval == str[k] && "string signal value at time_index k wrong");
        }
        db.close();
    }

    if (fs::exists(filename)) fs::remove(filename);
    std::cout << "OK: bulk timebase + per-slice numeric & string signal -> indices 0..N-1 (N=" << N << ")\n";
    return 0;
}
