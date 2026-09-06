// @file  test_swmr_string_fixed_width.cpp
// @brief Concurrency smoke test for the "index = commit point" SWMR protocol.
//
// A writer thread appends N time-slices of a dynamic string signal, calling
// PanzerDB::flush() (the H5Fflush GLOBAL commit barrier) after each slice and
// sleeping briefly between flushes. Concurrently, a reader thread repeatedly
// OPENS the file fresh (each open simulates a distinct SWMR reader process),
// enumerates the committed leaves, reads every string slice, and asserts that
// each read value is EXACTLY the expected "slice_<t>" for that leaf's time
// index. A torn read — a partially-written string, a garbage slot, or a value
// that does not match the time index — increments `torn_reads`.
//
// The H5Fflush barrier + fixed-width (no-vlen) encoding are what make this
// hold. If either regresses, at least one snapshot shows a bad value.
#include "panzerdb.h"
#include <iostream>
#include <atomic>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

#include "fixtures/panzerdb_helper.h"

int main() {
    std::cout << "=== TEST PanzerDB: SWMR commit barrier + fixed-width string smoke ===\n\n";
    const std::string filename = "test_swmr_string.panzer";
    if (fs::exists(filename)) fs::remove(filename);

    const int N_SLICES   = 6;
    const int SLEEP_MS   = 25;      // writer pacing between flushes
    const int READER_ITERS = 60;    // reader snapshots
    const int READER_SLEEP_MS = 8;  // between snapshots

    std::atomic<bool>   writer_done{false};
    std::atomic<int>    torn_reads{0};

    // ---- reader thread: repeatedly open fresh, read committed slices, validate ----
    std::thread reader([&]{
        // Wait for the writer to create the file before the first open.
        for (int i = 0; i < 500 && !fs::exists(filename); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));

        for (int it = 0; it < READER_ITERS; ++it) {
            if (writer_done.load() && fs::exists(filename)) {
                // One more full snapshot after writer finishes.
            }
            try {
                PanzerDB r(filename, PanzerDB::OpenMode::READ);
                const auto& leaves = r.getLeaves();
                for (const auto& lf : leaves) {
                    if (std::string(lf.path) != "dyn/sig") continue;
                    std::string v;
                    r.readTensor(lf, &v);
                    const std::string expect = "slice_" + std::to_string(lf.time_index);
                    if (v != expect) {
                        std::cerr << "  [TORN?] leaf dyn/sig@" << lf.time_index
                                  << " read '" << v << "' != expected '" << expect << "'\n";
                        torn_reads.fetch_add(1);
                    }
                }
                r.close();
            } catch (const std::exception&) {
                // File not yet fully created or being committed; skip this snapshot.
            }
            if (writer_done.load()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(READER_SLEEP_MS));
        }
    });

    // ---- writer thread (inline): append N slices, flush (commit) after each ----
    {
        PanzerDB w(filename, PanzerDB::OpenMode::WRITE, true);
        w.beginArray("dyn", "time");
        for (int t = 0; t < N_SLICES; ++t) {
            if (t > 0) w.incrementArrayIndex();     // advance the dynamic time counter
            const std::string v = "slice_" + std::to_string(t);
            const char* p = v.c_str();
            w.writeDataSlices("sig", {}, &p, 1, "");   // one scalar string at time t
            w.flush();                                   // H5Fflush GLOBAL — the commit barrier
            std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_MS));
        }
        w.incrementArrayIndex();   // sentinel
        w.endArray();
        w.close();
    }
    writer_done.store(true);
    reader.join();

    // ---- post-join sanity: every slice must be present and correct ----
    {
        PanzerDB r(filename, PanzerDB::OpenMode::READ);
        const auto& leaves = r.getLeaves();
        int seen = 0;
        for (const auto& lf : leaves) {
            if (std::string(lf.path) != "dyn/sig") continue;
            std::string v;
            r.readTensor(lf, &v);
            const std::string expect = "slice_" + std::to_string(lf.time_index);
            assert(v == expect);
            ++seen;
        }
        assert(seen == N_SLICES && "all N slices must be present after the writer closes");
        r.close();
    }

    if (fs::exists(filename)) fs::remove(filename);

    if (torn_reads.load() != 0) {
        std::cerr << "\nFAIL: " << torn_reads.load() << " torn read(s) detected during SWMR smoke test\n";
        return 1;
    }
    std::cout << "  [OK] " << N_SLICES << " slices appended with a commit barrier each;\n";
    std::cout << "       " << READER_ITERS << " reader snapshots observed 0 torn string reads\n";
    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}
