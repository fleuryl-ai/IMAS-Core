// @file  test_swmr_string_fixed_width.cpp
// @brief SWMR smoke test for the "index = commit point" protocol, driven by
//        TWO separate OS processes — a real shared-FS writer/reader pair.
//
// WHY TWO PROCESSES (not threads)
// -------------------------------
// The previous version ran the writer and the reader as two threads in ONE
// process. That made the reader thread's H5Dget() race the writer's
// flush()/H5Dwrite() against HDF5's per-process file cache / Virtual Object
// Layer, which is not safe to share across threads that touch the same file —
// it produced HDF5-core segfaults (e.g. in H5P_peek) that were NOT defects in
// PanzerDB. Real SWMR readers are *other processes*, so the honest simulation
// is two processes: after fork() each side owns its own HDF5 instance and file
// cache and the only thing they share is the byte stream on disk — exactly the
// single-writer / multi-reader contract under test.
//
// PROTOCOL UNDER TEST
// -------------------
// - Writer (parent): appends N time-slices of a dynamic string signal "sig",
//   calling PanzerDB::flush() (the H5Fflush GLOBAL commit barrier, with the
//   /index row written LAST inside flush()) after each slice, so each slice is
//   one self-contained commit point.
// - Reader (child): repeatedly opens the file fresh, enumerates the committed
//   leaves, reads every committed "dyn/sig" slice and asserts each value is
//   EXACTLY "slice_<time_index>". A torn read (partial / garbage / value that
//   does not match the time index) fails the test. Once it has read all N
//   slices correctly it does a final confirming pass and exits 0.
//
// A torn or garbled string — or a reader crash — means the fixed-width
// (no-vlen) string encoding or the commit barrier regressed.
#include "panzerdb.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <thread>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

#include "fixtures/panzerdb_helper.h"

namespace {

// ---------------------------------------------------------------------------
// Reader process: open fresh, read committed slices, validate exact values.
// Returns 0 on success, 1 on a torn read, 2 if the writer never committed all
// slices within the deadline (or a read failed to complete).
// ---------------------------------------------------------------------------
int run_reader(const std::string& filename, int N, int snap_sleep_ms, int deadline_s) {
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::seconds(deadline_s);

    // Wait for the writer process to create the file before the first open.
    for (int i = 0; i < 2000 && !fs::exists(filename); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    if (!fs::exists(filename)) {
        std::cerr << "FAIL: reader — file never appeared: " << filename << "\n";
        return 2;
    }

    // time_indexes of "dyn/sig" slices we have read back with the correct value.
    std::set<uint64_t> confirmed;

    while (Clock::now() < deadline) {
        bool snapshot_completed = true;
        try {
            PanzerDB r(filename, PanzerDB::OpenMode::READ);
            const auto& leaves = r.getLeaves();
            for (const auto& lf : leaves) {
                if (std::string(lf.path) != "dyn/sig") continue;
                std::string v;
                r.readTensor(lf, &v);
                const std::string expect = "slice_" + std::to_string(lf.time_index);
                if (v != expect) {
                    std::cerr << "  [TORN] dyn/sig@" << lf.time_index
                              << " read '" << v << "' != expected '" << expect << "'\n";
                    r.close();
                    return 1;
                }
                confirmed.insert(lf.time_index);
            }
            r.close();
        } catch (const std::exception&) {
            // Mid-commit / file still being created; retry next tick.
            snapshot_completed = false;
        }

        if (static_cast<int>(confirmed.size()) >= N) break;  // every slice confirmed
        if (snapshot_completed)
            std::this_thread::sleep_for(std::chrono::milliseconds(snap_sleep_ms));
    }

    if (static_cast<int>(confirmed.size()) < N) {
        std::cerr << "FAIL: reader confirmed only " << confirmed.size() << "/" << N
                  << " slices before the deadline\n";
        return 2;
    }

    // Final confirming pass: re-open and verify all N are present and correct.
    try {
        PanzerDB r(filename, PanzerDB::OpenMode::READ);
        const auto& leaves = r.getLeaves();
        int count = 0;
        for (const auto& lf : leaves) {
            if (std::string(lf.path) != "dyn/sig") continue;
            std::string v;
            r.readTensor(lf, &v);
            if (v != "slice_" + std::to_string(lf.time_index)) {
                std::cerr << "  [TORN] final pass dyn/sig@" << lf.time_index
                          << " read '" << v << "'\n";
                r.close();
                return 1;
            }
            ++count;
        }
        r.close();
        if (count != N) {
            std::cerr << "FAIL: reader final count of dyn/sig = " << count
                      << " != " << N << "\n";
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: reader final pass threw: " << e.what() << "\n";
        return 2;
    }

    std::cout << "  [reader] OK: confirmed all " << N
              << " committed slices with 0 torn string reads\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << "=== TEST PanzerDB: SWMR commit barrier + fixed-width string smoke (2-process) ===\n";

    const int N = (argc > 1) ? std::stoi(argv[1]) : 6;
    const std::string filename = (argc > 2) ? argv[2] : "test_swmr_string.panzer";
    if (fs::exists(filename)) fs::remove(filename);

    const int WRITER_SLEEP_MS = 25;       // pacing between commit barriers
    const int READER_SNAP_MS  = 8;        // reader pacing between snapshots
    const int READER_DEADLINE_S = 20;     // reader gives up if the writer stalls

    // Flush stdio BEFORE forking so parent/child never share a text buffer.
    std::cout.flush();
    std::cerr.flush();

    const pid_t child = fork();
    if (child < 0) {
        std::cerr << "FAIL: fork() failed\n";
        return 1;
    }

    if (child == 0) {
        // ---- CHILD: reader process (own HDF5 instance over the shared file) ----
        const int rc = run_reader(filename, N, READER_SNAP_MS, READER_DEADLINE_S);
        // Parent owns file cleanup; the child must not delete the file.
        std::exit(rc);
    }

    // ---- PARENT: writer process (and coordinator) ------------------------------
    int writer_rc = 0;
    try {
        PanzerDB w(filename, PanzerDB::OpenMode::WRITE, true);
        w.beginArray("dyn", "time");
        for (int t = 0; t < N; ++t) {
            if (t > 0) w.incrementArrayIndex();   // advance the dynamic time counter
            const std::string v = "slice_" + std::to_string(t);
            const char* p = v.c_str();
            w.writeDataSlices("sig", {}, &p, 1, "");   // one scalar string at time t
            w.flush();                                   // H5Fflush GLOBAL — commit barrier
            std::this_thread::sleep_for(std::chrono::milliseconds(WRITER_SLEEP_MS));
        }
        w.incrementArrayIndex();   // sentinel
        w.endArray();
        w.close();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: writer threw: " << e.what() << "\n";
        writer_rc = 1;
    }

    // Reap the reader child and interpret its exit code.
    int child_status = 0;
    const pid_t waited = waitpid(child, &child_status, 0);
    if (waited < 0) {
        std::cerr << "FAIL: waitpid() on reader child failed\n";
        if (fs::exists(filename)) fs::remove(filename);
        return 1;
    }
    bool child_ok = WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0;
    if (WIFSIGNALED(child_status))
        std::cerr << "FAIL: reader child killed by signal " << WTERMSIG(child_status) << "\n";
    else if (!child_ok)
        std::cerr << "FAIL: reader child exited with code "
                  << (WIFEXITED(child_status) ? WEXITSTATUS(child_status) : -1) << "\n";

    // Parent-side sanity read (solo, no concurrent reader now that the child is reaped).
    int parent_reread_rc = 0;
    try {
        PanzerDB r(filename, PanzerDB::OpenMode::READ);
        const auto& leaves = r.getLeaves();
        int seen = 0;
        for (const auto& lf : leaves) {
            if (std::string(lf.path) != "dyn/sig") continue;
            std::string v;
            r.readTensor(lf, &v);
            if (v != "slice_" + std::to_string(lf.time_index)) {
                std::cerr << "FAIL: parent re-read dyn/sig@" << lf.time_index
                          << " '" << v << "' != expected\n";
                parent_reread_rc = 1;
            }
            ++seen;
        }
        r.close();
        if (seen != N) {
            std::cerr << "FAIL: parent saw " << seen << "/" << N << " dyn/sig slices\n";
            parent_reread_rc = 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: parent re-read threw: " << e.what() << "\n";
        parent_reread_rc = 1;
    }

    if (fs::exists(filename)) fs::remove(filename);

    if (writer_rc == 0 && child_ok && parent_reread_rc == 0) {
        std::cout << "  [OK] " << N << " slices committed, one commit barrier each;\n";
        std::cout << "       a SEPARATE reader process read them all with 0 torn string reads\n";
        std::cout << "\nALL TESTS PASSED!\n";
        return 0;
    }

    std::cerr << "\nFAIL: writer_rc=" << writer_rc
              << " child_ok=" << (child_ok ? 1 : 0)
              << " parent_reread_rc=" << parent_reread_rc << "\n";
    return 1;
}
