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
// WHAT AN OPEN FAILURE DOES — AND DOES NOT — MEAN
// -----------------------------------------------
// Two windows make an open fail legitimately, and both were measured rather than
// assumed. (1) Between H5Fcreate and the writer's first H5Fflush the file holds a
// 48-byte superblock that already promises the aligned 2 MB layout, so the open
// fails with "truncated file: eof = 48 ... stored_eof = 2097152". There is no way
// to publish that skeleton earlier: PanzerDB::flush() early-returns while its
// buffers are empty. (2) HDF5 >= 1.14 puts a POSIX record lock and a "being
// written" flag on the file, so a second process is refused outright
// ("unable to lock file, errno = 11", then "file is already open for write")
// unless the engine stamps SWMR_WRITE at creation and opens readers with
// SWMR_READ and locking off — see the OpenMode branches of the PanzerDB
// constructor. Measured on this node: without that pair a reader only ever sees
// the writer's state as of close(), so it confirms all N slices in one snapshot
// and the commit barrier is never exercised (which is how this test used to pass
// for the wrong reason). Inside those windows an open attempt is EXPECTED to
// fail, and the attempt itself is the readiness probe: it is counted, not read as
// a defect — and HDF5's error stack is captured into one short line instead of
// dumped, otherwise a green run reads like a crash.
//
// A failure after the file HAS opened is a different animal: the reader is
// looking at a committed state, so a value mismatch or a throw from
// getLeaves()/readTensor() is a hard failure. One catch used to wrap the whole
// snapshot and silently turned those into "retry next tick" — which is exactly
// how this test could pass while its own log said otherwise.
//
// Green is only meaningful if the reader watched the commits arrive one at a
// time. On a close-to-open filesystem (AFS) it is handed the finished file in a
// single step, all N slices confirm, and nothing about the barrier was proven —
// so having seen at least one partial commit state is asserted too.
//
// A torn or garbled string — or a reader crash — means the fixed-width
// (no-vlen) string encoding or the commit barrier regressed.
//
// Reader exit codes: 0 ok; 1 a value did not match; 2 ran out of slices before
// the deadline; 3 a read threw on a file that opened cleanly; 4 never saw a
// partially committed file (the barrier was not observable).
#include "panzerdb.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
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
// Hdf5ErrorCapture — HDF5's own error printer, redirected into a string instead
// of stderr. Installed ONLY around an open attempt (see run_reader); the
// auto-printer stays active everywhere else so a genuine failure still dumps.
// ---------------------------------------------------------------------------
class Hdf5ErrorCapture {
public:
    std::string reason;   // what HDF5 would have dumped, as one line

    Hdf5ErrorCapture() {
        H5Eget_auto2(H5E_DEFAULT, &old_func_, &old_data_);
        H5Eset_auto2(H5E_DEFAULT, &Hdf5ErrorCapture::record, this);
    }
    ~Hdf5ErrorCapture() {
        H5Eset_auto2(H5E_DEFAULT, old_func_, old_data_);
        // A rejected open leaves its frames on the default stack, where the next
        // unrelated HDF5 error would drag them back into the log.
        H5Eclear2(H5E_DEFAULT);
    }

private:
    static herr_t record(hid_t estack, void* ctx) {
        H5Ewalk2(estack, H5E_WALK_UPWARD, &Hdf5ErrorCapture::innermost_frame, ctx);
        return 0;
    }
    // WALK_UPWARD starts at the most specific frame — "unable to locate root group
    // signature" rather than the "unable to open file" wrapper the dump starts with.
    static herr_t innermost_frame(unsigned, const H5E_error2_t* err, void* ctx) {
        auto* self = static_cast<Hdf5ErrorCapture*>(ctx);
        if (self->reason.empty() && err->desc && *err->desc) self->reason = err->desc;
        return 0;
    }

    H5E_auto2_t old_func_ = nullptr;
    void* old_data_ = nullptr;
};

// ---------------------------------------------------------------------------
// Reader process: open fresh, read committed slices, validate exact values.
// Returns 0 on success, 1 on a torn value, 2 if the writer never committed all
// slices within the deadline, 3 if a read threw on a file that opened cleanly,
// 4 if the reader never saw a partially committed file.
// ---------------------------------------------------------------------------
int run_reader(const std::string& filename, int N, int snap_sleep_ms, int deadline_s) {
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::seconds(deadline_s);

    // Cheap guard so we do not hammer a path that does not exist yet. It is NOT
    // the readiness signal: the file is on the filesystem long before it is an
    // openable HDF5 file, so readiness is decided by the open attempt itself.
    for (int i = 0; i < 2000 && !fs::exists(filename); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    if (!fs::exists(filename)) {
        std::cerr << "FAIL: reader — file never appeared: " << filename << "\n";
        return 2;
    }

    // time_indexes of "dyn/sig" slices we have read back with the correct value.
    std::set<uint64_t> confirmed;
    // How many slices each successful snapshot held, and whether any of them held
    // a partial commit: that is what makes the run a test of the barrier at all.
    std::set<int> snapshot_sizes;
    bool saw_partial_commit = false;
    int snapshots = 0;
    int rejects_before_first_commit = 0;
    int rejects_after_opening = 0;

    while (Clock::now() < deadline) {
        ++snapshots;

        // Phase 1 — open. Rejected here means "not published yet", which is the
        // state a real SWMR reader is expected to retry on.
        std::unique_ptr<PanzerDB> r;
        std::string reject_reason;
        {
            Hdf5ErrorCapture cap;
            try {
                r = std::make_unique<PanzerDB>(filename, PanzerDB::OpenMode::READ);
            } catch (const std::exception&) {
                reject_reason = cap.reason;
            }
        }
        if (!r) {
            const bool before_first_commit = confirmed.empty();
            const int n = before_first_commit ? ++rejects_before_first_commit
                                              : ++rejects_after_opening;
            if (n <= 2)
                std::cerr << "  [retry] open rejected #" << n
                          << (before_first_commit ? " before the first commit"
                                                  : " AFTER the file had opened — it stopped being openable")
                          << (reject_reason.empty() ? std::string() : ", HDF5: " + reject_reason)
                          << "\n";
            if (static_cast<int>(confirmed.size()) >= N) break;
            continue;   // no sleep: the failed open already burned the round trip
        }

        // Phase 2 — the file opened, so we are looking at a committed state.
        // Anything thrown from here is a read failure, not a timing artefact.
        try {
            const auto& leaves = r->getLeaves();
            int slices_in_snapshot = 0;
            for (const auto& lf : leaves) {
                if (std::string(lf.path) != "dyn/sig") continue;
                std::string v;
                r->readTensor(lf, &v);
                const std::string expect = "slice_" + std::to_string(lf.time_index);
                if (v != expect) {
                    std::cerr << "  [TORN] dyn/sig@" << lf.time_index
                              << " read '" << v << "' != expected '" << expect << "'\n";
                    r->close();
                    return 1;
                }
                confirmed.insert(lf.time_index);
                ++slices_in_snapshot;
            }
            r->close();
            snapshot_sizes.insert(slices_in_snapshot);
            if (slices_in_snapshot > 0 && slices_in_snapshot < N) saw_partial_commit = true;
        } catch (const std::exception& e) {
            std::cerr << "  [TORN] snapshot threw on a file that opened cleanly: "
                      << e.what() << "\n";
            return 3;
        }

        if (static_cast<int>(confirmed.size()) >= N) break;  // every slice confirmed
        std::this_thread::sleep_for(std::chrono::milliseconds(snap_sleep_ms));
    }

    const std::string tally = std::to_string(snapshots) + " snapshots, " +
        std::to_string(rejects_before_first_commit) + " open reject(s) before the first commit, " +
        std::to_string(rejects_after_opening) + " after the file had opened";

    if (static_cast<int>(confirmed.size()) < N) {
        std::cerr << "FAIL: reader confirmed only " << confirmed.size() << "/" << N
                  << " slices before the deadline (" << tally << ")\n";
        return 2;
    }

    // Final confirming pass: re-open and verify all N are present and correct.
    // The writer may still be finishing its last commit, so the two phases stay.
    std::unique_ptr<PanzerDB> r;
    std::string reject_reason;
    {
        Hdf5ErrorCapture cap;
        try {
            r = std::make_unique<PanzerDB>(filename, PanzerDB::OpenMode::READ);
        } catch (const std::exception&) {
            reject_reason = cap.reason;
        }
    }
    if (!r) {
        std::cerr << "FAIL: reader final pass could not open a file that held all "
                  << N << " slices ("
                  << (reject_reason.empty() ? std::string("no HDF5 reason captured") : reject_reason)
                  << ")\n";
        return 2;
    }
    try {
        const auto& leaves = r->getLeaves();
        int count = 0;
        for (const auto& lf : leaves) {
            if (std::string(lf.path) != "dyn/sig") continue;
            std::string v;
            r->readTensor(lf, &v);
            if (v != "slice_" + std::to_string(lf.time_index)) {
                std::cerr << "  [TORN] final pass dyn/sig@" << lf.time_index
                          << " read '" << v << "'\n";
                r->close();
                return 1;
            }
            ++count;
        }
        r->close();
        if (count != N) {
            std::cerr << "FAIL: reader final count of dyn/sig = " << count
                      << " != " << N << "\n";
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "  [TORN] reader final pass threw: " << e.what() << "\n";
        return 3;
    }

    if (!saw_partial_commit) {
        std::string sizes;
        for (const int k : snapshot_sizes) sizes += (sizes.empty() ? "" : ", ") + std::to_string(k);
        std::cerr << "FAIL: the reader never saw a partially committed file — every snapshot it "
                     "opened held 0 or all " << N << " slices (sizes seen: "
                  << (sizes.empty() ? "none" : sizes) << "; " << tally << "). The file reached it "
                     "whole in one step, so this run proves nothing about the commit barrier: the "
                     "reader was handed the writer's state at close(), not a sequence of commits.\n";
        return 4;
    }

    std::cout << "  [reader] OK: confirmed all " << N
              << " committed slices with 0 torn string reads; " << tally << "; "
              << snapshot_sizes.size() << " distinct commit states observed\n";
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
