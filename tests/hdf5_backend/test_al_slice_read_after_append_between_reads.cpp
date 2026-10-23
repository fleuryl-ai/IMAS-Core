// @file  test_al_slice_read_after_append_between_reads.cpp
// @brief Guards a reader session that already read a pulse against rows appended
//        after that read: the read session must notice the table grew, not keep
//        serving the rows it decoded before the append.
//
// A read session in the v2 backend keeps one PanzerDB engine (and the path index
// built over it) for as long as the reader lives, moving it onto the group id the
// AL opens for each operation. That is only safe while what the engine decoded is
// still the whole table: an append adds /index rows, and a session that missed
// them would answer a slice read with "not found" -- or with the wrong values --
// for a slice that is in the file.
//
// The phases therefore alternate read / write / read on the SAME backend object,
// which is what keeps one read session alive across the append. A test that wrote
// first and read afterwards (as the other append tests do) would build the read
// session after the append and never exercise this.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

const std::string URI = "imas:hdf5?path=./test_db_slice_read_after_append_between_reads";

const int POINTS = 10;          // radial points of one signal_1d slice
const int INITIAL_STEPS = 3;    // slices written up front (t = 0.1, 0.2, 0.3)

// Value of point x of the slice written at time index t_idx.
double expected_value(int t_idx, int x) { return 100.0 + t_idx * 10.0 + x; }

// Maps a time to its time index: the time base is 0.1, 0.2, ... so t*10 is the index.
int time_index_of(double t) { return (int)(t * 10.0 + 0.5); }

// Writes the slices [from_idx, to_idx] of one core_profiles pulse.
static void write_slices(HDF5Backend &backend, DataEntryContext &ctx,
                         int from_idx, int to_idx, bool first_pulse) {
    backend.openPulse(&ctx, first_pulse ? FORCE_CREATE_PULSE : OPEN_PULSE);
    const double t_first = 0.1 * from_idx;
    OperationContext opCtx(&ctx, "core_profiles", WRITE_OP, alconst::slice_op, t_first,
                           alconst::closest_interp);
    backend.beginAction(&opCtx);

    if (first_pulse) {
        int homogeneous_time = 1;
        backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time,
                          alconst::integer_data, 0, nullptr);
    }
    // The time base of this operation: the append path grows it, the create path sets it.
    const int n_steps = to_idx - from_idx + 1;
    std::vector<double> times(n_steps);
    for (int i = 0; i < n_steps; ++i) times[i] = 0.1 * (from_idx + i);
    int time_dim = 1;
    int time_size[] = {n_steps};
    backend.writeData(&opCtx, "time", "time", times.data(), alconst::double_data,
                      time_dim, time_size);

    int profiles_size = n_steps;
    ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
    backend.beginArraystructAction(&profilesCtx, &profiles_size);
    for (int i = 0; i < n_steps; ++i) {
        const int t_idx = from_idx + i;
        std::vector<double> val(POINTS);
        for (int x = 0; x < POINTS; ++x) val[x] = expected_value(t_idx, x);
        int sig_dim = 1;
        int sig_size[] = {POINTS};
        backend.writeData(&profilesCtx, "signal_1d", "time", val.data(),
                          alconst::double_data, sig_dim, sig_size);
        if (i < n_steps - 1) profilesCtx.nextIndex(1);
    }
    backend.endAction(&profilesCtx);
    backend.endAction(&opCtx);
    backend.closePulse(&ctx, first_pulse ? FORCE_CREATE_PULSE : OPEN_PULSE);
}

// Reads one slice and compares it against what the slice at that time was filled with.
static bool read_slice(HDF5Backend &backend, DataEntryContext &ctx, double target_time) {
    backend.openPulse(&ctx, OPEN_PULSE);
    OperationContext opCtx(&ctx, "core_profiles", READ_OP, alconst::slice_op, target_time,
                           alconst::closest_interp);
    backend.beginAction(&opCtx);

    const int t_idx = time_index_of(target_time);
    ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
    int profiles_size = 0;
    backend.beginArraystructAction(&profilesCtx, &profiles_size);
    if (profiles_size != 1) {
        std::cerr << "profiles_1d holds " << profiles_size << " instance(s) at t="
                  << target_time << ", expected 1" << std::endl;
        return false;
    }

    void *data = nullptr;
    int type = alconst::double_data;
    int dim = 0;
    int size[H5S_MAX_RANK];
    backend.readData(&profilesCtx, "signal_1d", "time", &data, &type, &dim, size);
    if (!data) {
        std::cerr << "signal_1d read returned nothing at t=" << target_time
                  << " -- the append is invisible to this read session" << std::endl;
        return false;
    }
    if (dim != 1 || size[0] != POINTS) {
        std::cerr << "signal_1d at t=" << target_time << " came back with dim=" << dim
                  << " size[0]=" << size[0] << ", expected 1/" << POINTS << std::endl;
        free(data);
        return false;
    }
    const double *vals = static_cast<const double *>(data);
    bool ok = true;
    for (int x = 0; x < POINTS; ++x) {
        const double want = expected_value(t_idx, x);
        if (std::abs(vals[x] - want) > 1e-9) {
            std::cerr << "signal_1d[" << x << "] at t=" << target_time << " = " << vals[x]
                      << ", expected " << want << std::endl;
            ok = false;
            break;
        }
    }
    free(data);
    backend.endAction(&profilesCtx);
    backend.endAction(&opCtx);
    backend.closePulse(&ctx, OPEN_PULSE);
    return ok;
}

int main() {
    try {
        std::cout << "\n=== Test: a read session sees slices appended after it read ===\n";
        // This gate pins the v2 session engine; the target version comes from the
        // environment, so without pinning it an ambient BACKEND_DEFAULT_WRITE_VERSION=1.0
        // moves the whole test onto the v1 path, where there is no engine to invalidate.
        setenv("BACKEND_DEFAULT_WRITE_VERSION", "2.0", 1);
        if (fs::exists("./test_db_slice_read_after_append_between_reads"))
            fs::remove_all("./test_db_slice_read_after_append_between_reads");

        // ONE backend for every phase: the read session built in phase 2 is the one
        // that has to notice the append of phase 3.
        HDF5Backend backend;

        std::cout << "--- Phase 1: write t = 0.0, 0.1, 0.2 ---\n";
        {
            DataEntryContext ctx(URI);
            write_slices(backend, ctx, 0, INITIAL_STEPS - 1, true);
        }

        std::cout << "--- Phase 2: read t = 0.2 (this decode is what phase 3 must invalidate) ---\n";
        {
            DataEntryContext ctx(URI);
            if (!read_slice(backend, ctx, 0.2)) return 1;
        }

        std::cout << "--- Phase 3: append t = 0.3 ---\n";
        {
            DataEntryContext ctx(URI);
            write_slices(backend, ctx, INITIAL_STEPS, INITIAL_STEPS, false);
        }

        // The slice written after the read session was built. A session still holding
        // the pre-append table has no index 3: closest_interp would answer with the
        // last slice it knows (120), which is exactly what this check rejects.
        std::cout << "--- Phase 4: read the appended slice t = 0.3 ---\n";
        {
            DataEntryContext ctx(URI);
            if (!read_slice(backend, ctx, 0.3)) return 1;
        }

        // And the slices that were already there must still read, so the invalidation
        // cannot be "throw the session away and lose what was fine".
        std::cout << "--- Phase 5: read t = 0.1 again ---\n";
        {
            DataEntryContext ctx(URI);
            if (!read_slice(backend, ctx, 0.1)) return 1;
        }

        std::cout << "[OK] read session saw the appended slice.\n";
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    if (fs::exists("./test_db_slice_read_after_append_between_reads"))
        fs::remove_all("./test_db_slice_read_after_append_between_reads");
    return 0;
}
