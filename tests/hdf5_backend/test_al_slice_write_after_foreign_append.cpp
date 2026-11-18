// @file  test_al_slice_write_after_foreign_append.cpp
// @brief Guards a slice-writing session whose engine is kept alive from one
//        putSlice() to the next, against rows another writer added meanwhile.
//
// The v2 write path keeps one APPEND engine for the whole slice session: rebuilding
// it per putSlice() cost an init() plus an H5Dclose over every table at each time
// step. Reusing it is only safe while this engine is the one that grew the tables —
// it must still know how many /index rows exist and where the raw data ends. If
// somebody else appended rows in between, that knowledge is stale and its next
// flush would place data over what the other writer wrote.
//
// So the session is written by TWO backends on purpose: the second one appends a
// slice in the middle of the first one's session. The first session must notice the
// foreign rows and rebuild, and the file must come out with both writers' slices
// intact.
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

const std::string URI = "imas:hdf5?path=./test_db_slice_write_after_foreign_append";

const int POINTS = 10;   // radial points of one signal_1d slice

// Value of point x of the slice written at time index t_idx.
double expected_value(int t_idx, int x) { return 100.0 + t_idx * 10.0 + x; }
// A signal only the first writer ever writes: it must land on its own time index.
double expected_extra(int t_idx, int x) { return 500.0 + t_idx * 10.0 + x; }

// Maps a time to its time index: the time base is 0.1, 0.2, ... so t*10 is the index.
int time_index_of(double t) { return (int)(t * 10.0 + 0.5); }

// Writes the slices [from_idx, to_idx] of one core_profiles pulse. The caller owns
// the pulse session (openPulse/closePulse), so a session can stay open while another
// backend writes to the same file.
static void write_slices(HDF5Backend &backend, DataEntryContext &ctx,
                         int from_idx, int to_idx, bool write_ids_properties,
                         bool write_extra) {
    const double t_first = 0.1 * from_idx;
    OperationContext opCtx(&ctx, "core_profiles", WRITE_OP, alconst::slice_op, t_first,
                           alconst::closest_interp);
    backend.beginAction(&opCtx);

    if (write_ids_properties) {
        int homogeneous_time = 1;
        backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time,
                          alconst::integer_data, 0, nullptr);
    }
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
        if (write_extra) {
            std::vector<double> extra(POINTS);
            for (int x = 0; x < POINTS; ++x) extra[x] = expected_extra(t_idx, x);
            backend.writeData(&profilesCtx, "extra_1d", "time", extra.data(),
                              alconst::double_data, sig_dim, sig_size);
        }
        if (i < n_steps - 1) profilesCtx.nextIndex(1);
    }
    backend.endAction(&profilesCtx);
    backend.endAction(&opCtx);
}

// Reads one slice and compares signal_1d (always) and extra_1d (when asked) against
// what was written there.
static bool read_slice(HDF5Backend &backend, DataEntryContext &ctx, double target_time,
                       bool check_extra) {
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

    bool ok = true;
    const auto check = [&](const char *name, double (*want)(int, int)) {
        void *data = nullptr;
        int type = alconst::double_data;
        int dim = 0;
        int size[H5S_MAX_RANK];
        backend.readData(&profilesCtx, name, "time", &data, &type, &dim, size);
        if (!data) {
            std::cerr << name << " read returned nothing at t=" << target_time << std::endl;
            ok = false;
            return;
        }
        if (dim != 1 || size[0] != POINTS) {
            std::cerr << name << " at t=" << target_time << " came back with dim=" << dim
                      << " size[0]=" << size[0] << ", expected 1/" << POINTS << std::endl;
            free(data);
            ok = false;
            return;
        }
        const double *vals = static_cast<const double *>(data);
        for (int x = 0; x < POINTS && ok; ++x) {
            const double w = want(t_idx, x);
            if (std::abs(vals[x] - w) > 1e-9) {
                std::cerr << name << "[" << x << "] at t=" << target_time << " = " << vals[x]
                          << ", expected " << w << std::endl;
                ok = false;
            }
        }
        free(data);
    };

    check("signal_1d", expected_value);
    if (ok && check_extra) check("extra_1d", expected_extra);

    backend.endAction(&profilesCtx);
    backend.endAction(&opCtx);
    return ok;
}

int main() {
    try {
        std::cout << "\n=== Test: a slice session notices rows another writer added ===\n";
        // This gate pins the v2 session engine; the target version comes from the
        // environment, so without pinning it an ambient BACKEND_DEFAULT_WRITE_VERSION=1.0
        // moves the whole test onto the v1 path, where there is no engine to invalidate.
        setenv("BACKEND_DEFAULT_WRITE_VERSION", "2.0", 1);
        if (fs::exists("./test_db_slice_write_after_foreign_append"))
            fs::remove_all("./test_db_slice_write_after_foreign_append");

        // Session A stays open across B's write: A's engine is the one that must notice.
        HDF5Backend backend_a;
        HDF5Backend backend_b;

        std::cout << "--- Phase 1: session A writes t = 0.0, 0.1, 0.2 ---\n";
        DataEntryContext ctx_a(URI);
        backend_a.openPulse(&ctx_a, FORCE_CREATE_PULSE);
        write_slices(backend_a, ctx_a, 0, 0, true, false);
        write_slices(backend_a, ctx_a, 1, 2, false, false);

        std::cout << "--- Phase 2: session B appends t = 0.3 into the same file ---\n";
        {
            DataEntryContext ctx_b(URI);
            backend_b.openPulse(&ctx_b, OPEN_PULSE);
            write_slices(backend_b, ctx_b, 3, 3, false, false);
            backend_b.closePulse(&ctx_b, OPEN_PULSE);
        }

        // A keeps going without knowing B was here. Its engine still counts the rows it
        // wrote and the end of the raw data it knows; both moved. It must rebuild, and
        // its fresh signal must land on index 4 rather than over B's slice.
        std::cout << "--- Phase 3: session A resumes at t = 0.4 with a fresh signal ---\n";
        write_slices(backend_a, ctx_a, 4, 4, false, true);
        backend_a.closePulse(&ctx_a, FORCE_CREATE_PULSE);

        std::cout << "--- Phase 4: read every slice back, from both writers ---\n";
        HDF5Backend reader;
        for (int t_idx = 0; t_idx <= 4; ++t_idx) {
            DataEntryContext ctx_r(URI);
            reader.openPulse(&ctx_r, OPEN_PULSE);
            if (!read_slice(reader, ctx_r, 0.1 * t_idx, t_idx == 4)) {
                reader.closePulse(&ctx_r, OPEN_PULSE);
                return 1;
            }
            reader.closePulse(&ctx_r, OPEN_PULSE);
        }

        std::cout << "[OK] both writers' slices survived the kept-alive session.\n";
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    if (fs::exists("./test_db_slice_write_after_foreign_append"))
        fs::remove_all("./test_db_slice_write_after_foreign_append");
    return 0;
}
