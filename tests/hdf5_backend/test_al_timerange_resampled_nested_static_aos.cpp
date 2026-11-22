// @file  test_al_timerange_resampled_nested_static_aos.cpp
// @brief Reproduces the v2 gap revealed by IMAS-Python
//        imas/test/test_get_sample.py::test_get_sample_heterogeneous_linear_interp
//        (and the *_previous_interp / *_closest_interp variants).
//
// Scenario (equilibrium, inhomogeneous time, i.e. each grids_ggd occurrence
// carries its own 0-D time):
//
//   equilibrium
//   +-- homogeneous_time = 0
//   +-- grids_ggd[]              (dynamic AoS, one stored occurrence)
//        +-- time = 0.0          (per-occurrence scalar, 0-D)
//        +-- grid[]              (STATIC AoS, maxoccur 1)
//             +-- path = "wall:0/description_ggd(1)/grid_ggd"
//
// The read is a TIMERANGE_OP with resampling: tmin=0.2, tmax=0.501,
// dtime={0.05} -> a new 7-point time grid 0.2, 0.25, ..., 0.50.
// The resampled occurrence count is 7 (that part works in v2 already).
//
// The AL contract for the nested STATIC AoS `grid`: it is not time-dependent,
// its size/content must be the SAME for every occurrence returned. The
// reference v1 backend answers size=1 for every resampled occurrence (the
// values come from the sole stored occurrence).
//
// The v2 backend answers size=1 ONLY for the occurrence that has its own
// stored `grid` leaf (resampled index 0) and size=0 for resampled indices
// 1..6, which do not have a stored slice at that index. The HLI therefore
// sees grids_ggd[i].grid with 0 items and the read fails.
//
// This test is deliberately self-contained (its own mini DB, no Python HLI):
// write the same pulse twice, once with BACKEND_DEFAULT_WRITE_VERSION=1.0
// (reference v1 semantics, run first) and once with the v2 default, and
// report the nested-AoS size for every resampled occurrence.
//
// Expected outcome while the gap is present:
//   v1 phase: sizes  [1,1,1,1,1,1,1]   -> phase OK
//   v2 phase: sizes  [1,0,0,0,0,0,0]   -> phase FAILS (bug reproduced)
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const char* IDS = "equilibrium";
static const char* PATH_CONST = "wall:0/description_ggd(1)/grid_ggd";

static int failures = 0;

// One stored occurrence of grids_ggd: time=0.0 and a 1-element static grid AoS.
// Mirrors the HLI fixture (test_get_sample.py): equilibrium has a top-level
// heterogeneous time (512 points) AND a single stored grids_ggd occurrence.
static void write_pulse(const std::string& db) {
    const std::string uri = std::string("imas:hdf5?path=") + db;
    DataEntryContext dec(uri);
    HDF5Backend backend;
    backend.openPulse(&dec, FORCE_CREATE_PULSE);
    OperationContext oc(&dec, IDS, "", WRITE_OP);
    backend.beginAction(&oc);

    int homogeneous_time = 0;   // inhomogeneous: per-occurrence time
    backend.writeData(&oc, "ids_properties/homogeneous_time", "", &homogeneous_time,
                      alconst::integer_data, 0, nullptr);

    const int n_top_time = 512; // eq.time = linspace(0, 2, 512)
    std::vector<double> t(n_top_time);
    for (int k = 0; k < n_top_time; ++k) t[k] = 2.0 * k / (n_top_time - 1);
    int dim1 = 1;
    int size1[1] = {n_top_time};
    backend.writeData(&oc, "time", "time", t.data(), alconst::double_data, dim1, size1);

    int aossz = 1;              // exactly ONE stored grids_ggd occurrence
    // grids_ggd is a DYNAMIC (timed) AoS: its time base is its own per-occurrence
    // time node (IMAS DD: timebasepath = "grids_ggd/time"). The context must carry
    // that timebase so the AL treats it as timed (otherwise it is read as a plain
    // stored AoS and never resampled).
    ArraystructContext ag(&oc, "grids_ggd", "grids_ggd/time");
    backend.beginArraystructAction(&ag, &aossz);
    double t0 = 0.0;            // occurrence time (0-D scalar)
    backend.writeData(&ag, "time", "", &t0, alconst::double_data, 0, nullptr);
    int gridsz = 1;             // nested STATIC AoS with a single element
    ArraystructContext grid(&ag, "grid", "");
    backend.beginArraystructAction(&grid, &gridsz);
    int path_dim = 1;           // AL exposes a C string as STR_1D
    int path_size[1] = {(int)std::strlen(PATH_CONST)};
    backend.writeData(&grid, "path", "", (void*)PATH_CONST, alconst::char_data, path_dim, path_size);
    backend.endAction(&grid);
    backend.endAction(&ag);

    backend.endAction(&oc);
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
}

// Read the resampled grids_ggd and check the nested static AoS size at every
// occurrence. AL contract (as served by the reference v1 backend, and asserted
// by IMAS-Python test_get_sample.py): a resampled dynamic AoS has one
// occurrence per requested time point, and its nested STATIC AoS has the SAME
// stored size (1) at EVERY occurrence -- the inner shape must come from a
// representative stored occurrence, not from the resampled index.
//
// Returns the number of occurrences with a wrong nested `grid` AoS size.
static int read_resampled(const std::string& db) {
    const std::string uri = std::string("imas:hdf5?path=") + db;
    DataEntryContext dec(uri);
    HDF5Backend backend;
    backend.openPulse(&dec, OPEN_PULSE);

    const double tmin = 0.2, tmax = 0.501;
    std::vector<double> dtime = {0.05};
    OperationContext oc(&dec, IDS, READ_OP, alconst::timerange_op, tmin, tmax,
                        dtime, alconst::linear_interp);
    backend.beginAction(&oc);

    int aossz = -1;
    // Same timed context as in the HLI (IMAS DD: grids_ggd timebasepath =
    // "grids_ggd/time"): the outer dynamic AoS is resampled to the requested
    // 7-point grid for both backends.
    ArraystructContext ag(&oc, "grids_ggd", "grids_ggd/time");
    backend.beginArraystructAction(&ag, &aossz);
    std::printf("    resampled grids_ggd size = %d (reference: 7)\n", aossz);
    if (aossz != 7) {
        std::printf("[FAIL] resampled grids_ggd size is %d, expected 7\n", aossz);
        ++failures;
    }

    int bad = 0;
    for (int i = 0; i < aossz; ++i) {
        int gridsz = -1;
        ArraystructContext grid(&ag, "grid", "");
        backend.beginArraystructAction(&grid, &gridsz);

        std::string content;
        if (gridsz == 1) {
            void* data = nullptr;
            int dt = alconst::char_data;
            int dim = -1, sz = 0;
            backend.readData(&grid, "path", "", &data, &dt, &dim, &sz);
            if (data) { content = std::string((char*)data).substr(0, 40); free(data); }
        }

        bool ok = (gridsz == 1);
        if (ok) {
            // Informational (correct in the v1 reference): the element content must
            // be the stored one. A 0-D string may come back dim 0 or 1, so check
            // bytes, not dim.
            if (content.find(PATH_CONST) == std::string::npos)
                std::printf("    occ %d: [info] path != stored value: '%s'\n", i, content.c_str());
        } else {
            std::printf("[FAIL] occ %d: nested grid AoS size = %d (expected 1) -- nested static "
                        "AoS read at a resampled occurrence whose index has no stored slice\n",
                        i, gridsz);
            ++bad;
        }
        backend.endAction(&grid);
        if (i + 1 < aossz) ag.nextIndex(1);
    }

    backend.endAction(&ag);
    backend.endAction(&oc);
    backend.closePulse(&dec, CLOSE_PULSE);
    return bad;
}

int main(int argc, char** argv) {
    // Optional: run only one phase ("v1" or "v2"); default runs both.
    bool only_v1 = (argc > 1 && std::strcmp(argv[1], "v1") == 0);
    bool only_v2 = (argc > 1 && std::strcmp(argv[1], "v2") == 0);
    bool both = !only_v1 && !only_v2;

    if (both || only_v1) {
        std::printf("=== PHASE A: reference v1 storage (BACKEND_DEFAULT_WRITE_VERSION=1.0) ===\n");
        setenv("BACKEND_DEFAULT_WRITE_VERSION", "1.0", 1);
        const std::string db = "./test_db_resampled_nested_static_aos_v1";
        fs::remove_all(db);
        write_pulse(db);
        int bad = read_resampled(db);
        fs::remove_all(db);
        if (bad) {
            std::printf("[FAIL] v1 reference phase: %d occurrence(s) with a wrong nested grid AoS\n", bad);
            ++failures;
        } else {
            std::printf("[OK] v1 reference phase served the nested static AoS for every occurrence\n");
        }
        unsetenv("BACKEND_DEFAULT_WRITE_VERSION");
    }

    if (both || only_v2) {
        std::printf("=== PHASE B: v2 default storage (the phase that must reproduce the bug) ===\n");
        unsetenv("BACKEND_DEFAULT_WRITE_VERSION");
        const std::string db = "./test_db_resampled_nested_static_aos_v2";
        fs::remove_all(db);
        write_pulse(db);
        int bad = read_resampled(db);
        fs::remove_all(db);
        if (bad) {
            std::printf("[FAIL] v2 phase: %d occurrence(s) with a wrong nested grid AoS "
                        "-> BUG REPRODUCED (matches test_get_sample.py heterogeneous interp failures)\n", bad);
            ++failures;
        } else {
            std::printf("[OK] v2 phase served the nested static AoS for every occurrence (gap fixed)\n");
        }
    }

    if (failures) {
        std::printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
