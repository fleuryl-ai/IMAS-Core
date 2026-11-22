// @file  test_al_list_filled_paths.cpp
//
// Reproduces and pins the backend-v2 bug reported by
// IMAS-Python `imas/test/test_list_filled_paths.py`:
//
//   * In a legacy (v1) group the first level holds the logical data nodes
//     (ids_properties&comment, profiles_1d[]&grid&psi, time, ...), so
//     `HDF5Reader::list_filled_paths` (which walks those links) returns the
//     IMAS logical paths correctly.
//
//   * In a PanzerDB (v2) group the first level holds only the physical
//     columns (`data_raw_*`, `data_scalar_*`, `data_time_f64`, `index`,
//     `path_table`, `list_spans`). The inherited v1 walk therefore returned
//     those physical names, which broke any caller expecting IMAS paths.
//
//   * The fix: `HDF5Reader_v2::list_filled_paths` enumerates the logical
//     leaves from the /index cache, strips occurrence indices and AoS
//     meta-nodes, and de-duplicates — yielding the same logical path set as
//     the legacy reader.
//
//   * A non-existent occurrence must come back with an empty list (size 0)
//     and no exception, in both backends.
//
// The test is self-contained (its own mini core_profiles, no Python HLI) and
// runs an A/B phase in one binary: v1 (reference) vs v2 (the code under test).
#include "hdf5_backend.h"
#include "al_backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const char* IDS = "core_profiles";
static int failures = 0;

// One minimal core_profiles pulse, close to what the HLI test writes.
static void write_pulse(const std::string& db) {
    const std::string uri = std::string("imas:hdf5?path=") + db;
    DataEntryContext dec(uri);
    HDF5Backend backend;
    backend.openPulse(&dec, FORCE_CREATE_PULSE);
    OperationContext oc(&dec, IDS, "", WRITE_OP);
    backend.beginAction(&oc);

    // ids_properties
    int homogeneous_time = 1;
    backend.writeData(&oc, "ids_properties/homogeneous_time", "", &homogeneous_time,
                      alconst::integer_data, 0, nullptr);
    const char* comment = "comment";
    int c1 = 1; int csize[1] = {(int)strlen(comment)};
    backend.writeData(&oc, "ids_properties/comment", "", (void*)comment,
                      alconst::char_data, c1, csize);

    // top-level 1-D time series
    const double t[2] = {0.1, 0.2};
    int d1 = 1; int dsize[1] = {2};
    backend.writeData(&oc, "time", "", (void*)t, alconst::double_data, d1, dsize);

    // profiles_1d: two occurrences, with nested nodes of varying depth
    {
        int aossz = 2;
        ArraystructContext ag(&oc, "profiles_1d", "time");
        backend.beginArraystructAction(&ag, &aossz);

        // occurrence 0: grid / rho_tor_norm
        {
            int gsz = 1;
            ArraystructContext g(&ag, "grid", "");
            backend.beginArraystructAction(&g, &gsz);
            const double v[2] = {1.0, 2.0};
            backend.writeData(&g, "rho_tor_norm", "", (void*)v, alconst::double_data, d1, dsize);
            backend.endAction(&g);
        }
        // occurrence 0: ion/1/temperature
        {
            int isz = 2;
            ArraystructContext ion(&ag, "ion", "");
            backend.beginArraystructAction(&ion, &isz);
            ion.nextIndex(1); // move to ion[1]
            const double v[2] = {1.0, 2.0};
            backend.writeData(&ion, "temperature", "", (void*)v, alconst::double_data, d1, dsize);
            backend.endAction(&ion);
        }
        ag.nextIndex(1); // move to profiles_1d[1]

        // occurrence 1: grid/psi, q, e_field/radial
        {
            int gsz = 1;
            ArraystructContext g(&ag, "grid", "");
            backend.beginArraystructAction(&g, &gsz);
            const double v[2] = {1.0, 2.0};
            backend.writeData(&g, "psi", "", (void*)v, alconst::double_data, d1, dsize);
            backend.endAction(&g);
        }
        {
            const double v[2] = {1.0, 2.0};
            backend.writeData(&ag, "q", "", (void*)v, alconst::double_data, d1, dsize);
        }
        {
            int fsz = 1;
            ArraystructContext ef(&ag, "e_field", "");
            backend.beginArraystructAction(&ef, &fsz);
            const double v[2] = {1.0, 2.0};
            backend.writeData(&ef, "radial", "", (void*)v, alconst::double_data, d1, dsize);
            backend.endAction(&ef);
        }
        backend.endAction(&ag);
    }

    // global_quantities/ip
    {
        int gsz = 1;
        ArraystructContext gq(&oc, "global_quantities", "");
        backend.beginArraystructAction(&gq, &gsz);
        const double v[2] = {1.0, 2.0};
        backend.writeData(&gq, "ip", "", (void*)v, alconst::double_data, d1, dsize);
        backend.endAction(&gq);
    }

    backend.endAction(&oc);
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
}

// Run `list_filled_paths` on occurrence 0 and compare against the expected set.
static bool test_occurrence_zero(const std::string& db, const std::string& tag) {
    const std::string uri = std::string("imas:hdf5?path=") + db;
    DataEntryContext dec(uri);
    HDF5Backend backend;
    backend.openPulse(&dec, OPEN_PULSE);
    OperationContext oc(&dec, IDS, "", READ_OP);
    backend.beginAction(&oc);

    char** paths = nullptr;
    int n = -1;
    backend.list_filled_paths(&oc, IDS, &paths, &n);
    std::set<std::string> got;
    for (int i = 0; i < n; ++i) got.insert(paths[i]);
    std::free(paths);

    // Note: the ids_properties/version_put/* trio is written by the HLI's put(),
    // not by the low-level writer this test drives, so it is not expected here.
    std::set<std::string> expected = {
        "ids_properties/comment",
        "ids_properties/homogeneous_time",
        "time",
        "profiles_1d/grid/rho_tor_norm",
        "profiles_1d/grid/psi",
        "profiles_1d/ion/temperature",
        "profiles_1d/q",
        "profiles_1d/e_field/radial",
        "global_quantities/ip",
    };
    std::printf("    [%s] occurrence 0: %d paths\n", tag.c_str(), n);
    if (got != expected) {
        std::printf("[FAIL] %s: occurrence-0 list mismatch (%zu vs %zu expected)\n", tag.c_str(), got.size(), expected.size());
        std::printf("      missing:");
        for (auto& e : expected) if (!got.count(e)) std::printf(" %s", e.c_str());
        std::printf("\n      extra:");
        for (auto& g : got) if (!expected.count(g)) std::printf(" %s", g.c_str());
        std::printf("\n");
        return false;
    }

    // Non-existent occurrence: empty list, no exception, same contract.
    char** paths2 = nullptr;
    int n2 = -1;
    backend.list_filled_paths(&oc, "core_profiles/99", &paths2, &n2);
    std::printf("    [%s] occurrence 99: %d paths\n", tag.c_str(), n2);
    std::free(paths2);
    if (n2 != 0) {
        std::printf("[FAIL] %s: non-existent occurrence reported %d paths (expected 0)\n", tag.c_str(), n2);
        return false;
    }

    backend.endAction(&oc);
    backend.closePulse(&dec, OPEN_PULSE);
    return true;
}

int main(int argc, char** argv) {
    bool only_v1 = (argc > 1 && std::strcmp(argv[1], "v1") == 0);
    bool only_v2 = (argc > 1 && std::strcmp(argv[1], "v2") == 0);
    bool both = !only_v1 && !only_v2;

    if (both || only_v1) {
        std::printf("=== PHASE A: v1 storage (reference) ===\n");
        setenv("BACKEND_DEFAULT_WRITE_VERSION", "1.0", 1);
        const std::string db = "./test_db_lffp_v1";
        fs::remove_all(db);
        write_pulse(db);
        bool ok = test_occurrence_zero(db, "v1");
        fs::remove_all(db);
        if (ok) std::printf("[OK] v1 reference phase served the expected logical path set\n");
        else ++failures;
        unsetenv("BACKEND_DEFAULT_WRITE_VERSION");
    }

    if (both || only_v2) {
        std::printf("=== PHASE B: v2 storage (code under test) ===\n");
        unsetenv("BACKEND_DEFAULT_WRITE_VERSION");
        const std::string db = "./test_db_lffp_v2";
        fs::remove_all(db);
        write_pulse(db);
        bool ok = test_occurrence_zero(db, "v2");
        fs::remove_all(db);
        if (ok) std::printf("[OK] v2 phase served the expected logical path set (bug fixed)\n");
        else { std::printf("[FAIL] v2 phase list mismatch -> BUG REPRODUCED (matches test_list_filled_paths.py)\n"); ++failures; }
    }

    if (failures) { std::printf("%d FAILURE(S)\n", failures); return 1; }
    std::printf("ALL PASSED\n");
    return 0;
}
