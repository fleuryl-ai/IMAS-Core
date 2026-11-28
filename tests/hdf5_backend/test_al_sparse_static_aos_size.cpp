// @file  test_al_sparse_static_aos_size.cpp
// @brief The reader must return the DECLARED size of a static AoS whose
//        elements are not all filled (pulse 65782 fidelity campaign).
//
// Measured defect (calorimetry, occ 0): the IDS declares a 19-element static
// AoS `group`; the pulse fills only group[0].name — the v1 backend reads the
// 19 elements back, the v2 backend (native HLI write) read back 1. The v2
// /index keeps the truth: the meta row declares shape[0]=19 and the writer
// records instance rows only for the filled elements, so the reader must size
// the AoS from the declared size, not from max(child index)+1.
//
// Phase A (v1 store, reference) must serve 19, Phase B (v2 store) must too.
// Before the fix Phase B served 1 -> the test failed.

#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

static const char* IDS = "calorimetry";
static const int N_GROUP = 19;    // declared size of the static AoS
static const char* NAME_VALUE = "wall:calorimetry/description(1)/group(1)";

static int failures = 0;

// calorimetry-like pulse: static IDS, group[] with 19 elements, only
// group[0].name filled — the HLI put() of 65782, kept minimal.
static void write_pulse(const std::string& db) {
    const std::string uri = std::string("imas:hdf5?path=") + db;
    DataEntryContext dec(uri);
    HDF5Backend backend;
    backend.openPulse(&dec, FORCE_CREATE_PULSE);
    OperationContext oc(&dec, IDS, "", WRITE_OP);
    backend.beginAction(&oc);

    int homogeneous_time = 1;    // static pulse
    backend.writeData(&oc, "ids_properties/homogeneous_time", "", &homogeneous_time,
                      alconst::integer_data, 0, nullptr);
    double t0 = 0.0;
    backend.writeData(&oc, "time", "time", &t0, alconst::double_data, 0, nullptr);

    int aossz = N_GROUP;
    ArraystructContext ag(&oc, "group", "");      // STATIC AoS (no timebase)
    backend.beginArraystructAction(&ag, &aossz);
    for (int i = 0; i < N_GROUP; ++i) {
        if (i == 0) {
            int sz1[1] = {(int)std::strlen(NAME_VALUE)};
            backend.writeData(&ag, "name", "", (void*)NAME_VALUE,
                              alconst::char_data, 1, sz1);
        }
        if (i + 1 < N_GROUP) ag.nextIndex(1);
    }
    backend.endAction(&ag);

    backend.endAction(&oc);
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
}

// Global read (same shape as the HLI ids.get() of the fidelity campaign).
// Returns the number of failed assertions.
static int read_global(const std::string& db) {
    const std::string uri = std::string("imas:hdf5?path=") + db;
    DataEntryContext dec(uri);
    HDF5Backend backend;
    backend.openPulse(&dec, OPEN_PULSE);
    OperationContext oc(&dec, IDS, "", READ_OP);
    backend.beginAction(&oc);

    int bad = 0;
    int aossz = -1;
    ArraystructContext ag(&oc, "group", "");
    backend.beginArraystructAction(&ag, &aossz);
    std::printf("    group size = %d (reference: %d)\n", aossz, N_GROUP);
    if (aossz != N_GROUP) {
        std::printf("[FAIL] static AoS with only element 0 filled reads %d elements, "
                    "expected %d (the empty elements are part of the pulse, and the "
                    "file stores the declared size even without values for them)\n",
                    aossz, N_GROUP);
        ++bad;
    } else {
        // The filled element must still be there, still correct.
        void* data = nullptr; int dt = alconst::char_data, dim = -1, sz = 0;
        backend.readData(&ag, "name", "", &data, &dt, &dim, &sz);
        if (dt != alconst::char_data || sz <= 0
            || std::strncmp((char*)data, NAME_VALUE, std::strlen(NAME_VALUE)) != 0) {
            std::printf("[FAIL] group[0].name content lost or corrupted\n");
            ++bad;
        }
        if (data) free(data);
    }
    backend.endAction(&ag);
    backend.endAction(&oc);
    backend.closePulse(&dec, CLOSE_PULSE);
    return bad;
}

int main(int argc, char** argv) {
    bool only_v1 = (argc > 1 && std::strcmp(argv[1], "v1") == 0);
    bool only_v2 = (argc > 1 && std::strcmp(argv[1], "v2") == 0);
    bool both = !only_v1 && !only_v2;

    if (both || only_v1) {
        std::printf("=== PHASE A: v1 storage (reference) ===\n");
        setenv("BACKEND_DEFAULT_WRITE_VERSION", "1.0", 1);
        const std::string db = "./test_db_sparse_static_aos_v1";
        fs::remove_all(db);
        write_pulse(db);
        int bad = read_global(db);
        fs::remove_all(db);
        if (bad) std::printf("[FAIL] v1 reference phase: %d assertion(s) failed\n", bad);
        else std::printf("[OK] v1 reference phase served the full declared AoS\n");
        failures += bad;
        unsetenv("BACKEND_DEFAULT_WRITE_VERSION");
    }

    if (both || only_v2) {
        std::printf("\n=== PHASE B: v2 default storage ===\n");
        unsetenv("BACKEND_DEFAULT_WRITE_VERSION");
        const std::string db = "./test_db_sparse_static_aos_v2";
        fs::remove_all(db);
        write_pulse(db);
        int bad = read_global(db);
        fs::remove_all(db);
        if (bad) std::printf("[FAIL] v2 phase: the declared size of the sparse static AoS was not served\n", bad);
        else std::printf("[OK] v2 phase served the full declared AoS (gap fixed)\n");
        failures += bad;
    }

    if (failures) {
        std::printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    std::printf("\nALL PASSED\n");
    return 0;
}
