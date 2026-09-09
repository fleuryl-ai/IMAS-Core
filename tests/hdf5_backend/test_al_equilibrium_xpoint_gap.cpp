// @file  test_al_equilibrium_xpoint_gap.cpp
// @brief Reproduces the "disappearing x-point" case (hdf5bug.py) in the C++ AL
//        API: a dynamic time-slice AoS (equilibrium/time_slice) inside which a
//        nested static AoS (contour_tree/node) DROPS in size from one slice to
//        the next (2 nodes at t=0.0, 1 node at t=0.1 -- the x-point vanishes),
//        and the second node (x-point) additionally carries a nested static
//        AoS (node/levelset) whose members are arrays (r[2], z[2]).
//
//        The test writes the two slices, then reads each one back by time with
//        CLOSEST (the primary read -- mirroring the Python
//        get_slice("equilibrium", 0.1, CLOSEST_INTERP)) and PREVIOUS, asserting
//        that the per-slice nested static-AoS size and all values come back
//        intact. This guards against the "gap" (smaller node list in a later
//        slice) leaking the earlier slice's extra x-point or corrupting the read.
//
//        Modeled on test_al_dynamic_aos_nested_static_slice.cpp (dynamic AoS ->
//        nested static AoS with a size that varies per slice), extended with one
//        extra nesting level (node/levelset/{r[], z[]}).
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""
#define YELLOW  ""

const std::string URI = "imas:hdf5?path=./test_db_equilibrium_xpoint_gap";
const std::string DB  = "test_db_equilibrium_xpoint_gap";

static bool near(double a, double b) { return std::abs(a - b) < 1e-12; }

// Snapshot of one equilibrium slice: the nested contour_tree/node structure.
struct Snapshot {
    int    node_size            = 0;
    double node_r[2]            = {0.0, 0.0};
    double node_z[2]            = {0.0, 0.0};
    bool   xpoint_has_levelset  = false;   // node[1] carries a levelset AoS
    double ls_r0 = 0.0, ls_r1   = 0.0;     // node[1]/levelset/r
    double ls_z0 = 0.0, ls_z1   = 0.0;     // node[1]/levelset/z
};

// ---------------------------------------------------------------------------
// Write: two time-slices; node list is 2 at t=0.0 (flux max + x-point) and
//       1 at t=0.1 (x-point removed, "keep=True" keeps the flux max).
// ---------------------------------------------------------------------------
static void write_fixture(HDF5Backend& backend, DataEntryContext& dec) {
    backend.openPulse(&dec, FORCE_CREATE_PULSE);
    OperationContext opCtx(&dec, "equilibrium", "", WRITE_OP);
    backend.beginAction(&opCtx);

    int homogeneous_time = 1;
    backend.writeData(&opCtx, "ids_properties/homogeneous_time",
                      "", &homogeneous_time, alconst::integer_data, 0, nullptr);

    int TS = 2;   // two time-slices: t=0.0 and t=0.1
    ArraystructContext tsCtx(&opCtx, "time_slice", "time");
    backend.beginArraystructAction(&tsCtx, &TS);

    for (int t = 0; t < TS; ++t) {
        double cur_time = (t == 0) ? 0.0 : 0.1;
        backend.writeData(&tsCtx, "time", "", &cur_time, alconst::double_data, 0, nullptr);

        // contour_tree: intermediate static AoS, a single element
        ArraystructContext ctCtx(&tsCtx, "contour_tree", "");
        int ct = 1;
        backend.beginArraystructAction(&ctCtx, &ct);

        // node: nested static AoS whose SIZE VARY between slices (2 -> 1)
        int node_size = (t == 0) ? 2 : 1;
        ArraystructContext nodeCtx(&ctCtx, "node", "");
        backend.beginArraystructAction(&nodeCtx, &node_size);

        for (int n = 0; n < node_size; ++n) {
            double r = 2.0;              // both nodes at r=2 (see hdf5bug.py)
            double z = (n == 0) ? 0.0 : -2.0;   // node[0] flux max, node[1] x-point
            backend.writeData(&nodeCtx, "r", "", &r, alconst::double_data, 0, nullptr);
            backend.writeData(&nodeCtx, "z", "", &z, alconst::double_data, 0, nullptr);

            if (n == 1) {   // the x-point carries a levelset { r=[1,1], z=[1,1] }
                ArraystructContext lsCtx(&nodeCtx, "levelset", "");
                int ls = 1;
                backend.beginArraystructAction(&lsCtx, &ls);

                const int LS = 2;
                std::vector<double> lr(LS, 1.0), lz(LS, 1.0);
                int ld = 1;   int ldim[] = {LS};
                backend.writeData(&lsCtx, "r", "", lr.data(), alconst::double_data, ld, ldim);
                backend.writeData(&lsCtx, "z", "", lz.data(), alconst::double_data, ld, ldim);

                backend.endAction(&lsCtx);
            }

            if (n < node_size - 1) nodeCtx.nextIndex(1);
        }
        backend.endAction(&nodeCtx);
        backend.endAction(&ctCtx);

        if (t < TS - 1) tsCtx.nextIndex(1);
    }
    backend.endAction(&tsCtx);

    // master time vector at the data-object level (matches test_al_dynamic_aos_nested_static_slice)
    std::vector<double> mt = {0.0, 0.1};
    int mdim = 1;   int msize[] = {2};
    backend.writeData(&opCtx, "time", "time", mt.data(), alconst::double_data, mdim, msize);

    backend.endAction(&opCtx);
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
    std::cout << GREEN << "  [write] t=0.0 -> node size 2 (flux max + x-point/levelset); "
                 "t=0.1 -> node size 1 (x-point gone)\n" << RESET;
}

// ---------------------------------------------------------------------------
// Read one slice and capture the nested structure.
// ---------------------------------------------------------------------------
static Snapshot read_one(HDF5Backend& backend, DataEntryContext& dec,
                         double t, int interp) {
    backend.openPulse(&dec, OPEN_PULSE);
    OperationContext opCtx(&dec, "equilibrium", READ_OP, alconst::slice_op, t, interp);
    backend.beginAction(&opCtx);

    Snapshot s;

    ArraystructContext tsCtx(&opCtx, "time_slice", "time");
    int ts = 0;
    backend.beginArraystructAction(&tsCtx, &ts);
    if (ts != 1) { std::cerr << RED << "    expected time_slice size 1, got " << ts << "\n" << RESET; }

    ArraystructContext ctCtx(&tsCtx, "contour_tree", "");
    int ct = 0;
    backend.beginArraystructAction(&ctCtx, &ct);
    if (ct != 1) { std::cerr << RED << "    expected contour_tree size 1, got " << ct << "\n" << RESET; }

    ArraystructContext nodeCtx(&ctCtx, "node", "");
    int ns = 0;
    backend.beginArraystructAction(&nodeCtx, &ns);
    s.node_size = ns;

    for (int n = 0; n < ns && n < 2; ++n) {
        void* d = nullptr;
        int dt = alconst::double_data, dim = 0;
        int sz[H5S_MAX_RANK];

        backend.readData(&nodeCtx, "r", "time", &d, &dt, &dim, sz);
        if (d) { s.node_r[n] = ((double*)d)[0]; free(d); }
        d = nullptr;
        backend.readData(&nodeCtx, "z", "time", &d, &dt, &dim, sz);
        if (d) { s.node_z[n] = ((double*)d)[0]; free(d); }

        if (n == 1) {   // x-point: inspect its levelset (present only at t=0.0)
            ArraystructContext lsCtx(&nodeCtx, "levelset", "");
            int ls = 0;
            backend.beginArraystructAction(&lsCtx, &ls);
            if (ls > 0) {
                s.xpoint_has_levelset = true;
                d = nullptr;
                backend.readData(&lsCtx, "r", "time", &d, &dt, &dim, sz);
                if (d) { if (sz[0] >= 1) s.ls_r0 = ((double*)d)[0]; if (sz[0] >= 2) s.ls_r1 = ((double*)d)[1]; free(d); }
                d = nullptr;
                backend.readData(&lsCtx, "z", "time", &d, &dt, &dim, sz);
                if (d) { if (sz[0] >= 1) s.ls_z0 = ((double*)d)[0]; if (sz[0] >= 2) s.ls_z1 = ((double*)d)[1]; free(d); }
            }
            backend.endAction(&lsCtx);
        }

        if (n < ns - 1) nodeCtx.nextIndex(1);
    }

    backend.endAction(&nodeCtx);
    backend.endAction(&ctCtx);
    backend.endAction(&tsCtx);
    backend.endAction(&opCtx);
    backend.closePulse(&dec, OPEN_PULSE);
    return s;
}

int main() {
    try {
        std::cout << BOLD << "\n=== equilibrium/time_slice nested static-AoS with a gap (x-point disappears) ===\n" << RESET;

        if (fs::exists(DB)) fs::remove_all(DB);

        { DataEntryContext dec(URI); HDF5Backend backend; write_fixture(backend, dec); }

        int nfail = 0;
        auto check = [&](bool cond, const std::string& what) {
            std::cout << (cond ? GREEN : RED) << "  " << (cond ? "OK  " : "FAIL")
                      << "  " << what << "\n" << RESET;
            if (!cond) ++nfail;
        };

        // ---- t=0.0 CLOSEST : 2 nodes; x-point present with its levelset ----
        std::cout << YELLOW << "  -- t=0.0 CLOSEST --\n" << RESET;
        {
            DataEntryContext dec(URI); HDF5Backend backend;
            Snapshot s = read_one(backend, dec, 0.0, alconst::closest_interp);
            check(s.node_size == 2,                    "node_size == 2");
            check(near(s.node_r[0], 2.0),               "node[0].r == 2.0");
            check(near(s.node_z[0], 0.0),               "node[0].z == 0.0");
            check(near(s.node_r[1], 2.0),               "node[1].r == 2.0");
            check(near(s.node_z[1], -2.0),              "node[1].z == -2.0");
            check(s.xpoint_has_levelset,                "node[1] carries a levelset");
            check(near(s.ls_r0, 1.0) && near(s.ls_r1, 1.0), "levelset.r == [1,1]");
            check(near(s.ls_z0, 1.0) && near(s.ls_z1, 1.0), "levelset.z == [1,1]");
        }

        // ---- t=0.1 CLOSEST : x-point gone -> node size 1 (the Python get_slice) ----
        std::cout << YELLOW << "  -- t=0.1 CLOSEST --\n" << RESET;
        {
            DataEntryContext dec(URI); HDF5Backend backend;
            Snapshot s = read_one(backend, dec, 0.1, alconst::closest_interp);
            check(s.node_size == 1,                    "node_size == 1 (x-point gone)");
            check(near(s.node_r[0], 2.0),               "node[0].r == 2.0");
            check(near(s.node_z[0], 0.0),               "node[0].z == 0.0");
        }

        // ---- t=0.1 PREVIOUS : exact match at t=0.1 -> node size 1 ----
        std::cout << YELLOW << "  -- t=0.1 PREVIOUS --\n" << RESET;
        {
            DataEntryContext dec(URI); HDF5Backend backend;
            Snapshot s = read_one(backend, dec, 0.1, alconst::previous_interp);
            check(s.node_size == 1,                    "node_size == 1");
        }

        // ---- t=0.05 PREVIOUS : resolves back to t=0.0 -> 2 nodes, x-point intact ----
        std::cout << YELLOW << "  -- t=0.05 PREVIOUS (earlier slice still carries the x-point) --\n" << RESET;
        {
            DataEntryContext dec(URI); HDF5Backend backend;
            Snapshot s = read_one(backend, dec, 0.05, alconst::previous_interp);
            check(s.node_size == 2,                    "node_size == 2");
            check(near(s.node_z[1], -2.0),              "node[1].z == -2.0 (x-point not corrupted/lost)");
            check(s.xpoint_has_levelset,                "node[1] still carries its levelset");
        }

        if (fs::exists(DB)) fs::remove_all(DB);

        if (nfail == 0) {
            std::cout << GREEN << BOLD << "\n[OK] All gap / nested-static-AoS assertions passed.\n" << RESET;
            return 0;
        }
        std::cerr << RED << BOLD << "\n[FAIL] " << nfail << " assertion(s) failed.\n" << RESET;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << "\n" << RESET;
        if (fs::exists(DB)) fs::remove_all(DB);
        return 1;
    }
}
