// @file  test_al_homog_aos_element_index.cpp
// @brief A homogeneous IDS (homogeneous_time = 1) read slice by slice through a dynamic
//        AoS: the AoS slice index is resolved on the IDS's own root time base, which a
//        bulk write leaves on a single index row, instead of the base the AoS keeps for
//        itself (one row per time step, interleaved with the payloads that separate the
//        steps -- the cost to_improve.md point 24 measures).
//
//        The second fixture is what forbids doing that unconditionally: when the AoS is
//        written only on some of the time steps, its elements are numbered in write order
//        (0,1,...) behind the time axis, so the root base would point at elements the
//        file never wrote. The guard is the element count: the root base answers only
//        when the AoS holds exactly one element per time point.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include "panzerdb.h"
#include <hdf5.h>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

static int nfail = 0;
#include "fixtures/check.h"

static const int N_SLICES = 10;
static const int N_POINTS = 4;

static std::string path_of(const std::string& uri) {
    return uri.substr(uri.find("path=") + 5);
}

// Writes homogeneous_time = 1 and a root time base of N_SLICES points in ONE global
// action, so the root base sits on a single index row.
static void open_and_write_root_time(HDF5Backend& backend, DataEntryContext& dec) {
    backend.openPulse(&dec, FORCE_CREATE_PULSE);
    OperationContext op(&dec, "test_ids", "", WRITE_OP);
    backend.beginAction(&op);
    int homogeneous_time = 1;
    backend.writeData(&op, "ids_properties/homogeneous_time", "", &homogeneous_time,
                      alconst::integer_data, 0, nullptr);
    std::vector<double> root_times(N_SLICES);
    for (int t = 0; t < N_SLICES; ++t) root_times[t] = 0.1 * t;
    int shape[1] = {N_SLICES};
    backend.writeData(&op, "time", "", root_times.data(), alconst::double_data, 1, shape);
    backend.endAction(&op);
}

// One time step of the dynamic AoS, inside the caller's open write session: element data
// 1000 + slice + 0.1*point.
// write_own_base = false leaves the AoS without a time base of its own, which is legal
// in a homogeneous IDS: the root base is the only one there.
static void write_aos_slice(HDF5Backend& backend, DataEntryContext& dec, int slice,
                            bool write_own_base) {
    OperationContext op(&dec, "test_ids", WRITE_OP, alconst::slice_op,
                        alconst::undefined_time, alconst::undefined_interp);
    backend.beginAction(&op);
    int size_B = 1;
    ArraystructContext ctxB(&op, "B", "time");
    backend.beginArraystructAction(&ctxB, &size_B);
    if (write_own_base) {
        double tv = 0.1 * slice;
        backend.writeData(&ctxB, "time", "", &tv, alconst::double_data, 0, nullptr);
    }
    std::vector<double> sig(N_POINTS);
    for (int k = 0; k < N_POINTS; ++k) sig[k] = 1000.0 + slice + 0.1 * k;
    int shape[1] = {N_POINTS};
    backend.writeData(&ctxB, "sig", "time", sig.data(), alconst::double_data, 1, shape);
    backend.endAction(&ctxB);
    backend.endAction(&op);
}

// Reads B/sig at one time and checks it is the slice the ROOT base points at.
static void check_slice(const std::string& uri, double t_req, int expect_slice,
                        const std::string& label) {
    HDF5Backend backend;
    DataEntryContext dec(uri);
    backend.openPulse(&dec, OPEN_PULSE);
    OperationContext op(&dec, "test_ids", READ_OP, alconst::slice_op, t_req,
                        alconst::closest_interp);
    backend.beginAction(&op);

    int size_B = -7;
    ArraystructContext ctxB(&op, "B", "time");
    backend.beginArraystructAction(&ctxB, &size_B);
    CHECK(size_B == 1, label + ": AoS reports size " + std::to_string(size_B) + " (want 1)");

    void* data = nullptr;
    int type = alconst::double_data, dim = 0;
    int size[H5S_MAX_RANK];
    int found = backend.readData(&ctxB, "sig", "time", &data, &type, &dim, size);
    CHECK(found == 1, label + ": B/sig not found at t=" + std::to_string(t_req));
    if (found && data && dim >= 1) {
        double* v = (double*)data;
        for (int k = 0; k < size[0] && k < N_POINTS; ++k)
            CHECK(std::abs(v[k] - (1000.0 + expect_slice + 0.1 * k)) < 1e-9,
                  label + ": wrong value at point " + std::to_string(k) + " of t=" +
                      std::to_string(t_req));
    }
    if (data) free(data);
    backend.endAction(&ctxB);
    backend.endAction(&op);
    backend.closePulse(&dec, OPEN_PULSE);
}

// The layout fact the guard rests on, read straight from the engine: a gapped AoS numbers
// its elements in write order, so its element count and the root time length disagree.
static void check_gapped_layout(const std::string& uri) {
    std::string file = path_of(uri) + "/test_ids.h5";
    hid_t h5 = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5 < 0) {
        std::cerr << "FAIL: cannot open " << file << " for the layout check\n";
        ++nfail;
        return;
    }
    hid_t grp = H5Gopen2(h5, "test_ids", H5P_DEFAULT);
    H5Fclose(h5);
    if (grp < 0) {
        std::cerr << "FAIL: cannot open the 'test_ids' group for the layout check\n";
        ++nfail;
        return;
    }
    PanzerDB db(grp, PanzerDB::OpenMode::READ, true, true);
    const size_t elements = db.getDynamicAOSSize("B");
    const size_t time_points = db.getWholeDynamicSignal("time").size();
    CHECK(elements == 2 && time_points == (size_t)N_SLICES,
          "gapped AoS: expected 2 elements behind " + std::to_string(N_SLICES) +
              " time points, got " + std::to_string(elements) + " / " +
              std::to_string(time_points));
    // The two bases answer differently for the same requested time: the root one points
    // at time index 9, the AoS one at the element that actually holds t=0.9. Asking the
    // root here would ask for an element the file never wrote.
    const int64_t on_root = db.getTimeIndex("time", 0.9, alconst::closest_interp);
    const int64_t on_aos = db.getTimeIndex("B/time", 0.9, alconst::closest_interp);
    CHECK(on_root == 9, "gapped AoS: the root base should answer 9 for t=0.9, got " +
                            std::to_string(on_root));
    CHECK(on_aos == 1, "gapped AoS: the AoS base should answer 1 for t=0.9, got " +
                           std::to_string(on_aos));
    H5Gclose(grp);
}

int main() {
    const std::string dense_uri = "imas:hdf5?path=./test_db_homog_aos_dense";
    const std::string nobase_uri = "imas:hdf5?path=./test_db_homog_aos_nobase";
    const std::string gapped_uri = "imas:hdf5?path=./test_db_homog_aos_gapped";
    try {
        // A. one element per time point: the root base and the AoS base agree, and the
        //    root one is what the reader is expected to consult.
        {
            std::filesystem::remove_all(path_of(dense_uri));
            HDF5Backend backend;
            DataEntryContext dec(dense_uri);
            open_and_write_root_time(backend, dec);
            for (int t = 0; t < N_SLICES; ++t) write_aos_slice(backend, dec, t, true);
            backend.closePulse(&dec, FORCE_CREATE_PULSE);
        }
        // B. same, but the AoS never writes a base of its own: the root base is the only
        //    one available, so it must answer on its own.
        {
            std::filesystem::remove_all(path_of(nobase_uri));
            HDF5Backend backend;
            DataEntryContext dec(nobase_uri);
            open_and_write_root_time(backend, dec);
            for (int t = 0; t < N_SLICES; ++t) write_aos_slice(backend, dec, t, false);
            backend.closePulse(&dec, FORCE_CREATE_PULSE);
        }
        // C. the AoS exists at t=0.8 and t=0.9 only: its two elements are numbered 0 and 1.
        {
            std::filesystem::remove_all(path_of(gapped_uri));
            HDF5Backend backend;
            DataEntryContext dec(gapped_uri);
            open_and_write_root_time(backend, dec);
            write_aos_slice(backend, dec, 8, true);
            write_aos_slice(backend, dec, 9, true);
            backend.closePulse(&dec, FORCE_CREATE_PULSE);
        }

        check_slice(dense_uri, 0.3, 3, "dense AoS");
        check_slice(dense_uri, 0.9, 9, "dense AoS, last slice");
        check_slice(nobase_uri, 0.3, 3, "AoS without own base");
        check_slice(nobase_uri, 0.7, 7, "AoS without own base");
        check_slice(gapped_uri, 0.9, 9, "gapped AoS, written step");
        check_slice(gapped_uri, 0.85, 9, "gapped AoS, between written steps");
        check_gapped_layout(gapped_uri);
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
    if (nfail == 0) {
        std::cout << "OK: homogeneous AoS slice index resolved on the root time base, "
                     "with the gapped-AoS element numbering preserved\n";
        return 0;
    }
    std::cerr << nfail << " check(s) failed\n";
    return 1;
}
