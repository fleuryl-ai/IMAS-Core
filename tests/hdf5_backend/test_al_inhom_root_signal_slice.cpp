// @file  test_al_inhom_root_signal_slice.cpp
// @brief Inhomogeneous IDS (homogeneous_time = 0) with a dynamic AoS AND a root-level
//        signal that owns its time base ("<sig>/time"), appended through ten put_slice:
//        a slice read must resolve that declared base, not go looking for a root "time"
//        that such an IDS never writes.
//
// Before the fix the read died inside DataInterpolation with
// "Unexpected time vector with size 0" (src/data_interpolation.cpp:119): with an
// OperationContext (no AoS to anchor on), getTimeValues() asked the engine for a root
// "time" and ignored the timebase path the caller declared.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <hdf5.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <filesystem>

static int nfail = 0;
#include "fixtures/check.h"

const std::string URI = "imas:hdf5?path=./test_db_inhom_root_signal_slice";

static const int N_SLICES = 10;
static const int N_POINTS = 5;

static void write_fixture() {
    std::filesystem::remove_all(URI.substr(URI.find("path=") + 5));
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, FORCE_CREATE_PULSE);

    {   // ids_properties, in one global action
        OperationContext op(&dec, "test_ids", "", WRITE_OP);
        backend.beginAction(&op);
        int homogeneous_time = 0;   // INHOMOGENEOUS
        backend.writeData(&op, "ids_properties/homogeneous_time", "", &homogeneous_time,
                          alconst::integer_data, 0, nullptr);
        backend.endAction(&op);
    }

    for (int t = 0; t < N_SLICES; ++t) {   // one put_slice per time step
        OperationContext op(&dec, "test_ids", WRITE_OP, alconst::slice_op,
                            alconst::undefined_time, alconst::undefined_interp);
        backend.beginAction(&op);
        double tv = 0.1 * t;

        // the dynamic AoS, base on the AoS itself
        int size_B = 1;
        ArraystructContext ctxB(&op, "B", "time");
        backend.beginArraystructAction(&ctxB, &size_B);
        backend.writeData(&ctxB, "time", "", &tv, alconst::double_data, 0, nullptr);
        std::vector<double> aos_sig(N_POINTS);
        for (int k = 0; k < N_POINTS; ++k) aos_sig[k] = 100.0 + t + 0.1 * k;
        int aos_shape[1] = {N_POINTS};
        backend.writeData(&ctxB, "sig", "time", aos_sig.data(), alconst::double_data, 1, aos_shape);
        backend.endAction(&ctxB);

        // the root-level signal, owning its base as a sibling leaf (sig1/time).
        // Outside a dynamic AoS the writer takes the LAST written dim as the time
        // axis, so one slice of a 1D signal is written as dim=2 [points, 1].
        backend.writeData(&op, "sig1/time", "time", &tv, alconst::double_data, 0, nullptr);
        std::vector<double> root_sig(N_POINTS);
        for (int k = 0; k < N_POINTS; ++k) root_sig[k] = t + 0.5 * k;
        int shape[2] = {N_POINTS, 1};
        backend.writeData(&op, "sig1/data", "time", root_sig.data(), alconst::double_data, 2, shape);

        backend.endAction(&op);
    }
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
}

// get_slice() on a node reached through the dynamic AoS (the shape that already worked)
static void check_inside_aos(double t_req) {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, OPEN_PULSE);
    OperationContext op(&dec, "test_ids", READ_OP, alconst::slice_op, t_req, alconst::closest_interp);
    backend.beginAction(&op);

    int size_B = 0;
    ArraystructContext ctxB(&op, "B", "time");
    backend.beginArraystructAction(&ctxB, &size_B);

    void* data = nullptr;
    int type = alconst::double_data, dim = 0;
    int size[H5S_MAX_RANK];
    backend.readData(&ctxB, "sig", "time", &data, &type, &dim, size);
    int slice = (int)std::lround(t_req * 10.0);
    CHECK(dim == 1 && size[0] == N_POINTS, "inside the AoS: unexpected shape");
    if (dim == 1 && size[0] == N_POINTS) {
        double* v = (double*)data;
        CHECK(std::abs(v[0] - (100.0 + slice)) < 1e-9, "inside the AoS: wrong slice");
    }
    free(data);

    backend.endAction(&ctxB);
    backend.endAction(&op);
    backend.closePulse(&dec, OPEN_PULSE);
}

// get_slice() on the root-level signal, naming its base as the dictionary declares it
static void check_root_signal(const char* path, const char* timebase, double t_req, int interp) {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, OPEN_PULSE);
    OperationContext op(&dec, "test_ids", READ_OP, alconst::slice_op, t_req, interp);
    backend.beginAction(&op);

    void* data = nullptr;
    int type = alconst::double_data, dim = 0;
    int size[H5S_MAX_RANK];
    backend.readData(&op, path, timebase, &data, &type, &dim, size);
    int slice = (int)std::lround(t_req * 10.0);
    // Outside a dynamic AoS the reader promotes a slice dimension of size 1 (same
    // shape the homogeneous IDS gives for this write pattern), so the slice of a
    // N-point signal comes back as [N, 1].
    std::string shape = std::to_string(dim);
    for (int k = 0; k < dim && k < H5S_MAX_RANK; ++k) shape += "x" + std::to_string(size[k]);
    CHECK(dim == 2 && size[0] == N_POINTS && size[1] == 1,
          std::string(path) + " tb=\"" + timebase + "\": unexpected shape " + shape +
              " (want " + std::to_string(N_POINTS) + "x1)");
    if (dim == 2 && size[0] == N_POINTS && size[1] == 1) {
        double* v = (double*)data;
        for (int k = 0; k < N_POINTS; ++k)
            CHECK(std::abs(v[k] - (slice + 0.5 * k)) < 1e-9,
                  std::string(path) + " tb=\"" + timebase + "\": wrong value at point " +
                      std::to_string(k));
    }
    free(data);

    backend.endAction(&op);
    backend.closePulse(&dec, OPEN_PULSE);
}

// Reading the base itself as a node: one time point per slice, and a time base is
// never given the promoted slice dimension, so the slice stays a 1-point 1D array.
static void check_timebase_leaf(const char* path, const char* timebase, double t_req) {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, OPEN_PULSE);
    OperationContext op(&dec, "test_ids", READ_OP, alconst::slice_op, t_req, alconst::closest_interp);
    backend.beginAction(&op);

    void* data = nullptr;
    int type = alconst::double_data, dim = 0;
    int size[H5S_MAX_RANK];
    backend.readData(&op, path, timebase, &data, &type, &dim, size);
    int slice = (int)std::lround(t_req * 10.0);
    std::string shape = "dim=" + std::to_string(dim) + " [";
    for (int k = 0; k < dim && k < H5S_MAX_RANK; ++k) shape += (k ? "," : "") + std::to_string(size[k]);
    shape += "]";
    CHECK(dim == 1 && size[0] == 1,
          std::string(path) + " tb=\"" + timebase + "\": unexpected shape " + shape + " (want dim=1 [1])");
    if (dim == 1 && size[0] == 1)
        CHECK(std::abs(*(double*)data - 0.1 * slice) < 1e-9,
              std::string(path) + ": wrong time point");
    free(data);

    backend.endAction(&op);
    backend.closePulse(&dec, OPEN_PULSE);
}

int main() {
    try {
        write_fixture();

        // The dictionary declares the base either as an absolute path or relative to
        // the signal's container; both are handed over verbatim by the AL and both
        // must resolve to sig1/time. The root "time" is not a fallback here: an
        // inhomogeneous IDS does not have one.
        check_root_signal("sig1/data", "sig1/time", 0.7, alconst::closest_interp);
        check_root_signal("sig1/data", "sig1/time", 0.7, alconst::linear_interp);
        check_root_signal("sig1/data", "sig1/time", 0.2, alconst::closest_interp);
        check_root_signal("sig1/data", "time", 0.7, alconst::closest_interp);
        check_timebase_leaf("sig1/time", "time", 0.7);
        check_inside_aos(0.7);

        if (nfail) {
            std::cout << nfail << " check(s) FAILED\n";
            return 1;
        }
        std::cout << "all checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
