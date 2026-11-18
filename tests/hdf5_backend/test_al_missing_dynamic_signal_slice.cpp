// @file  test_al_missing_dynamic_signal_slice.cpp
// @brief Inhomogeneous IDS (homogeneous_time = 0) where a dynamic signal the dictionary
//        declares was never written -- neither its data nor its time base. A slice read
//        must answer "not available" (readData returns 0, the node stays undefined) the
//        way the 1.0 reader does, and must not take the whole IDS down with it.
//
// Before the fix the read died inside DataInterpolation with
// "Unexpected time vector with size 0" (src/data_interpolation.cpp:119): the time base
// of the absent signal was resolved before anything asked whether the signal itself is
// filed in the index, and an absent signal has no base to find either -- so the file
// reported the missing base instead of the missing data.
//
// Measured on shot 54924 (V2): 9 of the 41 IDS are in this shape (barometry asks for
// code/output_flag with a declared "/time" that the pulse never wrote) and all 100 of
// their getSlice calls were refused, while the 1.0 leg served the same IDS with the
// node simply left undefined.
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

const std::string URI = "imas:hdf5?path=./test_db_missing_dynamic_signal_slice";

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

        // A dynamic AoS with its own base, read for the non-regression side.
        int size_B = 1;
        ArraystructContext ctxB(&op, "B", "time");
        backend.beginArraystructAction(&ctxB, &size_B);
        backend.writeData(&ctxB, "time", "", &tv, alconst::double_data, 0, nullptr);
        std::vector<double> aos_sig(N_POINTS);
        for (int k = 0; k < N_POINTS; ++k) aos_sig[k] = 100.0 + t + 0.1 * k;
        int aos_shape[1] = {N_POINTS};
        backend.writeData(&ctxB, "sig", "time", aos_sig.data(), alconst::double_data, 1, aos_shape);
        backend.endAction(&ctxB);

        // A root-level signal that owns its base as a sibling leaf, written.
        backend.writeData(&op, "sig1/time", "time", &tv, alconst::double_data, 0, nullptr);
        std::vector<double> root_sig(N_POINTS);
        for (int k = 0; k < N_POINTS; ++k) root_sig[k] = t + 0.5 * k;
        int shape[2] = {N_POINTS, 1};
        backend.writeData(&op, "sig1/data", "time", root_sig.data(), alconst::double_data, 2, shape);

        // sig2/data and sig2/time are never written: the shape that used to throw.
        backend.endAction(&op);
    }
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
}

// The absent signal: no exception, and the read reports "not available".
static void check_absent_signal(const char* path, const char* timebase, double t_req) {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, OPEN_PULSE);
    OperationContext op(&dec, "test_ids", READ_OP, alconst::slice_op, t_req, alconst::closest_interp);
    backend.beginAction(&op);

    void* data = nullptr;
    int type = alconst::double_data, dim = 0;
    int size[H5S_MAX_RANK];
    int available = -1;
    bool threw = false;
    try {
        available = backend.readData(&op, path, timebase, &data, &type, &dim, size);
    } catch (const std::exception& e) {
        threw = true;
        std::cout << "  readData(\"" << path << "\", \"" << timebase << "\") threw: "
                  << e.what() << "\n";
    }
    CHECK(!threw, std::string(path) + ": a signal this file does not hold must not throw");
    CHECK(available == 0, std::string(path) + ": expected \"not available\", got " +
                              std::to_string(available));
    free(data);

    backend.endAction(&op);
    backend.closePulse(&dec, OPEN_PULSE);
}

// Non-regression: a signal that IS written, with its base, still reads.
static void check_written_root_signal(const char* path, const char* timebase, double t_req) {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, OPEN_PULSE);
    OperationContext op(&dec, "test_ids", READ_OP, alconst::slice_op, t_req, alconst::closest_interp);
    backend.beginAction(&op);

    void* data = nullptr;
    int type = alconst::double_data, dim = 0;
    int size[H5S_MAX_RANK];
    int available = backend.readData(&op, path, timebase, &data, &type, &dim, size);
    int slice = (int)std::lround(t_req * 10.0);
    CHECK(available == 1, std::string(path) + ": a written signal must be available");
    // Outside a dynamic AoS the reader promotes a slice dimension of size 1.
    CHECK(dim == 2 && size[0] == N_POINTS && size[1] == 1,
          std::string(path) + ": unexpected shape from a written signal");
    if (dim == 2 && size[0] == N_POINTS && size[1] == 1) {
        double* v = (double*)data;
        for (int k = 0; k < N_POINTS; ++k)
            CHECK(std::abs(v[k] - (slice + 0.5 * k)) < 1e-9,
                  std::string(path) + ": wrong value at point " + std::to_string(k));
    }
    free(data);

    backend.endAction(&op);
    backend.closePulse(&dec, OPEN_PULSE);
}

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
    int available = backend.readData(&ctxB, "sig", "time", &data, &type, &dim, size);
    int slice = (int)std::lround(t_req * 10.0);
    CHECK(available == 1, "inside the AoS: a written signal must be available");
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

// Absent signal reached through a dynamic AoS: the family has no row with this tail,
// so the same answer must come back without walking the time base.
static void check_absent_signal_in_aos(double t_req) {
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
    int available = -1;
    bool threw = false;
    try {
        available = backend.readData(&ctxB, "absent", "time", &data, &type, &dim, size);
    } catch (const std::exception& e) {
        threw = true;
        std::cout << "  readData(B/absent) threw: " << e.what() << "\n";
    }
    CHECK(!threw, "B/absent: a signal this file does not hold must not throw");
    CHECK(available == 0, "B/absent: expected \"not available\", got " + std::to_string(available));
    free(data);

    backend.endAction(&ctxB);
    backend.endAction(&op);
    backend.closePulse(&dec, OPEN_PULSE);
}

int main() {
    try {
        write_fixture();

        check_written_root_signal("sig1/data", "sig1/time", 0.7);
        check_inside_aos(0.7);

        check_absent_signal("sig2/data", "sig2/time", 0.7);
        check_absent_signal("sig2/data", "time", 0.7);
        check_absent_signal_in_aos(0.7);

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
