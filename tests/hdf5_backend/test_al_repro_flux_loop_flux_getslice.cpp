// @file  test_al_repro_flux_loop_flux_getslice.cpp
// @brief Production regression test for the magnetics flux_loop(j)/flux/data
//        slice-read bug in an inhomogeneous (homogeneous_time = 0) IDS.
//
// The signal `flux_loop(j)/flux/data` is a 1-D signal whose time base is a
// sibling of the payload, inside the `flux` container:
//
//            flux_loop(j)/flux/data    (payload)
//            flux_loop(j)/flux/time    (time base)
//
// Before the fix, the inhomogeneous static-AoS branch of
// IReadStrategy::getTimeValues (src/hdf5/iread_strategy.h) anchored the base
// name on the AoS element (flux_loop(j)/time) and then on a root `time`,
// never on the signal's own container.  The stored base was therefore never
// found, the empty time vector reached DataInterpolation and threw
//   [ALBackendException = Unexpected time vector with size 0]
// (src/data_interpolation.cpp:119).
//
// The fix tries, in order, the siblings of the data node, the AoS element
// (kept for bases at the element level), and the root of the IDS -- the same
// candidate order the root-signal branch already used for non-AoS signals.
//
// This test writes both loops with their sibling base and asserts that a
// getSlice() at a non-grid time returns the correct closest / previous /
// linear value for each loop.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <hdf5.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

static int nfail = 0;
#include "fixtures/check.h"

const std::string URI = "imas:hdf5?path=./test_db_repro_flux_flux_getslice";

static const int N_LOOPS = 2;   // size of the static AoS `flux_loop`
static const int N_TIME  = 5;   // points in the 1-D signal / time base

static void write_fixture() {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, FORCE_CREATE_PULSE);

    OperationContext opCtx(&dec, "magnetics", "", WRITE_OP);
    backend.beginAction(&opCtx);

    int homogeneous_time = 0;   // INHOMOGENEOUS
    backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time,
                      alconst::integer_data, 0, nullptr);

    ArraystructContext fluxLoopCtx(&opCtx, "flux_loop", "");
    int fl_size = N_LOOPS;
    backend.beginArraystructAction(&fluxLoopCtx, &fl_size);

    for (int i = 0; i < N_LOOPS; ++i) {
        std::vector<double> times(N_TIME), payload(N_TIME);
        for (int t = 0; t < N_TIME; ++t) {
            times[t]   = 0.1 * t;              // 0.0, 0.1, 0.2, 0.3, 0.4
            payload[t] = 10.0 * i + t;         // 10*i + step index
        }
        int one[] = {N_TIME};
        // The base is a sibling of the payload, inside `flux` (magnetics layout).
        backend.writeData(&fluxLoopCtx, "flux/time", "time", times.data(),
                          alconst::double_data, 1, one);
        backend.writeData(&fluxLoopCtx, "flux/data", "time", payload.data(),
                          alconst::double_data, 1, one);
        if (i < N_LOOPS - 1) fluxLoopCtx.nextIndex(1);
    }
    backend.endAction(&fluxLoopCtx);
    backend.endAction(&opCtx);
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
}

static double read_slice(int loop_idx, double t_req, int interp, std::string& err) {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    OperationContext op(&dec, "magnetics", READ_OP, alconst::slice_op, t_req, interp);
    ArraystructContext ctxLoop(&op, "flux_loop", "");
    bool aos_begun = false;
    double value = std::nan("");
    void* data = nullptr;
    try {
        backend.openPulse(&dec, OPEN_PULSE);
        backend.beginAction(&op);
        int n = 0;
        backend.beginArraystructAction(&ctxLoop, &n);
        aos_begun = true;
        for (int k = 0; k < loop_idx; ++k) ctxLoop.nextIndex(1);
        int type = alconst::double_data, dim = 0;
        int sz[H5S_MAX_RANK] = {0};
        // field = "flux/data", timebase = "time"  -- the AL passes the base NAME;
        // the READER anchors it on the signal's own container (flux/time).
        backend.readData(&ctxLoop, "flux/data", "time", &data, &type, &dim, sz);
        if (type == alconst::double_data && data)
            value = *(double*)data;
        else
            err = "did not get a double back";
        backend.endAction(&ctxLoop);
        backend.endAction(&op);
    } catch (const std::exception& e) {
        err = std::string(e.what());
        // Unwind the action even on the exception path (skipping this segfaults
        // at exit on the shared backend).
        if (aos_begun) { try { backend.endAction(&ctxLoop); } catch (...) {} }
        try { backend.endAction(&op); } catch (...) {}
    }
    try { backend.closePulse(&dec, OPEN_PULSE); } catch (...) {}
    if (data) free(data);
    return value;
}

int main() {
    try {
        write_fixture();
        // Sample at t = 0.35 so closest/previous hit t = 0.3 (index 3) and
        // linear interpolates between indices 3 and 4.  For loop j:
        //   linear(0.35)  = 10*j + 3.5
        //   closest(0.35) = 10*j + 3.0
        //   previous(0.35)= 10*j + 3.0
        const double t = 0.35;
        const int interps[]   = {alconst::closest_interp, alconst::previous_interp, alconst::linear_interp};
        const char* names[]   = {"closest", "previous", "linear"};

        for (int j = 0; j < N_LOOPS; ++j)
            for (int m = 0; m < 3; ++m) {
                std::string err;
                double got = read_slice(j, t, interps[m], err);
                double want = (interps[m] == alconst::linear_interp) ? 10.0*j + 3.5 : 10.0*j + 3.0;
                std::cout << "  loop " << j << " " << names[m] << "(" << t << "): got "
                          << got << " want " << want
                          << (err.empty() ? "" : ("  [" + err + "]")) << "\n";
                CHECK(err.empty(), std::string("loop ") + std::to_string(j) + " " + names[m]
                        + ": read threw: " + err);
                CHECK(err.empty() && std::abs(got - want) < 1e-9,
                      std::string("loop ") + std::to_string(j) + " " + names[m] + ": got "
                      + std::to_string(got) + " want " + std::to_string(want));
            }

        if (nfail) {
            std::cerr << nfail << " check(s) FAILED\n";
            return 1;
        }
        std::cout << "all checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception in main: " << e.what() << std::endl;
        return 1;
    }
}
