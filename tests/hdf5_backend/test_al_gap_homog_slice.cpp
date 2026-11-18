// @file  test_al_gap_homog_slice.cpp
// @brief Verifies slice reads (closest / previous / linear) of a dynamic
//        signal in an IDS with HOMOGENEOUS time
//        (ids_properties&homogeneous_time = 1) that has gaps (some slices
//        missing).
//
// The inhomogeneous case (homogeneous_time = 0) is already covered by
// test_gap_{closest_prev,linear,timerange}_read.cpp. Here there is NO dynamic
// AoS: the signal is a standalone dynamic datum at the root, attached to the
// root timebase "time", written step by step (APPEND). This therefore
// validates CASE 2 of the writeDataSlicesImpl path and the homogeneous read.
//
// Values (base + i): sigA = 100+i, sigB = 200+i, sigC = 300+i, time = 0.1*i
//   - sigA: gap at i=2  -> present (0,100),(0.1,101),(0.3,103),(0.4,104)
//   - sigB: gaps i=1,3  -> present (0,200),(0.2,202),(0.4,204)
//   - sigC: complete    -> present (0,300)..(0.4,304)
// When the requested slice is absent, the returned value must be that of an
// EXISTING slice; the linear interpolation must never go through an absent
// slice.
#include "al_context.h"
#include "al_defs.h"
#include "al_const.h"
#include "hdf5_backend.h"
#include <iostream>
#include <cmath>

const std::string URI = "imas:hdf5?path=./test_db_gap_homog_slice";

static void write_fixture() {
    // Init phase: creation of the IDS + homogeneous_time = 1 marker
    {
        HDF5Backend backend;
        DataEntryContext dec(URI);
        backend.openPulse(&dec, FORCE_CREATE_PULSE);
        OperationContext opInit(&dec, "test_ids", "", WRITE_OP);
        backend.beginAction(&opInit);
        int homogeneous_time = 1; // HOMOGENEOUS
        backend.writeData(&opInit, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);
        backend.endAction(&opInit);
        backend.closePulse(&dec, FORCE_CREATE_PULSE);
    }

    // 5 temporal steps, each an append (slice_op).
    // The root timebase "time" is built by these additions (1 point/step).
    int n = 5;
    for (int i = 0; i < n; ++i) {
        HDF5Backend backend;
        DataEntryContext dec(URI);
        backend.openPulse(&dec, OPEN_PULSE);

        double t = 0.1 * i;
        double t_undef = alconst::undefined_time;
        int interp_undef = alconst::undefined_interp;
        OperationContext opCtx(&dec, "test_ids", WRITE_OP, alconst::slice_op, t_undef, interp_undef);
        backend.beginAction(&opCtx);

        // root timebase: 1 point per step
        backend.writeData(&opCtx, "time", "time", &t, alconst::double_data, 0, nullptr);

        if (i != 2)                  { double v = 100.0 + i; backend.writeData(&opCtx, "sigA", "time", &v, alconst::double_data, 0, nullptr); } // gap i=2
        if (i != 1 && i != 3)        { double v = 200.0 + i; backend.writeData(&opCtx, "sigB", "time", &v, alconst::double_data, 0, nullptr); } // gaps i=1,3
        { double v = 300.0 + i;      backend.writeData(&opCtx, "sigC", "time", &v, alconst::double_data, 0, nullptr); }                        // complete

        backend.endAction(&opCtx);
        backend.closePulse(&dec, OPEN_PULSE);
    }
}

static double read_scalar(int interp, double req_time,
                          const char* sig, bool* available) {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, OPEN_PULSE);
    double t_undef = alconst::undefined_time;
    OperationContext opCtx(&dec, "test_ids", READ_OP, alconst::slice_op, req_time, interp);
    backend.beginAction(&opCtx);

    void* data = nullptr;
    int type = alconst::double_data;
    int dim = 0;
    int size[1] = {0};
    int avail = backend.readData(&opCtx, sig, "time", &data, &type, &dim, size);
    if (available) *available = (avail == 0 ? false : true);

    double val = 0.0;
    if (data) { val = *(double*)data; free(data); }

    backend.endAction(&opCtx);
    backend.closePulse(&dec, OPEN_PULSE);
    return val;
}

static int nfail = 0;
#include "fixtures/check.h"

int main() {
    try {
        write_fixture();
        auto check = [&](double got, double exp, const char* what) {
            CHECK(std::abs(got - exp) < 1e-9, std::string(what) + ": got " + std::to_string(got) + " expected " + std::to_string(exp));
        };

        // ---- CLOSEST (absent slice -> closest existing one, tie -> lowest time) ----
        check(read_scalar(alconst::closest_interp, 0.2, "sigA", nullptr), 101.0, "closest sigA@0.2 (gap -> t=0.1)");
        check(read_scalar(alconst::closest_interp, 0.3, "sigA", nullptr), 103.0, "closest sigA@0.3 (present)");
        check(read_scalar(alconst::closest_interp, 0.1, "sigB", nullptr), 200.0, "closest sigB@0.1 (gap -> t=0)");
        check(read_scalar(alconst::closest_interp, 0.3, "sigB", nullptr), 202.0, "closest sigB@0.3 (gap -> t=0.2)");
        check(read_scalar(alconst::closest_interp, 0.2, "sigC", nullptr), 302.0, "closest sigC@0.2 (complete)");

        // ---- PREVIOUS (the last available value <= t) ----
        check(read_scalar(alconst::previous_interp, 0.2, "sigA", nullptr), 101.0, "prev sigA@0.2 (gap -> t=0.1)");
        check(read_scalar(alconst::previous_interp, 0.3, "sigB", nullptr), 202.0, "prev sigB@0.3 (gap -> t=0.2)");
        check(read_scalar(alconst::previous_interp, 0.4, "sigB", nullptr), 204.0, "prev sigB@0.4 (present)");
        check(read_scalar(alconst::previous_interp, 0.4, "sigC", nullptr), 304.0, "prev sigC@0.4 (present)");

        // ---- LINEAR (interp. between EXISTING slices only) ----
        // sigA@0.2 (gap) : lower=t=0.1(101), upper=t=0.3(103) -> f=0.5 -> 102
        check(read_scalar(alconst::linear_interp, 0.2, "sigA", nullptr), 102.0, "linear sigA@0.2 (gap)");
        // sigB@0.1 (gap) : lower=t=0(200), upper=t=0.2(202) -> f=0.5 -> 201
        check(read_scalar(alconst::linear_interp, 0.1, "sigB", nullptr), 201.0, "linear sigB@0.1 (gap)");
        // sigB@0.3 (gap) : lower=t=0.2(202), upper=t=0.4(204) -> f=0.5 -> 203
        check(read_scalar(alconst::linear_interp, 0.3, "sigB", nullptr), 203.0, "linear sigB@0.3 (gap)");
        // sigC@0.2 complete: no gap -> 302
        check(read_scalar(alconst::linear_interp, 0.2, "sigC", nullptr), 302.0, "linear sigC@0.2 (present)");

        // ---- Signal never written -> unavailable ----
        bool avail = false;
        read_scalar(alconst::closest_interp, 0.2, "sig_never", &avail);
        CHECK(!avail, "closest sig_never should be unavailable");
        avail = false;
        read_scalar(alconst::linear_interp, 0.2, "sig_never", &avail);
        CHECK(!avail, "linear sig_never should be unavailable");

        if (nfail == 0) { std::cout << "[OK] gap homog slice read passed\n"; return 0; }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n"; return 1;
    }
    return 1;
}
