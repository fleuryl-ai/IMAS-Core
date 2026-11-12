// @file test_al_repro_ece_beamtracing_sliceindex_getslice.cpp
// @brief Second axis of the ece inhomogeneous get_slice defect.
//
// getSliceIndex (slice_read_strategy.cpp) resolves, for a nested DYNAMIC AoS
// (`beam_tracing`), the requested time -> ELEMENT slice index by building a
// timebase path from `timed_parent->getTimebasePath()`.  When the HLI declares
// that base with the DICTIONARY-ABSOLUTE name ("channel/beam_tracing/time"),
// sanitisation yields "channel&beam_tracing&time"; the code appends that WHOLE
// string to the AoS path building "<aos>/channel&beam_tracing&time" and
// "<aos>&channel&beam_tracing&time" — neither exists (sz=0) — so the index
// resolves to -1 and the read is PINNED to ELEMENT index 0 (the FIRST slice).
//
// With a CONSTANT per-slice arraystruct size this is invisible (index 0 has the
// same size); with a TIME-VARYING beam count per slice, `ids_get_slice(t)` at a
// non-first slice then returns slice 0's element count instead of slice t's.
// This test pins exactly that: the beam AoS size at each requested slice.
//
// It is the SAME basename-extraction defect class fixed in getTimeValues
// (commit f425866) but on the slice-INDEX resolution path.  It must FAIL before
// the fix and pass after.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <hdf5.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

static const int N_T = 3;
// TIME-VARYING beam count per (channel c, time slice t) — this is legal for
// channel(i)/beam_tracing(time)/beam: beam sits under the time-indexed entry.
static int beams(int c, int t) {
    // channel 0: 2,1,1  |  channel 1: 2,1,1  (slice 0 = 2, slice 1.. = 1)
    static const int V[2][3] = {
        {2,1,1},
        {2,1,1},
    };
    return V[c % 2][t % N_T];
}
static const int N_CH = 2;
static double base_t(int t) { return 0.1 * t; }   // 0.0, 0.1, 0.2

static void write_fixture(const std::string& uri) {
    HDF5Backend backend;
    DataEntryContext dec(uri);
    backend.openPulse(&dec, FORCE_CREATE_PULSE);
    OperationContext op(&dec, "ece", "", WRITE_OP);
    backend.beginAction(&op);
    int ht = 0;
    backend.writeData(&op, "ids_properties/homogeneous_time", "", &ht, alconst::integer_data, 0, nullptr);
    ArraystructContext chCtx(&op, "channel", "");
    int n_ch = N_CH; backend.beginArraystructAction(&chCtx, &n_ch);
    for (int c = 0; c < N_CH; ++c) {
        std::vector<double> tb(N_T); for (int t=0;t<N_T;++t) tb[t]=base_t(t);
        int one = N_T;
        backend.writeData(&chCtx, "beam_tracing/time", "time", tb.data(), alconst::double_data, 1, &one);
        ArraystructContext btCtx(&chCtx, "beam_tracing", "time");
        int n_t = N_T; backend.beginArraystructAction(&btCtx, &n_t);
        for (int t = 0; t < N_T; ++t) {
            int nbe = beams(c, t);
            ArraystructContext beCtx(&btCtx, "beam", "");
            backend.beginArraystructAction(&beCtx, &nbe);
            for (int b = 0; b < nbe; ++b) {
                double v = 100.0*c + 10.0*t + b;
                backend.writeData(&beCtx, "power_initial", "time", &v, alconst::double_data, 0, nullptr);
                if (b < nbe - 1) beCtx.nextIndex(1);
            }
            backend.endAction(&beCtx);
            if (t < N_T - 1) btCtx.nextIndex(1);
        }
        backend.endAction(&btCtx);
        if (c < N_CH - 1) chCtx.nextIndex(1);
    }
    backend.endAction(&chCtx);
    backend.endAction(&op);
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
}

// HLI-shape: dynamic AoS declared with the DICTIONARY-ABSOLUTE timebase
// ("channel/beam_tracing/time"), read at `t_req`.  Returns the beam AoS size
// reported by beginArraystructAction (the defect-visible count).
static int slice_beam_count(const std::string& uri, int ch, double t_req, int interp, std::string& how) {
    HDF5Backend backend;
    DataEntryContext dec(uri);
    OperationContext op(&dec, "ece", READ_OP, alconst::slice_op, t_req, interp);
    bool c=false,bt=false,be=false;
    ArraystructContext chCtx(&op, "channel", "");
    ArraystructContext btCtx(&chCtx, "beam_tracing", "channel/beam_tracing/time");  // HLI
    ArraystructContext beCtx(&btCtx, "beam", "");
    int got = -1;
    auto unwind = [&](){
        if (be){try{backend.endAction(&beCtx);}catch(...){}}
        if (bt){try{backend.endAction(&btCtx);}catch(...){}}
        if (c ){try{backend.endAction(&chCtx);}catch(...){}}
        try{backend.endAction(&op);}catch(...){}
        try{backend.closePulse(&dec, OPEN_PULSE);}catch(...){}
    };
    try {
        backend.openPulse(&dec, OPEN_PULSE);
        backend.beginAction(&op);
        int n=0;
        backend.beginArraystructAction(&chCtx,&n); c=true;
        for (int i=0;i<ch;++i) chCtx.nextIndex(1);
        backend.beginArraystructAction(&btCtx,&n); bt=true;
        int nbe=0;
        backend.beginArraystructAction(&beCtx,&nbe); be=true;
        got = nbe;
        how = "count=" + std::to_string(nbe);
    } catch (const std::exception& e) { how = "THREW: " + std::string(e.what()); if (std::string(e.what()).find("size 0")!=std::string::npos) how += " <<< SIZE-0"; }
    unwind();
    return got;
}

int main(){
    int failures = 0;
    const std::string uri = "imas:hdf5?path=./test_db_ece_sliceidx";
    try {
        write_fixture(uri);
        std::cout << "[written] channel("<<N_CH<<")/beam_tracing("<<N_T<<",dynamic,ht=0)/beam(count varies: slice) / power_initial(0D)\n";
        // For each channel and each target slice, the beam count must equal the
        // per-slice written count.  Pick the exact slice time to avoid interpolation.
        struct R { int ch, t; } rs[] = { {0,0},{0,1},{0,2},{1,0},{1,1},{1,2} };
        for (auto& r : rs) {
            double got; std::string how;
            int cnt = slice_beam_count(uri, r.ch, base_t(r.t), alconst::closest_interp, how);
            int exp = beams(r.ch, r.t);
            bool ok = (cnt == exp);
            std::cout << "  ch" << r.ch << " t=" << base_t(r.t) << " beam_count got=" << cnt
                      << " expected=" << exp << (ok?"  OK":"  ** FAIL ** (returned slice-0 count)") << "\n";
            if (!ok) ++failures;
        }
        std::cout << (failures? (">>> REPRODUCED: " + std::to_string(failures) + " slice-index read(s) return the wrong slice <<<\n")
                              : "all slice-index reads OK\n");
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << "outer: " << e.what() << "\n"; return 1; }
}
