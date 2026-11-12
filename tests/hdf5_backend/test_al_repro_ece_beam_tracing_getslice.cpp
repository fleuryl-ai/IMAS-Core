// @file  test_al_repro_ece_beam_tracing_getslice.cpp
// @brief Reproduction of the ece IDS get_slice failure reported from MATLAB:
//
//     ids_get_slice: al_plugin_read_data: [ALBackendException =
//     Unexpected time vector with size 0]
//     ... in field channel/beam_tracing/beam/power_initial   (a FLT_0D)
//     ... in IDS ece
//
// Structure (ece), inhomogeneous (ids_properties.homogeneous_time = 0):
//     channel            AoS, index = channel     (STATIC)
//       beam_tracing     AoS, index = time sample (DYNAMIC — carries the time base)
//         beam           AoS, index = beam        (STATIC)
//           power_initial FLT_0D (scalar double)
//
// The time base is filled for every (channel c, time sample t).  A global
// put() is followed by ids_get_slice.  This is the exact shape the user asked
// for: a 0-D double signal, in a static AoS (`beam`), in a DYNAMIC AoS
// (`beam_tracing`), in another AoS (`channel`).
//
// BATTERY over the two base layouts:
//   1. CANONICAL (the HLI layout, as in real files — cf.
//      <ids>/time_slice[]&time, one 1-D vector per dynamic AoS):
//      channel(c)/beam_tracing/time.  In this tree every get_slice on it
//      THROWS "ALBackendException = Unexpected time vector with size 0"
//      (closest/linear/previous) — the exact error reported from MATLAB.
//   2. per-sample scalar channel(c)/beam_tracing(t)/time.  In this tree the
//      base is found but every get_slice silently RETURNS EMPTY (type=0/dim=0)
//      even though a plain (non-slice) read of the same element returns the
//      value — the slice path loses the value for a static AoS nested under a
//      dynamic AoS.
//
// Both variants failing is the reported defect; this test only pins them.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <hdf5.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

static const int N_CH = 2;   // channel AoS (static)
static const int N_T  = 3;   // beam_tracing (dynamic, time samples)
static const int N_BE = 2;   // beam AoS (static)

static double base_t(int t)             { return 0.1 * t; }            // 0,0.1,0.2
static double pval(int c, int t, int b) { return 100.0*c + 10.0*t + b; }

// LAYOUT parameter: how the beam_tracing base is stored
//   0 = CANONICAL (the AL/HLI layout, cf. real files: beam_tracing/time is a 1-D
//       sibling vector inside channel(c));
//   1 = per-sample scalar beam_tracing(t)/time inside each element.
static void write_fixture(const std::string& uri, int layout) {
    HDF5Backend backend;
    DataEntryContext dec(uri);
    backend.openPulse(&dec, FORCE_CREATE_PULSE);
    OperationContext op(&dec, "ece", "", WRITE_OP);
    backend.beginAction(&op);
    int homogeneous_time = 0;
    backend.writeData(&op, "ids_properties/homogeneous_time", "", &homogeneous_time,
                      alconst::integer_data, 0, nullptr);

    ArraystructContext chCtx(&op, "channel", "");
    int n_ch = N_CH;
    backend.beginArraystructAction(&chCtx, &n_ch);
    for (int c = 0; c < N_CH; ++c) {
        if (layout == 0) {
            // CANONICAL: 1-D base "channel(c)/beam_tracing/time" (sibling of the AoS)
            std::vector<double> tb(N_T);
            for (int t = 0; t < N_T; ++t) tb[t] = base_t(t);
            int one = N_T;
            backend.writeData(&chCtx, "beam_tracing/time", "time", tb.data(),
                              alconst::double_data, 1, &one);
        }
        ArraystructContext btCtx(&chCtx, "beam_tracing", "time");          // DYNAMIC
        int n_t = N_T;
        backend.beginArraystructAction(&btCtx, &n_t);
        for (int t = 0; t < N_T; ++t) {
            if (layout == 1) {
                double tv = base_t(t);
                backend.writeData(&btCtx, "time", "time", &tv, alconst::double_data, 0, nullptr);
            }
            ArraystructContext beCtx(&btCtx, "beam", "");                   // STATIC in dynamic
            int n_be = N_BE;
            backend.beginArraystructAction(&beCtx, &n_be);
            for (int b = 0; b < N_BE; ++b) {
                double pi = pval(c, t, b);
                backend.writeData(&beCtx, "power_initial", "time", &pi, alconst::double_data, 0, nullptr);
                if (b < N_BE - 1) beCtx.nextIndex(1);
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

// Plain (non-slice) read of one concrete (ch, t, beam) element.  Proves the
// data is stored and reachable outside the slice-interpolation path.
static bool plain_read(const std::string& uri, int ch, int t, int beam, double& out, std::string& why) {
    HDF5Backend backend;
    DataEntryContext dec(uri);
    OperationContext op(&dec, "ece", "", READ_OP);      // plain read, no slice_op
    bool c=false,bt=false,be=false;
    ArraystructContext chCtx(&op, "channel", "");
    ArraystructContext btCtx(&chCtx, "beam_tracing", "time");
    ArraystructContext beCtx(&btCtx, "beam", "");
    bool ok=false;
    auto unwind = [&](){
        if (be) { try{backend.endAction(&beCtx);}catch(...){} }
        if (bt) { try{backend.endAction(&btCtx);}catch(...){} }
        if (c ) { try{backend.endAction(&chCtx);}catch(...){} }
        try { backend.endAction(&op); } catch (...) {}
        try { backend.closePulse(&dec, OPEN_PULSE); } catch (...) {}
    };
    try {
        backend.openPulse(&dec, OPEN_PULSE);
        backend.beginAction(&op);
        int n=0;
        backend.beginArraystructAction(&chCtx,&n); c=true;
        for (int i=0;i<ch;++i) chCtx.nextIndex(1);      // to channel[ch]
        backend.beginArraystructAction(&btCtx,&n); bt=true;
        for (int i=0;i<t;++i) btCtx.nextIndex(1);       // to beam_tracing[t]
        backend.beginArraystructAction(&beCtx,&n); be=true;
        for (int i=0;i<beam;++i) beCtx.nextIndex(1);    // to beam[beam]
        void* data=nullptr; int type=0, dim=0; int sz[H5S_MAX_RANK]={0};
        backend.readData(&beCtx, "power_initial", "time", &data, &type, &dim, sz);
        if (type==alconst::double_data && data) { out = *(double*)data; free(data); ok=true; }
        else { free(data); why = "plain read returned non-scalar (type="+std::to_string(type)+")"; }
    } catch (const std::exception& e) {
        why = e.what();
    }
    unwind();
    return ok;
}

// ids_get_slice: read power_initial at time t_req using `interp`.
static bool slice_read(const std::string& uri, int ch, int beam, double t_req, int interp, const char* iname,
                       double& got, std::string& how) {
    HDF5Backend backend;
    DataEntryContext dec(uri);
    OperationContext op(&dec, "ece", READ_OP, alconst::slice_op, t_req, interp);
    bool c=false,bt=false,be=false;
    ArraystructContext chCtx(&op, "channel", "");
    ArraystructContext btCtx(&chCtx, "beam_tracing", "time");
    ArraystructContext beCtx(&btCtx, "beam", "");
    got = -1e30;
    bool okret = false;
    auto unwind = [&](){
        if (be) { try{backend.endAction(&beCtx);}catch(...){} }
        if (bt) { try{backend.endAction(&btCtx);}catch(...){} }
        if (c ) { try{backend.endAction(&chCtx);}catch(...){} }
        try { backend.endAction(&op); } catch (...) {}
        try { backend.closePulse(&dec, OPEN_PULSE); } catch (...) {}
    };
    try {
        backend.openPulse(&dec, OPEN_PULSE);
        backend.beginAction(&op);
        int n=0;
        backend.beginArraystructAction(&chCtx,&n); c=true;
        for (int i=0;i<ch;++i) chCtx.nextIndex(1);       // channel[ch] (static)
        backend.beginArraystructAction(&btCtx,&n); bt=true;  // beam_tracing (dynamic, slice-resolved)
        backend.beginArraystructAction(&beCtx,&n); be=true;
        for (int i=0;i<beam;++i) beCtx.nextIndex(1);    // beam[beam] (static)
        void* data=nullptr; int type=alconst::double_data, dim=0; int sz[H5S_MAX_RANK]={0};
        backend.readData(&beCtx, "power_initial", "time", &data, &type, &dim, sz);
        if (type==alconst::double_data && data) got = *(double*)data;
        free(data);
        how = "type=" + std::to_string(type) + " dim=" + std::to_string(dim);
        okret = (type==alconst::double_data && data != nullptr);
    } catch (const std::exception& e) {
        how = std::string("THREW: ") + e.what();
        if (std::string(e.what()).find("size 0") != std::string::npos) how += "   <<< SIZE-0 BUG";
    }
    unwind();
    return okret;
}

int main() {
    int failures = 0;
    try {
        // bases are base_t = {0, 0.1, 0.2}.  Pick t so each interpolation has an
        // unambiguous answer independent of tie-breaking:
        //   closest  t=0.11 -> nearest sample is index 1 (t=0.1)
        //   linear   t=0.15 -> midpoint of index 1 (t=0.1) and index 2 (t=0.2)
        //   previous t=0.15 -> last sample with base <= 0.15 is index 1 (t=0.1)
        struct Probe { int ch, be; double t; int interp; const char* name; } probes[] = {
            {0, 0, 0.11, alconst::closest_interp,  "closest "},
            {0, 1, 0.15, alconst::linear_interp,   "linear  "},
            {1, 0, 0.15, alconst::previous_interp, "previous"},
        };

        // Two battery variants: the SAME structure, two base layouts.
        struct Layout { const char* tag; const char* uri; int which; } layouts[] = {
            { "CANONICAL base: beam_tracing/time (1-D sibling) ", "imas:hdf5?path=./test_db_repro_ece_sib",    0 },
            { "per-sample   base: beam_tracing(t)/time          ", "imas:hdf5?path=./test_db_repro_ece_persmp", 1 },
        };

        for (auto& L : layouts) {
            std::cout << "============================================================\n";
            std::cout << "VARIANT: " << L.tag << "\n";
            std::cout << "[ece] channel(2) / beam_tracing(3, dynamic) / beam(2) / power_initial(0D), ht=0\n";
            write_fixture(L.uri, L.which);
            std::cout << "[fixture written]\n";

            // Probe: plain read of a known element — the data IS stored and addressable.
            double v; std::string why;
            if (plain_read(L.uri, 0,0,0, v, why) && std::fabs(v - pval(0,0,0)) < 1e-9)
                std::cout << "  [plain read] ch0/tr0/beam0 power_initial = " << v << "  (OK, data is there)\n";
            else { ++failures; std::cout << "  [plain read] ch0/tr0/beam0 FAILED: " << why << "\n"; }

            std::cout << "--- ids_get_slice on channel/beam_tracing/beam/power_initial ---\n";
            for (auto& p : probes) {
                double got; std::string how;
                bool okret = slice_read(L.uri, p.ch, p.be, p.t, p.interp, p.name, got, how);
                // expected value per interpolation (bases base_t = {0, 0.1, 0.2}):
                //   closest  t=0.11 -> sample index 1 (t=0.1)
                //   linear   t=0.15 -> midpoint of index 1 and index 2
                //   previous t=0.15 -> last sample with base <= 0.15 = index 1
                double expected;
                if (p.interp == alconst::linear_interp) {
                    double tlo = base_t(1), thi = base_t(2);
                    double f = std::min(1.0, std::max(0.0, (p.t - tlo) / (thi - tlo)));
                    expected = (1.0 - f)*pval(p.ch,1,p.be) + f*pval(p.ch,2,p.be);
                } else if (p.interp == alconst::previous_interp) {
                    int ip = 0; for (int k = 1; k < N_T; ++k) if (base_t(k) <= p.t + 1e-9) ip = k;
                    expected = pval(p.ch, ip, p.be);
                } else { // closest
                    int ic = 0; for (int k = 1; k < N_T; ++k)
                        if (std::fabs(base_t(k) - p.t) < std::fabs(base_t(ic) - p.t)) ic = k;
                    expected = pval(p.ch, ic, p.be);
                }
                bool good = okret && how.find("size 0") == std::string::npos && std::fabs(got - expected) < 1e-9;
                std::cout << "  [get_slice] ch" << p.ch << "/beam" << p.be << " t=" << p.t << " " << p.name
                          << " : " << (okret ? ("val=" + std::to_string(got) + " | " + how) : how)
                          << "  expected=" << expected
                          << (good ? "   OK" : "   ** FAIL **") << "\n";
                if (!good) ++failures;
            }
        }

        std::cout << "\n" << (failures ? ">>> REPRODUCED: " + std::to_string(failures) + " defective read(s) across the two layouts <<<"
                                      : "all reads OK (bug NOT reproduced in this tree)") << "\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "outer error: " << e.what() << "\n";
        return 1;
    }
}
