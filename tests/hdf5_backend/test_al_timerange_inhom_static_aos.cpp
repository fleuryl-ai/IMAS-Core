// @file  test_al_timerange_inhom_static_aos.cpp
// @brief TIMERANGE_OP on a dynamic signal that lives inside a STATIC AoS whose
//        time base sits on the AoS element (ids_properties/homogeneous_time = 0)
//        — the shape of reflectometer_profile/channel[]/n_e.
//
// Expected AL behaviour: the time window filters that signal, per element, using
// the element's own time base. Same rule as the other time bases; the request
// window must not be ignored, and must not make the read return nothing.
//
// What the v2 backend did instead: TimeRangeReadStrategy::read_ND_Data resolved
// the time values WITHOUT the node's timebasename (slice_read_strategy.cpp passes
// it, the timerange strategy did not), so for a static AoS the resolution found no
// time base, gave an empty time vector, and every read — signal AND time node —
// returned rc=0, even for a window covering the whole span.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const std::string URI = "imas:hdf5?path=./test_db_timerange_inhom_static_aos";
static const char* IDS = "test_ids";
static const int NCH = 2;

// channel(0): t = 1..10, channel(1): t = 1..20. Values: t*10 + channel index.
static std::vector<double> timebase_of(int j) {
    std::vector<double> tv;
    for (int k = 0; k < (j == 0 ? 10 : 20); ++k) tv.push_back(1.0 + k);
    return tv;
}

static int failures = 0;

static void write_pulse() {
    DataEntryContext dec(URI);
    HDF5Backend backend;
    backend.openPulse(&dec, FORCE_CREATE_PULSE);
    OperationContext oc(&dec, IDS, "", WRITE_OP);
    backend.beginAction(&oc);

    int homogeneous_time = 0;
    backend.writeData(&oc, "ids_properties/homogeneous_time", "", &homogeneous_time,
                      alconst::integer_data, 0, nullptr);

    int asize = NCH;
    ArraystructContext ch(&oc, "channel", "");
    backend.beginArraystructAction(&ch, &asize);
    for (int j = 0; j < asize; ++j) {
        std::vector<double> tv = timebase_of(j);
        std::vector<double> sig(tv.size());
        for (size_t k = 0; k < tv.size(); ++k) sig[k] = tv[k] * 10.0 + j;
        int dim = 1, size[1] = {(int)tv.size()};
        backend.writeData(&ch, "time", "", tv.data(), alconst::double_data, dim, size);
        backend.writeData(&ch, "dyn_sig", "time", sig.data(), alconst::double_data, dim, size);
        if (j + 1 < asize) ch.nextIndex(1);
    }
    backend.endAction(&ch);
    backend.endAction(&oc);
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
}

// Reads one node of the current AoS element and checks it comes back filtered to
// [tmin, tmax] on the element's own time base.
static void check_read(HDF5Backend& backend, ArraystructContext& ch, const char* field,
                       double tmin, double tmax, int j) {
    void* data = nullptr;
    int datatype = alconst::double_data;
    int dim = -99;
    int size[6] = {0, 0, 0, 0, 0, 0};
    int rc = backend.readData(&ch, field, "time", &data, &datatype, &dim, size);

    std::vector<double> tv = timebase_of(j);
    std::vector<double> want;
    for (double t : tv)
        if (t >= tmin && t <= tmax) want.push_back(std::strcmp(field, "time") == 0 ? t : t * 10.0 + j);

    // An empty window is reported as "nothing read" (rc = 0) by this backend;
    // both that and an empty success are acceptable, a non-empty answer is not.
    if (want.empty()) {
        if (rc == 1 && size[0] != 0) {
            std::printf("[FAIL] channel(%d)/%s window [%g, %g]: outside every time base but returned %d samples\n",
                        j, field, tmin, tmax, size[0]);
            ++failures;
        }
    } else if (rc != 1 || dim != 1 || size[0] != (int)want.size()) {
        std::printf("[FAIL] channel(%d)/%s window [%g, %g]: rc=%d dim=%d size[0]=%d, expected rc=1 dim=1 size[0]=%zu\n",
                    j, field, tmin, tmax, rc, dim, size[0], want.size());
        ++failures;
    } else if (data) {
        double* got = (double*)data;
        for (size_t k = 0; k < want.size(); ++k) {
            if (std::abs(got[k] - want[k]) > 1e-9) {
                std::printf("[FAIL] channel(%d)/%s window [%g, %g]: value %zu is %.6g, expected %.6g\n",
                            j, field, tmin, tmax, k, got[k], want[k]);
                ++failures;
                break;
            }
        }
    }
    if (data) free(data);
}

static void read_window(double tmin, double tmax) {
    DataEntryContext dec(URI);
    HDF5Backend backend;
    backend.openPulse(&dec, OPEN_PULSE);
    OperationContext oc(&dec, IDS, READ_OP, alconst::timerange_op, tmin, tmax,
                        std::vector<double>(), alconst::closest_interp);
    backend.beginAction(&oc);

    int asize = -1;
    ArraystructContext ch(&oc, "channel", "");
    backend.beginArraystructAction(&ch, &asize);
    if (asize != NCH) {
        std::printf("[FAIL] channel AoS size is %d, expected %d\n", asize, NCH);
        ++failures;
    }
    for (int j = 0; j < asize; ++j) {
        check_read(backend, ch, "time", tmin, tmax, j);
        check_read(backend, ch, "dyn_sig", tmin, tmax, j);
        if (j + 1 < asize) ch.nextIndex(1);
    }

    backend.endAction(&ch);
    backend.endAction(&oc);
    backend.closePulse(&dec, CLOSE_PULSE);
}

int main() {
    fs::remove_all(URI.substr(7));
    std::printf("=== TEST TIMERANGE_OP on a static AoS with per-element time base ===\n");
    try {
        write_pulse();
        read_window(1.0, 20.0);   // covers everything -> everything must come back
        read_window(4.0, 7.0);    // middle window -> 4 points per channel
        read_window(1.0, 3.0);    // leading window -> 3 points per channel
        read_window(25.0, 30.0);  // outside every time base -> nothing
    } catch (const std::exception& e) {
        std::printf("[FAIL] exception: %s\n", e.what());
        ++failures;
    }
    fs::remove_all(URI.substr(7));
    if (failures) {
        std::printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
