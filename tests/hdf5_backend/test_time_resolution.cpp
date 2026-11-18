// test_time_resolution.cpp
// Pins the exact slice-resolution contract of DataInterpolation before/after the
// linear scans are replaced by binary searches.
//
// The oracle below is a verbatim copy of the ORIGINAL implementation (forward
// scan for the bracketing slice, backward scan for tmax skipping the -9e40
// sentinel).  The tests therefore describe the behaviour in force when this file
// was written; any optimisation of these lookups must keep every column of the
// comparison identical.
//
// Precondition of the optimised lookups: a timebase is strictly increasing, with
// at most a trailing run of IMAS invalid values (-9e40).  A sentinel in the
// middle of a timebase makes the vector non-monotonic; the original scan had no
// defined meaning there either, so no case covers it.

#include "data_interpolation.h"
#include "al_const.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

const double kInvalid = -9.e40;
int g_checks = 0;

void fail(const std::string& what) {
    std::cerr << "FAIL: " << what << std::endl;
    assert(false);
}

// ---------------------------------------------------------------------------
// Oracles: the original algorithms, copied as-is
// ---------------------------------------------------------------------------

int oracle_slices(double t, const std::vector<double>& tv, std::map<std::string, int>& out, int interp) {
    int slice_sup = (int)tv.size() - 1;
    int slice_inf = 0;
    if (slice_sup > 0) slice_inf = slice_sup - 1;

    for (size_t i = 0; i < tv.size(); i++) {
        if (tv[i] >= t) {
            slice_sup = (int)i;
            if (interp != alconst::closest_interp) {
                if (slice_sup == 0 && tv.size() > 1) slice_sup = 1;
            }
            if (slice_sup > 0) slice_inf = slice_sup - 1;
            else               slice_inf = slice_sup;
            break;
        }
    }
    out[SLICE_INF] = slice_inf;
    out[SLICE_SUP] = slice_sup;

    int it = interp;
    if (it == alconst::undefined_interp) it = alconst::closest_interp;
    switch (it) {
    case alconst::closest_interp:
        return (std::fabs(t - tv[slice_sup]) <= std::fabs(t - tv[slice_inf])) ? slice_sup : slice_inf;
    case alconst::previous_interp: {
        int closest = slice_inf;
        if (slice_sup > 0 && ((std::fabs(t - tv[slice_sup]) < std::numeric_limits<double>::epsilon()) ||
                              (t > tv[slice_sup])))
            closest = slice_sup;
        return closest;
    }
    case alconst::linear_interp:
        return slice_inf;
    }
    return -1;
}

// Mirrors getTimeRangeIndices for every dtime shape, including the branches that
// leave 'range' untouched (the caller's initial value is part of the contract).
void oracle_range(double tmin, double tmax, std::vector<double> dtime, const std::vector<double>& tv,
                  int* time_min_index, int* time_max_index, int* range, int interp) {
    if (dtime.size() == 0) {
        *time_min_index = -1;
        *time_max_index = -1;
        for (size_t i = 0; i < tv.size(); i++) {
            if (tv[i] >= tmin) { *time_min_index = (int)i; break; }
        }
        if (*time_min_index == -1) return;
        for (int i = (int)tv.size() - 1; i >= 0; i--) {
            if (tv[i] != kInvalid && tv[i] <= tmax) { *time_max_index = i; break; }
        }
        if (*time_max_index == -1) return;
        *range = *time_max_index - *time_min_index + 1;
        return;
    }

    if (dtime.size() > 1) { tmin = dtime[0]; tmax = dtime.back(); }
    std::map<std::string, int> times_indices;
    if (tv.size() > 0) {
        *time_max_index = oracle_slices(tmax, tv, times_indices, interp);
        if (interp == alconst::linear_interp) {
            if (times_indices[SLICE_SUP] <= (int)tv.size()) *time_max_index = times_indices[SLICE_SUP];
        }
        *time_min_index = oracle_slices(tmin, tv, times_indices, interp);
    }
    if (dtime.size() == 1) {
        if (dtime[0] <= 0) *range = 0;
        else *range = (int)std::floor((tmax - tmin) / dtime[0] + 1e-9) + 1;
    } else {
        *range = (int)(dtime.size() - 1) + 1;
    }
    if (tv.size() == 0) { *time_min_index = 0; *time_max_index = *range - 1; }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

std::vector<double> ramp(size_t n, double step) {
    std::vector<double> tv(n);
    for (size_t i = 0; i < n; ++i) tv[i] = step * (double)i;
    return tv;
}

const std::vector<int> kInterps = {alconst::undefined_interp, alconst::closest_interp,
                                   alconst::previous_interp, alconst::linear_interp};

// Compares one lookup for every timebase of the fixture set.
void check_slices(const std::vector<std::vector<double>>& bases, const std::vector<double>& probes) {
    DataInterpolation di;
    for (const auto& tv : bases) {
        if (tv.empty()) continue;
        for (int interp : kInterps) {
            for (double t : probes) {
                std::map<std::string, int> ref_map, got_map;
                const int ref = oracle_slices(t, tv, ref_map, interp);
                const int got = di.getSlicesTimesIndices(t, tv, got_map, interp);
                ++g_checks;
                if (got != ref) {
                    fail("getSlicesTimesIndices(t=" + std::to_string(t) + ", n=" + std::to_string(tv.size()) +
                         ", interp=" + std::to_string(interp) + ") -> " + std::to_string(got) +
                         " (expected " + std::to_string(ref) + ")");
                }
                if (got_map[SLICE_INF] != ref_map[SLICE_INF] || got_map[SLICE_SUP] != ref_map[SLICE_SUP]) {
                    fail("slice bracket mismatch (t=" + std::to_string(t) + ", n=" + std::to_string(tv.size()) +
                         ", interp=" + std::to_string(interp) + "): inf " + std::to_string(got_map[SLICE_INF]) +
                         "/" + std::to_string(ref_map[SLICE_INF]) + " sup " + std::to_string(got_map[SLICE_SUP]) +
                         "/" + std::to_string(ref_map[SLICE_SUP]));
                }
            }
        }
    }
}

void check_ranges(const std::vector<std::vector<double>>& bases,
                  const std::vector<std::pair<double, double>>& windows,
                  const std::vector<std::vector<double>>& dtimes, int interp) {
    DataInterpolation di;
    for (const auto& tv : bases) {
        for (const auto& w : windows) {
            for (const auto& dt : dtimes) {
                int ref_min = -1, got_min = -1, ref_max = -1, got_max = -1;
                int ref_range = -777, got_range = -777;   // pins the "left untouched" branches
                oracle_range(w.first, w.second, dt, tv, &ref_min, &ref_max, &ref_range, interp);
                di.getTimeRangeIndices(w.first, w.second, dt, tv, &got_min, &got_max, &got_range, interp);
                ++g_checks;
                if (got_min != ref_min || got_max != ref_max || got_range != ref_range) {
                    fail("getTimeRangeIndices(tmin=" + std::to_string(w.first) + ", tmax=" + std::to_string(w.second) +
                         ", ndt=" + std::to_string(dt.size()) + ", n=" + std::to_string(tv.size()) +
                         ", interp=" + std::to_string(interp) + ") -> min=" + std::to_string(got_min) +
                         " max=" + std::to_string(got_max) + " range=" + std::to_string(got_range) +
                         " (expected min=" + std::to_string(ref_min) + " max=" + std::to_string(ref_max) +
                         " range=" + std::to_string(ref_range) + ")");
                }
            }
        }
    }
}

// Probes around the edges and between points: below the first point, exactly on
// it, just after it, mid-range (hit + in-between), above the last point.
std::vector<double> probes_for(const std::vector<double>& tv) {
    std::vector<double> out;
    const double lo = tv.front(), hi = tv.back();
    const double step = tv.size() > 1 ? tv[1] - tv[0] : 1.0;
    out.push_back(lo - 10.0 * step);
    out.push_back(lo - 0.5 * step);
    out.push_back(lo);
    out.push_back(lo + 0.5 * step);
    for (size_t i = 1; i + 1 < tv.size(); ++i) { out.push_back(tv[i]); out.push_back(tv[i] + 0.5 * step); }
    out.push_back(hi - 0.5 * step);
    out.push_back(hi);
    out.push_back(hi + 0.5 * step);
    out.push_back(hi + 10.0 * step);
    return out;
}

}  // namespace

void test_slices_small_bases() {
    std::cout << "Running test: test_slices_small_bases" << std::endl;
    const std::vector<std::vector<double>> bases = {
        {2.0},
        {0.0, 1.0},
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {0.5, 1.5, 2.0, 2.0001, 3.0},
        {1.0, 2.0, 3.0, 4.0, kInvalid},          // trailing invalid run
    };
    for (const auto& tv : bases) check_slices({tv}, probes_for(tv));
}

void test_slices_big_base() {
    std::cout << "Running test: test_slices_big_base" << std::endl;
    // The size that makes the old linear scan quadratic in a real get().
    const std::vector<double> tv = ramp(8150, 0.002);
    std::vector<double> probes;
    for (size_t i = 0; i < tv.size(); i += 137) {
        probes.push_back(tv[i]);
        probes.push_back(tv[i] + 0.001);
    }
    probes.push_back(tv.front() - 1.0);
    probes.push_back(tv.back() + 1.0);
    check_slices({tv}, probes);
    std::cout << "  (" << probes.size() << " probes x " << kInterps.size() << " interp modes)" << std::endl;
}

void test_ranges_no_resampling() {
    std::cout << "Running test: test_ranges_no_resampling" << std::endl;
    const std::vector<std::vector<double>> bases = {
        ramp(1000, 0.01),
        ramp(1, 0.0),
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {1.0, 2.0, 3.0, 4.0, kInvalid},
    };
    const std::vector<std::pair<double, double>> windows = {
        {0.0, 9.99}, {0.05, 9.9}, {2.0, 2.0}, {2.01, 2.02}, {-5.0, 100.0},
        {-5.0, -1.0}, {50.0, 60.0}, {9.9, 0.0},          // tmin > tmax, and out-of-range windows
    };
    for (int interp : kInterps) check_ranges(bases, windows, {}, interp);
}

void test_ranges_with_resampling() {
    std::cout << "Running test: test_ranges_with_resampling" << std::endl;
    const std::vector<std::vector<double>> bases = {ramp(1000, 0.01), {1.0, 2.0, 3.0, 4.0, 5.0}};
    const std::vector<std::pair<double, double>> windows = {
        {0.0, 9.99}, {1.0, 5.0}, {0.55, 3.9}, {2.0, 2.0},
    };
    const std::vector<std::vector<double>> dtimes = {
        {0.1}, {0.0}, {-1.0}, {0.55, 0.65, 0.75, 2.0, 9.9},
    };
    for (int interp : kInterps) check_ranges(bases, windows, dtimes, interp);
}

void test_resample_timebasis_matches_range() {
    std::cout << "Running test: test_resample_timebasis_matches_range" << std::endl;
    // resample_timebasis(timed_AOS_index == -1) must agree with getTimeRangeIndices
    // on the number of output slices it produces.
    DataInterpolation di;
    const std::vector<double> tv = ramp(1000, 0.01);
    for (double dt : {0.1, 0.25}) {
        for (auto w : std::vector<std::pair<double, double>>{{0.0, 9.99}, {1.0, 5.0}, {0.55, 3.9}}) {
            int rmin = -1, rmax = -1, range = -777;
            di.getTimeRangeIndices(w.first, w.second, std::vector<double>{dt}, tv, &rmin, &rmax, &range,
                                   alconst::closest_interp);
            void* res = nullptr;
            const int nb = di.resample_timebasis(w.first, w.second, std::vector<double>{dt}, -1,
                                                 const_cast<std::vector<double>&>(tv), &res);
            ++g_checks;
            if (nb != range) {
                fail("resample_timebasis gives " + std::to_string(nb) + " slices, getTimeRangeIndices gives " +
                     std::to_string(range) + " (window " + std::to_string(w.first) + ".." + std::to_string(w.second) +
                     ", dt=" + std::to_string(dt) + ")");
            }
            free(res);
        }
    }
}

int main() {
    test_slices_small_bases();
    test_slices_big_base();
    test_ranges_no_resampling();
    test_ranges_with_resampling();
    test_resample_timebasis_matches_range();
    std::cout << "ALL PASSED (" << g_checks << " comparisons)" << std::endl;
    return 0;
}
