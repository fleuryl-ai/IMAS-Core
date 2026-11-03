// @file  test_engine_time_base_layout.cpp
// @brief v4 layout contract: an IDS time base gets a raw block of its own.
//
// A time base written slice by slice used to occupy one or two FLOAT64 slots per
// payload block of data_raw_f64, so reading it back spanned the whole payload
// (measured on an 8000-slice core_profiles: 8000 doubles recovered from a
// 1 152 047 984-byte range, 2198 chunks decompressed -- to_improve.md 24/30).
// Index v4 routes both time bases to data_time_f64, which has its own append
// cursor, and records the choice in one bit of the packed row.
//
//   * after 100 slice writes, the rows of the root base ("time") and the
//     per-element base ("profiles_1d/time") carry the block bit, their offsets
//     fit inside the data_time_f64 extent, and their span stays within twice
//     their own count (the ratio that caused all of this);
//   * the payload block holds the payload only;
//   * both bases read back whole, slice-by-slice and bulk (put) writes alike,
//     and stay contiguous when a bulk APPEND follows slice-by-slice writes;
//   * the routing reads the timebase as a path, not as the string the Access
//     Layer happens to pass ("/time" from an AoS, "&time" from a standalone
//     write), which is what decides these rows in a file the AL itself wrote;
//   * a file stamped with an index_version this build cannot decode is refused
//     by name, not decoded as v2.
#include "panzerdb.h"
#include <hdf5.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

#include "fixtures/panzerdb_helper.h"

#define HARD_CHECK(cond) \
    do { if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        std::exit(1); \
    } } while (0)

static hsize_t dset_extent(hid_t file, const char* name) {
    if (H5Lexists(file, name, H5P_DEFAULT) <= 0) return 0;
    hid_t d = H5Dopen2(file, name, H5P_DEFAULT);
    HARD_CHECK(d >= 0);
    hid_t s = H5Dget_space(d);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(s, dims, nullptr);
    H5Sclose(s);
    H5Dclose(d);
    return dims[0];
}

static uint32_t read_index_version(hid_t file) {
    hid_t d = H5Dopen2(file, "index", H5P_DEFAULT);
    HARD_CHECK(d >= 0);
    uint32_t v = 0;
    if (H5Aexists(d, "index_version") > 0) {
        hid_t at = H5Aopen(d, "index_version", H5P_DEFAULT);
        if (at >= 0) { H5Aread(at, H5T_NATIVE_UINT32, &v); H5Aclose(at); }
    }
    H5Dclose(d);
    return v;
}

// What one path's rows occupy in the block they address: how many rows carry the
// v4 time-base bit, their total count, and the [min offset, max offset+count) span.
struct RowShape {
    uint64_t rows = 0;
    uint64_t block_rows = 0;
    uint64_t count = 0;
    uint64_t span = 0;
    uint64_t max_end = 0;
};

static RowShape shape_of_path(const std::vector<PanzerDB::Leaf>& leaves,
                              const std::string& path) {
    RowShape s;
    uint64_t lo = ~0ULL;
    for (const auto& leaf : leaves) {
        if (leaf.path != path) continue;
        ++s.rows;
        if (leaf.block == PanzerDB::RAW_BLOCK_TIME) ++s.block_rows;
        s.count += leaf.count;
        lo = std::min(lo, leaf.offset);
        s.max_end = std::max(s.max_end, leaf.offset + leaf.count);
    }
    if (s.rows) s.span = s.max_end - lo;
    return s;
}

// Both contracts of a time-base path: rows tagged with the block, offsets inside
// that block's extent, and a span that stays next to the data it recovers.
static void check_time_base(const std::vector<PanzerDB::Leaf>& leaves,
                            const std::string& path, hsize_t time_extent) {
    const RowShape s = shape_of_path(leaves, path);
    HARD_CHECK(s.rows > 0 && ("time base " + path + " must be in the index").c_str());
    HARD_CHECK(s.block_rows == s.rows &&
               ("every row of " + path + " must address the time-base block").c_str());
    HARD_CHECK(s.max_end <= time_extent &&
               ("offsets of " + path + " must fit in the data_time_f64 extent").c_str());
    HARD_CHECK(s.span <= 2 * s.count &&
               ("the rows of " + path + " must stay contiguous in their block").c_str());
}

static std::vector<double> make_times(int n) {
    std::vector<double> t(n);
    for (int i = 0; i < n; ++i) t[i] = 0.1 * i;
    return t;
}

int main() {
    std::cout << "=== TEST PanzerDB: v4 time-base block (data_time_f64) ===\n\n";
    const int N = 100;      // slices
    const int POINTS = 5;   // radial points of the per-slice signal

    // ------------------------------------------------------------------
    // 1. Slice by slice (what putSlice() does): both bases, one row per step.
    // ------------------------------------------------------------------
    const std::string slice_name = "test_time_base_layout.panzer";
    if (fs::exists(slice_name)) fs::remove(slice_name);
    const std::vector<double> times = make_times(N);
    {
        PanzerDB db(slice_name, PanzerDB::OpenMode::WRITE, true);
        for (int i = 0; i < N; ++i) {
            db.writeDataSlices("time", {1}, &times[i], 1, "time");
        }
        db.beginArray("profiles_1d", "time");
        for (int i = 0; i < N; ++i) {
            db.writeDataSlices("time", {1}, &times[i], 1, "time");
            std::vector<double> dens(POINTS);
            for (int p = 0; p < POINTS; ++p) dens[p] = 100.0 * i + p;
            db.writeDataSlices("dens", {POINTS}, dens.data(), 1, "time");
        }
        db.endArray();
        db.flush();
        db.close();
    }

    hid_t f = H5Fopen(slice_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    HARD_CHECK(f >= 0);
    HARD_CHECK(read_index_version(f) == PANZER_INDEX_VERSION_V5 && "new file must be stamped v5");
    const hsize_t time_extent = dset_extent(f, PANZER_TIME_DATASET);
    const hsize_t payload_extent = dset_extent(f, "data_raw_f64");
    H5Fclose(f);
    HARD_CHECK(time_extent == (hsize_t)(2 * N) &&
               "both bases must land in data_time_f64, nothing else");
    HARD_CHECK(payload_extent == (hsize_t)(N * POINTS) &&
               "data_raw_f64 must hold the payload only");

    {
        PanzerDB db(slice_name, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();
        check_time_base(leaves, "time", time_extent);
        check_time_base(leaves, "profiles_1d/time", time_extent);

        // A payload row keeps the bit at 0 and addresses the payload block.
        const RowShape dens = shape_of_path(leaves, "profiles_1d/dens");
        HARD_CHECK(dens.block_rows == 0 && "payload rows must not claim the time block");
        HARD_CHECK(dens.max_end <= payload_extent &&
                   "payload offsets must fit in the data_raw_f64 extent");

        // The read path this layout exists for: the whole base, once.
        const std::vector<double> root = db.getWholeDynamicSignal("time");
        HARD_CHECK(root.size() == times.size() && "root time base read whole");
        for (int i = 0; i < N; ++i) HARD_CHECK(root[i] == times[i] && "root time value");
        const std::vector<double> per_elem = db.getWholeDynamicSignal("profiles_1d/time");
        HARD_CHECK(per_elem.size() == times.size() && "per-element time base read whole");
        for (int i = 0; i < N; ++i) HARD_CHECK(per_elem[i] == times[i] && "per-element value");

        // The payload still reads slice by slice.
        uint64_t ndim = 0, shape[6] = {0};
        double* out = nullptr;
        HARD_CHECK(db.readDataByIndex("profiles_1d/dens", N - 1, &ndim, shape, &out) == 0 &&
                   "last slice of the payload reads back");
        HARD_CHECK(ndim == 1 && shape[0] == POINTS && out[0] == 100.0 * (N - 1));
        delete[] out;
        db.close();
        std::cout << "  [OK] 2 x " << N << " time-base rows contiguous in "
                  << PANZER_TIME_DATASET << " (" << time_extent
                  << " slots), payload alone in data_raw_f64 (" << payload_extent
                  << " slots)\n";
    }

    // ------------------------------------------------------------------
    // 2. Bulk (what put() does): the whole base in ONE row of the time block.
    // ------------------------------------------------------------------
    const std::string bulk_name = "test_time_base_layout_bulk.panzer";
    if (fs::exists(bulk_name)) fs::remove(bulk_name);
    {
        PanzerDB db(bulk_name, PanzerDB::OpenMode::WRITE, true);
        db.writeDataSlices("time", {N}, times.data(), 1, "time");
        db.beginArray("profiles_1d", "time");
        db.writeDataSlices("time", {N}, times.data(), 1, "time");
        std::vector<double> dens(N * POINTS);
        for (int i = 0; i < N; ++i)
            for (int p = 0; p < POINTS; ++p) dens[i * POINTS + p] = 100.0 * i + p;
        db.writeDataSlices("dens", {POINTS}, dens.data(), N, "time");
        db.endArray();
        db.flush();
        db.close();
    }
    f = H5Fopen(bulk_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hsize_t bulk_time_extent = dset_extent(f, PANZER_TIME_DATASET);
    H5Fclose(f);
    {
        PanzerDB db(bulk_name, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();
        const RowShape root = shape_of_path(leaves, "time");
        HARD_CHECK(root.rows == 1 && root.count == (uint64_t)N && "bulk root base: one row");
        check_time_base(leaves, "time", bulk_time_extent);
        check_time_base(leaves, "profiles_1d/time", bulk_time_extent);
        const std::vector<double> back = db.getWholeDynamicSignal("time");
        HARD_CHECK(back.size() == times.size() && "bulk root base read whole");
        for (int i = 0; i < N; ++i) HARD_CHECK(back[i] == times[i] && "bulk root value");
        const std::vector<double> elem = db.getWholeDynamicSignal("profiles_1d/time");
        HARD_CHECK(elem.size() == times.size() && "bulk per-element base read whole");
        for (int i = 0; i < N; ++i) HARD_CHECK(elem[i] == times[i] && "bulk per-element value");
        db.close();
        std::cout << "  [OK] bulk time bases: one row each, same block, values intact\n";
    }

    // ------------------------------------------------------------------
    // 3. Mixed lives: a bulk APPEND after slice-by-slice writes must continue
    //    the block where it stopped, not restart it (that is what the per-dtype
    //    cursor buys; a stale cursor would overwrite earlier values).
    // ------------------------------------------------------------------
    {
        PanzerDB db(slice_name, PanzerDB::OpenMode::APPEND);
        HARD_CHECK(db.getAOSShape("profiles_1d").front() == (uint64_t)N);
        db.beginArray("profiles_1d", "time");
        std::vector<double> more_times = make_times(N);
        for (int i = 0; i < N; ++i) more_times[i] = 10.0 + 0.1 * i;
        db.writeDataSlices("time", {N}, more_times.data(), 1, "time");
        for (int i = 0; i < N; ++i) {
            std::vector<double> dens(POINTS);
            for (int p = 0; p < POINTS; ++p) dens[p] = 100.0 * (N + i) + p;
            db.writeDataSlices("dens", {POINTS}, dens.data(), 1, "time");
        }
        db.endArray();
        db.flush();
        db.close();
    }
    f = H5Fopen(slice_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hsize_t appended_time_extent = dset_extent(f, PANZER_TIME_DATASET);
    H5Fclose(f);
    {
        PanzerDB db(slice_name, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();
        check_time_base(leaves, "time", appended_time_extent);
        check_time_base(leaves, "profiles_1d/time", appended_time_extent);
        const RowShape elem = shape_of_path(leaves, "profiles_1d/time");
        HARD_CHECK(elem.count == (uint64_t)(2 * N) && "both lives of the per-element base");
        HARD_CHECK(elem.span <= 2 * elem.count &&
                   "the appended rows continue where the first ones stopped");
        const std::vector<double> back = db.getWholeDynamicSignal("profiles_1d/time");
        HARD_CHECK(back.size() == (size_t)(2 * N) && "both lives read back");
        for (int i = 0; i < N; ++i) HARD_CHECK(back[N + i] == 10.0 + 0.1 * i &&
                                              "appended values did not clobber the first ones");
        db.close();
        std::cout << "  [OK] APPEND: " << appended_time_extent
                  << " slots in " << PANZER_TIME_DATASET << ", no overwrite\n";
    }

    // ------------------------------------------------------------------
    // 4. The spellings the Access Layer actually hands over. beginArray() gets the
    //    AoS timebase as "/time", and a standalone write arrives with "&time" (the
    //    writer converts separators for dataset names). Comparing against the string
    //    as it arrives -- which is how this predicate was first written -- matches
    //    neither, routes every time-base row to the payload block, and leaves the v3
    //    scatter in place while every other test here still passes.
    // ------------------------------------------------------------------
    const std::string al_name = "test_time_base_layout_al.panzer";
    if (fs::exists(al_name)) fs::remove(al_name);
    {
        PanzerDB db(al_name, PanzerDB::OpenMode::WRITE, true);
        db.beginArray("profiles_1d", "/time");
        for (int i = 0; i < N; ++i) db.writeDataSlices("time", {1}, &times[i], 1, "/time");
        db.endArray();
        for (int i = 0; i < N; ++i) db.writeDataSlices("time", {1}, &times[i], 1, "&time");
        db.flush();
        db.close();
    }
    f = H5Fopen(al_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hsize_t al_extent = dset_extent(f, PANZER_TIME_DATASET);
    H5Fclose(f);
    HARD_CHECK(al_extent == (hsize_t)(2 * N) && "both spellings reach data_time_f64");
    {
        PanzerDB db(al_name, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();
        check_time_base(leaves, "profiles_1d/time", al_extent);
        check_time_base(leaves, "time", al_extent);
        db.close();
        std::cout << "  [OK] the AL spellings (/time, &time) reach the time block\n";
    }

    // ------------------------------------------------------------------
    // 5. A file this build cannot decode must be refused by name. Without the
    //    refusal, a stamped table falls back to the v2 column map and silently
    //    reads garbage out of a 7-column table.
    // ------------------------------------------------------------------
    const std::string future_name = "test_time_base_layout_future.panzer";
    if (fs::exists(future_name)) fs::remove(future_name);
    {
        PanzerDB db(future_name, PanzerDB::OpenMode::WRITE, true);
        const double v = 1.0;
        db.writeData("s_f64", {}, &v, 1);
        db.flush();
        db.close();
        hid_t lf = H5Fopen(future_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        HARD_CHECK(lf >= 0);
        hid_t idx = H5Dopen2(lf, "index", H5P_DEFAULT);
        HARD_CHECK(idx >= 0);
        const uint32_t stamp = 99;
        if (H5Aexists(idx, "index_version") > 0) H5Adelete(idx, "index_version");
        hsize_t adim[1] = {1};
        hid_t asp = H5Screate_simple(1, adim, nullptr);
        hid_t at = H5Acreate2(idx, "index_version", H5T_STD_U32LE, asp, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(at, H5T_NATIVE_UINT32, &stamp);
        H5Aclose(at); H5Sclose(asp);
        H5Dclose(idx); H5Fclose(lf);
    }
    for (const PanzerDB::OpenMode mode : {PanzerDB::OpenMode::READ, PanzerDB::OpenMode::APPEND}) {
        bool threw = false;
        try {
            PanzerDB db(future_name, mode);
            db.getLeaves();
        } catch (const ALBackendException& e) {
            threw = true;
            const std::string msg = e.what();
            HARD_CHECK(msg.find("index_version=99") != std::string::npos &&
                       "the refusal must name the version it cannot read");
        }
        HARD_CHECK(threw && "a stamped-ahead index must be refused, not decoded");
    }
    std::cout << "  [OK] index_version=99 refused by name in READ and APPEND\n";
    fs::remove(future_name);

    fs::remove(slice_name);
    fs::remove(bulk_name);
    fs::remove(al_name);
    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}
