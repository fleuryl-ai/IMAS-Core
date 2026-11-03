// @file  test_engine_scalar_block.cpp
// @brief v5 layout contract: a node carrying one value per time step gets a raw
//        block of its own (data_scalar_f64 / _i32 / _c128).
//
// Such a node (global_quantities&ip on an 8000-step core_profiles) is 0-D in the
// IDS but written once per slice, so index v4 gave it one slot per AoS element
// inside the payload block. Reading the whole series back therefore spans the
// entire payload: measured on the 8000x2000 campaign file, 143 990 000 elements
// swept to recover 8 000 (17 998x), 799.8 MiB and 3.11 s of one get() --
// to_improve.md point 31. Index v5 routes those rows to a block of their own,
// with its own append cursor, and records the choice in two bits of the row.
//
//   * a per-step scalar written slice by slice lands in data_scalar_<type>: its
//     rows carry the scalar block, their offsets fit that block's extent, and
//     their span stays equal to their own count (the ratio this layout exists
//     to remove); the payload block keeps the payload only;
//   * all three numeric types are routed (a string per-step signal is not -- its
//     offset addresses data_raw_str);
//   * a bulk write of the same node is one contiguous row of that block;
//   * APPEND continues the block where it stopped (the per-block cursor, not a
//     restart that would clobber the first values);
//   * APPEND keeps the layout of the file it opens: on a v4 file these rows go
//     to the payload block with their block bits at 0, and a later read across
//     both blocks still returns every value.
#include "panzerdb.h"
#include <hdf5.h>

#include <complex>
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

static void stamp_index_version(const std::string& file_name, uint32_t stamp) {
    hid_t lf = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    HARD_CHECK(lf >= 0);
    hid_t idx = H5Dopen2(lf, "index", H5P_DEFAULT);
    HARD_CHECK(idx >= 0);
    if (H5Aexists(idx, "index_version") > 0) H5Adelete(idx, "index_version");
    hsize_t adim[1] = {1};
    hid_t asp = H5Screate_simple(1, adim, nullptr);
    hid_t at = H5Acreate2(idx, "index_version", H5T_STD_U32LE, asp, H5P_DEFAULT, H5P_DEFAULT);
    HARD_CHECK(at >= 0);
    H5Awrite(at, H5T_NATIVE_UINT32, &stamp);
    H5Aclose(at); H5Sclose(asp);
    H5Dclose(idx); H5Fclose(lf);
}

// What one path's rows occupy in the block they address: how many rows carry the
// scalar block, their total count, and the [min offset, max offset+count) span.
struct RowShape {
    uint64_t rows = 0;
    uint64_t scalar_rows = 0;
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
        if (leaf.block == PanzerDB::RAW_BLOCK_SCALAR) ++s.scalar_rows;
        s.count += leaf.count;
        lo = std::min(lo, leaf.offset);
        s.max_end = std::max(s.max_end, leaf.offset + leaf.count);
    }
    if (s.rows) s.span = s.max_end - lo;
    return s;
}

// Both contracts of a per-step-scalar path: rows tagged with the scalar block,
// offsets inside that block's extent, and a span that stays next to the values.
static void check_scalar_signal(const std::vector<PanzerDB::Leaf>& leaves,
                                const std::string& path, hsize_t scalar_extent) {
    const RowShape s = shape_of_path(leaves, path);
    HARD_CHECK(s.rows > 0 && ("per-step scalar " + path + " must be in the index").c_str());
    HARD_CHECK(s.scalar_rows == s.rows &&
               ("every row of " + path + " must address the scalar block").c_str());
    HARD_CHECK(s.max_end <= scalar_extent &&
               ("offsets of " + path + " must fit in the data_scalar_* extent").c_str());
    // One value per step, one after the other: the span IS the signal. The
    // payload layout made this 18 001x the count on the campaign file.
    HARD_CHECK(s.span == s.count &&
               ("the rows of " + path + " must be contiguous in the scalar block").c_str());
}

int main() {
    std::cout << "=== TEST PanzerDB: v5 per-step-scalar block (data_scalar_*) ===\n\n";
    const int N = 100;      // time steps
    const int POINTS = 5;   // radial points of the per-slice signal

    std::vector<double> times(N), ip(N);
    std::vector<int32_t> ip_i(N);
    std::vector<std::complex<double>> ip_c(N);
    for (int i = 0; i < N; ++i) {
        times[i] = 0.1 * i;
        ip[i] = 1.0e6 + i;
        ip_i[i] = 1000 + i;
        ip_c[i] = std::complex<double>(i, -i);
    }

    // ------------------------------------------------------------------
    // 1. Slice by slice (what putSlice() does): one row per step, three types.
    // ------------------------------------------------------------------
    const std::string slice_name = "test_scalar_block.panzer";
    if (fs::exists(slice_name)) fs::remove(slice_name);
    {
        PanzerDB db(slice_name, PanzerDB::OpenMode::WRITE, true);
        for (int i = 0; i < N; ++i) {
            db.writeDataSlices("time", {1}, &times[i], 1, "time");
            db.writeDataSlices("ip", {1}, &ip[i], 1, "time");
            db.writeDataSlices("ip_int", {1}, &ip_i[i], 1, "time");
            db.writeDataSlices("ip_cpx", {1}, &ip_c[i], 1, "time");
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
    const hsize_t scalar_f64 = dset_extent(f, PANZER_SCALAR_DATASET_F64);
    const hsize_t scalar_i32 = dset_extent(f, PANZER_SCALAR_DATASET_I32);
    const hsize_t scalar_c128 = dset_extent(f, PANZER_SCALAR_DATASET_C128);
    const hsize_t payload_extent = dset_extent(f, "data_raw_f64");
    H5Fclose(f);
    HARD_CHECK(time_extent == (hsize_t)(2 * N) &&
               "both time bases still own data_time_f64, nothing else");
    HARD_CHECK(scalar_f64 == (hsize_t)N && "ip must own data_scalar_f64, nothing else");
    HARD_CHECK(scalar_i32 == (hsize_t)N && "ip_int must own data_scalar_i32");
    HARD_CHECK(scalar_c128 == (hsize_t)N && "ip_cpx must own data_scalar_c128");
    HARD_CHECK(payload_extent == (hsize_t)(N * POINTS) &&
               "data_raw_f64 must hold the payload only, no scalar slots");

    {
        PanzerDB db(slice_name, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();
        check_scalar_signal(leaves, "ip", scalar_f64);
        check_scalar_signal(leaves, "ip_int", scalar_i32);
        check_scalar_signal(leaves, "ip_cpx", scalar_c128);

        // A per-slice array is payload: block 0, inside the payload extent.
        const RowShape dens = shape_of_path(leaves, "profiles_1d/dens");
        HARD_CHECK(dens.scalar_rows == 0 && "payload rows must not claim the scalar block");
        HARD_CHECK(dens.max_end <= payload_extent &&
                   "payload offsets must fit in the data_raw_f64 extent");
        // A time base is not a per-step scalar: it stays in its own block.
        HARD_CHECK(shape_of_path(leaves, "time").scalar_rows == 0 &&
                   "the time base must not move into the scalar block");

        // The read path this layout exists for: the whole series, once.
        const std::vector<double> whole = db.getWholeDynamicSignal("ip");
        HARD_CHECK(whole.size() == times.size() && "ip read whole");
        for (int i = 0; i < N; ++i) HARD_CHECK(whole[i] == ip[i] && "ip value");

        // Step-at-a-time reads, one per type.
        uint64_t ndim = 0, shape[6] = {0};
        double* out_d = nullptr;
        HARD_CHECK(db.readDataByIndex("ip", N - 1, &ndim, shape, &out_d) == 0 &&
                   "last step of ip reads back");
        HARD_CHECK(ndim == 1 && shape[0] == 1 && out_d[0] == ip[N - 1]);
        delete[] out_d;
        int32_t* out_i = nullptr;
        HARD_CHECK(db.readIntDataByIndex("ip_int", N - 1, &ndim, shape, &out_i) == 0 &&
                   "last step of ip_int reads back");
        HARD_CHECK(out_i[0] == ip_i[N - 1]);
        delete[] out_i;
        std::complex<double>* out_c = nullptr;
        HARD_CHECK(db.readComplexDataByIndex("ip_cpx", N - 1, &ndim, shape, &out_c) == 0 &&
                   "last step of ip_cpx reads back");
        HARD_CHECK(out_c[0] == ip_c[N - 1]);
        delete[] out_c;

        db.close();
        std::cout << "  [OK] " << N << " per-step scalars of 3 types in "
                  << PANZER_SCALAR_DATASET_F64 << "/" << PANZER_SCALAR_DATASET_I32 << "/"
                  << PANZER_SCALAR_DATASET_C128 << " (" << scalar_f64 << " slots each), "
                  << "payload alone in data_raw_f64 (" << payload_extent << " slots)\n";
    }

    // ------------------------------------------------------------------
    // 2. Bulk (what put() does): the whole series in ONE row of that block.
    // ------------------------------------------------------------------
    const std::string bulk_name = "test_scalar_block_bulk.panzer";
    if (fs::exists(bulk_name)) fs::remove(bulk_name);
    {
        PanzerDB db(bulk_name, PanzerDB::OpenMode::WRITE, true);
        db.writeDataSlices("time", {N}, times.data(), 1, "time");
        db.writeDataSlices("ip", {1}, ip.data(), N, "time");
        db.flush();
        db.close();
    }
    f = H5Fopen(bulk_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hsize_t bulk_scalar = dset_extent(f, PANZER_SCALAR_DATASET_F64);
    const hsize_t bulk_payload = dset_extent(f, "data_raw_f64");
    H5Fclose(f);
    HARD_CHECK(bulk_scalar == (hsize_t)N && "bulk ip in data_scalar_f64");
    HARD_CHECK(bulk_payload == 0 && "a bulk per-step scalar must not touch the payload block");
    {
        PanzerDB db(bulk_name, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();
        const RowShape s = shape_of_path(leaves, "ip");
        HARD_CHECK(s.rows == 1 && s.count == (uint64_t)N && "bulk ip: one row, N values");
        check_scalar_signal(leaves, "ip", bulk_scalar);
        const std::vector<double> back = db.getWholeDynamicSignal("ip");
        HARD_CHECK(back.size() == ip.size() && "bulk ip read whole");
        for (int i = 0; i < N; ++i) HARD_CHECK(back[i] == ip[i] && "bulk ip value");
        db.close();
        std::cout << "  [OK] bulk ip: one row of " << N << " slots, payload block untouched\n";
    }

    // ------------------------------------------------------------------
    // 3. APPEND: the cursor continues the block instead of restarting it (a
    //    stale cursor would overwrite the first N values).
    // ------------------------------------------------------------------
    {
        PanzerDB db(bulk_name, PanzerDB::OpenMode::APPEND);
        std::vector<double> more_ip(N);
        for (int i = 0; i < N; ++i) more_ip[i] = 1.0e6 + N + i;
        db.writeDataSlices("ip", {1}, more_ip.data(), N, "time");
        db.flush();
        db.close();
    }
    f = H5Fopen(bulk_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hsize_t appended_scalar = dset_extent(f, PANZER_SCALAR_DATASET_F64);
    H5Fclose(f);
    HARD_CHECK(appended_scalar == (hsize_t)(2 * N) && "both lives in data_scalar_f64");
    {
        PanzerDB db(bulk_name, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();
        const RowShape s = shape_of_path(leaves, "ip");
        HARD_CHECK(s.count == (uint64_t)(2 * N) && "both lives of ip");
        HARD_CHECK(s.span == s.count &&
                   "the appended rows continue where the first ones stopped");
        const std::vector<double> back = db.getWholeDynamicSignal("ip");
        HARD_CHECK(back.size() == (size_t)(2 * N) && "both lives read back");
        for (int i = 0; i < 2 * N; ++i)
            HARD_CHECK(back[i] == 1.0e6 + i && "appended values did not clobber the first ones");
        db.close();
        std::cout << "  [OK] APPEND: " << appended_scalar << " slots in "
                  << PANZER_SCALAR_DATASET_F64 << ", no overwrite\n";
    }

    // ------------------------------------------------------------------
    // 4. APPEND keeps the layout of the file it opens. On a v4 file there is no
    //    scalar block to route into, so these rows stay in the payload block
    //    with their block bits at 0 -- and a read that covers rows from BOTH
    //    blocks (the v5 rows already on disk, the v4 rows just appended) still
    //    returns every value in order.
    // ------------------------------------------------------------------
    stamp_index_version(bulk_name, PANZER_INDEX_VERSION_V4);
    {
        PanzerDB db(bulk_name, PanzerDB::OpenMode::APPEND);
        std::vector<double> older_ip(N);
        for (int i = 0; i < N; ++i) older_ip[i] = 1.0e6 + 2 * N + i;
        db.writeDataSlices("ip", {1}, older_ip.data(), N, "time");
        db.flush();
        db.close();
    }
    f = H5Fopen(bulk_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hsize_t v4_scalar = dset_extent(f, PANZER_SCALAR_DATASET_F64);
    const hsize_t v4_payload = dset_extent(f, "data_raw_f64");
    H5Fclose(f);
    HARD_CHECK(v4_scalar == appended_scalar && "a v4 APPEND must not extend the scalar block");
    HARD_CHECK(v4_payload == (hsize_t)N && "the v4 rows went to the payload block");
    {
        PanzerDB db(bulk_name, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();
        const RowShape s = shape_of_path(leaves, "ip");
        HARD_CHECK(s.rows == 3 && s.scalar_rows == 2 &&
                   "the v4-era row keeps the payload block, the v5 rows keep the scalar one");
        const std::vector<double> back = db.getWholeDynamicSignal("ip");
        HARD_CHECK(back.size() == (size_t)(3 * N) && "all three lives read back");
        for (int i = 0; i < 3 * N; ++i)
            HARD_CHECK(back[i] == 1.0e6 + i && "value across both blocks");
        db.close();
        std::cout << "  [OK] APPEND on a v4 file: payload block for the new rows, one read "
                     "across both blocks\n";
    }

    fs::remove(slice_name);
    fs::remove(bulk_name);
    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}
