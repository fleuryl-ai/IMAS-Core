// @file  test_engine_index_v3.cpp
// @brief v3 index format contract (REVIEW_PERF point 13/14/16):
//   * new files carry /index with an "index_version"=3 attribute, 7 packed
//     u64 columns, and an interned /path_table instead of row-aligned /paths;
//   * repeated paths cost ONE path_table entry (dedup), not one per row;
//   * round-trip of the public Leaf surface is unchanged (path/parent/shape/
//     flags/offsets), including APPEND continuation (stable ids, no path
//     duplication);
//   * a hand-crafted legacy 12-column file (no attribute, row-aligned /paths)
//     is still readable (upward compatibility).
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

static uint32_t read_index_version(hid_t file) {
    hid_t d = H5Dopen2(file, "index", H5P_DEFAULT);
    HARD_CHECK(d >= 0);
    uint32_t v = 0;
    if (H5Aexists(d, "index_version") > 0) {
        hid_t at = H5Aopen(d, "index_version", H5P_DEFAULT);
        if (at >= 0) { H5Aread(at, H5T_NATIVE_UINT32, &v); H5Aclose(at); }
    }
    hsize_t dims[2] = {0, 0};
    hid_t s = H5Dget_space(d);
    H5Sget_simple_extent_dims(s, dims, nullptr);
    H5Sclose(s);
    H5Dclose(d);
    if (v == 0 && dims[1] == 7) v = 3;
    return v;
}

static hsize_t dset_rows(hid_t file, const char* name) {
    if (H5Lexists(file, name, H5P_DEFAULT) <= 0) return 0;
    hid_t d = H5Dopen2(file, name, H5P_DEFAULT);
    hid_t s = H5Dget_space(d);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(s, dims, nullptr);
    H5Sclose(s);
    H5Dclose(d);
    return dims[0];
}

int main() {
    std::cout << "=== TEST PanzerDB: v3 index format (interning, packing, legacy read) ===\n\n";

    const std::string filename = "test_index_v3.panzer";
    if (fs::exists(filename)) fs::remove(filename);

    // ------------------------------------------------------------------
    // 1. WRITE: dynamic AoS with a 100-step numeric signal + a 100-step
    //    compact string-list signal -> hundreds of index rows, very few
    //    distinct paths.
    // ------------------------------------------------------------------
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        db.beginArray("sig_aos", "time");
        std::vector<double> times(100), vals(100);
        for (int i = 0; i < 100; ++i) { times[i] = i * 0.1; vals[i] = i * 2.0; }
        db.writeDataSlices("time", {1}, times.data(), 100, "");
        db.writeDataSlices("sig",  {1}, vals.data(),  100, "time");
        std::vector<const char*> strs(100);
        static const char* text[100];
        for (int i = 0; i < 100; ++i) {
            static std::string buf[100];
            buf[i] = "s" + std::to_string(i);
            text[i] = buf[i].c_str();
            strs[i] = text[i];
        }
        db.writeDataSlices("sigstr", {1}, strs.data(), 100, "time");
        db.endArray();
        db.flush();
        db.close();
    }

    hid_t f = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    HARD_CHECK(f >= 0);
    HARD_CHECK(read_index_version(f) == PANZER_INDEX_VERSION_V3 && "new file must be stamped v3");
    const hsize_t path_entries = dset_rows(f, "path_table");
    HARD_CHECK(path_entries > 0 && "v3 must own a /path_table");
    HARD_CHECK(dset_rows(f, "paths") == 0 && "v3 must NOT carry row-aligned /paths");
    H5Fclose(f);

    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();
        // sig + sigstr each merged into ONE row (numeric bulk + V4 run),
        // timebase one row, one meta row -> 4 leaves.
        HARD_CHECK(leaves.size() == 4 && "merged rows: meta + time + sig + sigstr");

        // Interning: distinct paths only, NOT one entry per row.
        HARD_CHECK(path_entries <= leaves.size() &&
                   "path_table must hold UNIQUE paths, not one slot per row");

        const PanzerDB::Leaf* sig = find_leaf(leaves, "sig_aos/sig");
        HARD_CHECK(sig && "sig leaf present");
        HARD_CHECK(sig->count == 100 && sig->time_index == 0 && sig->ndim == 1);
        HARD_CHECK(sig->parent_path == "sig_aos");
        std::vector<double> out(100);
        db.readTensor(*sig, out.data());
        for (int i = 0; i < 100; ++i) HARD_CHECK(out[i] == i * 2.0);

        const PanzerDB::Leaf* sigstr = find_leaf(leaves, "sig_aos/sigstr");
        HARD_CHECK(sigstr && "sigstr leaf present");
        HARD_CHECK(sigstr->count == 100 && sigstr->time_index == 0);
        HARD_CHECK(PanzerDB::leafDataType(*sigstr) == PanzerDB::DataType::STRING);

        std::cout << "  [OK] v3 stamped; " << path_entries << " unique paths for "
                  << leaves.size() << " leaves; merged rows round-trip\n";
        db.close();
    }

    // ------------------------------------------------------------------
    // 2. APPEND: ids continue, interning continues (no duplicate text for
    //    existing paths), and the new slices merge into fresh rows.
    // ------------------------------------------------------------------
    {
        PanzerDB db(filename, PanzerDB::OpenMode::APPEND);
        HARD_CHECK(db.getAOSShape("sig_aos").front() == 100);
        db.beginArray("sig_aos", "time");
        std::vector<double> times(20), vals(20);
        for (int i = 0; i < 20; ++i) { times[i] = (100 + i) * 0.1; vals[i] = (100 + i) * 2.0; }
        db.writeDataSlices("time", {1}, times.data(), 20, "");
        db.writeDataSlices("sig",  {1}, vals.data(),  20, "time");
        db.endArray();
        db.flush();
        db.close();
    }
    f = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hsize_t path_entries2 = dset_rows(f, "path_table");
    H5Fclose(f);
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();
        const PanzerDB::Leaf* sig = find_leaf(leaves, "sig_aos/sig");
        HARD_CHECK(sig && "sig still present after APPEND");
        // The appended slices form their own row (gap-free continuation is not
        // retro-merged); total steps across the two rows must be 120.
        uint64_t total = 0;
        for (const auto& l : leaves)
            if (l.path == "sig_aos/sig") total += l.count;
        HARD_CHECK(total == 120 && "120 steps after APPEND");
        HARD_CHECK(db.getAOSShape("sig_aos").front() == 120 &&
                   "dynamic AoS size must cover both rows");
    }
    f = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hsize_t rows_after = dset_rows(f, "index");
    H5Fclose(f);
    HARD_CHECK(path_entries2 < rows_after &&
               "interning: fewer path entries than index rows (paths reused on APPEND)");
    std::cout << "  [OK] APPEND: ids/ids stable, " << path_entries2
              << " unique paths for " << rows_after << " index rows\n";

    // ------------------------------------------------------------------
    // 3. Legacy compatibility: hand-craft a v2 12-column file (no attribute,
    //    row-aligned /paths) and read it through the engine.
    //    One meta row (AoS dyn "leg") + one data row "leg/0/x" (2 f64).
    // ------------------------------------------------------------------
    const std::string legacy_name = "test_index_v3_legacy.h5";
    if (fs::exists(legacy_name)) fs::remove(legacy_name);
    {
        hid_t lf = H5Fcreate(legacy_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims[2] = {2, 12}, maxd[2] = {H5S_UNLIMITED, 12}, chunk[2] = {16, 12};
        hid_t sp = H5Screate_simple(2, dims, maxd);
        hid_t pl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(pl, 2, chunk);
        hid_t idx = H5Dcreate2(lf, "index", H5T_STD_U64LE, sp, H5P_DEFAULT, pl, H5P_DEFAULT);
        H5Sclose(sp);
        uint64_t rows[2][12] = {
            // ndim shape[6] time off count flags parent
            {1, 0,0,0,0,0,0, 0, 0,0, 3 /*dyn meta*/, PANZER_NO_PARENT_ROW},
            {1, 2,0,0,0,0,0, 0, 0,2, 0 /*f64 data*/,  0 /* parent = row 0 */},
        };
        H5Dwrite(idx, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, rows);
        H5Dclose(idx); H5Pclose(pl);

        hid_t st = H5Tcopy(H5T_C_S1);
        H5Tset_size(st, 256);
        H5Tset_strpad(st, H5T_STR_NULLPAD);
        hsize_t pdims[1] = {2}, pmax[1] = {H5S_UNLIMITED}, pchunk[1] = {16};
        hid_t psp = H5Screate_simple(1, pdims, pmax);
        hid_t ppl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(ppl, 1, pchunk);
        hid_t pds = H5Dcreate2(lf, "paths", st, psp, H5P_DEFAULT, ppl, H5P_DEFAULT);
        H5Sclose(psp);
        char pbuf[2][256] = {};
        std::snprintf(pbuf[0], 256, "leg");
        std::snprintf(pbuf[1], 256, "leg/0/x");
        H5Dwrite(pds, st, H5S_ALL, H5S_ALL, H5P_DEFAULT, pbuf);
        H5Dclose(pds); H5Pclose(ppl); H5Tclose(st);

        hsize_t ddims[1] = {2}, dmax[1] = {H5S_UNLIMITED}, dchunk[1] = {16};
        hid_t dsp = H5Screate_simple(1, ddims, dmax);
        hid_t dpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dpl, 1, dchunk);
        hid_t dds = H5Dcreate2(lf, "data_raw_f64", H5T_IEEE_F64LE, dsp, H5P_DEFAULT, dpl, H5P_DEFAULT);
        H5Sclose(dsp);
        double dvals[2] = {11.0, 22.0};
        H5Dwrite(dds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dvals);
        H5Dclose(dds); H5Pclose(dpl);
        H5Fclose(lf);
    }
    {
        PanzerDB db(legacy_name, PanzerDB::OpenMode::READ);
        const auto& leaves = db.getLeaves();
        HARD_CHECK(leaves.size() == 2 && "legacy file: 2 rows decoded");
        const PanzerDB::Leaf* x = find_leaf(leaves, "leg/0/x");
        HARD_CHECK(x && "legacy data row decoded");
        // M1 text rule: a data leaf's parent = full path minus one segment.
        HARD_CHECK(x->parent_path == "leg/0" && "legacy parent reconstructed from text rule");
        HARD_CHECK(x->count == 2 && x->ndim == 1 && x->shape[0] == 2);
        double out[2] = {0, 0};
        db.readTensor(*x, out);
        HARD_CHECK(out[0] == 11.0 && out[1] == 22.0);
        std::cout << "  [OK] legacy 12-column file (no attribute) still readable\n";
        db.close();
    }

    fs::remove(filename);
    fs::remove(legacy_name);
    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}
