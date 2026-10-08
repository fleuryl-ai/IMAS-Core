// @file  test_engine_commit_order_guardrail.cpp
// @brief Deterministic regression for the crash-safe commit ordering of
//        PanzerDB::flush() plus the matching reader guardrail in getLeaves().
//
//   * writes one valid double scalar and verifies it round-trips (baseline);
//   * hand-crafts a DANGLING index row (kind 0 / FLOAT64) whose offset is far
//     past the materialised data_raw_f64 extent — the exact torn state the old
//     index-first ordering could leave behind after a mid-flush crash;
//   * verifies the reader SKIPS the dangling row (it never appears in the
//     leaves) and that the valid leaf still round-trips.
//
// The injection is layout-aware: v3 files (7 packed columns + interned
// /path_table) and v2/legacy files (12/14 columns + row-aligned /paths).
#include "panzerdb.h"
#include <hdf5.h>

#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

#include "fixtures/panzerdb_helper.h"

// Hard check: NDEBUG strips plain assert() from Release builds, and this test
// is worthless if the assertions silently vanish.
#define HARD_CHECK(cond) \
    do { if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        std::exit(1); \
    } } while (0)

// Append one fabricated DANGLING row to /index AND to the file's path storage.
// Its offset is set far past the data_raw_f64 extent so the reader's guardrail
// must skip it. The path datasets are extended consistently with the format
// version in effect (v3: one extra path_table entry; v2: one /paths row).
static void inject_dangling_row(const std::string& path) {
    hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    HARD_CHECK(f >= 0 && "inject: open RW");

    hid_t index_dset = H5Dopen2(f, "index", H5P_DEFAULT);
    HARD_CHECK(index_dset >= 0 && "inject: /index present");

    // Format version: attribute when present, else inferred from the columns.
    uint32_t version = 2;
    if (H5Aexists(index_dset, "index_version") > 0) {
        hid_t at = H5Aopen(index_dset, "index_version", H5P_DEFAULT);
        if (at >= 0) { H5Aread(at, H5T_NATIVE_UINT32, &version); H5Aclose(at); }
    }

    hsize_t irows = 0, ncols = 0;
    {
        hid_t iids = H5Dget_space(index_dset);
        hsize_t idims[2] = {0, 0};
        H5Sget_simple_extent_dims(iids, idims, NULL);
        H5Sclose(iids);
        irows = idims[0];
        ncols = idims[1];
    }

    // --- fabricate the dangling index row ----------------------------------
    uint64_t row[14] = {0};
    const char* ghost_path = "dangling_ghost";
    hid_t path_dset = -1;      // /paths (v2) or /path_table (v3)
    if (version == 3) {
        HARD_CHECK(ncols == 7 && "inject: v3 /index has 7 columns");
        // Append one unique path to /path_table and reference it by id.
        path_dset = H5Dopen2(f, "path_table", H5P_DEFAULT);
        HARD_CHECK(path_dset >= 0 && "inject: /path_table present in v3");
        hid_t ps = H5Dget_space(path_dset);
        hsize_t pdims[1] = {0};
        H5Sget_simple_extent_dims(ps, pdims, NULL);
        H5Sclose(ps);
        const uint32_t ghost_id = (uint32_t)pdims[0];

        row[0] = 999999;                                       // offset far past extent
        row[1] = 4;                                            // count
        row[2] = (uint64_t)ghost_id | ((uint64_t)PANZER_NO_PARENT_ROW_V3 << 32);
        row[3] = 0 | ((uint64_t)0 << 32);                      // time 0, flags 0, ndim 0
        // Extend /path_table by one entry, then the index row by one.
        hsize_t new_pdims[1] = {pdims[0] + 1};
        H5Dset_extent(path_dset, new_pdims);
        ps = H5Dget_space(path_dset);
        hsize_t poff[1] = {pdims[0]};
        hsize_t pcnt[1] = {1};
        H5Sselect_hyperslab(ps, H5S_SELECT_SET, poff, NULL, pcnt, NULL);
        hid_t pms = H5Screate_simple(1, pcnt, NULL);
        hid_t ptype = H5Tcopy(H5T_C_S1);
        H5Tset_size(ptype, 256);
        H5Tset_strpad(ptype, H5T_STR_NULLPAD);
        H5Dwrite(path_dset, ptype, pms, ps, H5P_DEFAULT, ghost_path);
        H5Tclose(ptype); H5Sclose(pms); H5Sclose(ps);
    } else {
        HARD_CHECK((ncols == 12 || ncols == 14) && "inject: known v2/legacy /index layout");
        const bool legacy = (ncols == 14);
        const unsigned off_c    = legacy ? 9  : 8;
        const unsigned cnt_c    = legacy ? 10 : 9;
        const unsigned flags_c  = legacy ? 11 : 10;
        const unsigned parent_c = legacy ? 12 : 11;
        row[off_c]    = 999999;                  // offset -> far past data extent
        row[cnt_c]    = 4;                       // count
        row[flags_c]  = 0;                       // flags  -> kind 0 (data), dtype FLOAT64
        row[parent_c] = PANZER_NO_PARENT_ROW;    // parent (root)

        path_dset = H5Dopen2(f, "paths", H5P_DEFAULT);
        HARD_CHECK(path_dset >= 0 && "inject: /paths present in v2/legacy");
        hid_t ids = H5Dget_space(path_dset);
        hsize_t pdims[1] = {0};
        H5Sget_simple_extent_dims(ids, pdims, NULL);
        H5Sclose(ids);
        hsize_t new_paths_dims[1] = {pdims[0] + 1};
        H5Dset_extent(path_dset, new_paths_dims);
        hid_t ploc  = H5Dget_space(path_dset);
        hsize_t poff[1] = {pdims[0]};
        hsize_t pcnt[1] = {1};
        H5Sselect_hyperslab(ploc, H5S_SELECT_SET, poff, NULL, pcnt, NULL);
        hid_t pms = H5Screate_simple(1, pcnt, NULL);
        char ghost_buf[256] = {0};
        std::snprintf(ghost_buf, sizeof ghost_buf, "%s", ghost_path);
        H5Dwrite(path_dset, H5Dget_type(path_dset), pms, ploc, H5P_DEFAULT, ghost_buf);
        H5Sclose(pms); H5Sclose(ploc);
    }

    // --- /index: extend by one row and write the fabricated row ------------
    {
        hsize_t new_index_dims[2] = {irows + 1, ncols};
        H5Dset_extent(index_dset, new_index_dims);
        hid_t iloc = H5Dget_space(index_dset);
        hsize_t ioff[2] = {irows, 0};
        hsize_t icnt[2] = {1, ncols};
        H5Sselect_hyperslab(iloc, H5S_SELECT_SET, ioff, NULL, icnt, NULL);
        hid_t ims = H5Screate_simple(2, icnt, NULL);
        H5Dwrite(index_dset, H5T_NATIVE_UINT64, ims, iloc, H5P_DEFAULT, row);
        H5Sclose(ims); H5Sclose(iloc);
    }

    H5Dclose(path_dset);
    H5Dclose(index_dset);
    H5Fflush(f, H5F_SCOPE_GLOBAL);
    H5Fclose(f);
}

int main() {
    std::cout << "=== TEST PanzerDB: commit ordering + dangling-row guardrail ===\n\n";
    const std::string filename = "test_commit_guardrail.panzer";
    if (fs::exists(filename)) fs::remove(filename);

    // --- write one valid scalar (baseline) --------------------------------
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        const double val = 3.14;
        db.writeData("s_f64", {}, &val, 1);
        db.flush();
        db.close();
    }

    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        std::vector<PanzerDB::Leaf> leaves = db.getLeaves();
        HARD_CHECK(leaves.size() == 1 && "baseline: exactly one leaf");

        const PanzerDB::Leaf* l = find_leaf(leaves, "s_f64");
        HARD_CHECK(l && "baseline: missing s_f64");
        double r = 0;
        db.readTensor(*l, &r);
        HARD_CHECK(r == 3.14);

        db.close();
        std::cout << "  [OK] baseline: s_f64 writes and round-trips (1 leaf)\n";
    }

    // --- inject the dangling row, then re-read ----------------------------
    inject_dangling_row(filename);

    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        std::vector<PanzerDB::Leaf> leaves = db.getLeaves();

        // The guardrail must have skipped the dangling row: still just one leaf,
        // and the ghost path must not be present in the leaf table.
        HARD_CHECK(leaves.size() == 1 && "dangling row must be skipped");
        HARD_CHECK(find_leaf(leaves, "dangling_ghost") == nullptr &&
                   "dangling row must not be exposed");

        // The valid leaf is unaffected.
        const PanzerDB::Leaf* l = find_leaf(leaves, "s_f64");
        HARD_CHECK(l && "valid leaf must remain after guardrail skip");
        double r = 0;
        db.readTensor(*l, &r);
        HARD_CHECK(r == 3.14);

        db.close();
        std::cout << "  [OK] dangling row (offset past extent) skipped; valid leaf intact\n";
    }

    fs::remove(filename);
    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}
