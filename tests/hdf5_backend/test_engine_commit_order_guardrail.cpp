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
#include "panzerdb.h"
#include <hdf5.h>

#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

#include "fixtures/panzerdb_helper.h"

// Append one fabricated DANGLING row to /index AND /paths of an existing file.
// Its offset is set far past the data_raw_f64 extent so the reader's guardrail
// must skip it. Both datasets are extended by exactly one row so their counts
// stay consistent (getLeaves reads /paths only when paths_count >= n_rows).
static void inject_dangling_row(const std::string& path) {
    hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    assert(f >= 0 && "inject: open RW");

    hid_t index_dset = H5Dopen2(f, "index", H5P_DEFAULT);
    hid_t paths_dset = H5Dopen2(f, "paths", H5P_DEFAULT);
    assert(index_dset >= 0 && paths_dset >= 0);

    // --- /index: extend by one row -----------------------------------------
    hid_t iids = H5Dget_space(index_dset);
    hsize_t idims[2];
    H5Sget_simple_extent_dims(iids, idims, NULL);
    H5Sclose(iids);
    hsize_t irows = idims[0];
    hsize_t new_index_dims[2] = {irows + 1, 14};
    H5Dset_extent(index_dset, new_index_dims);

    // --- /paths: keep count consistent, add a dummy path -------------------
    hid_t ids = H5Dget_space(paths_dset);
    hsize_t pdims[1];
    H5Sget_simple_extent_dims(ids, pdims, NULL);
    H5Sclose(ids);
    hsize_t pcount = pdims[0];
    hsize_t new_paths_dims[1] = {pcount + 1};
    H5Dset_extent(paths_dset, new_paths_dims);

    hid_t ptype = H5Dget_type(paths_dset);
    hid_t ploc  = H5Dget_space(paths_dset);
    hsize_t poff[1] = {pcount};
    hsize_t pcnt[1] = {1};
    H5Sselect_hyperslab(ploc, H5S_SELECT_SET, poff, NULL, pcnt, NULL);
    hid_t pms = H5Screate_simple(1, pcnt, NULL);
    char ghost_path[256];
    std::snprintf(ghost_path, sizeof ghost_path, "dangling_ghost");
    H5Dwrite(paths_dset, ptype, pms, ploc, H5P_DEFAULT, ghost_path);
    H5Sclose(pms); H5Sclose(ploc); H5Tclose(ptype);

    // --- fabricate the dangling index row ----------------------------------
    // Layout: {type, ndim, shape[6], time_idx, offset, count, flags, parent_id,
    // index_value}. flags = 0  =>  kind 0 (data), dtype FLOAT64. The offset is
    // intentionally far past the data_raw_f64 extent.
    uint64_t row[14] = {0};
    row[0]  = 0;                       // type (unused; dtype encoded in flags)
    row[1]  = 0;                       // ndim
    row[8]  = 0;                       // time_idx
    row[9]  = 999999;                  // offset -> far past data extent
    row[10] = 4;                       // count
    row[11] = 0;                       // flags  -> kind 0 (data), dtype FLOAT64
    row[12] = PANZER_NO_PARENT_ROW;    // parent (root)
    row[13] = 0;                       // instance

    hid_t iloc = H5Dget_space(index_dset);
    hsize_t ioff[2] = {irows, 0};
    hsize_t icnt[2] = {1, 14};
    H5Sselect_hyperslab(iloc, H5S_SELECT_SET, ioff, NULL, icnt, NULL);
    hid_t ims = H5Screate_simple(2, icnt, NULL);
    H5Dwrite(index_dset, H5T_NATIVE_UINT64, ims, iloc, H5P_DEFAULT, row);
    H5Sclose(ims); H5Sclose(iloc);

    H5Dclose(paths_dset);
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
        assert(leaves.size() == 1 && "baseline: exactly one leaf");

        const PanzerDB::Leaf* l = find_leaf(leaves, "s_f64");
        assert(l && "baseline: missing s_f64");
        double r = 0;
        db.readTensor(*l, &r);
        assert(r == 3.14);

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
        assert(leaves.size() == 1 && "dangling row must be skipped");
        assert(find_leaf(leaves, "dangling_ghost") == nullptr &&
               "dangling row must not be exposed");

        // The valid leaf is unaffected.
        const PanzerDB::Leaf* l = find_leaf(leaves, "s_f64");
        assert(l && "valid leaf must remain after guardrail skip");
        double r = 0;
        db.readTensor(*l, &r);
        assert(r == 3.14);

        db.close();
        std::cout << "  [OK] dangling row (offset past extent) skipped; valid leaf intact\n";
    }

    fs::remove(filename);
    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}
