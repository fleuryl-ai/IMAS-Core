// @file  test_engine_filter_pipeline_order.cpp
// @brief Regression: numeric datasets (e.g. data_raw_f64) must carry the HDF5
//        filter pipeline in the CORRECT order: SHUFFLE (preprocessing) first,
//        DEFLATE (compressor) last.
//
// Historical bug: `H5Pset_deflate` was called BEFORE `H5Pset_shuffle`
// (panzerdb.cpp, createDataset). HDF5 applies filters in add-order, so the
// pipeline on disk ended up [deflate, shuffle]: the 512 KB chunk was compressed
// WITHOUT its byte-lanes being grouped, then shuffled (a no-op permutation on
// already-compressed output). Measured 2026-10-09 on reflectometer_profile:
//   - buggy order:  data_raw_f64 stored at 11 322 578 B (1.817:1)
//   - correct order (h5repack SHUF+GZIP=6 on the same file): 9 138 929 B (2.251:1)
// i.e. ~20% of the file size wasted. Reading was always correct (filters are
// reversed on read), which is why the bug was invisible to data-checks.
#include "panzerdb.h"
#include <H5Fpublic.h>
#include <H5Dpublic.h>
#include <H5Ppublic.h>
#include <H5Zpublic.h>
#include <iostream>
#include <cstdlib>
#include <filesystem>

// Filter introspection: H5Pget_nfilters (count) + H5Pget_filter2 (id returned,
// standard HDF5). This particular build (HDF5 1.14.4 H5_PUBCONF) exposes only
// the H5Pget_filter2 signature, no H5Pget_filter_count.
static H5Z_filter_t filter_id(hid_t cpl, int idx) {
    unsigned flags = 0;
    return H5Pget_filter2(cpl, (unsigned)idx, &flags, (size_t*)nullptr,
                          (unsigned*)nullptr, (size_t)0, (char*)nullptr, (unsigned*)nullptr);
}

namespace fs = std::filesystem;

static void check_pipeline(const std::string& ffile, const char* dset) {
    hid_t f = H5Fopen(ffile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) { std::cerr << "FAIL: cannot open " << ffile << "\n"; exit(1); }
    hid_t d = H5Dopen2(f, dset, H5P_DEFAULT);
    if (d < 0) { std::cerr << "FAIL: cannot open dataset " << dset << "\n"; exit(1); }
    hid_t cpl = H5Dget_create_plist(d);
    if (cpl < 0) { std::cerr << "FAIL: cannot read creation plist\n"; exit(1); }
    int nf = H5Pget_nfilters(cpl);
    if (nf != 2) {
        std::cerr << "FAIL: " << dset << " has " << nf
                  << " filters (expected exactly 2: shuffle + deflate)\n";
        exit(1);
    }
    H5Z_filter_t f0 = filter_id(cpl, 0);
    H5Z_filter_t f1 = filter_id(cpl, 1);
    std::cout << "  " << dset << ": pipeline[0]=filter " << f0
              << " | pipeline[1]=filter " << f1
              << " (expected shuffle=2, deflate=" << H5Z_FILTER_DEFLATE << ")\n";
    if (f0 != H5Z_FILTER_SHUFFLE) {
        std::cerr << "REGRESSION: " << dset << " pipeline[0] is not SHUFFLE.\n";
        exit(1);
    }
    if (f1 != H5Z_FILTER_DEFLATE) {
        std::cerr << "REGRESSION: " << dset << " pipeline[1] is not DEFLATE (compressor must be last).\n";
        exit(1);
    }
    H5Pclose(cpl);
    H5Dclose(d);
    H5Fclose(f);
}

int main() {
    const std::string ffile = "test_filter_pipeline_order.panzer";
    fs::remove(ffile);

    {
        // Writing one double is enough to materialize `data_raw_f64` with the
        // compression pipeline; the pipeline is set at dataset creation.
        PanzerDB db(ffile, PanzerDB::OpenMode::WRITE, true);
        double v = 1.0;
        db.writeData("temperature", {}, &v, 1);
        db.close();
    }

    std::cout << "=== checking numeric buffer (data_raw_f64) ===\n";
    check_pipeline(ffile, "data_raw_f64");

    std::cout << "=== checking index table ===\n";
    check_pipeline(ffile, "index");

    fs::remove(ffile);
    std::cout << "\nPASS: filter pipeline order is SHUFFLE then DEFLATE\n";
    return 0;
}
