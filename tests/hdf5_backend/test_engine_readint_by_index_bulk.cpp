// @file  test_engine_readint_by_index_bulk.cpp
// @brief Parity of the single-leaf branch of PanzerDB::readIntDataByIndex with
//        PanzerDB::readDataByIndex (FLOAT64): a bulk-written time series stored
//        in ONE leaf (ndim=k, count=n_slices*prod(shape)) must report the time
//        dimension (+1, last dim = count/slice_vol), not just the spatial ndim.
//
//        Before the fix the int variant returned ndim_out = shape.size() for a
//        single leaf no matter how many slices were packed, so a scalar int
//        time series (ndim=0, count=N) read back as a 0D value.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    const std::string filename = "test_readint_by_index_bulk.pane";
    if (fs::exists(filename)) fs::remove(filename);

    const int N = 3;
    std::vector<double> times(N); for (int k = 0; k < N; ++k) times[k] = 100.0 + k;
    std::vector<int32_t> flags(N); for (int k = 0; k < N; ++k) flags[k] = 1 + 2 * k;

    // Phase 1: scalar (rank-0) time series, one leaf with count = N
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        db.writeDataSlices("time", {1}, times.data(), N, "");
        db.writeDataSlices("flag", {}, flags.data(), N, "");   // bulk, spatial scalar
        db.close();
    }

    // Phase 2: aggregate read (time_index = -1)
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        uint64_t ndim = 0;
        uint64_t shape[6] = {0};
        int32_t* d = nullptr;
        if (db.readIntDataByIndex("flag", -1, &ndim, shape, &d) != 0) {
            std::cerr << "FAIL: readIntDataByIndex aggregate failed\n";
            return 1;
        }

        if (ndim != 1) {
            std::cerr << "FAIL: bulk int series (ndim=0, count=" << N
                      << ") reported ndim_out=" << ndim << ", expected 1\n";
            if (d) free(d);
            return 1;
        }
        if (shape[0] != (uint64_t)N) {
            std::cerr << "FAIL: last dim=" << shape[0] << ", expected " << N << "\n";
            if (d) free(d);
            return 1;
        }
        for (int k = 0; k < N; ++k) {
            if (d[k] != flags[k]) {
                std::cerr << "FAIL: bulk int series value at " << k << ": got "
                          << d[k] << ", expected " << flags[k] << "\n";
                free(d);
                return 1;
            }
        }
        free(d);

        // Per-slice read: scalar at each time index, values aligned.
        for (int k = 0; k < N; ++k) {
            uint64_t nd = 0, sh[6] = {0};
            int32_t* v = nullptr;
            if (db.readIntDataByIndex("flag", k, &nd, sh, &v) != 0) {
                std::cerr << "FAIL: per-slice read at index " << k << "\n";
                return 1;
            }
            if (nd != 0 || *v != flags[k]) {
                std::cerr << "FAIL: per-slice value at " << k << ": got "
                          << (v ? *v : -1) << ", expected " << flags[k] << "\n";
                free(v);
                return 1;
            }
            free(v);
        }
    }

    std::cout << "[OK] readIntDataByIndex reports the time dimension for a bulk "
                 "leaf (parity with readDataByIndex)\n";
    return 0;
}
