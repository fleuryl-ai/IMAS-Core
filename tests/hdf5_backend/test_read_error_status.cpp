// test_read_error_status.cpp
// Pins the error contract of PanzerDB::readTensor for numeric payloads
// (REVIEW_PERF_hdf5_v2.md item E1).
//
// The numeric readTensor used to call H5Dread and test its return status in an
// EMPTY if-block, so an HDF5 failure returned as a success with whatever the
// caller's buffer happened to hold.  A Leaf whose [offset, offset+count) runs
// past the payload extent is exactly what a half-written index produces, so
// this is not a hypothetical: the caller used to get uninitialized memory
// instead of an error.  (The std::string specialization already threw.)
//
// Leaves are built by hand here rather than taken from getLeaves(): the file
// guardrail in getLeaves() skips data rows that run past the materialised
// extent, so an out-of-range leaf cannot be reached through the index -- it has
// to be handed to readTensor directly, which is also its public surface.

#include "panzerdb.h"

#include <hdf5.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool cond, const std::string& what) {
    if (!cond) {
        std::cout << "  FAIL: " << what << "\n";
        ++failures;
    } else {
        std::cout << "  ok:   " << what << "\n";
    }
}

// HDF5 dumps a full error stack on every expected failure; keep the report
// readable by muting it around the calls that are meant to fail.
class MutedHdf5Errors {
public:
    MutedHdf5Errors() { H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); }
    ~MutedHdf5Errors() { H5Eset_auto2(H5E_DEFAULT, (H5E_auto2_t)H5Eprint2, stderr); }
};

PanzerDB::Leaf make_leaf(PanzerDB::DataType dt, uint64_t offset, uint64_t count) {
    PanzerDB::Leaf leaf;
    leaf.shape[0] = count;
    leaf.ndim = 1;
    leaf.time_index = 0;
    leaf.offset = offset;
    leaf.count = count;
    leaf.flags = (static_cast<uint64_t>(dt) << 4) | 0; // upper bits: DataType, low 4 bits: node kind
    leaf.is_empty = false;
    return leaf;
}

template <typename Throwed>
bool throws(const PanzerDB& db, const PanzerDB::Leaf& leaf, double* buf) {
    MutedHdf5Errors mute;
    try {
        db.readTensor(leaf, buf);
    } catch (const Throwed&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

} // namespace

int main() {
    std::cout << "=== TEST PanzerDB: readTensor propagates HDF5 read errors ===\n\n";
    const std::string filename = "test_read_error_status.panzer";
    std::remove(filename.c_str());

    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        std::vector<double> payload = {1.5, 2.5, 3.5, 4.5};
        db.writeData("sig", {payload.size()}, payload.data(), payload.size());
        db.flush();
    }

    PanzerDB db(filename, PanzerDB::OpenMode::READ);

    std::cout << "-- a leaf inside the extent still reads back exactly\n";
    double buf[4] = {0.0, 0.0, 0.0, 0.0};
    db.readTensor(make_leaf(PanzerDB::DataType::FLOAT64, 0, 4), buf);
    check(buf[0] == 1.5 && buf[1] == 2.5 && buf[2] == 3.5 && buf[3] == 4.5,
          "offset 0, count 4 -> 1.5 2.5 3.5 4.5");

    std::cout << "-- a leaf starting past the extent must raise, not return garbage\n";
    for (uint64_t off : {uint64_t(1) << 30, uint64_t(1) << 62}) {
        check(throws<ALBackendException>(db, make_leaf(PanzerDB::DataType::FLOAT64, off, 4), buf),
              "offset " + std::to_string(off) + " past the extent -> ALBackendException");
    }

    std::cout << "-- a leaf starting inside the extent but running past it must raise\n";
    check(throws<ALBackendException>(db, make_leaf(PanzerDB::DataType::FLOAT64, 2, 100000), buf),
          "offset 2, count 100000 -> ALBackendException");

    std::cout << "-- an empty leaf keeps raising (unchanged behaviour)\n";
    {
        PanzerDB::Leaf empty = make_leaf(PanzerDB::DataType::FLOAT64, 0, 0);
        empty.is_empty = true;
        check(throws<std::runtime_error>(db, empty, buf), "empty leaf -> throw");
    }

    std::cout << "\n";
    if (failures == 0) {
        std::cout << "ALL PASSED\n";
        return 0;
    }
    std::cout << failures << " FAILURE(S)\n";
    return 1;
}
