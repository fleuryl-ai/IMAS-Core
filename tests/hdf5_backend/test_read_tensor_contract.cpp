// test_read_tensor_contract.cpp
// Pins the error contract of PanzerDB::readTensor for numeric payloads
// (REVIEW_PERF_hdf5_v2.md items E1 and E2).
//
// E1 -- the numeric readTensor used to call H5Dread and test its return status
// in an EMPTY if-block, so an HDF5 failure returned as a success with whatever
// the caller's buffer happened to hold.  A Leaf whose [offset, offset+count)
// runs past the payload extent is exactly what a half-written index produces,
// so this is not a hypothetical: the caller used to get uninitialized memory
// instead of an error.  (The std::string specialization already threw.)
//
// E2 -- readTensor picks the memory type from the LEAF (leaf.flags >> 4), never
// from its template argument.  A buffer whose element is smaller than the
// stored element is therefore overrun by H5Dread: readTensor<int32_t> on a
// FLOAT64 leaf writes 8 bytes per element into a 4-bytes-per-element buffer.
// The opposite direction is a live idiom, not a bug: PanzerDB::
// readInterpolatedData routes integer signals through readDataByIndex(double**),
// which over-allocates 8 bytes per element and whose consumer reads the buffer
// back as int32.  So the guard is about width, and this test pins both halves.
//
// Leaves are built by hand here rather than taken from getLeaves(): the file
// guardrail in getLeaves() skips data rows that run past the materialised
// extent, so an out-of-range leaf cannot be reached through the index -- it has
// to be handed to readTensor directly, which is also its public surface.

#include "panzerdb.h"

#include <hdf5.h>

#include <complex>
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

template <typename Throwed, typename T>
bool throws(const PanzerDB& db, const PanzerDB::Leaf& leaf, T* buf) {
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

template <typename T>
bool raised(const PanzerDB& db, const PanzerDB::Leaf& leaf, T* buf) {
    MutedHdf5Errors mute;
    try {
        db.readTensor(leaf, buf);
    } catch (...) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    std::cout << "=== TEST PanzerDB: readTensor error contract ===\n\n";
    const std::string filename = "test_read_tensor_contract.panzer";
    std::remove(filename.c_str());

    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        std::vector<double> payload = {1.5, 2.5, 3.5, 4.5};
        db.writeData("sig", {payload.size()}, payload.data(), payload.size());
        std::vector<int32_t> ints = {10, 20, 30, 40};
        db.writeData("isig", {ints.size()}, ints.data(), ints.size());
        db.flush();
    }

    PanzerDB db(filename, PanzerDB::OpenMode::READ);

    std::cout << "-- E1: a leaf inside the extent still reads back exactly\n";
    double buf[4] = {0.0, 0.0, 0.0, 0.0};
    db.readTensor(make_leaf(PanzerDB::DataType::FLOAT64, 0, 4), buf);
    check(buf[0] == 1.5 && buf[1] == 2.5 && buf[2] == 3.5 && buf[3] == 4.5,
          "offset 0, count 4 -> 1.5 2.5 3.5 4.5");

    std::cout << "-- E1: a leaf starting past the extent must raise, not return garbage\n";
    for (uint64_t off : {uint64_t(1) << 30, uint64_t(1) << 62}) {
        check(throws<ALBackendException>(db, make_leaf(PanzerDB::DataType::FLOAT64, off, 4), buf),
              "offset " + std::to_string(off) + " past the extent -> ALBackendException");
    }

    std::cout << "-- E1: a leaf starting inside the extent but running past it must raise\n";
    check(throws<ALBackendException>(db, make_leaf(PanzerDB::DataType::FLOAT64, 2, 100000), buf),
          "offset 2, count 100000 -> ALBackendException");

    std::cout << "-- E1: an empty leaf keeps raising (unchanged behaviour)\n";
    {
        PanzerDB::Leaf empty = make_leaf(PanzerDB::DataType::FLOAT64, 0, 0);
        empty.is_empty = true;
        check(throws<std::runtime_error>(db, empty, buf), "empty leaf -> throw");
    }

    // Slack so the pre-fix write past the intended 4 elements stays inside the
    // allocation: the point of the check is the missing raise, not the spill.
    std::cout << "-- E2: a buffer element narrower than the stored element must raise\n";
    {
        std::vector<int32_t> wide(64, 0x5a5a5a5a);
        check(throws<ALBackendException>(db, make_leaf(PanzerDB::DataType::FLOAT64, 0, 4), wide.data()),
              "readTensor<int32_t> on a FLOAT64 leaf -> ALBackendException");
    }
    {
        std::vector<std::complex<double>> wider(64);
        check(!raised(db, make_leaf(PanzerDB::DataType::INT32, 0, 4), wider.data()),
              "readTensor<complex> on an INT32 leaf -> accepted, the buffer is wider");
    }

    std::cout << "-- E2: exact match still reads back exactly\n";
    {
        int32_t ibuf[4] = {0, 0, 0, 0};
        db.readTensor(make_leaf(PanzerDB::DataType::INT32, 0, 4), ibuf);
        check(ibuf[0] == 10 && ibuf[1] == 20 && ibuf[2] == 30 && ibuf[3] == 40,
              "readTensor<int32_t> on an INT32 leaf -> 10 20 30 40");
    }

    std::cout << "-- E2: a WIDER buffer is the live integer-via-double idiom, must not raise\n";
    {
        double dbuf[4] = {0.0, 0.0, 0.0, 0.0};
        db.readTensor(make_leaf(PanzerDB::DataType::INT32, 0, 4), dbuf);
        const int32_t* as_ints = reinterpret_cast<const int32_t*>(dbuf);
        check(as_ints[0] == 10 && as_ints[1] == 20 && as_ints[2] == 30 && as_ints[3] == 40,
              "readTensor<double> on an INT32 leaf -> int32 values at the front of the buffer");
    }

    std::cout << "\n";
    if (failures == 0) {
        std::cout << "ALL PASSED\n";
        return 0;
    }
    std::cout << failures << " FAILURE(S)\n";
    return 1;
}
