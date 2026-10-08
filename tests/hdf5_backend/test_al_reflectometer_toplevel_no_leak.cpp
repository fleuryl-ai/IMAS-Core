// @file  test_al_reflectometer_toplevel_no_leak.cpp
// @brief Regression for the round-trip 2x inflation bug.
//
// Symptom: after a v2->v2 read->write round-trip, reflectometer_profile got a
// whole extra set of root-level arrays (n_e/data, position/r|phi|z) that were
// byte-identical to the channel(0)/... members, doubling the file size.
//
// Root cause: IReadStrategy's name_index "last path segment" fallback. When the
// AL asked for the record-top-level node n_e/data (legitimate IMAS "reconstruction"
// node, absent from the file), find_path() missed and find_name("n_e&data") hit
// the first leaf sharing that last segment — channel(0)/n_e&data — because the
// top-level context_prefix was empty and carried no parent constraint. The reader
// then returned the channel's array for the top-level node, and the writer
// faithfully duplicated it.
//
// Fix: when the query context is the record root (empty context prefix), the
// name-fallback must only accept leaves whose own parent is also the record root
// (leaf->parent_path.empty()), so an AoS member can never be returned in place of
// a root-level node.
//
// This test writes a reflectometer_profile with ONLY channel(0) filled (n_e/data
// and position/r) and, on read, asserts that:
//   * the top-level n_e/data and position/r are EMPTY, i.e. the channel is not
//     leaking into the root-level reconstruction node; AND
//   * channel(0)/n_e/data and channel(0)/position/r still return the written data
//     (no regression on the legitimate member read).
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <cstring>

namespace fs = std::filesystem;

#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define GREEN   "\033[32m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"

const std::string URI = "imas:hdf5?path=./test_db_reflectometer_toplevel_no_leak";

const int AOSSIZE = 1;   // one channel
const int N = 8;         // profile length

void refl_put() {
    std::cout << YELLOW << "\n--- Phase: reflectometer_put() ---" << RESET << std::endl;

    DataEntryContext dataEntryCtx(URI);
    HDF5Backend backend;
    backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

    OperationContext opCtx(&dataEntryCtx, "reflectometer_profile", "", WRITE_OP);
    backend.beginAction(&opCtx);

    // A single time point + a couple of top-level scalars so the record is well-formed.
    int homogeneous_time = 1;
    backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time,
                      alconst::integer_data, 0, nullptr);

    std::vector<double> time_values(1);
    time_values[0] = 0.0;
    int time_dim = 1; int time_size[] = {1};
    backend.writeData(&opCtx, "time", "time", time_values.data(),
                      alconst::double_data, time_dim, time_size);

    // channel(0) member arrays — the ONLY place the profile is stored.
    ArraystructContext channelCtx(&opCtx, "channel", "");
    backend.beginArraystructAction(&channelCtx, (int*)&AOSSIZE);

    std::vector<double> n_e_vals(N);
    std::vector<double> r_vals(N);
    for (int i = 0; i < N; ++i) {
        n_e_vals[i] = 1e19 * (1.0 + i);
        r_vals[i]   = 1.05 + 0.01 * i;
    }
    int dim = 1; int size[] = {N};
    backend.writeData(&channelCtx, "n_e/data", "", n_e_vals.data(), alconst::double_data, dim, size);
    backend.writeData(&channelCtx, "position/r", "", r_vals.data(), alconst::double_data, dim, size);

    backend.endAction(&channelCtx);
    backend.endAction(&opCtx);
    backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
    std::cout << GREEN << "[OK] reflectometer_put completed." << RESET << std::endl;
}

int main() {
    fs::remove_all(URI.substr(7)); // ./test_db_...

    refl_put();

    DataEntryContext dataEntryCtx(URI);
    HDF5Backend backend;
    backend.openPulse(&dataEntryCtx, OPEN_PULSE);

    OperationContext opCtx(&dataEntryCtx, "reflectometer_profile", "", READ_OP);
    backend.beginAction(&opCtx);

    void* data = nullptr;
    int type = alconst::double_data;
    int dim = -99;
    int size[5] = {0, 0, 0, 0, 0};
    int failed = 0;

    // 0) channel(0) member reads MUST return the written data (sanity — the data
    //    is present in the file under the channel; this is the "source of truth"
    //    the round-trip must not duplicate at the root).
    ArraystructContext channelCtx(&opCtx, "channel", "");
    int c_size = 0;
    backend.beginArraystructAction(&channelCtx, &c_size);

    dim = -99; size[0] = -1;
    int rc = backend.readData(&channelCtx, "n_e/data", "", &data, &type, &dim, size);
    if (rc != 1 || dim != 1 || size[0] != N) {
        std::cerr << RED << "[FAIL] channel(0)/n_e/data read sanity: rc=" << rc
                  << " dim=" << dim << " size[0]=" << size[0]
                  << " (expected rc=1 dim=1 size[0]=" << N << ")" << RESET << std::endl;
        failed = 1;
    }
    if (data && rc == 1) {
        double* n_e = (double*)data;
        for (int i = 0; i < N; ++i)
            if (std::abs(n_e[i] - 1e19 * (1.0 + i)) > 1.0) {
                std::cerr << RED << "[FAIL] channel(0)/n_e/data value mismatch" << RESET << std::endl;
                failed = 1;
            }
        free(data); data = nullptr;
    }

    dim = -99; size[0] = -1;
    rc = backend.readData(&channelCtx, "position/r", "", &data, &type, &dim, size);
    if (rc != 1 || dim != 1 || size[0] != N) {
        std::cerr << RED << "[FAIL] channel(0)/position/r read sanity: rc=" << rc
                  << " dim=" << dim << " size[0]=" << size[0]
                  << " (expected rc=1 dim=1 size[0]=" << N << ")" << RESET << std::endl;
        failed = 1;
    }
    if (data && rc == 1) {
        double* r = (double*)data;
        for (int i = 0; i < N; ++i)
            if (std::abs(r[i] - (1.05 + 0.01 * i)) > 1e-9) {
                std::cerr << RED << "[FAIL] channel(0)/position/r value mismatch" << RESET << std::endl;
                failed = 1;
            }
        free(data); data = nullptr;
    }
    backend.endAction(&channelCtx);

    // 1) TOP-LEVEL reconstruction node must NOT return the channel's N elements.
    //    This is the regression the fix addresses (name_index last-segment fallback).
    //    Accept: rc==1 && dim==0 (empty success), rc==0 (not found), dim<0, or
    //    anything else that is NOT dim==1 && size[0]==N. All of these mean the
    //    channel data did not leak into the top-level node.
    dim = -99; size[0] = -1;
    rc = backend.readData(&opCtx, "n_e/data", "", &data, &type, &dim, size);
    bool top_level_leak = (rc == 1 && dim == 1 && size[0] == N);
    if (top_level_leak) {
        std::cerr << RED << "[REGRESSION] top-level n_e/data returned the channel's array"
                  << " (dim=1 size[0]=" << N << "). Bug is back." << RESET << std::endl;
        failed = 1;
    }
    if (data) { free(data); data = nullptr; }
    std::cout << (top_level_leak ? RED : GREEN)
              << "  top-level n_e/data: rc=" << rc << " dim=" << dim
              << " size[0]=" << (dim==1 ? size[0] : -1) << RESET << std::endl;

    dim = -99; size[0] = -1;
    rc = backend.readData(&opCtx, "position/r", "", &data, &type, &dim, size);
    bool top_pos_leak = (rc == 1 && dim == 1 && size[0] == N);
    if (top_pos_leak) {
        std::cerr << RED << "[REGRESSION] top-level position/r returned the channel's array"
                  << " (dim=1 size[0]=" << N << "). Bug is back." << RESET << std::endl;
        failed = 1;
    }
    if (data) { free(data); data = nullptr; }
    std::cout << (top_pos_leak ? RED : GREEN)
              << "  top-level position/r: rc=" << rc << " dim=" << dim
              << " size[0]=" << (dim==1 ? size[0] : -1) << RESET << std::endl;

    backend.endAction(&opCtx);
    backend.closePulse(&dataEntryCtx, OPEN_PULSE);

    fs::remove_all(URI.substr(7));
    if (failed) {
        std::cerr << BOLD << RED << "\n[FAIL] reflectometer toplevel/no-leak test" << RESET << std::endl;
        return 1;
    }
    std::cout << BOLD << GREEN << "\n[PASS] reflectometer toplevel/no-leak test" << RESET << std::endl;
    return 0;
}
