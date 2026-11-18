// @file  test_engine_flush_dataspace_handles.cpp
// @brief Pins that flush() releases every dataspace handle it opens on the way to a commit.
//
// flush() grows each table in two steps: read the space to learn where the tail is, then
// read it again to build the hyperslab the new rows land in. Between the two the first
// handle has to be closed. Five of the six blocks do; the /index block -- the one that
// writes the commit marker, so the one that runs on every step -- did not: the variable
// holding it was overwritten by the second H5Dget_space (panzerdb.cpp:1100 vs :1106).
//
// The library never mentions it: no diagnostic at commit time, none at exit. What it costs
// is measurable -- 20 000 un-closed dataspaces took the process from 10,4 Mo to 21,9 Mo of
// resident memory, ~576 o each -- and it grows with the number of write steps, which is
// unbounded for an engine fed by a pulse.
#include "panzerdb.h"
#include <H5Ipublic.h>
#include <H5Spublic.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(const char* what, long got, long want) {
    if (got != want) {
        std::cout << "  FAIL: " << what << " = " << got << ", expected " << want << "\n";
        ++failures;
    }
}

static int count_one(hid_t /*id*/, void* op_data) {
    ++(*static_cast<long*>(op_data));
    return 0;
}

/// Registered dataspace ids in this process: what the library still has to release.
/// H5Inmembers refuses this id class ("cannot call public function on library type",
/// H5I.c:206); H5Iiterate walks it and returns the count through the callback.
/// Returns -1 when the counter itself is unusable, which the caller reports as a
/// failure -- a counter that stops working must not be able to report success.
static long live_dataspaces() {
    long n = 0;
    return H5Iiterate(H5I_DATASPACE, count_one, &n) < 0 ? -1 : n;
}

int main() {
    const std::string file = "test_engine_flush_dataspace_handles.panzer";
    std::filesystem::remove(file);
    std::cout << "=== TEST flush() closes every dataspace it opens ===\n";

    // Does the counter see a dataspace at all, and see it go away? Open one by hand, count
    // it, close it, count again -- the guard for the guard.
    {
        const hsize_t one = 1;
        const long before = live_dataspaces();
        const hid_t sp = H5Screate_simple(1, &one, nullptr);
        const long opened = live_dataspaces();
        check("the dataspace counter works", (before < 0 || opened < 0 || opened != before + 1) ? -1 : 0, 0);
        H5Sclose(sp);
        check("closing it is counted too", live_dataspaces() - before, 0);
    }

    const long base = live_dataspaces();
    {
        PanzerDB db(file, PanzerDB::OpenMode::WRITE, true);
        const long opened = live_dataspaces();
        check("opening an engine leaves no dataspace behind", opened - base, 0);

        // 200 commits: the defect costs exactly one id per commit, so this is 200 against
        // 0 rather than something inside a tolerance band.
        std::vector<double> v = {1.0, 2.0, 3.0};
        db.writeData("signal", {3}, v.data(), 3);
        for (int i = 0; i < 200; ++i) {
            db.writeData("signal", {3}, v.data(), 3);
            db.flush();
        }
        check("200 commits leave no dataspace behind", live_dataspaces() - opened, 0);
    }
    check("closing the engine leaves no dataspace behind", live_dataspaces() - base, 0);

    // The read path opens a dataspace per leaf too; it must stay clean while we are here.
    {
        PanzerDB db(file, PanzerDB::OpenMode::READ);
        const long opened = live_dataspaces();
        const auto& leaves = db.getLeaves();
        std::vector<double> buf;
        for (const auto& l: leaves) {
            buf.assign(l.count, 0.0);
            db.readTensor(l, buf.data());
        }
        check("reading every leaf leaves no dataspace behind", live_dataspaces() - opened, 0);
    }
    check("the read engine released its dataspaces", live_dataspaces() - base, 0);

    std::filesystem::remove(file);
    if (failures) { std::cout << failures << " FAILURE(S)\n"; return 1; }
    std::cout << "ALL PASSED\n";
    return 0;
}
