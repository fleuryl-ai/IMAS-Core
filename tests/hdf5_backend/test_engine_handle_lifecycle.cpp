// @file  test_engine_handle_lifecycle.cpp
// @brief Pins that an engine releases every HDF5 handle it opened, so the library can
//        terminate cleanly (see PanzerDB::close).
//
// An engine that opens a file itself owns that file handle. When the tables do not sit
// at the file root — the layout of every real IMAS pulse file, where /index lives inside
// the <ids> group — init() descends: it replaces file_id with the child group and marks
// it as closable (panzerdb.cpp:241-246). close() then closes the group and never the
// file, because the constructor that called H5Fopen never claimed ownership.
//
// Measured consequence on a 171k-row core_profiles copy: one live file id survives every
// engine (H5Fget_obj_count: FILE = 1, 2, 3, ... after each open/close cycle, while
// datasets and groups all return to 0), and the process segfaults at exit inside
// H5_term_library -> H5F_term_package -> H5I_clear_type -> H5VL_file_close -> H5I_find_id
// as HDF5 cleans up files it was never told to close. The data on disk is fine (flush()
// commits), which is why nothing else caught it.
#include "panzerdb.h"
#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

static void check(const char* what, long got, long want) {
    if (got != want) {
        std::cout << "  FAIL: " << what << " = " << got << ", expected " << want << "\n";
        ++failures;
    }
}

/// Live HDF5 ids in this process: what library termination still has to clean up.
static long live_objects() {
    const ssize_t n = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL);
    return n < 0 ? -1 : static_cast<long>(n);
}

static long live_files() {
    const ssize_t n = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_FILE);
    return n < 0 ? -1 : static_cast<long>(n);
}

/// A file laid out like an IMAS pulse file: the engine tables live in a child group, so
/// an engine opened on the FILE has to descend. The loc_id engine writes with
/// close_loc_id_on_exit = false: the test owns the group handle.
static void make_nested_file(const std::string& file, const char* group) {
    const hid_t f = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0) throw std::runtime_error("H5Fcreate failed");
    const hid_t g = H5Gcreate2(f, group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (g < 0) throw std::runtime_error("H5Gcreate2 failed");
    {
        PanzerDB db(g, PanzerDB::OpenMode::WRITE, true, false);
        std::vector<double> v = {1.0, 2.0};
        db.writeData("signal", {2}, v.data(), 2);
    }
    H5Gclose(g);
    H5Fclose(f);
}

/// Open and destroy `cycles` engines on one file; every handle they opened must be gone
/// when they are, and no more files must be registered than at the start.
static void check_cycles(const char* what, const std::string& file, int cycles) {
    const long base_ids = live_objects();
    const long base_files = live_files();
    for (int i = 0; i < cycles; ++i) {
        PanzerDB db(file, PanzerDB::OpenMode::APPEND);   // init() decodes /index
    }
    check(what, live_objects() - base_ids, 0);
    check(what, live_files() - base_files, 0);
}

int main() {
    const std::string nested = "test_engine_handle_lifecycle.panzer";
    const std::string flat = "test_engine_handle_lifecycle_flat.panzer";
    std::filesystem::remove(nested);
    std::filesystem::remove(flat);
    std::cout << "=== TEST engine releases every HDF5 handle it opened ===\n";
    const long base = live_objects();
    check("the id counter works", base < 0 ? -1 : 0, 0);

    // The pulse-file layout: /index inside a child group, so init() descends.
    make_nested_file(nested, "core_profiles_3");
    check("creation released its handles", live_objects(), base);
    check_cycles("nested layout, 5 APPEND engines", nested, 5);

    // The flat layout: /index at the file root, no descent. This one already worked —
    // close() recognises a file id and closes it — and must keep working.
    {
        PanzerDB db(flat, PanzerDB::OpenMode::WRITE, true);
        std::vector<double> v = {3.0};
        db.writeData("other", {1}, v.data(), 1);
    }
    check("flat creation released its handles", live_objects(), base);
    check_cycles("flat layout, 5 APPEND engines", flat, 5);

    std::filesystem::remove(nested);
    std::filesystem::remove(flat);
    if (failures) { std::cout << failures << " FAILURE(S)\n"; return 1; }
    std::cout << "ALL PASSED\n";
    return 0;
}
