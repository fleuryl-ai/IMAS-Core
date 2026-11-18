// @file  test_engine_list_spans_over_write.cpp
// @brief A second WRITE engine on one file must not inherit the previous life's span table.
//
// A WRITE engine handed a loc_id reuses the file instead of truncating it -- which is what
// the AL does for every ids_put(): HDF5Writer_v2::setWriteStrategy builds a PanzerDB(WRITE)
// on the IDS group it just opened. That engine deletes the tables it is about to recreate,
// and /list_spans has to be one of them: while the name is still there the create fails, the
// new span rows are dropped (flush has no dataset to put them in) and the OLD ones stay on
// disk keyed by row_id. Row ids restart at 0 after the truncate, so those stale rows describe
// the new data with the old element boundaries: the string comes back wrong, not truncated.
//
// The two lives are built so their span tables have the SAME row count -- a slice records one
// entry per element as soon as one of them overflows -- so what is compared is the recorded
// boundaries themselves, "row.element:slots", not how many rows there are.
//
// Phase C covers the other way the table can be missing: a file written before the span table
// existed, appended with an element that needs one. The table has to appear on demand, or the
// element is silently cut at one 512 B slot.
#include "panzerdb.h"
#include <hdf5.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int failures = 0;

static void fail(const std::string& what) {
    std::cout << "  FAIL: " << what << "\n";
    ++failures;
}

// A string longer than one 512 B slot, hence stored across several and described by a span.
static std::string long_string(size_t len, char base) {
    std::string s(len, 'x');
    for (size_t i = 0; i < len; ++i) s[i] = (char)(base + (i % 26));
    return s;
}

// The tables live in a child group, like a real pulse file, so the engine is opened on that
// group and its file is reused rather than truncated.
static const char* GROUP = "core_profiles";
static const std::string SPANS = std::string(GROUP) + "/list_spans";

// Slots one string occupies, mirroring the writer.
static long long slots_of(size_t len) { return len ? (long long)((len - 1) / 511 + 1) : 1; }

// Writes one slice of a list of strings, as a fresh engine would.
static void write_list(hid_t loc, PanzerDB::OpenMode mode, const std::vector<std::string>& items) {
    std::vector<const char*> flat;
    for (const std::string& s : items) flat.push_back(s.c_str());
    PanzerDB db(loc, mode, true, false);
    db.beginArray("A", 1);
    db.beginArray("B", "time");
    db.writeDataSlices("labels", {items.size()}, flat.data(), 1, "time");
    db.endArray();
    db.endArray();
    db.close();
}

// Reads back one slice and compares it against what was written.
static void check_list(PanzerDB& db, int slice, const std::vector<std::string>& want) {
    uint64_t ndim = 0, shape[6] = {0};
    char* data = nullptr;
    const int rc = db.readStringDataByIndex("A/0/B/labels", slice, &ndim, shape, &data);
    if (rc != 0 || !data) {
        fail("slice " + std::to_string(slice) + " of A/0/B/labels did not come back (rc="
             + std::to_string(rc) + ")");
        return;
    }
    if (ndim != 2 || shape[0] != want.size()) {
        std::cout << "  FAIL: slice " << slice << " came back as " << ndim << "D ["
                  << shape[0] << " x " << shape[1] << "], expected 2D [" << want.size()
                  << " x ...]\n";
        ++failures;
    } else {
        for (size_t i = 0; i < want.size(); ++i) {
            const std::string got(data + i * shape[1]);
            if (got != want[i]) {
                std::cout << "  FAIL: slice " << slice << " element " << i << " is "
                          << got.size() << " chars (" << got.substr(0, 24) << "...), expected "
                          << want[i].size() << " chars (" << want[i].substr(0, 24) << "...)\n";
                ++failures;
            }
        }
    }
    free(data);
}

// The span table as sorted "element:slots" pairs -- what tells one life of the file from
// another. Row ids are left out: they count the enclosing AoS rows too, so they are an
// internal detail, but they are printed when a check fails. Empty when the dataset is absent.
static std::vector<std::string> read_spans(hid_t file, std::vector<std::string>& rows_out) {
    std::vector<std::string> out;
    if (H5Lexists(file, SPANS.c_str(), H5P_DEFAULT) <= 0) return out;
    const hid_t d = H5Dopen2(file, SPANS.c_str(), H5P_DEFAULT);
    if (d < 0) { fail(SPANS + " cannot be opened"); return out; }
    const hid_t s = H5Dget_space(d);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(s, dims, nullptr);
    H5Sclose(s);
    std::vector<uint64_t> raw(dims[0]);
    if (!raw.empty() && H5Dread(d, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.data()) < 0)
        fail(SPANS + " cannot be read");
    H5Dclose(d);

    for (size_t i = 0; i + 3 < raw.size(); i += 4) {
        out.push_back(std::to_string(raw[i + 1]) + ":" + std::to_string(raw[i + 3]));
        const std::string row = std::to_string(raw[i]);
        if (std::find(rows_out.begin(), rows_out.end(), row) == rows_out.end())
            rows_out.push_back(row);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// The boundaries one slice of `items` must leave in the table: every element of a slice is
// recorded as soon as one of them overflows; a slice with no long element records nothing.
static std::vector<std::string> expected_spans(const std::vector<std::string>& items) {
    bool any = false;
    for (const std::string& s : items) if (slots_of(s.size()) > 1) any = true;
    std::vector<std::string> out;
    if (!any) return out;
    for (size_t i = 0; i < items.size(); ++i)
        out.push_back(std::to_string(i) + ":" + std::to_string(slots_of(items[i].size())));
    std::sort(out.begin(), out.end());
    return out;
}

static void check_spans(hid_t file, const std::vector<std::string>& want, const char* when) {
    std::vector<std::string> rows;
    const std::vector<std::string> got = read_spans(file, rows);
    if (got != want) {
        std::cout << "  FAIL: the span table " << when << " holds";
        for (const std::string& s : got) std::cout << " " << s;
        std::cout << " (rows";
        for (const std::string& r : rows) std::cout << " " << r;
        std::cout << "), expected";
        for (const std::string& s : want) std::cout << " " << s;
        std::cout << "\n";
        ++failures;
    }
}

// Reads a slice back through a fresh READ engine on the file root (init() descends into the
// group holding /index).
static void check_from_file(hid_t file, int slice, const std::vector<std::string>& want) {
    PanzerDB db(file, PanzerDB::OpenMode::READ, true, false);
    check_list(db, slice, want);
    db.close();
}

// The three lives of one file. A failure to lay a table down is reported by throwing, which
// main turns into a failure rather than a terminate.
static void run(const std::string& file) {
    std::cout << "=== TEST: a WRITE engine does not keep the previous span table ===\n";

    // The two lives differ in which element overflows: same number of span rows, different
    // boundaries, so only the content can tell them apart.
    const std::vector<std::string> first  = {"hello", long_string(749, 'a'), "world"};
    const std::vector<std::string> second = {long_string(900, 'b'), "tiny", long_string(600, 'c')};

    // ---- A: first life ----
    {
        const hid_t f = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (f < 0) throw std::runtime_error("H5Fcreate failed");
        const hid_t g = H5Gcreate2(f, GROUP, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (g < 0) { H5Fclose(f); throw std::runtime_error("H5Gcreate2 failed"); }
        write_list(g, PanzerDB::OpenMode::WRITE, first);
        H5Gclose(g);
        H5Fclose(f);
    }
    {
        const hid_t f = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (f < 0) throw std::runtime_error("the pulse file is gone after the first WRITE");
        check_from_file(f, 0, first);
        check_spans(f, expected_spans(first), "after the first WRITE");
        H5Fclose(f);
    }

    // ---- B: a second WRITE on the same file, with a different list ----
    {
        const hid_t f = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (f < 0) throw std::runtime_error("H5Fopen failed");
        const hid_t g = H5Gopen2(f, GROUP, H5P_DEFAULT);
        if (g < 0) { H5Fclose(f); throw std::runtime_error("H5Gopen2 failed"); }
        write_list(g, PanzerDB::OpenMode::WRITE, second);
        H5Gclose(g);
        H5Fclose(f);
    }
    {
        const hid_t f = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        // A's rows describe elements that no longer exist, and row 0 is now B's slice: reading
        // must give B, not A's boundaries applied to B's slots.
        check_from_file(f, 0, second);
        check_spans(f, expected_spans(second), "after the rewrite");
        H5Fclose(f);
    }

    // ---- C: appended into a file that never had a span table ----
    // Removing the table by hand is how a file written before it existed is simulated; it also
    // leaves slice 0 unreadable on purpose, which is why only slice 1 is read afterwards.
    {
        const hid_t f = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (H5Ldelete(f, SPANS.c_str(), H5P_DEFAULT) < 0) {
            std::cout << "  FAIL: could not remove " << SPANS << " to build the legacy case\n";
            ++failures;
        }
        H5Fclose(f);
    }
    {
        const std::vector<std::string> third = {long_string(1100, 'd'), "x", "y"};
        const hid_t f = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (f < 0) throw std::runtime_error("H5Fopen failed");
        write_list(f, PanzerDB::OpenMode::APPEND, third);
        H5Fclose(f);
        const hid_t r = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        check_from_file(r, 1, third);
        check_spans(r, expected_spans(third), "after appending into a file without one");
        H5Fclose(r);
    }
}

int main() {
    const std::string file = "test_engine_list_spans_over_write.panzer";
    try {
        run(file);
    } catch (const std::exception& e) {
        // A backend that cannot lay a table down must be reported, not left to terminate the
        // process from main.
        std::cout << "  FAIL: " << e.what() << "\n";
        ++failures;
    }
    if (fs::exists(file)) fs::remove(file);
    if (failures) {
        std::cout << failures << " failure(s).\n";
        return 1;
    }
    std::cout << "[OK] span table follows the life of the file.\n";
    return 0;
}
