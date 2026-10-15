// Verifies the user-friendly VDS view publisher (view_publisher.{h,cpp}).
//
// Fixture (one PanzerDB snapshot):
//   * A[2] x B[2] scalar grid            -> aggregated to A/B/val, dims [2,2]
//   * C[1] holding a 5-point vector      -> aggregated to C/vec5,   dims [5]   (C==1 squeezed)
//   * `series` written once per slice    -> folded into a leading time dim
//   * equilibrium[1]/time_slice[2]/r with slices of length 3 and 5
//                                        -> dims [2,5], the short slice padded
//                                          with the IMAS invalid value
//
// Collapse mode (the default) is checked first, then --expand restores the
// historical per-instance view and is checked against its own expectations.
// Both modes are checked for idempotency, and the original PanzerDB read path
// must be unaffected (no regression).
//
// NOTE: uses a hard `check()` (not assert()) so failures are honoured even in
//       a Release/NDEBUG build.
#include "panzerdb.h"
#include "view_publisher.h"
#include <hdf5.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

static constexpr double INVALID_F64 = -9.0e40;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "\nCHECK FAILED: %s\n", msg);
        std::fflush(stderr);
        std::exit(1);
    }
}

static void rm(const std::string& p) { if (fs::exists(p)) fs::remove(p); }

static hid_t openFile(const std::string& f) {
    hid_t fid = H5Fopen(f.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) throw std::runtime_error("cannot open " + f);
    return fid;
}

// H5Lexists on a path whose parent is absent prints a full HDF5 error trace, so
// the absence probes below mute the stack (they assert on the return value only).
struct QuietHdf5Errors {
    H5E_auto2_t oldFunc = nullptr;
    void* oldData = nullptr;
    QuietHdf5Errors()  { H5Eget_auto2(H5E_DEFAULT, &oldFunc, &oldData); H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); }
    ~QuietHdf5Errors() { H5Eset_auto2(H5E_DEFAULT, oldFunc, oldData); }
};

static bool exists(hid_t fid, const char* path) {
    QuietHdf5Errors q;
    return H5Lexists(fid, path, H5P_DEFAULT) > 0;
}

// rank + dimensions of a dataset; -1 when it does not exist
static int dimsOf(hid_t fid, const char* path, hsize_t* dims, int maxDims) {
    if (!exists(fid, path)) return -1;
    hid_t dset = H5Dopen2(fid, path, H5P_DEFAULT);
    hid_t sp   = H5Dget_space(dset);
    int rank   = H5Sget_simple_extent_ndims(sp);
    if (rank < 0) rank = 0;
    if (rank > 0) H5Sget_simple_extent_dims(sp, dims, nullptr);
    H5Sclose(sp);
    H5Dclose(dset);
    return (rank > maxDims) ? maxDims : rank;
}

static const char* readStrAttr(hid_t obj, const char* name, char* buf, size_t maxlen) {
    if (H5Aexists(obj, name) <= 0) return nullptr;
    buf[0] = 0;
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    hid_t t = H5Aget_type(a);
    hsize_t sz = (hsize_t)H5Tget_size(t);
    if (sz == 0 || sz > maxlen - 1) sz = maxlen - 1;
    H5Aread(a, t, buf);
    buf[sz] = 0;
    H5Tclose(t);
    H5Aclose(a);
    return buf;
}

// Reads a string attribute of a dataset, closing the dataset again.
static const char* attrOf(hid_t fid, const char* path, const char* name, char* buf, size_t maxlen) {
    hid_t dset = H5Dopen2(fid, path, H5P_DEFAULT);
    if (dset < 0) return nullptr;
    const char* r = readStrAttr(dset, name, buf, maxlen);
    H5Dclose(dset);
    return r;
}

static bool hasAttr(hid_t fid, const char* path, const char* name) {
    hid_t obj = H5Oopen(fid, path, H5P_DEFAULT);
    if (obj < 0) return false;
    const bool ok = (H5Aexists(obj, name) > 0);
    H5Oclose(obj);
    return ok;
}

static double readDoubleAttr(hid_t fid, const char* path, const char* name) {
    hid_t obj = H5Oopen(fid, path, H5P_DEFAULT);
    if (obj < 0) throw std::runtime_error(std::string("cannot open for attr: ") + path);
    double v = 0.0;
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    H5Aread(a, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(a);
    H5Oclose(obj);
    return v;
}

static void readDoubles(hid_t fid, const char* path, double* out, size_t n) {
    hid_t dset = H5Dopen2(fid, path, H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error(std::string("cannot open dataset: ") + path);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out);
    H5Dclose(dset);
}

// Reads a fixed-width text dataset into NUL-terminated strings.
static void readStrings(hid_t fid, const char* path, std::vector<std::string>& out) {
    hid_t dset = H5Dopen2(fid, path, H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error(std::string("cannot open dataset: ") + path);
    hid_t sp   = H5Dget_space(dset);
    const int rank = H5Sget_simple_extent_ndims(sp);
    hsize_t d[4] = {0};
    if (rank > 0) H5Sget_simple_extent_dims(sp, d, nullptr);
    const size_t n = (rank == 0) ? 1 : (size_t)d[0] * (rank > 1 ? (size_t)d[1] : 1);
    hid_t type = H5Dget_type(dset);
    const size_t w = H5Tget_size(type);
    std::vector<char> buf(n * w, 0);
    H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    for (size_t i = 0; i < n; ++i) out.emplace_back(buf.data() + i * w);
    H5Tclose(type);
    H5Sclose(sp);
    H5Dclose(dset);
}

// Top-level link names, optionally setting one of them aside (the view group).
static std::vector<std::string> rootLinks(hid_t fid, const char* ignore) {
    std::vector<std::string> out;
    H5G_info_t gi;
    if (H5Gget_info(fid, &gi) < 0) return out;
    for (hsize_t i = 0; i < gi.nlinks; ++i) {
        char name[512];
        if (H5Lget_name_by_idx(fid, ".", H5_INDEX_NAME, H5_ITER_INC, i, name, sizeof(name), H5P_DEFAULT) < 0)
            continue;
        if (ignore && std::strcmp(name, ignore) == 0) continue;
        out.emplace_back(name);
    }
    return out;
}

// Prints the published tree (name, kind, shape, key attributes) so that a
// failure below is readable straight from the test log.
static void dumpNode(hid_t fid, const char* name) {
    H5O_info2_t oinfo;
    if (H5Oget_info_by_name3(fid, name, &oinfo, H5O_INFO_BASIC, H5P_DEFAULT) < 0) return;
    std::string line = "    " + std::string(name);
    if (oinfo.type == H5O_TYPE_DATASET) {
        hid_t dset = H5Dopen2(fid, name, H5P_DEFAULT);
        hid_t sp   = H5Dget_space(dset);
        const int rank = H5Sget_simple_extent_ndims(sp);
        std::string shape = "{}";
        if (rank > 0) {
            std::vector<hsize_t> d(rank);
            H5Sget_simple_extent_dims(sp, d.data(), nullptr);
            shape = "{ ";
            for (int i = 0; i < rank; ++i) {
                if (i) shape += ", ";
                shape += std::to_string((long long)d[i]);
            }
            shape += " }";
        }
        H5Sclose(sp);
        char buf[128] = {0};
        std::string attrs;
        if (readStrAttr(dset, "dims_label", buf, sizeof(buf))) attrs += " @dims_label=" + std::string(buf);
        if (readStrAttr(dset, "schema", buf, sizeof(buf)))     attrs += " @schema=" + std::string(buf);
        line += "  Dataset " + shape;
        if (hasAttr(fid, name, "time_dim")) {
            char n[32]; snprintf(n, sizeof(n), "%g", readDoubleAttr(fid, name, "time_dim"));
            attrs += " @time_dim=" + std::string(n);
        }
        if (hasAttr(fid, name, "ragged"))     attrs += " @ragged";
        if (hasAttr(fid, name, "mapping_cap")) attrs += " @mapping_cap";
        line += attrs;
        H5Dclose(dset);
    } else if (oinfo.type == H5O_TYPE_GROUP) {
        line += "  Group";
        if (hasAttr(fid, name, "aos_size")) {
            char n[32]; snprintf(n, sizeof(n), "%g", readDoubleAttr(fid, name, "aos_size"));
            line += " @aos_size=" + std::string(n);
        }
        if (hasAttr(fid, name, "aos_dynamic")) line += " @aos_dynamic";
    }
    std::cout << line << '\n';
}

static void dumpTree(hid_t fid, const char* root) {
    std::cout << "  --- published view: /" << root << " ---\n";
    // Recursive listing, depth-first, so the printed tree mirrors h5ls -r.
    std::vector<std::string> stack{root};
    while (!stack.empty()) {
        const std::string dir = stack.back();
        stack.pop_back();
        hid_t g = H5Gopen2(fid, dir.c_str(), H5P_DEFAULT);
        if (g < 0) continue;
        H5G_info_t gi;
        H5Gget_info(g, &gi);
        std::vector<std::string> kids;
        for (hsize_t i = 0; i < gi.nlinks; ++i) {
            char name[512];
            if (H5Lget_name_by_idx(g, ".", H5_INDEX_NAME, H5_ITER_INC, i, name, sizeof(name), H5P_DEFAULT) < 0)
                continue;
            kids.push_back(name);
        }
        H5Gclose(g);
        for (const auto& k : kids) {
            std::string full = dir + "/" + k;
            dumpNode(fid, full.c_str());
            hid_t o = H5Oopen(fid, full.c_str(), H5P_DEFAULT);
            if (o >= 0) {
                H5O_info2_t oi;
                if (H5Oget_info3(o, &oi, H5O_INFO_BASIC) >= 0 && oi.type == H5O_TYPE_GROUP) stack.push_back(full);
                H5Oclose(o);
            }
        }
    }
    std::cout << "  --- end ---\n";
}

// Finds the listing line describing <path> (listView pads the path with spaces,
// so the character after it must be a space).
static bool findLine(const std::vector<std::string>& lines, const std::string& path, std::string& out) {
    for (const auto& l : lines) {
        if (l.size() > path.size() + 1 && l.compare(0, path.size(), path) == 0
            && l[path.size()] == ' ') { out = l; return true; }
    }
    return false;
}

int main() {
    const std::string filename = "test_publish_view.h5";
    const char* V = "imas_view";
    double g[2][2] = {{11.0, 12.0}, {21.0, 22.0}};   // A[a][b]
    double vec5[5] = {101, 102, 103, 104, 105};
    double series[4] = {100.0, 101.5, 102.5, 103.5};
    double bulk3[3]  = {5.5, 6.5, 7.5};
    double r3[3] = {1, 2, 3};
    double r5[5] = {4, 5, 6, 7, 8};

    // =====================================================================
    // 1. Write the PanzerDB snapshot
    // =====================================================================
    rm(filename);
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, /*preserve_empty*/true);
        // A[2] x B[2] scalar grid, explicit instance navigation
        db.beginArray("A", (size_t)2);        // A[0]
        db.beginArray("B", (size_t)2);        // A[0]/B[0]
        db.writeData("val", std::vector<size_t>{}, &g[0][0], 1);   // A/0/B/0/val
        db.incrementArrayIndex();             // A/0/B[1]
        db.writeData("val", std::vector<size_t>{}, &g[0][1], 1);   // A/0/B/1/val
        db.endArray();                        // pop B -> A[0]
        db.incrementArrayIndex();             // A[1]
        db.beginArray("B", (size_t)2);        // A[1]/B[0]
        db.writeData("val", std::vector<size_t>{}, &g[1][0], 1);   // A/1/B/0/val
        db.incrementArrayIndex();             // A/1/B[1]
        db.writeData("val", std::vector<size_t>{}, &g[1][1], 1);   // A/1/B/1/val
        db.endArray();                        // pop B
        db.endArray();                        // pop A

        // C[1] holding a 5-point vector
        db.beginArray("C", (size_t)1);
        db.writeData("vec5", std::vector<size_t>{5}, vec5, 5);     // C/0/vec5
        db.endArray();

        // One logical node stored as one row per time slice (the legacy per-slice
        // layout: identical path, row order == time) -> must become a time dim.
        for (int t = 0; t < 4; ++t)
            db.writeData("series", std::vector<size_t>{}, &series[t], 1);

        // The opposite layout: ONE row covering several time steps (ndim 0, count
        // == n_slices), which is how core_profiles stores its own `time` leaf.
        db.writeDataSlices("bulk", std::vector<size_t>{}, bulk3, 3, "time");   // -> bulk dims [3]

        // Ragged slices of the same logical node: length 3 then 5.
        db.beginArray("equilibrium", (size_t)1);
        db.beginArray("time_slice", (size_t)2);
        db.writeData("r", std::vector<size_t>{3}, r3, 3);          // .../time_slice/0/r
        { const char* l0[1] = {"alpha"};                            // same node, 1 text
          db.writeDataSlices("label", std::vector<size_t>{1}, l0, 1, ""); }
        db.incrementArrayIndex();
        db.writeData("r", std::vector<size_t>{5}, r5, 5);          // .../time_slice/1/r
        { const char* l1[2] = {"beta", "gamma"};                    // same node, 2 texts
          db.writeDataSlices("label", std::vector<size_t>{2}, l1, 1, ""); }
        db.endArray();
        db.endArray();

        db.close();
    }
    std::cout << "[write] OK\n";

    // Snapshot of the raw top-level layout, to prove publishing is purely additive.
    std::vector<std::string> beforePublish;
    {
        hid_t fid = openFile(filename);
        beforePublish = rootLinks(fid, nullptr);
        H5Fclose(fid);
    }
    check(!beforePublish.empty(), "raw file has top-level objects");
    // A file whose view was never published is a normal state, not an error.
    {
        const auto none = imas::view::listView(filename, V);
        check(!none.found && none.lines.empty(), "listView reports no view before publishing");
    }

    // =====================================================================
    // 2. Collapse mode (the default): one dataset per logical node
    // =====================================================================
    imas::view::PublishStats st;
    try {
        st = imas::view::publish(filename, V);
    } catch (const std::exception& e) {
        check(false, e.what());
    }
    std::cout << "[collapse] datasets=" << st.n_aggregated_datasets
              << " aos_groups=" << st.n_aos_groups
              << " mappings=" << st.n_mappings
              << " holes=" << st.n_fill_holes
              << " errors=" << st.n_errors << '\n';
    check(st.n_errors == 0, "collapse publish produced 0 errors");
    check(st.n_aggregated_datasets == 6, "6 aggregated datasets (A/B/val, C/vec5, series, bulk, equilibrium/time_slice/{r,label})");

    {
        hid_t fid = openFile(filename);
        dumpTree(fid, V);
        hsize_t d[8];

        // 2a. A[2]xB[2] scalars -> one dataset, dims [2,2], row-major A then B
        check(dimsOf(fid, "imas_view/A/B/val", d, 8) == 2 && d[0] == 2 && d[1] == 2,
              "A/B/val aggregated to dims [2,2]");
        check(!exists(fid, "imas_view/A/0/B/0/val"), "no per-instance path left in collapse mode");
        double v4[4] = {0};
        readDoubles(fid, "imas_view/A/B/val", v4, 4);
        check(v4[0] == 11 && v4[1] == 12 && v4[2] == 21 && v4[3] == 22, "A/B/val values match [A][B]");
        char buf[128];
        check(attrOf(fid, "imas_view/A/B/val", "dims_label", buf, sizeof(buf))
                  && std::strcmp(buf, "A|B") == 0, "@dims_label == A|B (scalar leaf adds no shape dim)");
        check(attrOf(fid, "imas_view/A/B/val", "schema", buf, sizeof(buf))
                  && std::strcmp(buf, "A/B/val") == 0, "@schema == A/B/val");
        check(attrOf(fid, "imas_view/A/B/val", "datatype", buf, sizeof(buf))
                  && std::strcmp(buf, "float64") == 0, "A/B/val @datatype == float64");
        check(readDoubleAttr(fid, "imas_view/A/B/val", "aos_ndim") == 2.0, "@aos_ndim == 2");
        check(readDoubleAttr(fid, "imas_view/A/B/val", "n_elements") == 4.0, "@n_elements == 4");
        // One mapping per A instance: within an instance the 4 scalars are stored
        // contiguously, so B coalesces into a single strided mapping per A.
        check(readDoubleAttr(fid, "imas_view/A/B/val", "n_mappings") == 2.0, "A/B/val uses 2 mappings (one per A)");

        // 2b. C[1] singleton level squeezed out: dims [5], not [1,5]
        check(dimsOf(fid, "imas_view/C/vec5", d, 8) == 1 && d[0] == 5, "C/vec5 dims [5] (C==1 squeezed)");
        double v5[5] = {0};
        readDoubles(fid, "imas_view/C/vec5", v5, 5);
        for (int i = 0; i < 5; ++i) check(v5[i] == vec5[i], "C/vec5 values match");
        check(readDoubleAttr(fid, "imas_view/C/vec5", "aos_ndim") == 0.0, "C/vec5 has no leading dim");
        check(attrOf(fid, "imas_view/C/vec5", "dims_label", buf, sizeof(buf))
                  && std::strcmp(buf, "vec5") == 0, "C/vec5 @dims_label == vec5");

        // 2c. Repeated identical path -> leading time dimension
        check(dimsOf(fid, "imas_view/series", d, 8) == 1 && d[0] == 4, "series folded to dims [4]");
        double sv[4] = {0};
        readDoubles(fid, "imas_view/series", sv, 4);
        for (int t = 0; t < 4; ++t) check(sv[t] == series[t], "series value matches row order");
        check(readDoubleAttr(fid, "imas_view/series", "time_dim") == 4.0, "series @time_dim == 4");
        check(attrOf(fid, "imas_view/series", "dims_label", buf, sizeof(buf))
                  && std::strcmp(buf, "time") == 0, "series @dims_label == time (scalar leaf adds no shape dim)");

        // 2c-bis. One row covering several time steps: the cell must fill every
        // time position it spans (a mapping whose target counted only run.len
        // elements silently produced an all-fill dataset).
        check(dimsOf(fid, "imas_view/bulk", d, 8) == 1 && d[0] == 3, "bulk row folded to dims [3]");
        double bv[3] = {0};
        readDoubles(fid, "imas_view/bulk", bv, 3);
        for (int t = 0; t < 3; ++t) check(bv[t] == bulk3[t], "bulk row value matches its slice");
        check(readDoubleAttr(fid, "imas_view/bulk", "n_mappings") == 1.0, "bulk row needs a single mapping");
        check(readDoubleAttr(fid, "imas_view/bulk", "time_dim") == 3.0, "bulk @time_dim == 3");

        // 2d. Ragged slices: [2,5], the short slice padded with the invalid value
        check(dimsOf(fid, "imas_view/equilibrium/time_slice/r", d, 8) == 2 && d[0] == 2 && d[1] == 5,
              "ragged node aggregated to dims [2,5]");
        double rv[10] = {0};
        readDoubles(fid, "imas_view/equilibrium/time_slice/r", rv, 10);
        check(rv[0] == 1 && rv[1] == 2 && rv[2] == 3, "ragged slice 0 values");
        check(rv[3] == INVALID_F64 && rv[4] == INVALID_F64, "ragged hole holds the invalid value");
        for (int i = 0; i < 5; ++i) check(rv[5 + i] == r5[i], "ragged slice 1 values");
        check(hasAttr(fid, "imas_view/equilibrium/time_slice/r", "ragged"), "ragged node carries @ragged");
        check(attrOf(fid, "imas_view/equilibrium/time_slice/r", "dims_label", buf, sizeof(buf))
                  && std::strcmp(buf, "time_slice|r") == 0, "ragged @dims_label == time_slice|r");

        // 2d-bis. Same ragged pattern in text form: the hole must read as the
        // empty string (the decided fill for text).
        check(dimsOf(fid, "imas_view/equilibrium/time_slice/label", d, 8) == 2 && d[0] == 2 && d[1] == 2,
              "ragged text node aggregated to dims [2,2]");
        std::vector<std::string> labels;
        readStrings(fid, "imas_view/equilibrium/time_slice/label", labels);
        check(labels.size() == 4, "ragged text node has 2x2 cells");
        check(labels[0] == "alpha" && labels[2] == "beta" && labels[3] == "gamma",
              "ragged text values match their instance");
        check(labels[1].empty(), "text hole reads as the empty string");
        check(attrOf(fid, "imas_view/equilibrium/time_slice/label", "datatype", buf, sizeof(buf))
                  && std::strcmp(buf, "string") == 0, "text node @datatype == string");

        // 2e. @aos_size on the AoS groups (singleton `equilibrium` keeps its size)
        check(readDoubleAttr(fid, "imas_view/A", "aos_size") == 2.0, "@aos_size(A) == 2");
        check(readDoubleAttr(fid, "imas_view/A/B", "aos_size") == 2.0, "@aos_size(A/B) == 2");
        check(readDoubleAttr(fid, "imas_view/C", "aos_size") == 1.0, "@aos_size(C) == 1");
        check(readDoubleAttr(fid, "imas_view/equilibrium/time_slice", "aos_size") == 2.0,
              "@aos_size(equilibrium/time_slice) == 2");

        // 2f. The raw PanzerDB tables are untouched (the view is additive): the
        // top-level object list is identical once the view group is set aside.
        check(rootLinks(fid, V) == beforePublish, "publishing only adds the view group");

        // 2g. listView: the annotations h5ls -r cannot show, one line per object.
        {
            const auto lv = imas::view::listView(filename, V);
            check(lv.found, "listView finds the published view");
            check(lv.lines.size() == 11, "listView lists 5 AoS groups + 6 aggregated datasets");
            for (const auto& l : lv.lines) std::cout << "    " << l << '\n';

            std::string line;
            check(findLine(lv.lines, "/imas_view/A/B/val", line)
                      && line.find("Dataset {2, 2}") != std::string::npos
                      && line.find("@dims_label=A|B") != std::string::npos
                      && line.find("holes") == std::string::npos,
                  "A/B/val listed with dims, dim labels, and no hole");
            check(findLine(lv.lines, "/imas_view/equilibrium/time_slice/r", line)
                      && line.find("Dataset {2, 5}") != std::string::npos
                      && line.find("@ragged") != std::string::npos
                      && line.find("holes=2") != std::string::npos,
                  "ragged node listed with @ragged and holes=2 (10 cells, 8 stored)");
            check(findLine(lv.lines, "/imas_view/equilibrium/time_slice/label", line)
                      && line.find("holes=1") != std::string::npos,
                  "ragged text node lists holes=1 (4 cells, 3 stored)");
            check(findLine(lv.lines, "/imas_view/C/vec5", line)
                      && line.find("Dataset {5}") != std::string::npos,
                  "C/vec5 listed squeezed to dims {5}");
            check(findLine(lv.lines, "/imas_view/series", line)
                      && line.find("Dataset {4}") != std::string::npos
                      && line.find("@dims_label=time") != std::string::npos,
                  "folded time series listed with its time dim label");
            check(findLine(lv.lines, "/imas_view/A", line)
                      && line.find("Group") != std::string::npos
                      && line.find("@aos_size=2") != std::string::npos,
                  "AoS group A listed with @aos_size=2");

            // Read-only proof: publish needs write access, the listing must not.
            std::error_code pec;
            fs::permissions(filename, fs::perms::owner_read, fs::perm_options::replace, pec);
            check(!pec, "could not make the snapshot read-only");
            const auto ro = imas::view::listView(filename, V);
            fs::permissions(filename, fs::perms::owner_all, fs::perm_options::add, pec);
            check(!pec, "could not restore write permission");
            check(ro.found && ro.lines.size() == lv.lines.size(),
                  "listView works on a file with no write permission");
        }
        H5Fclose(fid);
    }
    std::cout << "[collapse] OK\n";

    // Idempotency in collapse mode
    auto st2 = imas::view::publish(filename, V);
    check(st2.n_errors == 0, "collapse re-publish produced 0 errors");
    check(st2.n_aggregated_datasets == st.n_aggregated_datasets, "re-publish is stable");
    std::cout << "[idempotency/collapse] OK\n";

    // =====================================================================
    // 2h. keep_singletons: singleton AoS levels stay as leading dims, so the
    //     size of every level is visible in the shape h5ls prints (it shows no
    //     attribute, hence this option).
    // =====================================================================
    {
        imas::view::PublishOptions o;
        o.root_group      = V;
        o.keep_singletons = true;
        const auto stk = imas::view::publish(filename, o);
        check(stk.n_errors == 0, "keep_singletons publish produced 0 errors");
        check(stk.n_aggregated_datasets == 6, "keep_singletons publishes the same 6 datasets");

        hid_t fid = openFile(filename);
        hsize_t d[8];
        char buf[128];

        // C==1 kept: dims [1,5] instead of the squeezed [5]
        check(dimsOf(fid, "imas_view/C/vec5", d, 8) == 2 && d[0] == 1 && d[1] == 5,
              "keep_singletons: C/vec5 dims [1,5]");
        check(attrOf(fid, "imas_view/C/vec5", "dims_label", buf, sizeof(buf))
                  && std::strcmp(buf, "C|vec5") == 0, "keep_singletons: C/vec5 @dims_label == C|vec5");
        check(readDoubleAttr(fid, "imas_view/C/vec5", "aos_ndim") == 1.0,
              "keep_singletons: C/vec5 @aos_ndim == 1");
        double v5[5] = {0};
        readDoubles(fid, "imas_view/C/vec5", v5, 5);
        for (int i = 0; i < 5; ++i) check(v5[i] == vec5[i], "keep_singletons: C/vec5 values still match");

        // equilibrium==1 kept in front of the ragged dims: [1,2,5] not [2,5]
        check(dimsOf(fid, "imas_view/equilibrium/time_slice/r", d, 8) == 3
                  && d[0] == 1 && d[1] == 2 && d[2] == 5,
              "keep_singletons: ragged node dims [1,2,5]");
        double rv[10] = {0};
        readDoubles(fid, "imas_view/equilibrium/time_slice/r", rv, 10);
        check(rv[0] == 1 && rv[1] == 2 && rv[2] == 3, "keep_singletons: ragged slice 0 values");
        check(rv[3] == INVALID_F64 && rv[4] == INVALID_F64, "keep_singletons: ragged hole still invalid");
        for (int i = 0; i < 5; ++i) check(rv[5 + i] == r5[i], "keep_singletons: ragged slice 1 values");

        // Levels whose size is >1 are unaffected by the option
        check(dimsOf(fid, "imas_view/A/B/val", d, 8) == 2 && d[0] == 2 && d[1] == 2,
              "keep_singletons: A/B/val dims unchanged [2,2]");

        // The annotated listing reflects the un-squeezed dims
        {
            const auto lv = imas::view::listView(filename, V);
            std::string line;
            check(findLine(lv.lines, "/imas_view/C/vec5", line)
                      && line.find("Dataset {1, 5}") != std::string::npos,
                  "keep_singletons: C/vec5 listed with dims {1, 5}");
            check(findLine(lv.lines, "/imas_view/equilibrium/time_slice/r", line)
                      && line.find("Dataset {1, 2, 5}") != std::string::npos,
                  "keep_singletons: ragged node listed with dims {1, 2, 5}");
        }
        H5Fclose(fid);

        // Reverse idempotency: a plain re-publish squeezes again
        const auto stback = imas::view::publish(filename, V);
        check(stback.n_errors == 0, "default re-publish after keep_singletons: 0 errors");
        hid_t fid2 = openFile(filename);
        check(dimsOf(fid2, "imas_view/C/vec5", d, 8) == 1 && d[0] == 5,
              "default re-publish squeezes C/vec5 back to [5]");
        H5Fclose(fid2);
        std::cout << "[keep_singletons] OK\n";
    }

    // =====================================================================
    // 3. Expand mode: the historical per-instance view
    // =====================================================================
    {
        imas::view::PublishOptions o;
        o.root_group   = V;
        o.collapse_aos = false;
        const auto ste = imas::view::publish(filename, o);
        // The historical view maps one index row per HDF5 object, so the 4 rows
        // sharing the path `series` collide: 3 of them are lost. That is exactly
        // the defect the collapse mode fixes (equilibrium_1.h5 loses 12 998 rows
        // this way), so it is asserted rather than hidden.
        check(ste.n_data_leaves == 11, "expand: 14 rows -> 11 VDS (the 4 duplicate `series` paths collapse to 1)");
        check(ste.n_errors == 3, "expand: 3 duplicate-path rows cannot be published");

        hid_t fid = openFile(filename);
        hsize_t d[8];
        check(dimsOf(fid, "imas_view/A/0/B/0/val", d, 8) == 0, "expand: A/0/B/0/val is a scalar");
        double v = 0;
        readDoubles(fid, "imas_view/A/0/B/0/val", &v, 1);
        check(v == 11.0, "expand: A/0/B/0/val == 11");
        check(dimsOf(fid, "imas_view/A/1/B/1/val", d, 8) == 0, "expand: A/1/B/1/val exists");
        readDoubles(fid, "imas_view/A/1/B/1/val", &v, 1);
        check(v == 22.0, "expand: A/1/B/1/val == 22");
        check(dimsOf(fid, "imas_view/C/0/vec5", d, 8) == 1 && d[0] == 5, "expand: C/0/vec5 dims [5]");
        check(dimsOf(fid, "imas_view/equilibrium/0/time_slice/0/r", d, 8) == 1 && d[0] == 3,
              "expand: per-instance ragged slice keeps its own length");
        check(readDoubleAttr(fid, "imas_view/A", "aos_size") == 2.0, "expand: @aos_size(A) == 2");
        check(!exists(fid, "imas_view/A/B/val"), "expand: no aggregated dataset");
        H5Fclose(fid);
        std::cout << "[expand] OK (" << ste.n_data_leaves << " instance VDS, "
                  << ste.n_errors << " duplicate-path rows dropped)\n";
    }

    // Back to the default view, and check idempotency across a mode switch.
    {
        const auto st3 = imas::view::publish(filename, V);
        check(st3.n_errors == 0 && st3.n_aggregated_datasets == 6, "mode switch rebuilds cleanly");
        hid_t fid = openFile(filename);
        check(exists(fid, "imas_view/A/B/val") && !exists(fid, "imas_view/A/0/B/0/val"),
              "collapse view replaces the expanded one");
        H5Fclose(fid);
    }

    // =====================================================================
    // 4. No regression: original PanzerDB read path still works
    // =====================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        uint64_t ndim = 0; uint64_t shape[6] = {0}; double* out = nullptr;
        check(db.readDataByIndex("A/0/B/0/val", -1, &ndim, shape, &out) == 0, "readDataByIndex A/0/B/0/val");
        check(out && out[0] == g[0][0], "readDataByIndex value A/0/B/0/val");
        delete[] out; out = nullptr;
        check(db.readDataByIndex("C/0/vec5", -1, &ndim, shape, &out) == 0, "readDataByIndex C/0/vec5");
        check(shape[0] == 5 && out[0] == 101.0, "readDataByIndex value C/0/vec5");
        delete[] out;
        db.close();
    }
    std::cout << "[no regression] OK\n";

    std::cout << "\nALL TESTS PASSED\n";
    rm(filename);
    return 0;
}
