// Verifies the user-friendly VDS view publisher (view_publisher.{h,cpp}).
//
// 1. Build a small PanzerDB snapshot: a 2×2 AoS scalar grid (A[a]/B[b]/val)
//    plus a single-instance AoS (C) holding a 5-point vector (C/0/vec5).
// 2. Publish the VDS view under the default "imas_view" group.
// 3. Open the file RAW (plain HDF5 C, no PanzerDB API) and verify:
//    - every instance leaf is a VDS carrying the correct value and @datatype,
//    - @aos_size is set on every AoS group (A==2, A/a/B==2),
//    - the vector leaf has shape [5] and the correct values,
//    - idempotency (re-publish rebuilds, still readable),
//    - the ORIGINAL PanzerDB read path is unaffected (no regression).
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
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

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

static void* readScalar(hid_t dset, double& out) {
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &out);
    return nullptr;
}

static int dims(hid_t dset, hsize_t& d0) {
    hid_t sp = H5Dget_space(dset);
    int rank = H5Sget_simple_extent_ndims(sp);
    if (rank > 0) { hsize_t dim[1]; H5Sget_simple_extent_dims(sp, dim, nullptr); d0 = dim[0]; }
    H5Sclose(sp);
    return rank;
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

static double readDoubleAttr(hid_t obj, const char* name) {
    double v = 0.0;
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    H5Aread(a, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(a);
    return v;
}

int main() {
    const std::string filename = "test_publish_view.h5";
    const char* V = "imas_view";
    double g[2][2] = {{11.0, 12.0}, {21.0, 22.0}};   // A[a][b]
    double vec5[5] = {101, 102, 103, 104, 105};

    // =====================================================================
    // 1. Write the PanzerDB snapshot
    // =====================================================================
    rm(filename);
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, /*preserve_empty*/true);
        // A[2] x B[2] scalar grid, explicit instance navigation
        db.beginArray("A", 2);        // A[0]
        db.beginArray("B", 2);        // A[0]/B[0]
        db.writeData("val", std::vector<size_t>{}, &g[0][0], 1);   // A/0/B/0/val
        db.incrementArrayIndex();    // A/0/B[1]
        db.writeData("val", std::vector<size_t>{}, &g[0][1], 1);   // A/0/B/1/val
        db.endArray();               // pop B -> A[0]
        db.incrementArrayIndex();    // A[1]
        db.beginArray("B", 2);       // A[1]/B[0]
        db.writeData("val", std::vector<size_t>{}, &g[1][0], 1);   // A/1/B/0/val
        db.incrementArrayIndex();    // A[1]/B[1]
        db.writeData("val", std::vector<size_t>{}, &g[1][1], 1);   // A/1/B/1/val
        db.endArray();               // pop B
        db.endArray();               // pop A

        // C[1] holding a 5-point vector
        db.beginArray("C", 1);
        db.writeData("vec5", std::vector<size_t>{5}, vec5, 5);     // C/0/vec5
        db.endArray();

        db.close();
    }
    std::cout << "[write] OK\n";

    // =====================================================================
    // 2. Publish the VDS view
    // =====================================================================
    imas::view::PublishStats st;
    try {
        st = imas::view::publish(filename, V);
    } catch (const std::exception& e) {
        check(false, e.what());
    }
    std::cout << "[publish] leaves=" << st.n_data_leaves
              << " aos_groups=" << st.n_aos_groups
              << " errors=" << st.n_errors << '\n';
    check(st.n_errors == 0, "publish produced 0 errors");
    check(st.n_data_leaves >= 5, "at least 5 data leaves (4 scalars + 1 vector)");

    // =====================================================================
    // 3. Read with plain HDF5 C (NO PanzerDB API) — the whole point
    // =====================================================================
    {
        hid_t fid = openFile(filename);

        // 3a. Each scalar instance VDS
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                char path[256];
                snprintf(path, sizeof(path), "%s/A/%d/B/%d/val", V, a, b);
                check(H5Lexists(fid, path, H5P_DEFAULT) > 0, "instance VDS exists");
                hid_t dset = H5Dopen2(fid, path, H5P_DEFAULT);
                char buf[64];
                check(readStrAttr(dset, "datatype", buf, sizeof(buf)) && std::strcmp(buf, "float64") == 0,
                      "@datatype == float64");
                double v = 0.0;
                readScalar(dset, v);
                check(v == g[a][b], "scalar value matches");
                std::cout << "  " << path << " = " << v << " OK\n";
                H5Dclose(dset);
            }

        // 3b. Vector VDS (shape [5], values)
        {
            char path[256];
            snprintf(path, sizeof(path), "%s/C/0/vec5", V);
            check(H5Lexists(fid, path, H5P_DEFAULT) > 0, "vector VDS exists");
            hid_t dset = H5Dopen2(fid, path, H5P_DEFAULT);
            hsize_t d0 = 0;
            int rank = dims(dset, d0);
            check(rank == 1 && d0 == 5, "vector shape == [5]");
            double v[5] = {0};
            H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v);
            for (int i = 0; i < 5; ++i) check(v[i] == vec5[i], "vector value matches");
            char buf[64];
            check(readStrAttr(dset, "datatype", buf, sizeof(buf)) && std::strcmp(buf, "float64") == 0,
                  "vector @datatype == float64");
            std::cout << "  " << path << " shape=[5] values OK\n";
            H5Dclose(dset);
        }

        // 3c. @aos_size on the AoS groups
        {
            check(H5Lexists(fid, "imas_view/A", H5P_DEFAULT) > 0, "A group exists");
            hid_t grpA = H5Gopen2(fid, "imas_view/A", H5P_DEFAULT);
            check(readDoubleAttr(grpA, "aos_size") == 2.0, "@aos_size(A) == 2");
            H5Gclose(grpA);

            check(H5Lexists(fid, "imas_view/A/0/B", H5P_DEFAULT) > 0, "A/0/B group exists");
            hid_t grpAB = H5Gopen2(fid, "imas_view/A/0/B", H5P_DEFAULT);
            check(readDoubleAttr(grpAB, "aos_size") == 2.0, "@aos_size(A/0/B) == 2");
            H5Gclose(grpAB);

            hid_t grpC = H5Gopen2(fid, "imas_view/C", H5P_DEFAULT);
            check(readDoubleAttr(grpC, "aos_size") == 1.0, "@aos_size(C) == 1");
            H5Gclose(grpC);
            std::cout << "  @aos_size OK (A==2, A/0/B==2, C==1)\n";
        }
        H5Fclose(fid);
    }

    // =====================================================================
    // 4. Idempotency: publish AGAIN, still readable
    // =====================================================================
    auto st2 = imas::view::publish(filename, V);
    check(st2.n_errors == 0, "re-publish produced 0 errors");
    {
        hid_t fid = openFile(filename);
        check(H5Lexists(fid, "imas_view/A/0/B/0/val", H5P_DEFAULT) > 0, "idempotent: A/0/B/0/val");
        check(H5Lexists(fid, "imas_view/C/0/vec5", H5P_DEFAULT) > 0, "idempotent: C/0/vec5");
        H5Fclose(fid);
    }
    std::cout << "[idempotency] OK\n";

    // =====================================================================
    // 5. No regression: original PanzerDB read path still works
    // =====================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        uint64_t ndim = 0; uint64_t shape[6] = {0}; double* out = nullptr;
        check(db.readDataByIndex("A/0/B/0/val", -1, &ndim, shape, &out) == 0, "readDataByIndex A/0/B/0/val");
        check(out && out[0] == g[0][0], "readDataByIndex value A/0/B/0/val");
        delete[] out;
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
