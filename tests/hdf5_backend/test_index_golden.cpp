// @file  test_index_golden.cpp
// @brief Oracle (golden) test for the PanzerDB index "surface" -- the public
//        contract that the compact-index refactor (index_plan/...) must keep
//        byte-identical on the same corpus.
//
// The "surface" is everything a consumer can read off the index WITHOUT the raw
// data values (those are covered by the value-asserting engine/*al_* tests):
//
//   * PanzerDB::getLeaves()  -> every Leaf: path, parent_path, flags,
//     time_index, offset, count, shape, is_empty
//   * imas::direct_access::list_nodes(...) -> NodeInfo (path, type, dims, is_in_dynamic_aos)
//   * PanzerDB::getAOSShape(aos_path)       -> size vector per AoS
//   * PanzerDB::stripIndices(path)          -> schema of each instance path
//
// If the surface is byte-identical before/after, ALL lookups (leaf_lookup,
// parent_lookup), stripIndices consumers, list_nodes and the CLI tools are
// transitively unchanged.
//
// Modes (argv[1], default "verify"):
//   verify  (default) : rebuild the corpus, compute the surface, and compare it
//                        to PZGOLD_GOLDEN_FILE; exit 0 iff equal (this is the
//                        CTest-registered mode).
//   capture          : rebuild the corpus, compute the surface, and WRITE
//                        PZGOLD_GOLDEN_FILE (one-time oracle generation).
//
// The corpus is fully deterministic, so both modes are standalone.

#include "panzerdb.h"
#include "direct_access_api.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <complex>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef PZGOLD_GOLDEN_DIR
  #define PZGOLD_GOLDEN_DIR "goldens"
#endif
#define PZGOLD_GOLDEN_FILE PZGOLD_GOLDEN_DIR "/index_surface.golden"

using imas::direct_access::list_nodes;
using imas::direct_access::NodeInfo;
using imas::direct_access::NodeType;

namespace {

// ----------------------------------------------------------------------------
// Deterministic corpus: exercises every index emitter + data type + a mix of
// static / dynamic / nested / empty AoS, strings, complex and int32.
//   emitters mapped: beginArray(meta), writeData<T>(static), writeDataSlices<T>(dynamic)
// ----------------------------------------------------------------------------
void write_corpus(const std::string& filename) {
    PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

    // ---- ROOT static scalars/tensors: every static dtype (writeData<T>) ----
    double rd = 98.6;
    db.writeData("root_d", {}, &rd, 1);
    int32_t ri = 42;
    db.writeData("root_i", {}, &ri, 1);
    std::complex<double> rc(1.0, -2.0);
    db.writeData("root_c", {}, &rc, 1);
    double r1d[4] = {1, 2, 3, 4};
    db.writeData("root_1d", {4}, r1d, 4);
    double r2d[6] = {6, 5, 4, 3, 2, 1};
    db.writeData("root_2d", {2, 3}, r2d, 6);
    int32_t rsi[3] = {7, 8, 9};
    db.writeData("root_i1d", {3}, rsi, 3);
    std::complex<double> rc1d[2] = {{1, 2}, {3, 4}};
    db.writeData("root_c1d", {2}, rc1d, 2);
    const char* rss[1] = {"rootstring"};
    db.writeData("root_str", {}, rss, 1);                 // const char* scalar
    const char* rlist[3] = {"a", "bb", "ccc"};
    db.writeData("root_liststr", {3}, rlist, 3);          // const char* 1D list
    char bytes[4] = {'h', 'i', '!', '0'};
    db.writeData("root_bytes", {4}, bytes, 4);            // char raw scalar bytes

    // ---- STATIC nested AoS (mirror test_engine_readdata_by_index_ultradeep) ----
    db.beginArray("sd", 1);                                // sd[0]
    db.beginArray("b", 2);                                 // b (declared 2)
    // b[0] -> c declared 1, empty
    db.beginArray("c", 1);
    db.endArray();                                         // c
    db.incrementArrayIndex();                              // -> b[1]
    // b[1] -> c declared 2
    db.beginArray("c", 2);
    // c[0] -> d -> e -> tensors
    db.beginArray("d", 1);
    db.beginArray("e", 1);
    { double t3[3] = {10, 11, 12}; db.writeData("t3", {3}, t3, 3); }
    { double t2[2] = {20, 21};    db.writeData("t2", {2}, t2, 2); }
    { int32_t ti[4] = {1, 2, 3, 4}; db.writeData("ti", {4}, ti, 4); }
    db.endArray();                                         // e
    db.endArray();                                         // d
    db.incrementArrayIndex();                              // -> c[1]
    // c[1] -> d -> e -> tensor
    db.beginArray("d", 1);
    db.beginArray("e", 1);
    { double t4[5] = {30, 31, 32, 33, 34}; db.writeData("t4", {5}, t4, 5); }
    db.endArray();                                         // e
    db.endArray();                                         // d
    db.incrementArrayIndex();                              // -> c[2]
    db.endArray();                                         // c
    db.incrementArrayIndex();                              // -> b[2]
    db.endArray();                                         // b
    db.incrementArrayIndex();                              // -> sd[1]
    db.endArray();                                         // sd

    // ---- DYNAMIC standalone AoS with inhomogeneous numeric/dtype signals ----
    db.beginArray("dyn", "time");
    { double s0[3] = {1, 2, 3};                 db.writeDataSlices("sig", {1}, s0, 3, ""); }
    { double s1[2]; s1[0]=5; s1[1]=6;           db.writeDataSlices("sig_more", {1}, s1, 2, ""); }
    { double s2d[2*4] = {1,2,3,4,5,6,7,8};       db.writeDataSlices("sig1d", {4}, s2d, 2, ""); }
    { int32_t si[4] = {4,5,6,7};                 db.writeDataSlices("sigi", {1}, si, 4, ""); }
    { std::complex<double> sc[3] = {{0,1},{2,3},{4,5}}; db.writeDataSlices("sigc", {1}, sc, 3, ""); }
    { const char* ss[3] = {"a", "bb", "ccc"};    db.writeDataSlices("sigstr", {1}, ss, 3, ""); }
    db.endArray();                                        // dyn

    // ---- DYNAMIC under STATIC (mirror test_engine_write_read_nested_dynamic_aos)
    //      plus a string + complex signal to hit the char*const*/complex emitters ----
    db.beginArray("na", 3);                                // na[0]
    db.beginArray("nb", "time");
    { double n0[3] = {100,101,102}; db.writeDataSlices("sigsig", {1}, n0, 3, ""); }
    { const char* nn0[2]={"x","yy"}; db.writeDataSlices("sigstrd", {1}, nn0, 2, ""); }
    db.endArray();                                         // nb
    db.incrementArrayIndex();                              // -> na[1]
    db.beginArray("nb", "time");
    { double n1[5] = {200,201,202,203,204}; db.writeDataSlices("sigsig", {1}, n1, 5, ""); }
    db.endArray();                                         // nb
    db.incrementArrayIndex();                              // -> na[2]
    db.beginArray("nb", "time");
    { double n2[2] = {300,301}; std::complex<double> nc[2]={{9,1},{8,2}};
      db.writeDataSlices("sigsig", {1}, n2, 2, ""); }
    { std::complex<double> n2c[2]={{9,1},{8,2}}; db.writeDataSlices("sigcplx", {1}, n2c, 2, ""); }
    db.endArray();                                         // nb
    db.incrementArrayIndex();                              // -> na[3]
    db.endArray();                                         // na

    db.close();
}

// ----------------------------------------------------------------------------
// Serialize the full preserved surface to a canonical, deterministic string.
// ----------------------------------------------------------------------------
std::string serialize_surface(const std::string& filename) {
    std::ostringstream os;

    PanzerDB db(filename, PanzerDB::OpenMode::READ);
    const std::vector<PanzerDB::Leaf>& leaves = db.getLeaves();

    // Stable order: path, then time_index, then parent_path.
    std::vector<const PanzerDB::Leaf*> all;
    all.reserve(leaves.size());
    for (const auto& lf : leaves) all.push_back(&lf);
    std::sort(all.begin(), all.end(), [](const PanzerDB::Leaf* a, const PanzerDB::Leaf* b) {
        if (a->path != b->path) return std::string(a->path) < std::string(b->path);
        if (a->time_index != b->time_index) return a->time_index < b->time_index;
        return std::string(a->parent_path) < std::string(b->parent_path);
    });

    auto dims_str = [](const std::vector<size_t>& v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) { if (i) s += ","; s += std::to_string(v[i]); }
        return s;
    };

    os << "# PANZERDB INDEX SURFACE GOLDEN v1\n";

    os << "[leaves " << all.size() << "]\n";
    for (const auto* lf : all) {
        const std::string path(lf->path);
        os << path << " | "
           << std::string(lf->parent_path) << " | "
           << "flags=" << lf->flags
           << " kind=" << (lf->flags & 0xFULL)
           << " dtype=" << (lf->flags >> 4)
           << " time=" << lf->time_index
           << " off=" << lf->offset
           << " cnt=" << lf->count
           << " shape=(" << dims_str(lf->shape) << ")"
           << " empty=" << (lf->is_empty ? 1 : 0)
           << " schema=" << PanzerDB::stripIndices(path)
           << "\n";
    }

    // ----- public: list_nodes (CLI imas_h5ls / imas_h5dump contract) -----
    auto [nodes, aos_map] = imas::direct_access::list_nodes(filename, true, true, true);
    std::vector<NodeInfo> sorted(nodes.begin(), nodes.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const NodeInfo& a, const NodeInfo& b) { return a.path < b.path; });
    auto nt_name = [](NodeType t) {
        switch (t) {
            case NodeType::DATASET:     return "DATASET";
            case NodeType::AOS_STATIC:  return "AOS_STATIC";
            case NodeType::AOS_DYNAMIC: return "AOS_DYNAMIC";
        }
        return "?";
    };

    os << "[list_nodes " << sorted.size() << "]\n";
    for (const auto& n : sorted) {
        os << n.path << " | " << nt_name(n.type)
           << " dims=(" << dims_str(n.dims) << ")"
           << " dyn=" << (n.is_in_dynamic_aos ? 1 : 0) << "\n";
    }

    // ----- public: per-AoS shape (list_nodes + getAOSShape contract) -----
    std::vector<std::string> aos_paths;
    for (const auto& kv : aos_map) aos_paths.push_back(kv.first);
    std::sort(aos_paths.begin(), aos_paths.end());
    os << "[getAOSShape " << aos_paths.size() << "]\n";
    for (const auto& p : aos_paths) {
        std::vector<size_t> shape = db.getAOSShape(p);
        os << p << " | shape=(" << dims_str(shape) << ")\n";
    }

    os << "[end]\n";
    return os.str();
}

bool read_file_to_string(const std::string& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = (argc > 1) ? argv[1] : "verify";
    const std::string corpus = "test_index_golden_corpus.panzer";

    write_corpus(corpus);
    const std::string surface = serialize_surface(corpus);

    if (mode == "capture") {
        std::filesystem::create_directories(PZGOLD_GOLDEN_DIR);
        std::ofstream out(PZGOLD_GOLDEN_FILE, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::cerr << "FAIL: cannot write golden " << PZGOLD_GOLDEN_FILE << "\n";
            return 2;
        }
        out << surface;
        out.flush();
        std::cout << "GOLDEN captured -> " << PZGOLD_GOLDEN_FILE
                  << " (" << surface.size() << " bytes, "
                  << (surface.find('\n') == std::string::npos ? 0 : 1) << "+ lines)\n";
        return 0;
    }

    // default: verify
    std::string golden;
    if (!read_file_to_string(PZGOLD_GOLDEN_FILE, golden)) {
        std::cerr << "FAIL: golden not found at " << PZGOLD_GOLDEN_FILE
                  << " (run `test_index_golden capture` first)\n";
        return 2;
    }
    if (golden == surface) {
        std::cout << "GOLDEN verify OK: " << PZGOLD_GOLDEN_FILE << " matches ("
                  << surface.size() << " bytes)\n";
        return 0;
    }

    std::cerr << "GOLDEN verify FAILED: surface differs from " << PZGOLD_GOLDEN_FILE << "\n";
    std::vector<std::string> gs, cs;
    { std::istringstream g(golden), c(surface); std::string line;
      while (std::getline(g, line)) gs.push_back(line);
      while (std::getline(c, line)) cs.push_back(line); }
    const size_t n = std::max(gs.size(), cs.size());
    size_t shown = 0;
    for (size_t i = 0; i < n && shown < 20; ++i) {
        const std::string& a = i < gs.size() ? gs[i] : "<eof>";
        const std::string& b = i < cs.size() ? cs[i] : "<eof>";
        if (a != b) {
            std::cerr << "first diff @ line " << (i + 1) << ":\n"
                      << "  golden: " << a << "\n"
                      << "  actual: " << b << "\n";
            shown++;
        }
    }
    return 1;
}
