// @file  bench_read.cpp
// @brief Read-parameter research bench for the v2 (PanzerDB) backend.
//
// Measures, for one v2 file:
//   - the /index + /paths load (getLeaves()),
//   - the full re-read of every data leaf, in two modes:
//       perleaf : one H5Dread per index row (worst case),
//       grouped : leaves grouped per node and pulled with the hyperslab-union
//                 readLeavesUnion()/readStringBulk() used by the GLOBAL strategy.
// Storage/cache parameters can be swept to rank their influence.
//
// Usage: bench_read --file <file.h5> [--reps N] [--cold] [--mode all|perleaf|grouped]
//                  [--cache <nbytes>,<nslots>,<w0>]
// @note Not a ctest test: a measurement tool (src/hdf5/CMakeLists.txt builds it).
#include "panzerdb.h"
#include "hdf5_backend.h"
#include "al_context.h"
#include "al_defs.h"
#include "al_const.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fcntl.h>
#include <set>
#include <iostream>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

struct Options {
    std::string file;
    int reps = 3;
    bool cold = false;
    bool perleaf = true;
    bool grouped = true;
    bool cache_set = false;
    bool global = false;
    std::string dir, ids;
    size_t cache_bytes = 0, cache_slots = 0;
    double cache_w0 = 0.75;
};

static void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " --file <v2 file> [--reps N] [--cold] [--mode all|perleaf|grouped]"
                 " [--cache <nbytes>,<nslots>,<w0>]\n";
}

static bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--file" && i + 1 < argc) o.file = argv[++i];
        else if (a == "--reps" && i + 1 < argc) o.reps = std::atoi(argv[++i]);
        else if (a == "--cold") o.cold = true;
        else if (a == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            o.perleaf = (m == "all" || m == "perleaf");
            o.grouped = (m == "all" || m == "grouped");
        }
        else if (a == "--global") o.global = true;
        else if (a == "--dir" && i + 1 < argc) o.dir = argv[++i];
        else if (a == "--ids" && i + 1 < argc) o.ids = argv[++i];
        else if (a == "--cache" && i + 1 < argc) {
            unsigned long long nb = 0, ns = 0; double w0 = 0.75;
            if (sscanf(argv[++i], "%llu,%llu,%lf", &nb, &ns, &w0) >= 2) {
                o.cache_set = true; o.cache_bytes = nb; o.cache_slots = ns; o.cache_w0 = w0;
            }
        }
        else { usage(argv[0]); return false; }
    }
    if (o.file.empty()) { usage(argv[0]); return false; }
    if (o.global) {
        if (o.dir.empty() || o.ids.empty()) { usage(argv[0]); return false; }
        o.perleaf = o.grouped = false;   // --global replaces the engine modes
    }
    return true;
}

static void drop_page_cache(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
}

// Node key of a leaf: its path with the AoS element indices removed.
static std::string schema_key(const PanzerDB::Leaf& leaf) {
    std::string out;
    const std::string_view p = leaf.path;
    size_t pos = 0;
    while (pos <= p.size()) {
        size_t next = p.find('/', pos);
        std::string tok(p.substr(pos, (next == std::string_view::npos) ? std::string_view::npos : next - pos));
        const bool numeric = !tok.empty() &&
            std::all_of(tok.begin(), tok.end(), [](char c) { return c >= '0' && c <= '9'; });
        if (!numeric) { if (!out.empty()) out += '/'; out += tok; }
        if (next == std::string_view::npos) break;
        pos = next + 1;
    }
    return out;
}


// ---------------------------------------------------------------------------
// GLOBAL strategy mode: walk the whole IDS through HDF5Backend (GLOBAL_OP read)
// ---------------------------------------------------------------------------

struct GNode {
    std::map<std::string, GNode> children;   // AoS name -> subtree
    std::set<std::pair<std::string, int>> datasets;  // dataset name, is_dynamic
    bool dyn = false;                        // this AoS level declares a timebase
};

// Splits a stored path into '/'-separated tokens.
static std::vector<std::string> split_tokens(std::string_view p) {
    std::vector<std::string> tok;
    size_t pos = 0;
    while (pos <= p.size()) {
        size_t n = p.find('/', pos);
        tok.push_back(std::string(p.substr(pos, (n == std::string_view::npos) ? std::string_view::npos : n - pos)));
        if (n == std::string_view::npos) break;
        pos = n + 1;
    }
    return tok;
}

static bool is_numeric_token(const std::string& t) {
    if (t.empty()) return false;
    for (char c : t) if (c < '0' || c > '9') return false;
    return true;
}

// Builds the AoS tree of the IDS out of the index rows.
// Pass 1: the AoS meta rows give the schema tree + which levels are dynamic.
// Pass 2: every row is parsed greedily against that tree (a token that names a
// child AoS opens that level, an optional numeric token is its element index),
// what is left is the (flattened) dataset name of that node.
static void build_tree(const std::vector<PanzerDB::Leaf>& leaves, GNode& root) {
    auto descend = [](GNode& n, const std::string& name) -> GNode* {
        auto it = n.children.find(name);
        return (it == n.children.end()) ? nullptr : &it->second;
    };
    for (const auto& l : leaves) {
        const uint64_t role = l.flags & 0xFULL;
        if (role != 2 && role != 3) continue;
        GNode* cur = &root;
        for (const auto& t : split_tokens(l.path)) {
            if (is_numeric_token(t)) continue;
            GNode* nxt = descend(*cur, t);
            if (!nxt) nxt = &cur->children[t];
            cur = nxt;
        }
        cur->dyn = (role == 3);
    }
    for (const auto& l : leaves) {
        const uint64_t role = l.flags & 0xFULL;
        if (role > 1) continue;
        const auto tok = split_tokens(l.path);
        GNode* cur = &root;
        size_t i = 0;
        while (i < tok.size()) {
            GNode* nxt = descend(*cur, tok[i]);
            if (!nxt) break;
            cur = nxt;
            ++i;
            if (i < tok.size() && is_numeric_token(tok[i])) ++i;   // element index
        }
        std::string ds;
        for (size_t k = i; k < tok.size(); ++k) { if (k > i) ds += "/"; ds += tok[k]; }
        if (!ds.empty()) cur->datasets.insert({ds, l.time_index > 0 ? 1 : 0});
    }
}

struct GCounters { uint64_t d_hash = 0; long long i_sum = 0, str_len = 0, nodes = 0, d_elems = 0, i_elems = 0; };

static void mix64(uint64_t& acc, uint64_t v) {
    acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
}

static int al_type_of(PanzerDB::DataType dt) {
    switch (dt) {
        case PanzerDB::DataType::INT32: return alconst::integer_data;
        case PanzerDB::DataType::COMPLEX128: return alconst::complex_data;
        case PanzerDB::DataType::STRING:
        case PanzerDB::DataType::LIST_OF_STRINGS:
        case PanzerDB::DataType::STRING_CHUNKED: return alconst::char_data;
        default: return alconst::double_data;
    }
}

template <typename Ctx>
static void read_node_datasets(HDF5Backend& backend, Ctx* ctx, const GNode& node,
                               const std::map<std::string, PanzerDB::DataType>& dtype_of, GCounters& c) {
    for (const auto& [name, dyn] : node.datasets) {
        void* data = nullptr;
        int type = al_type_of(dtype_of.count(name) ? dtype_of.at(name) : PanzerDB::DataType::FLOAT64);
        int dim = 0, size[8] = {0};
        const int ok = backend.readData(ctx, name, dyn ? "time" : "", &data, &type, &dim, size);
        ++c.nodes;
        if (!ok || !data) continue;
        size_t n = 1;
        for (int d = 0; d < dim; ++d) n *= (size_t)size[d];
        switch (type) {
            case alconst::double_data: {
                const double* v = (const double*)data;
                c.d_elems += (long long)n;
                for (size_t k = 0; k < n; ++k) { uint64_t b; std::memcpy(&b, &v[k], 8); mix64(c.d_hash, b); }
                break; }
            case alconst::integer_data: {
                const int32_t* v = (const int32_t*)data;
                c.i_elems += (long long)n;
                for (size_t k = 0; k < n; ++k) c.i_sum += v[k];
                break; }
            case alconst::complex_data: {
                const double* v = (const double*)data;
                for (size_t k = 0; k < 2 * n; ++k) { uint64_t b; std::memcpy(&b, &v[k], 8); mix64(c.d_hash, b); }
                break; }
            default: {
                const char* s = (const char*)data;
                c.str_len += (long long)std::strlen(s);
                break; }
        }
        free(data);
    }
}

template <typename Ctx>
static void walk(HDF5Backend& backend, Ctx* ctx, const GNode& node,
                 const std::map<std::string, PanzerDB::DataType>& dtype_of, GCounters& c) {
    read_node_datasets(backend, ctx, node, dtype_of, c);
    for (const auto& [aname, child] : node.children) {
        ArraystructContext actx(ctx, aname, child.dyn ? "time" : "");
        int asize = 0;
        backend.beginArraystructAction(&actx, &asize);
        if (asize <= 0) continue;
        for (int j = 0; j < asize; ++j) {
            // One readData per AoS element, exactly like a GLOBAL_OP get():
            // an AoS is a list of structures, dynamic or not.
            read_node_datasets(backend, &actx, child, dtype_of, c);
            for (const auto& [cname, gchild] : child.children) {
                ArraystructContext gctx(&actx, cname, gchild.dyn ? "time" : "");
                int gsize = 0;
                backend.beginArraystructAction(&gctx, &gsize);
                for (int k = 0; k < gsize; ++k) {
                    walk(backend, &gctx, gchild, dtype_of, c);
                    if (k < gsize - 1) gctx.nextIndex(1);
                }
                backend.endAction(&gctx);
            }
            if (j < asize - 1) actx.nextIndex(1);
        }
        backend.endAction(&actx);
    }
}

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) return 1;

    struct Run { double index_ms = 0, perleaf_ms = 0, grouped_ms = 0; };
    std::vector<Run> runs;
    struct GRun { double index_ms; };
    std::vector<GRun> g_runs;

    for (int rep = 0; rep < o.reps; ++rep) {
        if (o.cold) drop_page_cache(o.file);

        Run r;
        auto t0 = clock_type::now();
        PanzerDB db(o.file, PanzerDB::OpenMode::READ);
        if (o.cache_set) db.setReadCache(o.cache_bytes, o.cache_slots, o.cache_w0);
        const auto& leaves = db.getLeaves();
        r.index_ms = ms_since(t0);

        // Classify the rows: data leaves only (skip AoS meta nodes and empty nodes).
        std::vector<const PanzerDB::Leaf*> data;
        std::map<std::pair<std::string, int>, std::vector<const PanzerDB::Leaf*>> groups;
        uint64_t raw_bytes = 0;
        for (const auto& l : leaves) {
            const uint64_t role = l.flags & 0xFULL;
            if (role == 1 || role == 2 || role == 3) continue;
            if (l.count == 0) continue;
            const auto dtype = static_cast<PanzerDB::DataType>(l.flags >> 4);
            data.push_back(&l);
            size_t es = 8;
            switch (dtype) {
                case PanzerDB::DataType::INT32: es = 4; break;
                case PanzerDB::DataType::COMPLEX128: es = 16; break;
                case PanzerDB::DataType::STRING:
                case PanzerDB::DataType::LIST_OF_STRINGS:
                case PanzerDB::DataType::STRING_CHUNKED: es = 512; break;
                default: break;
            }
            raw_bytes += l.count * es;
            groups[{schema_key(l), static_cast<int>(dtype)}].push_back(&l);
        }

        // Checksums, to prove every variant returns the same content.
        // NaN-safe checksums (this file holds NaN/inf values): XOR of bit patterns.
        uint64_t sum_d = 0; long long sum_i = 0, str_len = 0;
        auto mix = [](uint64_t& acc, uint64_t v) { acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2); };

        if (o.perleaf) {
            auto t1 = clock_type::now();
            for (const auto* l : data) {
                const auto dtype = static_cast<PanzerDB::DataType>(l->flags >> 4);
                switch (dtype) {
                    case PanzerDB::DataType::FLOAT64: {
                        std::vector<double> buf(l->count);
                        db.readTensor(*l, buf.data());
                        for (double v : buf) { uint64_t b; std::memcpy(&b, &v, 8); mix(sum_d, b); }
                        break; }
                    case PanzerDB::DataType::INT32: {
                        std::vector<int32_t> buf(l->count);
                        db.readTensor(*l, buf.data());
                        for (int32_t v : buf) sum_i += v;
                        break; }
                    case PanzerDB::DataType::COMPLEX128: {
                        std::vector<std::complex<double>> buf(l->count);
                        db.readTensor(*l, buf.data());
                        for (const auto& v : buf) { uint64_t b; std::memcpy(&b, &v, 8); mix(sum_d, b); }
                        break; }
                    default: {
                        auto strs = db.readStringBulk(l->offset, l->offset + l->count);
                        for (const auto& s : strs) str_len += (long long)s.size();
                        break; }
                }
            }
            r.perleaf_ms = ms_since(t1);
        }

        if (o.grouped) {
            auto t1 = clock_type::now();
            for (auto& [key, ls] : groups) {
                const auto dtype = static_cast<PanzerDB::DataType>(key.second);
                if (dtype == PanzerDB::DataType::STRING || dtype == PanzerDB::DataType::LIST_OF_STRINGS ||
                    dtype == PanzerDB::DataType::STRING_CHUNKED) {
                    uint64_t lo = ls.front()->offset, hi = lo;
                    for (const auto* l : ls) { lo = std::min(lo, l->offset); hi = std::max(hi, l->offset + l->count); }
                    auto strs = db.readStringBulk(lo, hi);
                    for (const auto& s : strs) str_len += (long long)s.size();
                } else {
                    uint64_t total = 0;
                    for (const auto* l : ls) total += l->count;
                    if (dtype == PanzerDB::DataType::FLOAT64) {
                        std::vector<double> buf(total);
                        db.readLeavesUnion(ls, buf.data(), dtype);
                        for (double v : buf) { uint64_t b; std::memcpy(&b, &v, 8); mix(sum_d, b); }
                    } else if (dtype == PanzerDB::DataType::INT32) {
                        std::vector<int32_t> buf(total);
                        db.readLeavesUnion(ls, buf.data(), dtype);
                        for (int32_t v : buf) sum_i += v;
                    } else if (dtype == PanzerDB::DataType::COMPLEX128) {
                        std::vector<std::complex<double>> buf(total);
                        db.readLeavesUnion(ls, buf.data(), dtype);
                        for (const auto& v : buf) { uint64_t b; std::memcpy(&b, &v, 8); mix(sum_d, b); }
                    }
                }
            }
            r.grouped_ms = ms_since(t1);
        }

        printf("rep %d: index %.1f ms | perleaf %.1f ms | grouped %.1f ms | nodes %zu | rows %zu | raw %.1f MB | checksum d=%016llx i=%lld str=%lld\n",
               rep, r.index_ms, r.perleaf_ms, r.grouped_ms, groups.size(), data.size(),
               raw_bytes / 1e6, (unsigned long long)sum_d, sum_i, str_len);
        runs.push_back(r);
    }

    if (o.global) {
        namespace fs = std::filesystem;
        const std::string uri = "imas:hdf5?path=" + o.dir;
        // A pulse directory needs its master file: create it once, then drop the
        // file under study in place of the stub IDS file.
        {
            fs::create_directories(o.dir);
            DataEntryContext dec(uri);
            HDF5Backend b;
            b.openPulse(&dec, FORCE_CREATE_PULSE);
            OperationContext oc(&dec, o.ids, "", WRITE_OP);
            b.beginAction(&oc);
            b.endAction(&oc);
            b.closePulse(&dec, CLOSE_PULSE);
        }
        fs::copy_file(o.file, o.dir + "/" + o.ids + ".h5", fs::copy_options::overwrite_existing);

        // Tree + types come from the index (outside of the timed sections).
        GNode root;
        std::map<std::string, PanzerDB::DataType> dtype_of;
        {
            PanzerDB db(o.file, PanzerDB::OpenMode::READ);
            for (const auto& l : db.getLeaves()) {
                if ((l.flags & 0xFULL) > 1) continue;
                const auto p = l.path.find_last_of('/');
                dtype_of[std::string(p == std::string_view::npos ? l.path : l.path.substr(p + 1))] =
                    static_cast<PanzerDB::DataType>(l.flags >> 4);
            }
            build_tree(db.getLeaves(), root);
            if (getenv("BENCH_DEBUG")) {
                int shown = 0;
                for (const auto& l : db.getLeaves()) {
                    if ((l.flags & 0xFULL) <= 1) {
                        std::cerr << "LEAF " << l.path << " role=" << (l.flags & 0xF)
                                  << " t=" << l.time_index << " cnt=" << l.count << "\n";
                        if (++shown >= 12) break;
                    }
                }
                std::map<std::string, int> hist;
                for (const auto& l : db.getLeaves()) {
                    if ((l.flags & 0xFULL) > 1 || l.count == 0) continue;
                    hist[std::string(l.path)]++;
                }
                std::vector<std::pair<int, std::string>> top;
                for (auto& [k, v] : hist) top.push_back({v, k});
                std::sort(top.rbegin(), top.rend());
                std::cerr << "DISTINCT DATA PATHS: " << hist.size() << "\n";
                for (size_t k = 0; k < top.size() && k < 6; ++k) std::cerr << "  " << top[k].second << " x" << top[k].first << "\n";
                std::vector<std::string> deep;
                for (const auto& l : db.getLeaves()) {
                    if ((l.flags & 0xFULL) > 1) continue;
                    const size_t slashes = l.path.find('/') == std::string_view::npos ? 0 : 1 + std::count(l.path.begin(), l.path.end(), '/');
                    if (slashes >= 3) { deep.push_back(std::string(l.path)); }
                }
                std::cerr << "DEEP DATA ROWS: " << deep.size() << "\n";
                for (size_t k = 0; k < deep.size() && k < 8; ++k) std::cerr << "  " << deep[k] << "\n";
                shown = 0;
                for (const auto& l : db.getLeaves()) {
                    if (l.flags < 0xF && l.path.find("ion") != std::string_view::npos) {
                        std::cerr << "IONLEAF " << l.path << " cnt=" << l.count << " t=" << l.time_index << "\n";
                        if (++shown >= 10) break;
                    }
                }
                shown = 0;
                for (const auto& l : db.getLeaves()) {
                    if ((l.flags & 0xFULL) >= 2) {
                        std::cerr << "META " << l.path << " role=" << (l.flags & 0xF)
                                  << " shape0=" << (l.shape.empty() ? 0 : l.shape[0]) << "\n";
                        if (++shown >= 12) break;
                    }
                }
            }
            if (getenv("BENCH_DEBUG")) {
                std::function<void(const GNode&, int)> dump = [&](const GNode& n, int d) {
                    fprintf(stderr, "%*s datasets=%zu children:", 2 * d, "", n.datasets.size());
                    for (const auto& [k, c] : n.children) fprintf(stderr, " %s%s", k.c_str(), c.dyn ? "(t)" : "");
                    fprintf(stderr, "\n");
                    for (const auto& [k, c] : n.children) if (d < 2) dump(c, d + 1);
                };
                dump(root, 0);
            }
        }

        for (int rep = 0; rep < o.reps; ++rep) {
            if (o.cold) drop_page_cache(o.dir + "/" + o.ids + ".h5");
            auto t0 = clock_type::now();
            DataEntryContext dec(uri);
            HDF5Backend backend;
            backend.openPulse(&dec, OPEN_PULSE);
            const double open_ms = ms_since(t0);

            auto t1 = clock_type::now();
            OperationContext oc(&dec, o.ids, "", READ_OP);
            backend.beginAction(&oc);
            GCounters c;
            walk(backend, &oc, root, dtype_of, c);
            backend.endAction(&oc);
            const double read_ms = ms_since(t1);

            backend.closePulse(&dec, CLOSE_PULSE);
            printf("rep %d: open %.1f ms | global read %.1f ms | nodes %lld | d_elems %lld i_elems %lld | hash %016llx i=%lld str=%lld\n",
                   rep, open_ms, read_ms, c.nodes, c.d_elems, c.i_elems, (unsigned long long)c.d_hash, c.i_sum, c.str_len);
            g_runs.push_back({read_ms});
        }
        std::vector<double> gv;
        for (const auto& r : g_runs) gv.push_back(r.index_ms);
        std::sort(gv.begin(), gv.end());
        printf("MEDIAN GLOBAL READ: %.1f ms\n", gv[gv.size() / 2]);
        return 0;
    }

    auto median = [&](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::vector<double> idx, pl, gr;
    for (const auto& r : runs) { idx.push_back(r.index_ms); pl.push_back(r.perleaf_ms); gr.push_back(r.grouped_ms); }
    printf("MEDIAN: index %.1f ms | perleaf %.1f ms | grouped %.1f ms\n",
           median(idx), o.perleaf ? median(pl) : 0.0, o.grouped ? median(gr) : 0.0);
    return 0;
}
