// imas_publish_view.cpp
// Publish a user-friendly Virtual Dataset (VDS) view on a PanzerDB snapshot so
// that standard HDF5 tools (h5ls, h5dump, h5py) can inspect the logical IMAS
// instance tree without any IMAS-API dependency.
//
// Usage:
//   imas_publish_view <file.h5> [--root <name>] [--expand]
//                                     [--max-mappings N] [--stats]
//
// The view is written under the top-level group <name> (default "imas_view").
// The operation is idempotent: re-running deletes and rebuilds <name>.
//
// Default (collapse) mode folds every Array-of-Structures level and every time
// step into leading dimensions: one dataset per logical IMAS node, so h5ls -r
// lists the IMAS tree instead of one path per instance.  --expand restores the
// historical view (one VDS per index row).
//
// After publishing:
//   imas_publish_view core_profiles_3.h5
//   h5ls     core_profiles_3.h5                 # now shows /imas_view/...
//   h5dump   core_profiles_3.h5 imas_view/...    # values via VDS
//   python3 -c "import h5py; f=h5py.File('core_profiles_3.h5','r'); print(f['/imas_view/profiles_1d/grid/rho_tor_norm'][:])"

#include <view_publisher.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {

void usage() {
    std::cerr << "usage: imas_publish_view <file.h5> [--root <name>] [--expand]\n"
              << "                          [--max-mappings N] [--stats]\n"
              << "  --root <name>       view group name (default: imas_view)\n"
              << "  --expand            historical view: one VDS per index row, every\n"
              << "                        AoS instance listed under its own subgroup\n"
              << "  --max-mappings N    safety valve: above N virtual mappings a dataset\n"
              << "                        is reduced to its first slices and flagged\n"
              << "                        @mapping_cap=1 (default: 0 == unlimited)\n"
              << "  --stats             report the mapping gauges used to size that valve\n";
}

// Parses an unsigned decimal option value; reports and clears ok on garbage.
bool parseCount(const std::string& text, uint64_t& out, bool& ok) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0') {
        std::cerr << "ERROR: --max-mappings expects a non-negative integer, got: " << text << '\n';
        ok = false;
        return false;
    }
    out = static_cast<uint64_t>(v);
    return true;
}

void printStats(const imas::view::PublishStats& st) {
    std::cout << "  datasets published:     " << st.n_aggregated_datasets << '\n';
    std::cout << "  AoS groups sized:       " << st.n_aos_groups << '\n';
    std::cout << "  virtual mappings:       " << st.n_mappings << '\n';
    std::cout << "  max in one dataset:     " << st.max_mappings_in_one_dataset << '\n';
    std::cout << "  cells left as fill:     " << st.n_fill_holes << '\n';
    std::cout << "  rank overflow skipped:  " << st.n_rank_overflow << '\n';
    std::cout << "  mapping-cap fallbacks:  " << st.n_mapping_cap_fallback << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    std::string filename = argv[1];
    imas::view::PublishOptions options;
    bool stats = false;
    bool ok = true;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto needValue = [&](const char* opt, std::string& out) {
            if (i + 1 >= argc) { std::cerr << "ERROR: " << opt << " needs an argument\n"; ok = false; return; }
            out = argv[++i];
        };
        if (a == "--root") {
            std::string v;
            needValue("--root", v);
            if (ok) {
                if (v.empty()) { std::cerr << "ERROR: --root must not be empty\n"; ok = false; }
                else options.root_group = v;
            }
        } else if (a == "--expand") {
            options.collapse_aos = false;
        } else if (a == "--max-mappings") {
            std::string v;
            needValue("--max-mappings", v);
            if (ok) parseCount(v, options.max_mappings_per_dataset, ok);
        } else if (a == "--stats") {
            stats = true;
        } else {
            std::cerr << "ERROR: unknown option: " << a << '\n';
            ok = false;
        }
    }
    if (!ok) return 2;

    int ec = 0;
    try {
        const auto st = imas::view::publish(filename, options);
        std::cout << "Published " << (options.collapse_aos ? "collapsed" : "expanded")
                  << " VDS view under '/" << options.root_group << "' in " << filename << '\n';
        std::cout << "  datasets published:   "
                  << (options.collapse_aos ? st.n_aggregated_datasets : st.n_data_leaves) << '\n';
        std::cout << "  AoS groups sized:     " << st.n_aos_groups << '\n';
        std::cout << "  empty nodes skipped:  " << st.n_empty_skipped << '\n';
        std::cout << "  errors (skipped):     " << st.n_errors << '\n';
        if (stats) printStats(st);
        if (st.n_errors || st.n_rank_overflow) ec = 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        ec = 1;
    }
    return ec;
}
