// imas_publish_view.cpp
// Publish a user-friendly Virtual Dataset (VDS) view on a PanzerDB snapshot so
// that standard HDF5 tools (h5ls, h5dump, h5py) can inspect the logical IMAS
// instance tree without any IMAS-API dependency.
//
// Usage:
//   imas_publish_view <file.h5> [--root <name>] [--expand]
//                                     [--max-mappings N] [--stats]
//   imas_publish_view <file.h5> --list [--root <name>]
//
// The view is written under the top-level group <name> (default "imas_view").
// The operation is idempotent: re-running deletes and rebuilds <name>.
//
// Default (collapse) mode folds every Array-of-Structures level and every time
// step into leading dimensions: one dataset per logical IMAS node, so h5ls -r
// lists the IMAS tree instead of one path per instance.  --expand restores the
// historical view (one VDS per index row).
//
// --list does NOT publish: it reads an existing view back, read-only, and prints
// one annotated line per object.  It exists because `h5ls -r` shows no attribute
// at all, so @dims_label / @ragged / @aos_size are invisible in a plain listing
// (h5ls -v -r shows attribute *names*, only h5dump -A and h5py show values).
//
// After publishing:
//   imas_publish_view core_profiles_3.h5
//   imas_publish_view core_profiles_3.h5 --list   # tree + annotations, one line each
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
              << "                          [--no-aos-groups] [--max-mappings N] [--stats]\n"
              << "       imas_publish_view <file.h5> --list [--root <name>]\n"
              << "  --root <name>       view group name (default: imas_view)\n"
              << "  --expand            historical view: one VDS per index row, every\n"
              << "                        AoS instance listed under its own subgroup\n"
              << "  --no-aos-groups     collapse mode: publish no AoS group at all, every\n"
              << "                        dataset flat under the view root with ':' joining\n"
              << "                        its path (imas_view/grids_ggd:grid:…:geometry).\n"
              << "                        h5ls then lists the datasets only; each AoS size\n"
              << "                        still shows in the shape.  No effect --expand\n"
              << "  --max-mappings N    safety valve: above N virtual mappings a dataset\n"
              << "                        is reduced to its first slices and flagged\n"
              << "                        @mapping_cap=1 (default: 0 == unlimited)\n"
              << "  --stats             report the mapping gauges used to size that valve\n"
              << "  --list              read-only: print an already published view, one\n"
              << "                        annotated line per object (path, kind, dims,\n"
              << "                        @dims_label, @ragged, @aos_size, holes=...).\n"
              << "                        Publishes nothing; exits 2 if no view exists\n";
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
    bool listOnly = false;
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
        } else if (a == "--no-aos-groups") {
            options.no_aos_groups = true;
        } else if (a == "--max-mappings") {
            std::string v;
            needValue("--max-mappings", v);
            if (ok) parseCount(v, options.max_mappings_per_dataset, ok);
        } else if (a == "--stats") {
            stats = true;
        } else if (a == "--list") {
            listOnly = true;
        } else {
            std::cerr << "ERROR: unknown option: " << a << '\n';
            ok = false;
        }
    }
    if (!ok) return 2;

    if (listOnly) {
        // Read-only path: the file is not opened for writing, so --list works on a
        // snapshot nobody is appending to.  It lists whatever view the file holds,
        // which is why the publishing options below do not apply here.
        if (!options.collapse_aos || options.max_mappings_per_dataset || stats)
            std::cerr << "note: --list prints the view as stored; "
                         "--expand/--max-mappings/--stats only apply when publishing\n";
        try {
            const auto res = imas::view::listView(filename, options.root_group);
            if (!res.found) {
                std::cerr << "ERROR: no '/" << options.root_group << "' view in " << filename
                          << " — publish it first: imas_publish_view " << filename << '\n';
                return 2;
            }
            for (const auto& line : res.lines) std::cout << line << '\n';
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << '\n';
            return 1;
        }
    }

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
