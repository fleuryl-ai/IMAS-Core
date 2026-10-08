// imas_publish_view.cpp
// Publish a user-friendly Virtual Dataset (VDS) view on a PanzerDB snapshot so
// that standard HDF5 tools (h5ls, h5dump, h5py) can inspect the logical IMAS
// instance tree without any IMAS-API dependency.
//
// Usage:
//   imas_publish_view <file.h5> [--root <name>]
//
// The view is written under the top-level group <name> (default "imas_view").
// The operation is idempotent: re-running deletes and rebuilds <name>.
//
// After publishing:
//   imas_publish_view core_profiles_3.h5
//   h5ls     core_profiles_3.h5                 # now shows /imas_view/...
//   h5dump   core_profiles_3.h5 imas_view/...    # values via VDS
//   python3 -c "import h5py; f=h5py.File('core_profiles_3.h5','r'); print(f['/imas_view/profiles_1d/0/grid/rho_tor_norm'][:])"

#include <view_publisher.h>

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: imas_publish_view <file.h5> [--root <name>]\n";
        return 2;
    }
    std::string filename = argv[1];
    std::string root = "imas_view";
    bool ok = true;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--root") {
            if (i + 1 < argc) root = argv[++i];
            else { std::cerr << "ERROR: --root needs an argument\n"; return 2; }
        } else {
            std::cerr << "ERROR: unknown option: " << a << "\n";
            ok = false;
        }
    }
    if (!ok) return 2;

    int ec = 0;
    try {
        auto st = imas::view::publish(filename, root);
        std::cout << "Published VDS view under '/" << root << "' in " << filename << '\n';
        std::cout << "  data leaves (VDS):     " << st.n_data_leaves  << '\n';
        std::cout << "  AoS groups sized:      " << st.n_aos_groups   << '\n';
        std::cout << "  empty nodes skipped:   " << st.n_empty_skipped << '\n';
        std::cout << "  errors (skipped):      " << st.n_errors << '\n';
        if (st.n_errors) ec = 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        ec = 1;
    }
    return ec;
}
