// @file  test_al_upgrade_v1_to_v2.cpp
// @brief Guards the v1 -> v2 direction of the version decision: rewriting a v1 pulse with a 2.0
//        target (BACKEND_ALLOW_AUTO_UPGRADE) has to leave a v2 pulse that the next open finds.
//
// Opening a v1 pulse while the target version is 2.0 is allowed under that flag, and the writer
// then lays the v2 tables down. Two things are needed for that to be a rewrite rather than a
// half-written file, and both are the backend's job:
//  - the v1 datasets have to go. The v2 tables and the v1 datasets live in the same group of the
//    same IDS file, so left alone the file ends up holding both layouts;
//  - master.h5 has to stop advertising "1.0". That attribute is what HDF5Utils::openPulse() reads
//    and what picks the reader of the next open, so a leftover "1.0" sends the next reader to the
//    v1 engine -- which then reads the datasets that are no longer there and never sees what this
//    session wrote. Worse, reopening without the flag throws outright.
//
// So the test writes a pulse in v1, rewrites it with a 2.0 target, and checks: the lock is still
// there when the flag is off (and a refusal touched nothing); afterwards the master file and the
// IDS file are both labelled 2.0; the v1 datasets are gone and the v2 tables took their place;
// and the pulse reopens without the flag and reads back through the v2 reader with its values.
//
// The pulse is not read through the AL while it is still v1: the v1 read path of this tree is not
// maintained (BACKEND_DEFAULT_WRITE_VERSION=1.0 makes 14 of the 53 test_al_* fail, two with a
// segfault), so an AL read there could fail for reasons that have nothing to do with this
// behaviour. What the pulse holds before the rewrite is therefore checked through the HDF5 API --
// which is also the only way to see that those datasets did go away afterwards.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <hdf5.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

const std::string URI = "imas:hdf5?path=./test_db_upgrade_v1_to_v2";
const std::string PULSE_DIR = "./test_db_upgrade_v1_to_v2";
const std::string MASTER_FILE = PULSE_DIR + "/master.h5";
const std::string IDS_FILE = PULSE_DIR + "/core_profiles.h5";

const int POINTS = 10;      // radial points of one signal_1d slice
const int STEPS = 3;        // slices in the pulse (t = 0.0, 0.1, 0.2)

// Both layouts put their objects in the group named after the IDS: "core_profiles/index" for the
// v2 table, "core_profiles/time" for a v1 dataset. The names below are therefore path-like.
const char* V2_TABLES[] = {"core_profiles/index", "core_profiles/path_table",
                           "core_profiles/data_raw_f64"};
const char* V1_DATASETS[] = {"core_profiles/time", "core_profiles/profiles_1d[]&signal_1d"};

double expected_value(int t_idx, int x) { return 100.0 + t_idx * 10.0 + x; }
// Maps a time to its time index: the time base is 0.0, 0.1, ... so t*10 is the index.
int time_index_of(double t) { return (int)(t * 10.0 + 0.5); }

void use_v1() { setenv("BACKEND_DEFAULT_WRITE_VERSION", "1.0", 1); }
void use_v2() { setenv("BACKEND_DEFAULT_WRITE_VERSION", "2.0", 1); }
void allow_upgrade(bool on) {
    if (on) setenv("BACKEND_ALLOW_AUTO_UPGRADE", "true", 1);
    else unsetenv("BACKEND_ALLOW_AUTO_UPGRADE");
}

// Writes the whole IDS in one GLOBAL_OP put -- what an upgrade is in practice: everything the
// caller has is written again in the target layout. mode is FORCE_CREATE_PULSE for the pulse this
// test creates itself, OPEN_PULSE for the session that rewrites it.
static void write_global(HDF5Backend &backend, DataEntryContext &ctx, int mode) {
    backend.openPulse(&ctx, mode);
    if (mode == OPEN_PULSE)
        backend.getVersion(&ctx);   // what the AL calls right after openPulse: that arms the upgrade

    OperationContext opCtx(&ctx, "core_profiles", "", WRITE_OP);   // the 4-arg form defaults to GLOBAL_OP
    backend.beginAction(&opCtx);

    int homogeneous_time = 1;
    backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time,
                      alconst::integer_data, 0, nullptr);

    std::vector<double> times(STEPS);
    for (int i = 0; i < STEPS; ++i) times[i] = 0.1 * i;
    int time_dim = 1;
    int time_size[] = {STEPS};
    backend.writeData(&opCtx, "time", "time", times.data(), alconst::double_data,
                      time_dim, time_size);

    int profiles_size = STEPS;
    ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
    backend.beginArraystructAction(&profilesCtx, &profiles_size);
    for (int i = 0; i < STEPS; ++i) {
        std::vector<double> val(POINTS);
        for (int x = 0; x < POINTS; ++x) val[x] = expected_value(i, x);
        int sig_dim = 1;
        int sig_size[] = {POINTS};
        backend.writeData(&profilesCtx, "signal_1d", "time", val.data(),
                          alconst::double_data, sig_dim, sig_size);
        if (i < STEPS - 1) profilesCtx.nextIndex(1);
    }
    backend.endAction(&profilesCtx);
    backend.endAction(&opCtx);
    backend.closePulse(&ctx, mode);
}

// Reads one slice and compares it against what was written there.
static bool read_slice(HDF5Backend &backend, DataEntryContext &ctx, double target_time) {
    OperationContext opCtx(&ctx, "core_profiles", READ_OP, alconst::slice_op, target_time,
                           alconst::closest_interp);
    backend.beginAction(&opCtx);

    const int t_idx = time_index_of(target_time);
    ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
    int profiles_size = 0;
    backend.beginArraystructAction(&profilesCtx, &profiles_size);
    if (profiles_size != 1) {
        std::cerr << "profiles_1d holds " << profiles_size << " instance(s) at t="
                  << target_time << ", expected 1" << std::endl;
        return false;
    }

    void *data = nullptr;
    int type = alconst::double_data;
    int dim = 0;
    int size[H5S_MAX_RANK];
    backend.readData(&profilesCtx, "signal_1d", "time", &data, &type, &dim, size);
    if (!data) {
        std::cerr << "signal_1d read returned nothing at t=" << target_time << std::endl;
        return false;
    }
    bool ok = (dim == 1 && size[0] == POINTS);
    if (!ok) {
        std::cerr << "signal_1d at t=" << target_time << " came back with dim=" << dim
                  << " size[0]=" << size[0] << ", expected 1/" << POINTS << std::endl;
    }
    const double *vals = static_cast<const double *>(data);
    for (int x = 0; x < POINTS && ok; ++x) {
        const double want = expected_value(t_idx, x);
        if (std::abs(vals[x] - want) > 1e-9) {
            std::cerr << "signal_1d[" << x << "] at t=" << target_time << " = " << vals[x]
                      << ", expected " << want << std::endl;
            ok = false;
        }
    }
    free(data);
    backend.endAction(&profilesCtx);
    backend.endAction(&opCtx);
    return ok;
}

// Reads the HDF5_BACKEND_VERSION attribute of a pulse file (the master file picks the reader
// for the next open; each IDS file carries the layout its writer laid down).
static std::string file_version(const std::string &path) {
    hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) return "<file unreadable>";
    std::string out = "<attribute absent>";
    if (H5Aexists(f, "HDF5_BACKEND_VERSION") > 0) {
        hid_t a = H5Aopen(f, "HDF5_BACKEND_VERSION", H5P_DEFAULT);
        if (a >= 0) {
            char buf[32] = {0};
            hid_t t = H5Tcopy(H5T_C_S1);
            H5Tset_size(t, 20);
            // Same encoding as the stored attribute: HDF5 refuses the read on a character
            // set mismatch, which would look like "the attribute is gone".
            H5Tset_cset(t, H5T_CSET_UTF8);
            if (H5Aread(a, t, buf) >= 0) out = buf;
            H5Tclose(t);
            H5Aclose(a);
        }
    }
    H5Fclose(f);
    return out;
}

// True when one IDS file holds that object. Names are given as "<IDS>/<object>", which is how
// both layouts store them; an IDS file that is not there holds nothing.
static bool object_in_IDS_file(const char *name) {
    hid_t f = H5Fopen(IDS_FILE.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) return false;
    const bool present = (H5Lexists(f, name, H5P_DEFAULT) > 0);
    H5Fclose(f);
    return present;
}

// Reads a double dataset of the IDS file: its extent in `shape`, its values in `values`.
static bool read_double_dataset(const char *name, std::vector<hsize_t> &shape,
                                std::vector<double> &values) {
    hid_t f = H5Fopen(IDS_FILE.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) {
        std::cerr << IDS_FILE << " cannot be opened to read '" << name << "'" << std::endl;
        return false;
    }
    bool ok = false;
    hid_t d = H5Dopen2(f, name, H5P_DEFAULT);
    if (d < 0) {
        std::cerr << "'" << name << "' is not in " << IDS_FILE << std::endl;
    } else {
        hid_t s = H5Dget_space(d);
        const int rank = H5Sget_simple_extent_ndims(s);
        shape.resize(rank);
        H5Sget_simple_extent_dims(s, shape.data(), nullptr);
        hsize_t count = 1;
        for (int r = 0; r < rank; ++r) count *= shape[r];
        values.resize(count);
        ok = (H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
        if (!ok) std::cerr << "'" << name << "' could not be read" << std::endl;
        H5Sclose(s);
        H5Dclose(d);
    }
    H5Fclose(f);
    return ok;
}

// Checks one dataset against the shape and the values that were written for it.
static bool check_double_dataset(const char *name, std::vector<hsize_t> want_shape,
                                 const std::vector<double> &want_values) {
    std::vector<hsize_t> shape;
    std::vector<double> values;
    if (!read_double_dataset(name, shape, values)) return false;
    if (shape != want_shape) {
        std::cerr << "'" << name << "' has " << shape.size() << " dimension(s) [";
        for (size_t r = 0; r < shape.size(); ++r)
            std::cerr << (r ? "," : "") << shape[r];
        std::cerr << "], expected [";
        for (size_t r = 0; r < want_shape.size(); ++r)
            std::cerr << (r ? "," : "") << want_shape[r];
        std::cerr << "]" << std::endl;
        return false;
    }
    for (size_t i = 0; i < want_values.size(); ++i) {
        if (i >= values.size() || std::abs(values[i] - want_values[i]) > 1e-9) {
            std::cerr << "'" << name << "' element " << i << " = "
                      << (i < values.size() ? values[i] : NAN) << ", expected "
                      << want_values[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    try {
        std::cout << "\n=== Test: rewrite a v1 pulse as a v2 pulse ===\n";
        if (fs::exists(PULSE_DIR)) fs::remove_all(PULSE_DIR);
        allow_upgrade(false);

        std::cout << "--- Phase 1: write the pulse in v1 ---\n";
        use_v1();
        {
            HDF5Backend backend;
            DataEntryContext ctx(URI);
            write_global(backend, ctx, FORCE_CREATE_PULSE);
        }
        std::cout << "    master file labelled " << file_version(MASTER_FILE)
                  << ", IDS file " << file_version(IDS_FILE) << "\n";
        if (file_version(MASTER_FILE) != "1.0" || file_version(IDS_FILE) != "1.0") {
            std::cerr << "phase 1 did not produce a v1 pulse" << std::endl;
            return 1;
        }
        std::vector<double> want_times;
        for (int i = 0; i < STEPS; ++i) want_times.push_back(0.1 * i);
        std::vector<double> want_signals;
        for (int i = 0; i < STEPS; ++i)
            for (int x = 0; x < POINTS; ++x) want_signals.push_back(expected_value(i, x));
        if (!check_double_dataset("core_profiles/time", {STEPS}, want_times) ||
            !check_double_dataset("core_profiles/profiles_1d[]&signal_1d", {STEPS, POINTS}, want_signals)) {
            std::cerr << "phase 1 did not leave the v1 datasets the rewrite has to replace" << std::endl;
            return 1;
        }
        std::cout << "    v1 datasets there: core_profiles/time {3}, "
                     "core_profiles/profiles_1d[]&signal_1d {3,10}\n";

        std::cout << "--- Phase 2: without the flag, opening it as 2.0 must still be refused ---\n";
        {
            use_v2();
            allow_upgrade(false);
            HDF5Backend backend;
            DataEntryContext ctx(URI);
            backend.openPulse(&ctx, OPEN_PULSE);
            bool refused = false;
            try {
                backend.getVersion(&ctx);   // what the AL calls right after openPulse
            } catch (const ALBackendException &) {
                refused = true;
            }
            if (!refused) {
                std::cerr << "a v1 pulse opened with a 2.0 target was accepted although "
                             "BACKEND_ALLOW_AUTO_UPGRADE is not set" << std::endl;
                return 1;
            }
            backend.closePulse(&ctx, OPEN_PULSE);
            std::cout << "    refused as expected\n";
        }
        if (file_version(MASTER_FILE) != "1.0") {
            std::cerr << "a refused upgrade changed the pulse label" << std::endl;
            return 1;
        }
        for (const char *name : V1_DATASETS)
            if (!object_in_IDS_file(name)) {
                std::cerr << "'" << name << "' is gone although the upgrade was refused" << std::endl;
                return 1;
            }

        std::cout << "--- Phase 3: with the flag, write the IDS again with a 2.0 target ---\n";
        {
            use_v2();
            allow_upgrade(true);
            HDF5Backend backend;
            DataEntryContext ctx(URI);
            write_global(backend, ctx, OPEN_PULSE);
            std::cout << "    rewritten in one GLOBAL put\n";
        }
        std::cout << "    master file now labelled " << file_version(MASTER_FILE)
                  << ", IDS file " << file_version(IDS_FILE) << "\n";
        if (file_version(MASTER_FILE) != "2.0" || file_version(IDS_FILE) != "2.0") {
            std::cerr << "the upgraded pulse is labelled " << file_version(MASTER_FILE)
                      << " in the master file and " << file_version(IDS_FILE)
                      << " in the IDS file -- a later open would pick the v1 reader" << std::endl;
            return 1;
        }

        std::cout << "--- Phase 4: the v2 tables replaced the v1 datasets, they did not join them ---\n";
        for (const char *table : V2_TABLES) {
            if (!object_in_IDS_file(table)) {
                std::cerr << "'" << table << "' is missing from " << IDS_FILE
                          << " -- the 2.0 target did not lay the v2 layout down" << std::endl;
                return 1;
            }
        }
        for (const char *name : V1_DATASETS) {
            if (object_in_IDS_file(name)) {
                std::cerr << "'" << name << "' is still in " << IDS_FILE
                          << " -- the file holds both layouts, and only one of them will be read"
                          << std::endl;
                return 1;
            }
        }
        std::cout << "    v2 tables in, v1 datasets out\n";

        std::cout << "--- Phase 5: reopen without the flag: a plain v2 pulse, and readable ---\n";
        {
            use_v2();
            allow_upgrade(false);
            HDF5Backend backend;
            DataEntryContext ctx(URI);
            backend.openPulse(&ctx, OPEN_PULSE);
            try {
                backend.getVersion(&ctx);   // must not throw: nothing to upgrade any more
            } catch (const ALBackendException &e) {
                std::cerr << "the upgraded pulse cannot be reopened without the flag: "
                          << e.what() << std::endl;
                return 1;
            }
            for (int i = 0; i < STEPS; ++i) {
                if (!read_slice(backend, ctx, 0.1 * i)) {
                    std::cerr << "the upgraded pulse does not read back as the v2 pulse it claims to be"
                              << std::endl;
                    return 1;
                }
            }
            backend.closePulse(&ctx, OPEN_PULSE);
            std::cout << "    opened as v2 with the flag unset, three slices read back\n";
        }

        std::cout << "[OK] v1 pulse rewritten as a plain v2 pulse.\n";
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    unsetenv("BACKEND_ALLOW_AUTO_UPGRADE");
    if (fs::exists(PULSE_DIR)) fs::remove_all(PULSE_DIR);
    return 0;
}
