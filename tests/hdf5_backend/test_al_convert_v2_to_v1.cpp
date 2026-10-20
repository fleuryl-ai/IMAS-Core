// @file  test_al_convert_v2_to_v1.cpp
// @brief Guards BACKEND_HDF5_FORCE_CONVERSION_TO_V1: a pulse written in the v2 (columnar)
//        layout is opened with a 1.0 target, read back through the v2 reader, rewritten as
//        v1, and must then be a plain v1 pulse holding its data, with no v2 table left behind.
//
// The v2 to v1 direction is refused by default -- a v2 file opened while the target version
// is 1.0 throws -- because it is not an edit of the same storage: the v2 tables have to go
// and the pulse re-annotated. The flag lifts that lock for testing, and the backend is
// responsible for making the file actually v1 afterwards: the first write of the session
// drops the IDS storage, the v1 writer lays it down again (file, external link and its own
// "1.0" header), and closePulse() rewrites the master file's version attribute.
//
// Four things therefore have to hold, and the test checks each of them:
//  - the lock is still there when the flag is off (otherwise the flag is not what gates it,
//    and a refused conversion must not have touched the pulse either);
//  - with the flag on, the session reads the v2 tables it was given -- the reader follows the
//    physical version, not the target;
//  - afterwards both the master file and the IDS file are labelled 1.0, and the pulse reopens
//    with the flag unset: a leftover "2.0" would send the next reader to the v2 engine and
//    hide the v1 datasets that were just written;
//  - the v1 datasets are there, with the shapes the v1 writer gives them and the values that
//    were written -- and none of the v2 tables survived next to them.
//
// Note the reads of phase 3 come BEFORE the conversion write: dropping the v2 tables is what
// makes room for the v1 ones, so a conversion session reads the old layout until it writes,
// and finds nothing afterwards.
//
// The last phase reads the datasets through the HDF5 API rather than through the AL. The v1
// reader of this tree cannot serve an AL read of a time-declared AoS written in one GLOBAL
// put: a slice get comes back as a single element and a global get hands back the caller's
// buffer untouched for every AoS element past the first -- and a pulse written natively in 1.0
// behaves exactly the same, so this is the v1 read path, not the conversion. Checking the
// storage directly still catches everything the conversion could get wrong: a label that stayed
// "2.0", a v2 table left in place of the v1 datasets, or a rewrite that lost a slice.
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

const std::string URI = "imas:hdf5?path=./test_db_convert_v2_to_v1";
const std::string PULSE_DIR = "./test_db_convert_v2_to_v1";
const std::string MASTER_FILE = PULSE_DIR + "/master.h5";
const std::string IDS_FILE = PULSE_DIR + "/core_profiles.h5";

const int POINTS = 10;      // radial points of one signal_1d slice
const int STEPS = 3;        // slices in the pulse (t = 0.0, 0.1, 0.2)

// Value of point x of the slice written at time index t_idx.
double expected_value(int t_idx, int x) { return 100.0 + t_idx * 10.0 + x; }
// Maps a time to its time index: the time base is 0.0, 0.1, ... so t*10 is the index.
int time_index_of(double t) { return (int)(t * 10.0 + 0.5); }

void use_v1() { setenv("BACKEND_DEFAULT_WRITE_VERSION", "1.0", 1); }
void use_v2() { setenv("BACKEND_DEFAULT_WRITE_VERSION", "2.0", 1); }
void allow_conversion(bool on) {
    if (on) setenv("BACKEND_HDF5_FORCE_CONVERSION_TO_V1", "true", 1);
    else unsetenv("BACKEND_HDF5_FORCE_CONVERSION_TO_V1");
}

// Writes the whole IDS in one GLOBAL_OP put: the shape a conversion takes, where everything
// that was read is written again in the target layout.
static void write_global(HDF5Backend &backend, DataEntryContext &ctx) {
    backend.openPulse(&ctx, FORCE_CREATE_PULSE);
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
    backend.closePulse(&ctx, FORCE_CREATE_PULSE);
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

// True when the IDS file still holds one of the v2 tables.
static bool v2_table_left(const char *name) {
    hid_t f = H5Fopen(IDS_FILE.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) return false;   // no IDS file at all: nothing left
    const bool present = (H5Lexists(f, name, H5P_DEFAULT) > 0);
    H5Fclose(f);
    return present;
}

// Reads a double dataset of the IDS file: its extent in `shape`, its values in `values`.
// Names carry the v1 encoding of an IDS path ("profiles_1d[]&signal_1d"), link name as stored.
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
        std::cerr << "'" << name << "' is not in " << IDS_FILE << " -- the v1 writer did not "
                  << "lay it down" << std::endl;
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
        std::cout << "\n=== Test: convert a v2 pulse into a v1 pulse ===\n";
        if (fs::exists(PULSE_DIR)) fs::remove_all(PULSE_DIR);
        allow_conversion(false);

        std::cout << "--- Phase 1: write the pulse in v2 ---\n";
        use_v2();
        {
            HDF5Backend backend;
            DataEntryContext ctx(URI);
            write_global(backend, ctx);
        }
        std::cout << "    master file labelled " << file_version(MASTER_FILE)
                  << ", IDS file " << file_version(IDS_FILE) << "\n";
        if (file_version(MASTER_FILE) != "2.0" || file_version(IDS_FILE) != "2.0") {
            std::cerr << "phase 1 did not produce a v2 pulse" << std::endl;
            return 1;
        }

        std::cout << "--- Phase 2: without the flag, opening it as 1.0 must still be refused ---\n";
        {
            use_v1();
            allow_conversion(false);
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
                std::cerr << "a v2 pulse opened with a 1.0 target was accepted although "
                             "BACKEND_HDF5_FORCE_CONVERSION_TO_V1 is not set" << std::endl;
                return 1;
            }
            backend.closePulse(&ctx, OPEN_PULSE);
            std::cout << "    refused as expected\n";
        }
        if (file_version(MASTER_FILE) != "2.0") {
            std::cerr << "a refused conversion changed the pulse label" << std::endl;
            return 1;
        }

        std::cout << "--- Phase 3: with the flag, read as v2 then write as v1 ---\n";
        {
            use_v1();
            allow_conversion(true);
            HDF5Backend backend;
            DataEntryContext ctx(URI);
            backend.openPulse(&ctx, OPEN_PULSE);
            const std::pair<int, int> v = backend.getVersion(&ctx);
            std::cout << "    session version " << v.first << "." << v.second
                      << " (reader follows the file: v2, writer follows the target: v1)\n";

            // Read through the v2 reader -- after the conversion write starts, the v2 tables
            // are gone and this would return nothing.
            for (int i = 0; i < STEPS; ++i) {
                if (!read_slice(backend, ctx, 0.1 * i)) {
                    std::cerr << "the v2 pulse could not be read before converting" << std::endl;
                    return 1;
                }
            }
            std::cout << "    read back through the v2 reader\n";

            // Rewrite it in the v1 layout. The first write of the session drops the IDS
            // storage, so the writer lays a new file down instead of mixing layouts.
            OperationContext opCtx(&ctx, "core_profiles", "", WRITE_OP);   // GLOBAL_OP
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
            backend.closePulse(&ctx, OPEN_PULSE);
        }
        std::cout << "    master file now labelled " << file_version(MASTER_FILE)
                  << ", IDS file " << file_version(IDS_FILE) << "\n";
        if (file_version(MASTER_FILE) != "1.0" || file_version(IDS_FILE) != "1.0") {
            std::cerr << "the converted pulse is labelled " << file_version(MASTER_FILE)
                      << " in the master file and " << file_version(IDS_FILE)
                      << " in the IDS file -- a later open would pick the v2 reader" << std::endl;
            return 1;
        }

        std::cout << "--- Phase 4: reopen without the flag: the pulse is a plain v1 pulse now ---\n";
        {
            use_v1();
            allow_conversion(false);
            HDF5Backend backend;
            DataEntryContext ctx(URI);
            backend.openPulse(&ctx, OPEN_PULSE);
            backend.getVersion(&ctx);   // must not throw: nothing to convert any more
            backend.closePulse(&ctx, OPEN_PULSE);
            std::cout << "    opened as v1 with the flag unset\n";
        }

        std::cout << "--- Phase 5: no v2 table left in the IDS file ---\n";
        for (const char *table : {"index", "paths", "path_table", "data_raw_f64", "list_spans"}) {
            if (v2_table_left(table)) {
                std::cerr << "'" << table << "' is still in " << IDS_FILE
                          << " -- the v2 storage survived the conversion" << std::endl;
                return 1;
            }
        }
        std::cout << "    index / paths / path_table / data_raw_f64 / list_spans: none left\n";

        std::cout << "--- Phase 6: the v1 datasets are there and hold the data ---\n";
        std::vector<double> want_times;
        for (int i = 0; i < STEPS; ++i) want_times.push_back(0.1 * i);
        std::vector<double> want_signals;
        for (int i = 0; i < STEPS; ++i)
            for (int x = 0; x < POINTS; ++x) want_signals.push_back(expected_value(i, x));
        struct { const char *name; std::vector<hsize_t> shape; std::vector<double> values; }
            expected[] = {
                // The v1 writer keeps the IDS name as a group inside its own file, unlike the
                // v2 tables, which sit at the root.
                {"core_profiles/time", {STEPS}, want_times},
                {"core_profiles/profiles_1d[]&signal_1d", {STEPS, POINTS}, want_signals},
            };
        for (const auto &e : expected) {
            if (!check_double_dataset(e.name, e.shape, e.values)) {
                std::cerr << "the converted pulse does not hold '" << e.name
                          << "' the way a v1 pulse should" << std::endl;
                return 1;
            }
        }
        std::cout << "    time {3}, profiles_1d[]&signal_1d {3,10}: shapes and values match\n";

        std::cout << "[OK] v2 pulse converted into a plain v1 pulse.\n";
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    unsetenv("BACKEND_HDF5_FORCE_CONVERSION_TO_V1");
    if (fs::exists(PULSE_DIR)) fs::remove_all(PULSE_DIR);
    return 0;
}
