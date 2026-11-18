// @file  test_read_node_roles.cpp
// @brief Read-side node-role tests for to_improve.md point 3 (conservative pass):
//        fields ending with "time" are not timebases, dynamic AoS timebase names
//        are honoured, and string scalar/list shapes stay explicit.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void remove_uri(const std::string& uri_path) {
    if (fs::exists(uri_path)) fs::remove_all(uri_path);
}

bool approx_equal(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

void test_time_step_is_not_a_timebase() {
    std::cout << "\n=== Test: fields ending with 'time' are not timebases ===\n";

    const std::string db_path = "test_db_read_node_roles_time_step";
    const std::string uri = "imas:hdf5?path=./" + db_path;
    remove_uri(db_path);

    constexpr int n_time = 10;
    std::vector<double> time_values(n_time);
    for (int i = 0; i < n_time; ++i) time_values[i] = i * 0.1;

    {
        DataEntryContext dec(uri);
        HDF5Backend backend;
        backend.openPulse(&dec, FORCE_CREATE_PULSE);

        OperationContext opCtx(&dec, "test_ids", "", WRITE_OP);
        backend.beginAction(&opCtx);

        int homogeneous_time = 1;
        backend.writeData(&opCtx, "ids_properties/homogeneous_time", "",
                          &homogeneous_time, alconst::integer_data, 0, nullptr);

        int time_size[] = {n_time};
        backend.writeData(&opCtx, "time", "time", time_values.data(),
                          alconst::double_data, 1, time_size);

        std::vector<double> time_step(n_time);
        std::vector<double> lifetime(n_time);
        for (int i = 0; i < n_time; ++i) {
            time_step[i] = time_values[i] * 100.0;
            lifetime[i] = time_values[i] * 200.0;
        }
        backend.writeData(&opCtx, "time_step", "time", time_step.data(),
                          alconst::double_data, 1, time_size);
        backend.writeData(&opCtx, "lifetime", "time", lifetime.data(),
                          alconst::double_data, 1, time_size);

        backend.endAction(&opCtx);
        backend.closePulse(&dec, FORCE_CREATE_PULSE);
    }

    {
        DataEntryContext dec(uri);
        HDF5Backend backend;
        backend.openPulse(&dec, OPEN_PULSE);

        const double tmin = 0.2;
        const double tmax = 0.6;
        const std::vector<double> dtime = {0.1};
        OperationContext opCtx(&dec, "test_ids", READ_OP, alconst::timerange_op,
                               tmin, tmax, dtime, alconst::linear_interp);
        backend.beginAction(&opCtx);

        const std::vector<std::pair<std::string, double>> cases = {
            {"time_step", 100.0},
            {"lifetime", 200.0},
        };

        for (const auto& [name, scale] : cases) {
            void* data = nullptr;
            int datatype = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            const int status = backend.readData(&opCtx, name, "time", &data,
                                                &datatype, &dim, size);
            if (status != 1) {
                std::cerr << "readData failed for field '" << name
                          << "' (it was treated as a timebase?)\n";
                throw std::runtime_error("timebase misdetection");
            }
            if (dim != 1 || size[0] != 5) {
                std::cerr << "Unexpected dimensions for '" << name << "': dim="
                          << dim << ", size[0]=" << size[0] << "\n";
                if (data) free(data);
                throw std::runtime_error("bad dimension");
            }

            const double* vals = static_cast<double*>(data);
            for (int i = 0; i < 5; ++i) {
                const double requested_time = tmin + i * dtime[0];
                const double expected = requested_time * scale;
                if (!approx_equal(vals[i], expected, 1e-8)) {
                    std::cerr << "Mismatch for '" << name << "'[" << i
                              << "]: expected " << expected << ", got "
                              << vals[i] << "\n";
                    free(data);
                    throw std::runtime_error("bad value");
                }
            }

            free(data);
        }

        backend.endAction(&opCtx);
        backend.closePulse(&dec, OPEN_PULSE);
    }
}

void test_dynamic_timebase_clock() {
    std::cout << "\n=== Test: dynamic AoS with a non-default timebase ('clock') ===\n";

    const std::string db_path = "test_db_read_node_roles_clock";
    const std::string uri = "imas:hdf5?path=./" + db_path;
    remove_uri(db_path);

    constexpr int n_slices = 4;

    {
        DataEntryContext dec(uri);
        HDF5Backend backend;
        backend.openPulse(&dec, FORCE_CREATE_PULSE);

        OperationContext opCtx(&dec, "test_ids", "", WRITE_OP);
        backend.beginAction(&opCtx);

        int homogeneous_time = 0;
        backend.writeData(&opCtx, "ids_properties/homogeneous_time", "",
                          &homogeneous_time, alconst::integer_data, 0, nullptr);

        int dyn_size = n_slices;
        ArraystructContext dynCtx(&opCtx, "dynamic_aos", "clock");
        backend.beginArraystructAction(&dynCtx, &dyn_size);

        for (int i = 0; i < n_slices; ++i) {
            double clock_value = i * 0.2;
            backend.writeData(&dynCtx, "clock", "", &clock_value,
                              alconst::double_data, 0, nullptr);

            double value = 100.0 + i;
            backend.writeData(&dynCtx, "value", "", &value,
                              alconst::double_data, 0, nullptr);

            double time_step = 1000.0 + i;
            backend.writeData(&dynCtx, "time_step", "", &time_step,
                              alconst::double_data, 0, nullptr);

            if (i < n_slices - 1) dynCtx.nextIndex(1);
        }

        backend.endAction(&dynCtx);
        backend.endAction(&opCtx);
        backend.closePulse(&dec, FORCE_CREATE_PULSE);
    }

    {
        DataEntryContext dec(uri);
        HDF5Backend backend;
        backend.openPulse(&dec, OPEN_PULSE);

        const double requested_time = 0.3;
        OperationContext opCtx(&dec, "test_ids", READ_OP, alconst::slice_op,
                               requested_time, alconst::linear_interp);
        backend.beginAction(&opCtx);

        int dyn_size = 0;
        ArraystructContext dynCtx(&opCtx, "dynamic_aos", "clock");
        backend.beginArraystructAction(&dynCtx, &dyn_size);
        if (dyn_size != 1) {
            std::cerr << "dynamic_aos slice size is " << dyn_size
                      << ", expected 1\n";
            throw std::runtime_error("bad dynamic AoS slice size");
        }

        const std::vector<std::pair<std::string, double>> cases = {
            {"value", 101.5},
            {"time_step", 1001.5},
        };

        for (const auto& [name, expected] : cases) {
            void* data = nullptr;
            int datatype = alconst::double_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            const int status = backend.readData(&dynCtx, name, "", &data,
                                                &datatype, &dim, size);
            if (status != 1) {
                std::cerr << "readData failed for dynamic signal '" << name << "'\n";
                throw std::runtime_error("read failed");
            }
            if (dim != 0) {
                std::cerr << "Unexpected dimension for '" << name
                          << "': dim=" << dim << ", expected 0\n";
                free(data);
                throw std::runtime_error("bad dimension");
            }
            const double value = *static_cast<double*>(data);
            if (!approx_equal(value, expected, 1e-8)) {
                std::cerr << "Mismatch for '" << name << "': expected "
                          << expected << ", got " << value << "\n";
                free(data);
                throw std::runtime_error("bad value");
            }
            free(data);
        }

        backend.endAction(&dynCtx);
        backend.endAction(&opCtx);
        backend.closePulse(&dec, OPEN_PULSE);
    }
}

void test_string_scalar_and_one_element_list_shapes() {
    std::cout << "\n=== Test: string scalar vs one-element list shapes ===\n";

    const std::string db_path = "test_db_read_node_roles_strings";
    const std::string uri = "imas:hdf5?path=./" + db_path;
    remove_uri(db_path);

    {
        DataEntryContext dec(uri);
        HDF5Backend backend;
        backend.openPulse(&dec, FORCE_CREATE_PULSE);

        OperationContext opCtx(&dec, "test_ids", "", WRITE_OP);
        backend.beginAction(&opCtx);

        const std::string scalar = "alpha";
        std::vector<char> scalar_buf(scalar.begin(), scalar.end());
        int scalar_size[] = {static_cast<int>(scalar.size())};
        backend.writeData(&opCtx, "scalar_str", "", scalar_buf.data(),
                          alconst::char_data, 1, scalar_size);

        const std::string item = "beta";
        const int n_strings = 1;
        const int max_len = static_cast<int>(item.size()) + 1;
        std::vector<char> list_buf(n_strings * max_len, '\0');
        std::memcpy(list_buf.data(), item.c_str(), item.size() + 1);
        int list_size[] = {n_strings, max_len};
        backend.writeData(&opCtx, "list_str", "", list_buf.data(),
                          alconst::char_data, 2, list_size);

        backend.endAction(&opCtx);
        backend.closePulse(&dec, FORCE_CREATE_PULSE);
    }

    {
        DataEntryContext dec(uri);
        HDF5Backend backend;
        backend.openPulse(&dec, OPEN_PULSE);

        OperationContext opCtx(&dec, "test_ids", "", READ_OP);
        backend.beginAction(&opCtx);

        void* scalar_data = nullptr;
        int scalar_datatype = alconst::char_data;
        int scalar_dim = 0;
        int scalar_size[H5S_MAX_RANK];
        const int scalar_status = backend.readData(&opCtx, "scalar_str", "",
                                                   &scalar_data, &scalar_datatype,
                                                   &scalar_dim, scalar_size);
        if (scalar_status != 1 || scalar_dim != 1 || scalar_size[0] != 5) {
            std::cerr << "scalar_str returned dim=" << scalar_dim
                      << ", size[0]=" << scalar_size[0] << "\n";
            if (scalar_data) free(scalar_data);
            throw std::runtime_error("bad scalar string shape");
        }
        if (std::strcmp(static_cast<char*>(scalar_data), "alpha") != 0) {
            std::cerr << "scalar_str content mismatch\n";
            free(scalar_data);
            throw std::runtime_error("bad scalar string value");
        }
        free(scalar_data);

        void* list_data = nullptr;
        int list_datatype = alconst::char_data;
        int list_dim = 0;
        int list_size[H5S_MAX_RANK];
        const int list_status = backend.readData(&opCtx, "list_str", "",
                                                 &list_data, &list_datatype,
                                                 &list_dim, list_size);
        if (list_status != 1 || list_dim != 2 || list_size[0] != 1) {
            std::cerr << "list_str returned dim=" << list_dim
                      << ", size[0]=" << list_size[0] << "\n";
            if (list_data) free(list_data);
            throw std::runtime_error("one-element list was not read as a list");
        }
        if (std::strncmp(static_cast<char*>(list_data), "beta", 4) != 0) {
            std::cerr << "list_str content mismatch\n";
            free(list_data);
            throw std::runtime_error("bad string list value");
        }
        free(list_data);

        backend.endAction(&opCtx);
        backend.closePulse(&dec, OPEN_PULSE);
    }
}

} // namespace

int main() {
    try {
        test_time_step_is_not_a_timebase();
        test_dynamic_timebase_clock();
        test_string_scalar_and_one_element_list_shapes();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nAll node-role tests passed!\n";
    return 0;
}
