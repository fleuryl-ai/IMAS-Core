// @file  test_al_core_profiles_put_get_slice.cpp
// @brief Put/get and put-slice/get-slice coverage for core_profiles:
//        1D and 2D signals, string scalars/lists, and static ion AoS.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <cstring>

// For cleaning up test directory
namespace fs = std::filesystem;

// Colors for debug output
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define GREEN   "\033[32m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"

// URI for the test database
const std::string URI = "imas:hdf5?path=./test_db_core_profiles_put_get_slice";

// Global constants for the test data
const int TIME_STEPS_PUT = 3;
const int SPATIAL_SIZE = 5;
const double SLICE_TIME = 0.25;
const int ION_SIZE = 3;
const int ELEMENT_SIZE = 2;
const int STR_1D_LEN = 32;
const int NUM_SLICES_SLICE = 2;
const int GRID_DIM1 = 3;
const int GRID_DIM2 = 4;

// Forward declarations of the functions
void core_profiles_put();
void core_profiles_get();
void core_profiles_putSlice();
void core_profiles_getSlice();

/**
 * @brief Phase 1: Write initial data using GLOBAL_OP (put).
 * This function creates a new pulse file and writes several time steps of data.
 */
void core_profiles_put() {
    std::cout << YELLOW << "\n--- Phase: core_profiles_put() ---" << RESET << std::endl;
    
    DataEntryContext dataEntryCtx(URI);
    HDF5Backend backend;
    backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

    OperationContext opCtx(&dataEntryCtx, "core_profiles", "", WRITE_OP);
    backend.beginAction(&opCtx);

    // Set homogeneous time
    int homogeneous_time = 1;
    backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

    // Write time array
    std::vector<double> time_values(TIME_STEPS_PUT);
    for (int i = 0; i < TIME_STEPS_PUT; ++i) {
        time_values[i] = (double)i * 0.1;
    }
    int time_dim = 1;
    int time_size[] = {TIME_STEPS_PUT};
    backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, time_dim, time_size);

    // Write global_quantities/ip (0D dynamic signal at root - scalar varying in time)
    std::vector<double> ip_values(TIME_STEPS_PUT);
    for (int t = 0; t < TIME_STEPS_PUT; ++t) {
        ip_values[t] = 8000.0 + t * 100.0;
    }
    int ip_dim = 1;
    int ip_size[] = {TIME_STEPS_PUT};
    backend.writeData(&opCtx, "global_quantities/ip", "time", ip_values.data(), alconst::double_data, ip_dim, ip_size);

    // Open profiles_1d Dynamic AoS
    ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
    backend.beginArraystructAction(&profilesCtx, (int*)&TIME_STEPS_PUT);

    for (int t = 0; t < TIME_STEPS_PUT; ++t) {
        // Write a 1D signal for the current time step
        std::vector<double> val(SPATIAL_SIZE);
        for(int x = 0; x < SPATIAL_SIZE; ++x) {
            val[x] = 100.0 + t * 10.0 + x;
        }
        int sig_dim = 1;
        int sig_size[] = {SPATIAL_SIZE};
        backend.writeData(&profilesCtx, "signal_1d", "", val.data(), alconst::double_data, sig_dim, sig_size);

        // Add 0D char_data signal 'string_0d'
        std::string str_val = "String_" + std::to_string(t);
        int str_dim = 1;
        int str_size_arr[] = {(int)str_val.length()};
        backend.writeData(&profilesCtx, "string_0d", "", (void*)str_val.c_str(), alconst::char_data, str_dim, str_size_arr);

        // Add 1D char_data signal 'string_1d'
        int str_1d_dim = 2;
        int str_1d_size_arr[] = {SPATIAL_SIZE, STR_1D_LEN};
        std::vector<char> str_1d_buf(SPATIAL_SIZE * STR_1D_LEN, 0);
        for(int x=0; x<SPATIAL_SIZE; ++x) {
            std::string s = "S_" + std::to_string(t) + "_" + std::to_string(x);
            strncpy(str_1d_buf.data() + x * STR_1D_LEN, s.c_str(), STR_1D_LEN - 1);
        }
        backend.writeData(&profilesCtx, "string_1d", "", str_1d_buf.data(), alconst::char_data, str_1d_dim, str_1d_size_arr);

        // Add static AoS 'ion' with 0D datum 'z_ion'
        ArraystructContext ionCtx(&profilesCtx, "ion", "");
        backend.beginArraystructAction(&ionCtx, (int*)&ION_SIZE);
        for (int i = 0; i < ION_SIZE; ++i) {
            double z_ion_val = 1.0 + t + i;
            backend.writeData(&ionCtx, "z_ion", "", &z_ion_val, alconst::double_data, 0, nullptr);

            ArraystructContext elementCtx(&ionCtx, "element", "");
            backend.beginArraystructAction(&elementCtx, (int*)&ELEMENT_SIZE);
            for (int j = 0; j < ELEMENT_SIZE; ++j) {
                double a_dyn_val = 2.0 + t + i + j;
                backend.writeData(&elementCtx, "a_dyn", "", &a_dyn_val, alconst::double_data, 0, nullptr);
                if (j < ELEMENT_SIZE - 1) elementCtx.nextIndex(1);
            }
            backend.endAction(&elementCtx);
            
            if (i < ION_SIZE - 1) ionCtx.nextIndex(1);
        }
        backend.endAction(&ionCtx);
        
        if (t < TIME_STEPS_PUT - 1) {
            profilesCtx.nextIndex(1);
        }
    }

    backend.endAction(&profilesCtx);
    backend.endAction(&opCtx);
    backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
    std::cout << GREEN << "[OK] core_profiles_put completed." << RESET << std::endl;
}

/**
 * @brief Phase 2: Read and validate the initial data (get).
 */
void core_profiles_get() {
    std::cout << YELLOW << "\n--- Phase: core_profiles_get() ---" << RESET << std::endl;

    DataEntryContext dataEntryCtx(URI);
    HDF5Backend backend;
    backend.openPulse(&dataEntryCtx, OPEN_PULSE);

    OperationContext opCtx(&dataEntryCtx, "core_profiles", "", READ_OP);
    backend.beginAction(&opCtx);

    // Read and validate time array
    void* data = nullptr;
    int type = alconst::double_data;
    int dim = 0;
    int size[H5S_MAX_RANK];
    backend.readData(&opCtx, "time", "time", &data, &type, &dim, size);
    assert(dim == 1);
    assert(size[0] == TIME_STEPS_PUT);
    free(data);

    // Validation global_quantities/ip
    backend.readData(&opCtx, "global_quantities/ip", "time", &data, &type, &dim, size);
    assert(dim == 1);
    assert(size[0] == TIME_STEPS_PUT);
    double* ip_vals = (double*)data;
    for (int t = 0; t < TIME_STEPS_PUT; ++t) {
        double expected = 8000.0 + t * 100.0;
        assert(std::abs(ip_vals[t] - expected) < 1e-9);
    }
    free(data);

    // Read and validate profiles_1d
    ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
    int p_size = 0;
    backend.beginArraystructAction(&profilesCtx, &p_size);
    assert(p_size == TIME_STEPS_PUT);

    for (int t = 0; t < TIME_STEPS_PUT; ++t) {
        backend.readData(&profilesCtx, "signal_1d", "", &data, &type, &dim, size);
        
        assert(dim == 1);
        assert(size[0] == SPATIAL_SIZE);
        
        double* vals = (double*)data;
        for(int x = 0; x < SPATIAL_SIZE; ++x) {
            double expected = 100.0 + t * 10.0 + x;
            assert(std::abs(vals[x] - expected) < 1e-9);
        }
        free(data);

        // Validation string_0d
        void* str_data = nullptr;
        int str_type = alconst::char_data;
        int str_dim = 0;
        int str_size_arr[H5S_MAX_RANK];
        backend.readData(&profilesCtx, "string_0d", "", &str_data, &str_type, &str_dim, str_size_arr);
        assert(str_dim == 1);
        std::string read_str((char*)str_data);
        std::string expected_str = "String_" + std::to_string(t);
        assert(read_str == expected_str);
        delete[] (char*)str_data;

        // Validation string_1d
        void* str_1d_data = nullptr;
        int str_1d_rdim = 0;
        int str_1d_rsize[H5S_MAX_RANK];
        backend.readData(&profilesCtx, "string_1d", "", &str_1d_data, &str_type, &str_1d_rdim, str_1d_rsize);
        assert(str_1d_rdim == 2);
        assert(str_1d_rsize[0] == SPATIAL_SIZE);
        int r_len = str_1d_rsize[1];
        char* r_buf = (char*)str_1d_data;
        for(int x=0; x<SPATIAL_SIZE; ++x) {
            std::string read_s(r_buf + x * r_len);
            std::string expected_s = "S_" + std::to_string(t) + "_" + std::to_string(x);
            assert(read_s == expected_s);
        }
        delete[] (char*)str_1d_data;

        // Validate static AoS 'ion'
        ArraystructContext ionCtx(&profilesCtx, "ion", "");
        int ion_size = 0;
        backend.beginArraystructAction(&ionCtx, &ion_size);
        assert(ion_size == ION_SIZE);
        for (int i = 0; i < ion_size; ++i) {
            void* z_data = nullptr;
            backend.readData(&ionCtx, "z_ion", "", &z_data, &type, &dim, size);
            assert(dim == 0);
            assert(std::abs(*(double*)z_data - (1.0 + t + i)) < 1e-9);
            free(z_data);

            ArraystructContext elementCtx(&ionCtx, "element", "");
            int element_size = 0;
            backend.beginArraystructAction(&elementCtx, &element_size);
            assert(element_size == ELEMENT_SIZE);
            for (int j = 0; j < element_size; ++j) {
                void* a_data = nullptr;
                backend.readData(&elementCtx, "a_dyn", "", &a_data, &type, &dim, size);
                assert(dim == 0);
                assert(std::abs(*(double*)a_data - (2.0 + t + i + j)) < 1e-9);
                free(a_data);
                if (j < element_size - 1) elementCtx.nextIndex(1);
            }
            backend.endAction(&elementCtx);
            
            if (i < ion_size - 1) ionCtx.nextIndex(1);
        }
        backend.endAction(&ionCtx);
        
        if (t < TIME_STEPS_PUT - 1) {
            profilesCtx.nextIndex(1);
        }
    }

    backend.endAction(&profilesCtx);
    backend.endAction(&opCtx);
    backend.closePulse(&dataEntryCtx, OPEN_PULSE);
    std::cout << GREEN << "[OK] core_profiles_get validation passed." << RESET << std::endl;
}

/**
 * @brief Phase 3: Append a new slice of data using SLICE_OP (putSlice).
 */
void core_profiles_putSlice() {
    std::cout << YELLOW << "\n--- Phase: core_profiles_putSlice() ---" << RESET << std::endl;

    DataEntryContext dataEntryCtx(URI);
    HDF5Backend backend;
    backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);
    
    for (int k = 0; k < NUM_SLICES_SLICE; ++k) {
        double current_time = SLICE_TIME + k * 0.1;

        // Operation context for writing a single slice
        OperationContext opCtx(&dataEntryCtx, "core_profiles", WRITE_OP, alconst::slice_op, current_time, alconst::undefined_interp);
        backend.beginAction(&opCtx);

        if (k == 0) {
            // Set homogeneous time (required since we recreate the file on first slice)
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);
        }

        // Write the time value for the new slice
        backend.writeData(&opCtx, "time", "time", &current_time, alconst::double_data, 0, nullptr);

        // Write global_quantities/ip for the new slice (scalar)
        double ip_val = 9000.0 + k * 100.0;
        int ip_dim_slice = 0;
        int* ip_size_slice = nullptr;
        backend.writeData(&opCtx, "global_quantities/ip", "time", &ip_val, alconst::double_data, ip_dim_slice, ip_size_slice);

        // Open profiles_1d AoS for writing one slice
        int p_size = 1;
        ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
        backend.beginArraystructAction(&profilesCtx, &p_size);

        // Write the 1D signal data for the new slice
        std::vector<double> val(SPATIAL_SIZE);
        for(int x = 0; x < SPATIAL_SIZE; ++x) {
            val[x] = 200.0 + k * 10.0 + x;
        }
        int sig_dim = 1;
        int sig_size[] = {SPATIAL_SIZE};
        backend.writeData(&profilesCtx, "signal_1d", "time", val.data(), alconst::double_data, sig_dim, sig_size);

        // Write string_0d for the new slice
        std::string str_val = "String_slice_" + std::to_string(k);
        int str_dim = 1;
        int str_size[] = {(int)str_val.length()};
        backend.writeData(&profilesCtx, "string_0d", "time", (void*)str_val.c_str(), alconst::char_data, str_dim, str_size);

        // Write string_1d for the new slice
        int str_1d_dim = 2;
        int str_1d_size_arr[] = {SPATIAL_SIZE, STR_1D_LEN};
        std::vector<char> str_1d_buf(SPATIAL_SIZE * STR_1D_LEN, 0);
        for(int x=0; x<SPATIAL_SIZE; ++x) {
            std::string s = "S_slice_" + std::to_string(k) + "_" + std::to_string(x);
            strncpy(str_1d_buf.data() + x * STR_1D_LEN, s.c_str(), STR_1D_LEN - 1);
        }
        backend.writeData(&profilesCtx, "string_1d", "time", str_1d_buf.data(), alconst::char_data, str_1d_dim, str_1d_size_arr);

        // Write ion AoS for the new slice
        int ion_size = ION_SIZE;
        ArraystructContext ionCtx(&profilesCtx, "ion", "");
        backend.beginArraystructAction(&ionCtx, &ion_size);
        for (int i = 0; i < ion_size; ++i) {
            double z_ion_val = 300.0 + k * 10.0 + i;
            backend.writeData(&ionCtx, "z_ion", "", &z_ion_val, alconst::double_data, 0, nullptr);

            int element_size = ELEMENT_SIZE;
            ArraystructContext elementCtx(&ionCtx, "element", "");
            backend.beginArraystructAction(&elementCtx, &element_size);
            for (int j = 0; j < element_size; ++j) {
                double a_dyn_val = 400.0 + k * 10.0 + i + j;
                backend.writeData(&elementCtx, "a_dyn", "", &a_dyn_val, alconst::double_data, 0, nullptr);
                if (j < element_size - 1) elementCtx.nextIndex(1);
            }
            backend.endAction(&elementCtx);
            
            if (i < ion_size - 1) ionCtx.nextIndex(1);
        }
        backend.endAction(&ionCtx);

        backend.endAction(&profilesCtx);

        // profiles_2D (Dynamic AoS)
        int p2d_size = 1;
        ArraystructContext profiles2DCtx(&opCtx, "profiles_2D", "time");
        backend.beginArraystructAction(&profiles2DCtx, &p2d_size);

        // ion (Static AoS inside Dynamic AoS)
        int ion2d_size = 2;
        ArraystructContext ion2DCtx(&profiles2DCtx, "ion", "");
        backend.beginArraystructAction(&ion2DCtx, &ion2d_size);
        
        for (int i = 0; i < ion2d_size; ++i) {
            std::vector<double> sig2d(GRID_DIM1 * GRID_DIM2);
            for(int x=0; x<GRID_DIM1; ++x) {
                for(int y=0; y<GRID_DIM2; ++y) {
                    sig2d[x*GRID_DIM2 + y] = 500.0 + k*100.0 + i*10.0 + x*GRID_DIM2 + y;
                }
            }
            int sig2d_dim = 2;
            int sig2d_size[] = {GRID_DIM1, GRID_DIM2};
            backend.writeData(&ion2DCtx, "signal_2d", "time", sig2d.data(), alconst::double_data, sig2d_dim, sig2d_size);
            
            if (i < ion2d_size - 1) ion2DCtx.nextIndex(1);
        }
        backend.endAction(&ion2DCtx);
        backend.endAction(&profiles2DCtx);

        backend.endAction(&opCtx);
    }
    backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
    std::cout << GREEN << "[OK] core_profiles_putSlice completed." << RESET << std::endl;
}

/**
 * @brief Phase 4: Read and validate the appended slice (getSlice).
 */
void core_profiles_getSlice() {
    std::cout << YELLOW << "\n--- Phase: core_profiles_getSlice() ---" << RESET << std::endl;

    DataEntryContext dataEntryCtx(URI);
    HDF5Backend backend;
    backend.openPulse(&dataEntryCtx, OPEN_PULSE);
    
    for (int k = 0; k < NUM_SLICES_SLICE; ++k) {
        double current_slice_time = SLICE_TIME + k * 0.1;

        // Operation context for reading a single slice
        OperationContext opCtx(&dataEntryCtx, "core_profiles", READ_OP, alconst::slice_op, current_slice_time, alconst::closest_interp);
        backend.beginAction(&opCtx);

        // Validation global_quantities/ip (Slice)
        void* data = nullptr;
        int type = alconst::double_data;
        int dim = 0;
        int size[H5S_MAX_RANK];
        backend.readData(&opCtx, "global_quantities/ip", "time", &data, &type, &dim, size);
        printf("dim=%d size[0]=%d\n", dim, size[0]);
        // Root scalar slice is promoted to 1D array of size 1
        assert(dim == 1);
        assert(size[0] == 1);
        double* ip_vals_slice = (double*)data;
        double expected = 9000.0 + k * 100.0;
        assert(std::abs(ip_vals_slice[0] - expected) < 1e-9);
        free(data);

        // Open profiles_1d AoS and check size is 1 for slice mode
        ArraystructContext profilesCtx(&opCtx, "profiles_1d", "time");
        int p_size = 0;
        backend.beginArraystructAction(&profilesCtx, &p_size);
        assert(p_size == 1);

        // Read and validate the 1D signal data for the slice
        data = nullptr;
        backend.readData(&profilesCtx, "signal_1d", "time", &data, &type, &dim, size);

        assert(dim == 1);
        assert(size[0] == SPATIAL_SIZE);
        
        double* vals = (double*)data;
        for(int x = 0; x < SPATIAL_SIZE; ++x) {
            double expected = 200.0 + k * 10.0 + x;
            assert(std::abs(vals[x] - expected) < 1e-9);
        }
        free(data);

        // Validation string_0d (Slice)
        void* str_data = nullptr;
        int str_type = alconst::char_data;
        int str_dim = 0;
        int str_size_arr[H5S_MAX_RANK];
        backend.readData(&profilesCtx, "string_0d", "time", &str_data, &str_type, &str_dim, str_size_arr);
        assert(str_dim == 1);
        std::string read_str_slice((char*)str_data);
        std::string expected_str_slice = "String_slice_" + std::to_string(k);
        assert(read_str_slice == expected_str_slice);
        free(str_data);

        // Validation string_1d (Slice)
        void* str_1d_data = nullptr;
        int str_1d_rdim = 0;
        int str_1d_rsize[H5S_MAX_RANK];
        backend.readData(&profilesCtx, "string_1d", "time", &str_1d_data, &str_type, &str_1d_rdim, str_1d_rsize);
        assert(str_1d_rdim == 2);
        assert(str_1d_rsize[0] == SPATIAL_SIZE);
        int r_len_slice = str_1d_rsize[1];
        char* r_buf_slice = (char*)str_1d_data;
        for(int x=0; x<SPATIAL_SIZE; ++x) {
            std::string read_s(r_buf_slice + x * r_len_slice);
            std::string expected_s = "S_slice_" + std::to_string(k) + "_" + std::to_string(x);
            assert(read_s == expected_s);
        }
        free(str_1d_data);

        // Validation ion AoS (Slice)
        ArraystructContext ionCtx(&profilesCtx, "ion", "");
        int ion_size = 0;
        backend.beginArraystructAction(&ionCtx, &ion_size);
        assert(ion_size == ION_SIZE);
        for (int i = 0; i < ion_size; ++i) {
            backend.readData(&ionCtx, "z_ion", "", &data, &type, &dim, size);
            assert(dim == 0);
            assert(std::abs(*(double*)data - (300.0 + k * 10.0 + i)) < 1e-9);
            free(data);

            ArraystructContext elementCtx(&ionCtx, "element", "");
            int element_size = 0;
            backend.beginArraystructAction(&elementCtx, &element_size);
            assert(element_size == ELEMENT_SIZE);
            for (int j = 0; j < element_size; ++j) {
                backend.readData(&elementCtx, "a_dyn", "", &data, &type, &dim, size);
                assert(dim == 0);
                assert(std::abs(*(double*)data - (400.0 + k * 10.0 + i + j)) < 1e-9);
                free(data);
                if (j < element_size - 1) elementCtx.nextIndex(1);
            }
            backend.endAction(&elementCtx);
            
            if (i < ion_size - 1) ionCtx.nextIndex(1);
        }
        backend.endAction(&ionCtx);

        backend.endAction(&profilesCtx);

        // Validation profiles_2D
        ArraystructContext profiles2DCtx(&opCtx, "profiles_2D", "time");
        int p2d_size = 0;
        backend.beginArraystructAction(&profiles2DCtx, &p2d_size);
        assert(p2d_size == 1);

        ArraystructContext ion2DCtx(&profiles2DCtx, "ion", "");
        int ion2d_size = 0;
        backend.beginArraystructAction(&ion2DCtx, &ion2d_size);
        assert(ion2d_size == 2);

        for (int i = 0; i < ion2d_size; ++i) {
            void* data_2d = nullptr;
            int type_2d = alconst::double_data;
            int dim_2d = 0;
            int size_2d[H5S_MAX_RANK];
            backend.readData(&ion2DCtx, "signal_2d", "time", &data_2d, &type_2d, &dim_2d, size_2d);
            
            assert(dim_2d == 2);
            assert(size_2d[0] == GRID_DIM1);
            assert(size_2d[1] == GRID_DIM2);
            
            double* vals_2d = (double*)data_2d;
            for(int x=0; x<GRID_DIM1; ++x) {
                for(int y=0; y<GRID_DIM2; ++y) {
                    double expected = 500.0 + k*100.0 + i*10.0 + x*GRID_DIM2 + y;
                    assert(std::abs(vals_2d[x*GRID_DIM2 + y] - expected) < 1e-9);
                }
            }
            free(data_2d);
            
            if (i < ion2d_size - 1) ion2DCtx.nextIndex(1);
        }
        backend.endAction(&ion2DCtx);
        backend.endAction(&profiles2DCtx);

        backend.endAction(&opCtx);
    }
    backend.closePulse(&dataEntryCtx, OPEN_PULSE);
    std::cout << GREEN << "[OK] core_profiles_getSlice validation passed." << RESET << std::endl;
}

int main(int argc, char** argv) {
    // Disable AL validation as we fill all fields, including mutually exclusive coordinates
    putenv((char*)"IMAS_AL_DISABLE_VALIDATE=1");

    try {
        // Cleanup previous test run
        if (fs::exists("./test_db_core_profiles_put_get_slice")) {
            fs::remove_all("./test_db_core_profiles_put_get_slice");
        }

        printf("---> Using backend : HDF5_BACKEND\n");
        printf("Processing IDS: core_profiles\n");
        DataEntryContext dataEntryCtx(URI);
        HDF5Backend backend;
        backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);
        auto version = backend.getVersion(&dataEntryCtx);
            if (version == std::make_pair(1,0)) {
                backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
                return 0;
            } 
        //backend.closePulse(&dataEntryCtx, OPEN_PULSE);

        // Execute the test phases
        core_profiles_put();
        core_profiles_get();
        core_profiles_putSlice();
        core_profiles_getSlice();

        std::cout << BOLD << GREEN << "\nALL PHASES COMPLETED SUCCESSFULLY!\n" << RESET << std::endl;

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception caught: " << e.what() << RESET << std::endl;
        return 1;
    }

    return 0;
}