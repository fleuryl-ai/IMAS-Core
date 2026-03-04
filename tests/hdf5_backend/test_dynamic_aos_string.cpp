// test_dynamic_aos_string.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <complex>
#include <filesystem>

namespace fs = std::filesystem;

// Colors for debug
#define RESET ""
#define BOLD ""
#define GREEN ""
#define RED ""

const std::string URI = "imas:hdf5?path=./test_db_dynamic_aos_string";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Dynamic AoS with String Signal ===\n" << RESET;

        // 1. Writing
        {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Timebase
            int time_dim = 10;
            std::vector<double> time_values(time_dim);
            for(int i=0; i<time_dim; ++i) time_values[i] = (double)i * 0.1;
            int time_arr_size[] = {time_dim};
            backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, 1, time_arr_size);

            // AoS A (Static)
            int size_A = 1;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);

            // AoS B (Static)
            int size_B = 1;
            ArraystructContext ctxB(&ctxA, "B", "");
            backend.beginArraystructAction(&ctxB, &size_B);

            // AoS C (Dynamic)
            int size_C = 10;
            ArraystructContext ctxC(&ctxB, "C", "time");
            backend.beginArraystructAction(&ctxC, &size_C);

            for(int t=0; t<size_C; ++t) {
                std::string val = "String_" + std::to_string(t);
                int str_dim = 1;
                int str_size[] = {(int)val.length()};
                // Writing scalar string inside dynamic AoS
                backend.writeData(&ctxC, "dyn_str", "time", (void*)val.c_str(), alconst::char_data, str_dim, str_size);

                // Writing scalar complex inside dynamic AoS
                std::complex<double> cplx_val((double)t * 10.0, (double)t * -1.1);
                backend.writeData(&ctxC, "dyn_complex", "time", &cplx_val, alconst::complex_data, 0, nullptr);
                
                if (t < size_C - 1) ctxC.nextIndex(1);
            }
            backend.endAction(&ctxC);
            backend.endAction(&ctxB);
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // 2. Reading (Slice)
        std::vector<double> slice_times = {0.2, 0.5, 0.8}; // Indices 2, 5, 8
        for(double t_req : slice_times) {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            // Use closest interp for strings to ensure we get the exact string for the time step
            int interpmode = alconst::closest_interp; 
            
            OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP, alconst::slice_op, t_req, interpmode);
            backend.beginAction(&opCtx);

            int size_A = 0;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);
            assert(size_A == 1);

            int size_B = 0;
            ArraystructContext ctxB(&ctxA, "B", "");
            backend.beginArraystructAction(&ctxB, &size_B);
            assert(size_B == 1);

            // AoS C is dynamic. In slice mode, size should be 1 (the slice).
            int size_C = 0;
            ArraystructContext ctxC(&ctxB, "C", "time");
            backend.beginArraystructAction(&ctxC, &size_C);
            assert(size_C == 1);

            void* data_ptr = nullptr;
            int datatype = alconst::char_data;
            int dim = 0;
            int size[H5S_MAX_RANK];

            backend.readData(&ctxC, "dyn_str", "time", &data_ptr, &datatype, &dim, size);
            
            char* str_data = (char*)data_ptr;
            std::string read_val(str_data);
            
            int expected_idx = (int)(t_req * 10 + 0.001);
            std::string expected_val = "String_" + std::to_string(expected_idx);

            if (read_val != expected_val) {
                 std::cerr << RED << "Mismatch at t=" << t_req << ": expected " << expected_val << ", got " << read_val << RESET << std::endl;
                 return 1;
            } else {
                 std::cout << "  t=" << t_req << " -> " << read_val << " [OK]" << std::endl;
            }
            free(data_ptr);

            // Read and validate complex signal
            void* cplx_ptr = nullptr;
            int cplx_datatype = alconst::complex_data;
            int cplx_dim = 0;
            int cplx_size[H5S_MAX_RANK];
            backend.readData(&ctxC, "dyn_complex", "time", &cplx_ptr, &cplx_datatype, &cplx_dim, cplx_size);
            printf("Read complex at t=%f: dim=%d\n", t_req, cplx_dim);
            //assert(cplx_dim == 0); // Scalar complex
            
            std::complex<double>* cplx_data = (std::complex<double>*)cplx_ptr;
            std::complex<double> expected_cplx((double)expected_idx * 10.0, (double)expected_idx * -1.1);

            if (std::abs(cplx_data->real() - expected_cplx.real()) > 1e-9 || std::abs(cplx_data->imag() - expected_cplx.imag()) > 1e-9) {
                 std::cerr << RED << "Mismatch for complex at t=" << t_req << ": expected " << expected_cplx << ", got " << *cplx_data << RESET << std::endl;
                 return 1;
            } else {
                 std::cout << "  t=" << t_req << " -> " << *cplx_data << " [OK]" << std::endl;
            }
            free(cplx_ptr);


            backend.endAction(&ctxC);
            backend.endAction(&ctxB);
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }
        std::cout << GREEN << "[OK] Slice validation completed.\n" << RESET;

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}
