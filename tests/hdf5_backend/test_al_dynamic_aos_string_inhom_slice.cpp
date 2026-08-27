// @file  test_al_dynamic_aos_string_inhom_slice.cpp
// @brief Dynamic AoS with string and complex signals under inhomogeneous time:
//        global write, slice reads, slice appends, then slice read-back.
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

const std::string URI = "imas:hdf5?path=./test_db_dynamic_aos_string_inhom";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Dynamic AoS with String Signal (Inhomogeneous Time) ===\n" << RESET;

        // 1. Writing
        {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Define the time as inhomogeneous
            int homogeneous_time = 0;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // In inhomogeneous mode, the time base is defined inside the data structure.
            // There is no global 'time' vector.

            // AoS A (static)
            int size_A = 1;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);

            // AoS B (static)
            int size_B = 1;
            ArraystructContext ctxB(&ctxA, "B", "");
            backend.beginArraystructAction(&ctxB, &size_B);

            // AoS C (dynamic)
            int size_C = 10;
            ArraystructContext ctxC(&ctxB, "C", "time"); // C is temporal, its time base is named "time"
            backend.beginArraystructAction(&ctxC, &size_C);

            for(int t=0; t<size_C; ++t) {
                // For each slice of the dynamic AoS, the time value must be provided.
                double current_time = (double)t * 0.1;
                backend.writeData(&ctxC, "time", "", &current_time, alconst::double_data, 0, nullptr);

                // Write the other dynamic signals
                std::string val = "String_" + std::to_string(t);
                int str_dim = 1;
                int str_size[] = {(int)val.length()};
                backend.writeData(&ctxC, "dyn_str", "time", (void*)val.c_str(), alconst::char_data, str_dim, str_size);

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

        // 2. Reading (Slice) - this part must behave exactly like the homogeneous case.
        // The backend must be able to locate the inhomogeneous time base correctly.
        std::vector<double> slice_times = {0.2, 0.5, 0.8}; // Indices 2, 5, 8
        for(double t_req : slice_times) {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

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

            // AoS C is dynamic. In slice mode, its size must be 1.
            int size_C = 0;
            ArraystructContext ctxC(&ctxB, "C", "time");
            backend.beginArraystructAction(&ctxC, &size_C);
            assert(size_C == 1);

            // Validate dyn_str
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
                 std::cout << "  t=" << t_req << " -> dyn_str: " << read_val << " [OK]" << std::endl;
            }
            free(data_ptr);

            // Validate dyn_complex
            void* cplx_ptr = nullptr;
            int cplx_datatype = alconst::complex_data;
            int cplx_dim = 0;
            int cplx_size[H5S_MAX_RANK];
            backend.readData(&ctxC, "dyn_complex", "time", &cplx_ptr, &cplx_datatype, &cplx_dim, cplx_size);
            
            assert(cplx_dim == 0);
            
            std::complex<double>* cplx_data = (std::complex<double>*)cplx_ptr;
            std::complex<double> expected_cplx((double)expected_idx * 10.0, (double)expected_idx * -1.1);

            if (std::abs(cplx_data->real() - expected_cplx.real()) > 1e-9 || std::abs(cplx_data->imag() - expected_cplx.imag()) > 1e-9) {
                 std::cerr << RED << "Mismatch for complex at t=" << t_req << ": expected " << expected_cplx << ", got " << *cplx_data << RESET << std::endl;
                 return 1;
            } else {
                 std::cout << "  t=" << t_req << " -> dyn_complex: " << *cplx_data << " [OK]" << std::endl;
            }
            free(cplx_ptr);

            backend.endAction(&ctxC);
            backend.endAction(&ctxB);
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }
        std::cout << GREEN << "[OK] Slice validation completed.\n" << RESET;

        // 3. Appending Slices
        {
            std::cout << BOLD << "\n=== 3. Appending 5 new slices ===\n" << RESET;
            for(int t = 0; t < 5; ++t) {
                DataEntryContext dataEntryCtx(URI);
                HDF5Backend backend;
                backend.openPulse(&dataEntryCtx, OPEN_PULSE);

                double time_undef = alconst::undefined_time;
                int interpmode_undef = alconst::undefined_interp;
                OperationContext opCtx(&dataEntryCtx, "test_ids", WRITE_OP, alconst::slice_op, time_undef, interpmode_undef);
                backend.beginAction(&opCtx);

                int size_A = 1;
                ArraystructContext ctxA(&opCtx, "A", "");
                backend.beginArraystructAction(&ctxA, &size_A);

                int size_B = 1;
                ArraystructContext ctxB(&ctxA, "B", "");
                backend.beginArraystructAction(&ctxB, &size_B);

                int size_C_append = 1;
                ArraystructContext ctxC(&ctxB, "C", "time");
                backend.beginArraystructAction(&ctxC, &size_C_append);

                int global_t = 10 + t;
                double current_time = (double)global_t * 0.1;
                backend.writeData(&ctxC, "time", "", &current_time, alconst::double_data, 0, nullptr);

                std::string val = "String_Appended_" + std::to_string(global_t);
                int str_dim = 1;
                int str_size[] = {(int)val.length()};
                backend.writeData(&ctxC, "dyn_str", "time", (void*)val.c_str(), alconst::char_data, str_dim, str_size);

                std::complex<double> cplx_val((double)global_t * 10.0, (double)global_t * -1.1);
                backend.writeData(&ctxC, "dyn_complex", "time", &cplx_val, alconst::complex_data, 0, nullptr);
                
                std::cout << "  Appending slice t=" << current_time << " -> " << val << std::endl;

                backend.endAction(&ctxC);
                backend.endAction(&ctxB);
                backend.endAction(&ctxA);
                backend.endAction(&opCtx);
                backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            }
            std::cout << GREEN << "[OK] Appending completed.\n" << RESET;
        }

        // 4. Reading (Appended Slices)
        {
            std::cout << BOLD << "\n=== 4. Reading appended slices ===\n" << RESET;
            std::vector<double> slice_times_appended = {1.1, 1.3}; // Indices 11, 13
            for(double t_req : slice_times_appended) {
                DataEntryContext dataEntryCtx(URI);
                HDF5Backend backend;
                backend.openPulse(&dataEntryCtx, OPEN_PULSE);

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

                int size_C = 0;
                ArraystructContext ctxC(&ctxB, "C", "time");
                backend.beginArraystructAction(&ctxC, &size_C);
                assert(size_C == 1);

                // Validation of dyn_str
                void* data_ptr = nullptr;
                int datatype = alconst::char_data;
                int dim = 0;
                int size[H5S_MAX_RANK];
                backend.readData(&ctxC, "dyn_str", "time", &data_ptr, &datatype, &dim, size);
                
                char* str_data = (char*)data_ptr;
                std::string read_val(str_data);
                
                int expected_idx = (int)(t_req * 10 + 0.001);
                std::string expected_val = "String_Appended_" + std::to_string(expected_idx);

                if (read_val != expected_val) {
                     std::cerr << RED << "Mismatch at t=" << t_req << ": expected " << expected_val << ", got " << read_val << RESET << std::endl;
                     return 1;
                } else {
                     std::cout << "  t=" << t_req << " -> dyn_str: " << read_val << " [OK]" << std::endl;
                }
                free(data_ptr);

                // Validation of dyn_complex
                void* cplx_ptr = nullptr;
                int cplx_datatype = alconst::complex_data;
                int cplx_dim = 0;
                int cplx_size[H5S_MAX_RANK];
                backend.readData(&ctxC, "dyn_complex", "time", &cplx_ptr, &cplx_datatype, &cplx_dim, cplx_size);
                
                assert(cplx_dim == 0);
                
                std::complex<double>* cplx_data = (std::complex<double>*)cplx_ptr;
                std::complex<double> expected_cplx((double)expected_idx * 10.0, (double)expected_idx * -1.1);

                if (std::abs(cplx_data->real() - expected_cplx.real()) > 1e-9 || std::abs(cplx_data->imag() - expected_cplx.imag()) > 1e-9) {
                     std::cerr << RED << "Mismatch for complex at t=" << t_req << ": expected " << expected_cplx << ", got " << *cplx_data << RESET << std::endl;
                     return 1;
                } else {
                     std::cout << "  t=" << t_req << " -> dyn_complex: " << *cplx_data << " [OK]" << std::endl;
                }
                free(cplx_ptr);

                backend.endAction(&ctxC);
                backend.endAction(&ctxB);
                backend.endAction(&ctxA);
                backend.endAction(&opCtx);
                backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            }
            std::cout << GREEN << "[OK] Appended slice validation completed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}
