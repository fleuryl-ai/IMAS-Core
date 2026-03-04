// test_homog_2d.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

// Colors for debug
#define RESET ""
#define BOLD ""
#define GREEN ""
#define RED ""

const std::string URI = "imas:hdf5?path=./test_db_homog_2d";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Homogeneous Timebase with 2D signal ===\n" << RESET;

        const int time_dim_size = 5;
        const int spatial_dim_size = 4;

        // 1. Writing
        {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Set homogeneous_time = 1 (Homogeneous)
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Static char_data 'ids_properties/comment'
            std::string comment = "Test Homog 2D";
            int comment_dim = 1;
            int comment_size[] = {(int)comment.length()};
            backend.writeData(&opCtx, "ids_properties/comment", "", (void*)comment.c_str(), alconst::char_data, comment_dim, comment_size);

            // Write root timebase 'time' (5 points)
            std::vector<double> time_values(time_dim_size);
            for (int k = 0; k < time_dim_size; ++k) time_values[k] = (double)k * 0.1;
            int time_dim = 1;
            int time_size[] = {time_dim_size};
            backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, time_dim, time_size);

            int size_A = 3;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);

            for (int i = 0; i < size_A; ++i) {
                int size_B = 3;
                ArraystructContext ctxB(&ctxA, "B", "");
                backend.beginArraystructAction(&ctxB, &size_B);

                for (int j = 0; j < size_B; ++j) {
                    // Dynamic 2D signal in B (5 slices of 1D array of size 4)
                    std::vector<double> dyn_sig_2d(spatial_dim_size * time_dim_size);
                    for(int t=0; t<time_dim_size; ++t) {
                        for(int x=0; x<spatial_dim_size; ++x) {
                            dyn_sig_2d[t * spatial_dim_size + x] = (double)(i * 1000 + j * 100 + t * 10 + x);
                        }
                    }
                    
                    int dyn_sig_dim = 2;
                    int dyn_sig_size[] = {spatial_dim_size, time_dim_size};
                    backend.writeData(&ctxB, "dyn_sig", "time", dyn_sig_2d.data(), alconst::double_data, dyn_sig_dim, dyn_sig_size);

                    // Dynamic char* signal (scalar string)
                    /*for (int t = 0; t < time_dim_size; ++t) {
                        std::string str_val = "Start_" + std::to_string(t);
                        int str_dim_scalar = 1;
                        int str_size_scalar[] = {(int)str_val.length()};
                        backend.writeData(&ctxB, "dyn_str_scalar", "time", (void*)str_val.c_str(), alconst::char_data, str_dim_scalar, str_size_scalar);
                    }*/

                    // Dynamic char* signal (array of strings)
                    // We write 5 strings corresponding to the 5 time steps.
                    // We must use a flat buffer to be consistent with AL conventions.
                    const int max_len = 10;
                    char flat_strs[time_dim_size][max_len];
                    memset(flat_strs, 0, sizeof(flat_strs));
                    for(int t=0; t<time_dim_size; ++t) {
                        std::string s = "S_" + std::to_string(t);
                        strncpy(flat_strs[t], s.c_str(), max_len - 1);
                    }
                    int str_dim = 2;
                    int str_size[] = {time_dim_size, max_len};
                    backend.writeData(&ctxB, "dyn_strs", "time", (void*)flat_strs, alconst::char_data, str_dim, str_size);

                    int size_C = 1;
                    ArraystructContext ctxC(&ctxB, "C", "");
                    backend.beginArraystructAction(&ctxC, &size_C);
                    backend.endAction(&ctxC);

                    if (j < size_B - 1) ctxB.nextIndex(1);
                }
                backend.endAction(&ctxB);
                if (i < size_A - 1) ctxA.nextIndex(1);
            }
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // 2. Reading
        {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", READ_OP);
            backend.beginAction(&opCtx);

            int size_A = 0;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);
            assert(size_A == 3);

            // Read static comment
            void* comment_ptr = nullptr;
            int comment_datatype = alconst::char_data;
            int comment_dim_out = 0;
            int comment_size_out[H5S_MAX_RANK];
            backend.readData(&opCtx, "ids_properties/comment", "", &comment_ptr, &comment_datatype, &comment_dim_out, comment_size_out);
            assert(comment_dim_out == 1);
            assert(std::string((char*)comment_ptr) == "Test Homog 2D");
            free(comment_ptr);

            for (int i = 0; i < size_A; ++i) {
                int size_B = 0;
                ArraystructContext ctxB(&ctxA, "B", "");
                backend.beginArraystructAction(&ctxB, &size_B);
                assert(size_B == 3);

                for (int j = 0; j < size_B; ++j) {
                    void* data_ptr = nullptr;
                    int datatype = alconst::double_data;
                    int dim = 0;
                    int size[H5S_MAX_RANK];

                    // Read dyn_sig
                    backend.readData(&ctxB, "dyn_sig", "time", &data_ptr, &datatype, &dim, size);
                    
                    assert(dim == 2);
                    assert(size[0] == spatial_dim_size);
                    assert(size[1] == time_dim_size);
                    
                    double* sig_vals = (double*)data_ptr;
                    for(int t=0; t<time_dim_size; ++t) {
                        for(int x=0; x<spatial_dim_size; ++x) {
                            double expected = (double)(i * 1000 + j * 100 + t * 10 + x);
                            // Data is stored as [time][space] in memory after reading
                            assert(std::abs(sig_vals[t * spatial_dim_size + x] - expected) < 1e-9);
                        }
                    }
                    free(data_ptr);

                    // Read dyn_strs (initial 5 values)
                    void* str_ptr = nullptr;
                    int str_datatype = alconst::char_data;
                    int str_dim_out = 0;
                    int str_size_out[H5S_MAX_RANK];
                    backend.readData(&ctxB, "dyn_strs", "time", &str_ptr, &str_datatype, &str_dim_out, str_size_out);
                    
                    assert(str_dim_out == 2); // 1D array of strings is dim 2 in AL (count, max_len)
                    assert(str_size_out[0] == 5);
                    
                    char* char_data = (char*)str_ptr;
                    int max_len = str_size_out[1];
                    for(int k=0; k<5; ++k) {
                        std::string val(char_data + k * max_len);
                        assert(val == "S_" + std::to_string(k));
                    }
                    delete[] char_data;

                    // Read dyn_str_scalar (initial 5 values)
                    /*void* str_scalar_ptr = nullptr;
                    int str_scalar_datatype = alconst::char_data;
                    int str_scalar_dim_out = 0;
                    int str_scalar_size_out[H5S_MAX_RANK];
                    backend.readData(&ctxB, "dyn_str_scalar", "time", &str_scalar_ptr, &str_scalar_datatype, &str_scalar_dim_out, str_scalar_size_out);
                    printf("Read dyn_str_scalar: dim=%d, size[0]=%d\n", str_scalar_dim_out, str_scalar_size_out[0]);
                    assert(str_scalar_dim_out == 2); // 1D array of strings is dim 2 in AL
                    assert(str_scalar_size_out[1] == 5); // 5 strings
                    
                    char* char_scalar_data = (char*)str_scalar_ptr;
                    int max_scalar_len = str_scalar_size_out[0];
                    for(int k=0; k<5; ++k) {
                        char* current_str = char_scalar_data + k * max_scalar_len;
                        std::string val(current_str);
                        assert(val == "Start_" + std::to_string(k));
                    }
                    delete[] char_scalar_data;*/

                    if (j < size_B - 1) ctxB.nextIndex(1);
                }
                backend.endAction(&ctxB);
                if (i < size_A - 1) ctxA.nextIndex(1);
            }
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Reading validation completed.\n" << RESET;
        }


        // 3. Appending Slices
        std::cout << BOLD << "\n=== 3. Appending 5 Slices (Slice-by-Slice) ===\n" << RESET;
        for (int k = 0; k < 5; ++k) {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            printf("Appending slice %d (t=%.1f)...\n", k, 0.5 + k * 0.1);

            double time_val = 0.5 + k * 0.1; // 0.5, 0.6, ...
            
            double time_undef = alconst::undefined_time;
            int interpmode_undef = alconst::undefined_interp;
            OperationContext opCtx(&dataEntryCtx, "test_ids", WRITE_OP,
                                   alconst::slice_op, time_undef, interpmode_undef);
            
            backend.beginAction(&opCtx);

            // Write time scalar for this slice
            int time_dim = 1;
            int time_size[] = {1};
            backend.writeData(&opCtx, "time", "time", &time_val, alconst::double_data, time_dim, time_size);

            int size_A = 3;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);

            for (int i = 0; i < size_A; ++i) {
                int size_B = 3;
                ArraystructContext ctxB(&ctxA, "B", "");
                backend.beginArraystructAction(&ctxB, &size_B);

                for (int j = 0; j < size_B; ++j) {
                    // Dynamic 2D signal slice in B
                    // It's a 1D vector of size 4 for this time step
                    std::vector<double> dyn_sig_slice(spatial_dim_size);
                    int t_idx = time_dim_size + k; // 5, 6, ...
                    for(int x=0; x<spatial_dim_size; ++x) {
                        dyn_sig_slice[x] = (double)(i * 1000 + j * 100 + t_idx * 10 + x);
                    }
                    
                    // We write it as a 2D array [4, 1] to indicate 1 slice of size 4
                    int dyn_sig_dim = 2;
                    int dyn_sig_size[] = {spatial_dim_size, 1};
                    backend.writeData(&ctxB, "dyn_sig", "time", dyn_sig_slice.data(), alconst::double_data, dyn_sig_dim, dyn_sig_size);

                    // Dynamic char* signal slice
                    std::string str_slice_scalar = "Slice_" + std::to_string(5 + k);
                    int str_dim_scalar = 1;
                    int str_size_scalar[] = {(int)str_slice_scalar.length()};
                    backend.writeData(&ctxB, "dyn_str_scalar", "time", (void*)str_slice_scalar.c_str(), alconst::char_data, str_dim_scalar, str_size_scalar);

                    // Dynamic char* signal slice
                    // We write 1 string for this time step
                    std::string str_slice = "S_" + std::to_string(5 + k);
                    int str_dim = 1; // Scalar string for this slice
                    int str_size[] = {(int)str_slice.length()};
                    backend.writeData(&ctxB, "dyn_strs", "time", (void*)str_slice.c_str(), alconst::char_data, str_dim, str_size);

                    if (j < size_B - 1) ctxB.nextIndex(1);
                }
                backend.endAction(&ctxB);
                if (i < size_A - 1) ctxA.nextIndex(1);
            }
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Slice " << k << " (t=" << time_val << ") appended.\n" << RESET;
        }

        // 4. Reading Validation (Global)
        std::cout << BOLD << "\n=== 4. Reading Validation (Global) ===\n" << RESET;
        {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", READ_OP);
            backend.beginAction(&opCtx);

            int size_A = 0;
            ArraystructContext ctxA(&opCtx, "A", "");
            backend.beginArraystructAction(&ctxA, &size_A);
            assert(size_A == 3);

            for (int i = 0; i < size_A; ++i) {
                int size_B = 0;
                ArraystructContext ctxB(&ctxA, "B", "");
                backend.beginArraystructAction(&ctxB, &size_B);
                assert(size_B == 3);

                for (int j = 0; j < size_B; ++j) {
                    void* data_ptr = nullptr;
                    int datatype = alconst::double_data;
                    int dim = 0;
                    int size[H5S_MAX_RANK];

                    // Read dyn_sig
                    backend.readData(&ctxB, "dyn_sig", "time", &data_ptr, &datatype, &dim, size);
                    
                    assert(dim == 2);
                    assert(size[0] == spatial_dim_size);
                    assert(size[1] == time_dim_size + 5); // 5 initial + 5 appended
                    
                    double* sig_vals = (double*)data_ptr;
                    for(int t=0; t<size[1]; ++t) {
                        for(int x=0; x<spatial_dim_size; ++x) {
                            double expected = (double)(i * 1000 + j * 100 + t * 10 + x);
                            // Data is stored as [time][space] in memory after reading
                            assert(std::abs(sig_vals[t * spatial_dim_size + x] - expected) < 1e-9);
                        }
                    }
                    free(data_ptr);

                    // Read dyn_strs (10 values: 5 initial + 5 appended)
                    void* str_ptr = nullptr;
                    int str_datatype = alconst::char_data;
                    int str_dim_out = 0;
                    int str_size_out[H5S_MAX_RANK];
                    backend.readData(&ctxB, "dyn_strs", "time", &str_ptr, &str_datatype, &str_dim_out, str_size_out);
                    
                    assert(str_dim_out == 2);
                    assert(str_size_out[0] == 10); // 10 strings total
                    
                    char* char_data = (char*)str_ptr;
                    int max_len = str_size_out[1];
                    for(int k=0; k<10; ++k) {
                        std::string val(char_data + k * max_len);
                        assert(val == "S_" + std::to_string(k));
                    }
                    delete[] char_data;

                    
                    if (j < size_B - 1) ctxB.nextIndex(1);
                }
                backend.endAction(&ctxB);
                if (i < size_A - 1) ctxA.nextIndex(1);
            }
            backend.endAction(&ctxA);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Reading validation completed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}