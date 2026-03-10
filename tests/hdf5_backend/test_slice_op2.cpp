// tests/hdf5_backend/test_slice_op2.cpp
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
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_slice_op2";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Slice Op 2: code/output_flag (1D dynamic, spatial 0D) ===\n" << RESET;

        // Cleanup
        if (fs::exists("test_db_slice_op2")) {
            fs::remove_all("test_db_slice_op2");
        }

        const int TIME_STEPS = 3;
        std::vector<double> time_values = {10.0, 11.0, 12.0};
        std::vector<int> flag_values = {1, 0, 1};
        std::vector<double> pressure_values = {1000.0, 1100.0, 1200.0};
        std::vector<int> test_code_flag_values = {10, 20, 30};

        // 1. Writing
        {
            std::cout << "\n--- Phase 1: Writing Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Homogeneous time
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Time
            int time_dim = 1;
            int time_size[] = {TIME_STEPS};
            backend.writeData(&opCtx, "time", "time", time_values.data(), alconst::double_data, time_dim, time_size);

            // Structure 'code' (Static AoS size 1)
            ArraystructContext codeCtx(&opCtx, "code", "");
            int code_size = 1;
            backend.beginArraystructAction(&codeCtx, &code_size);

            // output_flag (Dynamic 1D signal, spatial 0D)
            // dim=1 because it has a time dimension
            int flag_dim = 1;
            int flag_size[] = {TIME_STEPS};
            backend.writeData(&opCtx, "output_flag", "time", flag_values.data(), alconst::integer_data, flag_dim, flag_size);

            backend.endAction(&codeCtx);

            // Structure 'pressure' (Static AoS size 1)
            ArraystructContext pressureCtx(&opCtx, "pressure", "");
            int pressure_size = 1;
            backend.beginArraystructAction(&pressureCtx, &pressure_size);

            int pressure_dim = 1;
            int pressure_dims[] = {TIME_STEPS};
            backend.writeData(&pressureCtx, "data", "time", pressure_values.data(), alconst::double_data, pressure_dim, pressure_dims);
            backend.endAction(&pressureCtx);

            // test/code_flag (Dynamic 1D signal at root, spatial 0D)
            int test_flag_dim = 1;
            int test_flag_size[] = {TIME_STEPS};
            backend.writeData(&opCtx, "test/code_flag", "time", test_code_flag_values.data(), alconst::integer_data, test_flag_dim, test_flag_size);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // 2. Reading Slices
        std::cout << "\n--- Phase 2: Reading Slices ---\n";
        for (int i = 0; i < TIME_STEPS; ++i) {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            double target_time = time_values[i];
            std::cout << "Reading slice at t=" << target_time << "...\n";

            OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP, alconst::slice_op, target_time, alconst::closest_interp);
            backend.beginAction(&opCtx);

            ArraystructContext codeCtx(&opCtx, "code", "");
            int size = 0;
            backend.beginArraystructAction(&codeCtx, &size);
            assert(size == 1);

            void* data = nullptr;
            int type = alconst::integer_data;
            int dim = 0;
            int dims[H5S_MAX_RANK];

            backend.readData(&codeCtx, "output_flag", "time", &data, &type, &dim, dims);
            
            // Verification dimensions
            // Expecting dim=1, size[0]=1 because it's a scalar slice of a dynamic signal not in a dynamic AoS
            printf("  Read dim: %d, size[0]: %d\n", dim, dims[0]);
            assert(dim == 1);
            assert(dims[0] == 1);

            int val = *(int*)data;
            int expected = flag_values[i];
            
            if (val != expected) {
                std::cerr << RED << "Mismatch at t=" << target_time << ": expected " << expected << ", got " << val << RESET << std::endl;
                return 1;
            }
            std::cout << "  Value: " << val << " [OK]\n";

            free(data);

            backend.endAction(&codeCtx);

            ArraystructContext pressureCtx(&opCtx, "pressure", "");
            int p_size = 0;
            backend.beginArraystructAction(&pressureCtx, &p_size);
            assert(p_size == 1);

            void* p_data = nullptr;
            int p_type = alconst::double_data;
            int p_dim = 0;
            int p_dims[H5S_MAX_RANK];

            backend.readData(&pressureCtx, "data", "time", &p_data, &p_type, &p_dim, p_dims);
            
            assert(p_dim == 1);
            assert(p_dims[0] == 1);

            double p_val = *(double*)p_data;
            double p_expected = pressure_values[i];
            
            if (std::abs(p_val - p_expected) > 1e-9) {
                std::cerr << RED << "Mismatch pressure at t=" << target_time << ": expected " << p_expected << ", got " << p_val << RESET << std::endl;
                return 1;
            }
            std::cout << "  Pressure Value: " << p_val << " [OK]\n";

            free(p_data);
            backend.endAction(&pressureCtx);

            // Read test/code_flag
            void* t_data = nullptr;
            int t_type = alconst::integer_data;
            int t_dim = 0;
            int t_dims[H5S_MAX_RANK];

            backend.readData(&opCtx, "test/code_flag", "time", &t_data, &t_type, &t_dim, t_dims);
            assert(t_dim == 1);
            assert(t_dims[0] == 1);
            int t_val = *(int*)t_data;
            int t_expected = test_code_flag_values[i];
            if (t_val != t_expected) {
                std::cerr << RED << "Mismatch test/code_flag at t=" << target_time << ": expected " << t_expected << ", got " << t_val << RESET << std::endl;
                return 1;
            }
            std::cout << "  Test Code Flag Value: " << t_val << " [OK]\n";
            free(t_data);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }
        std::cout << GREEN << "[OK] Reading validation completed.\n" << RESET;

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}