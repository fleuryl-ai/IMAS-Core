#include "imas_h5read/imas_h5read.h"
#include "panzerdb.h"
#include "al_defs.h"
#include <vector>
#include <string>
#include <numeric>
#include <filesystem>
#include <cassert>
#include <iostream>
#include <cmath> // For std::abs

// Use the fully qualified names for clarity
using imas::direct_access::DataType;
using imas::direct_access::TensorView;

// The filename base, .h5 will be added by the underlying API
const std::string FILENAME_WITHOUT_EXT = "core_profiles";
const std::string FILENAME_WITH_EXT = FILENAME_WITHOUT_EXT + ".h5";

void generate_test_file() {
    if (std::filesystem::exists(FILENAME_WITH_EXT)) {
        std::filesystem::remove(FILENAME_WITH_EXT);
    }
    PanzerDB db(FILENAME_WITH_EXT, PanzerDB::OpenMode::WRITE);

    // --- Scalar ---
    int32_t scalar_int_val = 42;
    db.writeData("test/scalar_int", {}, &scalar_int_val, 1);

    // --- 1D Vector ---
    std::vector<double> vector_double_data = {1.1, 2.2, 3.3, 4.4};
    std::vector<size_t> vector_dims = { vector_double_data.size() };
    db.writeDataSlices("test/vector_double", vector_dims, vector_double_data.data(), vector_double_data.size(), "");

    // --- 3D Array (2x3x4) ---
    // Values: i*100 + j*10 + k
    std::vector<size_t> dims_3d = {2, 3, 4};
    std::vector<int> array_3d_data(2 * 3 * 4);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                array_3d_data[i * 12 + j * 4 + k] = i * 100 + j * 10 + k;
            }
        }
    }
    db.writeDataSlices("test/array_3d", dims_3d, array_3d_data.data(), array_3d_data.size(), "");
    
    db.close();
}

void test_read_scalar_int() {
    std::cout << "--- Testing ReadScalarInt..." << std::endl;
    TensorView tv = imas_h5read::read(FILENAME_WITH_EXT, "core_profiles/test/scalar_int");
    assert(tv.type() == DataType::INT32);
    assert(*tv.as<int>() == 42);
    std::cout << "--- PASSED" << std::endl;
}

void test_read_vector_double() {
    std::cout << "--- Testing ReadVectorDouble..." << std::endl;
    TensorView tv = imas_h5read::read(FILENAME_WITH_EXT, "core_profiles/test/vector_double");
    // Flexible dimension check (IMAS might add a dim 1 at the start)
    const auto& d = tv.dims();
    if (d.size() == 2) assert(d[0] == 1 && d[1] == 4);
    else assert(d.size() == 1 && d[0] == 4);
    std::cout << "--- PASSED" << std::endl;
}

void test_read_multidim_slice() {
    std::cout << "--- Testing ReadMultidimSlice..." << std::endl;
    TensorView tv = imas_h5read::read(FILENAME_WITH_EXT, "core_profiles/test/array_3d", {0, 0, 0}, {2, 2, 2});
    const auto& d = tv.dims();
    // Allow for a leading dimension of 1 if rank > 3
    if (d.size() == 4) assert(d[0] == 1 && d[1] == 2 && d[2] == 2 && d[3] == 2);
    else assert(d.size() == 3 && d[0] == 2 && d[1] == 2 && d[2] == 2);
    std::cout << "--- PASSED" << std::endl;
}

void test_read_stride() {
    std::cout << "--- Testing ReadStride..." << std::endl;
    using imas_h5read::INF;

    TensorView tv = imas_h5read::read(FILENAME_WITH_EXT, "core_profiles/test/array_3d", {0, 0, 0}, {INF, INF, INF}, {1, 2, 2});

    const auto& d = tv.dims();
    std::cout << "  Actual dims: "; for(auto dim : d) std::cout << dim << " "; std::cout << std::endl;

    // Based on actual dims, verify the structure
    size_t r = d.size();
    assert(d[r-3] == 2); // Planes
    assert(d[r-2] == 2); // Rows (sampled 0, 2)
    assert(d[r-1] == 2); // Cols (sampled 0, 2)

    const int* data = tv.as<int>();
    assert(data[0] == 0);   // (0,0,0)
    assert(data[1] == 2);   // (0,0,2)
    assert(data[2] == 20);  // (0,2,0)
    assert(data[4] == 100); // (1,0,0)

    std::cout << "--- PASSED" << std::endl;
}

int main() {
    try {
        generate_test_file();
        test_read_scalar_int();
        test_read_vector_double();
        test_read_multidim_slice();
        test_read_stride();
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "All imas_h5read tests passed!" << std::endl;
    return 0;
}
