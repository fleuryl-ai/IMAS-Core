// @file  test_engine_write_read_root_scalar.cpp
// @brief PanzerDB root scalar test: writes two scalars directly at the root
//        ('temperature' and 'pression') and reads them back with readScalar.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== TEST PanzerDB: Root Scalar Write/Read ===\n\n";
    const std::string filename = "test_root_scalar.panzer";

    // ===================================================================
    // 1. Write a scalar at the root
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::cout << "Writing a scalar 'temperature' at the root...\n";
        double temp_value = 98.6;
        // We call writeData directly, without beginArray
        db.writeData("temperature", {}, &temp_value, 1);

        std::cout << "Writing a second scalar 'pression' at the root...\n";
        double pressure_value = 1013.25;
        db.writeData("pression", {}, &pressure_value, 1);

        db.close();
        std::cout << "File closed.\n\n";
    }

    // ===================================================================
    // 2. Reopen and read the scalar
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);

        /*std::cout << "Reading the scalar 'temperature'...\n";
        auto leaves = db.getLeaves();

        assert(leaves.size() == 2 && "There should be exactly two leaves.");

        const PanzerDB::Leaf* temperature_leaf = nullptr;
        for (const auto& leaf : leaves) {
            if (leaf.path == "temperature") {
                temperature_leaf = &leaf;
                break;
            }
        }

        assert(temperature_leaf != nullptr && "The leaf 'temperature' must be found.");

        assert(temperature_leaf->parent_path == "" && "The parent of the root scalar must be empty.");
        assert(temperature_leaf->shape.empty() && "The shape of a scalar must be empty.");
        assert(temperature_leaf->count == 1 && "The count of a scalar must be 1.");

        std::vector<double> data = db.readTensor(*temperature_leaf);
        assert(data.size() == 1 && "The read tensor must contain a single element.");
        assert(std::abs(data[0] - 98.6) < 1e-9 && "The value of the scalar must be 98.6.");

        std::cout << "Scalar read successfully: " << data[0] << " (expected: 98.6)\n";*/

        std::cout << "Reading the scalar 'temperature'...\n";
        int status = -1;
        double temp = db.readScalar<double>("temperature", &status);
        assert(status == 0);
        assert(std::abs(temp - 98.6) < 1e-9);
        std::cout << "Scalar 'temperature' read successfully: " << temp << "\n";

        std::cout << "Reading the scalar 'pression'...\n";
        double pressure = db.readScalar<double>("pression", &status);
        assert(status == 0);
        assert(std::abs(pressure - 1013.25) < 1e-9);
        std::cout << "Scalar 'pression' read successfully: " << pressure << "\n";
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}