// @file  test_engine_group_constructor.cpp
// @brief Tests the PanzerDB group-ID constructor: writes a scalar into an existing
//        HDF5 group via its gid, reopens it for reading and reads the scalar back.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <hdf5.h> // Needed for HDF5 file/group operations

int main() {
    std::cout << "=== TEST PanzerDB: Constructor with a group identifier ===\n\n";
    const std::string filename = "test_group_constructor.h5";
    const std::string group_name = "magnetics";
    hid_t file_id = -1;
    hid_t group_id = -1;

    try {
        // ===================================================================
        // 1. Create an HDF5 file and a group, then write a scalar into it via PanzerDB
        // ===================================================================
        {
            // Create the HDF5 file
            file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            assert(file_id >= 0);
            std::cout << "HDF5 file '" << filename << "' created.\n";

            // Create a group in the file
            group_id = H5Gcreate2(file_id, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            assert(group_id >= 0);
            std::cout << "Group '" << group_name << "' created.\n";

            // Instantiate PanzerDB with the group ID for writing
            // The last 'false' tells PanzerDB not to close the group ID on destruction
            PanzerDB db(group_id, PanzerDB::OpenMode::WRITE, true, false);

            std::cout << "Writing a scalar 'temperature' into the group '" << group_name << "'...\n";
            double temp_value = 123.45;
            db.writeData("temperature", {}, &temp_value, 1);

            db.close(); // Flushes the data to disk
            std::cout << "PanzerDB (writing) closed.\n\n";
            
            // We do not close the group here, as we will reuse it for reading.
        }

        // ===================================================================
        // 2. Reopen the group and read the scalar
        // ===================================================================
        {
            // Instantiate PanzerDB with the same group ID for reading
            // The last 'true' tells PanzerDB to close the group ID on destruction
            PanzerDB db(group_id, PanzerDB::OpenMode::READ, true, true);
            std::cout << "PanzerDB (reading) opened on the existing group.\n";

            std::cout << "Reading the scalar 'temperature'...\n";
            int status = -1;
            double temp = db.readScalar<double>("temperature", &status);
            assert(status == 0);
            assert(std::abs(temp - 123.45) < 1e-9);
            std::cout << "Scalar 'temperature' read successfully: " << temp << "\n";

            db.close(); // PanzerDB also closes group_id since close_loc_id_on_exit=true
            group_id = -1; // The ID is no longer valid
            std::cout << "PanzerDB (reading) and HDF5 group closed.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "An exception was caught: " << e.what() << std::endl;
        if (group_id >= 0) H5Gclose(group_id);
        if (file_id >= 0) H5Fclose(file_id);
        return 1;
    }

    // ===================================================================
    // 3. Final cleanup
    // ===================================================================
    if (file_id >= 0) {
        H5Fclose(file_id);
        std::cout << "HDF5 file closed.\n";
    }

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}