// test_nested_aos_regression.cpp
// Regression test for nested AOS + putSlice with multiple time slices.
// Verifies that AOS_SHAPE datasets are created with correct sizes (no spurious
// extension) and that data (z_ion) is written to correct positions with
// expected values.

#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

const std::string URI = "imas:hdf5?path=./test_db_nested_aos_regression";
const std::string file_path = "./test_db_nested_aos_regression";

// Helper to read a dataset value from HDF5 file
static int read_h5_dataset_int(hid_t loc_id, const char *dataset_name) {
  hid_t dataset_id = H5Dopen2(loc_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::cerr << "[ERROR] Cannot open dataset: " << dataset_name << std::endl;
    return -999;
  }
  int value = 0;
  herr_t status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, &value);
  H5Dclose(dataset_id);
  if (status < 0) {
    std::cerr << "[ERROR] Cannot read dataset: " << dataset_name << std::endl;
    return -999;
  }
  return value;
}

// Helper to read a 1D slice of int dataset
static std::vector<int> read_h5_dataset_1d_int(hid_t loc_id,
                                               const char *dataset_name) {
  hid_t dataset_id = H5Dopen2(loc_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    std::cerr << "[ERROR] Cannot open dataset: " << dataset_name << std::endl;
    return {};
  }
  hid_t dataspace = H5Dget_space(dataset_id);
  hssize_t npoints = H5Sget_simple_extent_npoints(dataspace);
  H5Sclose(dataspace);

  if (npoints <= 0) {
    H5Dclose(dataset_id);
    return {};
  }

  std::vector<int> values(npoints);
  herr_t status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, values.data());
  H5Dclose(dataset_id);

  if (status < 0) {
    std::cerr << "[ERROR] Cannot read dataset: " << dataset_name << std::endl;
    return {};
  }
  return values;
}

int main() {
  try {
    std::cout
        << "\n=== Regression Test: Nested AOS + putSlice (s=1, t=2, p=3) ===\n";

    // Clean up previous run
    if (fs::exists(URI.substr(16))) { // Extract path from URI
      fs::remove_all(URI.substr(16));
    }

    int s = 1, t = 10, p = 3;
    std::vector<double> time(t);
    for (int i = 0; i < t; ++i) {
      time[i] = 1.0 * (i + 1);
    }

    std::vector<std::unique_ptr<ArraystructContext>> contexts;

    DataEntryContext dataEntryCtx(URI);
    HDF5Backend backend;
    backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

    // Write homogeneous_time
    OperationContext opCtx1(&dataEntryCtx, "core_sources", "", WRITE_OP);
    backend.beginAction(&opCtx1);
    int homogeneous_time = 1;
    backend.writeData(&opCtx1, "ids_properties/homogeneous_time", "",
                      &homogeneous_time, alconst::integer_data, 0, nullptr);
    backend.endAction(&opCtx1);

    // Begin putSlice operation
    double undefined_time = alconst::undefined_time;
    int interpmode = alconst::undefined_interp;
    OperationContext opCtx(&dataEntryCtx, "core_sources", WRITE_OP,
                           alconst::slice_op, undefined_time, interpmode);
    backend.beginAction(&opCtx);

    auto *sourceCtx = new ArraystructContext(&opCtx, "source", "");
    contexts.push_back(std::unique_ptr<ArraystructContext>(sourceCtx));
    backend.beginArraystructAction(sourceCtx, &s);

    for (int i = 0; i < s; ++i) {
      auto *profilesCtx =
          new ArraystructContext(sourceCtx, "profiles_1d", "time");
      contexts.push_back(std::unique_ptr<ArraystructContext>(profilesCtx));
      int profiles_size = t;
      backend.beginArraystructAction(profilesCtx, &profiles_size);

      for (int j = 0; j < t; ++j) {

        // Write z_ion for each ion
        auto *ionCtx = new ArraystructContext(profilesCtx, "ion", "");
        contexts.push_back(std::unique_ptr<ArraystructContext>(ionCtx));
        backend.beginArraystructAction(ionCtx, &p);

        // Create ion_values: [1.0, 2.0, 3.0]
        std::vector<double> ion_values(p);
        for (int k = 0; k < p; ++k) {
          ion_values[k] = (double)(k + 1);
        }

        for (int k = 0; k < p; ++k) {
          backend.writeData(ionCtx, "z_ion", "", &ion_values[k],
                            alconst::double_data, 0, nullptr);
          ionCtx->nextIndex(1);
        }
        backend.endAction(ionCtx);
        profilesCtx->nextIndex(1);
      }
      backend.endAction(profilesCtx);
      sourceCtx->nextIndex(1);
    }
    backend.endAction(sourceCtx);

    // Write time
    int size_time[1] = {10};
    backend.writeData(&opCtx, "time", "time", &time[0], alconst::double_data, 1,
                      size_time);

    backend.endAction(&opCtx);

    std::cout << "[OK] core_sources IDS written\n";

    // ========== VALIDATION ==========
    std::string ids_file = URI.substr(16) + "/core_sources.h5";
    ids_file = "./" + ids_file;
    printf("ids_file= %s\n ", ids_file.c_str());
    if (!fs::exists(ids_file)) {
      std::cerr << "[FAIL] File not created: " << ids_file << std::endl;
      return 1;
    }
    std::cout << "[OK] File exists: " << ids_file << "\n";

    // Open HDF5 file to read and validate datasets
    hid_t file_id = H5Fopen(ids_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
      std::cerr << "[FAIL] Cannot open HDF5 file\n";
      return 1;
    }

    hid_t group_id = H5Gopen2(file_id, "core_sources", H5P_DEFAULT);
    if (group_id < 0) {
      std::cerr << "[FAIL] Cannot open group 'core_sources'\n";
      H5Fclose(file_id);
      return 1;
    }

    // Validate AOS_SHAPE datasets
    std::cout << "\n[VALIDATE] Checking AOS_SHAPE datasets...\n";

    // source[]&AOS_SHAPE should be 1
    int source_aos_shape = read_h5_dataset_int(group_id, "source[]&AOS_SHAPE");
    std::cout << "  source[]&AOS_SHAPE = " << source_aos_shape
              << " (expected 1)\n";
    assert(source_aos_shape == 1 && "source[]&AOS_SHAPE must be 1");

    // source[]&profiles_1d[]&AOS_SHAPE should be [2] (one value for the first
    // time slice) Note: AOS_SHAPE for profiles_1d has shape (1, 2, 1) but we
    // read at [0,0]
    hid_t prof_aos_id =
        H5Dopen2(group_id, "source[]&profiles_1d[]&AOS_SHAPE", H5P_DEFAULT);
    if (prof_aos_id < 0) {
      std::cerr << "[FAIL] Cannot open profiles_1d AOS_SHAPE\n";
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }
    int prof_aos_shape = 0;
    H5Dread(prof_aos_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            &prof_aos_shape);
    H5Dclose(prof_aos_id);
    std::cout << "  source[]&profiles_1d[]&AOS_SHAPE[0,0] = " << prof_aos_shape
              << " (expected 10)\n";
    assert(prof_aos_shape == 10 && "profiles_1d[]&AOS_SHAPE must be 10");

    // source[]&profiles_1d[]&ion[]&AOS_SHAPE should be [3, 3] for [0,0] and
    // [0,1]
    hid_t ion_aos_id = H5Dopen2(
        group_id, "source[]&profiles_1d[]&ion[]&AOS_SHAPE", H5P_DEFAULT);
    if (ion_aos_id < 0) {
      std::cerr << "[FAIL] Cannot open ion AOS_SHAPE\n";
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }
    std::vector<int> ion_aos_values = read_h5_dataset_1d_int(
        group_id, "source[]&profiles_1d[]&ion[]&AOS_SHAPE");
    H5Dclose(ion_aos_id);
    std::cout << "  source[]&profiles_1d[]&ion[]&AOS_SHAPE size = "
              << ion_aos_values.size() << " (expected 2)\n";
    assert(ion_aos_values.size() >= 2 &&
           "ion[]&AOS_SHAPE must have at least 2 entries");
    std::cout << "    [0] = " << ion_aos_values[0] << " (expected 3)\n";
    std::cout << "    [1] = " << ion_aos_values[1] << " (expected 3)\n";
    assert(ion_aos_values[0] == 3 && "ion[]&AOS_SHAPE[0] must be 3");
    assert(ion_aos_values[1] == 3 && "ion[]&AOS_SHAPE[1] must be 3");

    // Validate z_ion data: should have values [1, 2, 3] and [1, 2, 3] for the
    // two time slices
    std::cout << "\n[VALIDATE] Checking z_ion data...\n";
    hid_t zion_id =
        H5Dopen2(group_id, "source[]&profiles_1d[]&ion[]&z_ion", H5P_DEFAULT);
    if (zion_id < 0) {
      std::cerr << "[FAIL] Cannot open z_ion dataset\n";
      H5Gclose(group_id);
      H5Fclose(file_id);
      return 1;
    }

    hid_t zion_space = H5Dget_space(zion_id);
    hssize_t zion_npoints = H5Sget_simple_extent_npoints(zion_space);
    H5Sclose(zion_space);

    std::vector<double> zion_data(zion_npoints);
    H5Dread(zion_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            zion_data.data());
    H5Dclose(zion_id);

    std::cout << "  z_ion data size = " << zion_npoints
              << " (expected 30: 10 time slices × 3 ions)\n";
    assert(zion_npoints == 30 && "z_ion must have 30 values");

    // Expected order: (0,0,0):1, (0,0,1):2, (0,0,2):3, (0,1,0):1, (0,1,1):2,
    // (0,1,2):3
    std::cout << "  z_ion values: [";
    for (size_t i = 0; i < zion_data.size(); ++i) {
      std::cout << zion_data[i];
      if (i < zion_data.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]\n";

    // Check that no value is 0 (which would indicate a fill value or error)
    for (size_t i = 0; i < zion_data.size(); ++i) {
      assert(zion_data[i] > 0.5 &&
             "z_ion values must not be zero (fill value)");
      // Each value should be 1, 2, or 3
      assert((zion_data[i] > 0.5 && zion_data[i] < 3.5) &&
             "z_ion values must be in range [1, 3]");
    }

    // Pattern validation: first 3 should be [1,2,3], next 3 should be [1,2,3]
    for (int slice = 0; slice < 10; ++slice) {
      for (int ion = 0; ion < 3; ++ion) {
        int idx = slice * 3 + ion;
        double expected = (double)(ion + 1);
        assert(std::abs(zion_data[idx] - expected) < 0.01 &&
               "z_ion pattern mismatch");
      }
    }
    std::cout << "  [OK] z_ion pattern validated: all slices have [1, 2, 3]\n";

    // Cleanup
    H5Gclose(group_id);
    H5Fclose(file_id);

    std::cout << "\n=== [SUCCESS] All regression tests passed ===\n";
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "\n[EXCEPTION] " << e.what() << std::endl;
    return 1;
  }
}
