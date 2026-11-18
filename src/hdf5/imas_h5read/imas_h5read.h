#pragma once

#include <string>
#include <vector>
#include "direct_access_api.h" // Provides TensorView definition

namespace imas_h5read {

/**
 * @brief Constant representing reading until the end of a dimension.
 * Matching the behavior of MATLAB's Inf in ncread.
 */
const int INF = -1;

/**
 * @brief Reads all data from a variable and returns it as a TensorView.
 * This function is a simple wrapper around the direct access API.
 *
 * @param source The path to the HDF5 source file (or IDS name).
 * @param varname The path to the variable (dataset) within the HDF5 file.
 * @return An imas::direct_access::TensorView object.
 * @throws ALException on read errors.
 */
imas::direct_access::TensorView read(const std::string& source, const std::string& varname);

/**
 * @brief Reads a multidimensional slice of data from a variable.
 * This is the closest equivalent to MATLAB's ncread(source, varname, start, count).
 *
 * @param source The path to the HDF5 source file (or IDS name).
 * @param varname The path to the variable (dataset).
 * @param start Vector of starting indices for each dimension (0-based).
 * @param count Vector of number of elements to read along each dimension. 
 *              Use imas_h5read::INF (-1) for a dimension to read until its end.
 * @return An imas::direct_access::TensorView object.
 */
imas::direct_access::TensorView read(const std::string& source, const std::string& varname, 
                                     const std::vector<int>& start, const std::vector<int>& count);

/**
 * @brief Reads a multidimensional slice with stride (sampling interval).
 * This is the closest equivalent to MATLAB's ncread(source, varname, start, count, stride).
 *
 * @param source The path to the HDF5 source file (or IDS name).
 * @param varname The path to the variable (dataset).
 * @param start Vector of starting indices for each dimension (0-based).
 * @param count Vector of number of elements to read along each dimension.
 * @param stride Vector of sampling intervals for each dimension.
 * @return An imas::direct_access::TensorView object.
 */
imas::direct_access::TensorView read(const std::string& source, const std::string& varname, 
                                     const std::vector<int>& start, const std::vector<int>& count, 
                                     const std::vector<int>& stride);

} // namespace imas_h5read
