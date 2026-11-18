#pragma once

#include <string>
#include "direct_access_api.h" // Provides TensorView definition

namespace imas_h5read {

/**
 * @brief Reads all data from a variable and returns it as a TensorView.
 * This function is a simple wrapper around the direct access API.
 *
 * @param source The path to the HDF5 source file.
 * @param varname The path to the variable (dataset) within the HDF5 file.
 * @return An imas::direct_access::TensorView object.
 * @throws ALException on read errors.
 */
imas::direct_access::TensorView read(const std::string& source, const std::string& varname);

} // namespace imas_h5read
