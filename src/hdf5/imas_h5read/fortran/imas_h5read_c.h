#ifndef IMAS_H5READ_C_H
#define IMAS_H5READ_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief Reads data from an IMAS HDF5 file via the C wrapper.
 * 
 * @param source Path to the HDF5 file (base name).
 * @param varname Path to the variable.
 * @param start Array of starting indices (0-based). Can be NULL for full read.
 * @param count Array of counts. Can be NULL for full read.
 * @param stride Array of strides. Can be NULL for no stride.
 * @param rank The dimensionality of the start/count/stride arrays.
 * @param data_out [OUT] Pointer to the allocated data buffer. Caller must free using imas_h5read_free_c.
 * @param type_out [OUT] Integer representing the DataType (mapped from imas::direct_access::DataType).
 * @param dims_out [OUT] Pointer to an array of long long containing the dimensions.
 * @param rank_out [OUT] The rank of the returned data.
 * @return 0 on success, -1 on error.
 */
int imas_h5read_c(const char* source, const char* varname,
                  const int* start, const int* count, const int* stride, int rank,
                  void** data_out, int* type_out, long long** dims_out, int* rank_out);

/**
 * @brief Frees memory allocated by imas_h5read_c.
 */
void imas_h5read_free_c(void* data, long long* dims);

#ifdef __cplusplus
}
#endif

#endif
