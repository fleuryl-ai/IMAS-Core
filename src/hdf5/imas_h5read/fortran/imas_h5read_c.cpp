#include "imas_h5read_c.h"
#include "imas_h5read.h"
#include <vector>
#include <cstring>
#include <cstdlib>

using namespace imas::direct_access;

extern "C" {

int imas_h5read_c(const char* source, const char* varname,
                  const int* start, const int* count, const int* stride, int rank,
                  void** data_out, int* type_out, long long** dims_out, int* rank_out)
{
    try {
        TensorView tv;

        if (start == nullptr || count == nullptr) {
            tv = imas_h5read::read(source, varname);
        } else {
            std::vector<int> v_start(start, start + rank);
            std::vector<int> v_count(count, count + rank);
            
            if (stride == nullptr) {
                tv = imas_h5read::read(source, varname, v_start, v_count);
            } else {
                std::vector<int> v_stride(stride, stride + rank);
                tv = imas_h5read::read(source, varname, v_start, v_count, v_stride);
            }
        }

        if (tv.type() == DataType::UNKNOWN || tv.total_elements() == 0) {
            return -1;
        }

        // Export data type
        *type_out = static_cast<int>(tv.type());

        // Export dimensions
        const auto& dims = tv.dims();
        *rank_out = static_cast<int>(dims.size());
        *dims_out = (long long*)std::malloc(dims.size() * sizeof(long long));
        for (size_t i = 0; i < dims.size(); ++i) {
            (*dims_out)[i] = static_cast<long long>(dims[i]);
        }

        // Export data (copy to a newly allocated buffer that C/Fortran can manage)
        size_t total_bytes = tv.size_in_bytes();
        *data_out = std::malloc(total_bytes);
        std::memcpy(*data_out, tv.data(), total_bytes);

        return 0; // Success

    } catch (...) {
        return -1; // General error
    }
}

void imas_h5read_free_c(void* data, long long* dims) {
    if (data) std::free(data);
    if (dims) std::free(dims);
}

} // extern "C"
