#include "imas_h5read.h"
#include "direct_access_api.h"
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace imas_h5read {

using imas::direct_access::TensorView;
using imas::direct_access::DataType;

static void copy_with_stride_recursive(
    const char* src_base,
    char* dst_base,
    const std::vector<size_t>& src_dims,
    const std::vector<int>& strides,
    const std::vector<size_t>& src_strides_bytes,
    size_t element_size,
    size_t dim_idx,
    size_t& dst_offset_bytes)
{
    if (dim_idx == src_dims.size()) {
        std::memcpy(dst_base + dst_offset_bytes, src_base, element_size);
        dst_offset_bytes += element_size;
        return;
    }

    for (size_t i = 0; i < src_dims[dim_idx]; i += strides[dim_idx]) {
        copy_with_stride_recursive(
            src_base + i * src_strides_bytes[dim_idx],
            dst_base, src_dims, strides, src_strides_bytes,
            element_size, dim_idx + 1, dst_offset_bytes
        );
    }
}

TensorView read(const std::string& source, const std::string& varname)
{
    return imas::direct_access::read_tensor(source, varname);
}

TensorView read(const std::string& source, const std::string& varname, 
                const std::vector<int>& start, const std::vector<int>& count)
{
    std::string augmented_path = varname + "[";
    for (size_t i = 0; i < start.size(); ++i) {
        if (i > 0) augmented_path += ",";
        augmented_path += std::to_string(start[i]) + ":";
        if (count[i] != INF) {
            augmented_path += std::to_string(start[i] + count[i]);
        }
    }
    augmented_path += "]";
    return imas::direct_access::read_tensor(source, augmented_path);
}

TensorView read(const std::string& source, const std::string& varname, 
                const std::vector<int>& start, const std::vector<int>& count, 
                const std::vector<int>& stride)
{
    TensorView block = read(source, varname, start, count);
    const std::vector<size_t>& src_dims = block.dims();
    if (src_dims.empty()) return block;

    // Align strides with actual dimensions (match from the right)
    std::vector<int> effective_strides(src_dims.size(), 1);
    size_t n_user = stride.size();
    size_t n_actual = src_dims.size();
    for (size_t i = 0; i < std::min(n_user, n_actual); ++i) {
        effective_strides[n_actual - 1 - i] = stride[n_user - 1 - i];
    }

    bool needs_sampling = false;
    for (int s : effective_strides) {
        if (s <= 0) throw std::invalid_argument("Stride must be greater than 0");
        if (s > 1) needs_sampling = true;
    }
    if (!needs_sampling) return block;

    // Calculate destination dimensions
    std::vector<size_t> dst_dims;
    size_t total_dst_elements = 1;
    for (size_t i = 0; i < src_dims.size(); ++i) {
        size_t d = (src_dims[i] + effective_strides[i] - 1) / effective_strides[i];
        dst_dims.push_back(d);
        total_dst_elements *= d;
    }

    if (total_dst_elements == 0) return TensorView();

    size_t total_src_elements = block.total_elements();
    if (total_src_elements == 0) return block;

    size_t element_size = block.size_in_bytes() / total_src_elements;
    auto new_buffer = std::shared_ptr<char[]>(new char[total_dst_elements * element_size], std::default_delete<char[]>());

    std::vector<size_t> src_strides_bytes(src_dims.size());
    src_strides_bytes.back() = element_size;
    for (int i = (int)src_dims.size() - 2; i >= 0; --i) {
        src_strides_bytes[i] = src_strides_bytes[i + 1] * src_dims[i + 1];
    }

    size_t dst_offset_bytes = 0;
    copy_with_stride_recursive(
        static_cast<const char*>(block.data()),
        new_buffer.get(),
        src_dims, effective_strides, src_strides_bytes,
        element_size, 0, dst_offset_bytes
    );

    return TensorView(std::move(new_buffer), dst_dims, block.type(), block.metadata());
}

} // namespace imas_h5read
