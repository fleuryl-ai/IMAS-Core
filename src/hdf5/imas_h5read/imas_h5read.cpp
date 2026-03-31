#include "imas_h5read.h"
#include "direct_access_api.h"
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <hdf5.h>

namespace fs = std::filesystem;

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

// Helper structures for HDF5 external link resolution
struct LinkSearchData {
    std::string target_suffix;
    std::string found_file_name;
};

static herr_t find_elink_callback(hid_t loc_id, const char* name, const H5L_info_t* linfo, void* opdata) {
    if (linfo->type == H5L_TYPE_EXTERNAL) {
        LinkSearchData* data = static_cast<LinkSearchData*>(opdata);
        
        char* buf = new char[linfo->u.val_size];
        if (H5Lget_val(loc_id, name, buf, linfo->u.val_size, H5P_DEFAULT) >= 0) {
            unsigned flags = 0;
            const char* file_name = nullptr;
            const char* obj_name = nullptr;
            if (H5Lunpack_elink_val(buf, linfo->u.val_size, &flags, &file_name, &obj_name) >= 0) {
                if (obj_name != nullptr) {
                    std::string obj_str(obj_name);
                    // Check if obj_str ends with target_suffix
                    if (obj_str.length() >= data->target_suffix.length()) {
                        if (obj_str.compare(obj_str.length() - data->target_suffix.length(), data->target_suffix.length(), data->target_suffix) == 0) {
                            if (file_name != nullptr) {
                                data->found_file_name = file_name;
                            }
                            delete[] buf;
                            return 1; // Stop iteration, found it
                        }
                    }
                }
            }
        }
        delete[] buf;
    }
    return 0; // Continue iteration
}

static void parse_varname(const std::string& varname, std::string& ids_name, int& occurrence, std::string& dataset_path) {
    size_t first_slash = varname.find('/');
    std::string prefix = (first_slash != std::string::npos) ? varname.substr(0, first_slash) : varname;
    
    // Le dataset_path est ce qui suit le premier slash (peut être vide)
    dataset_path = (first_slash != std::string::npos) ? varname.substr(first_slash + 1) : "";

    size_t colon_pos = prefix.find(':');
    if (colon_pos != std::string::npos) {
        ids_name = prefix.substr(0, colon_pos);
        try {
            occurrence = std::stoi(prefix.substr(colon_pos + 1));
        } catch (...) {
            occurrence = 0;
        }
    } else {
        ids_name = prefix;
        occurrence = 0;
    }
}

static void resolve_source(const std::string& source, const std::string& ids_name, int occurrence, std::string& final_h5_file) {
    std::string path = source;
    std::string protocol = "imas:hdf5?path=";
    if (path.find(protocol) == 0) {
        path = path.substr(protocol.length());
    }

    // Is it a direct .h5 file?
    if (path.length() >= 3 && path.substr(path.length() - 3) == ".h5") {
        if (!fs::exists(path)) {
            throw std::runtime_error("HDF5 file does not exist: " + path);
        }
        final_h5_file = path;
        return;
    }

    // Otherwise, assume it's a directory containing master.h5
    fs::path master_path = fs::path(path) / "master.h5";
    if (!fs::exists(master_path)) {
        throw std::runtime_error("Source is neither a valid .h5 file nor a directory containing master.h5: " + path);
    }

    hid_t file_id = H5Fopen(master_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to open master file: " + master_path.string());
    }

    std::string target_suffix = "//" + ids_name;
    if (occurrence > 0) {
        target_suffix += "_" + std::to_string(occurrence);
    }

    LinkSearchData search_data;
    search_data.target_suffix = target_suffix;

    hsize_t idx = 0;
    // Iterate over root group links
    herr_t status = H5Literate(file_id, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, find_elink_callback, &search_data);

    H5Fclose(file_id);

    if (status > 0 && !search_data.found_file_name.empty()) {
        fs::path resolved_path = fs::path(path) / search_data.found_file_name;
        // weakly_canonical resolves ../ sequences if possible
        final_h5_file = fs::weakly_canonical(resolved_path).string();
    } else {
        throw std::runtime_error("Could not find appropriate external link in master.h5 for target: " + target_suffix);
    }
}

TensorView read(const std::string& source, const std::string& varname)
{
    std::string ids_name, dataset_path;
    int occurrence = 0;
    parse_varname(varname, ids_name, occurrence, dataset_path);

    std::string final_h5_file;
    resolve_source(source, ids_name, occurrence, final_h5_file);

    return imas::direct_access::read_tensor(final_h5_file, dataset_path);
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

    std::string ids_name, dataset_path;
    int occurrence = 0;
    parse_varname(augmented_path, ids_name, occurrence, dataset_path);

    std::string final_h5_file;
    resolve_source(source, ids_name, occurrence, final_h5_file);

    return imas::direct_access::read_tensor(final_h5_file, dataset_path);
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
