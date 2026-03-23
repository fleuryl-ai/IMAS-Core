#include "direct_access_api.h"
#include "hdf5_backend.h"
#include "al_exception.h"
#include "path_parser.h"
#include "panzerdb.h"

#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>
#include <complex>
#include <memory>
#include <limits>

namespace imas {
namespace direct_access {

void collect_leaf_paths_recursive(
    PanzerDB& db,
    const std::vector<PathSegment>& segments,
    size_t segment_idx,
    std::string current_path,
    std::vector<std::string>& leaf_paths,
    std::vector<size_t>& selection_dims);

void collect_leaf_paths(
    PanzerDB& db,
    const std::vector<PathSegment>& segments,
    std::vector<std::string>& leaf_paths,
    std::vector<size_t>& selection_dims)
{
    collect_leaf_paths_recursive(db, segments, 0, "", leaf_paths, selection_dims);
}

void collect_leaf_paths_recursive(
    PanzerDB& db,
    const std::vector<PathSegment>& segments,
    size_t segment_idx,
    std::string current_path,
    std::vector<std::string>& leaf_paths,
    std::vector<size_t>& selection_dims)
{
    if (segment_idx >= segments.size()) {
        leaf_paths.push_back(current_path);
        return;
    }

    const auto& seg = segments[segment_idx];
    std::string new_path_base = current_path.empty() ? seg.node_name : current_path + "/" + seg.node_name;

    if (seg.selection != SelectionType::NONE) {
        size_t start = 0;
        size_t end = 0;

        std::vector<size_t> aos_shape = db.getAOSShape(new_path_base);
        if (aos_shape.empty()) {
            throw std::runtime_error("Could not determine size of AoS for path: " + new_path_base);
        }
        size_t aos_size = aos_shape[0];

        if (seg.selection == SelectionType::INDEX) {
            start = seg.index;
            end = start + 1;
        } else if (seg.selection == SelectionType::SLICE || seg.selection == SelectionType::ALL) {
            start = seg.has_start ? seg.start_index : 0;
            end = seg.has_end ? seg.end_index : aos_size;
        } else if (seg.selection == SelectionType::TIME_SLICE) {
            // ** Logique pour la sélection temporelle **
            // 1. Trouver le chemin de la time base pour cet AoS.
            //    (Suppose une méthode db.getAOSTimeBasePath())
            std::string time_path = new_path_base + "/time"; // Hypothèse sur le nom de la time base

            // 2. Convertir le temps en indice.
            start = seg.has_start_time ? db.getTimeIndex(time_path, seg.start_time, 0) : 0;
            end = seg.has_end_time ? db.getTimeIndex(time_path, seg.end_time, 0) + 1 : aos_size;
        }
        
        selection_dims.push_back(end - start);
        
        for (size_t i = start; i < end; ++i) {
            std::string indexed_path = new_path_base + "/" + std::to_string(i);
            collect_leaf_paths_recursive(db, segments, segment_idx + 1, indexed_path, leaf_paths, selection_dims);
        }
    } else {
        std::vector<size_t> aos_shape = db.getAOSShape(new_path_base);
        if (!aos_shape.empty() && segment_idx < segments.size() - 1) {
            size_t aos_size = aos_shape[0];
            selection_dims.push_back(aos_size);
            for (size_t i = 0; i < aos_size; ++i) {
                std::string indexed_path = new_path_base + "/" + std::to_string(i);
                collect_leaf_paths_recursive(db, segments, segment_idx + 1, indexed_path, leaf_paths, selection_dims);
            }
        } else {
            collect_leaf_paths_recursive(db, segments, segment_idx + 1, new_path_base, leaf_paths, selection_dims);
        }
    }
}

template <typename T>
TensorView read_typed_tensor(
    PanzerDB& db,
    const std::vector<std::string>& leaf_paths,
    const std::vector<size_t>& final_dims,
    DataType data_type)
{
    size_t total_elements = std::accumulate(final_dims.begin(), final_dims.end(), 1, std::multiplies<size_t>());
    if (total_elements == 0) total_elements = leaf_paths.size();
    
    size_t total_bytes = total_elements * sizeof(T);
    
    auto final_buffer = std::make_unique<char[]>(total_bytes);
    T* final_data_ptr = reinterpret_cast<T*>(final_buffer.get());

    for (size_t i = 0; i < leaf_paths.size(); ++i) {
        uint64_t ndim = 0;
        uint64_t shape[6] = {0};

        if constexpr (std::is_same_v<T, double>) {
            double* temp_data = nullptr;
            int status = db.pz_readData_by_index(leaf_paths[i].c_str(), -1, &ndim, shape, &temp_data);
            if (status == 0 && temp_data != nullptr) {
                final_data_ptr[i] = temp_data[0];
                free(temp_data);
            } else {
                final_data_ptr[i] = std::numeric_limits<double>::quiet_NaN();
            }
        } else {
             throw std::runtime_error("Unsupported type in read_typed_tensor.");
        }
    }
    
    return TensorView(std::move(final_buffer), final_dims, data_type);
}

TensorView read_tensor_impl(const std::string& ids_name, const std::vector<PathSegment>& segments)
{
    PanzerDB db(ids_name + ".h5", PanzerDB::OpenMode::READ);

    std::vector<std::string> leaf_paths;
    std::vector<size_t> selection_dims;
    collect_leaf_paths(db, segments, leaf_paths, selection_dims);

    if (leaf_paths.empty()) {
        return TensorView();
    }

    DataType data_type = db.get_leaf_type(leaf_paths[0]);

    std::vector<size_t> final_dims = selection_dims;
    
    switch (data_type) {
        case DataType::DOUBLE:
            return read_typed_tensor<double>(db, leaf_paths, final_dims, data_type);
        case DataType::INT32:
            return read_typed_tensor<int>(db, leaf_paths, final_dims, data_type);
        case DataType::COMPLEX_DOUBLE:
            return read_typed_tensor<std::complex<double>>(db, leaf_paths, final_dims, data_type);
        case DataType::STRING:
            return read_typed_tensor<std::string>(db, leaf_paths, final_dims, data_type);
        default:
            throw std::runtime_error("Unsupported data type for direct tensor read.");
    }
}

TensorView read_tensor(const std::string& ids_name, const std::string& path)
{
    PathParser parser(path);
    const auto& segments = parser.segments();
    return read_tensor_impl(ids_name, segments);
}

TensorView read_tensor(
    const std::string& ids_name,
    const std::string& path_template,
    const std::vector<int>& aos_indices)
{
    std::vector<PathSegment> segments;
    return read_tensor_impl(ids_name, segments);
}

} // namespace direct_access
} // namespace imas
