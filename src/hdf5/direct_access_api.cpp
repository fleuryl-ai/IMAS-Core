#include "direct_access_api.h"
#include "hdf5_backend.h"
#include "al_exception.h"
#include "path_parser.h"
#include "panzerdb.h"
#include "al_const.h"
#include "data_interpolation.h"

#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>
#include <memory>
#include <limits>
#include <set>
#include <map>
#include <algorithm>
#include <sstream>

namespace imas {
namespace direct_access {

// Déclarations anticipées
void collect_leaf_paths_recursive(PanzerDB& db, const std::vector<PathSegment>& segments, size_t segment_idx, std::string current_real_path, std::string current_schema_path, int64_t current_time_index, std::vector<std::pair<std::string, int64_t>>& leaf_paths, std::vector<size_t>& selection_dims, const std::map<std::string, NodeType>& aos_paths, const std::map<std::string, size_t>& aos_sizes);
TensorView read_tensor_impl_core(PanzerDB& db, const std::vector<PathSegment>& segments, const std::map<std::string, NodeType>& aos_paths, const std::map<std::string, size_t>& aos_sizes);

// Helpers
std::vector<double> get_time_vector(PanzerDB& db, const std::string& aos_path, size_t aos_size) {
    int status = 0;
    int homogeneous_time = 1;
    try {
        homogeneous_time = db.readScalar<int32_t>("ids_properties/homogeneous_time", &status);
        if (status != 0) homogeneous_time = 1;
    } catch (...) { homogeneous_time = 1; }

    if (db.is_dynamic_aos(aos_path)) {
        std::vector<double> time_values;
        time_values.reserve(aos_size);
        for (size_t i = 0; i < aos_size; ++i) {
            std::string time_path = aos_path + "/" + std::to_string(i) + "/time";
            double time_val = db.readScalar<double>(time_path, &status);
            if (status == 0) time_values.push_back(time_val);
            else throw std::runtime_error("Could not read time value at path: " + time_path);
        }
        return time_values;
    } else {
        if (homogeneous_time == 1) return db.getWholeDynamicSignal("time");
        else if (homogeneous_time == 2) return {};
        else throw std::runtime_error("Inhomogeneous static AoS time-slicing not supported.");
    }
}

// ------------------------------------------------------------------------------------------------
// 1. LIST NODES (pour imas_h5ls)
// ------------------------------------------------------------------------------------------------

std::pair<std::vector<NodeInfo>, std::map<std::string, NodeType>>
list_nodes(const std::string& ids_name, bool recursive, bool show_aos, bool show_metadata) {
    PanzerDB db(ids_name + ".h5", PanzerDB::OpenMode::READ);
    const auto& leaves = db.getLeaves();

    std::map<std::string, NodeType> aos_paths;
    std::map<std::string, size_t> aos_sizes;
    std::map<std::string, const PanzerDB::Leaf*> schema_to_leaf_map;

    for (const auto& leaf : leaves) {
        if (leaf.flags == 2 || leaf.flags == 3) {
            std::string schema_path = PanzerDB::stripIndices(std::string(leaf.path));
            aos_paths[schema_path] = (leaf.flags == 3) ? NodeType::AOS_DYNAMIC : NodeType::AOS_STATIC;
            
            if (leaf.flags == 2 && !leaf.shape.empty()) {
                aos_sizes[schema_path] = leaf.shape[0];
            } else if (leaf.flags == 3) {
                auto s = db.getAOSShape(std::string(leaf.path));
                aos_sizes[schema_path] = s.empty() ? 0 : s.front();
            }
        }
        
        if ((leaf.flags & 0xF) != 0) continue; 
        std::string schema_path = PanzerDB::stripIndices(std::string(leaf.path));
        if (!show_metadata && schema_path.find('@') != std::string::npos) continue;

        if (schema_to_leaf_map.find(schema_path) == schema_to_leaf_map.end()) {
            schema_to_leaf_map[schema_path] = &leaf;
        }
    }

    std::vector<NodeInfo> result;
    for (const auto& pair : schema_to_leaf_map) {
        const std::string& schema_path = pair.first;
        const PanzerDB::Leaf* rep_leaf = pair.second;
        
        std::vector<size_t> logical_dims;
        bool is_in_dynamic_aos = false;
        
        std::string prefix;
        std::stringstream ss(schema_path);
        std::string segment;
        
        while(std::getline(ss, segment, '/')) {
             if (ss.peek() == EOF && schema_path.find('/') != std::string::npos) break;
             prefix += (prefix.empty() ? "" : "/") + segment;
             if (aos_paths.count(prefix)) {
                 if (aos_paths.at(prefix) == NodeType::AOS_DYNAMIC) is_in_dynamic_aos = true;
                 if (aos_sizes.count(prefix) && aos_sizes[prefix] > 0) logical_dims.push_back(aos_sizes[prefix]);
             }
        }
        
        if (logical_dims.empty() && !is_in_dynamic_aos && rep_leaf->count > 1) {
            logical_dims.push_back(rep_leaf->count);
        }
        
        if (!rep_leaf->shape.empty() && !(rep_leaf->shape.size() == 1 && rep_leaf->shape[0] <= 1)) {
            logical_dims.insert(logical_dims.end(), rep_leaf->shape.begin(), rep_leaf->shape.end());
        }

        result.push_back({schema_path, NodeType::DATASET, logical_dims, is_in_dynamic_aos});
    }

    if (show_aos) {
        for (const auto& pair : aos_paths) {
            result.push_back({pair.first, pair.second, {}, false});
        }
    }
    
    std::sort(result.begin(), result.end(), [](const NodeInfo& a, const NodeInfo& b) { return a.path < b.path; });

    if (!recursive) {
        std::vector<NodeInfo> filtered_result;
        for (const auto& node : result) {
            if (node.path.find('/') == std::string::npos) filtered_result.push_back(node);
        }
        return {filtered_result, aos_paths};
    }
    
    return {result, aos_paths};
}

// ------------------------------------------------------------------------------------------------
// 2. LECTURE DES TENSEURS (pour imas_h5dump et tests)
// ------------------------------------------------------------------------------------------------

void collect_leaf_paths_recursive(
    PanzerDB& db,
    const std::vector<PathSegment>& segments,
    size_t segment_idx,
    std::string current_real_path,
    std::string current_schema_path,
    int64_t current_time_index,
    std::vector<std::pair<std::string, int64_t>>& leaf_paths,
    std::vector<size_t>& selection_dims,
    const std::map<std::string, NodeType>& aos_paths,
    const std::map<std::string, size_t>& aos_sizes)
{
    if (segment_idx >= segments.size()) {
        leaf_paths.push_back({current_real_path, current_time_index});
        return;
    }

    const auto& seg = segments[segment_idx];
    std::string new_schema_path = current_schema_path.empty() ? seg.node_name : current_schema_path + "/" + seg.node_name;
    std::string new_real_path = current_real_path.empty() ? seg.node_name : current_real_path + "/" + seg.node_name;

    bool is_aos = aos_paths.count(new_schema_path) > 0;
    bool is_dyn_aos = is_aos && aos_paths.at(new_schema_path) == NodeType::AOS_DYNAMIC;
    
    // --- LA REGLE D'OR CORRIGEE ---
    bool should_add_index = false;
    if (is_aos) {
        if (is_dyn_aos) {
            // C'est un AoS dynamique. On n'ajoute l'index que si le segment SUIVANT n'est PAS un champ final.
            if (segment_idx + 1 < segments.size()) {
                bool is_next_segment_last = (segment_idx + 1 == segments.size() - 1);
                if (!is_next_segment_last) {
                    should_add_index = true; // Le prochain segment est un sous-AoS, donc il faut un index
                }
            }
        } else {
            should_add_index = true; // C'est un AoS statique, on ajoute toujours l'index
        }
    }

    size_t aos_size = aos_sizes.count(new_schema_path) ? aos_sizes.at(new_schema_path) : 0;

    if (seg.selection != SelectionType::NONE) {
        size_t start = 0, end = 0;
        if (seg.selection == SelectionType::INDEX) {
            start = seg.index; end = start + 1;
        } else if (seg.selection == SelectionType::SLICE || seg.selection == SelectionType::ALL) {
            start = seg.has_start ? seg.start_index : 0;
            end = seg.has_end ? seg.end_index : aos_size;
            if (selection_dims.size() <= segment_idx) selection_dims.push_back(end - start);
        } else if (seg.selection == SelectionType::TIME) {
            std::vector<double> time_values = get_time_vector(db, new_schema_path, aos_size);
            DataInterpolation interpolator;
            std::map<std::string, int> times_indices;
            start = seg.has_start_time ? interpolator.getSlicesTimesIndices(seg.start_time, time_values, times_indices, CLOSEST_INTERP) : 0;
            end = seg.has_end_time ? interpolator.getSlicesTimesIndices(seg.end_time, time_values, times_indices, CLOSEST_INTERP) + 1 : aos_size;
            if (selection_dims.size() <= segment_idx) selection_dims.push_back(end - start);
        }

        if (should_add_index) {
            for (size_t i = start; i < end; ++i) {
                std::string indexed_path = new_real_path + "/" + std::to_string(i);
                collect_leaf_paths_recursive(db, segments, segment_idx + 1, indexed_path, new_schema_path, current_time_index, leaf_paths, selection_dims, aos_paths, aos_sizes);
            }
        } else if (is_dyn_aos) {
            if (seg.selection == SelectionType::ALL || (start == 0 && end == aos_size)) {
                collect_leaf_paths_recursive(db, segments, segment_idx + 1, new_real_path, new_schema_path, -1, leaf_paths, selection_dims, aos_paths, aos_sizes);
            } else {
                for (size_t i = start; i < end; ++i) {
                    collect_leaf_paths_recursive(db, segments, segment_idx + 1, new_real_path, new_schema_path, i, leaf_paths, selection_dims, aos_paths, aos_sizes);
                }
            }
        } else {
             collect_leaf_paths_recursive(db, segments, segment_idx + 1, new_real_path, new_schema_path, current_time_index, leaf_paths, selection_dims, aos_paths, aos_sizes);
        }
    } else { // Pas de sélection demandée
        if (aos_size > 0 && segment_idx < segments.size() - 1) {
            if (should_add_index) {
                if (selection_dims.size() <= segment_idx) selection_dims.push_back(aos_size);
                for (size_t i = 0; i < aos_size; ++i) {
                    std::string indexed_path = new_real_path + "/" + std::to_string(i);
                    collect_leaf_paths_recursive(db, segments, segment_idx + 1, indexed_path, new_schema_path, current_time_index, leaf_paths, selection_dims, aos_paths, aos_sizes);
                }
            } else if (is_dyn_aos) {
                collect_leaf_paths_recursive(db, segments, segment_idx + 1, new_real_path, new_schema_path, -1, leaf_paths, selection_dims, aos_paths, aos_sizes);
            }
        } else {
            collect_leaf_paths_recursive(db, segments, segment_idx + 1, new_real_path, new_schema_path, current_time_index, leaf_paths, selection_dims, aos_paths, aos_sizes);
        }
    }
}

template <typename T>
TensorView read_typed_tensor(
    PanzerDB& db,
    const std::vector<std::pair<std::string, int64_t>>& leaf_paths,
    const std::vector<size_t>& selection_dims,
    DataType data_type,
    const std::map<std::string, std::string>& metadata)
{
    if (leaf_paths.empty()) return TensorView();

    // 1. FAST PATH
    if (leaf_paths.size() == 1) {
        uint64_t ndim = 0;
        uint64_t shape[6] = {0};
        void* temp_data = nullptr;
        int status = -1;

        if constexpr (std::is_same_v<T, double>) {
            status = db.pz_readData_by_index(leaf_paths[0].first.c_str(), leaf_paths[0].second, &ndim, shape, (double**)&temp_data);
        } else if constexpr (std::is_same_v<T, int>) {
            status = db.pz_readIntData_by_index(leaf_paths[0].first.c_str(), leaf_paths[0].second, &ndim, shape, (int32_t**)&temp_data);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            status = db.pz_readComplexData_by_index(leaf_paths[0].first.c_str(), leaf_paths[0].second, &ndim, shape, (std::complex<double>**)&temp_data);
        }

        if (status == 0 && temp_data) {
            std::vector<size_t> actual_dims = selection_dims;
            
            if (leaf_paths[0].second == -1 && ndim > 0 && db.is_dynamic_aos(PanzerDB::stripIndices(leaf_paths[0].first))) {
                actual_dims.push_back(shape[ndim - 1]); // Temps au début
                for (uint64_t d = 0; d < ndim - 1; ++d) actual_dims.push_back(shape[d]); // Reste
            } else {
                for (uint64_t d = 0; d < ndim; ++d) actual_dims.push_back(shape[d]);
            }

            auto final_buffer = std::shared_ptr<char[]>(reinterpret_cast<char*>(temp_data), [](char* p){ if(p) free(p); });
            return TensorView(std::move(final_buffer), actual_dims, data_type, metadata);
        }
        return TensorView();
    }

    // 2. SLOW PATH
    std::vector<size_t> final_dims = selection_dims;
    uint64_t leaf_ndim = 0;
    uint64_t leaf_shape[6] = {0};
    
    if constexpr (std::is_same_v<T, double>) {
        double* temp = nullptr;
        if (db.pz_readData_by_index(leaf_paths[0].first.c_str(), leaf_paths[0].second, &leaf_ndim, leaf_shape, &temp) == 0) if(temp) free(temp);
    } else if constexpr (std::is_same_v<T, int>) {
        int32_t* temp = nullptr;
        if (db.pz_readIntData_by_index(leaf_paths[0].first.c_str(), leaf_paths[0].second, &leaf_ndim, leaf_shape, &temp) == 0) if(temp) free(temp);
    }
    
    if (leaf_paths[0].second == -1 && leaf_ndim > 0 && db.is_dynamic_aos(PanzerDB::stripIndices(leaf_paths[0].first))) {
        final_dims.push_back(leaf_shape[leaf_ndim - 1]);
        for (uint64_t d = 0; d < leaf_ndim - 1; ++d) final_dims.push_back(leaf_shape[d]);
    } else {
        for (uint64_t d = 0; d < leaf_ndim; ++d) final_dims.push_back(leaf_shape[d]);
    }
    
    size_t total_elements = std::accumulate(final_dims.begin(), final_dims.end(), 1, std::multiplies<size_t>());
    if (total_elements == 0) return TensorView();

    auto final_buffer = std::shared_ptr<char[]>(new char[total_elements * sizeof(T)], std::default_delete<char[]>());
    T* final_data_ptr = reinterpret_cast<T*>(final_buffer.get());
    size_t global_offset = 0;

    for (size_t i = 0; i < leaf_paths.size(); ++i) {
        uint64_t ndim = 0;
        uint64_t shape[6] = {0};
        const std::string& path = leaf_paths[i].first;
        int64_t time_idx = leaf_paths[i].second;
        
        if constexpr (std::is_same_v<T, double>) {
            double* temp_data = nullptr;
            if (db.pz_readData_by_index(path.c_str(), time_idx, &ndim, shape, &temp_data) == 0 && temp_data) {
                size_t count = 1; for(uint64_t d=0; d<ndim; ++d) count *= shape[d];
                size_t to_copy = std::min(count, total_elements - global_offset);
                std::copy(temp_data, temp_data + to_copy, final_data_ptr + global_offset);
                global_offset += to_copy; free(temp_data);
            }
        } else if constexpr (std::is_same_v<T, int>) {
            int32_t* temp_data = nullptr;
            if (db.pz_readIntData_by_index(path.c_str(), time_idx, &ndim, shape, &temp_data) == 0 && temp_data) {
                size_t count = 1; for(uint64_t d=0; d<ndim; ++d) count *= shape[d];
                size_t to_copy = std::min(count, total_elements - global_offset);
                std::copy(temp_data, temp_data + to_copy, final_data_ptr + global_offset);
                global_offset += to_copy; free(temp_data);
            }
        }
    }
    
    return TensorView(std::move(final_buffer), final_dims, data_type, metadata);
}

TensorView read_list_of_strings(
    PanzerDB& db,
    const std::vector<std::pair<std::string, int64_t>>& leaf_paths,
    const std::vector<size_t>& selection_dims,
    const std::map<std::string, std::string>& metadata)
{
    if (leaf_paths.empty()) return TensorView();

    if (leaf_paths.size() == 1) {
        char* temp_data = nullptr;
        uint64_t ndim = 0;
        uint64_t shape[6] = {0};
        
        int status = db.pz_readStringData_by_index(leaf_paths[0].first.c_str(), leaf_paths[0].second, &ndim, shape, &temp_data);
        if (status == 0 && temp_data) {
            std::vector<size_t> actual_dims = selection_dims;
            for(uint64_t d=0; d<ndim; ++d) actual_dims.push_back(shape[d]);
            auto final_buffer = std::shared_ptr<char[]>(temp_data, [](char* p){ if(p) free(p); });
            return TensorView(std::move(final_buffer), actual_dims, DataType::LIST_OF_STRINGS, metadata);
        }
        return TensorView();
    }

    std::vector<std::string> temp_strings;
    size_t max_len = 0;

    for (const auto& item : leaf_paths) {
        char* temp_data = nullptr;
        uint64_t ndim = 0;
        uint64_t shape[6] = {0};
        int status = db.pz_readStringData_by_index(item.first.c_str(), item.second, &ndim, shape, &temp_data);
        if (status == 0 && temp_data != nullptr) {
            std::string s(temp_data);
            if (s.length() > max_len) max_len = s.length();
            temp_strings.push_back(s);
            free(temp_data);
        } else {
            temp_strings.push_back("");
        }
    }
    
    size_t string_dim = max_len + 1;
    size_t num_strings = temp_strings.size();
    size_t total_bytes = num_strings * string_dim;

    auto final_buffer = std::shared_ptr<char[]>(new char[total_bytes], std::default_delete<char[]>());
    memset(final_buffer.get(), 0, total_bytes);

    for (size_t i = 0; i < num_strings; ++i) {
        strncpy(final_buffer.get() + i * string_dim, temp_strings[i].c_str(), max_len);
    }
    
    std::vector<size_t> final_dims = selection_dims;
    if (final_dims.empty()) final_dims.push_back(num_strings);
    final_dims.push_back(string_dim);
    
    return TensorView(std::move(final_buffer), final_dims, DataType::LIST_OF_STRINGS, metadata);
}

TensorView read_tensor_impl_core(PanzerDB& db, const std::vector<PathSegment>& segments, const std::map<std::string, NodeType>& aos_paths, const std::map<std::string, size_t>& aos_sizes)
{
    std::vector<std::pair<std::string, int64_t>> leaf_paths;
    std::vector<size_t> selection_dims;
    
    collect_leaf_paths_recursive(db, segments, 0, "", "", -1, leaf_paths, selection_dims, aos_paths, aos_sizes);

    if (leaf_paths.empty()) {
        return TensorView();
    }

    auto metadata = db.readMetadata(leaf_paths[0].first);
    DataType data_type = db.get_leaf_type(leaf_paths[0].first);
    
    switch (data_type) {
        case DataType::DOUBLE:
            return read_typed_tensor<double>(db, leaf_paths, selection_dims, data_type, metadata);
        case DataType::INT32:
            return read_typed_tensor<int>(db, leaf_paths, selection_dims, data_type, metadata);
        case DataType::COMPLEX_DOUBLE:
            return read_typed_tensor<std::complex<double>>(db, leaf_paths, selection_dims, data_type, metadata);
        case DataType::STRING:
        case DataType::LIST_OF_STRINGS:
            return read_list_of_strings(db, leaf_paths, selection_dims, metadata);
        default:
            throw std::runtime_error("Unsupported data type for direct tensor read.");
    }
}

TensorView read_tensor(const std::string& ids_name, const std::string& path)
{
    PathParser parser(path);
    const auto& segments = parser.segments();

    auto list_result = list_nodes(ids_name, true, true, false);
    const auto& aos_paths = list_result.second;
    
    PanzerDB db(ids_name + ".h5", PanzerDB::OpenMode::READ);
    
    std::map<std::string, size_t> aos_sizes;
    for (const auto& leaf : db.getLeaves()) {
        if (leaf.flags == 2 || leaf.flags == 3) {
            std::string schema_path = PanzerDB::stripIndices(std::string(leaf.path));
            if (leaf.flags == 2 && !leaf.shape.empty()) aos_sizes[schema_path] = leaf.shape[0];
            else if (leaf.flags == 3) {
                auto s = db.getAOSShape(std::string(leaf.path));
                aos_sizes[schema_path] = s.empty() ? 0 : s.front();
            }
        }
    }
    
    return read_tensor_impl_core(db, segments, aos_paths, aos_sizes);
}

// Stub for remaining overload
TensorView read_tensor(const std::string& ids_name, const std::string& path_template, const std::vector<int>& aos_indices) {
    return read_tensor(ids_name, path_template);
}

} // namespace direct_access
} // namespace imas
