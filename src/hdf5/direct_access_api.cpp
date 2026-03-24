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

// Déclaration anticipée pour résoudre l'ambiguïté de nom avec uri_parser.h
//std::vector<PathSegment> parse_path(const std::string& path);

TensorView read_tensor_impl_core(PanzerDB& db, const std::vector<PathSegment>& segments);


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

// Helper function to resolve the time vector based on the rules.
std::vector<double> get_time_vector(PanzerDB& db, const std::string& aos_path, size_t aos_size) {
    int status = 0;
    int homogeneous_time = 1; // Default to homogeneous if property is not found
    try {
        homogeneous_time = db.readScalar<int32_t>("ids_properties/homogeneous_time", &status);
        if (status != 0) homogeneous_time = 1; // Fallback on read error
    } catch (const std::exception&) {
        // Fallback if 'ids_properties/homogeneous_time' does not exist
        homogeneous_time = 1;
    }
    db.dumpLeavesCache();
    if (db.is_dynamic_aos(aos_path)) {
        // Rule: For a dynamic AoS, the time of each slice is at <aos_path>/<i>/time,
        // regardless of the homogeneous_time flag.
        std::vector<double> time_values;
        time_values.reserve(aos_size);
        for (size_t i = 0; i < aos_size; ++i) {
            std::string time_path = aos_path + "/" + std::to_string(i) + "/time";
            double time_val = db.readScalar<double>(time_path, &status);
            if (status == 0) {
                time_values.push_back(time_val);
            } else {
                // If a time value is missing, we cannot proceed.
                throw std::runtime_error("Could not read time value at path: " + time_path);
            }
        }
        return time_values;
    } else {
        // Logic for a static AoS
        if (homogeneous_time == 1) {
            return db.getWholeDynamicSignal("time");
        } else if (homogeneous_time == 2) {
            return {}; // No time base
        } else { // Inhomogeneous time for static AoS
            throw std::runtime_error("Time-slicing on a static AoS with inhomogeneous time is not yet supported.");
        }
    }
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
            // Une sélection par INDEX unique supprime la dimension, on n'ajoute rien à selection_dims.
        } else if (seg.selection == SelectionType::SLICE || seg.selection == SelectionType::ALL) {
            start = seg.has_start ? seg.start_index : 0;
            end = seg.has_end ? seg.end_index : aos_size;
            // CORRECTION: Ajouter la dimension uniquement si c'est le premier passage à ce niveau
            if (selection_dims.size() <= segment_idx) {
                selection_dims.push_back(end - start);
            }
        } else if (seg.selection == SelectionType::TIME) {
            std::vector<double> time_values = get_time_vector(db, new_path_base, aos_size);
            DataInterpolation interpolator;
            std::map<std::string, int> times_indices;

            if (seg.interp == InterpolationMethod::NONE) {
                start = seg.has_start_time ? interpolator.getSlicesTimesIndices(seg.start_time, time_values, times_indices, CLOSEST_INTERP) : 0;
                end = seg.has_end_time ? interpolator.getSlicesTimesIndices(seg.end_time, time_values, times_indices, CLOSEST_INTERP) + 1 : aos_size;
                if (selection_dims.size() <= segment_idx) {
                    selection_dims.push_back(end - start);
                }
            } else {
                start = interpolator.getSlicesTimesIndices(seg.start_time, time_values, times_indices, CLOSEST_INTERP);
                end = start + 1;
                // L'interpolation produit une seule tranche, la dimension est supprimée (ou de taille 1).
            }
        }
        
        for (size_t i = start; i < end; ++i) {
            std::string indexed_path = new_path_base + "/" + std::to_string(i);
            collect_leaf_paths_recursive(db, segments, segment_idx + 1, indexed_path, leaf_paths, selection_dims);
        }
    } else {
        std::vector<size_t> aos_shape = db.getAOSShape(new_path_base);
        if (!aos_shape.empty() && segment_idx < segments.size() - 1) {
            size_t aos_size = aos_shape[0];
            // CORRECTION: Ajouter la dimension uniquement si c'est le premier passage à ce niveau
            if (selection_dims.size() <= segment_idx) {
                selection_dims.push_back(aos_size);
            }
            for (size_t i = 0; i < aos_size; ++i) {
                std::string indexed_path = new_path_base + "/" + std::to_string(i);
                collect_leaf_paths_recursive(db, segments, segment_idx + 1, indexed_path, leaf_paths, selection_dims);
            }
        } else {
            // Dans le cas d'une descente sans AOS, on assure la cohérence des indices de dimensions
            if (selection_dims.size() <= segment_idx) {
                // On pourrait ajouter un placeholder ici si nécessaire, 
                // mais pour l'instant on se contente de continuer.
            }
            collect_leaf_paths_recursive(db, segments, segment_idx + 1, new_path_base, leaf_paths, selection_dims);
        }
    }
}

template <typename T>
TensorView read_typed_tensor(
    PanzerDB& db,
    const std::vector<std::string>& leaf_paths,
    const std::vector<size_t>& final_dims,
    DataType data_type,
    const std::map<std::string, std::string>& metadata)
{
    size_t total_elements = std::accumulate(final_dims.begin(), final_dims.end(), 1, std::multiplies<size_t>());
    if (total_elements == 0 && !leaf_paths.empty()) total_elements = leaf_paths.size();
    
    size_t total_bytes = total_elements * sizeof(T);
    auto final_buffer = std::shared_ptr<char[]>(new char[total_bytes], std::default_delete<char[]>());
    T* final_data_ptr = reinterpret_cast<T*>(final_buffer.get());

    size_t global_offset = 0;
    for (size_t i = 0; i < leaf_paths.size(); ++i) {
        uint64_t ndim = 0;
        uint64_t shape[6] = {0};
        
        if constexpr (std::is_same_v<T, double>) {
            double* temp_data = nullptr;
            if (db.pz_readData_by_index(leaf_paths[i].c_str(), -1, &ndim, shape, &temp_data) == 0 && temp_data) {
                size_t count = 1;
                for(uint64_t d=0; d<ndim; ++d) count *= shape[d];
                size_t to_copy = std::min(count, total_elements - global_offset);
                std::copy(temp_data, temp_data + to_copy, final_data_ptr + global_offset);
                global_offset += to_copy;
                free(temp_data);
            }
        } else if constexpr (std::is_same_v<T, int>) {
            int32_t* temp_data = nullptr;
            if (db.pz_readIntData_by_index(leaf_paths[i].c_str(), -1, &ndim, shape, &temp_data) == 0 && temp_data) {
                size_t count = 1;
                for(uint64_t d=0; d<ndim; ++d) count *= shape[d];
                size_t to_copy = std::min(count, total_elements - global_offset);
                std::copy(temp_data, temp_data + to_copy, final_data_ptr + global_offset);
                global_offset += to_copy;
                free(temp_data);
            }
        }
    }
    
    return TensorView(std::move(final_buffer), final_dims, data_type, metadata);
}



// Nouvelle fonction dédiée à la lecture des listes de chaînes de caractères
TensorView read_list_of_strings(
    PanzerDB& db,
    const std::vector<std::string>& leaf_paths,
    const std::vector<size_t>& selection_dims,
    const std::map<std::string, std::string>& metadata)
{
    std::vector<std::string> temp_strings;
    size_t max_len = 0;

    // 1. Lire toutes les chaînes et trouver la longueur maximale
    for (const auto& path : leaf_paths) {
        char* temp_data = nullptr;
        uint64_t ndim = 0;
        uint64_t shape[6] = {0};
        int status = db.pz_readStringData_by_index(path.c_str(), -1, &ndim, shape, &temp_data);
        if (status == 0 && temp_data != nullptr) {
            std::string s(temp_data);
            if (s.length() > max_len) {
                max_len = s.length();
            }
            temp_strings.push_back(s);
            free(temp_data);
        } else {
            temp_strings.push_back("");
        }
    }
    
    size_t string_dim = max_len + 1; // +1 pour le caractère nul
    size_t num_strings = temp_strings.size();
    size_t total_bytes = num_strings * string_dim;

    // 2. Allouer le buffer final et copier les données
    auto final_buffer = std::shared_ptr<char[]>(new char[total_bytes], std::default_delete<char[]>());
    memset(final_buffer.get(), 0, total_bytes);

    for (size_t i = 0; i < num_strings; ++i) {
        strncpy(final_buffer.get() + i * string_dim, temp_strings[i].c_str(), max_len);
    }
    
    // 3. Construire les dimensions finales (2D)
    std::vector<size_t> final_dims = selection_dims;
    if (final_dims.empty()) { // Si aucune sélection n'a défini les dims
        final_dims.push_back(num_strings);
    }
    final_dims.push_back(string_dim);
    
    return TensorView(std::move(final_buffer), final_dims, DataType::LIST_OF_STRINGS, metadata);
}

// Fonction corrigée pour gérer la logique d'interpolation linéaire
TensorView read_interpolated_tensor(
    PanzerDB& db,
    const std::vector<PathSegment>& segments)
{
    // 1. Trouver le segment qui demande l'interpolation
    const PathSegment* interp_segment = nullptr;
    size_t interp_segment_idx = -1;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (segments[i].interp == InterpolationMethod::LINEAR) {
            interp_segment = &segments[i];
            interp_segment_idx = i;
            break;
        }
    }
    if (!interp_segment) {
        throw std::runtime_error("read_interpolated_tensor appelé sans segment d'interpolation linéaire.");
    }

    // 2. Déterminer le chemin de l'AoS et récupérer le vecteur de temps complet
    std::string aos_path;
    for (size_t i = 0; i <= interp_segment_idx; ++i) {
        if (i > 0) {
            aos_path += "/";
        }
        aos_path += segments[i].node_name;
    }
    size_t aos_size = db.getAOSShape(aos_path)[0];
    std::vector<double> time_values = get_time_vector(db, aos_path, aos_size);
    
    if (time_values.empty()) {
        throw std::runtime_error("Impossible de récupérer le vecteur de temps pour l'interpolation.");
    }

    // 3. CORRECTION: Appeler getSlicesTimesIndices une seule fois et utiliser la map
    DataInterpolation interpolator;
    std::map<std::string, int> times_indices; // Cette map sera remplie par la fonction
    interpolator.getSlicesTimesIndices(interp_segment->start_time, time_values, times_indices, LINEAR_INTERP);

    int slice_inf = times_indices["slice_inf"];
    int slice_sup = times_indices["slice_sup"];

    if (slice_inf < 0 || slice_sup < 0) {
        throw std::runtime_error("Impossible de trouver les temps encadrants pour l'interpolation.");
    }

    if (slice_inf == slice_sup) {
        std::vector<PathSegment> new_segments = segments;
        new_segments[interp_segment_idx].selection = SelectionType::INDEX;
        new_segments[interp_segment_idx].index = slice_inf;
        new_segments[interp_segment_idx].interp = InterpolationMethod::NONE;
        return read_tensor_impl_core(db, new_segments);
    }
    
    // 4. Lire les données pour les deux tranches de temps
    std::vector<PathSegment> segments_inf = segments;
    segments_inf[interp_segment_idx].selection = SelectionType::INDEX;
    segments_inf[interp_segment_idx].index = slice_inf;
    segments_inf[interp_segment_idx].interp = InterpolationMethod::NONE;
    TensorView view_inf = read_tensor_impl_core(db, segments_inf);
    
    std::vector<PathSegment> segments_sup = segments;
    segments_sup[interp_segment_idx].selection = SelectionType::INDEX;
    segments_sup[interp_segment_idx].index = slice_sup;
    segments_sup[interp_segment_idx].interp = InterpolationMethod::NONE;
    TensorView view_sup = read_tensor_impl_core(db, segments_sup);

    // 5. Effectuer l'interpolation avec les vraies valeurs de temps
    std::map<std::string, void*> y_slices;
    y_slices["slice_inf"] = view_inf.data();
    y_slices["slice_sup"] = view_sup.data();

    std::map<std::string, double> slices_times;
    slices_times["slice_inf"] = time_values[slice_inf];
    slices_times["slice_sup"] = time_values[slice_sup];
    
    void* interpolated_data = nullptr;
    size_t total_elements = view_inf.total_elements();
    int data_type_code = (view_inf.type() == DataType::DOUBLE) ? DOUBLE_DATA : -1;
    
    interpolator.interpolate(data_type_code, total_elements, y_slices, slices_times, interp_segment->start_time, &interpolated_data, LINEAR_INTERP);

    // 6. Créer le TensorView final
    auto final_buffer = std::shared_ptr<char[]>(reinterpret_cast<char*>(interpolated_data), [](char* p){ free(p); });
    std::vector<size_t> final_dims = view_inf.dims();
    
    return TensorView(std::move(final_buffer), final_dims, view_inf.type(), view_inf.metadata());
}

std::pair<std::vector<NodeInfo>, std::map<std::string, NodeType>>
list_nodes(const std::string& ids_name, bool recursive, bool show_aos, bool show_metadata) {
    PanzerDB db(ids_name + ".h5", PanzerDB::OpenMode::READ);
    const auto& leaves = db.getLeaves();

    std::map<std::string, NodeType> aos_schemas;
    std::map<std::string, const PanzerDB::Leaf*> dataset_schemas;

    // 1. Identification
    for (const auto& leaf : leaves) {
        std::string schema = PanzerDB::stripIndices(std::string(leaf.path));
        if (leaf.flags == 2 || leaf.flags == 3) {
            aos_schemas[schema] = (leaf.flags == 3) ? NodeType::AOS_DYNAMIC : NodeType::AOS_STATIC;
        } else if ((leaf.flags & 0xF) == 0) {
            if (!show_metadata && schema.find('@') != std::string::npos) continue;
            if (dataset_schemas.find(schema) == dataset_schemas.end()) dataset_schemas[schema] = &leaf;
        }
    }

    std::vector<NodeInfo> result;

    // 2. Build Dataset Info
    for (const auto& pair : dataset_schemas) {
        const std::string& schema_path = pair.first;
        const PanzerDB::Leaf* rep_leaf = pair.second;
        
        std::vector<size_t> logical_dims;
        bool is_dyn = false;
        
        std::string instance_prefix;
        std::stringstream ss(schema_path);
        std::string segment;
        std::string current_schema;
        
        while(std::getline(ss, segment, '/') && ss.peek() != EOF) {
            current_schema += (current_schema.empty() ? "" : "/") + segment;
            if (aos_schemas.count(current_schema)) {
                // Use getAOSShape on a representative instance path
                auto shape = db.getAOSShape(instance_prefix + segment);
                if (!shape.empty()) logical_dims.push_back(shape[0]);
                if (aos_schemas[current_schema] == NodeType::AOS_DYNAMIC) is_dyn = true;
                instance_prefix += segment + "/0/";
            } else {
                instance_prefix += segment + "/";
            }
        }
        
        // Root dynamic datasets (like 'time')
        if (logical_dims.empty()) {
            size_t total = 0;
            for(const auto& l : leaves) if (PanzerDB::stripIndices(std::string(l.path)) == schema_path) total += l.count;
            if (total > 1) { logical_dims.push_back(total); is_dyn = true; }
        }
        
        // Add leaf shape if not scalar (ignores {1} or {})
        if (!rep_leaf->shape.empty() && !(rep_leaf->shape.size() == 1 && rep_leaf->shape[0] <= 1)) {
            logical_dims.insert(logical_dims.end(), rep_leaf->shape.begin(), rep_leaf->shape.end());
        }

        result.push_back({schema_path, NodeType::DATASET, logical_dims, is_dyn});
    }

    if (show_aos) {
        for (const auto& p : aos_schemas) result.push_back({p.first, p.second, {}, false});
    }
    
    std::sort(result.begin(), result.end(), [](const NodeInfo& a, const NodeInfo& b) { return a.path < b.path; });
    
    if (!recursive) {
        std::vector<NodeInfo> filtered;
        for (const auto& n : result) if (n.path.find('/') == std::string::npos) filtered.push_back(n);
        return {filtered, aos_schemas};
    }
    return {result, aos_schemas};
}

TensorView read_tensor_impl(const std::string& ids_name, const std::vector<PathSegment>& segments)
{
    PanzerDB db(ids_name + ".h5", PanzerDB::OpenMode::READ);

    bool needs_linear_interp = false;
    for (const auto& seg : segments) {
        if (seg.interp == InterpolationMethod::LINEAR) {
            needs_linear_interp = true;
            break;
        }
    }

    if (needs_linear_interp) {
        return read_interpolated_tensor(db, segments);
    } else {
        return read_tensor_impl_core(db, segments);
    }
}

TensorView read_tensor_impl_core(PanzerDB& db, const std::vector<PathSegment>& segments)
{
    std::vector<std::string> leaf_paths;
    std::vector<size_t> selection_dims;
    collect_leaf_paths(db, segments, leaf_paths, selection_dims);

    if (leaf_paths.empty()) {
        return TensorView();
    }

    // --- NOUVEAU: Lire les métadonnées ---
    auto metadata = db.readMetadata(leaf_paths[0]);

    DataType data_type = db.get_leaf_type(leaf_paths[0]);
    std::vector<size_t> final_dims = selection_dims;
    
    switch (data_type) {
        case DataType::DOUBLE:
            return read_typed_tensor<double>(db, leaf_paths, final_dims, data_type, metadata); // Passe metadata
        case DataType::INT32:
            return read_typed_tensor<int>(db, leaf_paths, final_dims, data_type, metadata); // Passe metadata
        case DataType::COMPLEX_DOUBLE:
            return read_typed_tensor<std::complex<double>>(db, leaf_paths, final_dims, data_type, metadata); // Passe metadata
        case DataType::STRING:
        case DataType::LIST_OF_STRINGS:
            return read_list_of_strings(db, leaf_paths, selection_dims, metadata); // Passe metadata
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
