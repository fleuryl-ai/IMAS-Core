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

// Helper ultra-robuste pour trouver la taille réelle d'un AoS en inspectant les chemins
size_t get_actual_aos_size(PanzerDB& db, const std::string& aos_path) {
    size_t max_idx = 0;
    bool found = false;
    std::string search_str = aos_path + "/";
    for (const auto& leaf : db.getLeaves()) {
        if (leaf.path.compare(0, search_str.length(), search_str) == 0) {
            std::string remainder = std::string(leaf.path).substr(search_str.length());
            size_t slash_pos = remainder.find('/');
            std::string idx_str = (slash_pos != std::string::npos) ? remainder.substr(0, slash_pos) : remainder;
            if (!idx_str.empty() && std::isdigit(idx_str[0])) {
                try {
                    size_t idx = std::stoul(idx_str);
                    if (idx >= max_idx) { max_idx = idx; found = true; }
                } catch(...) {}
            }
        }
    }
    if (found) return max_idx + 1;
    
    auto shape = db.getAOSShape(aos_path);
    return shape.empty() ? 0 : shape[0];
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

    // --- LOGIQUE DE PATCH CIBLÉE ---
    // Cette fonction est appelée récursivement. À chaque étape, 'new_path_base' est le chemin du "parent".
    // On vérifie s'il est un AoS dynamique.
    bool is_parent_dyn_aos = db.is_dynamic_aos(new_path_base);

    // On vérifie si son "enfant" (le segment suivant dans le chemin) est le dernier segment (c'est donc un champ).
    bool is_child_a_field = (segment_idx + 1 == segments.size() - 1);

    // La condition pour NE PAS ajouter d'indice est : si le parent est dynamique ET que l'enfant est un champ.
    bool should_skip_indexing = (is_parent_dyn_aos && is_child_a_field);
    
    // --- FIN DE LA LOGIQUE DE PATCH ---

    if (seg.selection != SelectionType::NONE) {
        size_t start = 0;
        size_t end = 0;

        size_t aos_size = get_actual_aos_size(db, new_path_base);
        if (aos_size == 0 && seg.selection != SelectionType::INDEX) {
            throw std::runtime_error("Could not determine size of AoS for path: " + new_path_base);
        }

        if (seg.selection == SelectionType::INDEX) {
            start = seg.index;
            end = start + 1;
        } else if (seg.selection == SelectionType::SLICE || seg.selection == SelectionType::ALL) {
            start = seg.has_start ? seg.start_index : 0;
            end = seg.has_end ? seg.end_index : aos_size;
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
                if (selection_dims.size() <= segment_idx) selection_dims.push_back(end - start);
            } else {
                start = interpolator.getSlicesTimesIndices(seg.start_time, time_values, times_indices, CLOSEST_INTERP);
                end = start + 1;
            }
        }
        
        // Appliquer la règle : si on ne doit pas indexer (cas sig_dyn), on continue sans ajouter /i.
        if (should_skip_indexing) {
             collect_leaf_paths_recursive(db, segments, segment_idx + 1, new_path_base, leaf_paths, selection_dims);
        } else {
             for (size_t i = start; i < end; ++i) {
                std::string indexed_path = new_path_base + "/" + std::to_string(i);
                collect_leaf_paths_recursive(db, segments, segment_idx + 1, indexed_path, leaf_paths, selection_dims);
            }
        }

    } else { // Pas de sélection sur ce segment
        size_t aos_size = get_actual_aos_size(db, new_path_base);
        if (aos_size > 0 && segment_idx < segments.size() - 1) {
            
            // Appliquer la règle ici aussi
            if (should_skip_indexing) {
                collect_leaf_paths_recursive(db, segments, segment_idx + 1, new_path_base, leaf_paths, selection_dims);
            } else {
                if (selection_dims.size() <= segment_idx) {
                    selection_dims.push_back(aos_size);
                }
                for (size_t i = 0; i < aos_size; ++i) {
                    std::string indexed_path = new_path_base + "/" + std::to_string(i);
                    collect_leaf_paths_recursive(db, segments, segment_idx + 1, indexed_path, leaf_paths, selection_dims);
                }
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

// Fonction dédiée à la lecture des listes de chaînes de caractères
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

    // 3. Appeler getSlicesTimesIndices une seule fois et utiliser la map
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

    // 5. Effectuer l'interpolation
    // Créer des copies manuelles pour éviter le double free
    void* data_inf_copy = malloc(view_inf.size_in_bytes());
    if (data_inf_copy) memcpy(data_inf_copy, view_inf.data(), view_inf.size_in_bytes());

    void* data_sup_copy = malloc(view_sup.size_in_bytes());
    if (data_sup_copy) memcpy(data_sup_copy, view_sup.data(), view_sup.size_in_bytes());
    
    std::map<std::string, void*> y_slices;
    y_slices["slice_inf"] = data_inf_copy;
    y_slices["slice_sup"] = data_sup_copy;

    std::map<std::string, double> slices_times;
    slices_times["slice_inf"] = time_values[slice_inf];
    slices_times["slice_sup"] = time_values[slice_sup];
    
    void* interpolated_data = nullptr;
    size_t total_elements = view_inf.total_elements();
    int data_type_code = (view_inf.type() == DataType::DOUBLE) ? alconst::double_data : -1;
    
    interpolator.interpolate(data_type_code, total_elements, y_slices, slices_times, interp_segment->start_time, &interpolated_data, LINEAR_INTERP);

    // 6. Créer le TensorView final avec un deleter personnalisé
    auto final_buffer = std::shared_ptr<char[]>(reinterpret_cast<char*>(interpolated_data), [](char* p){ if(p) free(p); });
    std::vector<size_t> final_dims = view_inf.dims();
    
    return TensorView(std::move(final_buffer), final_dims, view_inf.type(), view_inf.metadata());
}


std::pair<std::vector<NodeInfo>, std::map<std::string, NodeType>>
list_nodes(const std::string& ids_name, bool recursive, bool show_aos, bool show_metadata) {
    PanzerDB db(ids_name + ".h5", PanzerDB::OpenMode::READ);
    const auto& leaves = db.getLeaves();

    std::map<std::string, NodeType> aos_paths;
    std::map<std::string, const PanzerDB::Leaf*> schema_to_leaf_map;

    // 1. Collecter les schémas et identifier les AOS de manière fiable
    for (const auto& leaf : leaves) {
        if (leaf.flags == 2 || leaf.flags == 3) {
            std::string schema_aos = PanzerDB::stripIndices(std::string(leaf.path));
            aos_paths[schema_aos] = (leaf.flags == 3) ? NodeType::AOS_DYNAMIC : NodeType::AOS_STATIC;
        }
        
        if ((leaf.flags & 0xF) != 0) continue; // Uniquement les données
        std::string schema_path = PanzerDB::stripIndices(std::string(leaf.path));
        if (!show_metadata && schema_path.find('@') != std::string::npos) continue;

        if (schema_to_leaf_map.find(schema_path) == schema_to_leaf_map.end()) {
            schema_to_leaf_map[schema_path] = &leaf;
        }
    }

    std::vector<NodeInfo> result;

    // 2. Construire le NodeInfo pour les datasets
    for (const auto& pair : schema_to_leaf_map) {
        const std::string& schema_path = pair.first;
        const PanzerDB::Leaf* rep_leaf = pair.second;
        
        std::vector<size_t> logical_dims;
        bool is_in_dynamic_aos = false;
        bool is_metadata = (schema_path.find('@') != std::string::npos);
        
        // Les métadonnées sont toujours scalaires globales, on ne remonte pas les AOS pour elles
        if (!is_metadata) {
            std::string schema_prefix;
            std::string instance_prefix;
            std::stringstream ss(schema_path);
            std::string segment;
            
            while(std::getline(ss, segment, '/') && ss.peek() != EOF) {
                schema_prefix += (schema_prefix.empty() ? "" : "/") + segment;
                std::string query_path = instance_prefix + segment;

                if (aos_paths.count(schema_prefix)) {
                    if (aos_paths.at(schema_prefix) == NodeType::AOS_DYNAMIC) {
                        is_in_dynamic_aos = true;
                    }
                    size_t aos_size = get_actual_aos_size(db, query_path);
                    if (aos_size > 0) logical_dims.push_back(aos_size);
                    
                    instance_prefix += segment + "/0/";
                } else {
                    instance_prefix += segment + "/";
                }
            }
            
            // Cas du signal dynamique autonome (ex: "time" global)
            if (logical_dims.empty() && !is_in_dynamic_aos) {
                size_t total_count = 0;
                for(const auto& leaf : leaves) {
                    if ((leaf.flags & 0xF) == 0 && PanzerDB::stripIndices(std::string(leaf.path)) == schema_path) {
                        total_count += leaf.count;
                    }
                }
                if (total_count > 1) {
                    logical_dims.push_back(total_count);
                    is_in_dynamic_aos = true; // Force l'affichage du /Inf
                }
            }
            
            // Ajouter la forme intrinsèque si ce n'est pas un scalaire
            bool is_scalar_leaf = rep_leaf->shape.empty() || (rep_leaf->shape.size() == 1 && rep_leaf->shape[0] <= 1);
            if (!is_scalar_leaf) {
                logical_dims.insert(logical_dims.end(), rep_leaf->shape.begin(), rep_leaf->shape.end());
            }
        }

        result.push_back({schema_path, NodeType::DATASET, logical_dims, is_in_dynamic_aos});
    }

    // 3. Ajouter les AOS si demandé
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
