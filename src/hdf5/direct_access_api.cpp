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
        } else if (seg.selection == SelectionType::SLICE || seg.selection == SelectionType::ALL) {
            start = seg.has_start ? seg.start_index : 0;
            end = seg.has_end ? seg.end_index : aos_size;
        } else if (seg.selection == SelectionType::TIME) {
            std::vector<double> time_values = get_time_vector(db, new_path_base, aos_size);
            if (time_values.empty()) {
                throw std::runtime_error("Could not resolve time vector for path: " + new_path_base);
            }

            DataInterpolation interpolator;
            std::map<std::string, int> times_indices; // Dummy map required by the interpolator API

            if (seg.interp == InterpolationMethod::NONE) { // Time range [start:end]
                start = seg.has_start_time ? interpolator.getSlicesTimesIndices(seg.start_time, time_values, times_indices, CLOSEST_INTERP) : 0;
                end = seg.has_end_time ? interpolator.getSlicesTimesIndices(seg.end_time, time_values, times_indices, CLOSEST_INTERP) + 1 : aos_size;
            } else { // Interpolation point (CLOSEST or LINEAR)
                start = interpolator.getSlicesTimesIndices(seg.start_time, time_values, times_indices, CLOSEST_INTERP);
                end = start + 1;
            }
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
    
    auto final_buffer = std::shared_ptr<char[]>(new char[total_bytes], std::default_delete<char[]>());
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
        } else if constexpr (std::is_same_v<T, int>) {
            int32_t* temp_data = nullptr;
            int status = db.pz_readIntData_by_index(leaf_paths[i].c_str(), -1, &ndim, shape, &temp_data);
            if (status == 0 && temp_data != nullptr) {
                final_data_ptr[i] = temp_data[0];
                free(temp_data);
            } else {
                final_data_ptr[i] = 0; // Valeur par défaut en cas d'erreur
            }
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            std::complex<double>* temp_data = nullptr;
            int status = db.pz_readComplexData_by_index(leaf_paths[i].c_str(), -1, &ndim, shape, &temp_data);
            if (status == 0 && temp_data != nullptr) {
                final_data_ptr[i] = temp_data[0];
                free(temp_data);
            } else {
                final_data_ptr[i] = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
            }
        } else if constexpr (std::is_same_v<T, std::string>) {
            char* temp_data = nullptr;
            int status = db.pz_readStringData_by_index(leaf_paths[i].c_str(), -1, &ndim, shape, &temp_data);
             if (status == 0 && temp_data != nullptr) {
                final_data_ptr[i] = temp_data;
                free(temp_data);
            } else {
                final_data_ptr[i] = "";
            }
        }
    }
    
    return TensorView(std::move(final_buffer), final_dims, data_type);
}

// Nouvelle fonction dédiée à la lecture des listes de chaînes de caractères
TensorView read_list_of_strings(
    PanzerDB& db,
    const std::vector<std::string>& leaf_paths,
    const std::vector<size_t>& selection_dims)
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
    
    return TensorView(std::move(final_buffer), final_dims, DataType::LIST_OF_STRINGS);
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
    
    return TensorView(std::move(final_buffer), final_dims, view_inf.type());
}

// Le nouveau "chef d'orchestre"
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

    DataType data_type = db.get_leaf_type(leaf_paths[0]);

    std::vector<size_t> final_dims = selection_dims;
    
    switch (data_type) {
        case DataType::DOUBLE:
            return read_typed_tensor<double>(db, leaf_paths, final_dims, data_type);
        case DataType::INT32:
            return read_typed_tensor<int>(db, leaf_paths, final_dims, data_type);
        case DataType::COMPLEX_DOUBLE:
            return read_typed_tensor<std::complex<double>>(db, leaf_paths, final_dims, data_type);
    
        // CORRECTION: Dévier tous les types de chaînes vers la fonction correcte
        case DataType::STRING:
        case DataType::LIST_OF_STRINGS:
            return read_list_of_strings(db, leaf_paths, selection_dims);
    
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
