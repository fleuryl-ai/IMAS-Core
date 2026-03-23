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

namespace imas {
namespace direct_access {

// Déclaration anticipée de la fonction récursive
void collect_leaf_paths_recursive(
    PanzerDB& db,
    const std::vector<PathSegment>& segments,
    size_t segment_idx,
    std::string current_path,
    std::vector<std::string>& leaf_paths,
    std::vector<size_t>& selection_dims);


/**
 * @brief Pièce centrale : résout récursivement l'arbre de segments en une liste
 * plate de chemins de feuilles, en interrogeant la taille des AoS à la volée.
 */
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
    std::string new_path = current_path.empty() ? seg.name : current_path + "/" + seg.name;

    if (seg.selection == SelectionType::NONE) {
        collect_leaf_paths_recursive(db, segments, segment_idx + 1, new_path, leaf_paths, selection_dims);
    } else {
        size_t start = 0;
        size_t end = 0;

        if (seg.selection == SelectionType::INDEX) {
            start = seg.start_index;
            end = start + 1;
        } else { // ALL ou SLICE
            // NOTE: C'est ici que nous interrogeons PanzerDB pour la taille de l'AoS
            // Pour l'instant, on utilise une taille fixe en attendant la vraie API.
            std::vector<size_t> aos_shape = {10}; // db.getAOSShape(new_path);
            if (aos_shape.empty()) {
                throw AlException(NOT_FOUND, "Could not determine size of AoS for path: " + new_path);
            }
            size_t aos_size = aos_shape[0];

            start = seg.has_start ? seg.start_index : 0;
            end = seg.has_end ? seg.end_index : aos_size;
            
            if (segment_idx == 0) { // Uniquement pour la première dimension de sélection
                 selection_dims.push_back(end - start);
            }
        }
        
        for (size_t i = start; i < end; ++i) {
            std::string indexed_path = new_path + "[" + std::to_string(i) + "]";
            collect_leaf_paths_recursive(db, segments, segment_idx + 1, indexed_path, leaf_paths, selection_dims);
        }
    }
}


/**
 * @brief Helper template pour lire les données typées et construire le TensorView.
 * Appelle la surcharge pz_read... appropriée en fonction du type.
 */
template <typename T>
TensorView read_typed_tensor(
    PanzerDB& db,
    const std::vector<std::string>& leaf_paths,
    const std::vector<size_t>& final_dims,
    DataType data_type)
{
    size_t total_elements = std::accumulate(final_dims.begin(), final_dims.end(), 1, std::multiplies<size_t>());
    if (leaf_paths.size() != total_elements && total_elements > 0) {
        // Cas simple : la sélection a "aplati" la lecture.
        // La taille des leaf_paths correspond au nombre d'éléments.
        total_elements = leaf_paths.size();
    }


    auto data_buffer = std::make_shared<std::vector<T>>(total_elements);
    T* buffer_ptr = data_buffer->data();

    for (size_t i = 0; i < leaf_paths.size(); ++i) {
        if constexpr (std::is_same_v<T, double>) {
            db.pz_readData_by_index(leaf_paths[i], buffer_ptr + i, 0, 1);
        } else if constexpr (std::is_same_v<T, int>) {
            // Supposition: db.pz_readIntData_by_index(leaf_paths[i], buffer_ptr + i, 0, 1);
            throw AlException(UNSUPPORTED_FEATURE, "Reading INT32 data is not yet implemented.");
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            // Supposition: db.pz_readComplexData_by_index(leaf_paths[i], reinterpret_cast<double*>(buffer_ptr + i), 0, 1);
            throw AlException(UNSUPPORTED_FEATURE, "Reading COMPLEX128 data is not yet implemented.");
        } else if constexpr (std::is_same_v<T, std::string>) {
            throw AlException(UNSUPPORTED_FEATURE, "Reading STRING data is not yet implemented.");
        }
    }
    
    return TensorView(data_buffer, final_dims, data_type);
}


TensorView read_tensor_impl(const std::string& ids_name, const std::vector<PathSegment>& segments)
{
    PanzerDB& db = PanzerDB::get_instance(ids_name, "r");

    // 1. Collecter les chemins de feuilles plats
    std::vector<std::string> leaf_paths;
    std::vector<size_t> selection_dims;
    collect_leaf_paths(db, segments, leaf_paths, selection_dims);

    if (leaf_paths.empty()) {
        return TensorView(); // Retourne un tenseur vide si aucun chemin n'est trouvé
    }

    // 2. Déterminer le type de la donnée depuis le premier chemin (en supposant qu'ils sont tous du même type)
    // NOTE: Dépendance clé sur une nouvelle méthode de PanzerDB
    // DataType data_type = db.get_leaf_type(leaf_paths[0]);
    DataType data_type = DataType::DOUBLE; // Pour l'instant, on fixe le type à DOUBLE

    // 3. Calculer les dimensions finales
    std::vector<size_t> final_dims = selection_dims;
    // NOTE: Ici, on pourrait ajouter les dimensions intrinsèques de la feuille si elle n'est pas un scalaire.
    // std::vector<size_t> intrinsic_dims = db.get_leaf_dims(leaf_paths[0]);
    // final_dims.insert(final_dims.end(), intrinsic_dims.begin(), intrinsic_dims.end());
    if (final_dims.empty()) { // Si c'est un scalaire, la dimension est 1
        final_dims.push_back(1);
    }
    
    // 4. Dispatch vers le lecteur typé approprié
    switch (data_type) {
        case DataType::DOUBLE:
            return read_typed_tensor<double>(db, leaf_paths, final_dims, data_type);
        case DataType::INT32:
            return read_typed_tensor<int>(db, leaf_paths, final_dims, data_type);
        case DataType::COMPLEX128:
            return read_typed_tensor<std::complex<double>>(db, leaf_paths, final_dims, data_type);
        case DataType::STRING:
            return read_typed_tensor<std::string>(db, leaf_paths, final_dims, data_type);
        default:
            throw AlException(UNSUPPORTED_TYPE, "Unsupported data type for direct tensor read.");
    }
}


TensorView read_tensor(const std::string& ids_name, const std::string& path)
{
    auto segments = parse_path(path);
    return read_tensor_impl(ids_name, segments);
}

// Ancienne fonction conservée pour la compatibilité, maintenant un wrapper
TensorView read_tensor(
    const std::string& ids_name,
    const std::string& path_template,
    const std::vector<int>& aos_indices)
{
    auto segments = parse_path(path_template);

    // Substituer les sélections ALL/SLICE par des INDEX concrets
    size_t idx_cursor = 0;
    for (auto& seg : segments) {
        if (seg.selection != SelectionType::NONE) {
            if (idx_cursor < aos_indices.size()) {
                seg.selection  = SelectionType::INDEX;
                seg.start_index = aos_indices[idx_cursor++];
                seg.has_start  = true;
            }
        }
    }

    return read_tensor_impl(ids_name, segments);
}

} // namespace direct_access
} // namespace imas
