#include "direct_reader.h"
#include <stdexcept>

// Pour l'instant, on inclut les headers HDF5 ici.
// Si le problème de build persiste, il faudra trouver une autre stratégie.
// #include "hdf5.h" 

namespace imas {
namespace direct_access {

DirectReader::DirectReader(const std::string& ids_name) : ids_name_(ids_name) {
    // La logique d'ouverture de fichier ira ici
    // open_ids();
}

DirectReader::~DirectReader() {
    // La logique de fermeture de fichier ira ici
    // if (file_id_ >= 0) {
    //     H5Fclose(file_id_);
    // }
}

void DirectReader::open_ids() {
    // TODO: Déterminer le nom du fichier HDF5 à partir du nom de l'IDS
    // std::string filename = ids_name_ + ".h5"; // Simplification
    // file_id_ = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    // if (file_id_ < 0) {
    //     throw std::runtime_error("Failed to open IDS file for: " + ids_name_);
    // }
    throw std::runtime_error("HDF5 backend is currently disabled.");
}

TensorView DirectReader::read(const std::string& path_template, const std::vector<int>& aos_indices) {
    // 1. Transformer le path template et les indices en un chemin HDF5 complet
    // Ex: "A[:]/B[:]/data" + {3, 5} -> "A/3/B/5/data"
    
    // 2. Récupérer le type et les dimensions
    // DataType type = get_data_type(hdf5_path);
    // std::vector<size_t> dims = get_dimensions(hdf5_path);

    // 3. Créer une sélection (hyperslab)
    // ...

    // 4. Lire les données
    // ...

    // 5. Retourner le TensorView
    throw std::runtime_error("DirectReader::read not implemented.");
}

DataType DirectReader::get_data_type(const std::string& path) {
    // Lire l'attribut de type depuis le dataset HDF5
    return DataType::UNKNOWN;
}

std::vector<size_t> DirectReader::get_dimensions(const std::string& path) {
    // Lire l'attribut de dimension ou l'espace de données HDF5
    return {};
}


} // namespace direct_access
} // namespace imas
