#include "direct_reader.h"
#include "panzerdb.h"
#include <stdexcept>
#include <numeric>
#include <iostream>

namespace imas {
namespace direct_access {

// --- Helper pour convertir les types ---
DataType convert_panzer_type_to_api(PanzerDB::DataType pz_type) {
    switch (pz_type) {
        case PanzerDB::DataType::FLOAT64: return DataType::DOUBLE;
        case PanzerDB::DataType::INT32: return DataType::INT32;
        case PanzerDB::DataType::COMPLEX128: return DataType::COMPLEX_DOUBLE;
        case PanzerDB::DataType::STRING: return DataType::STRING;
        default: return DataType::UNKNOWN;
    }
}


// --- Constructeur/Destructeur ---
DirectReader::DirectReader(const std::string& ids_name) : ids_name_(ids_name) {}
DirectReader::~DirectReader() {}


// --- Implémentation de la lecture ---
TensorView DirectReader::read(const std::string& path_with_indices, const std::vector<int>& aos_indices) {
    
    // NOTE: Le path_template est maintenant un chemin avec indices, on le reconstruit
    // depuis l'API publique. On pourrait optimiser ça plus tard.
    
    // On ouvre le fichier en mode lecture. Le destructeur de PanzerDB s'occupera de fermer.
    PanzerDB panzer_db(ids_name_ + ".h5", PanzerDB::OpenMode::READ);
    
    // On charge l'index complet du fichier.
    const auto& leaves = panzer_db.getLeaves();
    
    // On recherche la "leaf" (la donnée terminale) qui correspond à notre chemin.
    // On utilise stripIndices pour obtenir un chemin "schema" (sans les indices numériques)
    // afin de trouver la leaf qui décrit le signal, même pour un AoS dynamique.
    std::string schema_path = PanzerDB::stripIndices(path_with_indices);
    const PanzerDB::Leaf* target_leaf = nullptr;
    for (const auto& leaf : leaves) {
        if (PanzerDB::stripIndices(std::string(leaf.path)) == schema_path) {
            target_leaf = &leaf;
            break;
        }
    }

    if (!target_leaf) {
        throw std::runtime_error("No data schema found for path: " + schema_path);
    }
    
    // --- Logique de lecture ---
    // On suppose pour l'instant un seul type de lecture : par indice de temps.
    // Plus tard, on pourra différencier AoS statique et dynamique.
    
    // Pour l'instant, on lit juste une slice de temps (time_index = 0) comme démo.
    int64_t time_index_to_read = 0;
    
    uint64_t ndim_out = 0;
    uint64_t shape_out[6] = {0};
    
    PanzerDB::DataType pz_type = static_cast<PanzerDB::DataType>(target_leaf->flags & 0xFF);
    DataType api_type = convert_panzer_type_to_api(pz_type);

    std::unique_ptr<char[]> buffer;
    
    int status = -1;

    // On utilise la C-API de PanzerDB qui est bien testée.
    switch(pz_type) {
        case PanzerDB::DataType::FLOAT64: {
            double* data_out = nullptr;
            status = panzer_db.pz_readData_by_index(path_with_indices.c_str(), time_index_to_read, &ndim_out, shape_out, &data_out);
            buffer.reset(reinterpret_cast<char*>(data_out));
            break;
        }
        case PanzerDB::DataType::COMPLEX128: {
            std::complex<double>* data_out = nullptr;
            status = panzer_db.pz_readComplexData_by_index(path_with_indices.c_str(), time_index_to_read, &ndim_out, shape_out, &data_out);
            buffer.reset(reinterpret_cast<char*>(data_out));
            break;
        }
        // TODO: Ajouter INT32 et STRING
        default:
            throw std::runtime_error("Reading this data type is not yet implemented in DirectReader.");
    }

    if (status != 0) {
        throw std::runtime_error("Failed to read data for path: " + path_with_indices);
    }

    std::vector<size_t> dims(shape_out, shape_out + ndim_out);
    return TensorView(std::move(buffer), dims, api_type);
}


} // namespace direct_access
} // namespace imas
