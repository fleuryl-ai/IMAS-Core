#include "direct_access_api.h"
#include "hdf5/path_parser.h"
#include "hdf5/direct_reader.h"
#include <stdexcept>
#include <iostream>

namespace imas {
namespace direct_access {

// ... (code de TensorView)

TensorView read_tensor(const std::string& ids_name, const std::string& path) {
    std::cout << "Attempting to read tensor for IDS '" << ids_name << "' with path '" << path << "'" << std::endl;

    // 1. Analyser le chemin
    PathParser parser(path);
    const auto& segments = parser.segments();

    if (segments.empty()) {
        throw std::runtime_error("Cannot read data for an empty path.");
    }

    // 2. Extraire les informations
    std::vector<int> aos_indices;
    std::string path_template;

    for (const auto& segment : segments) {
        if (!path_template.empty()) {
            path_template += "/";
        }
        path_template += segment.node_name;

        switch (segment.selection) {
            case SelectionType::NONE:
                // Rien à faire
                break;
            case SelectionType::INDEX:
                path_template += "[:]";
                aos_indices.push_back(segment.index);
                break;
            case SelectionType::ALL:
                path_template += "[:]";
                aos_indices.push_back(-1); // Convention pour "tous les indices"
                break;
            default:
                throw std::runtime_error("Unsupported selection type in path.");
        }
    }
    
    std::cout << "Parsed path template: " << path_template << std::endl;

    // 3. Appeler la surcharge avec les indices
    return read_tensor(ids_name, path_template, aos_indices);
}

TensorView read_tensor(const std::string& ids_name, const std::string& path_template, const std::vector<int>& aos_indices) {
    std::cout << "Reading tensor with path template: " << path_template << std::endl;
    
    // Créer un lecteur pour l'IDS demandé
    DirectReader reader(ids_name);

    // Déléguer la lecture au lecteur
    return reader.read(path_template, aos_indices);
}

} // namespace direct_access
} // namespace imas
