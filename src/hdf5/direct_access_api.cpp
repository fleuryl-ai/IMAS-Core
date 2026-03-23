#include "direct_access_api.h"
#include "path_parser.h"
#include "direct_reader.h"
#include <stdexcept>
#include <iostream>

namespace imas {
namespace direct_access {

// ... (définitions de TensorView)

TensorView read_tensor(const std::string& ids_name, const std::string& path) {
    // 1. Analyser le chemin en segments
    PathParser parser(path);
    
    // 2. Créer un lecteur pour l'IDS
    DirectReader reader(ids_name);

    // 3. Déléguer la lecture directement avec les segments
    return reader.read(parser.segments());
}

// NOTE: La deuxième surcharge n'est plus directement utilisée par la première,
// mais on la garde pour la compatibilité de l'API si nécessaire.
// Son implémentation est maintenant redondante.
TensorView read_tensor(const std::string& ids_name, const std::string& path_template, const std::vector<int>& aos_indices) {
    // Cette fonction pourrait reconstruire un chemin et appeler la première,
    // ou être marquée comme obsolète. Pour l'instant, on la laisse non implémentée.
    throw std::runtime_error("This overload of read_tensor is deprecated. Use the version with a single path string.");
}

} // namespace direct_access
} // namespace imas
