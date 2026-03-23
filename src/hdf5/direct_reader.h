#ifndef DIRECT_READER_H
#define DIRECT_READER_H

#include "direct_access_api.h"
#include "path_parser.h" // Inclure pour PathSegment
#include <string>
#include <vector>
#include <hdf5.h>

namespace imas {
namespace direct_access {

/**
 * @class DirectReader
 * @brief Gère la lecture de données directement depuis le backend de stockage.
 */
class DirectReader {
public:
    /**
     * @brief Constructeur.
     * @param ids_name Le nom de l'IDS à lire.
     */
    explicit DirectReader(const std::string& ids_name);

    /**
     * @brief Destructeur.
     */
    ~DirectReader();

    /**
     * @brief Lit un tenseur basé sur une séquence de segments de chemin analysés.
     * @param segments Les segments du chemin analysés par PathParser.
     * @return Un TensorView avec les données lues.
     */
    TensorView read(const std::vector<PathSegment>& segments);

private:
    std::string ids_name_;
};

} // namespace direct_access
} // namespace imas

#endif // DIRECT_READER_H
