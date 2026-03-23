#ifndef DIRECT_READER_H
#define DIRECT_READER_H

#include "direct_access_api.h"
#include <string>
#include <vector>
#include <hdf5.h> // Inclure directement l'en-tête HDF5

namespace imas {
namespace direct_access {

/**
 * @class DirectReader
 * @brief Gère la lecture de données directement depuis le backend de stockage.
 */
class DirectReader {
public:
    /**
     * @brief Constructeur. Ouvre une connexion à un IDS.
     * @param ids_name Le nom de l'IDS à lire.
     */
    explicit DirectReader(const std::string& ids_name);

    /**
     * @brief Destructeur. Ferme les ressources.
     */
    ~DirectReader();

    /**
     * @brief Lit un tenseur basé sur un chemin et des indices.
     * @param path_template Le chemin vers les données, avec placeholders.
     * @param aos_indices Les indices à appliquer.
     * @return Un TensorView avec les données lues.
     */
    TensorView read(const std::string& path_template, const std::vector<int>& aos_indices);

private:
    // Méthodes internes pour interagir avec HDF5
    void open_ids();
    DataType get_data_type(const std::string& path);
    std::vector<size_t> get_dimensions(const std::string& path);


    std::string ids_name_;
    hid_t file_id_ = -1; // Handle HDF5
};

} // namespace direct_access
} // namespace imas

#endif // DIRECT_READER_H
