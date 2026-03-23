#ifndef DIRECT_ACCESS_API_H
#define DIRECT_ACCESS_API_H

#include <vector>
#include <string>
#include <complex>
#include <memory>
#include <stdexcept>

// Forward declaration pour éviter d'inclure al_const.h dans un header public si possible
namespace alconst {
    enum data_type : int;
}

namespace imas {
namespace direct_access {

/**
 * @enum DataType
 * @brief Énumère les types de données possibles pouvant être retournés par l'API.
 */
enum class DataType {
    FLOAT,
    DOUBLE,
    INT32,
    INT64,
    STRING,
    COMPLEX_FLOAT,
    COMPLEX_DOUBLE,
    LIST_OF_STRINGS, // Ajout pour gérer les tableaux de chaînes de caractères
    UNKNOWN
};

/**
 * @class TensorView
 * @brief Un conteneur non-template pour visualiser les données de tenseur.
 *
 * Cette classe fournit une vue sur un buffer de données. Elle contient les
 * métadonnées nécessaires pour interpréter le buffer, comme les dimensions
 * et le type de données. L'objet TensorView est propriétaire du buffer de données.
 */
class TensorView {
public:
    /**
     * @brief Constructeur par défaut. Crée une vue vide.
     */
    TensorView() : data_type_(DataType::UNKNOWN) {}

    /**
     * @brief Construit un TensorView en déplaçant les données.
     * @param buffer Le buffer de données brutes.
     * @param dims Les dimensions du tenseur.
     * @param type Le type des données dans le buffer.
     */
    TensorView(std::unique_ptr<char[]> buffer, std::vector<size_t> dims, DataType type)
        : buffer_(std::move(buffer)), dimensions_(std::move(dims)), data_type_(type) {}

    /**
     * @brief Retourne les dimensions du tenseur.
     */
    const std::vector<size_t>& dims() const { return dimensions_; }

    /**
     * @brief Retourne le type de données du tenseur.
     */
    DataType type() const { return data_type_; }

    /**
     * @brief Retourne un pointeur brut vers le début des données.
     */
    const void* data() const { return buffer_.get(); }
    void* data() { return buffer_.get(); }

    /**
    * @brief Retourne la taille totale du buffer en octets.
    */
    size_t size_in_bytes() const {
        size_t element_size = 0;
        switch (data_type_) {
            case DataType::FLOAT:         element_size = sizeof(float); break;
            case DataType::DOUBLE:        element_size = sizeof(double); break;
            case DataType::INT32:         element_size = sizeof(int32_t); break;
            case DataType::INT64:         element_size = sizeof(int64_t); break;
            case DataType::COMPLEX_FLOAT: element_size = sizeof(std::complex<float>); break;
            case DataType::COMPLEX_DOUBLE:element_size = sizeof(std::complex<double>); break;
            case DataType::STRING:        element_size = sizeof(char); break; // Buffer de char
            case DataType::LIST_OF_STRINGS: element_size = sizeof(char); break; // Buffer de char
            default:                      element_size = 0; break;
        }
        return total_elements() * element_size;
    }

    /**
    * @brief Retourne le nombre total d'éléments dans le tenseur.
    */
    size_t total_elements() const {
        if (dimensions_.empty()) {
            return 0;
        }
        size_t total = 1;
        for (size_t dim : dimensions_) {
            total *= dim;
        }
        return total;
    }

    /**
     * @brief Accesseur typé aux données.
     * @tparam T Le type vers lequel caster les données.
     * @return Un pointeur de type T vers les données.
     * @throw std::runtime_error si le cast est invalide.
     */
    template<typename T>
    const T* as() const {
        // TODO: Ajouter une vérification de type entre T et data_type_
        return reinterpret_cast<const T*>(buffer_.get());
    }

    template<typename T>
    T* as() {
        // TODO: Ajouter une vérification de type entre T et data_type_
        return reinterpret_cast<T*>(buffer_.get());
    }

private:
    std::unique_ptr<char[]> buffer_;
    std::vector<size_t> dimensions_;
    DataType data_type_;
};

/**
 * @brief Lit un tenseur ou une partie d'un tenseur depuis un IDS.
 *
 * Analyse le chemin pour extraire les indices et les sélections, puis lit
 * les données correspondantes depuis le backend.
 *
 * Syntaxe de chemin : "node[index]/node2[:]/leaf_node"
 *
 * @param ids_name Le nom de l'IDS (ex: "magnetics").
 * @param path Le chemin vers la donnée avec la syntaxe de sélection.
 * @return Un TensorView contenant les données lues.
 */
TensorView read_tensor(
    const std::string& ids_name,
    const std::string& path
);

/**
 * @brief Lit un tenseur ou une partie d'un tenseur depuis un IDS.
 *
 * Version où les indices des tableaux de structures (AoS) sont fournis
 * séparément d'un chemin template.
 *
 * @param ids_name Le nom de l'IDS.
 * @param path_template Le chemin vers la donnée avec des placeholders (ex: "node[:]/node2[:]/leaf").
 * @param aos_indices Les indices à appliquer aux placeholders dans l'ordre.
 * @return Un TensorView contenant les données lues.
 */
TensorView read_tensor(
    const std::string& ids_name,
    const std::string& path_template,
    const std::vector<int>& aos_indices
);

} // namespace direct_access
} // namespace imas

#endif // DIRECT_ACCESS_API_H
