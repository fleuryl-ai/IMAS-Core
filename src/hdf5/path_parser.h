#ifndef PATH_PARSER_H
#define PATH_PARSER_H

#include <string>
#include <vector>
#include <stdexcept>

namespace imas {
namespace direct_access {

/**
 * @enum SelectionType
 * @brief Définit le type de sélection appliqué à un segment de chemin.
 */
enum class SelectionType {
    NONE,       // Aucune sélection, ex: "data"
    ALL,        // Sélection de tous les éléments, ex: "node[:]"
    INDEX,      // Sélection par un indice unique, ex: "node[3]"
    SLICE,      // Sélection par une tranche d'indices, ex: "node[2:10]"
    TIME_SLICE  // Sélection par une tranche temporelle, ex: "node[time=1.2:3.4]"
};

/**
 * @struct PathSegment
 * @brief Représente un segment unique d'un chemin d'accès.
 *
 * Un chemin comme "A[3]/B[time=1.0:2.0]/data" est décomposé en trois segments.
 */
struct PathSegment {
    std::string node_name;
    SelectionType selection = SelectionType::NONE;
    
    // Utilisé pour INDEX
    size_t index = 0;

    // Utilisé pour SLICE (par indice)
    size_t start_index = 0;
    size_t end_index = 0;
    bool has_start = false;
    bool has_end = false;

    // Utilisé pour TIME_SLICE
    double start_time = 0.0;
    double end_time = 0.0;
    bool has_start_time = false;
    bool has_end_time = false;
};

/**
 * @class PathParser
 * @brief Analyse une chaîne de caractères représentant un chemin d'accès.
 *
 * La classe décompose un chemin (ex: "A[3]/B[:]/data") en une séquence
 * de PathSegments, chacun contenant le nom du noeud et la sélection associée.
 */
class PathParser {
public:
    /**
     * @brief Construit un analyseur et traite le chemin fourni.
     * @param path Le chemin d'accès brut.
     * @throw std::runtime_error si le chemin est mal formé.
     */
    explicit PathParser(const std::string& path);

    /**
     * @brief Retourne les segments analysés du chemin.
     */
    const std::vector<PathSegment>& segments() const;

private:
    void parse();

    std::string raw_path_;
    std::vector<PathSegment> segments_;
};

// Fonction utilitaire pour utiliser le parser plus simplement
inline std::vector<PathSegment> parse_path(const std::string& path) {
    PathParser parser(path);
    return parser.segments();
}

} // namespace direct_access
} // namespace imas

#endif // PATH_PARSER_H
