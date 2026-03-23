#ifndef PATH_PARSER_H
#define PATH_PARSER_H

#include <string>
#include <vector>
#include <stdexcept>

namespace imas {
namespace direct_access {

enum class SelectionType {
    NONE,
    ALL,
    INDEX,
    SLICE,
    TIME // Un seul type pour toutes les sélections temporelles
};

enum class InterpolationMethod {
    NONE,    // Pour les plages de temps [time=1:2]
    CLOSEST, // Valeur par défaut pour un temps unique [time=2.7]
    LINEAR   // Pour [time=2.7,interp=linear]
};

struct PathSegment {
    std::string node_name;
    SelectionType selection = SelectionType::NONE;
    InterpolationMethod interp = InterpolationMethod::NONE;
    
    // Pour INDEX
    size_t index = 0;

    // Pour SLICE
    size_t start_index = 0;
    size_t end_index = 0;
    bool has_start = false;
    bool has_end = false;

    // Pour TIME
    double start_time = 0.0;
    double end_time = 0.0;
    bool has_start_time = false;
    bool has_end_time = false;
};

class PathParser {
public:
    explicit PathParser(const std::string& path);
    const std::vector<PathSegment>& segments() const;
private:
    void parse();
    std::string raw_path_;
    std::vector<PathSegment> segments_;
};

inline std::vector<PathSegment> parse_path(const std::string& path) {
    PathParser parser(path);
    return parser.segments();
}

} // namespace direct_access
} // namespace imas

#endif // PATH_PARSER_H
