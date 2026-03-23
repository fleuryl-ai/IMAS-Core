#include "direct_access_api.h"
#include "path_parser.h"
#include "direct_reader.h"
#include <stdexcept>
#include <iostream>

namespace imas {
namespace direct_access {

// Définitions manquantes de TensorView qui étaient dans l'ancien .cpp
size_t TensorView::size_in_bytes() const {
    if (total_elements() == 0) return 0;
    size_t element_size = 0;
    switch (data_type_) {
        case DataType::FLOAT: case DataType::INT32: element_size = 4; break;
        case DataType::DOUBLE: case DataType::INT64: element_size = 8; break;
        case DataType::COMPLEX_FLOAT: element_size = 8; break;
        case DataType::COMPLEX_DOUBLE: element_size = 16; break;
        case DataType::STRING: throw std::runtime_error("size_in_bytes() not implemented for STRING type yet.");
        case DataType::UNKNOWN: return 0;
    }
    return total_elements() * element_size;
}

size_t TensorView::total_elements() const {
    if (dimensions_.empty()) return 0;
    size_t total = 1;
    for (const auto& dim : dimensions_) total *= dim;
    return total;
}
// Fin des définitions de TensorView


TensorView read_tensor(const std::string& ids_name, const std::string& path) {
    std::cout << "Attempting to read tensor for IDS '" << ids_name << "' with path '" << path << "'" << std::endl;

    PathParser parser(path);
    const auto& segments = parser.segments();

    if (segments.empty()) {
        throw std::runtime_error("Cannot read data for an empty path.");
    }

    std::vector<int> aos_indices;
    std::string path_template;

    for (const auto& segment : segments) {
        if (!path_template.empty()) path_template += "/";
        path_template += segment.node_name;
        switch (segment.selection) {
            case SelectionType::NONE: break;
            case SelectionType::INDEX:
                path_template += "[:]";
                aos_indices.push_back(segment.index);
                break;
            case SelectionType::ALL:
                path_template += "[:]";
                aos_indices.push_back(-1);
                break;
            default:
                throw std::runtime_error("Unsupported selection type in path.");
        }
    }
    
    std::cout << "Parsed path template: " << path_template << std::endl;
    return read_tensor(ids_name, path_template, aos_indices);
}

TensorView read_tensor(const std::string& ids_name, const std::string& path_template, const std::vector<int>& aos_indices) {
    std::cout << "Reading tensor with path template: " << path_template << std::endl;
    DirectReader reader(ids_name);
    return reader.read(path_template, aos_indices);
}

} // namespace direct_access
} // namespace imas
