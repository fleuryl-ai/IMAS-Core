#ifndef DIRECT_ACCESS_API_H
#define DIRECT_ACCESS_API_H

#include <vector>
#include <string>
#include <complex>
#include <memory>
#include <map>
#include <stdexcept>


// Forward declaration pour éviter d'inclure al_const.h dans un header public si possible
namespace alconst {
    enum data_type : int;
}

namespace imas {
namespace direct_access {

enum class DataType {
    FLOAT, DOUBLE, INT32, INT64, STRING,
    COMPLEX_FLOAT, COMPLEX_DOUBLE, LIST_OF_STRINGS, UNKNOWN
};

struct Slice {
    size_t start;
    size_t end;
};

// Structure pour représenter une sélection sur une dimension (indice, plage, ou tout)
struct SliceSelection {
    enum class Type { Index, Range, All };
    Type type = Type::All;
    size_t index = 0;
    Slice range = {0, 0};

    // Fonctions d'aide pour une API lisible, ex: SliceSelection::at(1)
    static SliceSelection at(size_t i) { return {Type::Index, i, {0,0}}; }
    static SliceSelection from(size_t start) { return {Type::Range, 0, {start, (size_t)-1}}; }
    static SliceSelection to(size_t end) { return {Type::Range, 0, {0, end}}; }
    static SliceSelection between(size_t start, size_t end) { return {Type::Range, 0, {start, end}}; }
    static SliceSelection all() { return {Type::All, 0, {0,0}}; }
};

enum class NodeType {
    DATASET,
    AOS_STATIC,
    AOS_DYNAMIC
};

struct NodeInfo {
    std::string path;
    NodeType type;
    std::vector<size_t> dims;
    bool is_in_dynamic_aos = false; // Pour un dataset: est-il dans un AoS dynamique?
};

class TensorView {

private:
    static size_t get_element_size(DataType type) {
        switch (type) {
            case DataType::FLOAT:         return sizeof(float);
            case DataType::DOUBLE:        return sizeof(double);
            case DataType::INT32:         return sizeof(int32_t);
            case DataType::INT64:         return sizeof(int64_t);
            case DataType::COMPLEX_FLOAT: return sizeof(std::complex<float>);
            case DataType::COMPLEX_DOUBLE:return sizeof(std::complex<double>);
            case DataType::STRING:
            case DataType::LIST_OF_STRINGS: return sizeof(char);
            default:                      return 0;
        }
    }
    std::map<std::string, std::string> metadata_;
    
public:
    TensorView() : data_type_(DataType::UNKNOWN) {}

    TensorView(std::shared_ptr<char[]> buffer, std::vector<size_t> dims, DataType type, std::map<std::string, std::string> metadata = {})
    : buffer_(std::move(buffer)), dimensions_(std::move(dims)), data_type_(type), offset_in_bytes_(0) {
    
        // Calcule automatiquement les strides pour un bloc de mémoire contigu
        if (!dimensions_.empty()) {
            strides_in_bytes_.resize(dimensions_.size());
            size_t element_size = get_element_size(type);
            if (element_size == 0) {
                throw std::runtime_error("Impossible de calculer les strides pour un type de donnée inconnu.");
            }
            
            strides_in_bytes_.back() = element_size;
            for (int i = dimensions_.size() - 2; i >= 0; --i) {
                strides_in_bytes_[i] = strides_in_bytes_[i + 1] * dimensions_[i + 1];
            }
        }
    }

    const std::map<std::string, std::string>& metadata() const { return metadata_; }

    const std::vector<size_t>& dims() const { return dimensions_; }
    DataType type() const { return data_type_; }
    const void* data() const { return buffer_.get() + offset_in_bytes_; }
    void* data() { return buffer_.get() + offset_in_bytes_; }


    size_t size_in_bytes() const {
        size_t element_size = 0;
        switch (data_type_) {
            case DataType::FLOAT:         element_size = sizeof(float); break;
            case DataType::DOUBLE:        element_size = sizeof(double); break;
            case DataType::INT32:         element_size = sizeof(int32_t); break;
            case DataType::INT64:         element_size = sizeof(int64_t); break;
            case DataType::COMPLEX_FLOAT: element_size = sizeof(std::complex<float>); break;
            case DataType::COMPLEX_DOUBLE:element_size = sizeof(std::complex<double>); break;
            case DataType::STRING:
            case DataType::LIST_OF_STRINGS: element_size = sizeof(char); break;
            default:                      element_size = 0; break;
        }
        return total_elements() * element_size;
    }

    size_t total_elements() const {
        if (!buffer_) return 0; // Empty tensor
        if (dimensions_.empty()) return 1; // Scalar has 1 element
        size_t total = 1;
        for (size_t dim : dimensions_) total *= dim;
        return total;
    }

    template<typename T>
    const T* as() const {
        return reinterpret_cast<const T*>(buffer_.get());
    }

    template<typename T>
    T* as() {
        return reinterpret_cast<T*>(buffer_.get());
    }

    TensorView slice(const std::vector<SliceSelection>& selections) const {
        if (selections.size() > dimensions_.size()) {
            throw std::runtime_error("Trop de sélecteurs pour l'opération de slice.");
        }

        std::vector<size_t> new_dims;
        std::vector<size_t> new_strides;
        size_t new_offset = offset_in_bytes_;

        for (size_t i = 0; i < dimensions_.size(); ++i) {
            if (i < selections.size()) {
                const auto& sel = selections[i];
                if (sel.type == SliceSelection::Type::Index) {
                    if (sel.index >= dimensions_[i]) {
                        throw std::out_of_range("L'indice de slice est hors limites.");
                    }
                    // La dimension est supprimée, sa contribution est ajoutée à l'offset.
                    new_offset += sel.index * strides_in_bytes_[i];
                } else {
                    // La dimension est conservée.
                    size_t start = 0;
                    size_t end = dimensions_[i];
                    if (sel.type == SliceSelection::Type::Range) {
                        start = sel.range.start;
                        if (sel.range.end != (size_t)-1) {
                           end = sel.range.end;
                        }
                    }
                    if (start >= end || end > dimensions_[i]) {
                        throw std::out_of_range("La plage de slice est hors limites.");
                    }
                    new_offset += start * strides_in_bytes_[i];
                    new_dims.push_back(end - start);
                    new_strides.push_back(strides_in_bytes_[i]);
                }
            } else {
                // Si moins de sélecteurs que de dimensions, on garde le reste.
                new_dims.push_back(dimensions_[i]);
                new_strides.push_back(strides_in_bytes_[i]);
            }
        }
        
        // Appel du constructeur privé pour créer la nouvelle vue.
        return TensorView(buffer_, new_dims, data_type_, new_strides, new_offset, metadata_);
    }

private:
private:
    // Constructeur pour les vues dérivées (utilisé par slice())
    TensorView(std::shared_ptr<char[]> buffer, std::vector<size_t> dims, DataType type, std::vector<size_t> strides, size_t offset, const std::map<std::string, std::string>& metadata)
    : buffer_(std::move(buffer)), dimensions_(std::move(dims)), data_type_(type), strides_in_bytes_(std::move(strides)), offset_in_bytes_(offset) {}

    std::shared_ptr<char[]> buffer_;
    std::vector<size_t> dimensions_;
    DataType data_type_;
    std::vector<size_t> strides_in_bytes_;
    size_t offset_in_bytes_ = 0;
};

TensorView read_tensor(const std::string& ids_name, const std::string& path);

TensorView read_tensor(
    const std::string& ids_name,
    const std::string& path_template,
    const std::vector<int>& aos_indices
);

// Retourne à la fois les nœuds et une map des chemins d'AOS pour aider à l'affichage
std::pair<std::vector<NodeInfo>, std::map<std::string, NodeType>>
list_nodes(const std::string& ids_name, bool recursive, bool show_aos, bool show_metadata);

// True if the signal (data leaf) at `signal_path` varies over time — either because
// it lives under a dynamic Array of Structures, or because it owns its own time axis.
// False if the signal is static or the path is not found.
bool isDynamicSignal(const std::string& ids_name, const std::string& signal_path);


} // namespace direct_access
} // namespace imas

#endif // DIRECT_ACCESS_API_H
