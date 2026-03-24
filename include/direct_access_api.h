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

enum class DataType {
    FLOAT, DOUBLE, INT32, INT64, STRING,
    COMPLEX_FLOAT, COMPLEX_DOUBLE, LIST_OF_STRINGS, UNKNOWN
};

class TensorView {
public:
    TensorView() : data_type_(DataType::UNKNOWN) {}

    // CORRECTION: Le constructeur accepte maintenant un std::shared_ptr
    TensorView(std::shared_ptr<char[]> buffer, std::vector<size_t> dims, DataType type)
        : buffer_(std::move(buffer)), dimensions_(std::move(dims)), data_type_(type) {}

    const std::vector<size_t>& dims() const { return dimensions_; }
    DataType type() const { return data_type_; }
    const void* data() const { return buffer_.get(); }
    void* data() { return buffer_.get(); }

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
        if (dimensions_.empty()) return 0;
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

private:
    // CORRECTION: Utilisation d'un shared_ptr pour permettre le partage de la mémoire
    std::shared_ptr<char[]> buffer_;
    std::vector<size_t> dimensions_;
    DataType data_type_;
};

TensorView read_tensor(const std::string& ids_name, const std::string& path);

TensorView read_tensor(
    const std::string& ids_name,
    const std::string& path_template,
    const std::vector<int>& aos_indices
);

} // namespace direct_access
} // namespace imas

#endif // DIRECT_ACCESS_API_H
