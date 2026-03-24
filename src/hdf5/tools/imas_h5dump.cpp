#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <map>
#include "direct_access_api.h"

using namespace imas::direct_access;

// Helper to print indentation
void print_indent(int level) {
    for (int i = 0; i < level; ++i) std::cout << "   ";
}

// Core function to print tensor data recursively for multi-dimensional arrays
template<typename T>
void print_data_recursive(const T* data, const std::vector<size_t>& dims, size_t dim_idx, size_t& offset, int indent_level) {
    if (dim_idx == dims.size() - 1) {
        // Last dimension: print values
        std::cout << "{";
        for (size_t i = 0; i < dims[dim_idx]; ++i) {
            std::cout << data[offset++];
            if (i < dims[dim_idx] - 1) std::cout << ", ";
        }
        std::cout << "}";
    } else {
        // Nested dimensions
        std::cout << "{" << std::endl;
        for (size_t i = 0; i < dims[dim_idx]; ++i) {
            print_indent(indent_level + 1);
            print_data_recursive(data, dims, dim_idx + 1, offset, indent_level + 1);
            if (i < dims[dim_idx] - 1) std::cout << "," << std::endl;
        }
        std::cout << std::endl;
        print_indent(indent_level);
        std::cout << "}";
    }
}

// Specialization for strings (they are already in a flat char buffer)
void print_strings(const TensorView& view) {
    const char* buffer = view.as<char>();
    // If it's a list of strings, it's 2D: {num_strings, max_len}
    // If it's a single string, it's 1D: {len}
    if (view.dims().size() >= 2) {
        size_t num_strings = view.dims()[0];
        size_t stride = view.dims()[1];
        std::cout << "{" << std::endl;
        for (size_t i = 0; i < num_strings; ++i) {
            print_indent(1);
            std::cout << "\"" << (buffer + i * stride) << "\"";
            if (i < num_strings - 1) std::cout << ",";
            std::cout << std::endl;
        }
        std::cout << "}";
    } else {
        // Single scalar string
        std::cout << "\"" << buffer << "\"";
    }
}

void dump_dataset(const std::string& ids_name, const std::string& path) {
    try {
        auto view = read_tensor(ids_name, path);
        if (view.total_elements() == 0) return;

        std::cout << "DATASET \"" << path << "\" {" << std::endl;
        
        // 1. Print Metadata
        if (!view.metadata().empty()) {
            print_indent(1);
            std::cout << "METADATA {" << std::endl;
            for (auto const& [key, val] : view.metadata()) {
                print_indent(2);
                std::cout << "\"" << key << "\": \"" << val << "\"" << std::endl;
            }
            print_indent(1);
            std::cout << "}" << std::endl;
        }

        // 2. Print Data
        print_indent(1);
        std::cout << "DATA ";
        size_t offset = 0;
        
        if (view.type() == DataType::DOUBLE) {
            print_data_recursive(view.as<double>(), view.dims(), 0, offset, 1);
        } else if (view.type() == DataType::INT32) {
            print_data_recursive(view.as<int>(), view.dims(), 0, offset, 1);
        } else if (view.type() == DataType::LIST_OF_STRINGS || view.type() == DataType::STRING) {
            print_strings(view);
        } else if (view.type() == DataType::COMPLEX_DOUBLE) {
            // Complex format: (real, imag)
            print_data_recursive(view.as<std::complex<double>>(), view.dims(), 0, offset, 1);
        }

        std::cout << std::endl << "}" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "  Error dumping " << path << ": " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: imas_h5dump <file> [path]" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    if (filename.size() > 3 && filename.substr(filename.size() - 3) == ".h5") {
        filename = filename.substr(0, filename.size() - 3);
    }

    if (argc > 2) {
        // Targeted dump
        dump_dataset(filename, argv[2]);
    } else {
        // Full dump
        auto result = list_nodes(filename, true, false, false);
        for (const auto& node : result.first) {
            if (node.type == NodeType::DATASET) {
                dump_dataset(filename, node.path);
                std::cout << std::endl;
            }
        }
    }

    return 0;
}