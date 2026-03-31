#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <map>
#include <sstream>
#include <complex>
#include <numeric>
#include "direct_access_api.h"

// Déclarations
void print_usage();
void parse_arguments(int argc, char* argv[], std::string& filename, std::vector<std::string>& paths);
void dump_node(const std::string& ids_name, const imas::direct_access::NodeInfo& node);

template<typename T>
void print_data(const imas::direct_access::TensorView& tensor, size_t max_elements_to_print = 20);
void print_string_data(const imas::direct_access::TensorView& tensor, size_t max_elements_to_print = 20);

int main(int argc, char* argv[]) {
    std::string filename;
    std::vector<std::string> paths;

    parse_arguments(argc, argv, filename, paths);

    if (filename.empty()) {
        print_usage();
        return 1;
    }

    try {
        bool recursive = paths.empty(); // List all nodes if no specific path is given
        auto result = imas::direct_access::list_nodes(filename, recursive, false, false);
        
        if (paths.empty()) { // Dump all nodes
            for (const auto& node : result.first) {
                if (node.type == imas::direct_access::NodeType::DATASET) {
                    dump_node(filename, node);
                }
            }
        } else { // Dump only specified nodes
             for (const auto& path : paths) {
                 // Create a dummy NodeInfo to pass to dump_node
                 imas::direct_access::NodeInfo node_to_dump;
                 node_to_dump.path = path;
                 node_to_dump.type = imas::direct_access::NodeType::DATASET; // Assume it's a dataset
                 dump_node(filename, node_to_dump);
             }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

void parse_arguments(int argc, char* argv[], std::string& filename, std::vector<std::string>& paths) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (!arg.empty() && arg[0] != '-') {
            if (filename.empty()) {
                filename = arg;
            } else {
                paths.push_back(arg);
            }
        }
    }
}

void print_usage() {
    std::cerr << "Usage: imas_h5dump <file> [path1] [path2] ..." << std::endl;
    std::cerr << "  <file>: The IMAS HDF5 file (with .h5 extension)." << std::endl;
    std::cerr << "  [path...]: Optional. Specific paths to dump. If not provided, all datasets are dumped." << std::endl;
}

void dump_node(const std::string& ids_name, const imas::direct_access::NodeInfo& node) {
    try {
        std::cout << "HDF5 \"" << ids_name << "\" {" << std::endl;
        std::cout << "DATASET \"" << node.path << "\" {" << std::endl;

        auto tensor = imas::direct_access::read_tensor(ids_name, node.path);
        
        // Print shape
        std::cout << "   DATASPACE  ";
        if (tensor.dims().empty()) {
            std::cout << "SCALAR" << std::endl;
        } else {
            std::cout << "SIMPLE { ( ";
            for (size_t i = 0; i < tensor.dims().size(); ++i) {
                std::cout << tensor.dims()[i] << (i < tensor.dims().size() - 1 ? ", " : " )");
            }
            std::cout << " }" << std::endl;
        }

        // Print data
        std::cout << "   DATA {" << std::endl;
        std::cout << "      ";
        
        switch (tensor.type()) {
            case imas::direct_access::DataType::DOUBLE:
                print_data<double>(tensor);
                break;
            case imas::direct_access::DataType::INT32:
                print_data<int>(tensor);
                break;
            case imas::direct_access::DataType::COMPLEX_DOUBLE:
                print_data<std::complex<double>>(tensor);
                break;
            case imas::direct_access::DataType::STRING:
            case imas::direct_access::DataType::LIST_OF_STRINGS:
                 print_string_data(tensor);
                 break;
            default:
                std::cout << "      (Unsupported data type)" << std::endl;
        }

        std::cout << "   }" << std::endl; // End DATA
        std::cout << "}" << std::endl;     // End DATASET
        std::cout << "}" << std::endl;     // End FILE
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "   Error processing node " << node.path << ": " << e.what() << std::endl;
        std::cout << "}" << std::endl;
        std::cout << "}" << std::endl;
        std::cout << std::endl;
    }
}

template<typename T>
void print_data_recursive(const T* data, const std::vector<size_t>& dims, size_t dim_index, size_t& offset, std::string indent, size_t& printed_count, size_t max_count) {
    if (printed_count >= max_count) return;
    
    std::cout << indent;
    if (dim_index == dims.size() - 1) { // Innermost dimension
        std::cout << "{ ";
        for (size_t i = 0; i < dims[dim_index]; ++i) {
            if (printed_count >= max_count) { std::cout << "..."; break; }
            if constexpr (std::is_same_v<T, std::complex<double>>) {
                std::cout << "(" << data[offset].real() << "," << data[offset].imag() << ")";
            } else {
                std::cout << data[offset];
            }
            offset++;
            printed_count++;
            if (i < dims[dim_index] - 1) std::cout << ", ";
        }
        std::cout << " }";
    } else { // Outer dimensions
        std::cout << "{" << std::endl;
        for (size_t i = 0; i < dims[dim_index]; ++i) {
            if (printed_count >= max_count) { std::cout << indent << "   ..." << std::endl; break; }
            print_data_recursive(data, dims, dim_index + 1, offset, indent + "   ", printed_count, max_count);
            if (i < dims[dim_index] - 1) std::cout << ",";
            std::cout << std::endl;
        }
        std::cout << indent << "}";
    }
}

template<typename T>
void print_data(const imas::direct_access::TensorView& tensor, size_t max_elements_to_print) {
    if (!tensor.data()) {
        std::cout << "No data" << std::endl;
        return;
    }

    const T* data_ptr = reinterpret_cast<const T*>(tensor.data());
    const auto& dims = tensor.dims();
    size_t num_elements = tensor.total_elements();

    if (num_elements == 0) {
        return; // Print nothing if no elements
    }

    if (dims.empty()) { // Scalar
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            std::cout << "(" << data_ptr[0].real() << "," << data_ptr[0].imag() << ")";
        } else {
            std::cout << data_ptr[0];
        }
        std::cout << std::endl;
        return;
    }

    if (dims.size() > 1) {
        size_t offset = 0;
        size_t printed_count = 0;
        // The "      " is already printed by dump_node, so start with empty indent
        print_data_recursive<T>(data_ptr, dims, 0, offset, "", printed_count, max_elements_to_print);
        if (printed_count < num_elements) {
            std::cout << std::endl << "      ...";
        }
        std::cout << std::endl;
    } else { // 1D array
        size_t count = std::min(num_elements, max_elements_to_print);
        for (size_t i = 0; i < count; ++i) {
            if constexpr (std::is_same_v<T, std::complex<double>>) {
                std::cout << "(" << data_ptr[i].real() << "," << data_ptr[i].imag() << ")";
            } else {
                std::cout << data_ptr[i];
            }
            if (i < count - 1) std::cout << ", ";
        }
        if (num_elements > count) {
            std::cout << ", ...";
        }
        std::cout << std::endl;
    }
}

void print_string_data(const imas::direct_access::TensorView& tensor, size_t max_elements_to_print) {
     if (!tensor.data()) {
        std::cout << "No data" << std::endl;
        return;
    }

    if (tensor.dims().size() < 2) {
         std::cout << "(Invalid string format)" << std::endl;
         return;
    }
    
    size_t num_strings = tensor.dims()[0];
    size_t string_len = tensor.dims()[1];
    const char* data_ptr = reinterpret_cast<const char*>(tensor.data());
    
    size_t count = std::min(num_strings, max_elements_to_print);

    for (size_t i = 0; i < count; ++i) {
        std::string str(data_ptr + i * string_len, string_len);
        // Trim trailing nulls for cleaner output
        str.erase(str.find_last_not_of('\0') + 1);
        std::cout << "\"" << str << "\"" << (i < count - 1 ? ", " : "");
    }

    if (num_strings > max_elements_to_print) {
        std::cout << ", ...";
    }
    std::cout << std::endl;
}
