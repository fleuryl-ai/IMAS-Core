#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <libgen.h> // Pour dirname() et basename()
#include <unistd.h> // Pour chdir() et getcwd()
#include <limits.h> // Pour PATH_MAX
#include <string.h> // Pour strdup()
#include "direct_access_api.h"

// Helper pour l'indentation
void print_indent(int level) {
    for (int i = 0; i < level; ++i) std::cout << "  ";
}

// Fonction pour afficher récursivement les données numériques
template<typename T>
void print_data_recursive(const T* data, const std::vector<size_t>& dims, size_t dim_idx, size_t& offset, int indent_level) {
    if (dims.empty()) { // Cas scalaire
        std::cout << data[0];
        return;
    }
    if (dim_idx >= dims.size()) return;

    if (dim_idx == dims.size() - 1) {
        // Dernière dimension: on affiche les valeurs
        std::cout << "{ ";
        for (size_t i = 0; i < dims[dim_idx]; ++i) {
            std::cout << data[offset++];
            if (i < dims[dim_idx] - 1) std::cout << ", ";
        }
        std::cout << " }";
    } else {
        // Dimensions imbriquées
        std::cout << "{" << std::endl;
        for (size_t i = 0; i < dims[dim_idx]; ++i) {
            print_indent(indent_level + 1);
            print_data_recursive(data, dims, dim_idx + 1, offset, indent_level + 1);
            if (i < dims[dim_idx] - 1) std::cout << ",";
            std::cout << std::endl;
        }
        print_indent(indent_level);
        std::cout << "}";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <hdf5_file> [path_to_dump]" << std::endl;
        return 1;
    }

    std::string full_path = argv[1];
    std::string path_to_dump = (argc > 2) ? argv[2] : "/";

    char original_cwd[PATH_MAX];
    if (getcwd(original_cwd, sizeof(original_cwd)) == NULL) {
        std::cerr << "Error: Cannot get current working directory." << std::endl;
        return 1;
    }

    try {
        char* path_copy1 = strdup(full_path.c_str());
        char* path_copy2 = strdup(full_path.c_str());
        char* dir = dirname(path_copy1);
        char* base = basename(path_copy2);

        if (chdir(dir) != 0) {
            std::cerr << "Error: Cannot change directory to " << dir << std::endl;
            free(path_copy1);
            free(path_copy2);
            return 1;
        }

        std::string filename_only = base;
        free(path_copy1);
        free(path_copy2);
        
        std::string ids_name = filename_only;
        std::string extension = ".h5";
        if (ids_name.size() > extension.size() && 
            ids_name.substr(ids_name.size() - extension.size()) == extension) 
        {
            ids_name = ids_name.substr(0, ids_name.size() - extension.size());
        }

        auto result = imas::direct_access::list_nodes(ids_name, true, true, false);
        auto nodes = result.first;

        for (const auto& node_info : nodes) {
            if (path_to_dump != "/" && node_info.path.rfind(path_to_dump, 0) != 0) {
                continue;
            }

            if (node_info.type == imas::direct_access::NodeType::DATASET) {
                imas::direct_access::TensorView tensor = imas::direct_access::read_tensor(ids_name, node_info.path);

                std::cout << "DATASET \"" << node_info.path << "\"";
                if (!tensor.dims().empty()) {
                    std::cout << " {";
                    for (size_t i = 0; i < tensor.dims().size(); ++i) {
                        std::cout << tensor.dims()[i] << (i < tensor.dims().size() - 1 ? ", " : "");
                    }
                    std::cout << "}";
                } else {
                     std::cout << " {SCALAR}";
                }
                std::cout << std::endl;
                print_indent(1);
                std::cout << "Data: ";

                if (tensor.type() == imas::direct_access::DataType::STRING || tensor.type() == imas::direct_access::DataType::LIST_OF_STRINGS) {
                     std::cout << std::endl;
                    if (!tensor.dims().empty() && tensor.dims().size() == 2) {
                        const char* data_ptr = tensor.as<char>();
                        size_t num_strings = tensor.dims()[0];
                        size_t string_len = tensor.dims()[1];
                        for (size_t i = 0; i < num_strings; ++i) {
                            print_indent(2);
                            std::cout << "(" << i << "): \"" << (data_ptr + i * string_len) << "\"" << std::endl;
                        }
                    }
                } else {
                    size_t offset = 0;
                    if (tensor.type() == imas::direct_access::DataType::DOUBLE) {
                        print_data_recursive(tensor.as<double>(), tensor.dims(), 0, offset, 2);
                    } else if (tensor.type() == imas::direct_access::DataType::INT32) {
                        print_data_recursive(tensor.as<int32_t>(), tensor.dims(), 0, offset, 2);
                    } else {
                        std::cout << "(unsupported type for dumping)";
                    }
                    std::cout << std::endl;
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (chdir(original_cwd) != 0) { /* handle error */ }
        return 1;
    }

    if (chdir(original_cwd) != 0) { /* handle error */ }
    return 0;
}
