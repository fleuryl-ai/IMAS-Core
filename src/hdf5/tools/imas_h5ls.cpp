#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <map>
#include <sstream>
#include "direct_access_api.h"

// Déclarations
void print_usage();
void parse_arguments(int argc, char* argv[], std::string& filename, bool& recursive, bool& show_aos, bool& show_metadata);
void display_nodes(const std::vector<imas::direct_access::NodeInfo>& nodes, const std::map<std::string, imas::direct_access::NodeType>& aos_paths);

int main(int argc, char* argv[]) {
    std::string filename;
    bool recursive = false;
    bool show_aos = false;
    bool show_metadata = false;

    parse_arguments(argc, argv, filename, recursive, show_aos, show_metadata);

    if (filename.empty()) {
        print_usage();
        return 1;
    }

    try {
        auto result = imas::direct_access::list_nodes(filename, recursive, show_aos, show_metadata);
        display_nodes(result.first, result.second);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

void parse_arguments(int argc, char* argv[], std::string& filename, bool& recursive, bool& show_aos, bool& show_metadata) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r") recursive = true;
        else if (arg == "--show-aos") show_aos = true;
        else if (arg == "--show-metadata") show_metadata = true;
        else if (!arg.empty() && arg[0] != '-') {
            filename = arg;
        }
    }
}

void print_usage() {
    std::cerr << "Usage: imas_h5ls [options] <file>" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -r              Recursively list contents." << std::endl;
    std::cerr << "  --show-aos      Display logical Array of Structures (AOS)." << std::endl;
    std::cerr << "  --show-metadata Display metadata datasets (e.g., @units)." << std::endl;
}

void display_nodes(const std::vector<imas::direct_access::NodeInfo>& nodes, const std::map<std::string, imas::direct_access::NodeType>& aos_paths) {
    size_t max_path_len = 0;
    for (const auto& node : nodes) {
        size_t display_len = node.path.length();
        std::string current_prefix;
        std::stringstream ss(node.path);
        std::string segment;
        while(std::getline(ss, segment, '/')) {
            current_prefix += (current_prefix.empty() ? "" : "/") + segment;
            if (aos_paths.count(current_prefix)) {
                display_len += 2; // Pour '[]'
            }
        }
        if (display_len > max_path_len) {
            max_path_len = display_len;
        }
    }
    max_path_len += 4;

    for (const auto& node : nodes) {
        std::string display_path;
        std::string current_prefix;
        std::stringstream ss(node.path);
        std::string segment;
        bool first_segment = true;
        while(std::getline(ss, segment, '/')) {
            if (!first_segment) display_path += "/";
            current_prefix += (first_segment ? "" : "/") + segment;
            display_path += segment;
            if (aos_paths.count(current_prefix)) {
                display_path += "[]";
            }
            first_segment = false;
        }
        
        std::cout << std::left << std::setw(max_path_len) << display_path;

        switch (node.type) {
            case imas::direct_access::NodeType::AOS_STATIC:
                std::cout << "AOS(static)" << std::endl;
                break;
            case imas::direct_access::NodeType::AOS_DYNAMIC:
                std::cout << "AOS(dynamic)" << std::endl;
                break;
            case imas::direct_access::NodeType::DATASET:
                std::cout << "Dataset {";
                if (node.dims.empty()) {
                     std::cout << "SCALAR";
                } else {
                    for (size_t i = 0; i < node.dims.size(); ++i) {
                        std::cout << node.dims[i];
                        // Add /Inf only for the time-dimension of the dataset
                        if (i == 0 && node.is_in_dynamic_aos) std::cout << "/Inf";
                        if (i < node.dims.size() - 1) std::cout << ", ";
                    }
                }
                std::cout << "}" << std::endl;
                break;
        }
    }
}
