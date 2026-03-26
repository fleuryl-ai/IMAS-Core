#include <iostream>
#include "metadata_extractor.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <xml_file_path> <ids_name>" << std::endl;
        return 1;
    }

    std::string xml_file_path = argv[1];
    std::string ids_name = argv[2];

    MetadataExtractor extractor(xml_file_path);
    std::map<std::string, std::string> metadata = extractor.extract_metadata(ids_name);

    for (const auto& pair : metadata) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return 0;
}
