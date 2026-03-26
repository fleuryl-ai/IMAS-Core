#ifndef METADATA_EXTRACTOR_H
#define METADATA_EXTRACTOR_H

#include <string>
#include <map>
#include "pugixml.hpp"

class MetadataExtractor {
public:
    explicit MetadataExtractor(const std::string& xml_file_path);
    std::map<std::string, std::string> extract_metadata(const std::string& ids_name);

private:
    void traverse_nodes(pugi::xml_node node, std::map<std::string, std::string>& metadata);
    pugi::xml_document doc;
    bool doc_loaded = false;
};

#endif // METADATA_EXTRACTOR_H
