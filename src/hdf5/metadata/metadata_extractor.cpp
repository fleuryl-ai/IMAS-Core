#include "metadata_extractor.h"
#include <iostream>

MetadataExtractor::MetadataExtractor(const std::string& xml_file_path) {
    pugi::xml_parse_result result = doc.load_file(xml_file_path.c_str());
    if (result) {
        doc_loaded = true;
    } else {
        std::cerr << "XML [" << xml_file_path << "] parsed with errors, attr value: [" << doc.child("node").attribute("attr").value() << "]\n";
        std::cerr << "Error description: " << result.description() << "\n";
        std::cerr << "Error offset: " << result.offset << " (error at [..." << (xml_file_path.c_str() + result.offset) << "]\n\n";
    }
}

std::map<std::string, std::string> MetadataExtractor::extract_metadata(const std::string& ids_name) {
    std::map<std::string, std::string> metadata;
    if (!doc_loaded) {
        return metadata;
    }

    pugi::xml_node ids_node = doc.child("IDSs").find_child_by_attribute("IDS", "name", ids_name.c_str());
    if (ids_node) {
        traverse_nodes(ids_node, metadata);
    }

    return metadata;
}

void MetadataExtractor::traverse_nodes(pugi::xml_node node, std::map<std::string, std::string>& metadata) {
    for (pugi::xml_node child = node.first_child(); child; child = child.next_sibling()) {
        std::string name = child.name();
        if (name == "field") {
            std::string data_type = child.attribute("data_type").value();
            if (data_type.rfind("FLT_", 0) == 0 || data_type.rfind("STR_", 0) == 0 || data_type.rfind("CPX_", 0) == 0) {
                std::string path = child.attribute("path").value();
                for (pugi::xml_attribute attr = child.first_attribute(); attr; attr = attr.next_attribute()) {
                    std::string attr_name = attr.name();
                    if (attr_name == "path_doc" || attr_name == "documentation" || attr_name == "data_type" ||
                        attr_name == "type" || attr_name == "units" || attr_name == "timebasepath" ||
                        attr_name.rfind("coordinate", 0) == 0) {
                        metadata[path + "@" + attr_name] = attr.value();
                    }
                }
            }
        }
        traverse_nodes(child, metadata);
    }
}
