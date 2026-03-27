#include "path_parser.h"
#include <regex>
#include <stdexcept>

namespace imas {
namespace direct_access {

PathParser::PathParser(const std::string& path) : raw_path_(path) {
    parse();
}

const std::vector<PathSegment>& PathParser::segments() const {
    return segments_;
}

void PathParser::parse() {
    // Add '&' and '@' to the list of allowed characters. 
    std::regex segment_regex("([\\w&@-]*)(?:\\[(.*?)\\])?");
    
    // Regex pour les différentes parties de la sélection
    std::regex time_regex("time=([\\d\\.]+)?:?([\\d\\.]+)?");
    std::regex interp_regex("interp=(\\w+)");
    std::regex index_slice_regex("(\\d+)?:(\\d+)?");

    std::string path_to_parse = raw_path_;
    size_t start = 0;
    size_t end = path_to_parse.find('/');
    
    while (start < path_to_parse.length()) {
        std::string part = path_to_parse.substr(start, (end == std::string::npos) ? std::string::npos : (end - start));
        
        std::smatch match;
        if (std::regex_match(part, match, segment_regex)) {
            PathSegment segment;
            segment.node_name = match[1].str();

            if (match[2].matched) {
                std::string sel_content = match[2].str();
                
                // Diviser le contenu par des virgules (ex: "time=2.7,interp=linear")
                std::string token;
                std::stringstream ss(sel_content);
                while(std::getline(ss, token, ',')) {
                    std::smatch content_match;
                    if (token.rfind("time=", 0) == 0) {
                        segment.selection = SelectionType::TIME;
                        if (std::regex_match(token, content_match, time_regex)) {
                            if (content_match[1].matched) {
                                segment.start_time = std::stod(content_match[1].str());
                                segment.has_start_time = true;
                            }
                            if (content_match[2].matched) {
                                segment.end_time = std::stod(content_match[2].str());
                                segment.has_end_time = true;
                            }
                        }
                    } else if (std::regex_match(token, content_match, interp_regex)) {
                        std::string method = content_match[1].str();
                        if (method == "linear") {
                            segment.interp = InterpolationMethod::LINEAR;
                        } else if (method == "closest") {
                            segment.interp = InterpolationMethod::CLOSEST;
                        }
                    } else if (token == ":") {
                        segment.selection = SelectionType::ALL;
                    } else if (token.find(':') != std::string::npos) {
                        segment.selection = SelectionType::SLICE;
                        if (std::regex_match(token, content_match, index_slice_regex)) {
                           if (content_match[1].matched) { segment.start_index = std::stoul(content_match[1].str()); segment.has_start = true; }
                           if (content_match[2].matched) { segment.end_index = std::stoul(content_match[2].str()); segment.has_end = true; }
                        }
                    } else {
                        segment.selection = SelectionType::INDEX;
                        segment.index = std::stoul(token);
                    }
                }

                // Définir la méthode d'interpolation par défaut si nécessaire
                if (segment.selection == SelectionType::TIME && !segment.has_end_time && segment.interp == InterpolationMethod::NONE) {
                    segment.interp = InterpolationMethod::CLOSEST;
                }

            } else {
                segment.selection = SelectionType::NONE;
            }
            segments_.push_back(segment);
        } else {
            throw std::runtime_error("Invalid path segment format: " + part);
        }

        if (end == std::string::npos) break;
        start = end + 1;
        end = path_to_parse.find('/', start);
    }
}

} // namespace direct_access
} // namespace imas
