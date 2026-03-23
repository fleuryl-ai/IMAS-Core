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
    // Regex pour un segment complet, ex: "node[...]"
    std::regex segment_regex("([\\w-]+)(?:\\[(.*?)\\])?");
    
    // Regex pour les différents types de sélections à l'intérieur de [...]
    std::regex time_slice_regex("time=([\\d\\.]+)?:([\\d\\.]+)?"); // Ex: "time=1.2:3.4", "time=5.0:", "time=:10.0"
    std::regex index_slice_regex("(\\d+)?:(\\d+)?");           // Ex: "3:10", "5:", ":20"

    std::string path_to_parse = raw_path_;
    
    size_t start = 0;
    size_t end = path_to_parse.find('/');
    
    while (start < path_to_parse.length()) {
        std::string part = path_to_parse.substr(start, (end == std::string::npos) ? std::string::npos : (end - start));
        
        std::smatch match;
        if (std::regex_match(part, match, segment_regex)) {
            PathSegment segment;
            segment.node_name = match[1].str();

            if (match[2].matched) { // Si une sélection [...] est présente
                std::string sel_content = match[2].str();
                std::smatch content_match;
                
                if (sel_content.rfind("time=", 0) == 0) { // Démarre par "time="
                    segment.selection = SelectionType::TIME_SLICE;
                    if (std::regex_match(sel_content, content_match, time_slice_regex)) {
                        if (content_match[1].matched) {
                            segment.start_time = std::stod(content_match[1].str());
                            segment.has_start_time = true;
                        }
                        if (content_match[2].matched) {
                            segment.end_time = std::stod(content_match[2].str());
                            segment.has_end_time = true;
                        }
                    } else {
                        throw std::runtime_error("Invalid time slice format: " + sel_content);
                    }
                } else if (sel_content == ":") {
                    segment.selection = SelectionType::ALL;
                } else if (sel_content.find(':') != std::string::npos) {
                    segment.selection = SelectionType::SLICE;
                    if (std::regex_match(sel_content, content_match, index_slice_regex)) {
                        if (content_match[1].matched) {
                            segment.start_index = std::stoul(content_match[1].str());
                            segment.has_start = true;
                        }
                        if (content_match[2].matched) {
                            segment.end_index = std::stoul(content_match[2].str());
                            segment.has_end = true;
                        }
                    } else {
                        throw std::runtime_error("Invalid index slice format: " + sel_content);
                    }
                } else { // C'est un simple indice
                    try {
                        segment.selection = SelectionType::INDEX;
                        segment.index = std::stoul(sel_content);
                    } catch (const std::exception&) {
                        throw std::runtime_error("Invalid index format: " + sel_content);
                    }
                }
            } else {
                segment.selection = SelectionType::NONE;
            }
            segments_.push_back(segment);
        } else {
            throw std::runtime_error("Invalid path segment format: " + part);
        }

        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
        end = path_to_parse.find('/', start);
    }

    if (segments_.empty() && !raw_path_.empty()) {
        throw std::runtime_error("Path parsing failed to produce any segments for non-empty path.");
    }
}

} // namespace direct_access
} // namespace imas
