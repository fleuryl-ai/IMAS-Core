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
    // Regex étendue pour capturer les slices:
    // - Groupe 1: ([\w-]+) -> Nom du nœud
    // - Groupe 2: \[(...?)\] -> Contenu de la sélection (non-gourmand)
    //   - Groupe 3: (\d+:\d*|\d*:\d+|\d+|:) -> Le contenu effectif : "3:10", "5:", ":20", "3", ":"
    std::regex segment_regex("([\\w-]+)(?:\\[(.*?)\\])?");
    std::regex slice_regex("(\\d+)?:(\\d+)?"); // Pour analyser le contenu d'une slice

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
                
                if (sel_content == ":") {
                    segment.selection = SelectionType::ALL;
                } else if (sel_content.find(':') != std::string::npos) {
                    segment.selection = SelectionType::SLICE;
                    std::smatch slice_match;
                    if (std::regex_match(sel_content, slice_match, slice_regex)) {
                        if (slice_match[1].matched) {
                            segment.start_index = std::stoul(slice_match[1].str());
                            segment.has_start = true;
                        }
                        if (slice_match[2].matched) {
                            segment.end_index = std::stoul(slice_match[2].str());
                            segment.has_end = true;
                        }
                    } else {
                        throw std::runtime_error("Invalid slice format in path: " + part);
                    }
                } else {
                    try {
                        segment.selection = SelectionType::INDEX;
                        segment.index = std::stoul(sel_content);
                    } catch (const std::invalid_argument&) {
                        throw std::runtime_error("Invalid index format in path: " + part);
                    } catch (const std::out_of_range&) {
                        throw std::runtime_error("Index value '" + sel_content + "' is too large.");
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
