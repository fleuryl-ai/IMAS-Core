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
    // Regex pour capturer le nom du nœud et la sélection (optionnelle)
    // - Groupe 1: ([\w-]+) -> Nom du nœud (lettres, chiffres, '_', '-')
    // - Groupe 2: (\[(\d*|:|\d*:\d*)\])? -> Sélection optionnelle
    //   - Groupe 3: (\d*|:|\d*:\d*) -> Contenu de la sélection (ex: "3", ":", "1:5")
    std::regex segment_regex("([\\w-]+)(\\[(\\d*|:)\\])?");

    std::string path_to_parse = raw_path_;
    
    // Divise le chemin par le délimiteur '/'
    size_t start = 0;
    size_t end = path_to_parse.find('/');
    while (end != std::string::npos || start < path_to_parse.length()) {
        std::string part = path_to_parse.substr(start, end - start);
        if(end == std::string::npos) {
             part = path_to_parse.substr(start);
        }

        std::smatch match;
        if (std::regex_match(part, match, segment_regex)) {
            PathSegment segment;
            segment.node_name = match[1].str();

            if (match[2].matched) { // Si une sélection [...] est présente
                std::string selection_content = match[3].str();
                if (selection_content == ":") {
                    segment.selection = SelectionType::ALL;
                } else if (!selection_content.empty()) {
                    segment.selection = SelectionType::INDEX;
                    try {
                        segment.index = std::stoul(selection_content);
                    } catch (const std::out_of_range&) {
                        throw std::runtime_error("Index value '" + selection_content + "' is too large.");
                    }
                } else {
                     throw std::runtime_error("Invalid selection format in path: " + part);
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
