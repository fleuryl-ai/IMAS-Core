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
    std::regex segment_regex("([\\w-]+)(?:\\[(.*?)\\])?");
    
    // Regex étendues
    std::regex time_interp_regex("time=([\\d\\.]+)");             // time=2.3
    std::regex time_slice_regex("time=([\\d\\.]+)?:([\\d\\.]+)?"); // time=1.2:3.4
    std::regex index_slice_regex("(\\d+)?:(\\d+)?");              // 3:10

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
                std::smatch content_match;
                
                if (sel_content.rfind("time=", 0) == 0) {
                    if (sel_content.find(':') != std::string::npos) {
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
                        }
                    } else {
                        segment.selection = SelectionType::TIME_INTERP;
                        if (std::regex_match(sel_content, content_match, time_interp_regex)) {
                           segment.start_time = std::stod(content_match[1].str());
                           segment.has_start_time = true;
                        }
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
                    }
                } else {
                    segment.selection = SelectionType::INDEX;
                    segment.index = std::stoul(sel_content);
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
