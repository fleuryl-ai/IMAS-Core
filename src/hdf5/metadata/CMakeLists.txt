target_sources(al PRIVATE
    metadata_extractor.cpp
    pugixml.cpp
)

# Add the current directory to includes so pugixml.hpp can be found
# by metadata_extractor.cpp and other consumers of the 'al' library.
target_include_directories(al PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
