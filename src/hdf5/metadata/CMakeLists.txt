# Add metadata_extractor.cpp to the 'al' library
target_sources(al PRIVATE
    metadata_extractor.cpp
)

# Only add pugixml.cpp if it hasn't been added by the UDA backend
# to avoid multiple definition errors during linking.
if(NOT (AL_BACKEND_UDA OR AL_BACKEND_UDAFAT))
    target_sources(al PRIVATE
        pugixml.cpp
    )
endif()

# Add the current directory to includes so pugixml.hpp can be found
# by metadata_extractor.cpp and other consumers of the 'al' library.
target_include_directories(al PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# If UDA is enabled, we might need its directory for pugixml headers 
# if they were only there, but since you copied them to src/hdf5/metadata, 
# the include above is sufficient for metadata_extractor.cpp.
