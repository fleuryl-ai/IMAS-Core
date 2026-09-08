add_executable(test_metadata_extractor test_metadata_extractor.cpp)

target_include_directories(test_metadata_extractor PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/../../src/hdf5/metadata
)

target_link_libraries(test_metadata_extractor PRIVATE al)
