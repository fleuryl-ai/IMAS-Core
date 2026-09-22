add_executable(test_metadata_extractor test_metadata_extractor.cpp)

target_include_directories(test_metadata_extractor PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/../../src/hdf5/metadata
)

target_link_libraries(test_metadata_extractor PRIVATE al)

# Regression: INT_* leaves (e.g. ids_properties/homogeneous_time) must be extracted.
add_executable(test_metadata_extractor_inttype test_metadata_extractor_inttype.cpp)
target_include_directories(test_metadata_extractor_inttype PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/../../src/hdf5/metadata
)
target_link_libraries(test_metadata_extractor_inttype PRIVATE al)
add_test(NAME test_metadata_extractor_inttype COMMAND test_metadata_extractor_inttype)
