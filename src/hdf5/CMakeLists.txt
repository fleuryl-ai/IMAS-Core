# CMake configuration for the HDF5 backend
if(WIN32)
	# Try modern CONFIG mode first (vcpkg), fallback to legacy FindHDF5 module
	find_package( hdf5 CONFIG QUIET )
	if( NOT hdf5_FOUND )
	  find_package( HDF5 COMPONENTS C HL REQUIRED )
	endif()
else()
	find_package( HDF5 COMPONENTS C HL REQUIRED )
endif()

target_sources( al PRIVATE
    hdf5_backend.cpp
    hdf5_writer.cpp
    hdf5_reader.cpp
    hdf5_utils.cpp
    hdf5_hs_selection_reader.cpp
    hdf5_hs_selection_writer.cpp
    hdf5_dataset_handler.cpp
    hdf5_events_handler.cpp
    hdf5_backend_factory.cpp
	panzerdb.h
    panzerdb.cpp
    hdf5_writer_v2.h
    hdf5_writer_v2.cpp
    hdf5_reader_v2.h
    hdf5_reader_v2.cpp
    global_read_strategy.h
    global_read_strategy.cpp
    slice_read_strategy.h
    slice_read_strategy.cpp
    timerange_read_strategy.h
    timerange_read_strategy.cpp
)
# target_compile_definitions( al PRIVATE -DHDF5 -DDEBUG_HDF5_WRITER_V2 )
# target_compile_definitions( al PRIVATE -DHDF5 -DDEBUG_HDF5_READER_V2 )
target_compile_definitions( al PRIVATE -DHDF5 )
if(WIN32)
	# Use modern CMake targets if available (vcpkg), otherwise legacy variables
	if( TARGET hdf5::hdf5-shared )
	  target_link_libraries( al PRIVATE hdf5::hdf5-shared )
	  if( TARGET hdf5::hdf5_hl-shared )
	    target_link_libraries( al PRIVATE hdf5::hdf5_hl-shared )
	  endif()
	elseif( TARGET hdf5::hdf5-static )
	  target_link_libraries( al PRIVATE hdf5::hdf5-static )
	  if( TARGET hdf5::hdf5_hl-static )
	    target_link_libraries( al PRIVATE hdf5::hdf5_hl-static )
	  endif()
	else()
	  # Legacy FindHDF5 module
	  target_include_directories( al PRIVATE ${HDF5_C_INCLUDE_DIRS} )
	  target_link_libraries( al PRIVATE ${HDF5_C_LIBRARIES} )
	  target_compile_definitions( al PRIVATE ${HDF5_C_DEFINITIONS} )
	endif()
	
	target_include_directories( al PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} )
else()
  target_include_directories( al PRIVATE ${HDF5_C_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
  target_link_libraries( al PRIVATE ${HDF5_C_LIBRARIES} )
  target_compile_definitions( al PRIVATE ${HDF5_C_DEFINITIONS} )
endif()

macro(al_add_hdf5_test test_name)
    add_executable(${test_name} ${EXE_FLAG} ${PROJECT_SOURCE_DIR}/tests/hdf5_backend/${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${test_name} PRIVATE al z dl ${HDF5_LIBRARIES} )
    if(NOT WIN32 AND NOT APPLE)
        target_link_libraries(${test_name} PRIVATE stdc++fs)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

    al_add_hdf5_test(test_panzer)
    al_add_hdf5_test(test_panzer_read_by_index)
    al_add_hdf5_test(test_panzer_read_by_index_deep)
    al_add_hdf5_test(test_panzer_read_by_index_deep_2_signals)
    al_add_hdf5_test(test_panzer_read_by_index_complex_aos)
    al_add_hdf5_test(test_panzer_read_by_index_very_deep)
    al_add_hdf5_test(test_panzer_read_by_index_ultra_deep)
    al_add_hdf5_test(test_panzer_empty_aos)
    al_add_hdf5_test(test_panzer_dynamic_aos)
    al_add_hdf5_test(test_panzer_nested_dynamic_aos)
    al_add_hdf5_test(test_panzer_read_2d_signal)
    al_add_hdf5_test(test_panzer_root_scalar)
    al_add_hdf5_test(test_magnetics_flux_loop)
    al_add_hdf5_test(test_magnetics_repro)
    al_add_hdf5_test(test_panzer_group_constructor)
    al_add_hdf5_test(test_panzer_dynamic_data_put_and_slice)
    al_add_hdf5_test(test_panzer_complex_append)
    al_add_hdf5_test(test_panzer_char_data)
    al_add_hdf5_test(test_panzer_string_complex)
    al_add_hdf5_test(test_panzer_list_of_strings)
    al_add_hdf5_test(test_nested_dynamic_timebase)
    al_add_hdf5_test(test_inhom_timebase)
    al_add_hdf5_test(test_homog_2d)
    al_add_hdf5_test(test_dynamic_aos_string)
    al_add_hdf5_test(test_dynamic_aos_string_inhom)
    al_add_hdf5_test(test_dynamic_aos_gap)
    al_add_hdf5_test(test_timerange_simple)
    al_add_hdf5_test(test_timerange_aos)
    al_add_hdf5_test(test_timerange_nested_dynamic_aos)
    al_add_hdf5_test(test_timerange_nested_dynamic_aos_resampling)
    al_add_hdf5_test(test_timerange_nested_dynamic_aos_resampling_vector)
    al_add_hdf5_test(test_magnetics_resampling_vector)
    al_add_hdf5_test(test_profiles_1d_dynamic)
    al_add_hdf5_test(test_profiles_1d_dynamic_signal_1d)
    al_add_hdf5_test(test_profiles_1d_dynamic_signal_2d)
    al_add_hdf5_test(test_nested_static_aos)
    al_add_hdf5_test(test_profiles_1d_dynamic_0d_append)
    al_add_hdf5_test(test_time_slice_nested_x_point)
    al_add_hdf5_test(test_profiles_1d_dynamic_append_slice)
    al_add_hdf5_test(test_profiles_1d_dynamic_append_slice_large)
    al_add_hdf5_test(test_core_profiles_put_get_slice)
    al_add_hdf5_test(test_timerange_aos_string_list)
    al_add_hdf5_test(test_timerange_root_string)
    al_add_hdf5_test(test_root_1d_dynamic_append)
    al_add_hdf5_test(test_magnetics_repro_slice)
    al_add_hdf5_test(test_nested_aos_char_data)
    al_add_hdf5_test(test_bug_string_list_1d)
    al_add_hdf5_test(test_bug_static_aos_slice)
    al_add_hdf5_test(test_bug_dynamic_aos_nested_static)
    al_add_hdf5_test(test_bug_cic_string_slice)
    al_add_hdf5_test(test_bug_b_field_na_slice)
    al_add_hdf5_test(test_static_aos_dynamic_signal_slice)
    al_add_hdf5_test(test_timerange)
    al_add_hdf5_test(test_python_bug)
    al_add_hdf5_test(test_slice_op2)