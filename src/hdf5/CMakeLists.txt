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
    direct_access_api.cpp
    path_parser.cpp
    direct_reader.cpp
    view_publisher.h
    view_publisher.cpp
    imas_h5read/imas_h5read.cpp
)
# target_compile_definitions( al PRIVATE -DHDF5 -DDEBUG_HDF5_WRITER_V2 )
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

# Measurement tool for the read-parameter study (NOT registered as a test)
add_executable(bench_read ${EXE_FLAG} ${PROJECT_SOURCE_DIR}/tests/hdf5_backend/bench_read.cpp)
target_include_directories(bench_read PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench_read PRIVATE al z dl ${HDF5_LIBRARIES})
if(NOT WIN32 AND NOT APPLE)
    target_link_libraries(bench_read PRIVATE stdc++fs)
endif()

# --------------------------------------------------------------------------
# Engine tests (PanzerDB API directly)
# --------------------------------------------------------------------------
al_add_hdf5_test(test_engine_readdata_by_index)
al_add_hdf5_test(test_engine_readdata_by_index_deep)
al_add_hdf5_test(test_engine_readdata_by_index_2signals)
al_add_hdf5_test(test_engine_readdata_by_index_complex_aos)
al_add_hdf5_test(test_engine_readdata_by_index_verydeep)
al_add_hdf5_test(test_engine_readdata_by_index_ultradeep)
al_add_hdf5_test(test_engine_readdata_by_index_2d)
al_add_hdf5_test(test_engine_write_read_char)
al_add_hdf5_test(test_engine_write_read_string_complex)
al_add_hdf5_test(test_engine_write_read_listofstrings)
al_add_hdf5_test(test_engine_write_read_root_scalar)
al_add_hdf5_test(test_engine_empty_aos_getshape)
al_add_hdf5_test(test_engine_group_constructor)
al_add_hdf5_test(test_engine_write_read_dynamic_aos)
al_add_hdf5_test(test_engine_write_read_dynamic_put_slice)
al_add_hdf5_test(test_engine_write_read_nested_dynamic_aos)
# Bulk timebase + per-slice signal: fresh signal must land at 0..N-1, not shift
# onto the timebase's last written index (see hasFileData gating in panzerdb.cpp).
al_add_hdf5_test(test_engine_write_bulk_time_slice_signal)
al_add_hdf5_test(test_engine_append_complex)
# readIntDataByIndex single-leaf branch: report the time dimension for a bulk
# leaf (count = n_slices * shape), same rule as readDataByIndex (FLOAT64).
al_add_hdf5_test(test_engine_readint_by_index_bulk)

# --- SWMR: fixed-width data_raw_str boundary + writer/reader smoke --------
# `test_swmr_string_fixed_width` forks a separate reader process (a real shared-FS
# writer/reader pair) and still uses std::this_thread::sleep_for for pacing.
find_package(Threads REQUIRED)
al_add_hdf5_test(test_engine_write_read_string_512)
# Span-table (Option B): list elements > 511B chunked across 512B slots,
# recorded in the /list_spans dataset; backward-compatible with compact lists.
al_add_hdf5_test(test_engine_write_read_list_spanned)
al_add_hdf5_test(test_swmr_string_fixed_width)
# Crash-safe commit ordering: reader guardrail against a dangling index row
# (offset past the materialised data_raw_* extent — see PanzerDB::getLeaves).
al_add_hdf5_test(test_engine_commit_order_guardrail)
# v3 index format: packed 7-column /index + interned /path_table, version
# attribute, and upward compatibility with legacy 12-column files.
al_add_hdf5_test(test_engine_index_v3)
target_link_libraries(test_swmr_string_fixed_width PRIVATE Threads::Threads)

# Golden/oracle: pins the PanzerDB index surface (Leaf + list_nodes +
# getAOSShape + stripIndices). Default CTest mode is `verify`; `capture`
# regenerates the committed oracle. See tests/hdf5_backend/test_index_golden.cpp.
al_add_hdf5_test(test_index_golden)
al_add_hdf5_test(test_index_lifetime_guard)
target_compile_definitions(test_index_golden PRIVATE
    PZGOLD_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/hdf5_backend/fixtures/goldens")
target_include_directories(test_index_golden PRIVATE ${PROJECT_SOURCE_DIR}/include)

# --------------------------------------------------------------------------
# AL-level tests (via HDF5Backend / ALContext, dispatching to v2)
# --------------------------------------------------------------------------
# regression reproductions
al_add_hdf5_test(test_al_bfield_na_slice)
al_add_hdf5_test(test_al_cic_string_slice)
al_add_hdf5_test(test_al_dynamic_aos_nested_static_slice)
al_add_hdf5_test(test_al_static_aos_signal_slice)
al_add_hdf5_test(test_al_string_list_1d_slice)

# core IDS structures
al_add_hdf5_test(test_al_core_profiles_put_get_slice)
al_add_hdf5_test(test_al_core_sources_put_slice)

# magnetics flux_loop
al_add_hdf5_test(test_al_flux_loop_aos)
al_add_hdf5_test(test_al_flux_loop_deep_data_slice)
al_add_hdf5_test(test_al_flux_loop_deep_data_getslice)

# dynamic-AoS behaviour
al_add_hdf5_test(test_al_dynamic_aos_gap)
al_add_hdf5_test(test_al_dynamic_aos_nested_only_getshape)
# signal dynamism detection (under dynamic AoS OR own time axis, cf.
# PanzerDB::isDynamicSignal; tests/hdf5_backend/test_al_is_dynamic_signal.cpp)
al_add_hdf5_test(test_al_is_dynamic_signal)
al_add_hdf5_test(test_al_dynamic_aos_string_slice)
al_add_hdf5_test(test_al_dynamic_aos_string_inhom_slice)
al_add_hdf5_test(test_al_nested_dynamic_timebase)
al_add_hdf5_test(test_al_nested_static_aos)
# nested static AoS whose size varies across a dynamic time-slice (a "gap"):
# the x-point node disappears between two slices (mirrors hdf5bug.py).
al_add_hdf5_test(test_al_equilibrium_xpoint_gap)
al_add_hdf5_test(test_al_nested_aos_listofstrings)
al_add_hdf5_test(test_al_static_aos_dynamic_signal_slice)
al_add_hdf5_test(test_al_timeslice_nested_x_point)

# homogeneous / inhomogeneous timebases
al_add_hdf5_test(test_al_homog_timebase_2d_signal)
al_add_hdf5_test(test_al_inhom_timebase)

# gap handling
al_add_hdf5_test(test_al_gap_closest_prev)
al_add_hdf5_test(test_al_gap_homog_slice)
al_add_hdf5_test(test_al_gap_homog_timerange)
al_add_hdf5_test(test_al_gap_linear)
al_add_hdf5_test(test_al_gap_timerange)

# profiles_1d / root 1D signals
al_add_hdf5_test(test_al_profiles_1d_dynamic)
al_add_hdf5_test(test_al_profiles_1d_dynamic_0d_append)
al_add_hdf5_test(test_al_profiles_1d_dynamic_append_slice)
al_add_hdf5_test(test_al_profiles_1d_dynamic_append_slice_large)
al_add_hdf5_test(test_al_profiles_1d_dynamic_signal_1d)
al_add_hdf5_test(test_al_profiles_1d_dynamic_signal_2d)
al_add_hdf5_test(test_al_root_1d_dynamic_append)

# slice operators
al_add_hdf5_test(test_al_slice_op_code_output)
# Global read of a 1-slice scalar time series must stay 1D(1) (AL convention,
# to_improve.md point 3); static scalars must remain 0D.
al_add_hdf5_test(test_al_global_code_output_1slice)

# time-range operator
al_add_hdf5_test(test_al_timerange_basic)
al_add_hdf5_test(test_al_timerange_aos)
al_add_hdf5_test(test_al_timerange_aos_string_list)
al_add_hdf5_test(test_al_timerange_nested_dynamic_aos)
al_add_hdf5_test(test_al_timerange_nested_dynamic_aos_resampling)
al_add_hdf5_test(test_al_timerange_nested_dynamic_aos_resampling_vector)
al_add_hdf5_test(test_al_timerange_root_string)
al_add_hdf5_test(test_al_timerange_simple)

# resampling
al_add_hdf5_test(test_al_resampling_user_vector)

# metadata
al_add_hdf5_test(test_al_metadata_strategies)
# XML-driven (@IMAS_PREFIX) metadata extraction: INT_* leaves + multi-level path lookup
al_add_hdf5_test(test_al_metadata_extract_inttype)

# --------------------------------------------------------------------------
# Direct-access API
# --------------------------------------------------------------------------
al_add_hdf5_test(test_direct_api_validation)

# User-friendly VDS view (snapshot): standard h5ls/h5dump/h5py can inspect the
# IMAS instance tree without any IMAS-API dependency. See view_publisher.{h,cpp}.
al_add_hdf5_test(test_publish_view)

# --------------------------------------------------------------------------
# h5read (imas_h5read library)
# --------------------------------------------------------------------------
al_add_hdf5_test(test_h5read_overloads)

# --------------------------------------------------------------------------
# Path parser
# --------------------------------------------------------------------------
al_add_hdf5_test(test_path_parser_syntax)
al_add_hdf5_test(test_path_sanitization)
# Explicit read-side node-role rules (to_improve.md point 3, conservative pass).
al_add_hdf5_test(test_read_node_roles)

# Ajouter les outils HDF5
add_subdirectory(tools)
add_subdirectory(metadata )
