add_executable(imas_h5ls imas_h5ls.cpp)
target_link_libraries(imas_h5ls PRIVATE al ${HDF5_LIBRARIES})

add_executable(imas_h5dump imas_h5dump.cpp)
target_link_libraries(imas_h5dump PRIVATE al ${HDF5_LIBRARIES})

# User-friendly VDS view publisher (view_publisher.{h,cpp} live in the parent
# engine dir; link al which contains the object).
add_executable(imas_publish_view imas_publish_view.cpp)
target_include_directories(imas_publish_view PRIVATE ..)
target_link_libraries(imas_publish_view PRIVATE al ${HDF5_LIBRARIES})
