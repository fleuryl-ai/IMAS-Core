add_executable(imas_h5ls imas_h5ls.cpp)
target_link_libraries(imas_h5ls PRIVATE al)

add_executable(imas_h5dump imas_h5dump.cpp)
target_link_libraries(imas_h5dump PRIVATE al)
