valgrind --leak-check=full ./build/test_magnetics_flux_loop
valgrind --leak-check=full ./build/test_magnetics_flux_loop_data_put_slice
valgrind --leak-check=full ./build/test_magnetics_flux_loop_position_data
valgrind --leak-check=full ./build/test_equilibrium_time_slice
valgrind --leak-check=full ./build/test_equilibrium_time_slice_put_slice

