program test_imas_h5read_fortran
    use iso_c_binding
    use imas_h5read_mod
    implicit none

    character(len=64) :: filename = "test_imas_h5read"
    type(c_ptr)       :: data_ptr
    integer(kind=8), pointer :: dims(:)
    integer           :: dtype
    integer           :: i, j, k
    
    ! Pointers to access the raw data as Fortran arrays
    integer(c_int), pointer    :: array_3d(:,:,:)
    double precision, pointer  :: vector_1d(:)

    print *, "--- Fortran: Testing imas_h5read ---"

    ! 1. Read a 3D array slice: [1:2, 1:2, 1:2] (1-based)
    ! Original C++ data: i*100 + j*10 + k (0-based)
    print *, "Reading 3D slice [1:2, 1:2, 1:2]..."
    call imas_h5read(filename, "/test/array_3d", data_ptr, dims, dtype, &
                     start=[1, 1, 1], count=[2, 2, 2])

    if (c_associated(data_ptr)) then
        print *, "  Rank:", size(dims)
        print *, "  Dimensions:", dims
        
        ! Map C pointer to Fortran 3D array
        ! Note: Fortran is column-major, HDF5 is row-major. 
        ! The dims returned are [2, 2, 2].
        call c_f_pointer(data_ptr, array_3d, [dims(3), dims(2), dims(1)])
        
        ! Check values (indices are 1-based in Fortran)
        ! array_3d(k, j, i) maps to C array[i][j][k]
        print *, "  Value at (1,1,1):", array_3d(1,1,1) ! Expected 0
        print *, "  Value at (1,1,2):", array_3d(1,1,2) ! Expected 100
        
        ! Free memory
        call imas_h5read_free_c(data_ptr, c_loc(dims(1)))
    else
        print *, "  Error: Could not read 3D slice."
    end if

    ! 2. Read full 1D vector with INF
    print *, "Reading full 1D vector using IMAS_INF..."
    call imas_h5read(filename, "/test/vector_double", data_ptr, dims, dtype, &
                     start=[1], count=[IMAS_INF])

    if (c_associated(data_ptr)) then
        print *, "  Dimensions:", dims
        call c_f_pointer(data_ptr, vector_1d, [dims(1)])
        print *, "  First element:", vector_1d(1)
        print *, "  Last element:", vector_1d(dims(1))
        
        call imas_h5read_free_c(data_ptr, c_loc(dims(1)))
    end if

    print *, "--- Fortran tests completed ---"

end program test_imas_h5read_fortran
