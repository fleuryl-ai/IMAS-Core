module imas_h5read_mod
    use iso_c_binding
    implicit none

    ! Constant for reading until the end
    integer, parameter :: IMAS_INF = -1

    ! Data types matching imas::direct_access::DataType
    integer, parameter :: IMAS_TYPE_FLOAT          = 0
    integer, parameter :: IMAS_TYPE_DOUBLE         = 1
    integer, parameter :: IMAS_TYPE_INT32          = 2
    integer, parameter :: IMAS_TYPE_INT64          = 3
    integer, parameter :: IMAS_TYPE_STRING         = 4
    integer, parameter :: IMAS_TYPE_COMPLEX_FLOAT  = 5
    integer, parameter :: IMAS_TYPE_COMPLEX_DOUBLE = 6

    interface
        ! Low-level C interface
        function imas_h5read_c_internal(source, varname, start, count, stride, rank, &
                                        data_out, type_out, dims_out, rank_out) &
                                        bind(C, name="imas_h5read_c")
            import :: c_char, c_int, c_ptr, c_long_long
            integer(c_int) :: imas_h5read_c_internal
            character(c_char), intent(in) :: source(*), varname(*)
            integer(c_int), intent(in)    :: start(*), count(*), stride(*)
            integer(c_int), value         :: rank
            type(c_ptr), intent(out)      :: data_out, dims_out
            integer(c_int), intent(out)   :: type_out, rank_out
        end function imas_h5read_c_internal

        subroutine imas_h5read_free_c(data_out, dims_out) bind(C, name="imas_h5read_free_c")
            import :: c_ptr
            type(c_ptr), value :: data_out, dims_out
        end subroutine imas_h5read_free_c
    end interface

contains

    ! High-level Fortran wrapper
    ! Note: start and count are optional and 1-based.
    subroutine imas_h5read(source, varname, data_ptr, dims, data_type, start, count, stride)
        character(len=*), intent(in) :: source, varname
        type(c_ptr), intent(out)     :: data_ptr
        integer(kind=8), pointer, intent(out) :: dims(:)
        integer, intent(out)         :: data_type
        integer, intent(in), optional :: start(:), count(:), stride(:)

        character(kind=c_char, len=len_trim(source)+1)  :: source_c
        character(kind=c_char, len=len_trim(varname)+1) :: varname_c
        integer(c_int), allocatable :: start_c(:), count_c(:), stride_c(:)
        integer(c_int) :: rank_c, status, type_c, rank_out_c
        type(c_ptr) :: dims_c_ptr
        integer(kind=8), pointer :: dims_tmp(:)

        source_c  = trim(source) // c_null_char
        varname_c = trim(varname) // c_null_char

        rank_c = 0
        if (present(start)) then
            rank_c = size(start)
            allocate(start_c(rank_c), count_c(rank_c))
            ! Convert 1-based (Fortran) to 0-based (C)
            start_c = int(start - 1, c_int)
            count_c = int(count, c_int)
            
            if (present(stride)) then
                allocate(stride_c(rank_c))
                stride_c = int(stride, c_int)
                status = imas_h5read_c_internal(source_c, varname_c, start_c, count_c, stride_c, rank_c, &
                                                data_ptr, type_c, dims_c_ptr, rank_out_c)
            else
                status = imas_h5read_c_internal(source_c, varname_c, start_c, count_c, c_null_ptr, rank_c, &
                                                data_ptr, type_c, dims_c_ptr, rank_out_c)
            end if
        else
            status = imas_h5read_c_internal(source_c, varname_c, c_null_ptr, c_null_ptr, c_null_ptr, 0, &
                                            data_ptr, type_c, dims_c_ptr, rank_out_c)
        end if

        if (status /= 0) then
            data_ptr = c_null_ptr
            return
        end if

        data_type = int(type_c)
        
        ! Map the C dims array to a Fortran pointer
        call c_f_pointer(dims_c_ptr, dims_tmp, [int(rank_out_c, 8)])
        
        ! Note: Caller is responsible for freeing data_ptr and dims_c_ptr 
        ! via imas_h5read_free_c when finished.
        dims => dims_tmp

    end subroutine imas_h5read

end module imas_h5read_mod
