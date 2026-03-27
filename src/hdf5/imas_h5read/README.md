# IMAS H5Read API

A simplified, cross-language API for reading IMAS HDF5 data, modeled after the intuitive MATLAB `ncread` function. This project provides a unified interface for C++, MATLAB, and Fortran to access IMAS datasets directly and efficiently.

## Overview

The `imas_h5read` API abstracts the complexities of the underlying IMAS Access Layer and Direct Access API. It allows users to read entire signals or specific hyperslabs (subsets) using `start`, `count`, and `stride` parameters.

- **Simplified Access**: Minimal boilerplate code compared to the standard Access Layer.
- **Multidimensional Slicing**: Effortlessly read subsets of large arrays.
- **Strided Sampling**: Efficiently downsample data during the read process.
- **Cross-Language**: Identical behavior and logic across C++, MATLAB, and Fortran.

## C++ API Guide

The C++ API is the core implementation. It utilizes the `imas::direct_access::TensorView` to provide managed access to data buffers.

### Signatures

```cpp
namespace imas_h5read {
    const int INF = -1; // Use to read until the end of a dimension

    // Read full dataset
    TensorView read(const std::string& source, const std::string& varname);

    // Read multidimensional slice
    TensorView read(const std::string& source, const std::string& varname, 
                    const std::vector<int>& start, const std::vector<int>& count);

    // Read multidimensional slice with stride
    TensorView read(const std::string& source, const std::string& varname, 
                    const std::vector<int>& start, const std::vector<int>& count, 
                    const std::vector<int>& stride);
}
```

### Example

```cpp
#include "imas_h5read/imas_h5read.h"

// Read a 2x2x2 sub-block from a 3D array
auto tv = imas_h5read::read("my_data", "core_profiles/profiles_1d[0]/grid/rho_tor_norm", 
                            {0, 0, 0}, {2, 2, 2});

// Access the data
const double* data = tv.as<double>();
```

---

## MATLAB API Guide

The MATLAB interface consists of a MEX binary and a `.m` wrapper. It perfectly mimics the standard `ncread` syntax.

### Usage

```matlab
% Read full signal
data = imas_h5read('my_ids', 'equilibrium/time');

% Read a subset (1-based indexing, matching ncread)
% Read 10 elements starting from index 5
data = imas_h5read('my_ids', 'equilibrium/time', 5, 10);

% Multidimensional read with Stride and Inf
start  = [1, 1, 1];
count  = [Inf, Inf, Inf];
stride = [2, 2, 1];
data = imas_h5read('my_ids', 'array_3d', start, count, stride);
```

---

## Fortran API Guide

The Fortran API provides a high-level module (`imas_h5read_mod`) that handles C-interoperability and memory management.

### Usage

```fortran
use imas_h5read_mod
type(c_ptr) :: data_ptr
integer(kind=8), pointer :: dims(:)
integer :: dtype
real(8), pointer :: my_data(:)

! Read using optional arguments (1-based indexing)
call imas_h5read("my_ids", "equilibrium/time", data_ptr, dims, dtype, &
                 start=[1], count=[100])

if (c_associated(data_ptr)) then
    call c_f_pointer(data_ptr, my_data, dims)
    ! ... use data ...
    call imas_h5read_free_c(data_ptr, c_loc(dims(1)))
end if
```

## Implementation Details

The API is built on top of `direct_access_api.h` and `PanzerDB`. 
- **Slicing**: Performed at the HDF5 level whenever possible for optimal performance.
- **Striding**: Currently performed as an in-memory sampling of the requested block to ensure compatibility with all IMAS data structures.
- **Indexing**: 
    - C++ uses **0-based** indexing.
    - MATLAB and Fortran use **1-based** indexing for user convenience.
- **Dimensions**: In MATLAB, dimensions are automatically reversed to account for the Row-Major (C) to Column-Major (Fortran/MATLAB) difference, ensuring that `data(i,j,k)` in MATLAB matches the expected physical layout.
