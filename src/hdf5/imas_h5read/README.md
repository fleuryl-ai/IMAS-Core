# IMAS H5Read API

A simplified, cross-language API for reading IMAS HDF5 data, modeled after the intuitive MATLAB `ncread` function. This project provides a unified interface for C++, MATLAB, and Fortran to access IMAS datasets directly and efficiently.

## Overview

The `imas_h5read` API abstracts the complexities of the underlying IMAS Access Layer and Direct Access API. It allows users to read entire signals or specific hyperslabs (subsets) using `start`, `count`, and `stride` parameters.

- **Simplified Access**: Minimal boilerplate code compared to the standard Access Layer.
- **Unified URI & Master Support**: New support for `imas:hdf5?path=...` URIs and automatic resolution of `master.h5` files.
- **IDS Occurrence Management**: Easily access specific IDS occurrences using the `ids_name:N` syntax.
- **Multidimensional Slicing**: Effortlessly read subsets of large arrays.
- **Strided Sampling**: Efficiently downsample data during the read process.
- **Cross-Language**: Identical behavior and logic across C++, MATLAB, and Fortran.

## Data Source Resolution

The `source` parameter in the `read` function has been enhanced to support flexible data locations:

1.  **Direct HDF5 File**: Provide the absolute or relative path to an IDS HDF5 file (must end with `.h5`).
    *   *Example*: `/path/to/core_profiles.h5`
2.  **IMAS Database Directory**: Provide the path to a directory containing a `master.h5` file. The API will automatically resolve the correct IDS file by inspecting the external links within the master file.
    *   *Example*: `/imas_public/imasdb/west/3/55000/0`
3.  **IMAS HDF5 URI**: Use the `imas:hdf5` protocol to specify the database path.
    *   *Example*: `imas:hdf5?path=/imas_public/imasdb/west/3/55000/0`

### IDS Occurrences

When using a `master.h5` file, you can specify which occurrence of an IDS to read by appending `:N` to the IDS name in the `varname` parameter:

*   `core_profiles/profiles_1d...` : Reads from the default occurrence (0).
*   `core_profiles:0/profiles_1d...` : Also reads from occurrence 0.
*   `core_profiles:1/profiles_1d...` : Reads from occurrence 1 (resolves to the link ending in `//core_profiles_1`).

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

### Examples

#### Basic Read
```cpp
#include "imas_h5read/imas_h5read.h"

std::string source = "imas:hdf5?path=/imas_public/imasdb/west/3/55000/0";
std::string varname = "core_profiles/profiles_1d[0]/grid/rho_tor_norm";

// Read the entire dataset
auto tv = imas_h5read::read(source, varname);
const double* data = tv.as<double>();
```

#### Slicing (Start and Count)
```cpp
// Read 10 elements starting from index 5 (0-based)
auto tv_slice = imas_h5read::read(source, "equilibrium/time", {5}, {10});

// Read a 2x5 sub-block from a 2D array
// Start at (0, 10), take 2 elements along dim 0 and 5 elements along dim 1
auto tv_2d = imas_h5read::read(source, "core_profiles/profiles_1d[0]/t_i_average", 
                               {0, 10}, {2, 5});
```

#### Using INF and Stride
```cpp
using imas_h5read::INF;

// Read from index 100 to the end, with a stride of 2 (skip every other element)
auto tv_stride = imas_h5read::read(source, "equilibrium/time", 
                                   {100}, {INF}, {2});

// Multidimensional stride: sample every 2nd radial point and every 5th time point
// start=[time_idx, radial_idx], count=[how_many_times, how_many_radial]
auto tv_multi = imas_h5read::read(source, "core_profiles/profiles_1d/t_e",
                                  {0, 0}, {INF, INF}, {5, 2});
```

---

## MATLAB API Guide

The MATLAB interface consists of a MEX binary and a `.m` wrapper. It perfectly mimics the standard `ncread` syntax.

### Usage

```matlab
% Read from an IMAS database path
source = 'imas:hdf5?path=/imas_public/imasdb/west/3/55000/0';
data = imas_h5read(source, 'core_profiles:0/equilibrium/time');

% Read a subset (1-based indexing, matching ncread)
% Read 10 elements starting from index 5
data = imas_h5read(source, 'equilibrium/time', 5, 10);

% Multidimensional read with Stride and Inf
start  = [1, 1, 1];
count  = [Inf, Inf, Inf];
stride = [2, 2, 1];
data = imas_h5read(source, 'array_3d', start, count, stride);
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

! Read from a master file directory
call imas_h5read("/path/to/shot/run", "core_profiles:1/equilibrium/time", &
                 data_ptr, dims, dtype, start=[1], count=[100])

if (c_associated(data_ptr)) then
    call c_f_pointer(data_ptr, my_data, dims)
    ! ... use data ...
    call imas_h5read_free_c(data_ptr, c_loc(dims(1)))
end if
```

## Implementation Details

The API is built on top of `direct_access_api.h` and `PanzerDB`. 
- **Source Resolution**: Automatically handles direct file paths, master directories, and URIs.
- **Slicing**: Performed at the HDF5 level whenever possible for optimal performance.
- **Striding**: Currently performed as an in-memory sampling of the requested block to ensure compatibility with all IMAS data structures.
- **Indexing**: 
    - C++ uses **0-based** indexing.
    - MATLAB and Fortran use **1-based** indexing for user convenience.
- **Dimensions**: In MATLAB, dimensions are automatically reversed to account for the Row-Major (C) to Column-Major (Fortran/MATLAB) difference, ensuring that `data(i,j,k)` in MATLAB matches the expected physical layout.
