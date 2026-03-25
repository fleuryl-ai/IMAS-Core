# Direct Access API: Usage Scenarios and Examples

This document provides a series of detailed examples on how to use the `imas::direct_access` API to read data from an IMAS database. It covers basic reading, handling of complex data structures, time-based selections, and advanced in-memory operations.

---

### 1. Basic Reading: Accessing a Scalar Value

The most fundamental use case is reading a single data point, or scalar. The API returns a `TensorView` object, which provides access to the data, its type, and its dimensions.

```cpp
// main.cpp

#include "direct_access_api.h"
#include <iostream>
#include <cassert>

void read_scalar_example() {
    // Define the IDS name and the path to the data.
    // This assumes a data file named 'my_ids.h5' exists.
    const std::string ids_name = "my_ids";
    const std::string path = "summary/time"; // Path to a scalar double

    // Read the data using the direct access API.
    imas::direct_access::TensorView view = imas::direct_access::read_tensor(ids_name, path);

    // --- Validate the returned TensorView ---

    // 1. Check the dimensions. For a scalar, it should be a 1D tensor with 1 element.
    assert(view.dims().size() == 1);
    assert(view.dims()[0] == 1);

    // 2. Check the data type.
    assert(view.type() == imas::direct_access::DataType::DOUBLE);

    // 3. Access the data using the typed accessor `as<T>()`.
    const double* data_ptr = view.as<double>();
    double value = data_ptr[0];

    std::cout << "Successfully read scalar value: " << value << std::endl;
}
```

---

### 2. Working with Arrays of Structures (AoS)

The API provides a powerful path syntax to select data from within nested Arrays of Structures.

#### 2.1 Index Selection (Reading a single element)

To select a single element from an array, use the `[index]` syntax.

```cpp
// Reads the name of the second diagnostic (index 1).
const std::string path = "diagnostics[1]/name";
auto view = imas::direct_access::read_tensor("my_ids", path);
// view will contain a single string.
```

#### 2.2 Index Slicing (Reading a range of elements)

To select a contiguous range of elements, use the `[start:end]` syntax. The range is half-open, `[start, end)`.

```cpp
// Reads `z_ion` for the time steps with index 1 and 2.
// This path implicitly traverses all 'ion' and 'state' elements within each time step.
const std::string path = "profiles_1d[1:3]/ion/state/z_ion";
auto view = imas::direct_access::read_tensor("my_ids", path);

// The resulting dimensions will reflect the selections.
// If there are 5 ions and 2 states, the dims will be: {2, 5, 2}
// Dim 0: 2 time steps (from the [1:3] slice)
// Dim 1: 5 ions (from the implicit traversal of 'ion')
// Dim 2: 2 states (from the implicit traversal of 'state')
```

---

### 3. Time-Based Selections in Dynamic AoS

For dynamic AoS, the API supports selections based on time values, including interpolation.

#### 3.1 Time Slicing (Reading a time range)

Use `[time=start:end]` to select all time steps whose time values fall within a given range.

```cpp
// Assume the time vector for 'profiles_1d' is [1.0, 2.0, 3.0, 4.0, 5.0].
// This path will select the data for time steps at 2.0, 3.0, and 4.0.
const std::string path = "profiles_1d[time=1.5:4.5]/ion/state/z_ion";
auto view = imas::direct_access::read_tensor("my_ids", path);

// The first dimension of the resulting TensorView will have a size of 3.
```

#### 3.2 Time Interpolation (Closest)

To get data at a specific point in time, the default behavior is to find the nearest available time step.

```cpp
// Requests data at time t=2.7.
// If the available times are [..., 2.0, 3.0, ...], the API will return the data for t=3.0.
const std::string path = "profiles_1d[time=2.7]/ion/state/z_ion";
auto view = imas::direct_access::read_tensor("my_ids", path);

// The first dimension of the resulting TensorView will have a size of 1.
```

#### 3.3 Time Interpolation (Linear)

To get linearly interpolated data between two time steps, specify the interpolation method.

```cpp
// Requests data at t=2.5, exactly halfway between the time steps 2.0 and 3.0.
const std::string path = "profiles_1d[time=2.5,interp=linear]/ion/state/z_ion";
auto view = imas::direct_access::read_tensor("my_ids", path);

// The returned TensorView will contain a single time slice where each data point
// is the linear interpolation of the values at t=2.0 and t=3.0.
```

---

### 4. Reading a List of Strings (`STR_1D`)

When reading a 1D array of strings, the API returns a 2D `char` buffer for C-style compatibility.

```cpp
// Read all diagnostic names.
const std::string path = "diagnostics/name";
auto view = imas::direct_access::read_tensor("my_ids", path);

assert(view.type() == imas::direct_access::DataType::LIST_OF_STRINGS);
assert(view.dims().size() == 2);

// Dimension 0 is the number of strings.
size_t num_strings = view.dims()[0];
// Dimension 1 is the length of the longest string + 1 (for null terminator).
size_t max_len = view.dims()[1];

const char* data_buffer = view.as<char>();
for (size_t i = 0; i < num_strings; ++i) {
    std::string s(data_buffer + i * max_len);
    std::cout << "Diagnostic " << i << ": " << s << std::endl;
}
```

---

### 5. Accessing Metadata

The API automatically reads associated metadata and attaches it to the `TensorView`.

```cpp
// Assume the node 'z_ion' has a metadata attribute 'units' set to 'eV'.
const std::string path = "profiles_1d[0]/ion[0]/state[0]/z_ion";
auto view = imas::direct_access::read_tensor("my_ids", path);

// Access the metadata map.
const auto& metadata = view.metadata();

if (metadata.count("units")) {
    std::cout << "Units of z_ion: " << metadata.at("units") << std::endl; // Prints "eV"
}
```

---

### 6. Advanced: In-Memory Slicing (Zero-Copy Views)

Once a `TensorView` is loaded into memory, you can create sub-views from it without any data duplication. This is extremely fast and efficient for data exploration.

```cpp
// 1. Load a large, 3D chunk of data from disk into memory.
const std::string path = "profiles_1d[0:5]/ion/state/z_ion"; // Dims: {5, 3, 2}
auto main_view = imas::direct_access::read_tensor("my_ids", path);

// 2. Create a sub-view for the second ion (index 1) across all time steps.
// This operation is nearly instantaneous as it only calculates new metadata (offset, strides).
// No data is copied.
auto ion1_view = main_view.slice({
    imas::direct_access::SliceSelection::all(),   // All time steps from the main_view
    imas::direct_access::SliceSelection::at(1)    // The ion at relative index 1
    // The 'state' dimension is kept implicitly
});

// 3. Validate and use the new view.
// The new view is 2D because the 'ion' dimension was removed by selecting a single index.
assert(ion1_view.dims().size() == 2);
assert(ion1_view.dims()[0] == 5); // 5 time steps
assert(ion1_view.dims()[1] == 2); // 2 states

// The data pointer of 'ion1_view' points to a specific offset within the
// original 'main_view' buffer. You can now work with this smaller, non-contiguous
// slice of data as if it were a simple 2D array.
const double* ion1_data = ion1_view.as<double>();
std::cout << "Data for the first time step of the second ion: " << ion1_data[0] << std::endl;
```
