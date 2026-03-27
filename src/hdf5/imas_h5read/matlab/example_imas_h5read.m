% EXAMPLE_IMAS_H5READ
% This script demonstrates how to use the imas_h5read API in MATLAB.
% It mimics the behavior of the standard 'ncread' function.

filename = 'test_imas_h5read'; % .h5 is added automatically by the API

fprintf('--- MATLAB: Testing imas_h5read ---\n');

%% 1. Read a simple scalar
% Equivalent to: ncread(filename, varname)
scalar_val = imas_h5read(filename, '/test/scalar_int');
fprintf('Scalar value read: %d\n', scalar_val);

%% 2. Read a full 1D vector
% Equivalent to: ncread(filename, varname)
vector_data = imas_h5read(filename, '/test/vector_double');
fprintf('Vector size: %d\n', length(vector_data));
disp('Vector content:');
disp(vector_data');

%% 3. Read a slice of data (start, count)
% Read 2 elements starting from the 2nd element.
% Note: Indices are 1-based in MATLAB, consistent with ncread.
start = 2;
count = 2;
slice_data = imas_h5read(filename, '/test/vector_double', start, count);
fprintf('Slice data (indices 2 and 3): [%.1f, %.1f]\n', slice_data(1), slice_data(2));

%% 4. Read multidimensional data with Inf
% Read everything from the 2nd plane onwards in a 3D array.
% The 3D array generated in tests is (2 planes x 3 rows x 4 columns).
% In MATLAB, dimensions are reversed to maintain Column-Major compatibility.
start_3d = [1, 1, 2]; % [Column, Row, Plane]
count_3d = [Inf, Inf, Inf];
array_3d = imas_h5read(filename, '/test/array_3d', start_3d, count_3d);

fprintf('3D Array size read with Inf: %dx%dx%d\n', size(array_3d, 1), size(array_3d, 2), size(array_3d, 3));

%% 5. Read with a stride (sampling)
% Read every 2nd element in the columns and rows of the 3D array.
start_s  = [1, 1, 1];
count_s  = [Inf, Inf, Inf];
stride_s = [2, 2, 1]; % Sample every 2nd column and row, but all planes.
sampled_data = imas_h5read(filename, '/test/array_3d', start_s, count_s, stride_s);

fprintf('Sampled array size (stride [2,2,1]): %dx%dx%d\n', ...
        size(sampled_data, 1), size(sampled_data, 2), size(sampled_data, 3));

fprintf('--- MATLAB examples completed ---\n');
