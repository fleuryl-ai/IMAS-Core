function vardata = imas_h5read(source, varname, start, count, stride)
%IMAS_H5READ Read data from IMAS HDF5 file.
%   VARDATA = IMAS_H5READ(SOURCE, VARNAME) reads all the data from the 
%   variable VARNAME contained in SOURCE.
%
%   VARDATA = IMAS_H5READ(SOURCE, VARNAME, START, COUNT) reads a subset of
%   data. START is the 1-based starting index for each dimension. COUNT is
%   the number of elements to read along each dimension. Use Inf in COUNT
%   to read until the end of a dimension.
%
%   VARDATA = IMAS_H5READ(SOURCE, VARNAME, START, COUNT, STRIDE) reads a 
%   subset with the interval specified by STRIDE between indices.
%
%   Examples:
%      % Read full signal
%      data = imas_h5read('my_ids', 'equilibrium/time');
%
%      % Read first 100 elements
%      data = imas_h5read('my_ids', 'equilibrium/time', 1, 100);
%
%      % Read multidimensional slice with stride
%      data = imas_h5read('my_ids', 'array_3d', [1 1 1], [2 2 2], [1 2 2]);
%
%   See also: ncread

% This is a stub file. The actual execution is performed by the 
% imas_h5read_mex binary.

error('imas_h5read:mexMissing', ...
      'The imas_h5read MEX file is missing or not compiled for your architecture.');
end
