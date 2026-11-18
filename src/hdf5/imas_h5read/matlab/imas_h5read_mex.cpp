#include "mex.h"
#include "imas_h5read.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>

// Helper to convert MATLAB numeric vector to std::vector<int>
std::vector<int> mxToVectorInt(const mxArray* mxV, const char* argName) {
    if (!mxIsNumeric(mxV)) {
        mexErrMsgIdAndTxt("imas_h5read:invalidInput", "Argument '%s' must be numeric.", argName);
    }
    size_t n = mxGetNumberOfElements(mxV);
    double* ptr = mxGetPr(mxV);
    std::vector<int> result(n);
    for (size_t i = 0; i < n; ++i) {
        if (mxIsInf(ptr[i])) {
            result[i] = imas_h5read::INF;
        } else {
            result[i] = static_cast<int>(ptr[i]);
        }
    }
    return result;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Check input arguments
    if (nrhs != 2 && nrhs != 4 && nrhs != 5) {
        mexErrMsgIdAndTxt("imas_h5read:nargin", "Wrong number of input arguments. Expected 2, 4, or 5.");
    }

    // Parse source and varname
    if (!mxIsChar(prhs[0]) || !mxIsChar(prhs[1])) {
        mexErrMsgIdAndTxt("imas_h5read:invalidInput", "Source and varname must be strings.");
    }
    char* source_c = mxArrayToUTF8String(prhs[0]);
    char* varname_c = mxArrayToUTF8String(prhs[1]);
    std::string source(source_c);
    std::string varname(varname_c);
    mxFree(source_c);
    mxFree(varname_c);

    try {
        imas::direct_access::TensorView tv;

        if (nrhs == 2) {
            tv = imas_h5read::read(source, varname);
        } else {
            std::vector<int> start = mxToVectorInt(prhs[2], "start");
            std::vector<int> count = mxToVectorInt(prhs[3], "count");
            
            // Adjust MATLAB 1-based indexing to C++ 0-based indexing
            for (auto& s : start) {
                if (s != imas_h5read::INF) s -= 1;
            }

            if (nrhs == 4) {
                tv = imas_h5read::read(source, varname, start, count);
            } else {
                std::vector<int> stride = mxToVectorInt(prhs[4], "stride");
                tv = imas_h5read::read(source, varname, start, count, stride);
            }
        }

        // Handle empty or unknown results
        if (tv.type() == imas::direct_access::DataType::UNKNOWN || tv.total_elements() == 0) {
             plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
             return;
        }

        // Prepare MATLAB dimensions
        const auto& dims = tv.dims();
        std::vector<mwSize> mwDims;
        if (dims.empty()) {
            mwDims = {1, 1}; // Scalar
        } else {
            // Important: IMAS/HDF5 uses Row-Major (C-style). 
            // MATLAB uses Column-Major (Fortran-style).
            // To match ncread's behavior where the first dimension in the file
            // remains the first dimension in MATLAB, we must reverse dimensions
            // before creating the array and then the data will be transposable.
            for (auto d : dims) mwDims.push_back(static_cast<mwSize>(d));
            std::reverse(mwDims.begin(), mwDims.end());
            if (mwDims.size() == 1) mwDims.push_back(1);
        }

        // Map imas::direct_access::DataType to MATLAB mxClassID
        mxClassID classId;
        mxComplexity complexity = mxREAL;
        size_t elementSize = 0;

        switch (tv.type()) {
            case imas::direct_access::DataType::DOUBLE:
                classId = mxDOUBLE_CLASS; elementSize = sizeof(double); break;
            case imas::direct_access::DataType::FLOAT:
                classId = mxSINGLE_CLASS; elementSize = sizeof(float); break;
            case imas::direct_access::DataType::INT32:
                classId = mxINT32_CLASS; elementSize = sizeof(int32_t); break;
            case imas::direct_access::DataType::INT64:
                classId = mxINT64_CLASS; elementSize = sizeof(int64_t); break;
            case imas::direct_access::DataType::COMPLEX_DOUBLE:
                classId = mxDOUBLE_CLASS; complexity = mxCOMPLEX; 
                elementSize = sizeof(std::complex<double>); break;
            default:
                mexErrMsgIdAndTxt("imas_h5read:type", "Unsupported data type for MATLAB.");
        }

        // Create the output array
        plhs[0] = mxCreateNumericArray(mwDims.size(), mwDims.data(), classId, complexity);
        
        if (complexity == mxREAL) {
            void* dst = mxGetData(plhs[0]);
            std::memcpy(dst, tv.data(), tv.total_elements() * elementSize);
        } else {
            // Complex data requires separate buffers in MATLAB MEX API (non-interleaved)
            double* pr = mxGetPr(plhs[0]);
            double* pi = mxGetPi(plhs[0]);
            const std::complex<double>* src = static_cast<const std::complex<double>*>(tv.data());
            for (size_t i = 0; i < tv.total_elements(); ++i) {
                pr[i] = src[i].real();
                pi[i] = src[i].imag();
            }
        }

    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("imas_h5read:exception", e.what());
    }
}
