// @file  test_al_global_code_output_1slice.cpp
// @brief Regression: a scalar 1D dynamic signal with exactly ONE time slice
//        (e.g. code/output_flag with a single flag, the reflectometer_profile
//        case) must read back as 1D(1) on a GLOBAL read, not as a 0D scalar.
//
//        The slice path already applies the AL convention (to_improve.md
//        point 3, shouldPromoteTimeScalarOnSlice): a time-dependent scalar
//        outside a dynamic AoS gets a slice dimension of size 1. The global
//        path used to add the time dimension only when >1 slices were
//        present, so a 1-slice scalar series degraded to 0D and AL raised:
//          "Wrong dimension of Data returned by backend:
//           expected int in 1D but got int in 0D"  (al_lowlevel.cpp)
//
//        The control case (a purely static scalar, no timebase) must remain
//        0D — the promotion must not over-fire.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    const std::string URI = "imas:hdf5?path=./test_db_global_code_output1";
    if (fs::exists("test_db_global_code_output1")) fs::remove_all("test_db_global_code_output1");

    try {
        // ---------------- Phase 1: write 1 time point + 1-slice signals ----
        {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            int homogeneous_time = 1;  // static scalar int (the 0D control case)
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "",
                              &homogeneous_time, alconst::integer_data, 0, nullptr);

            double t0 = 10.0;
            int time_size[] = {1};
            backend.writeData(&opCtx, "time", "time", &t0, alconst::double_data, 1, time_size);

            // code/output_flag: dynamic 1D, spatial 0D, exactly ONE slice
            ArraystructContext codeCtx(&opCtx, "code", "");
            int code_size = 1;
            backend.beginArraystructAction(&codeCtx, &code_size);
            int flag = 7;
            int flag_size[] = {1};
            backend.writeData(&opCtx, "output_flag", "time", &flag,
                              alconst::integer_data, 1, flag_size);
            backend.endAction(&codeCtx);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
        }

        // ---------------- Phase 2: GLOBAL read (the failing path) ---------
        {
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", READ_OP);  // GLOBAL_OP
            backend.beginAction(&opCtx);

            // Control: static scalar (no timebase) must stay 0D.
            void* h = nullptr;
            int h_type = alconst::integer_data;
            int h_dim = 0;
            backend.readData(&opCtx, "ids_properties/homogeneous_time", "",
                             &h, &h_type, &h_dim, nullptr);
            if (h_dim != 0 || h == nullptr) {
                std::cerr << "FAIL: homogeneous_time should read as 0D scalar, got dim="
                          << h_dim << "\n";
                return 1;
            }
            if (*((int*)h) != 1) { std::cerr << "FAIL: homogeneous_time value\n"; return 1; }
            free(h);

            // The node in question: 1-slice scalar time series -> 1D(1).
            ArraystructContext codeCtx(&opCtx, "code", "");
            int size = 0;
            backend.beginArraystructAction(&codeCtx, &size);
            assert(size == 1);

            void* data = nullptr;
            int type = alconst::integer_data;
            int dim = 1;             // AL declares the node as 1D (IDS: INT_1D, time)
            int dims[6] = {0};

            backend.readData(&codeCtx, "output_flag", "time", &data, &type, &dim, dims);

            if (dim != 1) {
                std::cerr << "FAIL: output_flag (1-slice scalar time series) read as "
                          << dim << "D on a GLOBAL read; expected 1D(1)\n";
                return 1;
            }
            if (dims[0] != 1) {
                std::cerr << "FAIL: output_flag size[0]=" << dims[0] << ", expected 1\n";
                return 1;
            }
            if (data == nullptr || *((int*)data) != 7) {
                std::cerr << "FAIL: output_flag value\n";
                return 1;
            }
            free(data);
            backend.endAction(&codeCtx);

            // The 1-slice timebase itself is the same rank-0 case.
            void* td = nullptr;
            int tt_type = alconst::double_data;
            int tt_dim = 1;
            int tt_dims[6] = {0};
            backend.readData(&opCtx, "time", "time", &td, &tt_type, &tt_dim, tt_dims);
            if (tt_dim != 1 || tt_dims[0] != 1) {
                std::cerr << "FAIL: time (1-slice) read as " << tt_dim
                          << "D; expected 1D(1)\n";
                return 1;
            }
            if (td == nullptr || std::abs(*((double*)td) - 10.0) > 1e-9) {
                std::cerr << "FAIL: time value\n";
                return 1;
            }
            free(td);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
        }

        std::cout << "[OK] 1-slice scalar time series reads back as 1D(1) on a "
                     "GLOBAL read (and static scalars stay 0D)\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception (reproduces the AL dimension error): " << e.what() << "\n";
        return 1;
    }
    return 0;
}
