// @file  test_al_dynamic_aos_nested_only_getshape.cpp
// @brief Regression test for getAOSShape on a DYNAMIC AoS whose leaves are all
//        behind NESTED static AoS (e.g. time_slice/ggd/theta/values).
//
//
// The size of a dynamic AoS must be the number of time slices that were
// written, even when the only data leaves are deeply nested (no direct data
// child carries the time index). Previously the size was derived from direct
// children only and came back 0, truncating global reads to nothing.
//
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

#define RESET ""
#define BOLD ""
#define GREEN ""
#define RED ""

const std::string URI = "imas:hdf5?path=./test_db_dynamic_aos_nested_only";

// time_slice (dynamic, timebase "time")
//   / ggd   (static, size 1)
//   /   / theta (static, size 2)
//   /   /   / values (1D field, size 3)
static const int N_T  = 5;   // time slices
static const int N_TH = 2;   // theta
static const int N_V  = 3;   // values (1D)

int main() {
    try {
        std::cout << BOLD
                  << "\n=== Dynamic AoS with ONLY nested static AoS (time_slice/ggd/theta/values) ===\n"
                  << RESET;

        // ---- Phase 1: GLOBAL write of N_T slices ----------------------------
        {
            std::cout << "\n--- Phase 1: Writing " << N_T << " slices (global) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            auto version = backend.getVersion(&dataEntryCtx);
            if (version == std::make_pair(1, 0)) {
                backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
                return 0;
            }

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", WRITE_OP);
            backend.beginAction(&opCtx);

            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "",
                              &homogeneous_time, alconst::integer_data, 0, nullptr);

            std::vector<double> times(N_T);
            for (int t = 0; t < N_T; ++t) times[t] = 0.1 * t;
            int time_dim = 1;
            int time_size[] = {N_T};
            backend.writeData(&opCtx, "time", "time", times.data(),
                              alconst::double_data, time_dim, time_size);

            ArraystructContext tsCtx(&opCtx, "time_slice", "time");
            int ts_size = N_T;
            backend.beginArraystructAction(&tsCtx, &ts_size);

            for (int t = 0; t < N_T; ++t) {
                ArraystructContext ggdCtx(&tsCtx, "ggd", "");
                int ggd_size = 1;
                backend.beginArraystructAction(&ggdCtx, &ggd_size);

                ArraystructContext thetaCtx(&ggdCtx, "theta", "");
                int theta_size = N_TH;
                backend.beginArraystructAction(&thetaCtx, &theta_size);

                for (int th = 0; th < N_TH; ++th) {
                    double vals[N_V];
                    for (int v = 0; v < N_V; ++v) {
                        // value uniquely identifiable by (t, th, v)
                        vals[v] = 1000.0 * t + 10.0 * th + v + 0.5;
                    }
                    int dim = 1;
                    int size[] = {N_V};
                    backend.writeData(&thetaCtx, "values", "", vals,
                                      alconst::double_data, dim, size);
                    if (th < N_TH - 1) thetaCtx.nextIndex(1);
                }
                backend.endAction(&thetaCtx);
                backend.endAction(&ggdCtx);

                if (t < N_T - 1) tsCtx.nextIndex(1);
            }

            backend.endAction(&tsCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Write completed.\n" << RESET;
        }

        // ---- Phase 2: GLOBAL read — size must equal N_T ---------------------
        {
            std::cout << "\n--- Phase 2: Global read (size must be " << N_T << ") ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            OperationContext opCtx(&dataEntryCtx, "test_ids", "", READ_OP);
            backend.beginAction(&opCtx);

            ArraystructContext tsCtx(&opCtx, "time_slice", "time");
            int ts_size = 0;
            backend.beginArraystructAction(&tsCtx, &ts_size);

            std::cout << "  time_slice global size = " << ts_size
                      << " (expected " << N_T << ")\n";
            if (ts_size != N_T) {
                std::cerr << RED
                          << "[FAIL] global size != " << N_T << " (got " << ts_size
                          << "). Dynamic AoS with only nested leaves is truncated.\n"
                          << RESET;
                return 1;
            }

            for (int t = 0; t < ts_size; ++t) {
                ArraystructContext ggdCtx(&tsCtx, "ggd", "");
                int ggd_size = 0;
                backend.beginArraystructAction(&ggdCtx, &ggd_size);
                if (ggd_size != 1) {
                    std::cerr << RED << "[FAIL] ggd size != 1 (got " << ggd_size << ")\n" << RESET;
                    return 1;
                }

                ArraystructContext thetaCtx(&ggdCtx, "theta", "");
                int theta_size = 0;
                backend.beginArraystructAction(&thetaCtx, &theta_size);
                if (theta_size != N_TH) {
                    std::cerr << RED << "[FAIL] theta size != " << N_TH
                              << " (got " << theta_size << ")\n" << RESET;
                    return 1;
                }

                for (int th = 0; th < theta_size; ++th) {
                    void* data = nullptr;
                    int type = alconst::double_data;
                    int dim = 0;
                    int size[H5S_MAX_RANK];
                    backend.readData(&thetaCtx, "values", "", &data, &type, &dim, size);
                    if (type != alconst::double_data || dim != 1 || size[0] != N_V) {
                        std::cerr << RED
                                  << "[FAIL] values meta wrong at t=" << t << " th=" << th
                                  << " (type=" << type << " dim=" << dim << ")\n" << RESET;
                        return 1;
                    }
                    double* vp = static_cast<double*>(data);
                    for (int v = 0; v < N_V; ++v) {
                        double expected = 1000.0 * t + 10.0 * th + v + 0.5;
                        if (std::abs(vp[v] - expected) > 1e-9) {
                            std::cerr << RED
                                      << "[FAIL] values mismatch t=" << t << " th=" << th
                                      << " v=" << v << ": expected " << expected
                                      << " got " << vp[v] << RESET << std::endl;
                            return 1;
                        }
                    }
                    free(data);
                    if (th < theta_size - 1) thetaCtx.nextIndex(1);
                }
                backend.endAction(&thetaCtx);
                backend.endAction(&ggdCtx);

                if (t < ts_size - 1) tsCtx.nextIndex(1);
            }

            backend.endAction(&tsCtx);
            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Global read validated.\n" << RESET;
        }

        // ---- Phase 3: SLICE read at every time ------------------------------
        {
            std::cout << "\n--- Phase 3: Slice read at each slice ---\n";
            for (int t = 0; t < N_T; ++t) {
                double t_req = 0.1 * t;
                DataEntryContext dataEntryCtx(URI);
                HDF5Backend backend;
                backend.openPulse(&dataEntryCtx, OPEN_PULSE);

                OperationContext opCtx(&dataEntryCtx, "test_ids", READ_OP,
                                       alconst::slice_op, t_req, alconst::closest_interp);
                backend.beginAction(&opCtx);

                ArraystructContext tsCtx(&opCtx, "time_slice", "time");
                int ts_size = 0;
                backend.beginArraystructAction(&tsCtx, &ts_size);
                if (ts_size != 1) {
                    std::cerr << RED << "[FAIL] slice size != 1 (got " << ts_size << ")\n" << RESET;
                    return 1;
                }

                ArraystructContext ggdCtx(&tsCtx, "ggd", "");
                int ggd_size = 0;
                backend.beginArraystructAction(&ggdCtx, &ggd_size);
                if (ggd_size != 1) {
                    std::cerr << RED << "[FAIL] ggd slice size != 1 (got " << ggd_size << ")\n" << RESET;
                    return 1;
                }

                ArraystructContext thetaCtx(&ggdCtx, "theta", "");
                int theta_size = 0;
                backend.beginArraystructAction(&thetaCtx, &theta_size);
                if (theta_size != N_TH) {
                    std::cerr << RED << "[FAIL] theta slice size != " << N_TH
                              << " (got " << theta_size << ")\n" << RESET;
                    return 1;
                }

                bool ok = true;
                for (int th = 0; th < theta_size; ++th) {
                    void* data = nullptr;
                    int type = alconst::double_data;
                    int dim = 0;
                    int size[H5S_MAX_RANK];
                    backend.readData(&thetaCtx, "values", "", &data, &type, &dim, size);
                    double* vp = static_cast<double*>(data);
                    int n = (dim >= 1) ? size[0] : 0;
                    for (int v = 0; v < N_V; ++v) {
                        double expected = 1000.0 * t + 10.0 * th + v + 0.5;
                        if (n <= v || std::abs(vp[v] - expected) > 1e-9) { ok = false; break; }
                    }
                    free(data);
                    if (!ok) break;
                    if (th < theta_size - 1) thetaCtx.nextIndex(1);
                }
                backend.endAction(&thetaCtx);
                backend.endAction(&ggdCtx);
                backend.endAction(&tsCtx);
                backend.endAction(&opCtx);
                backend.closePulse(&dataEntryCtx, OPEN_PULSE);

                if (!ok) {
                    std::cerr << RED << "[FAIL] slice read at t=" << t_req << "\n" << RESET;
                    return 1;
                }
            }
            std::cout << GREEN << "[OK] Slice reads validated.\n" << RESET;
        }

        std::cout << GREEN << "\n=== [SUCCESS] dynamic AoS with nested-only leaves ===\n" << RESET;
    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << std::endl << RESET;
        return 1;
    }
    return 0;
}
