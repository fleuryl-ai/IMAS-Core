/*
 * @file check.h
 * @brief Shared `CHECK(cond, msg)` macro + `nfail` counter for the
 *        AL-level backend tests that need to accumulate several
 *        assertions before deciding pass/fail.
 *
 * Five test files (test_al_gap_*) used to ship their own byte-identical
 * copies of `static int nfail = 0;` and `#define CHECK(...)`. They now
 * `#include "fixtures/check.h"` instead.
 *
 * Each including file is expected to:
 *   - have a TU-local `static int nfail = 0;` (so the macro works);
 *   - print and `return 0` when `nfail == 0`; otherwise `return 1`.
 */
#ifndef HDF5_BACKEND_TESTS_CHECK_H
#define HDF5_BACKEND_TESTS_CHECK_H

#include <iostream>

#ifndef CHECK
#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++nfail; } } while (0)
#endif

#endif  // HDF5_BACKEND_TESTS_CHECK_H
