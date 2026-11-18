/*
 * @file panzerdb_helper.h
 * @brief Shared helpers for the `test_engine_*` tests that drive the
 *        PanzerDB engine directly.
 *
 * `find_leaf()` was previously copy-pasted (byte-identical in two of the
 * three files) at the top of the engine write/read tests. It is now
 * provided once here and `#include`d by the consuming tests.
 */
#ifndef HDF5_BACKEND_TESTS_PANZERDB_HELPER_H
#define HDF5_BACKEND_TESTS_PANZERDB_HELPER_H

#include <cstdint>
#include <string>
#include <vector>

#include "panzerdb.h"

/**
 * @brief Finds an index leaf by its full instance path and optional time index.
 * @param leaves    The index table returned by PanzerDB::getLeaves().
 * @param path      Full instance path, e.g. "A/0/B/signal".
 * @param time_index When >= 0, the leaf's first time step must match; -1 (default)
 *                   matches any leaf at `path` regardless of its time step.
 * @return A pointer to the matching leaf, or nullptr when none is found.
 */
inline const PanzerDB::Leaf* find_leaf(const std::vector<PanzerDB::Leaf>& leaves,
                                       const std::string& path,
                                       int64_t time_index = -1) {
    for (const auto& leaf : leaves) {
        if (leaf.path == path &&
            (time_index == -1 || leaf.time_index == static_cast<uint64_t>(time_index))) {
            return &leaf;
        }
    }
    return nullptr;
}

#endif  // HDF5_BACKEND_TESTS_PANZERDB_HELPER_H
