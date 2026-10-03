#ifndef HDF5_AOS_PATH_HELPERS_H
#define HDF5_AOS_PATH_HELPERS_H 1

#include <string>
#include <vector>
#include <algorithm>
#include "al_backend.h"

/**
 * @brief Shared AoS path helpers for the v2 backend (writer + read strategies).
 *
 * The AL hands full paths to each ArraystructContext (e.g. "core_sources/source"),
 * while the PanzerDB storage levels carry ONE name each. The local name of a
 * level is therefore its full path minus the parent's, with '/' flattened to '&'
 * so that a nested name ("constraints/x_point") stays ONE storage level
 * ("constraints&x_point"). This block used to be copy-pasted ~6 times across
 * the backend (writer + read strategies); it now has exactly one definition
 * here (to_improve.md point 4).
 */

/**
 * @brief Local name of an AoS level relative to its parent ('/' -> '&').
 * @param ctx The context (must be an ArraystructContext).
 * @return The flattened local name of this level.
 */
inline std::string localAosName(Context* ctx) {
    ArraystructContext* arr = static_cast<ArraystructContext*>(ctx);
    std::string full_path = arr->getPath();
    ArraystructContext* parent = arr->getParent();
    if (parent && parent->getType() == CTX_ARRAYSTRUCT_TYPE) {
        const std::string parent_path = parent->getPath();
        if (full_path.size() > parent_path.size() && full_path.rfind(parent_path + "/", 0) == 0) {
            full_path = full_path.substr(parent_path.size() + 1);
        }
    }
    std::replace(full_path.begin(), full_path.end(), '/', '&');
    return full_path;
}

/**
 * @brief Collect the open AoS chain (names + indices, root first).
 * @param curr Start of the walk (stops at the first non-ArraystructContext).
 * @param names Output: one local name per open AoS level, root first.
 * @param indices Output: the element index of each of those levels.
 * @param timed Optional output: getTimed() per level (same order as names).
 */
inline void collectAosChain(Context* curr,
                            std::vector<std::string>& names,
                            std::vector<int>& indices,
                            std::vector<bool>* timed = nullptr) {
    while (curr != nullptr && curr->getType() == CTX_ARRAYSTRUCT_TYPE) {
        ArraystructContext* arr = static_cast<ArraystructContext*>(curr);
        names.insert(names.begin(), localAosName(arr));
        indices.insert(indices.begin(), arr->getIndex());
        if (timed) timed->insert(timed->begin(), arr->getTimed());
        curr = arr->getParent();
    }
}

#endif
