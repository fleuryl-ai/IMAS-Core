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

/**
 * @brief Normalises a (potentially partial) path so it can be matched against
 *        the flat leaf paths of the PanzerDB index.
 *
 * The AL gives CUMULATIVE paths to every AoS level ("core_sources/source"), so a
 * path that runs through an open context necessarily STARTS with that context's
 * path; the part behind it is the (single, '&'-flattened) name stored under that
 * AoS. Matching that context path anywhere else in the string (the old rfind)
 * picked the wrong occurrence whenever a segment name was repeated further down
 * the hierarchy (to_improve.md point 5).
 *
 * @param ctx_paths Open AoS chain paths, deepest context first ("" entries ignored).
 * @param path Raw path (may be empty, "/", "time", "A/B", "a/b/F/g/data", ...).
 * @return The normalised path ("" for empty input).
 */
inline std::string sanitizeAosPath(const std::vector<std::string>& ctx_paths,
                                   const std::string& path) {
    if (path.empty()) return "";
    if (path == "/time") return "time";
    if (path == "time") return path;

    std::string remaining = path;
    const bool has_leading_slash = (remaining[0] == '/');
    if (has_leading_slash) remaining.erase(0, 1);

    std::vector<std::string> segments; // pushed from the end to the start

    for (const std::string& ctx_path : ctx_paths) {
        const size_t len = ctx_path.size();
        if (len == 0) continue;
        const bool matched = (remaining.size() == len)
                             || (remaining.size() > len && remaining[len] == '/'
                                 && remaining.compare(0, len, ctx_path) == 0);
        if (!matched) continue;

        std::string suffix = remaining.substr(std::min(len + 1, remaining.size()));
        std::replace(suffix.begin(), suffix.end(), '/', '&');
        if (!suffix.empty()) segments.push_back(suffix);

        std::string level = ctx_path;
        std::replace(level.begin(), level.end(), '/', '&');
        segments.push_back(level);

        remaining.clear();
        break;
    }

    // Levels not covered by the open AoS chain stay distinct '/'-separated levels
    if (!remaining.empty()) {
        size_t pos = 0;
        while ((pos = remaining.rfind('/')) != std::string::npos) {
            std::string part = remaining.substr(pos + 1);
            if (!part.empty()) segments.push_back(part);
            remaining = remaining.substr(0, pos);
        }
        if (!remaining.empty()) segments.push_back(remaining);
    }

    std::string result;
    for (size_t i = segments.size(); i-- > 0;) result += "/" + segments[i];

    if (!has_leading_slash && !result.empty()) result.erase(0, 1);
    return result;
}

#endif
