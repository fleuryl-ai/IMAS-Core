#ifndef I_READ_STRATEGY_H
#define I_READ_STRATEGY_H 1

#include <string>
#include <string_view>
#include <vector>
#include <memory> // For unique_ptr
#include "al_backend.h"
#include <sstream>
#include "panzerdb.h"
#include <algorithm>
#include <cstring>
#include <complex>
#include "al_defs.h"
#include "aos_path_helpers.h"
#include <set>
#include <map>

// Sentinel for the cached ids_properties&homogeneous_time flag (valid values: -1, 0, 1)
constexpr int HOMOG_TIME_UNRESOLVED = -2;

// Forward declarations
class Context;

// Debug macro
#ifdef DEBUG_HDF5_READER_V2
#define DEBUG_PRINT(msg) \
  std::cerr << "[DEBUG " << __func__ << "] " << msg << std::endl
#else
#define DEBUG_PRINT(msg) \
  do {                   \
  } while (0)
#endif

/**
 * @class ReadIndex
 * @brief Read-only index (path/name -> leaves) shared by the strategies over one PanzerDB.
 *
 * The three read strategies (Global, Slice, TimeRange) used by HDF5Reader_v2
 * observe the same file, so they must NOT each open their own PanzerDB and
 * rebuild their own path index (that was 3x the /index load + 3x the cache
 * memory on 10^5-slice files). This class holds the engine-wide indexes once,
 * owned by the reader and shared by every strategy created in the session.
 *
 * The string_view keys and the Leaf pointers reference PanzerDB's internal
 * path/leaf storage, so the ReadIndex keeps the PanzerDB alive through a
 * shared_ptr. Validity is not assumed any more: the generation reported by
 * PanzerDB::leaf_cache_generation() is stamped at build time and checked with
 * ensure_current(), so an engine whose leaf cache gets rebuilt (SWMR refresh, a
 * write+flush on the same engine, ...) forces a rebuild of these maps instead of
 * leaving them dangling (to_improve.md point 6).
 */
class ReadIndex {
    std::shared_ptr<PanzerDB> panzer_db_ptr; // Keeps the indexed engine alive.
    uint64_t built_generation = 0;           // leaf_cache_generation() when built.

    /**
     * @brief Cache for fast access path -> leaves.
     * Key: Full path (string view).
     * Value: Vector of pointers to leaves sharing this path (different time_index).
     */
    std::unordered_map<std::string_view, std::vector<const PanzerDB::Leaf*>> path_cache;

    /**
     * @brief Index by dataset NAME (last path segment) -> leaves.
     *
     * Used by the linear-scan fallbacks: the old code did an O(N) full scan of
     * `getLeaves()` per dataset read because the exact `path_cache` lookups miss
     * when the AL reconstructs a per-element index (e.g. "profiles_1d/5/..." vs
     * stored "profiles_1d/..."). The predicate "leaf.path == name OR (ends with
     * name preceded by '/')" is EXACTLY "last segment == name" (since `name` has
     * no '/', having had '/' replaced by '&'), so keying on the last segment
     * returns the identical candidate set at O(1) — the same leaf.path data that
     * `path_cache` references, so the string_view keys stay valid for the
     * PanzerDB lifetime.
     */
    std::unordered_map<std::string_view, std::vector<const PanzerDB::Leaf*>> name_index;

public:
    /**
     * @brief Constructor. Builds every index from the engine's leaf cache.
     * @param panzer_db The read-only engine to index (READ mode).
     */
    ReadIndex(std::shared_ptr<PanzerDB> panzer_db) : panzer_db_ptr(std::move(panzer_db)) {
        build_indexes();
    }

    ReadIndex(const ReadIndex&) = delete;
    ReadIndex& operator=(const ReadIndex&) = delete;

    /**
     * @brief Builds the path + dataset-name indexes over the current leaf cache.
     */
    void build_indexes() {
        path_cache.clear();
        name_index.clear();
        built_generation = 0;
        if (!panzer_db_ptr) return;

        const auto& leaves = panzer_db_ptr->getLeaves();
        built_generation = panzer_db_ptr->leaf_cache_generation();

        // Pre-reserve to avoid reallocations
        path_cache.reserve(leaves.size());
        name_index.reserve(leaves.size() / 16 + 16);

        for (const auto& leaf : leaves) {
            // Store the pointer to the leaf in the map
            path_cache[leaf.path].push_back(&leaf);

            // Also index by the last path segment (dataset name). `leaf.path` has no
            // '/' in its last segment's key because dataset names use '&' (see
            // clean_ds_name), so "last segment == name" is the exact fallback predicate.
            const std::string_view p = leaf.path;
            const auto slash = p.rfind('/');
            const std::string_view key = (slash == std::string_view::npos) ? p : p.substr(slash + 1);
            name_index[key].push_back(&leaf);
        }

        // Time-ordered buckets: reading one AoS element of a dynamic signal used
        // to linearly scan every slice of that signal (8k slices -> 8k candidates
        // per element read = O(slices^2) for a full get()). Sorting the buckets
        // lets the time filter binary-search instead (to_improve.md point 13).
        const auto by_time = [](const PanzerDB::Leaf* a, const PanzerDB::Leaf* b) {
            return a->time_index < b->time_index;
        };
        for (auto& [path, bucket] : path_cache) std::stable_sort(bucket.begin(), bucket.end(), by_time);
        for (auto& [name, bucket] : name_index) std::stable_sort(bucket.begin(), bucket.end(), by_time);

        DEBUG_PRINT("Path index built with " << path_cache.size() << " unique paths, "
                  << name_index.size() << " dataset names.");
    }

    /**
     * @brief True when no leaf-cache rebuild happened since this index was built.
     */
    bool is_current() const {
        return panzer_db_ptr && built_generation == panzer_db_ptr->leaf_cache_generation();
    }

    /**
     * @brief Rebuilds the maps if (and only if) the engine's leaf cache was rebuilt.
     *
     * Call it at OPERATION ENTRY only, never in the middle of one: the Leaf*
     * gathered during an operation (leaf lists, sorted time slices) must stay
     * usable until that operation ends. Cost when nothing changed: one uint64
     * compare; cost when the index really moved: one rebuild, same as the ctor.
     * @return true if the index was rebuilt (derived caches must be dropped).
     */
    bool ensure_current() {
        if (is_current()) return false;
        build_indexes();
        return true;
    }

    /**
     * @brief Leaves stored under exactly this path, or nullptr when absent.
     */
    const std::vector<const PanzerDB::Leaf*>* find_path(std::string_view path) const {
        const auto it = path_cache.find(path);
        return (it == path_cache.end()) ? nullptr : &it->second;
    }

    /**
     * @brief Range of the leaves of `bucket` holding a given time_index.
     * Buckets are time-ordered (build_indexes), so this is a binary search
     * instead of a linear scan over every slice of the signal.
     */
    using TimeRange = std::pair<std::vector<const PanzerDB::Leaf*>::const_iterator,
                                std::vector<const PanzerDB::Leaf*>::const_iterator>;
    static TimeRange equal_time(const std::vector<const PanzerDB::Leaf*>& bucket, uint64_t time_index) {
        const auto lo = std::lower_bound(bucket.begin(), bucket.end(), time_index,
                                         [](const PanzerDB::Leaf* l, uint64_t t) { return l->time_index < t; });
        const auto hi = std::upper_bound(lo, bucket.end(), time_index,
                                         [](uint64_t t, const PanzerDB::Leaf* l) { return t < l->time_index; });
        return {lo, hi};
    }

    /**
     * @brief Leaves whose last path segment is this dataset name, or nullptr.
     */
    const std::vector<const PanzerDB::Leaf*>* find_name(std::string_view name) const {
        const auto it = name_index.find(name);
        return (it == name_index.end()) ? nullptr : &it->second;
    }
};

/**
 * @class IReadStrategy
 * @brief Abstract base class defining the strategy for reading data from HDF5 via PanzerDB.
 * 
 * This class provides the common infrastructure for different reading strategies
 * (Global, Slice, TimeRange) used by the HDF5 backend.
 *
 * @note The PanzerDB engine and the path indexes are NOT owned here: one shared
 *       pair per read session is injected by HDF5Reader_v2 (see ReadIndex),
 *       so a session loads the /index once whatever the range modes used.
 */
class IReadStrategy {

protected:
     // =================================================================================
    //                                  Members
    // =================================================================================

     /**
     * @brief Cache for context paths to avoid rebuilding the parent path on every read call.
     * Key: Pointer to context.
     * Value: Pre-calculated path string.
     */
    mutable std::unordered_map<Context*, std::string> context_path_cache;

     /**
     * @brief Pointer to the PanzerDB instance handling low-level HDF5 operations
     *        (shared with the other strategies of the read session).
     */
    std::shared_ptr<PanzerDB> panzer_db_ptr;

    /**
     * @brief Shared engine-wide index (path/name -> leaves), shared with the
     *        other strategies of the read session.
     */
    std::shared_ptr<ReadIndex> read_index_ptr;

    std::unordered_map<std::string, std::vector<double>> time_values_cache;

    // Cached ids_properties&homogeneous_time flag for the read session (see getHomogeneousTime())
    mutable int homogeneous_time_cache = HOMOG_TIME_UNRESOLVED;
    
    // Cache for path sanitization (Context + Path -> Sanitized Path)
    mutable std::map<std::pair<Context*, std::string>, std::string> sanitized_path_cache;

public:
    // =================================================================================
    //                            Constructor / Destructor
    // =================================================================================

    /**
     * @brief Constructor. Adopts the session's read-only engine and its shared index.
     * @param panzer_db   The shared PanzerDB opened in READ mode for the session.
     * @param read_index  The shared ReadIndex built once over that engine.
     */
    IReadStrategy(std::shared_ptr<PanzerDB> panzer_db, std::shared_ptr<ReadIndex> read_index)
        : panzer_db_ptr(std::move(panzer_db)), read_index_ptr(std::move(read_index)) {}

    virtual ~IReadStrategy() = default;

    /**
     * @brief Re-synchronises this strategy with its engine index (SWMR / index move).
     *
     * Must be called at the ENTRY of an operation (read_ND_Data,
     * beginReadArraystructAction, getTimeValues, ...), never in the middle: the
     * Leaf* gathered during an operation then stay usable until its end. If the
     * engine rebuilt its leaf cache in the meantime, the shared ReadIndex is
     * rebuilt here and every value derived from the old leaves is dropped; cost
     * when nothing moved is one uint64 compare (to_improve.md point 6).
     */
    void refresh_index_if_needed() {
        if (!read_index_ptr || !read_index_ptr->ensure_current()) return;
        time_values_cache.clear();
        homogeneous_time_cache = HOMOG_TIME_UNRESOLVED;
    }
    
    // =================================================================================
    //                            Virtual Interface
    // =================================================================================

    /**
     * @brief Prepares reading an Array of Structures (AoS).
     * @param ctx The array structure context.
     * @param size Output pointer to store the size of the array.
     */
    virtual void beginReadArraystructAction(ArraystructContext * ctx, int *size) = 0;

     /**
     * @brief Finalizes an action on a context.
     * @param ctx The context to close/finalize.
     */
    virtual void endAction(Context * ctx) = 0;
    

     /**
     * @brief Reads N-Dimensional data.
     * @param ctx Current context.
     * @param dataset_name Name of the dataset.
     * @param timebasename Name of the timebase (if dynamic).
     * @param datatype Output pointer for data type.
     * @param data Output pointer for data buffer.
     * @param dim Output pointer for number of dimensions.
     * @param size Output pointer for dimensions sizes.
     * @return 0 on failure, 1 on success.
     */
    virtual int read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                             int* datatype, void **data, int *dim, int *size) = 0;

     // =================================================================================
    //                            Time Management
    // =================================================================================

    /**
     * @brief Retrieves the homogeneous time status from ids_properties.
     * @return 1 if homogeneous, 0 otherwise (or -1 on error/default).
     * @note Cached for the whole read session: the flag cannot change while
     *        the engine is open in READ mode, and this getter was called on
     *        every read_ND_Data / beginReadArraystructAction.
     */
    int getHomogeneousTime() {
        if (homogeneous_time_cache != HOMOG_TIME_UNRESOLVED) return homogeneous_time_cache;
        if (!panzer_db_ptr) return -1;
        int homogeneous_time = 1;
        int status = -1;
        int temp = panzer_db_ptr->readScalar<int32_t>("ids_properties&homogeneous_time", &status);
        if (status == 0)
           homogeneous_time = temp;
        homogeneous_time_cache = homogeneous_time;
        return homogeneous_time;
    }

     /**
     * @brief Retrieves time values associated with a context.
     * @param ctx The context.
     * @param homogeneous_time Flag indicating if time is homogeneous.
     * @return Vector of time values.
     */
    std::vector<double> getTimeValues(Context *ctx, int homogeneous_time, const std::string& timebasename = "") {
    refresh_index_if_needed();
    std::vector<double> time_values;

    std::string timebasename_copy = timebasename;
    if (!timebasename_copy.empty() && timebasename_copy[0] == '/') {
        timebasename_copy.erase(0, 1); // Remove 1 character at index 0
    }
    // Use smart sanitization
    timebasename_copy = sanitize_path(ctx, timebasename_copy);

    if (homogeneous_time == 1) {
        if (time_values_cache.count("HOMOGENEOUS_TIME")) {
            return time_values_cache["HOMOGENEOUS_TIME"];
        }
        time_values = panzer_db_ptr->getWholeDynamicSignal("time");
        time_values_cache["HOMOGENEOUS_TIME"] = time_values;
        return time_values;
    }

    // --- Logic for homogeneous_time == 0 ---
    if (ctx == nullptr) {
        return time_values; // Return empty vector
    }

    ArraystructContext *arrCtx = dynamic_cast<ArraystructContext*>(ctx);
    if (!arrCtx) { // Case where context is OperationContext
        time_values = panzer_db_ptr->getWholeDynamicSignal("time"); // Fallback for root time
        return time_values;
    }

    // Find the "timed" parent to build the timebase path
    ArraystructContext* timed_ctx = arrCtx;
    while(timed_ctx != nullptr && !timed_ctx->getTimed()) {
        timed_ctx = timed_ctx->getParent();
    }

    // If no dynamic parent is found, it could be a dynamic signal inside a static AoS.
    // In this case, the time vector is also static relative to the current context.
    if (!timed_ctx) {
        if (!timebasename_copy.empty()) {
            // For homogeneous_time=0, the timebase can be relative to the static AoS element,
            // or fall back to a common timebase at the root of the IDS.

            // 1. Try to find timebase relative to the current static AoS element.
            std::replace(timebasename_copy.begin(), timebasename_copy.end(), '/', '&');
            std::string local_timebase_path = buildFullPath(ctx, timebasename_copy);
            
            time_values = panzer_db_ptr->getWholeDynamicSignal(local_timebase_path);
            

            // 2. If not found locally, fall back to the root timebase.
            if (time_values.empty()) {
                time_values = panzer_db_ptr->getWholeDynamicSignal(timebasename_copy);
            }
            return time_values;
        }
        return {}; // No timebase found
    }

    // Filter to keep ONLY leaves corresponding to the exact path
    std::map<uint64_t, const PanzerDB::Leaf*> time_leaves_map;
    std::string timed_aos_path = getPath(timed_ctx, false);
    std::string timebase_name_str = timed_ctx->getTimebasePath();
    // Use smart sanitization
    timebase_name_str = sanitize_path(timed_ctx, timebase_name_str);

    // Extract basename of timebase to handle both relative ("time") and absolute/generic ("path/to/time") paths
    std::string timebase_basename = timebase_name_str;
    size_t last_slash_tb = timebase_name_str.find_last_of('/');
    if (last_slash_tb != std::string::npos) {
        timebase_basename = timebase_name_str.substr(last_slash_tb + 1);
    }

    std::string cache_key = timed_aos_path + "/" + timebase_basename;
    if (time_values_cache.count(cache_key)) {
        return time_values_cache[cache_key];
    }

    const auto& leaves = panzer_db_ptr->getLeaves();

    // HYBRID STRATEGY:
    // 1. Attempt direct access (Optimization if the timebase is stored in a single block under the AoS)
    //    We look for "AoS_Path/time".
    std::string direct_tb_path = timed_aos_path + "/" + timebase_basename;
    const auto* direct_leaves = read_index_ptr->find_path(direct_tb_path);

    if (direct_leaves && !direct_leaves->empty()) {
        // Homogeneous case found in cache!
        for (const auto* leaf : *direct_leaves) {
            if (!leaf->is_empty) {
                time_leaves_map[leaf->time_index] = leaf;
            }
        }
    } else {
        // 2. Optimized Fallback: Iteration by index (instead of linear scan)
        // Use the known size of the AoS to generate probable paths.
        size_t aos_size = panzer_db_ptr->getDynamicAOSSize(timed_aos_path);
        
        if (aos_size > 0) {
            for (size_t i = 0; i < aos_size; ++i) {
                // Construct path: AoS/i/time
                // Note: timebase_name_str is already relative to the AoS (e.g., "time" or "nested/time")
                std::string slice_tb_path = timed_aos_path + "/" + std::to_string(i) + "/" + timebase_name_str;
                
                const auto* slice_leaves = read_index_ptr->find_path(slice_tb_path);
                if (slice_leaves) {
                    for (const auto* leaf : *slice_leaves) {
                        if (!leaf->is_empty) {
                            time_leaves_map[leaf->time_index] = leaf;
                        }
                    }
                }
            }
        } else {
            // 3. Last resort: Linear scan (if size is unknown, e.g., complex static case)
            for (const auto& leaf : leaves) {
                if (leaf.parent_path.rfind(timed_aos_path, 0) == 0) {
                    size_t last_slash = leaf.path.find_last_of('/');
                    if (last_slash != std::string::npos) {
                        std::string_view leaf_name = leaf.path.substr(last_slash + 1);
                        if (leaf_name == timebase_basename && !leaf.is_empty) {
                            time_leaves_map[leaf.time_index] = &leaf;
                        }
                    }
                }
            }
        }
    }

    // Read values in time_index order
    for (const auto& [time_idx, leaf_ptr] : time_leaves_map) {
        if (leaf_ptr->count > 0) {
            std::vector<double> temp_data(leaf_ptr->count);
            panzer_db_ptr->readTensor(*leaf_ptr, temp_data.data());
            time_values.insert(time_values.end(), temp_data.begin(), temp_data.end());
        }
    }
    time_values_cache[cache_key] = time_values;
    return time_values;
}

    // =================================================================================
    //                            Path & Indexing
    //                            (indexes live in the shared ReadIndex)
    // =================================================================================

    /**
     * @brief Finds a specific leaf in the PanzerDB index for a given context and dataset.
     * @param ctx The current context.
     * @param dataset_name The name of the dataset.
     * @param timebasename The timebase name (used to determine if dynamic).
     * @param homogeneous_time Homogeneous time flag.
     * @return Pointer to the Leaf, or nullptr if not found.
     */
     const PanzerDB::Leaf* find_leaf_for_context(Context* ctx, std::string_view dataset_name, std::string_view timebasename, int homogeneous_time) {
    DEBUG_PRINT("Searching for dataset '" << dataset_name << "' with timebasename='" << timebasename << "' and homogeneous_time=" << homogeneous_time);
    refresh_index_if_needed();


    // --- Path reconstruction WITH the FULL parent path ---
    std::vector<std::string> path_segments;
    std::vector<int> indices;
    collectAosChain(ctx, path_segments, indices);

    std::string clean_ds_name(dataset_name);
    std::replace(clean_ds_name.begin(), clean_ds_name.end(), '/', '&');

    // ✅ Build the FULL path with indices
    // OPTIMIZATION: Use string reserve instead of stringstream
    std::string strict_target_path;
    size_t estimated_len = clean_ds_name.size() + path_segments.size() * 10; 
    strict_target_path.reserve(estimated_len);

    for (size_t i = 0; i < path_segments.size(); ++i) {
        if (i > 0) strict_target_path += "/";
        strict_target_path += path_segments[i];
        strict_target_path += "/";
        strict_target_path += std::to_string(indices[i]);
    }
    
    // ✅ Calculate context_prefix (path without the final dataset)
    std::string context_prefix = strict_target_path;
    if (!path_segments.empty()) context_prefix += "/";
    
    // ✅ Add the dataset to get the full path
    if (!path_segments.empty()) strict_target_path += "/";
    strict_target_path += clean_ds_name;
    
    // Ex: "core_sources/source/0/species&neutral&state&vibrational_level"

    DEBUG_PRINT("Reconstructed strict_target_path: '" << strict_target_path << "'");
    DEBUG_PRINT("Context prefix: '" << context_prefix << "'");

    int64_t target_time = -1;
    if (!timebasename.empty()) {
        target_time = 0;
    }

    // --- OPTIMIZED PASS 1: Search via Hash Map (O(1)) ---
    if (const auto* candidates = read_index_ptr->find_path(strict_target_path)) {
        const ReadIndex::TimeRange scan = (target_time == -1)
            ? ReadIndex::TimeRange{candidates->begin(), candidates->end()}
            : ReadIndex::equal_time(*candidates, static_cast<uint64_t>(target_time));
        for (auto it = scan.first; it != scan.second; ++it) {
            const auto* leaf = *it;
            DEBUG_PRINT("  -> Candidate from cache: " << leaf->path << " (time_index: " << leaf->time_index << ")");
            {
                return leaf;
            }
        }
    }

    // --- PASS 2: Fallback (indexed by dataset name) ---
    // The old O(N) full scan of getLeaves() is EXACTLY equivalent to looking up
    // the last path segment (the dataset name) in name_index and re-applying the
    // same time / prefix filters, because clean_ds_name has no '/' (it has had
    // '/' replaced by '&'), so "last segment == name" == the old suffix predicate.
    DEBUG_PRINT("[WARN] Optimized search failed. Falling back to name_index for: " << clean_ds_name);
    if (const auto* by_name = read_index_ptr->find_name(clean_ds_name)) {
        for (const auto* leaf : *by_name) {
            if (target_time != -1 && leaf->time_index != static_cast<uint64_t>(target_time)) continue;

            // ✅ FIX: Enforce context prefix constraint
            if (!context_prefix.empty()) {
                if (leaf->path.size() < context_prefix.size() || leaf->path.compare(0, context_prefix.size(), context_prefix) != 0) {
                    continue;
                }
            }

            return leaf;
        }
    }

    DEBUG_PRINT("Leaf not found for path: '" << strict_target_path << "' and time_index: " << target_time);
    return nullptr;
}

   protected: // This method is `protected` to be accessible by derived classes

     // =================================================================================
    //                            Path Construction Helpers
    // =================================================================================

    /**
     * @brief Constructs the path string for an ArraystructContext.
     * @param ctx The array structure context.
     * @param include_self_index Whether to include the index of the current context in the path.
     * @param override_timed_index When set (>= 0), the index to use for timed (dynamic) levels.
     * @return The constructed path string.
     */
    std::string getPath(ArraystructContext *ctx, bool include_self_index = true, int64_t override_timed_index = -1) {
    if (!ctx) return "";

    std::vector<std::pair<std::string, int>> segments;
    {
        std::vector<std::string> node_names;
        std::vector<int> raw_indices;
        std::vector<bool> timed_flags;
        collectAosChain(ctx, node_names, raw_indices, &timed_flags);
        for (size_t i = 0; i < node_names.size(); ++i) {
            const int index_to_use = (override_timed_index != -1 && timed_flags[i])
                                     ? static_cast<int>(override_timed_index) : raw_indices[i];
            segments.push_back({node_names[i], index_to_use});
        }
    }

    std::stringstream path_stream;
    for (size_t i = 0; i < segments.size(); ++i) {
        path_stream << segments[i].first; // Append node name (e.g., "A", then "B")

        bool is_last_segment = (i == segments.size() - 1);

        if (!is_last_segment) {
            // For non-last segments, always add index and separator. e.g., "A/0/"
            path_stream << "/" << segments[i].second << "/";
        } else { // For the last segment
            if (include_self_index) {
                path_stream << "/" << segments[i].second;
            }
        }
    }

    std::string result = path_stream.str();

    return result;
}

    /**
     * @brief Builds the full hierarchical path to a dataset inside nested AoS.
     *        The path is formatted as "AOS1/index1/AOS2/index2/.../dataset".
     * 
     * @param ctx The current context, must be an ArraystructContext or one of its children.
     * @param dataset_name The name of the final dataset.
     * @return std::string The full path, e.g., "A/0/B/0/data".
     */
   std::string buildFullPath(Context* ctx, const std::string& dataset_name) {
    std::vector<std::string> node_names;
    std::vector<int> raw_indices;
    collectAosChain(ctx, node_names, raw_indices);
    std::vector<std::pair<std::string, int>> segments;
    for (size_t i = 0; i < node_names.size(); ++i)
        segments.push_back({node_names[i], raw_indices[i]});

       std::stringstream path_stream;
       for (const auto& segment : segments) {
           path_stream << segment.first << "/" << segment.second << "/";
       }
       
       path_stream << dataset_name;
       
       return path_stream.str();
   }

 public:

    /**
     * @brief Normalizes a (potentially partial) path relative to `ctx` so it can be
     *        matched against the leaf paths of the PanzerDB index.
     *
     * Handles:
     *  - leading '/' (stripped);
     *  - '/' -> '&' flattening of the matched AoS block and of what hangs below it;
     *  - the context path is matched as a HEAD anchor (AL paths are cumulative
     *    from the IDS root), so a repeated segment name higher up the hierarchy
     *    cannot steal the match any more (to_improve.md point 5);
     *  - caching keyed on (ctx, input path) so repeated reads are O(1).
     * @param ctx  The context against which the path is to be interpreted.
     * @param path The raw path (may be empty, "/", "time", "A/B", etc.).
     * @return The normalized path ("" for empty input).
     */
      std::string sanitize_path(Context* ctx, const std::string& path) {
         // 1. Check the cache
         auto cache_key = std::make_pair(ctx, path);
         auto cached = sanitized_path_cache.find(cache_key);
         if (cached != sanitized_path_cache.end()) {
             return cached->second;
         }

         // 2. Collect the open AoS chain paths (deepest context first)
         std::vector<std::string> ctx_paths;
         Context* current = ctx;
         while (current != nullptr && current->getType() == CTX_ARRAYSTRUCT_TYPE) {
             ctx_paths.push_back(static_cast<ArraystructContext*>(current)->getPath());
             current = static_cast<ArraystructContext*>(current)->getParent();
         }

         // 3. Pure string normalisation (shared with the tests, aos_path_helpers.h)
         std::string result = sanitizeAosPath(ctx_paths, path);

         sanitized_path_cache[cache_key] = result;
         return result;
    }

   protected: // This method is `protected` to be accessible by derived classes

     // =================================================================================
    //                            Context Utilities
    // =================================================================================

    /**
     * @brief Checks if a context or any of its parents is timed.
     * @param ctx The context to check.
     * @return True if timed, false otherwise.
     */
    ArraystructContext* nearestTimedContext(Context *ctx) {
        if (!ctx || ctx->getType() != CTX_ARRAYSTRUCT_TYPE) {
            return nullptr;
        }

        ArraystructContext* arr_ctx = static_cast<ArraystructContext*>(ctx);
        while (arr_ctx != nullptr) {
            if (arr_ctx->getTimed()) {
                return arr_ctx;
            }
            arr_ctx = arr_ctx->getParent();
        }
        return nullptr;
    }

    bool isTimedContext(Context *ctx) {
        return nearestTimedContext(ctx) != nullptr;
    }

    // =================================================================================
    //                  Explicit read-side node-role helpers (to_improve.md #3)
    // =================================================================================

    /**
     * @brief Strips leading '/' and flattens the remaining separators to '&'.
     *
     * This mirrors the dataset-name normalisation done before the read strategies
     * are called, but can also be applied safely to timebase paths coming from
     * the AL context.
     */
    static std::string cleanFlatPath(std::string_view path) {
        std::string clean(path);
        if (!clean.empty() && clean.front() == '/') clean.erase(0, 1);
        std::replace(clean.begin(), clean.end(), '/', '&');
        return clean;
    }

    static std::string lastFlatSegment(std::string_view path) {
        const auto sep = path.find_last_of("/&");
        return (sep == std::string_view::npos)
                   ? std::string(path)
                   : std::string(path.substr(sep + 1));
    }

    /**
     * @brief Schema-level candidate paths for the timebase associated with a read.
     *
     * Two sources are considered explicitly, in contrast to the old suffix rule:
     *  - the `timebasename` argument supplied by the AL for this read;
     *  - the nearest timed AoS context's declared `getTimebasePath()`.
     *
     * The returned paths are stripped of AoS instance indices so that an instance
     * path (`A/0/time`) and the generic storage path (`A/time`) compare equal.
     */
    std::vector<std::string> collectTimebaseSchemaPaths(Context* ctx,
                                                        std::string_view timebasename) {
        std::vector<std::string> out;

        const auto add_schema = [&](std::string path) {
            path = PanzerDB::stripIndices(path);
            if (!path.empty()) out.push_back(std::move(path));
        };

        if (!timebasename.empty()) {
            std::string tb = cleanFlatPath(timebasename);
            if (!tb.empty()) {
                add_schema(buildFullPath(ctx, tb));
                add_schema(tb);
            }
        }

        if (ArraystructContext* timed = nearestTimedContext(ctx)) {
            std::string tb = timed->getTimebasePath();
            if (tb.empty()) tb = "time";
            tb = cleanFlatPath(sanitize_path(timed, tb));

            std::string generic = getPath(timed, false);
            if (!generic.empty()) generic += "/";
            generic += tb;
            add_schema(generic);
            add_schema(buildFullPath(timed, tb));
        }

        return out;
    }

    /**
     * @brief True when `dataset_name` denotes the timebase itself.
     *
     * The old rule was `ends_with(dataset_name, "time")`, which swallowed fields
     * such as `time_step` or `lifetime`. This version first uses the explicit
     * timebase path from the AL call/context and only falls back to an exact
     * basename "time" for legacy paths that do not expose a timebase path.
     */
    bool isTimebaseDataset(Context* ctx,
                           std::string_view dataset_name,
                           std::string_view timebasename,
                           int homogeneous_time) {
        const std::string clean_ds = cleanFlatPath(dataset_name);
        if (homogeneous_time == 1 && clean_ds == "time") return true;

        const std::string ds_schema = PanzerDB::stripIndices(buildFullPath(ctx, clean_ds));
        for (const auto& tb_schema : collectTimebaseSchemaPaths(ctx, timebasename)) {
            if (tb_schema == ds_schema) return true;
        }

        // Conservative legacy fallback: the dataset itself is a node whose final
        // storage segment is exactly "time". It no longer matches any name merely
        // ending with "time".
        return lastFlatSegment(clean_ds) == "time";
    }

    /**
     * @brief Explicit rule for slice promotion of a time-dependent scalar.
     *
     * Root/static-AoS scalar signals get a slice dimension of size 1; scalar
     * signals belonging to a dynamic AoS are already positioned by the dynamic
     * slice and remain scalars.
     */
    bool shouldPromoteTimeScalarOnSlice(Context* ctx,
                                        std::string_view timebasename,
                                        int datatype) {
        const bool inside_dynamic_aos = isTimedContext(ctx);
        const bool time_dependent = inside_dynamic_aos || !timebasename.empty();
        return time_dependent && !inside_dynamic_aos && datatype != alconst::char_data;
    }

    
   protected: // This method is `protected` to be accessible by derived classes

    // =================================================================================
    //                            Data Reading Helpers
    // =================================================================================

    /**
     * @brief Reads data from a specific PanzerDB leaf.
     * @param leaf Pointer to the leaf to read.
     * @param datatype Expected data type.
     * @param data Output pointer for the data buffer.
     * @param dim Output pointer for dimensions count.
     * @param size Output pointer for dimensions sizes.
     * @return 1 on success, 0 on failure.
     */
    int readLeafData(const PanzerDB::Leaf* leaf, int datatype, void **data, int *dim, int *size) {
        if (!panzer_db_ptr) {
            throw ALBackendException("PanzerDB not initialized", LOG);
        }
        if (!leaf) {
            return 0;
        }

        try {
            // RAII buffers: on any throw (readTensor, HDF5 errors) the block is
            // freed automatically; on success ownership is handed to the caller
            // via release() (same malloc'd block, caller keeps free()-ing it).
            if (datatype == alconst::double_data) {
                auto buf = std::make_unique<double[]>(leaf->count);
                panzer_db_ptr->readTensor<double>(*leaf, buf.get());
                *data = static_cast<void*>(buf.release());
            } else if (datatype == alconst::integer_data) {
                auto buf = std::make_unique<int32_t[]>(leaf->count);
                panzer_db_ptr->readTensor<int32_t>(*leaf, buf.get());
                *data = static_cast<void*>(buf.release());
            } else if (datatype == alconst::complex_data) {
                auto buf = std::make_unique<std::complex<double>[]>(leaf->count);
                panzer_db_ptr->readTensor<std::complex<double>>(*leaf, buf.get());
                *data = static_cast<void*>(buf.release());
            } else if (datatype == alconst::char_data) {
                if (leaf->shape_span().size() <= 1) { // 1D list or Scalar
                    std::vector<std::string> str_list(leaf->count);
                    panzer_db_ptr->readTensor<std::string>(*leaf, str_list.data());

                    // Array of char*: the array block is RAII-owned, the elements
                    // keep the historical per-string malloc (AL frees them one by one).
                    auto out_arr = std::make_unique<char*[]>(leaf->count);
                    char** out_ptr = out_arr.get();
                    for (size_t i = 0; i < leaf->count; ++i) {
                         size_t len = str_list[i].size() + 1;
                         out_ptr[i] = (char*)malloc(len);
                         std::memcpy(out_ptr[i], str_list[i].c_str(), len);
                    }
                    *data = static_cast<void*>(out_arr.release());
                } else {
                    throw ALBackendException("HDF5Reader_v2: Multidimensional strings not supported", LOG);
                }
            } else {
                throw ALBackendException("HDF5Reader_v2: Unknown datatype", LOG);
            }

            // Update output dimensions
            *dim = leaf->shape_span().size();
            for (size_t i = 0; i < leaf->shape_span().size(); ++i) {
                size[i] = leaf->shape[i];
            }

            return 1; // Success
        } catch (const std::exception& e) {
            throw ALBackendException(std::string("PanzerDB read error: ") + e.what(), LOG);
        }
    }

     /**
     * @brief Reads a whole dataset globally (concatenating all time slices).
     *        Refactored from GlobalReadStrategy to be used by TimeRangeReadStrategy.
     * @param ctx The context.
     * @param dataset_name The dataset name.
     * @param datatype Output pointer for data type.
     * @param data Output pointer for data buffer.
     * @param dim Output pointer for dimensions count.
     * @param size Output pointer for dimensions sizes.
     * @param timebasename The timebase argument supplied by the AL for this read
     *        (empty for purely static nodes); drives the rank-0 scalar
     *        promotion rule (see the numeric section below).
     * @return 1 on success, 0 on failure.
     */
    int read_dataset_globally(Context *ctx, std::string &dataset_name, int* datatype, void **data, int *dim, int *size,
                              std::string_view timebasename = "") {
        DEBUG_PRINT("--> Entering read_dataset_globally for dataset: " << dataset_name);

        int type = ctx->getType();

        if (!panzer_db_ptr) {
            throw ALBackendException("PanzerDB not initialized", LOG);
        }
        refresh_index_if_needed();

        // `*dim` arrives holding the caller's EXPECTED dimension (the AL layer
        // initialises retDim to the dictionary dim before calling the backend).
        // Capture it before any branch below overwrites it: it is the only
        // context-independent way to tell a 1-point scalar time series apart
        // from a genuine 0D scalar on disk (both are stored as shape=[],
        // count=1 — indistinguishable to the engine alone).
        const int expected_dim = dim ? *dim : 0;

        // ✅ OPTIMIZATION: Use the context path cache
        std::string context_prefix;
        auto cache_it = context_path_cache.find(ctx);
        if (cache_it != context_path_cache.end()) {
            context_prefix = cache_it->second;
        } else {
            // Path is not in cache, build it and store it
            std::stringstream ss_prefix;
            // ... (path construction logic remains here)
            // After construction, store in cache:
            // context_path_cache[ctx] = ss_prefix.str();
            // For now, we integrate the logic directly below.
        }

        // 1. Path reconstruction
        std::vector<std::string> path_segments;
        std::vector<int> indices;
        std::vector<bool> is_dynamic_level;
        collectAosChain(ctx, path_segments, indices, &is_dynamic_level);

        int64_t target_time_index = -1;
        for (size_t i = 0; i < path_segments.size(); ++i) {
            if (is_dynamic_level[i]) {
                target_time_index = indices[i];
                break;
            }
        }

        std::string clean_ds_name = dataset_name;
        std::replace(clean_ds_name.begin(), clean_ds_name.end(), '/', '&');

        // 2. Build the full path and retrieve the leaves
        // (Integration of cache logic here)
        std::stringstream ss_specific;
        for (size_t i = 0; i < path_segments.size(); ++i) {
            ss_specific << path_segments[i] << "/" << indices[i] << "/";
        }
        context_prefix = ss_specific.str();
        context_path_cache[ctx] = context_prefix; // Caching

        ss_specific << clean_ds_name;
        std::string specific_path = ss_specific.str();

        // --- METADATA HANDLING ---
        // Try to read metadata for this path (once per schema)
        std::map<std::string, std::string> meta = panzer_db_ptr->readMetadata(specific_path);
        if (!meta.empty()) {
            DEBUG_PRINT("Loaded " << meta.size() << " metadata entries for " << specific_path);
        }

        std::vector<const PanzerDB::Leaf*> sorted_leaves;
        bool found = false;

        const auto* specific_leaves = read_index_ptr->find_path(specific_path);
        if (specific_leaves && !specific_leaves->empty()) {
            DEBUG_PRINT("Found specific path: " << specific_path);
            // FIX: Always filter by time index if in a dynamic context,
            // even for a specific path. This handles cases where multiple time
            // slices might be associated with the same path string in the cache.
            if (target_time_index != -1) {
                const auto range = ReadIndex::equal_time(*specific_leaves, static_cast<uint64_t>(target_time_index));
                sorted_leaves.assign(range.first, range.second);
            } else {
                sorted_leaves = *specific_leaves;
            }
            found = !sorted_leaves.empty();
        }
        else {
            // Strategy B: Try Generic Path (skip index for dynamic levels)
            // This handles dynamic signals directly under dynamic AoS (e.g. profiles_1d/signal_1d)
            std::stringstream ss_generic;
            for (size_t i = 0; i < path_segments.size(); ++i) {
                ss_generic << path_segments[i];
                if (!is_dynamic_level[i]) {
                    ss_generic << "/" << indices[i];
                }
                ss_generic << "/";
            }
            ss_generic << clean_ds_name;
            std::string generic_path = ss_generic.str();
            
            const auto* generic_leaves = read_index_ptr->find_path(generic_path);
            if (generic_leaves && !generic_leaves->empty()) {
                DEBUG_PRINT("Found generic path: " << generic_path);
                // Filter by time index if we are in a dynamic context
                if (target_time_index != -1) {
                    const auto range = ReadIndex::equal_time(*generic_leaves, static_cast<uint64_t>(target_time_index));
                    sorted_leaves.assign(range.first, range.second);
                } else {
                    sorted_leaves = *generic_leaves;
                }
                found = true;
            }
        }

        if (!found || sorted_leaves.empty()) {
            // Fallback: dynamic signals read from a parent (e.g. read "profiles_1d/signal"
            // from the root). The old O(N) full scan of getLeaves() is EXACTLY equivalent
            // to looking up the last path segment (the dataset name) in name_index and
            // re-applying the same filters: "leaf.path == name OR (ends with name preceded
            // by '/')" == "last segment == name", because clean_ds_name has no '/' (it has
            // had '/' replaced by '&'). Same candidate set, O(1) instead of O(N).
            if (const auto* by_name = read_index_ptr->find_name(clean_ds_name)) {
                // Same candidate set as the old full scan, but the time filter is a
                // binary search on the time-ordered bucket when a time is requested.
                const ReadIndex::TimeRange range = (target_time_index == -1)
                    ? ReadIndex::TimeRange{by_name->begin(), by_name->end()}
                    : ReadIndex::equal_time(*by_name, static_cast<uint64_t>(target_time_index));
                for (auto it = range.first; it != range.second; ++it) {
                    const auto* leaf = *it;

                    // Check if path starts with context prefix (if any)
                    if (!context_prefix.empty()) {
                        if (leaf->path.compare(0, context_prefix.size(), context_prefix) == 0) {
                            sorted_leaves.push_back(leaf);
                        }
                    } else {
                        sorted_leaves.push_back(leaf);
                    }
                }
            }

            if (sorted_leaves.empty()) {
                DEBUG_PRINT("Leaf not found for: " << specific_path << " or generic variant (after fallback)");
                return 0;
            }
        }
        else {
            DEBUG_PRINT("Number of candidate leaves found: " << sorted_leaves.size());
        }

        std::sort(sorted_leaves.begin(), sorted_leaves.end(), 
             [](const PanzerDB::Leaf* a, const PanzerDB::Leaf* b) {
                return a->time_index < b->time_index;
            });

        size_t total_elements = 0;
        for (const auto* leaf : sorted_leaves) {
            if (static_cast<uint64_t>(leaf->flags >> 4) ==
                static_cast<uint64_t>(PanzerDB::DataType::STRING_CHUNKED)) {
                total_elements += 1;  // one logical scalar (its `count` slots are its chunks)
            } else {
                total_elements += leaf->count;
            }
        }

        DEBUG_PRINT("Total elements to read (after filtering by time index): " << total_elements);

        if (total_elements == 0) {
            *data = nullptr;
            *dim = 0;
            // size can be left uninitialized as dim is 0
            return 1; // Success, dataset is empty.
        }

        const PanzerDB::Leaf* first_leaf = sorted_leaves[0];
        
        // FIX: Handle multiple static writes for scalars (overwrites).
        // A scalar leaf is one with count=1 (for strings) or an empty shape (for numerics).
        PanzerDB::DataType first_leaf_type = static_cast<PanzerDB::DataType>(first_leaf->flags >> 4);
        bool is_numeric_scalar_leaf = (first_leaf_type != PanzerDB::DataType::STRING && first_leaf->shape_span().empty());
        bool is_string_scalar_leaf =
            (first_leaf_type == PanzerDB::DataType::STRING && first_leaf->count == 1) ||
            (first_leaf_type == PanzerDB::DataType::STRING_CHUNKED); // multi-slot scalar

        if ((is_numeric_scalar_leaf || is_string_scalar_leaf) && total_elements > 1) {
             bool all_static = true;
             for (const auto* leaf : sorted_leaves) {
                 if (leaf->time_index != 0) { all_static = false; break; }
             }
             if (all_static) {
                 const PanzerDB::Leaf* last_leaf = sorted_leaves.back();
                 sorted_leaves.clear();
                 sorted_leaves.push_back(last_leaf);
                 total_elements = last_leaf->count;
                 first_leaf = last_leaf;
             }
        }

        // Determine the actual type from the leaf flags
        PanzerDB::DataType actual_type_enum = static_cast<PanzerDB::DataType>(first_leaf->flags >> 4);
        int actual_datatype = 0;
        switch(actual_type_enum) {
            case PanzerDB::DataType::FLOAT64: actual_datatype = alconst::double_data; break;
            case PanzerDB::DataType::INT32: actual_datatype = alconst::integer_data; break;
            case PanzerDB::DataType::STRING: actual_datatype = alconst::char_data; break;
            case PanzerDB::DataType::STRING_CHUNKED: actual_datatype = alconst::char_data; break;
            case PanzerDB::DataType::COMPLEX128: actual_datatype = alconst::complex_data; break;
            default:
                DEBUG_PRINT("Unknown data type in leaf flags: " << (first_leaf->flags >> 4));
                return 0; // Unknown type
        }

        size_t leaf_rank = first_leaf->shape_span().size();

        // 3. CHAR / STRING Processing
        // Always read with the actual file type. Conversion will be done by al_lowlevel.
        DEBUG_PRINT("Actual data type determined from leaf flags: " << actual_datatype);
        if (actual_datatype == alconst::char_data) {
             DEBUG_PRINT("Processing string data...");
             auto is_chunked = [](const PanzerDB::Leaf* leaf) {
                 return static_cast<uint64_t>(leaf->flags >> 4) ==
                        static_cast<uint64_t>(PanzerDB::DataType::STRING_CHUNKED);
             };
             // BULK READER: fetch the whole union range of slots in ONE H5Dread
             // (instead of one tiny H5Dread per leaf — that was O(N) on 10^5-slice
             // files and the "reading never returns" hang). Then scatter locally:
             //   - a STRING_CHUNKED scalar : its `count` slots joined into one string
             //   - a plain (string) leaf   : its `count` slots, one per element
             uint64_t lo = UINT64_MAX, hi = 0;
             for (const auto* leaf : sorted_leaves) {
                 lo = std::min(lo, leaf->offset);
                 hi = std::max(hi, leaf->offset + leaf->count);
             }
             auto slots = panzer_db_ptr->readStringBulk(lo, hi);  // one H5Dread

             std::vector<std::string> temp_buffer;
             temp_buffer.reserve(total_elements);
             for (const auto* leaf : sorted_leaves) {
                 const uint64_t base = leaf->offset - lo;

                 // Spanned list (IMAS: element > 512B chunked across slots)
                 if (panzer_db_ptr->isListSpanned(leaf->row_id)) {
                     const auto& spans = panzer_db_ptr->getListSpans(leaf->row_id);
                     for (const auto& se : spans) {
                         std::string joined;
                         for (uint64_t k = 0; k < se.slot_span; ++k)
                             joined += slots[base + se.slot_offset + k];
                         temp_buffer.push_back(std::move(joined));
                     }
                 }
                 // STRING_CHUNKED scalar (concatenate leaf->count slots into one string)
                 else if (is_chunked(leaf)) {
                     std::string joined;
                     for (uint64_t i = 0; i < leaf->count; ++i) joined += slots[base + i];
                     temp_buffer.push_back(std::move(joined));
                 }
                 // Compact list (one slot per element)
                 else {
                     for (uint64_t i = 0; i < leaf->count; ++i) temp_buffer.push_back(std::move(slots[base + i]));
                 }
             }
 
             size_t max_str_len = 0;
             for (const auto& s : temp_buffer) if (s.size() > max_str_len) max_str_len = s.size();
             max_str_len += 1; // Null terminator

             // Scalar vs list contract (to_improve.md point 3): the leaf shape is
             // the authoritative shape role. A scalar string has an empty shape;
             // a list of strings has rank 1, even when it contains one element.
             const bool is_scalar_leaf = PanzerDB::isStringScalarLeaf(*first_leaf);
             DEBUG_PRINT("Is scalar leaf: " << is_scalar_leaf);

             bool return_as_scalar = (is_scalar_leaf && total_elements == 1);

            if (return_as_scalar) {
                DEBUG_PRINT("Detected scalar string leaf. Returning as single string.");
                 *dim = 1;
                 size[0] = (int)temp_buffer[0].size(); // Length excluding null
                 auto buf = std::make_unique<char[]>(size[0] + 1);
                 std::memcpy(buf.get(), temp_buffer[0].c_str(), size[0] + 1);
                 *data = static_cast<void*>(buf.release());
            } else {
                 // `temp_buffer.size()` = logical element count (correct for spanned lists).
                 size_t n_logical = (size_t)temp_buffer.size();
                 DEBUG_PRINT("Detected list of strings. Returning as 2D char array with max string length: " << max_str_len);
                 *dim = 2;
                 size[0] = (int)n_logical;
                 size[1] = (int)max_str_len;

                 size_t buffer_bytes = n_logical * max_str_len;
                 auto buf = std::make_unique<char[]>(buffer_bytes);
                 char* char_buffer = buf.get();
                 std::memset(char_buffer, 0, buffer_bytes);
                 for (size_t i = 0; i < n_logical; ++i) {
                     if (!temp_buffer[i].empty()) strncpy(char_buffer + (i * max_str_len), temp_buffer[i].c_str(), max_str_len);
                 }
                 *data = static_cast<void*>(buf.release());
             }
 
             *datatype = actual_datatype; // Return the type that was read
             return 1;
        }

        // 4. NUMERICAL Processing
        if (leaf_rank == 0) {
            // 0D signal (scalar) -> becomes 1D with the time dimension.
            // A 1-point scalar time series is stored identically to a genuine
            // 0D scalar (shape=[], count=1), so the engine cannot tell them
            // apart. Disambiguate with the caller's intent: promote that
            // single slice to 1D(1) when the node reads as time-dependent
            // (same rule as the slice path, shouldPromoteTimeScalarOnSlice)
            // OR the AL layer expects a 1D result (the dictionary says e.g.
            // code/output_flag is INT_1D over time). Without this the read
            // degrades to a 0D scalar and AL raised "expected int in 1D but
            // got int in 0D".
            const bool want_1d = (expected_dim == 1) ||
                shouldPromoteTimeScalarOnSlice(ctx, timebasename, actual_datatype);
            if (total_elements > 1) {
                *dim = 1;
                size[0] = (int)total_elements;
            } else if (total_elements == 1 && want_1d) {
                *dim = 1;
                size[0] = 1;
            } else {
                *dim = 0;
                // size[0] = 1; // Implicit for a scalar
            }
        } else {
            // N-D signal -> Add time dimension IF multiple slices
            size_t spatial_product = 1;
            for (size_t i = 0; i < leaf_rank; ++i) { 
                size[i] = (int)first_leaf->shape[i]; 
                spatial_product *= first_leaf->shape[i]; 
            }
            
            size_t n_time_slices = (spatial_product > 0) ? (total_elements / spatial_product) : 1;
            
            if (n_time_slices > 1) {
                *dim = (int)(leaf_rank + 1); // ✅ +1 for time dimension
                size[leaf_rank] = (int)n_time_slices;
            } else {
                *dim = (int)leaf_rank; // No time dimension if only 1 slice
            }
        }

        // Allocation (RAII: a throw inside readLeavesUnion used to leak the raw
        // buffer; ownership is only handed over on success via release()).
        std::unique_ptr<double[]> buf_d;
        std::unique_ptr<int32_t[]> buf_i;
        std::unique_ptr<std::complex<double>[]> buf_c;
        void* raw = nullptr;
        if (actual_datatype == alconst::double_data) { buf_d = std::make_unique<double[]>(total_elements); raw = buf_d.get(); }
        else if (actual_datatype == alconst::integer_data) { buf_i = std::make_unique<int32_t[]>(total_elements); raw = buf_i.get(); }
        else if (actual_datatype == alconst::complex_data) { buf_c = std::make_unique<std::complex<double>[]>(total_elements); raw = buf_c.get(); }
        else return 0;
        *data = raw;

        // OPTIMIZATION: Grouped read (Hyperslab Union)
        int res = panzer_db_ptr->readLeavesUnion(sorted_leaves, *data, actual_type_enum); 
        if (res < 0) {
            // buf_d/buf_i/buf_c freed by RAII here
            *data = nullptr;
            return 0;
        }

        if (buf_d) *data = buf_d.release();
        else if (buf_i) *data = buf_i.release();
        else if (buf_c) *data = buf_c.release();

        *datatype = actual_datatype; // Return the type that was read
        
        return 1;
    }
    
};

#endif // I_READ_STRATEGY_H
