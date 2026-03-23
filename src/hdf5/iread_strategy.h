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
#include <set>
#include <map>

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
 * @class IReadStrategy
 * @brief Abstract base class defining the strategy for reading data from HDF5 via PanzerDB.
 * 
 * This class provides the common infrastructure for different reading strategies
 * (Global, Slice, TimeRange) used by the HDF5 backend.
 */
class IReadStrategy {

protected:
     // =================================================================================
    //                                  Members
    // =================================================================================

    /**
     * @brief Cache for fast access path -> leaves.
     * Key: Full path (string view).
     * Value: Vector of pointers to leaves sharing this path (different time_index).
     */
    std::unordered_map<std::string_view, std::vector<const PanzerDB::Leaf*>> path_cache;

     /**
     * @brief Cache for context paths to avoid rebuilding the parent path on every read call.
     * Key: Pointer to context.
     * Value: Pre-calculated path string.
     */
    mutable std::unordered_map<Context*, std::string> context_path_cache;

     /**
     * @brief Pointer to the PanzerDB instance handling low-level HDF5 operations.
     */
    std::unique_ptr<PanzerDB> panzer_db_ptr;

    std::unordered_map<std::string, std::vector<double>> time_values_cache;
    
    // Cache for path sanitization (Context + Path -> Sanitized Path)
    mutable std::map<std::pair<Context*, std::string>, std::string> sanitized_path_cache;
    mutable std::set<std::string> schema_aos_paths; // Cache for "schema" AoS paths (without indices)

public:
    // =================================================================================
    //                            Constructor / Destructor
    // =================================================================================

    /**
     * @brief Constructor. Initializes PanzerDB in READ mode and builds the path index.
     * @param loc_id HDF5 location ID (group or file).
     */
    IReadStrategy(hid_t loc_id) {
        panzer_db_ptr = std::make_unique<PanzerDB>(loc_id, PanzerDB::OpenMode::READ);
        build_path_index(); // Build the index right after PanzerDB initialization
        build_aos_schema_index();
    } 

    virtual ~IReadStrategy() = default;
    
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
     */
    int getHomogeneousTime() {
        if (!panzer_db_ptr) return -1;
        int homogeneous_time = 1;
        int status = -1;
        int temp = panzer_db_ptr->readScalar<int32_t>("ids_properties&homogeneous_time", &status);
        if (status == 0)
           homogeneous_time = temp;
        return homogeneous_time;
    }

     /**
     * @brief Retrieves time values associated with a context.
     * @param ctx The context.
     * @param homogeneous_time Flag indicating if time is homogeneous.
     * @return Vector of time values.
     */
    std::vector<double> getTimeValues(Context *ctx, int homogeneous_time, const std::string& timebasename = "") {
    std::vector<double> time_values;
    //std::cout << "[DEBUG getTimeValues] homogeneous_time=" << homogeneous_time << " timebasename='" << timebasename << "'" << std::endl;

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
        //std::cout << "[DEBUG getTimeValues] getWholeDynamicSignal('time') returned size: " << time_values.size() << std::endl;
        time_values_cache["HOMOGENEOUS_TIME"] = time_values;
        return time_values;
    }

    // --- Logic for homogeneous_time == 0 ---
    //std::cout << "[DEBUG getTimeValues] Searching for time values in dynamic AoS context." << std::endl;
    if (ctx == nullptr) {
        //std::cout << "[DEBUG getTimeValues] ctx is nullptr" << std::endl;
        return time_values; // Return empty vector
    }

    ArraystructContext *arrCtx = dynamic_cast<ArraystructContext*>(ctx);
    if (!arrCtx) { // Case where context is OperationContext
        //std::cout << "[DEBUG getTimeValues] ctx is not ArraystructContext, trying fallback to root time" << std::endl;
        time_values = panzer_db_ptr->getWholeDynamicSignal("time"); // Fallback for root time
        //printf("[DEBUG getTimeValues] Fallback getWholeDynamicSignal('time') returned size: %zu\n", time_values.size());
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
        //std::cout << "[DEBUG getTimeValues] No timed parent context found." << std::endl;
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
        //printf("[DEBUG getTimeValues] No timed parent context and no static timebase found. Returning empty time vector.\n");
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
    //std::cout << "[DEBUG getTimeValues] Searching for timebase leaves with AoS path '" << timed_aos_path << "' and timebase basename '" << timebase_basename << "'" << std::endl;

    auto leaves = panzer_db_ptr->getLeaves();

    // HYBRID STRATEGY:
    // 1. Attempt direct access (Optimization if the timebase is stored in a single block under the AoS)
    //    We look for "AoS_Path/time".
    std::string direct_tb_path = timed_aos_path + "/" + timebase_basename;
    auto it_cache = path_cache.find(direct_tb_path);
    
    if (it_cache != path_cache.end() && !it_cache->second.empty()) {
        // Homogeneous case found in cache!
        for (const auto* leaf : it_cache->second) {
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
                
                auto it = path_cache.find(slice_tb_path);
                if (it != path_cache.end()) {
                    for (const auto* leaf : it->second) {
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
            //std::cout << "[DEBUG getTimeValues] Read " << temp_data.size() << " time values from leaf with time_index=" << time_idx << ". Total time values: " << time_values.size() << std::endl;
        }
    }
    //printf("[DEBUG getTimeValues] Final time values size: %zu\n", time_values.size());
    time_values_cache[cache_key] = time_values;
    return time_values;
}

    // =================================================================================
    //                            Path & Indexing
    // =================================================================================

    /**
     * @brief Builds an in-memory index of paths to leaves for fast lookup.
     */
    void build_path_index() {
        path_cache.clear();
        if (!panzer_db_ptr) return;

        const auto& leaves = panzer_db_ptr->getLeaves();
        
        // Pre-reserve to avoid reallocations
        path_cache.reserve(leaves.size());

        for (const auto& leaf : leaves) {
            // Store the pointer to the leaf in the map
            path_cache[leaf.path].push_back(&leaf);
        }
        
        DEBUG_PRINT("Path index built with " << path_cache.size() << " unique paths.");
    }

    void build_aos_schema_index() {
        schema_aos_paths.clear();
        if (!panzer_db_ptr) return;
        
        const auto& leaves = panzer_db_ptr->getLeaves();
        for (const auto& leaf : leaves) {
            if (leaf.flags != 2 && leaf.flags != 3) continue; // Keep only AoS (Static=2, Dynamic=3)
            std::string root(leaf.path);
            // Convert "A/0/B/0/C" -> "A/B/C"
            std::string schema_path;
            std::stringstream ss(root);
            std::string segment;
            while (std::getline(ss, segment, '/')) {
                // If the segment is numeric, ignore it (it's an index)
                if (segment.empty() || std::all_of(segment.begin(), segment.end(), ::isdigit)) {
                    continue;
                }
                if (!schema_path.empty()) schema_path += "/";
                schema_path += segment;
            }
            schema_aos_paths.insert(schema_path);
        }
    }

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

    // --- Path reconstruction WITH the FULL parent path ---
    std::vector<std::string> path_segments;
    std::vector<int> indices;
    Context* curr = ctx;
    
    // ✅ FIX: Go up to OperationContext to capture the full path
    while (curr != nullptr) {
        if (curr->getType() == CTX_ARRAYSTRUCT_TYPE) {
            ArraystructContext* arr = static_cast<ArraystructContext*>(curr);
            std::string full_path = arr->getPath();  // Ex: "core_sources/source"
            
            std::string node_name;
            Context* parent = arr->getParent();
            if (parent && parent->getType() == CTX_ARRAYSTRUCT_TYPE) {
                ArraystructContext* parent_arr = static_cast<ArraystructContext*>(parent);
                std::string parent_path = parent_arr->getPath();
                if (full_path.size() > parent_path.size() && full_path.rfind(parent_path + "/", 0) == 0) {
                    node_name = full_path.substr(parent_path.size() + 1);
                } else {
                    node_name = full_path;
                }
            } else {
                node_name = full_path;
            }
            std::replace(node_name.begin(), node_name.end(), '/', '&');
            
            path_segments.insert(path_segments.begin(), node_name);
            indices.insert(indices.begin(), arr->getIndex());
            curr = arr->getParent();
        }
        else if (curr->getType() == CTX_OPERATION_TYPE) {
            curr = nullptr;
        }
        else {
            curr = nullptr;
        }
    }

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
    auto it = path_cache.find(strict_target_path);
    if (it != path_cache.end()) {
        const std::vector<const PanzerDB::Leaf*>& candidates = it->second;
        for (const auto* leaf : candidates) {
            DEBUG_PRINT("  -> Candidate from cache: " << leaf->path << " (time_index: " << leaf->time_index << ")");
            if (target_time == -1 || leaf->time_index == static_cast<uint64_t>(target_time)) {
                return leaf;
            }
        }
    }

    // --- PASS 2: Fallback (Linear Search) ---
    DEBUG_PRINT("[WARN] Optimized search failed. Falling back to linear scan for: " << clean_ds_name);
    const auto& leaves = panzer_db_ptr->getLeaves();
    for (const auto& leaf : leaves) {
        if (target_time != -1 && leaf.time_index != static_cast<uint64_t>(target_time)) continue;
        
        // ✅ FIX: Enforce context prefix constraint
        if (!context_prefix.empty()) {
            if (leaf.path.size() < context_prefix.size() || leaf.path.compare(0, context_prefix.size(), context_prefix) != 0) {
                continue;
            }
        }

        if (leaf.path == clean_ds_name) return &leaf;
        if (leaf.path.size() > clean_ds_name.size() && leaf.path.compare(leaf.path.size() - clean_ds_name.size(), clean_ds_name.size(), clean_ds_name) == 0) {
            if (leaf.path[leaf.path.size() - clean_ds_name.size() - 1] == '/') return &leaf;
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
     * @return The constructed path string.
     */
    std::string getPath(ArraystructContext *ctx, bool include_self_index = true, int64_t override_timed_index = -1) {
    if (!ctx) return "";

    std::vector<std::pair<std::string, int>> segments;
    Context* current_ctx = ctx;

    while (current_ctx != nullptr && current_ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
        ArraystructContext* arr_ctx = static_cast<ArraystructContext*>(current_ctx);
        
        // 1. Get full path (e.g. "static_aos/dynamic_aos")
        std::string full_path = arr_ctx->getPath();
        std::string node_name;
        
        // 2. Extract local node name relative to parent
        Context* parent = arr_ctx->getParent();
        if (parent && parent->getType() == CTX_ARRAYSTRUCT_TYPE) {
            ArraystructContext* parent_arr = static_cast<ArraystructContext*>(parent);
            std::string parent_path = parent_arr->getPath();
            if (full_path.size() > parent_path.size() && full_path.rfind(parent_path + "/", 0) == 0) {
                node_name = full_path.substr(parent_path.size() + 1); // +1 for '/'
            } else {
                node_name = full_path; // Should not happen
            }
        } else {
            node_name = full_path;
        }
        
        std::replace(node_name.begin(), node_name.end(), '/', '&');
        
        int index_to_use = arr_ctx->getIndex();
        // If an override is provided AND the current context is dynamic, use it.
        if (override_timed_index != -1 && arr_ctx->getTimed()) {
            index_to_use = override_timed_index;
        }

        segments.insert(segments.begin(), {node_name, index_to_use});
        
        current_ctx = arr_ctx->getParent();
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
       std::vector<std::pair<std::string, int>> segments;
       Context* current_ctx = ctx;

       // Go up the context hierarchy to collect AoS names and indices.
       while (current_ctx != nullptr && current_ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
           ArraystructContext* arr_ctx = static_cast<ArraystructContext*>(current_ctx);
           
           std::string full_path = arr_ctx->getPath();
           std::string node_name;
           
           Context* parent = arr_ctx->getParent();
           if (parent && parent->getType() == CTX_ARRAYSTRUCT_TYPE) {
               ArraystructContext* parent_arr = static_cast<ArraystructContext*>(parent);
               std::string parent_path = parent_arr->getPath();
               if (full_path.size() > parent_path.size() && full_path.rfind(parent_path + "/", 0) == 0) {
                   node_name = full_path.substr(parent_path.size() + 1);
               } else {
                   node_name = full_path;
               }
           } else {
               node_name = full_path;
           }
           
           std::replace(node_name.begin(), node_name.end(), '/', '&');
           
           segments.insert(segments.begin(), {node_name, arr_ctx->getIndex()});
           
           current_ctx = arr_ctx->getParent();
       }

       std::stringstream path_stream;
       for (const auto& segment : segments) {
           path_stream << segment.first << "/" << segment.second << "/";
       }
       
       path_stream << dataset_name;
       
       return path_stream.str();
   }

    void replaceSlashWithAmpersand(std::string& s) {
        std::replace(s.begin(), s.end(), '/', '&');
    }

public:
    std::string sanitize_path(Context* ctx, const std::string& path) {
        // We build the full path to apply the logic
        std::string fullPath = path;

        if (fullPath.empty()) return "";
        if (fullPath == "/time") return "time";
        if (fullPath == "time") return fullPath;

        // 1. Check the cache
        auto cache_key = std::make_pair(ctx, path);
        if (sanitized_path_cache.count(cache_key)) {
            return sanitized_path_cache[cache_key];
        }

        std::vector<std::string> segments;
        std::string remaining = fullPath;
        
        // Remove the initial '/' if it exists to simplify splitting
        bool hasLeadingSlash = (fullPath[0] == '/');
        if (hasLeadingSlash) {
            remaining.erase(0, 1);
        }

        Context* current = ctx;

        // 1. Traverse contexts from deepest (F) to highest (A)
        while (current != nullptr) {
            std::string ctxPath;
            if (current->getType() == CTX_ARRAYSTRUCT_TYPE) {
                ctxPath = static_cast<ArraystructContext*>(current)->getPath();
            } else {
                // Stop if not ArraystructContext
                break;
            }
            
            if (!ctxPath.empty()) {
                size_t pos = remaining.rfind(ctxPath);
                
                if (pos != std::string::npos) {
                    // Check boundaries to ensure we matched a full path segment
                    bool boundaryStart = (pos == 0 || remaining[pos - 1] == '/');
                    bool boundaryEnd = (pos + ctxPath.length() == remaining.length() || remaining[pos + ctxPath.length()] == '/');

                    if (boundaryStart && boundaryEnd) {
                        // Extract the suffix (what comes AFTER the current context, e.g., "g/data")
                        std::string suffix = remaining.substr(pos + ctxPath.length());
                        if (!suffix.empty()) {
                            if (suffix[0] == '/') suffix.erase(0, 1);
                            replaceSlashWithAmpersand(suffix);
                            segments.push_back(suffix);
                        }

                        // Transform the context block itself (e.g., "d/e/F" -> "d&e&F")
                        replaceSlashWithAmpersand(ctxPath);
                        segments.push_back(ctxPath);

                        // Reduce the string for the next iteration
                        remaining = remaining.substr(0, pos);
                        if (!remaining.empty() && remaining.back() == '/') {
                            remaining.pop_back();
                        }
                    }
                }
            }
            
            if (current->getType() == CTX_ARRAYSTRUCT_TYPE) {
                current = static_cast<ArraystructContext*>(current)->getParent();
            } else {
                current = nullptr;
            }
        }

        // 2. Process what remains at the beginning of the string (e.g., "A/B/a/C" if not covered by ctx)
        // Split by '/' as these are normally distinct levels (Arrays/Structures)
        if (!remaining.empty()) {
            size_t pos = 0;
            while ((pos = remaining.rfind('/')) != std::string::npos) {
                std::string part = remaining.substr(pos + 1);
                if (!part.empty()) segments.push_back(part);
                remaining = remaining.substr(0, pos);
            }
            if (!remaining.empty()) {
                segments.push_back(remaining);
            }
        }

        // 3. Reconstruct the final string
        std::string result = "";
        // Iterate through the vector in reverse because we pushed from end to start
        for (int i = segments.size() - 1; i >= 0; --i) {
            result += "/" + segments[i];
        }

        // If the original string did not have a '/', remove the first one added
        if (!hasLeadingSlash && !result.empty()) {
            result.erase(0, 1);
        }

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
    bool isTimedContext(Context *ctx) {
        if (!ctx || ctx->getType() != CTX_ARRAYSTRUCT_TYPE) {
            return false;
        }

        ArraystructContext* arr_ctx = static_cast<ArraystructContext*>(ctx);
        while (arr_ctx != nullptr) {
            if (arr_ctx->getTimed()) {
                return true;
            }
            arr_ctx = arr_ctx->getParent();
        }
        return false;
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
            if (datatype == alconst::double_data) {
                *data = malloc(leaf->count * sizeof(double));
                panzer_db_ptr->readTensor<double>(*leaf, static_cast<double*>(*data));
            } else if (datatype == alconst::integer_data) {
                *data = malloc(leaf->count * sizeof(int32_t));
                panzer_db_ptr->readTensor<int32_t>(*leaf, static_cast<int32_t*>(*data));
            } else if (datatype == alconst::complex_data) {
                *data = malloc(leaf->count * sizeof(std::complex<double>));
                panzer_db_ptr->readTensor<std::complex<double>>(*leaf, static_cast<std::complex<double>*>(*data));
            } else if (datatype == alconst::char_data) {
                if (leaf->shape.size() <= 1) { // 1D list or Scalar
                    std::vector<std::string> str_list(leaf->count);
                    panzer_db_ptr->readTensor<std::string>(*leaf, str_list.data());
                    
                    *data = malloc(leaf->count * sizeof(char*));
                    char** out_ptr = static_cast<char**>(*data);
                    for (size_t i = 0; i < leaf->count; ++i) {
                         size_t len = str_list[i].size() + 1;
                         out_ptr[i] = (char*)malloc(len);
                         std::memcpy(out_ptr[i], str_list[i].c_str(), len);
                    }
                } else {
                    throw ALBackendException("HDF5Reader_v2: Multidimensional strings not supported", LOG);
                }
            } else {
                throw ALBackendException("HDF5Reader_v2: Unknown datatype", LOG);
            }

            // Update output dimensions
            *dim = leaf->shape.size();
            for (size_t i = 0; i < leaf->shape.size(); ++i) {
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
     * @return 1 on success, 0 on failure.
     */
    int read_dataset_globally(Context *ctx, std::string &dataset_name, int* datatype, void **data, int *dim, int *size) {
        DEBUG_PRINT("--> Entering read_dataset_globally for dataset: " << dataset_name);

        int type = ctx->getType();

        if (!panzer_db_ptr) {
            throw ALBackendException("PanzerDB not initialized", LOG);
        }

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
        int64_t target_time_index = -1;
        Context* curr = ctx;
        
    
        while (curr != nullptr) {
            // Potential crash here if curr is invalid
            if (curr->getType() == CTX_ARRAYSTRUCT_TYPE) {
                ArraystructContext* arr = static_cast<ArraystructContext*>(curr);
                std::string full_path = arr->getPath(); 
                std::string node_name;
                
                Context* parent = arr->getParent();
                if (parent && parent->getType() == CTX_ARRAYSTRUCT_TYPE) {
                    ArraystructContext* parent_arr = static_cast<ArraystructContext*>(parent);
                    std::string parent_path = parent_arr->getPath();
                    if (full_path.size() > parent_path.size() && full_path.rfind(parent_path + "/", 0) == 0) {
                        node_name = full_path.substr(parent_path.size() + 1);
                    } else {
                        node_name = full_path;
                    }
                } else {
                    node_name = full_path;
                }
                
                std::replace(node_name.begin(), node_name.end(), '/', '&');
                
                path_segments.insert(path_segments.begin(), node_name);
                
                if (arr->getTimed()) {
                    target_time_index = arr->getIndex();
                    is_dynamic_level.insert(is_dynamic_level.begin(), true);
                } else {
                    is_dynamic_level.insert(is_dynamic_level.begin(), false);
                }
                indices.insert(indices.begin(), arr->getIndex());
                curr = arr->getParent();
            }
            else {
                curr = nullptr;
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

        auto it = path_cache.find(specific_path);
        if (it != path_cache.end() && !it->second.empty()) {
            DEBUG_PRINT("Found specific path: " << specific_path);
            // FIX: Always filter by time index if in a dynamic context,
            // even for a specific path. This handles cases where multiple time
            // slices might be associated with the same path string in the cache.
            if (target_time_index != -1) {
                for (const auto* leaf : it->second) {
                    if (leaf->time_index == static_cast<uint64_t>(target_time_index)) {
                        sorted_leaves.push_back(leaf);
                    }
                }
            } else {
                sorted_leaves = it->second;
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
            
            it = path_cache.find(generic_path);
            if (it != path_cache.end() && !it->second.empty()) {
                DEBUG_PRINT("Found generic path: " << generic_path);
                // Filter by time index if we are in a dynamic context
                if (target_time_index != -1) {
                    for (const auto* leaf : it->second) {
                        if (leaf->time_index == static_cast<uint64_t>(target_time_index)) {
                            sorted_leaves.push_back(leaf);
                        }
                    }
                } else {
                    sorted_leaves = it->second;
                }
                found = true;
            }
        }

        if (!found || sorted_leaves.empty()) {
            // Fallback: Linear search for dynamic signals read from a parent
            // (e.g., read "profiles_1d/signal" from the root)
            // Note: This case is rare if contexts are used correctly.
            
            // ✅ FIX: FALLBACK LINEAR SEARCH IS NECESSARY!
            const auto& leaves = panzer_db_ptr->getLeaves();
            for (const auto& leaf : leaves) {
                if (target_time_index != -1 && leaf.time_index != static_cast<uint64_t>(target_time_index)) continue;
                
                // Check if path ends with dataset name
                if (leaf.path == clean_ds_name || 
                    (leaf.path.size() > clean_ds_name.size() && 
                     leaf.path.compare(leaf.path.size() - clean_ds_name.size(), clean_ds_name.size(), clean_ds_name) == 0 &&
                     leaf.path[leaf.path.size() - clean_ds_name.size() - 1] == '/')) {
                    
                    // Check if path starts with context prefix (if any)
                    if (!context_prefix.empty()) {
                        if (leaf.path.compare(0, context_prefix.size(), context_prefix) == 0) {
                            sorted_leaves.push_back(&leaf);
                        }
                    } else {
                        sorted_leaves.push_back(&leaf);
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
            total_elements += leaf->count;
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
        bool is_numeric_scalar_leaf = (first_leaf_type != PanzerDB::DataType::STRING && first_leaf->shape.empty());
        bool is_string_scalar_leaf = (first_leaf_type == PanzerDB::DataType::STRING && first_leaf->count == 1);

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
            case PanzerDB::DataType::COMPLEX128: actual_datatype = alconst::complex_data; break;
            default:
                DEBUG_PRINT("Unknown data type in leaf flags: " << (first_leaf->flags >> 4));
                return 0; // Unknown type
        }

        size_t leaf_rank = first_leaf->shape.size();

        // 3. CHAR / STRING Processing
        // Always read with the actual file type. Conversion will be done by al_lowlevel.
        DEBUG_PRINT("Actual data type determined from leaf flags: " << actual_datatype);
        if (actual_datatype == alconst::char_data) {
             DEBUG_PRINT("Processing string data...");
             std::vector<std::string> temp_buffer;
             temp_buffer.reserve(total_elements);
             for (const auto* leaf : sorted_leaves) {
                 std::vector<std::string> leaf_strings(leaf->count);
                 panzer_db_ptr->readTensor(*leaf, leaf_strings.data());
                 temp_buffer.insert(temp_buffer.end(), leaf_strings.begin(), leaf_strings.end());
             }
 
             size_t max_str_len = 0;
             for (const auto& s : temp_buffer) if (s.size() > max_str_len) max_str_len = s.size();
             max_str_len += 1; // Null terminator

             // Heuristic: Scalar vs List (Corrected)
             // Use the shape of the first leaf to determine if it was written as a scalar or an array.
             // Scalar string: shape is empty (rank 0).
             // List of strings: shape has rank 1.
             bool is_scalar_leaf = (first_leaf->shape.empty());
             DEBUG_PRINT("Is scalar leaf: " << is_scalar_leaf);

             bool return_as_scalar = (is_scalar_leaf && total_elements == 1);

             if (return_as_scalar) {
                DEBUG_PRINT("Detected scalar string leaf. Returning as single string.");
                 *dim = 1;
                 size[0] = (int)temp_buffer[0].size(); // Length excluding null
                 *data = (char*)malloc(size[0] + 1);
                 memcpy(*data, temp_buffer[0].c_str(), size[0] + 1);
             } else {
                 DEBUG_PRINT("Detected list of strings. Returning as 2D char array with max string length: " << max_str_len);
                 *dim = 2;
                 size[0] = (int)total_elements;
                 size[1] = (int)max_str_len;
                 
                 size_t buffer_bytes = total_elements * max_str_len;
                 char* char_buffer = (char*)malloc(buffer_bytes);
                 std::memset(char_buffer, 0, buffer_bytes);
                 for (size_t i = 0; i < total_elements; ++i) {
                     if (!temp_buffer[i].empty()) strncpy(char_buffer + (i * max_str_len), temp_buffer[i].c_str(), max_str_len);
                 }
                 *data = char_buffer;
             }
 
             *datatype = actual_datatype; // Return the type that was read
             return 1;
        }

        // 4. NUMERICAL Processing
        if (leaf_rank == 0) {
            // 0D signal (scalar) -> becomes 1D with time dimension
            if (total_elements > 1) { 
                *dim = 1; 
                size[0] = (int)total_elements; 
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

        // Allocation
        if (actual_datatype == alconst::double_data) *data = malloc(total_elements * sizeof(double));
        else if (actual_datatype == alconst::integer_data) *data = malloc(total_elements * sizeof(int32_t));
        else if (actual_datatype == alconst::complex_data) *data = malloc(total_elements * sizeof(std::complex<double>));
        else return 0;

        // OPTIMIZATION: Grouped read (Hyperslab Union)
        int res = panzer_db_ptr->readLeavesUnion(sorted_leaves, *data, actual_type_enum); 
        if (res < 0) {
            free(*data);
            *data = nullptr;
            return 0;
        }

        *datatype = actual_datatype; // Return the type that was read
        
        return 1;
    }
    
};

#endif // I_READ_STRATEGY_H
