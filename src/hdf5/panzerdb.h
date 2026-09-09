/**
 * @file panzerdb.h
 * @brief PanzerDB: a high-performance, columnar, index-table-based HDF5 storage engine
 *        used by the IMAS-core HDF5 backend (v2) for reading and writing time-dependent
 *        scientific data in a flat, append-friendly layout.
 *
 * This header declares the PanzerDB class (the storage engine itself) and the small
 * helper types it exposes (Leaf, ChunkingConfig, ArrayLevel, PathComponents,
 * ChunkingStats). All read and write operations go through the single central
 * "/index" table; see the PanzerDB class documentation for the design rationale.
 */
#pragma once
#include <hdf5.h>
#include <string>
#include <vector>
#include <cstdint>
#include "al_exception.h"
#include <iostream>
#include <complex>
#include <numeric>
#include <algorithm> 
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <map>
#include "metadata/metadata_extractor.h"

#include "direct_access_api.h" // for imas::direct_access::DataType (return type of getLeafType)

/**
 * @class PanzerDB
 * @brief A high-performance HDF5-based data storage engine.
 *
 * PanzerDB provides a high-level C++ interface for storing and retrieving large-scale
 * scientific data, designed as a replacement for the legacy HDF5 backend in the IMAS
 * data model. It uses a columnar storage approach, where data is organized by type
 * into flat, one-dimensional datasets, and a central index table maps structured data
 * paths to raw data offsets.
 *
 * Key Features:
 * - **Columnar Storage**: All data of the same type (e.g., double, int32) is appended
 *   to a single, large, chunked HDF5 dataset. An index table (`/index`) stores metadata
 *   for each data entry, including its path, shape, time index, and location (offset, count)
 *   in the raw data dataset. This design is highly efficient for both writing and for
 *   reading specific time slices or data subsets.
 *
 * - **Array of Structures (AoS) Emulation**: PanzerDB emulates complex, nested data
 *   structures (Arrays of Structures) using a path-based system. For example, a path
 *   like "profiles_1d/0/ion/0/density" represents a specific data node within a
 *   hierarchical structure.
 *
 * - **Static and Dynamic AoS**: It supports both statically-sized AoS (size known at
 *   creation) and dynamically-sized AoS (size grows over time). Dynamic AoS are ideal
 *   for time-evolving data, where each "slice" corresponds to a new time step.
 *
 * - **Optimized I/O**:
 *   - **Write Buffering**: Data writes are buffered in memory and flushed to disk in
 *     large, contiguous blocks, minimizing the number of costly HDF5 I/O operations.
 *   - **Dynamic Chunking**: HDF5 chunk sizes are automatically configured based on a
 *     "usage hint" (e.g., "time_series", "bulk_write", "interactive") to optimize
 *     performance for the intended access pattern.
 *   - **Optimized Reads**: Features a fast, cached index (`getLeaves`), direct-to-buffer
 *     hyperslab reads (`readSliceDirect`), and batched reads for multiple data blocks
 *     (`readLeavesUnion`).
 *
 * - **Time-Series Data Handling**: Provides first-class support for time-dependent data,
 *   including efficient time-based lookups and data interpolation (`readInterpolatedData`).
 *
 * - **Flexible Instantiation**: Can be instantiated to manage an entire HDF5 file or to
 *   operate within a specific HDF5 group, allowing it to be seamlessly embedded into
 *   existing HDF5 file structures.
 *
 * - **Multiple Open Modes**:
 *   - `WRITE`: Creates a new file, truncating if it exists.
 *   - `APPEND`: Opens an existing file for reading and writing, preserving its content.
 *   - `READ`: Opens an existing file for read-only access.
 */

/**
 * @struct ChunkingConfig
 * @brief Holds configuration parameters for HDF5 chunking, compression, and caching.
 *
 * This structure allows fine-tuning of storage performance by adjusting how data
 * is laid out on disk and cached in memory.
 */
struct ChunkingConfig {
    // Index table chunking
    size_t index_chunk_rows = 8192;           // Number of rows per chunk
    
    // Data chunking (by type)
    size_t data_chunk_f64 = 131072;           // 1 MB per chunk (131k doubles)
    size_t data_chunk_i32 = 262144;           // 1 MB per chunk (262k int32)
    size_t data_chunk_c128 = 65536;           // 1 MB per chunk (65k complex)
    size_t data_chunk_str = 8192;             // 8k strings per chunk
    
    // Path chunking
    size_t path_chunk_entries = 8192;         // 8k paths per chunk
    
    // Compression settings
    bool enable_compression = true;
    int compression_level = 6;                // 0-9, 6 is a good compromise
    
    // Cache settings (HDF5 metadata cache)
    size_t chunk_cache_size = 64 * 1024 * 1024;     // 64 MB
    size_t chunk_cache_nslots = 10007;               // Prime number for hash
};

/**
 * @struct ArrayLevel
 * @brief Descriptor of one level (element) on the AoS iteration stack.
 *
 * An ArrayLevel represents one AoS in the hierarchy of beginArray()/endArray()
 * calls currently open. It is both the record kept on `array_stack` and the
 * argument of the "manual" beginArray(ArrayLevel&) overload.
 */
struct ArrayLevel {
    std::string name;              // Name of this AoS (e.g. "ion").
    std::string saved_path_prefix; // Path prefix before entering this level (restored by endArray()).
    std::string timebase;          // Time base (non-empty only for dynamic AoS).
    bool is_dynamic = false;       // True if this AoS is time-evolving.
    size_t declared_size = 0;      // Fixed size of a static AoS (0 for dynamic).
    size_t current_index = 0;      // Index of the element currently being processed.
    size_t actual_count = 0;       // Number of elements actually written so far.
    bool had_write = false;        // True if at least one data node was written at this level.

    // Full path of the AoS (for lookup in aos_time_counters).
    std::string aos_full_path;

    // Row position in /index of this AoS's meta-node, assigned by beginArray.
    // Children record it as their parent_id so parent_path can be rebuilt on read
    // without duplicating the full parent text on every leaf.
    uint64_t container_row_id = 0;
};

// Sentinel for the parent_id column of a root row (no enclosing AoS meta node).
inline constexpr uint64_t PANZER_NO_PARENT_ROW = 0xFFFFFFFFFFFFFFFFULL;

/**
 * @struct PathComponents
 * @brief Parsed components of an instance path inside a (possibly dynamic) AoS.
 *
 * Splits e.g. "profiles_2d/3/ion/1/state/2/z_min" into
 * prefix="profiles_2d", index_str="3", suffix="/ion/1/state/2/z_min",
 * so that the AoS index can be substituted when resolving dynamic nodes.
 */

struct PathComponents {
    std::string prefix;      // "profiles_2d"
    std::string index_str;   // "3"
    std::string suffix;      // "/ion/1/state/2/z_min"
    bool has_index;

    PathComponents() : has_index(false) {}
};

/**
 * @struct ChunkingStats
 * @brief Aggregate I/O counters exposed for diagnostics (getChunkingStats()).
 *
 * All counters are cumulative for the life of the PanzerDB instance.
 */
struct ChunkingStats {
    size_t total_chunks_written = 0;
    size_t total_chunks_read = 0;
    size_t total_bytes_written = 0;
    size_t total_bytes_read = 0;
    size_t compression_ratio = 100;  // Percentage (100 = no compression)
};

class PanzerDB {
public:
    /**
     * @struct Leaf
     * @brief One entry of the PanzerDB index table: a fully self-describing data node.
     *
     * Each leaf corresponds to one row of the central "/index" dataset and holds every
     * piece of metadata needed to locate, interpret and read one logical node
     * (a static tensor, one chunk of a time signal, or an Array-of-Structures meta-node).
     *
     * Notes:
     * - `path` / `parent_path` are zero-copy views into the internal path storage
     *   (the cached_paths_blocks). They are valid only while the leaf cache lives;
     *   copy them out if you need to keep them longer.
     * - The low 4 bits of `flags` encode the node role, the upper bits encode the
     *   stored data type (see DataType). Typical values in production:
     *   flags = 0        : normal data node (double)
     *   flags = 4        : data node of type (4>>4)=INT32, etc.
     *   flags = 2        : static AoS meta-node (no raw data, size in `shape[0]`)
     *   flags = 3        : dynamic AoS meta-node (time-evolving; see aos_time_counters)
     *   flags = 1        : explicitly preserved empty node.
     */
    struct Leaf {
        std::vector<size_t> shape;      // Dimensions of a single time slice (empty for scalars/meta-nodes).
        uint64_t time_index = 0;        // First time step stored in this leaf (0 for static data).
        std::string_view path;          // Full instance path, e.g. "profiles_1d/0/ion/0/density".
        std::string_view parent_path;   // Path of the immediate parent node.
        uint64_t offset = 0;            // Element offset of this leaf inside its raw data dataset.
        uint64_t count = 0;             // Number of stored elements (slice_volume * n_time_steps).
        uint64_t flags = 0;             // low 4 bits: node kind; upper bits: DataType. See struct doc.
        bool is_empty = false;          // Convenience flag mirroring (flags & 0xF) == 1.
    };

private:
    
    hid_t file_id = -1;
    hid_t index_dset = -1;
    hid_t paths_dset = -1;
    hid_t parent_paths_dset = -1;

    // Datasets for each data type
    mutable hid_t data_dset_f64 = -1; // double
    mutable hid_t data_dset_i32 = -1; // int64_t
    mutable hid_t data_dset_c128 = -1; // complex
    mutable hid_t data_dset_str = -1; // string

    // Cache for dataset sizes to avoid H5Dget_space calls
    hsize_t disk_size_f64 = 0;
    hsize_t disk_size_i32 = 0;
    hsize_t disk_size_c128 = 0;
    hsize_t disk_size_str = 0;

    /**
     * @brief (Re)loads the on-disk size of every raw data dataset into disk_size_*.
     * @note Called once after opening (WRITE or APPEND) and after each flush(), so
     *       that append offsets (disk_size_* + in-memory buffer size) stay correct.
     */
    void updateDiskSizes();

    std::unordered_map<std::string, uint64_t> aos_time_counters;
    std::string dynamic_level;
    std::vector<std::string> current_path;
    std::string path_prefix;
    std::vector<ArrayLevel> array_stack;

    bool preserve_empty_nodes = false;
    bool last_level_had_write = false;
    bool should_close_loc_id = false;

    // Running /index row id. In WRITE mode it starts at 0; in APPEND mode it is
    // seeded from the on-disk /index row count so appended rows keep the same ids
    // a later READ observes (parent-first ordering).
    uint64_t next_row_id = 0;

    // RAM Buffers (only what is necessary)
    std::vector<uint64_t> index_buffer;   // 14 columns: name_id, ndim, shape[6], time_index, offset, count, flags, parent_id, index_value
    std::vector<double> data_buffer_f64;
    std::vector<int32_t> data_buffer_i32;
    std::vector<std::complex<double>> data_buffer_c128;
    std::vector<std::string> data_buffer_str;

    std::vector<char> paths_buffer;       // Buffer for plain text paths (fixed size)
    std::vector<char> parent_paths_buffer;

     // Constants for buffer management
    static constexpr size_t INITIAL_INDEX_BUFFER_SIZE = 1000;      
    static constexpr size_t INITIAL_DATA_BUFFER_SIZE = 8192;       
    static constexpr size_t INITIAL_STRING_BUFFER_SIZE = 512;      
    static constexpr size_t BUFFER_GROWTH_FACTOR = 2;              
    
    // Cache to avoid reconstructions
    /**
     * @brief Joins the given AoS stack levels into a single fully-qualified path.
     *        Root-to-leaf order is preserved.
     * @param stack_vector The AoS levels, from root to the current level.
     * @return The joined path string (e.g. "profiles_1d/3/ion").
     */
    std::string buildPathFromStack(const std::vector<ArrayLevel>& stack_vector) const;

    mutable std::string cached_dynamic_aos_path;
    mutable bool dynamic_aos_path_valid = false;

    /**
     * @brief Invalidates the cached dynamic-AoS path so it is recomputed on next use.
     * @note Called by every beginArray()/endArray() variant because the active
     *       dynamic AoS can change whenever the stack changes.
     */
    void invalidateDynamicAOSCache();
    
    /**
     * @brief Grows an in-memory write buffer to hold at least `required_space` elements.
     * @param required_space Minimum number of elements the buffer must be able to hold.
     */
    void growBufferIfNeeded(size_t required_space);

    /**
     * @brief Appends the buffered complex values into the complex raw dataset via the writer.
     */
    void flushComplexBuffer();

    /**
     * @brief Appends the buffered string values into the string raw dataset via the writer.
     */
    void flushStringBuffer() ;

    /**
     * @brief Appends a chunk of plain-text path entries into the given path dataset.
     * @param dataset_id Target path dataset (paths_dset or parent_paths_dset).
     * @param buffer The path characters to append (PATH_MAX_LEN per entry).
     */
    void flushPathBuffer(hid_t dataset_id, std::vector<char>& buffer);

    /**
     * @brief Appends a chunk of numeric values into a given raw data dataset.
     * @tparam T Value type (double, int32_t, std::complex<double>).
     * @param dataset_id Target raw data dataset.
     * @param buffer The values to append.
     * @param h5_type The matching native HDF5 type id.
     */
    template<typename T>
    void flushDataBuffer(hid_t dataset_id, std::vector<T>& buffer, hid_t h5_type);
   

    // Cache for reading
    mutable std::vector<Leaf> cached_leaves;
    mutable bool leaves_cache_valid = false;

    mutable std::list<std::vector<char>> cached_paths_blocks;
    mutable std::list<std::vector<char>> cached_parent_paths_blocks;
    mutable std::unordered_map<std::string_view, std::vector<size_t>> leaf_lookup;
    mutable std::unordered_map<std::string_view, std::vector<size_t>> parent_lookup;
    mutable std::vector<std::string> cached_dynamic_aos_roots;

    // For each dynamic AoS root: the highest time_index found among ALL of its
    // descendants (any depth), built in getLeaves(). Used by getAOSShape so a
    // dynamic AoS that only contains nested static AoS (e.g. time_slice/ggd/
    // theta/values) still reports the correct size.
    mutable std::unordered_map<std::string_view, uint64_t> max_time_at_dynamic_root;

    // Reusable scratch buffers to avoid repetitive malloc/free on reads
    mutable std::vector<double> scratch_f64;
    mutable std::vector<std::string> scratch_str;

    std::unordered_set<std::string> written_metadata_schema_paths;

    /**
     * @brief Rebuilds max_time_at_dynamic_root from the currently cached leaves:
     *        for every dynamic AoS root, the max time_index of ALL descendant
     *        data leaves (any depth).
     * @note Called once per cache (re)build, by getLeaves().
     */
    void rebuildDynamicRootTimeIndex() const;

    /**
     * @brief Splits `path` relative to the `aos_path` root (dynamic AoS path substitution).
     * @param path     The full instance path to parse.
     * @param aos_path The AoS root it belongs to.
     * @return The parsed prefix / index / suffix components.
     */
    PathComponents parsePath(const std::string& path,
                                    const std::string& aos_path) const;

    ChunkingConfig chunk_config;

    /**
     * @brief Applies a chunking/compression configuration preset selected by a usage hint.
     * @param usage_hint One of "time_series", "array_of_structures", "bulk_write",
     *                   "interactive", or an empty string for the default preset.
     */
    void configureChunking(const std::string& usage_hint);

    /**
     * @brief Creates a chunked (and optionally deflated) HDF5 dataset with the given layout.
     * @param name             Name of the dataset under the PanzerDB root location.
     * @param type             HDF5 native type id.
     * @param chunk_size       Number of elements per chunk (1-D layout).
     * @param enable_compression If true, applies GZIP at chunk_config.compression_level.
     * @param dapl             Data access property list (chunk cache), or H5P_DEFAULT.
     * @return The new dataset id (>=0), or a negative id on failure.
     */
    hid_t createOptimizedDataset(const std::string& name,
                                   hid_t type,
                                   size_t chunk_size,
                                   bool enable_compression,
                                   hid_t dapl = H5P_DEFAULT);

    /**
     * @brief Restores chunking/compression settings persisted in an existing file (APPEND).
     */
    void readChunkingConfig();

    /**
     * @brief Installs the HDF5 raw-data (chunk) cache on the datasets for read optimization.
     */
    void configureReadCache();

    //==========================================================================
    // Configuration API
    //==========================================================================

    /**
     * @brief Sets a usage hint to optimize chunking for new files (in WRITE mode).
     * @param hint A string hint, e.g., "time_series", "bulk_write", "interactive".
     */
    void setChunkingHint(const std::string& hint);

     /**
     * @brief Sets the GZIP compression level for new datasets.
     * @param level An integer from 0 (no compression) to 9 (max compression).
     */
    void setCompressionLevel(int level);

    /**
     * @brief Disables compression for new datasets.
     */
    void disableCompression();

    //==========================================================================
    // Statistics and Debugging
    //==========================================================================

    /**
     * @brief Retrieves statistics about chunking performance.
     * @return A ChunkingStats struct.
     */
    ChunkingStats getChunkingStats() const;

    /**
     * @brief Prints chunking statistics to standard output.
     */
    void printChunkingStats() const ;



public:
    /**
     * @enum OpenMode
     * @brief Defines how the PanzerDB file should be opened.
     */
    enum class OpenMode { WRITE, READ, APPEND };

    /**
     * @enum DataType
     * @brief Internal enumeration for data types, stored in the index flags.
     */
    enum class DataType : uint64_t {
        FLOAT64 = 0,
        INT32 = 1,
        COMPLEX128 = 2,
        STRING = 3,
        LIST_OF_STRINGS = 4,
        UNKNOWN = 99
    };

    /**
     * @brief Constructs a PanzerDB instance to manage an HDF5 file.
     * @param filename The path to the HDF5 file.
     * @param mode The mode in which to open the file (WRITE, READ, APPEND).
     * @param preserve_empty_nodes If true, metadata for empty nodes is preserved.
     */
    PanzerDB(const std::string& filename, OpenMode mode, bool preserve_empty_nodes = false);

     /**
     * @brief Constructs a PanzerDB instance within an existing HDF5 group.
     * @param loc_id The HDF5 identifier of the parent group.
     * @param mode The mode for operations within the group (WRITE, READ, APPEND).
     * @param preserve_empty_nodes If true, metadata for empty nodes is preserved.
     * @param close_loc_id_on_exit If true, PanzerDB will call H5Gclose on loc_id upon destruction.
     */
    PanzerDB(hid_t loc_id, OpenMode mode, bool preserve_empty_nodes = false, bool close_loc_id_on_exit = false);

    // Disable copy to prevent accidental closure of HDF5 handles by temporary copies
    PanzerDB(const PanzerDB&) = delete;
    PanzerDB& operator=(const PanzerDB&) = delete;

    /**
      * @brief Destructor. Flushes any remaining data and closes the file.
      */
    ~PanzerDB();


    /**
      * @brief Gets the current open mode of the database.
      * @return The current OpenMode (READ, WRITE, or APPEND).
      */
    OpenMode getOpenMode() const { return mode; }



    //==========================================================================
    // Write API - Array of Structures
    //==========================================================================

    /**
     * @brief Begins a new statically-sized Array of Structures (AoS) level.
     * @param name The name of the AoS.
     * @param size The fixed number of elements in the array.
     */
    void beginArray(const std::string& name, size_t size);

     /**
     * @brief Begins a new dynamically-sized (time-evolving) Array of Structures (AoS) level.
     * @param name The name of the AoS.
     * @param timebase The name of the dataset that serves as the time coordinate for this AoS.
     */
    void beginArray(const std::string& name, const std::string& timebase);

    /**
     * @brief Enters a caller-constructed AoS level (full control over the ArrayLevel fields).
     * @param level The level descriptor to push (name, size, dynamic flag, time base, index).
     * @note Registers a meta-node in the index table unless it already exists (APPEND/READ),
     *       updates path_prefix and pushes the level on the AoS stack.
     */
     void beginArray(ArrayLevel& level);

    /**
     * @brief Moves to the next element of the current AoS level.
     * @note Advances current_index of the stack top by one; a no-op when no AoS is open.
     */
     void incrementArrayIndex();

    /**
     * @brief Jumps to a specific element of the current AoS level.
     * @param new_index The element index to set on the stack top.
     * @note No-op when no AoS is open.
     */
     void setCurrentArrayIndex(size_t new_index);

    /**
     * @brief Checks if the current write position is inside a dynamic AoS.
     * @param timebase If inside a dynamic AoS, this string is filled with the name of the timebase.
     * @return True if inside a dynamic AoS, false otherwise.
     */
    bool isInsideDynamicAOS(std::string* timebase) const;

    /**
     * @brief Exits the current AoS level, restoring the path prefix and counts.
     * @note Symmetric to beginArray(). When leaving a dynamic AoS, the corresponding
     *       dynamic-level state is cleared and caches that depend on the stack are
     *       invalidated.
     */
    void endArray();

     /**
     * @brief Gets the current depth of the AoS stack.
     * @return The number of active `beginArray` calls.
     */
    size_t getArrayStackSize() const { return array_stack.size(); }

    /**
     * @brief Synchronizes the PanzerDB array stack with the provided indices
     * This method rebuilds the internal state (array_stack) to exactly match
     * the AoS indices in the AL context.
     * 
     * @param aos_names Names of the AoS in hierarchical order (e.g., ["flux_loop", "channel"])
     * @param indices Current indices for each level (e.g., [2, 5])
     */
    void synchronizeArrayStack(const std::vector<std::string>& aos_names, 
                            const std::vector<int>& indices);

    /**
     * @brief Checks if the AoS stack is empty.
     * @return True if not inside any AoS, false otherwise.
     */
    bool isArrayStackEmpty() const { return array_stack.empty(); }

    /**
     * @brief Advances the time counter of the enclosing dynamic AoS by n_steps.
     * @param timebase_name Name of the time base (used for diagnostics).
     * @param n_steps Number of time steps to skip.
     * @pre Must be called while inside a dynamic AoS.
     * @throws std::runtime_error if not inside a dynamic AoS.
     */
    void advanceTimebase(const std::string& timebase_name, uint64_t n_steps = 1);

    //==========================================================================
    // Write API - Data
    //==========================================================================

    /**
     * @brief Writes a static (non-time-dependent) data tensor of any supported type.
     * @tparam T The element type (double, int32_t, std::complex<double>, char, const char*).
     * @param name     Name of the data node under the current path.
     * @param shape    Dimensions of the tensor (empty for a scalar).
     * @param data     Pointer to the element buffer.
     * @param count    Total number of elements.
     * @param timebase Must be empty for static data.
     * @note Appends to the in-memory buffers; physically stored on flush() or destruction.
     */
    template<typename T>
    void writeData(const std::string& name,
                   const std::vector<size_t>& shape,
                   const T* data, size_t count,
                   const std::string& timebase = "");

     /**
      * @brief Writes one or more time slices of a dynamic double-precision floating point signal.
     * @param name The name of the signal.
     * @param base_shape The shape of a single time slice.
     * @param data Pointer to the contiguous data for all slices.
     * @param n_slices The number of slices to write.
     * @param timebase The name of the associated timebase.
     */
    void writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const double* data, size_t n_slices,
                               const std::string& timebase);

    /**
     * @brief Writes one or more time slices of a dynamic 32-bit integer signal.
     * @param name       Name of the signal under the current path.
     * @param base_shape Shape of a single time slice.
     * @param data       Pointer to the contiguous buffer holding all slices.
     * @param n_slices   Number of time slices to append.
     * @param timebase   Name of the associated time base (may be empty).
     * @note Time steps are numbered using aos_time_counters, preserving gaps and
     *       aligning with the enclosing dynamic AoS.
     */
     void writeDataSlices(const std::string& name,
                                const std::vector<size_t>& base_shape,
                                const int32_t* data, size_t n_slices,
                                const std::string& timebase);

    /**
     * @brief Writes one or more time slices of a dynamic complex signal.
     * @param name       Name of the signal under the current path.
     * @param base_shape Shape of a single time slice.
     * @param data       Pointer to the contiguous buffer holding all slices.
     * @param n_slices   Number of time slices to append.
     * @param timebase   Name of the associated time base (may be empty).
     */
     void writeDataSlices(const std::string& name,
                                const std::vector<size_t>& base_shape,
                                const std::complex<double>* data, size_t n_slices,
                                const std::string& timebase);

    /**
     * @brief Writes one or more time slices of a dynamic string signal.
     * @param name       Name of the signal under the current path.
     * @param base_shape Shape of a single time slice.
     * @param data       Array of null-terminated string pointers (one per slice element).
     * @param n_slices   Number of time slices to append.
     * @param timebase   Name of the associated time base (may be empty).
     */
     void writeDataSlices(const std::string& name,
                                const std::vector<size_t>& base_shape,
                                const char* const* data, size_t n_slices,
                                const std::string& timebase);

    //==========================================================================
    // Write API - Metadata
    //==========================================================================

    /**
      * @brief Writes a metadata entry for a given path.
      * @param path The base path (e.g., "profiles_1d/t_e")
      * @param value The metadata value to store
      */
    void writeMetadata(const std::string& path, const std::string& value);

    /**
     * @brief Utility to convert an instance path (e.g. "A/0/B/signal") to a schema path ("A/B/signal").
     * Removes all purely numeric path segments. Useful for retrieving shared metadata.
     */
    static std::string stripIndices(const std::string& path);

    //==========================================================================
    // Lifecycle - Terminal
    //==========================================================================

    /**
      * @brief Flushes all in-memory write buffers to the HDF5 file.
      * This makes the written data visible to other readers without closing the file.
      */
    void flush();

    /**
      * @brief Flushes all in-memory write buffers to the HDF5 file and closes all handles.
      */
    void close();

    /**
      * @brief Dumps the cached index table (leaves) to standard output, for debugging.
      */
    void dumpLeafIndex() const;

    //==========================================================================
    // Read API - Index
    //==========================================================================

    /**
     * @brief Retrieves the entire database index as a vector of Leaf objects.
    * The result is cached for subsequent calls. This is the primary entry point for read operations.
    * @return A constant reference to the cached vector of leaves.
    */
    const std::vector<Leaf>& getLeaves() const;

     /**
      * @brief Gets the effective size of an Array of Structures.
      * For static AoS, it returns the declared size. For dynamic AoS, it returns the number of time steps written.
      * @param level_name The full path to the AoS meta-node (e.g., "profiles_1d" or "profiles_1d/0/ion").
      * @return A vector containing the size of the AoS.
      */
     std::vector<size_t> getAOSShape(const std::string& level_name) const;

    /**
     * @brief Gets the next free time index of a dynamic AoS (number of slices written so far).
     * @param aos_path The full path of the dynamic AoS meta-node.
     * @return The slice counter, or 0 if the AoS is unknown.
     */
    size_t getDynamicAOSSize(const std::string& aos_path) const;

    /**
     * @brief Determines the stored data type of a node from its full path.
     * @param path The full instance path to the data node.
     * @return The matching imas::direct_access::DataType.
     * @throws ALBackendException if the path is not found in the index.
     */
    imas::direct_access::DataType getLeafType(const std::string& path);

    /**
     * @brief Checks if a given path corresponds to a dynamic Array of Structures (AoS).
     * @param aos_path The full path to the AoS meta-node (e.g., "profiles_1d").
     * @return True if the path points to a dynamic AoS, false otherwise.
     */
     bool isDynamicAOS(const std::string& aos_path) const;

    /**
     * @brief Reports whether a data signal (leaf) varies over time.
     * @details A signal is *dynamic* when it lives under a dynamic Array of Structures
     *          (the time axis comes from the AoS iteration), or — with no dynamic AoS —
     *          when it owns its own time axis, i.e. it stores more than one spatial slice
     *          (`count > ∏shape`) or appears as several rows at the same path.
     *          This mirrors the reader's own rule, which adds one implicit time dimension
     *          whenever `count > ∏shape` (cf. readDataByIndex / isTimeInLeaf).
     * @param signal_path Full instance path of the data leaf (e.g. "flux", "profiles_1d/0/ion/sig").
     * @return True if the signal is dynamic, false if it is static or the path is not found.
     */
    bool isDynamicSignal(const std::string& signal_path) const;

    /**
     * @brief Checks whether a time index falls within the time steps stored in a leaf.
     * @param leaf       The index leaf under test (must be a data leaf).
     * @param time_index The time index to test.
     * @return True if the leaf holds the requested time step.
     */
    bool isTimeInLeaf(const Leaf& leaf, int64_t time_index) const;

     /**
      * @brief Finds the index in a time base vector that is closest to a requested time.
      * @param timebase_path The full path to the 1D dataset representing the time base.
      * @param requested_time The time value to search for.
      * @param interp_mode The interpolation mode hint (see DataInterpolation).
      * @return The index of the element in the time base closest to the requested time.
      */
     int64_t getTimeIndex(const std::string& timebase_path, double requested_time, int interp_mode) const;

    /**
     * @brief Finds the time index of the nearest available slice for a data path.
     *
     * Resolves missing slices ("gaps"): if the requested index has no data,
     * returns the closest available one according to `prefer_direction`:
     * - -1: prefer the largest index <= requested (fall back to the smallest
     *      index >= requested if none below exists)
     * - +1: prefer the smallest index >= requested (fall back to the largest
     *      index <= requested if none above exists)
     * -  0: nearest in time (ties resolved to the smaller index)
     *
     * @param full_data_path Full path of the data node.
     * @param requested_idx The requested time index.
     * @param prefer_direction -1, +1 or 0 (see above).
     * @param time_basis Time basis of the dynamic structure.
     * @param requested_time The actual requested time (used for distance
     *        computation when prefer_direction == 0).
     * @return The available time index, or -1 if no data exists for the path.
     */
    int64_t nearestAvailableSliceIndex(const char* full_path, int64_t requested_idx,
                                       int64_t prefer_direction,
                                       const std::vector<double>& time_basis,
                                       double requested_time = -1.0) const;

    //==========================================================================
    // Read API - Data Retrieval
    //==========================================================================

    /**
     * @brief Reads the entire data tensor associated with a specific Leaf.
     * @tparam T The data type to read into (e.g., double, int32_t).
     * @param leaf The Leaf object from getLeaves() representing the data to read.
     * @param out_buffer A pre-allocated buffer to store the read data.
     */
    template<typename T>
    void readTensor(const Leaf& leaf, T* out_buffer) const;

     /**
     * @brief Reads a single time slice of data directly into an output buffer.
     * This is a highly optimized read that uses an HDF5 hyperslab to avoid intermediate copies.
     * @tparam T The data type.
     * @param leaf The Leaf representing the dynamic signal.
     * @param time_index The specific time index of the slice to read.
     * @param out_buffer A pre-allocated buffer to hold the slice data.
     * @return 0 on success, -1 on failure.
     */
    template<typename T>
    int readSliceDirect(const Leaf& leaf, int64_t time_index, T* out_buffer) const;

    /**
     * @brief Reads multiple leaves into a contiguous buffer using Hyperslab Union (H5S_SELECT_OR).
     *        Optimized for monotonic offsets. Falls back to sequential reads if offsets are not monotonic.
     * 
     * @param leaves Vector of leaves to read (must be sorted by desired output order, typically time)
     * @param buffer Output buffer (pre-allocated)
     * @param dtype Data type of the leaves
     * @return int 0 on success, -1 on failure
     */
    int readLeavesUnion(const std::vector<const Leaf*>& leaves, void* buffer, DataType dtype) const;

     /**
     * @brief Reads a single scalar value by its full path.
     * @tparam T The scalar type.
     * @param path The full path to the scalar node.
     * @param status Output parameter, set to 0 on success, -1 on failure.
     * @return The read scalar value.
     */
    template<typename T>
    T readScalar(const std::string& path, int *status) const {
        for (const auto& leaf : getLeaves()) {
            if (leaf.path == path) {
                if (!leaf.shape.empty() || leaf.count != 1) {
                    throw ALBackendException("Leaf at path '" + path + "' is not a scalar.", LOG);
                }
                T value;
                readTensor(leaf, &value);
                *status = 0;
                return value;
            }
        }
        *status = -1;
        return static_cast<T>(-1);
    }

    /**
     * @brief Reads data at a specific time, performing interpolation if necessary.
     * @param full_data_path The full path to the data node.
     * @param time The requested time point.
     * @param time_basis The time vector to use for interpolation.
     * @param interp_mode The interpolation mode (e.g., linear, closest).
     * @param datatype The type of data to read.
     * @param ndim_out Pointer to store the number of dimensions of the output.
     * @param shape_out Array to store the shape of the output.
     * @param data_out Pointer to store the allocated output data buffer.
     * @param expect_time_dim If true, expects the data to have a time dimension.
     * @return 0 on success, -1 on failure.
     */
    int readInterpolatedData(
                          const char* full_data_path, 
                          double time,
                          const std::vector<double>& time_basis,
                          int interp_mode,
                          int datatype,
                          uint64_t* ndim_out,
                          uint64_t shape_out[6],
                          void** data_out,
                          bool expect_time_dim = true);

    /**
      * @brief Reads a whole dynamic (time-evolving) double signal into memory.
      * @param dataset_name The full path of the signal (outside an AoS instance).
      * @return The concatenated time series, or an empty vector if not found.
      */
    std::vector<double> getWholeDynamicSignal(const std::string& dataset_name);



    /**
     * @brief C-style API to read double-precision data by path and time index.
     * Handles static data (time_index = -1), single time slices, and path substitution for dynamic AoS.
     * @param full_data_path The full path to the data node (e.g., "A/0/B/signal").
     * @param time_index The time index to read, or -1 for static data or to aggregate all time slices.
     * @param ndim_out Pointer to store the number of dimensions.
     * @param shape_out Array to store the output shape.
     * @param data_out Pointer to store the allocated output data buffer.
     * @return 0 on success, -1 on failure.
     */
    int readDataByIndex(
        const char* full_data_path,  // ✅ Full path: "profiles_2d/1/ion/0/state/0/z_min"
        int64_t time_index,
        uint64_t* ndim_out,
        uint64_t shape_out[6],
        double** data_out);

    /**
     * @brief C-style API to read string data by path and time index.
     * @param full_data_path Full path to the string signal (e.g. "profiles_2d/1/label").
     * @param time_index     Time index to read, or -1 for static / all-slices aggregation.
     * @param ndim_out       [out] Number of output dimensions.
     * @param shape_out      [out] Output shape (up to 6 dimensions).
     * @param data_out       [out] Newly allocated buffer of null-terminated strings (caller frees).
     * @return 0 on success, -1 on failure.
     */
     int readStringDataByIndex(
        const char* full_data_path,
        int64_t time_index,
        uint64_t* ndim_out,
        uint64_t shape_out[6],
        char** data_out);

    /**
     * @brief C-style API to read complex data by path and time index.
     * @param full_data_path Full path to the complex signal.
     * @param time_index     Time index to read, or -1 for static / all-slices aggregation.
     * @param ndim_out       [out] Number of output dimensions.
     * @param shape_out      [out] Output shape (up to 6 dimensions).
     * @param data_out       [out] Newly allocated buffer of std::complex<double> (caller frees).
     * @return 0 on success, -1 on failure.
     */
    int readComplexDataByIndex(
        const char* full_data_path,
        int64_t time_index,
        uint64_t* ndim_out,
        uint64_t shape_out[6],
        std::complex<double>** data_out);

    /**
     * @brief C-style API to read 32-bit integer data by path and time index.
     * @param full_data_path Full path to the int32 signal.
     * @param time_index     Time index to read, or -1 for static / all-slices aggregation.
     * @param ndim_out       [out] Number of output dimensions.
     * @param shape_out      [out] Output shape (up to 6 dimensions).
     * @param data_out       [out] Newly allocated buffer of int32_t (caller frees).
     * @return 0 on success, -1 on failure.
     */
      int readIntDataByIndex(
        const char* full_data_path,
        int64_t time_index,
        uint64_t* ndim_out,
        uint64_t shape_out[6],
        int32_t** data_out);

    //==========================================================================
    // Read API - Metadata
    //==========================================================================

    /**
     * @brief Reads metadata associated with a dataset instance.
     * Checks if metadata for the corresponding schema has already been read to avoid redundancy.
     * @param instance_path The full path of the data instance (e.g. "profiles_1d/0/t_e")
     * @return A map of metadata key-values found (e.g. {"units": "eV"}).
     */
    std::map<std::string, std::string> readMetadata(const std::string& instance_path);

private:
    // Current open mode (read-only; see getOpenMode()).
    OpenMode mode = OpenMode::READ;

    /**
     * @brief Re-establishes the write-time context (aos_time_counters) from the
     *        on-disk leaves after opening an existing file (APPEND/READ).
     * @note Ensures that appending continues where the previous session stopped,
     *        including gap alignment on the enclosing time base.
     */
    void restoreTimeContext();

    /**
     * @brief Common initialization path shared by both constructors.
     * @param mode The open mode being set up (WRITE / READ / APPEND).
     */
    void init(OpenMode mode);

    /**
     * @brief Appends one 14-column row to the index buffer (no HDF5 I/O).
     * @param full_path    Full instance path of the node (written to /paths).
     * @param parent_path  Path of the parent node (reconstructed on read; not stored).
     * @param shape        Node shape (up to 6 dimensions).
     * @param type         Data type (DataType value; M1 keeps the always-zero value in row[0]).
     * @param time_idx     First time step of the row (0 for static data).
     * @param offset       Element offset inside the raw data dataset.
     * @param count        Number of stored elements.
     * @param flags        Node kind (low 4 bits) or metadata marker.
     * @param parent_id    Row id of the enclosing AoS meta node, or PANZER_NO_PARENT_ROW.
     * @param index_value  Instance index within that parent AoS (0 if absent).
     * @return The row id assigned to the appended row (stable parent-first order).
     */
    uint64_t append_index_row(const std::string& full_path, const std::string& parent_path,
                                 const std::vector<size_t>& shape, uint64_t type,
                                 uint64_t time_idx, uint64_t offset, uint64_t count, uint64_t flags,
                                 uint64_t parent_id, uint64_t index_value);

    /**
     * @brief Returns the path of the enclosing dynamic AoS, or an empty string.
     */
    std::string getDynamicAOSPath() const;

    /**
     * @brief Gets the current next slice index of a dynamic AoS (aos_time_counters).
     * @param aos_path Full path of the dynamic AoS.
     * @return The counter value.
     */
    uint64_t getCurrentTimeForAOS(const std::string& aos_path);

    /**
     * @brief Advances the aos_time_counters entry of a dynamic AoS by delta.
     * @param aos_path Full path of the dynamic AoS.
     * @param delta    Number of time steps to add.
     */
    void advanceTimeForAOS(const std::string& aos_path, uint64_t delta);

    /**
     * @brief Shared implementation of writeData<T>.
     * Resolves the target raw dataset and its buffer from dtype, grows the buffer,
     * appends `count` elements, appends the 14-column index row, and bumps
     * aos_time_counters.
     */
    template<typename T>
    void writeDataImpl(const std::string& name,
                       const std::vector<size_t>& shape,
                       const T* data,
                       size_t count,
                       const std::string& timebase,
                       DataType dtype,
                       hid_t dataset_id,
                       std::vector<T>& buffer);

    /**
     * @brief Shared implementation of writeDataSlices<T>.
     * Computes the base time from aos_time_counters, appends the slices to the raw
     * dataset buffer and one index row, then advances the counters.
     */
    template<typename T>
    void writeDataSlicesImpl(const std::string& name,
                             const std::vector<size_t>& base_shape,
                             const T* data,
                             size_t n_slices,
                             const std::string& timebase,
                             DataType dtype,
                             hid_t dataset_id,
                             std::vector<T>& buffer);

};

// Declarations of explicit specializations OUTSIDE the class
template<>
void PanzerDB::writeData<double>(const std::string& name,
                       const std::vector<size_t>& shape,
                       const double* data, size_t count,
                       const std::string& timebase);

template<>
void PanzerDB::writeData<int32_t>(const std::string& name,
                     const std::vector<size_t>& shape,
                     const int32_t* data, size_t count,
                     const std::string& timebase);

template<>
void PanzerDB::writeData<std::complex<double>>(const std::string& name,
                     const std::vector<size_t>& shape,
                     const std::complex<double>* data, size_t count,
                     const std::string& timebase);

template<>
void PanzerDB::writeData<char>(const std::string& name,
                     const std::vector<size_t>& shape,
                     const char* data,  
                     size_t count,
                     const std::string& timebase);

template<>
void PanzerDB::writeData<const char*>(const std::string& name,
                     const std::vector<size_t>& shape,
                     const char* const* data,
                     size_t count,
                     const std::string& timebase);
