// panzerdb.h ULTIMATE LIGHTWEIGHT VERSION (without index_names)
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
#include <list>
#include <map>

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
    size_t index_chunk_rows = 8192;           ///< Number of rows per chunk in the main index table.
    
    // Data chunking (by type)
    size_t data_chunk_f64 = 131072;           ///< Chunk size in elements for float64 datasets (e.g., 1MB per chunk).
    size_t data_chunk_i32 = 262144;           ///< Chunk size in elements for int32 datasets (e.g., 1MB per chunk).
    size_t data_chunk_c128 = 65536;           ///< Chunk size in elements for complex128 datasets (e.g., 1MB per chunk).
    size_t data_chunk_str = 8192;             ///< Chunk size in elements for string datasets.
    
    // Path chunking
    size_t path_chunk_entries = 8192;         ///< Number of entries per chunk for path datasets.
    
    // Compression settings
    bool enable_compression = true;           ///< If true, enables GZIP compression.
    int compression_level = 6;                ///< GZIP compression level (0-9). 6 is a good balance.
    
    // Cache settings (HDF5 raw data chunk cache)
    size_t metadata_cache_size = 16 * 1024 * 1024;  ///< HDF5 metadata cache size (not directly used by PanzerDB cache config).
    size_t chunk_cache_size = 64 * 1024 * 1024;     ///< Total size of the raw data chunk cache.
    size_t chunk_cache_nslots = 10007;              ///< Number of chunk slots in the cache (prime number is good for hashing).
};

/**
 * @struct ArrayLevel
 * @brief Represents a level in the nested Array of Structures (AoS) hierarchy.
 *
 * When navigating into an AoS using `beginArray`, an `ArrayLevel` is pushed
 * onto an internal stack to track the current position, name, and properties
 * of the array.
 */
struct ArrayLevel {
    std::string name;                   ///< The name of this AoS level.
    std::string saved_path_prefix;      ///< The path prefix before entering this level.
    std::string timebase;               ///< The name of the timebase if this is a dynamic AoS.
    bool is_dynamic = false;            ///< True if this is a dynamic (time-evolving) AoS.
    size_t declared_size = 0;           ///< The declared size for a static AoS.
    size_t current_index = 0;           ///< The current index within this AoS.
    size_t actual_count = 0;            ///< The number of children written to.
    bool had_write = false;             ///< Flag to track if any data was written in this level.
    std::string aos_full_path;          ///< The full, unique path to this AoS instance.
};

/**
 * @struct TimeRangeP
 * @brief Represents a time range [start, end) for indexing.
 */
struct TimeRangeP {
    uint64_t start;
    uint64_t end;
    
    bool contains(uint64_t t) const {
        return t >= start && t < end;
    }
    
    bool operator<(const TimeRangeP& other) const {
        if (start != other.start) return start < other.start;
        return end < other.end;
    }
};

/**
 * @struct PathComponents
 * @brief Helper struct to hold the parsed components of a data path within a dynamic AoS.
 */
struct PathComponents {
    std::string prefix;      // "profiles_2d"
    std::string index_str;   // "0"
    std::string suffix;      // "/ion/0/state/0/z_min"
    bool has_index;
    
    PathComponents() : has_index(false) {}
};

/**
 * @struct ChunkingStats
 * @brief Holds statistics about HDF5 chunking performance.
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
     * @enum OpenMode
     * @brief Defines how the PanzerDB file should be opened.
     */
    enum class OpenMode {
        WRITE,  ///< Create a new file, truncating if it exists.
        READ,   ///< Open an existing file for read-only access.
        APPEND  ///< Open an existing file for reading and writing.
    };

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
     * @struct Leaf
     * @brief Represents a terminal node (a data entry) in the PanzerDB index.
     *
     * Each leaf corresponds to one row in the main index table and holds all
     * metadata required to locate and interpret a piece of data.
     */
    struct Leaf {
        std::vector<size_t> shape;      ///< The shape of the data tensor.
        uint64_t time_index = 0;        ///< The starting time index for dynamic data.
        std::string_view path;          ///< The full path to the data node.
        std::string_view parent_path;   ///< The path of the parent node.
        uint64_t offset = 0;            ///< The starting offset in the raw data dataset.
        uint64_t count = 0;             ///< The number of elements in the raw data dataset.
        uint64_t flags = 0;             ///< Flags: 0=normal, 1=empty, 2=static AoS, 3=dynamic AoS.
        bool is_empty = false;          ///< Convenience flag, true if flags == 1.
    };

    //==========================================================================
    // Lifecycle and Core Methods
    //==========================================================================

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

    /**
     * @brief Destructor. Flushes any remaining data and closes the file.
     */
    ~PanzerDB();

    /**
     * @brief Flushes all in-memory write buffers to the HDF5 file and closes all handles.
     */
    void close();

    /**
     * @brief Flushes all in-memory write buffers to the HDF5 file.
     * This makes the written data visible to other readers without closing the file.
     */
    void flush();

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

    OpenMode mode;

private:
    
    hid_t file_id = -1;
    hid_t index_dset = -1;
    hid_t paths_dset = -1;
    hid_t parent_paths_dset = -1;

    // Datasets for each data type
    hid_t data_dset_f64 = -1; // double
    hid_t data_dset_i32 = -1; // int64_t
    hid_t data_dset_c128 = -1; // complex
    hid_t data_dset_str = -1; // string

    // Cache for dataset sizes to avoid H5Dget_space calls
    hsize_t disk_size_f64 = 0;
    hsize_t disk_size_i32 = 0;
    hsize_t disk_size_c128 = 0;
    hsize_t disk_size_str = 0;
    
    void updateDiskSizes();

    //uint64_t global_time = 0;
    std::unordered_map<std::string, uint64_t> aos_time_counters;
    std::string dynamic_level;
    std::vector<std::string> current_path;
    std::string path_prefix;
    std::vector<ArrayLevel> array_stack;

    bool preserve_empty_nodes = false;
    bool last_level_had_write = false;
    bool should_close_loc_id = false;

    // RAM Buffers (only what is necessary)
    std::vector<uint64_t> index_buffer;   // 14 columns: type, ndim, shape[6], time_index, offset, count, flags
    std::vector<double> data_buffer_f64;
    std::vector<int32_t> data_buffer_i32;
    std::vector<std::complex<double>> data_buffer_c128;
    std::vector<std::string> data_buffer_str;

    std::vector<char> paths_buffer;       // Buffer for plain text paths (fixed size)
    std::vector<char> parent_paths_buffer;

     // New constants for memory management
    static constexpr size_t INITIAL_INDEX_BUFFER_SIZE = 1000;      // 112 KB instead of 11 MB
    static constexpr size_t INITIAL_DATA_BUFFER_SIZE = 8192;       // 64 KB instead of 1 MB
    static constexpr size_t INITIAL_STRING_BUFFER_SIZE = 512;      // 4 KB instead of 128 KB
    static constexpr size_t BUFFER_GROWTH_FACTOR = 2;              // Double when full
    
    // Thresholds for automatic flush
    static constexpr size_t AUTO_FLUSH_THRESHOLD = 100'000'000;    // 100 MB
    static constexpr size_t CRITICAL_FLUSH_THRESHOLD = 500'000'000; // 500 MB

    // Cache to avoid reconstructions
    mutable std::string cached_path_prefix;
    mutable bool path_prefix_dirty = true;
    
    void rebuildPathPrefix();
    std::string buildPathFromStack(const std::vector<ArrayLevel>& stack_vector) const;

    mutable std::string cached_dynamic_aos_path;
    mutable bool dynamic_aos_path_valid = false;
    
    void invalidateDynamicAOSCache();
    
    size_t getTotalBufferSize() const;
    void autoFlushIfNeeded();
    void growBufferIfNeeded(size_t required_space);
    void flushComplexBuffer();
    void flushStringBuffer() ;

    void flushPathBuffer(hid_t dataset_id, std::vector<char>& buffer);

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
    mutable std::map<TimeRangeP, size_t> time_range_index;

    // NEW: Reusable scratch buffers to avoid repetitive malloc/free during read
    mutable std::vector<double> scratch_f64;
    mutable std::vector<int32_t> scratch_i32; // Not used in current code but ready
    mutable std::vector<std::complex<double>> scratch_c128;
    mutable std::vector<std::string> scratch_str;

    // NEW: Optimized time index
    mutable bool time_index_valid = false;
    
    // NEW: Metadata cache to avoid recalculations
    struct LeafMetadata {
        size_t slice_volume;
        size_t n_time_steps;
        TimeRangeP time_range;
    };
    mutable std::vector<LeafMetadata> leaf_metadata_cache;
    
    void buildTimeIndex() const;
    const Leaf* findLeafByTime(const std::string& path, int64_t time_index) const;

    PathComponents parsePath(const std::string& path, 
                                   const std::string& aos_path) const;

    ChunkingConfig chunk_config;
    
    void configureChunking(const std::string& usage_hint);
    hid_t createOptimizedDataset(const std::string& name,
                                  hid_t type,
                                  size_t chunk_size,
                                  bool enable_compression,
                                  hid_t dapl = H5P_DEFAULT);

    void readChunkingConfig();
    void configureReadCache();
   



public:
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
     * @brief Ends the current AoS level and pops it from the internal stack.
     */
    void endArray();

    /**
     * @brief Manually increments the index of the current AoS level.
     */
    void beginArray(ArrayLevel& level);

    /**
     * @brief Manually sets the index of the current AoS level.
     * @param new_index The new index to set.
     */
    void incrementArrayIndex();

    void setCurrentArrayIndex(size_t new_index);

    //==========================================================================
    // Write API - Data
    //==========================================================================

    /**
     * @brief Writes a static (non-time-dependent) data tensor.
     * @tparam T The data type (e.g., double, int32_t, std::complex<double>).
     * @param name The name of the data node.
     * @param shape The dimensions of the tensor.
     * @param data A pointer to the data buffer.
     * @param count The total number of elements in the buffer.
     * @param timebase (Unsupported for this function, must be empty).
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
     */
    void writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const int32_t* data, size_t n_slices,
                               const std::string& timebase);

    /**
     * @brief Writes one or more time slices of a dynamic complex double signal.
     */
    void writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const std::complex<double>* data, size_t n_slices,
                               const std::string& timebase);

    /**
     * @brief Writes one or more time slices of a dynamic string signal.
     */
    void writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const char* const* data, size_t n_slices,
                               const std::string& timebase);

    //==========================================================================
    // Read API - Metadata and Index
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
     * @brief Finds the index in a time base vector that is closest to a requested time.
     * @param timebase_path The full path to the 1D dataset representing the time base.
     * @param requested_time The time value to search for.
     * @return The index of the element in the time base closest to the requested time.
     */
    int64_t getTimeIndex(const std::string& timebase_path, double requested_time) const;

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
     * @brief Reads a single scalar value by its full path.
     * @tparam T The scalar type.
     * @param path The full path to the scalar node.
     * @param status Output parameter, set to 0 on success, -1 on failure.
     * @return The read scalar value.
     */
    template<typename T>
    T readScalar(const std::string& path, int *status) const;

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
     * @brief Reads multiple contiguous data chunks in a single HDF5 operation.
     * @tparam T The data type.
     * @param leaves A vector of Leaf pointers that are contiguous in the raw data file.
     * @param output A pre-allocated buffer to hold the combined data.
     * @return 0 on success, -1 on failure.
     */
    template<typename T>
    int readMultipleSlices(const std::vector<const Leaf*>& leaves, T* output) const;

    /**
     * @brief Reads a collection of leaves into a contiguous buffer using HDF5's Hyperslab Union.
     * This is efficient for reading multiple, potentially non-contiguous, time slices of a signal at once.
     * @param leaves Vector of leaves to read (must be sorted by desired output order, typically time).
     * @param buffer Output buffer (pre-allocated).
     * @param dtype The data type of the leaves.
     * @return 0 on success, -1 on failure.
     */
    int readLeavesUnion(const std::vector<const Leaf*>& leaves, void* buffer, DataType dtype) const;

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
     * @brief C-style API to read double-precision data by path and time index.
     * Handles static data (time_index = -1), single time slices, and path substitution for dynamic AoS.
     * @param full_data_path The full path to the data node (e.g., "A/0/B/signal").
     * @param time_index The time index to read, or -1 for static data or to aggregate all time slices.
     * @param ndim_out Pointer to store the number of dimensions.
     * @param shape_out Array to store the output shape.
     * @param data_out Pointer to store the allocated output data buffer.
     * @return 0 on success, -1 on failure.
     */
    int pz_readData_by_index(
        const char* full_data_path,
        int64_t time_index,
        uint64_t* ndim_out,
        uint64_t shape_out[6],
        double** data_out);

    /**
     * @brief C-style API to read string data by path and time index.
     */
    int pz_readStringData_by_index(
        const char* full_data_path,
        int64_t time_index,
        uint64_t* ndim_out,
        uint64_t shape_out[6],
        char** data_out);

    /**
     * @brief C-style API to read complex data by path and time index.
     */
    int pz_readComplexData_by_index(
        const char* full_data_path,
        int64_t time_index,
        uint64_t* ndim_out,
        uint64_t shape_out[6],
        std::complex<double>** data_out);

    //==========================================================================
    // State and Synchronization
    //==========================================================================

    /**
     * @brief Synchronizes the internal PanzerDB array stack with an external context.
     * This is crucial for correctly positioning writes within nested AoS when the
     * context is managed externally.
     * @param aos_names A vector of AoS names in hierarchical order.
     * @param indices A vector of corresponding indices for each AoS level.
     */
    void synchronizeArrayStack(const std::vector<std::string>& aos_names, 
                               const std::vector<int>& indices);

    /**
     * @brief Checks if the current write position is inside a dynamic AoS.
     * @param timebase If inside a dynamic AoS, this string is filled with the name of the timebase.
     * @return True if inside a dynamic AoS, false otherwise.
     */
    bool isInsideDynamicAOS(std::string* timebase) const;

    /**
     * @brief Gets the current open mode of the database.
     * @return The current OpenMode (READ, WRITE, or APPEND).
     */
    OpenMode getOpenMode() const { return mode; }

    /**
     * @brief Gets the current depth of the AoS stack.
     * @return The number of active `beginArray` calls.
     */
    size_t getArrayStackSize() const { return array_stack.size(); }

    /**
     * @brief Checks if the AoS stack is empty.
     * @return True if not inside any AoS, false otherwise.
     */
    bool isArrayStackEmpty() const { return array_stack.empty(); }

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
    void printChunkingStats() const;

    /**
     * @brief Prints the content of the cached index table (leaves) to standard output for debugging.
     */
    void dumpLeavesCache() const;

    //==========================================================================
    // Deprecated or Internal-Use-Only
    //==========================================================================
    
    uint64_t getTimeBaseLength(const std::string& timebase_name) const;
    uint64_t getLastTimeIndex(const std::string& data_path) const;
    bool isTimeInLeaf(const Leaf& leaf, int64_t time_index) const;
    size_t getCurrentTotalSize(const std::string& level_name) const;
    void dumpIndexBuffer() const;
    void advanceTimebase(const std::string& timebase_name, uint64_t n_steps = 1);
    std::vector<double> getWholeDynamicSignal(const std::string& dataset_name);

private:
    template<typename T>
    void writeDataImpl(const std::string& name,
                              const std::vector<size_t>& shape,
                              const T* data,
                              size_t count,
                              const std::string& timebase,
                              DataType dtype,
                              hid_t dataset_id,
                              std::vector<T>& buffer);
    
    template<typename T>
    void writeDataSlicesImpl(const std::string& name,
                                    const std::vector<size_t>& base_shape,
                                    const T* data,
                                    size_t n_slices,
                                    const std::string& timebase,
                                    DataType dtype,
                                    hid_t dataset_id,
                                    std::vector<T>& buffer);
    void restoreTimeContext();
    void init(OpenMode mode);
    void append_index_row(const std::string& full_path, const std::string& parent_path,
                                const std::vector<size_t>& shape, uint64_t type,
                                uint64_t time_idx, uint64_t offset, uint64_t count, uint64_t flags);
    // Helper to get the path of the dynamic parent AOS
    std::string getDynamicAOSPath() const;
    uint64_t getCurrentTimeForAOS(const std::string& aos_path);
    void advanceTimeForAOS(const std::string& aos_path, uint64_t delta);
    std::string findDynamicAOSParent(const std::string& parent_path,
                                           const std::vector<Leaf>& leaves);
    
};

// Explicit template specializations for writeData
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