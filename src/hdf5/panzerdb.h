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
#include <unordered_set>
#include <list>
#include <map>
#include "metadata/metadata_extractor.h"

// Ajout pour la nouvelle méthode get_leaf_type
#include "direct_access_api.h"

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
    size_t metadata_cache_size = 16 * 1024 * 1024;  // 16 MB
    size_t chunk_cache_size = 64 * 1024 * 1024;     // 64 MB
    size_t chunk_cache_nslots = 10007;               // Prime number for hash
};

struct ArrayLevel {
    std::string name;
    std::string saved_path_prefix;
    std::string timebase;
    bool is_dynamic = false;
    size_t declared_size = 0;
    size_t current_index = 0;
    size_t actual_count = 0;
    bool had_write = false;
    
    // Full path of the AoS (for lookup in aos_time_counters)
    std::string aos_full_path;
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
     * @struct Leaf
     * @brief Represents a terminal node (a data entry) in the PanzerDB index.
     *
     * Each leaf corresponds to one row in the main index table and holds all
     * metadata required to locate and interpret a piece of data.
     */
    struct Leaf {
        std::vector<size_t> shape;
        uint64_t time_index = 0;
        std::string_view path; // OPTIMIZATION: Zero-copy view into cached_paths_blob
        std::string_view parent_path;
        uint64_t offset = 0;
        uint64_t count = 0;
        uint64_t flags = 0;  // 0=normal, 1=empty, 2=static AoS meta-node, 3=dynamic AoS meta-node
        bool is_empty = false; // Helper for compatibility
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
    
    void updateDiskSizes();

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

     // Constants for buffer management
    static constexpr size_t INITIAL_INDEX_BUFFER_SIZE = 1000;      
    static constexpr size_t INITIAL_DATA_BUFFER_SIZE = 8192;       
    static constexpr size_t INITIAL_STRING_BUFFER_SIZE = 512;      
    static constexpr size_t BUFFER_GROWTH_FACTOR = 2;              
    
    // Cache to avoid reconstructions
    mutable std::string cached_path_prefix;
    mutable bool path_prefix_dirty = true;
    
    void rebuildPathPrefix();
    std::string buildPathFromStack(const std::vector<ArrayLevel>& stack_vector) const;

    mutable std::string cached_dynamic_aos_path;
    mutable bool dynamic_aos_path_valid = false;
    
    void invalidateDynamicAOSCache();
    
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

    // For each dynamic AoS root: the highest time_index found among ALL of its
    // descendants (any depth), built in getLeaves(). Used by getAOSShape so a
    // dynamic AoS that only contains nested static AoS (e.g. time_slice/ggd/
    // theta/values) still reports the correct size.
    mutable std::unordered_map<std::string_view, uint64_t> max_time_at_dynamic_root;

    // Reusable scratch buffers to avoid repetitive malloc/free on reads
    mutable std::vector<double> scratch_f64;
    mutable std::vector<int32_t> scratch_i32; // Not used in current code but ready
    mutable std::vector<std::complex<double>> scratch_c128;
    mutable std::vector<std::string> scratch_str;

    std::unordered_set<std::string> written_metadata_schema_paths;
    std::map<std::string, std::string> metadata_map;

    // Rebuilds max_time_at_dynamic_root from the currently cached leaves:
    // for every dynamic AoS root, the max time_index of ALL descendant
    // data leaves (any depth). Called once per cache (re)build.
    void rebuildDynamicRootTimeIndex() const;

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

    OpenMode mode;

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
     * @brief Manually increments the index of the current AoS level.
     */
    void beginArray(ArrayLevel& level);

    /**
     * @brief Manually sets the index of the current AoS level.
     * @param new_index The new index to set.
     */
    void incrementArrayIndex();
    void setCurrentArrayIndex(size_t new_index);

    /**
     * @brief Checks if the current write position is inside a dynamic AoS.
     * @param timebase If inside a dynamic AoS, this string is filled with the name of the timebase.
     * @return True if inside a dynamic AoS, false otherwise.
     */
    bool isInsideDynamicAOS(std::string* timebase) const;


    uint64_t getTimeBaseLength(const std::string& timebase_name) const;
    uint64_t getLastTimeIndex(const std::string& data_path) const;

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
     * @brief Writes one or more time slices of a dynamic complex signal.
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
                              
    template<typename T>
    void writeDataSlicesImpl(const std::string& name,
                                    const std::vector<size_t>& base_shape,
                                    const T* data,
                                    size_t n_slices,
                                    const std::string& timebase,
                                    DataType dtype,
                                    hid_t dataset_id,
                                    std::vector<T>& buffer);
 
    
     /**
     * @brief Prints the content of the cached index table (leaves) to standard output for debugging.
     */
    void dumpLeavesCache() const;
    void endArray();

     /**
     * @brief Flushes all in-memory write buffers to the HDF5 file.
     * This makes the written data visible to other readers without closing the file.
     */
    void flush();

     /**
     * @brief Flushes all in-memory write buffers to the HDF5 file and closes all handles.
     */
    void close();

    //==========================================================================
    // Read API - Metadata and Index
    //==========================================================================

    /**
     * @brief Retrieves the entire database index as a vector of Leaf objects.
     * The result is cached for subsequent calls. This is the primary entry point for read operations.
     * @return A constant reference to the cached vector of leaves.
     */
    const std::vector<Leaf>& getLeaves() const;
    //const std::vector<std::string>& getDynamicAOSRoots() const { return cached_dynamic_aos_roots; }

     /**
     * @brief Gets the effective size of an Array of Structures.
     * For static AoS, it returns the declared size. For dynamic AoS, it returns the number of time steps written.
     * @param level_name The full path to the AoS meta-node (e.g., "profiles_1d" or "profiles_1d/0/ion").
     * @return A vector containing the size of the AoS.
     */
    std::vector<size_t> getAOSShape(const std::string& level_name) const;
    size_t getDynamicAOSSize(const std::string& aos_path) const;

    /**
     * @brief Finds the index in a time base vector that is closest to a requested time.
     * @param timebase_path The full path to the 1D dataset representing the time base.
     * @param requested_time The time value to search for.
     * @return The index of the element in the time base closest to the requested time.
     */
    int64_t getTimeIndex(const std::string& timebase_path, double requested_time, int interp_mode) const;


    bool isTimeInLeaf(const Leaf& leaf, int64_t time_index) const;

    /**
     * @brief Gets the current open mode of the database.
     * @return The current OpenMode (READ, WRITE, or APPEND).
     */
    OpenMode getOpenMode() const { return mode; }

    /**
     * @brief Détermine le type de donnée d'une feuille (dataset) à partir de son chemin.
     * @param path Le chemin complet vers le dataset dans le fichier HDF5.
     * @return Le type de la donnée sous forme d'enum DataType.
     */
    imas::direct_access::DataType get_leaf_type(const std::string& path);

    /**
     * @brief Checks if a given path corresponds to a dynamic Array of Structures (AoS).
     * @param aos_path The full path to the AoS meta-node (e.g., "profiles_1d").
     * @return True if the path points to a dynamic AoS, false otherwise.
     */
     bool is_dynamic_aos(const std::string& aos_path) const;

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
     * @brief Reads multiple contiguous data chunks in a single HDF5 operation.
     * @tparam T The data type.
     * @param leaves A vector of Leaf pointers that are contiguous in the raw data file.
     * @param output A pre-allocated buffer to hold the combined data.
     * @return 0 on success, -1 on failure.
     */
    template<typename T>
    int readMultipleSlices(const std::vector<const Leaf*>& leaves, T* output) const;

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
        const char* full_data_path,  // ✅ Full path: "profiles_2d/1/ion/0/state/0/z_min"
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

    /**
     * @brief C-style API to read 32-bit integer data by path and time index.
     */
     int pz_readIntData_by_index(
        const char* full_data_path,
        int64_t time_index,
        uint64_t* ndim_out,
        uint64_t shape_out[6],
        int32_t** data_out);

    void advanceTimebase(const std::string& timebase_name, uint64_t n_steps = 1);
    std::vector<double> getWholeDynamicSignal(const std::string& dataset_name);

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
     * @brief Utility to convert an instance path (e.g. "A/0/B/signal") to a schema path ("A/B/signal").
     * Removes all purely numeric path segments. Useful for retrieving shared metadata.
     */
    static std::string stripIndices(const std::string& path);

    /**
     * @brief Writes a set of metadata for a given path.
     * @param path The base path (e.g., "profiles_1d/t_e")
     * @param metadata_map Map containing {metadata_name, value} pairs
     */
    void writeMetaData(const std::string& path, const std::string& value);
    

    /**
     * @brief Reads metadata associated with a dataset instance.
     * Checks if metadata for the corresponding schema has already been read to avoid redundancy.
     * @param instance_path The full path of the data instance (e.g. "profiles_1d/0/t_e")
     * @return A map of metadata key-values found (e.g. {"units": "eV"}).
     */
    std::map<std::string, std::string> readMetadata(const std::string& instance_path);

private:
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
