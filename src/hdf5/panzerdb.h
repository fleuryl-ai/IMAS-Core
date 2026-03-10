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

// ============================================================================
// 1. CONFIGURATION DYNAMIQUE DU CHUNKING
// ============================================================================

struct ChunkingConfig {
    // Index table chunking
    size_t index_chunk_rows = 8192;           // Nombre de rows par chunk
    
    // Data chunking (par type)
    size_t data_chunk_f64 = 131072;           // 1 MB par chunk (131k doubles)
    size_t data_chunk_i32 = 262144;           // 1 MB par chunk (262k int32)
    size_t data_chunk_c128 = 65536;           // 1 MB par chunk (65k complex)
    size_t data_chunk_str = 8192;             // 8k strings par chunk
    
    // Path chunking
    size_t path_chunk_entries = 8192;         // 8k paths par chunk
    
    // Compression settings
    bool enable_compression = true;
    int compression_level = 6;                // 0-9, 6 est un bon compromis
    
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
    
    // ✅ Full path of the AoS (for lookup in aos_time_counters)
    std::string aos_full_path;
};

// ============================================================================
// OPTIMISATIONS PERFORMANCE LECTURE - PANZERDB
// ============================================================================

// 1. NOUVEAU: Index temporel pour recherche O(log n)
// ============================================================================

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

struct PathComponents {
    std::string prefix;      // "profiles_2d"
    std::string index_str;   // "0"
    std::string suffix;      // "/ion/0/state/0/z_min"
    bool has_index;
    
    PathComponents() : has_index(false) {}
};

struct ChunkingStats {
    size_t total_chunks_written = 0;
    size_t total_chunks_read = 0;
    size_t total_bytes_written = 0;
    size_t total_bytes_read = 0;
    size_t compression_ratio = 100;  // Pourcentage (100 = pas de compression)
};

class PanzerDB {
public:
    // Read
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

     // Nouvelles constantes pour gestion mémoire
    static constexpr size_t INITIAL_INDEX_BUFFER_SIZE = 1000;      // 112 KB au lieu de 11 MB
    static constexpr size_t INITIAL_DATA_BUFFER_SIZE = 8192;       // 64 KB au lieu de 1 MB
    static constexpr size_t INITIAL_STRING_BUFFER_SIZE = 512;      // 4 KB au lieu de 128 KB
    static constexpr size_t BUFFER_GROWTH_FACTOR = 2;              // Doubler quand plein
    
    // Seuils pour flush automatique
    static constexpr size_t AUTO_FLUSH_THRESHOLD = 100'000'000;    // 100 MB
    static constexpr size_t CRITICAL_FLUSH_THRESHOLD = 500'000'000; // 500 MB

    // Cache pour éviter reconstructions
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

    // NOUVEAU: Scratch buffers réutilisables pour éviter malloc/free répétitifs en lecture
    mutable std::vector<double> scratch_f64;
    mutable std::vector<int32_t> scratch_i32; // Pas utilisé dans le code actuel mais prêt
    mutable std::vector<std::complex<double>> scratch_c128;
    mutable std::vector<std::string> scratch_str;

    // NOUVEAU: Index temporel optimisé
    mutable bool time_index_valid = false;
    
    // NOUVEAU: Cache de métadonnées pour éviter recalculs
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
    void setChunkingHint(const std::string& hint);
    void setCompressionLevel(int level);
    void disableCompression();

    ChunkingStats getChunkingStats() const;
    void printChunkingStats() const ;



public:
    enum class OpenMode { WRITE, READ, APPEND };
    enum class DataType : uint64_t {
        FLOAT64 = 0,
        INT32 = 1,
        COMPLEX128 = 2,
        STRING = 3,
        LIST_OF_STRINGS = 4,
        UNKNOWN = 99
    };

    PanzerDB(const std::string& filename, OpenMode mode, bool preserve_empty_nodes = false);
    PanzerDB(hid_t loc_id, OpenMode mode, bool preserve_empty_nodes = false, bool close_loc_id_on_exit = false);

    // Disable copy to prevent accidental closure of HDF5 handles by temporary copies
    PanzerDB(const PanzerDB&) = delete;
    PanzerDB& operator=(const PanzerDB&) = delete;

    ~PanzerDB();

    OpenMode mode;

    void beginArray(const std::string& name, size_t size);
    void beginArray(const std::string& name, const std::string& timebase);
    void beginArray(ArrayLevel& level);

    void incrementArrayIndex();
    void setCurrentArrayIndex(size_t new_index);
    bool isInsideDynamicAOS(std::string* timebase) const;
    uint64_t getTimeBaseLength(const std::string& timebase_name) const;
    uint64_t getLastTimeIndex(const std::string& data_path) const;

    // 1. Declaration of the generic template writeData<T>
    // Note the absence of the function body; it will be defined in the .cpp
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

  
    void writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const double* data, size_t n_slices,
                               const std::string& timebase);

    // int32_t - uses data_dset_i32 and data_buffer_i32
    void writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const int32_t* data, size_t n_slices,
                               const std::string& timebase);

    // complex - uses data_dset_c128 and data_buffer_c128
    void writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const std::complex<double>* data, size_t n_slices,
                               const std::string& timebase);

    void writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const char* const* data, size_t n_slices,
                               const std::string& timebase);
                              
    // 1. Declaration of the generic template writeDataSlices<T>
    template<typename T>
    void writeDataSlicesImpl(const std::string& name,
                                    const std::vector<size_t>& base_shape,
                                    const T* data,
                                    size_t n_slices,
                                    const std::string& timebase,
                                    DataType dtype,
                                    hid_t dataset_id,
                                    std::vector<T>& buffer);

    // VERSION WITH root_aos_name to handle multiple root hierarchies
    // In panzerdb.h
    int pz_readData_by_index(
        const char* full_data_path,  // ✅ Full path: "profiles_2d/1/ion/0/state/0/z_min"
        int64_t time_index,
        uint64_t* ndim_out,
        uint64_t shape_out[6],
        double** data_out);

    int pz_readStringData_by_index(
        const char* full_data_path,
        int64_t time_index,
        uint64_t* ndim_out,
        uint64_t shape_out[6],
        char** data_out);

    int pz_readComplexData_by_index(
        const char* full_data_path,
        int64_t time_index,
        uint64_t* ndim_out,
        uint64_t shape_out[6],
        std::complex<double>** data_out);

    int readInterpolatedData(
                         const char* full_data_path,  // ✅ Changed
                         double time,
                         const std::vector<double>& time_basis,
                         int interp_mode,
                         int datatype,
                         uint64_t* ndim_out,
                         uint64_t shape_out[6],
                         void** data_out,
                         bool expect_time_dim = true);

    void dumpIndexBuffer() const;                 
    void dumpLeavesCache() const;
    void endArray();
    void flush();
    void close();

    const std::vector<Leaf>& getLeaves() const;
    const std::vector<std::string>& getDynamicAOSRoots() const { return cached_dynamic_aos_roots; }
    std::vector<size_t> getAOSShape(const std::string& level_name) const;
    size_t getCurrentTotalSize(const std::string& level_name) const;
    size_t getDynamicAOSSize(const std::string& aos_path) const;
    int64_t getTimeIndex(const std::string& timebase_path, double requested_time, int interp_mode) const;
    bool isTimeInLeaf(const Leaf& leaf, int64_t time_index) const;
    //uint64_t getCurrentTime() const { return global_time; }

    
    OpenMode getOpenMode() const { return mode; }

    template<typename T>
    void readTensor(const Leaf& leaf, T* out_buffer) const;

    template<typename T>
    int readSliceDirect(const Leaf& leaf, int64_t time_index, T* out_buffer) const;

    template<typename T>
    int readMultipleSlices(const std::vector<const Leaf*>& leaves, T* output) const;
    // Requested magic function
    /*template<typename T>
    void getSlice(double temps, std::vector<T>& data_out);*/

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

    // Function to read a scalar directly by its path
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

    void advanceTimebase(const std::string& timebase_name, uint64_t n_steps = 1);
    std::vector<double> getWholeDynamicSignal(const std::string& dataset_name);
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
     * @brief Checks if the array stack is empty
     */
    bool isArrayStackEmpty() const { return array_stack.empty(); }

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