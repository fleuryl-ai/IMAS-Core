#include "panzerdb.h"
#include "data_interpolation.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <string_view>
#include <sstream>
#include <string>
#include <map>

/*
 * ###############################################################################
 * # Important Points to Understand
 * ###############################################################################
 *
 * 1.  **Columnar Storage & Indexing**:
 *     - PanzerDB does not store data in a hierarchical HDF5 structure (groups/datasets).
 *       Instead, it flattens data into a few large 1D datasets based on type
 *       (e.g., `data_raw_f64` for doubles, `data_raw_i32` for integers).
 *     - A central `index` table stores metadata for every logical node (path, shape,
 *       time index, offset in the raw dataset). This allows for extremely fast
 *       writes (append-only) and flexible reads.
 *
 * 2.  **Dynamic vs. Static AoS (Array of Structures)**:
 *     - **Static AoS**: Fixed size, known at creation. The index in the path
 *       (e.g., `profiles_1d/0/ion`) is explicit.
 *     - **Dynamic AoS**: Time-evolving structures. The "time" dimension is implicit
 *       in the structure's growth. The code handles "time steps" by creating new
 *       entries in the index for the same path but with an incremented `time_index`.
 *
 * 3.  **Optimization Strategies**:
 *     - **Write Buffering**: Data is accumulated in memory (`data_buffer_f64`, etc.)
 *       and flushed to disk in large chunks to minimize HDF5 I/O overhead.
 *     - **Direct Hyperslab Reads**: `readSliceDirect` reads specific data slices
 *       directly from the HDF5 file into the user's buffer without intermediate copies.
 *     - **Batched Reads**: `readLeavesUnion` combines multiple non-contiguous data
 *       chunks (e.g., a time series scattered across the file) into a single HDF5
 *       read operation using `H5S_SELECT_OR`.
 *     - **Caching**: The `getLeaves` method caches the entire index table in memory.
 *
 * 4.  **Path Substitution**:
 *     - When reading dynamic data (e.g., `readDataByIndex`), the code must often
 *       translate a logical path like `A/0/B` (requested at time `t=1`) into the
 *       actual stored path `A/1/B`. This is handled by finding the "Dynamic AoS Root"
 *       and substituting the index.
 */

constexpr size_t PATH_MAX_LEN = 256;

// Fixed maximum bytes for a single string element in the data_raw_str column
// (variable-length -> fixed-width conversion, SWMR-safe like `paths`).
// A string value longer than (STRING_MAX_LEN - 1) bytes is rejected at write time.
constexpr size_t STRING_MAX_LEN = 512;

// PanzerDB constructor modification
PanzerDB::PanzerDB(const std::string& filename, OpenMode mode, bool preserve_empty)
    : preserve_empty_nodes(preserve_empty)
{
     // OPTIMIZATION: File Access Property List (FAPL) for massive writes
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    
    // 1. Alignment on file system blocks (e.g., Lustre stripe size = 4MB)
    // Any object > 4KB will be aligned on a 4MB boundary.
    H5Pset_alignment(fapl, 4096, 4 * 1024 * 1024);
    
    // 2. Use the latest HDF5 format (better chunk indexing)
    H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    
    // 3. Aggregate metadata in 2MB blocks to reduce IOPS
    H5Pset_meta_block_size(fapl, 2 * 1024 * 1024);

    if (mode == OpenMode::WRITE) {
        file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    } else if (mode == OpenMode::APPEND) {
        // Check if file exists
        if (H5Fis_hdf5(filename.c_str()) > 0) {
            // File exists, open in read/write
            file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, fapl);
        } else {
            // File does not exist, create it
            file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
            mode = OpenMode::WRITE; // Switch to WRITE mode for initialization
        }
    } else { // OpenMode::READ
        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, fapl);
    }
    H5Pclose(fapl);

    if (file_id < 0) throw std::runtime_error("Failed to open or create file: " + filename);
    init(mode);
}

PanzerDB::PanzerDB(hid_t loc_id, OpenMode mode, bool preserve_empty, bool close_loc_id_on_exit)
    : file_id(loc_id), preserve_empty_nodes(preserve_empty), should_close_loc_id(close_loc_id_on_exit)
{
    if (file_id < 0) throw std::runtime_error("Invalid loc_id provided to constructor.");
    // In this case, file_id is actually a loc_id (group)
    init(mode);
}

// init function modification
void PanzerDB::init(OpenMode mode) {
    this->mode = mode; // Ensure member mode is set

    // --- DEBUT DU PATCH: AUTO-DETECTION DU GROUPE RACINE ---
    if (mode == OpenMode::READ || mode == OpenMode::APPEND) {
        bool is_at_root = H5Lexists(file_id, "index", H5P_DEFAULT) > 0;
        if (!is_at_root) {
            H5G_info_t group_info;
            if (H5Gget_info(file_id, &group_info) >= 0) {
                for (hsize_t i = 0; i < group_info.nlinks; ++i) {
                    char name[256];
                    H5Lget_name_by_idx(file_id, ".", H5_INDEX_NAME, H5_ITER_INC, i, name, sizeof(name), H5P_DEFAULT);

                    // Tente d'ouvrir le lien comme un groupe, en supprimant temporairement les erreurs HDF5
                    H5E_auto2_t old_func;
                    void *old_client_data;
                    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
                    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
                    hid_t group_id = H5Gopen2(file_id, name, H5P_DEFAULT);
                    H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);

                    if (group_id >= 0) {
                        if (H5Lexists(group_id, "index", H5P_DEFAULT) > 0) {
                            // Found the right group (the one holding the index).
                            if (should_close_loc_id) {
                                if (H5Iget_type(file_id) == H5I_FILE) H5Fclose(file_id);
                                else H5Gclose(file_id);
                            }
                            file_id = group_id; // Use this group as the new root
                            should_close_loc_id = true; // Ensure this new handle will be closed
                            break; 
                        }
                        H5Gclose(group_id);
                    }
                }
            }
        }
    }
    // --- FIN DU PATCH ---

    std::string usage_hint = "time_series";  // default
    if (mode == OpenMode::READ) {
        usage_hint = "interactive";  // read mode favors fast access
    }

    configureChunking(usage_hint);

     // OPTIMIZATION: Create a DAPL (Dataset Access Property List) with a large cache
    // Applies to both creation (WRITE) and opening (APPEND/READ)
    hid_t dapl = H5Pcreate(H5P_DATASET_ACCESS);
    H5Pset_chunk_cache(dapl, chunk_config.chunk_cache_nslots, 
                       chunk_config.chunk_cache_size, 0.75);

    if (mode == OpenMode::WRITE) {
        // Delete existing datasets
        if (H5Lexists(file_id, "index", H5P_DEFAULT) > 0) H5Ldelete(file_id, "index", H5P_DEFAULT);
        if (H5Lexists(file_id, "data_raw_f64", H5P_DEFAULT) > 0) H5Ldelete(file_id, "data_raw_f64", H5P_DEFAULT);
        if (H5Lexists(file_id, "data_raw_i32", H5P_DEFAULT) > 0) H5Ldelete(file_id, "data_raw_i32", H5P_DEFAULT);
        if (H5Lexists(file_id, "data_raw_str", H5P_DEFAULT) > 0) H5Ldelete(file_id, "data_raw_str", H5P_DEFAULT);
        if (H5Lexists(file_id, "data_raw_c128", H5P_DEFAULT) > 0) H5Ldelete(file_id, "data_raw_c128", H5P_DEFAULT);
        if (H5Lexists(file_id, "paths", H5P_DEFAULT) > 0) H5Ldelete(file_id, "paths", H5P_DEFAULT);
        if (H5Lexists(file_id, "parent_paths", H5P_DEFAULT) > 0) H5Ldelete(file_id, "parent_paths", H5P_DEFAULT);
        
        // OPTIMIZATION: Create index dataset with optimal chunking
        hsize_t dims[2] = {0, 14};
        hsize_t maxdims[2] = {H5S_UNLIMITED, 14};
        hsize_t chunk[2] = {chunk_config.index_chunk_rows, 14};
        
        hid_t space = H5Screate_simple(2, dims, maxdims);
        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 2, chunk);
        
        // Compression for the index
        if (chunk_config.enable_compression) {
            H5Pset_deflate(plist, chunk_config.compression_level);
            H5Pset_shuffle(plist);  // Improves compression
        }
        
        H5Pset_fill_time(plist, H5D_FILL_TIME_NEVER);
        H5Pset_alloc_time(plist, H5D_ALLOC_TIME_INCR);
        
        index_dset = H5Dcreate2(file_id, "index", H5T_STD_U64LE, space, 
                                H5P_DEFAULT, plist, dapl);
        H5Pclose(plist);
        H5Sclose(space);
        
         // OPTIMIZATION: Create data datasets with optimal chunking
        data_dset_f64 = createOptimizedDataset("data_raw_f64", H5T_IEEE_F64LE, 
                                                chunk_config.data_chunk_f64, true, dapl);
        data_dset_i32 = createOptimizedDataset("data_raw_i32", H5T_STD_I32LE, 
                                                chunk_config.data_chunk_i32, true, dapl);
        
         // String dataset: FIXED-WIDTH, NUL-padded C1 (SWMR-safe; no vlen heap).
         // Element = STRING_MAX_LEN bytes; a value longer than STRING_MAX_LEN-1 is
         // rejected at write time (see flush()). Mirrors the `paths` column.
        hid_t str_type_fixed = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_type_fixed, STRING_MAX_LEN);
        H5Tset_strpad(str_type_fixed, H5T_STR_NULLPAD);
        H5Tset_cset(str_type_fixed, H5T_CSET_UTF8);
        data_dset_str = createOptimizedDataset("data_raw_str", str_type_fixed,
                                                chunk_config.data_chunk_str, false, dapl);
        H5Tclose(str_type_fixed);
        
        // Complex dataset
        hsize_t complex_dims[1] = {2};
        hid_t complex_tid = H5Tarray_create2(H5T_IEEE_F64LE, 1, complex_dims);
        data_dset_c128 = createOptimizedDataset("data_raw_c128", complex_tid, 
                                                 chunk_config.data_chunk_c128, true, dapl);
        H5Tclose(complex_tid);
        
        // Path datasets (fixed-length strings)
        hid_t str_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_type, PATH_MAX_LEN);
        H5Tset_strpad(str_type, H5T_STR_NULLPAD);
        
        paths_dset = createOptimizedDataset("paths", str_type,
                                            chunk_config.path_chunk_entries, true, dapl);
        H5Tclose(str_type);
        // M1: /parent_paths is no longer materialised; parent_path is reconstructed
        // on read from (parent_id, kind, full path).  See getLeaves().
        parent_paths_dset = H5I_INVALID_HID;
        
        // Optimized buffers
        index_buffer.reserve(chunk_config.index_chunk_rows * 14);
        data_buffer_f64.reserve(chunk_config.data_chunk_f64);
        data_buffer_i32.reserve(chunk_config.data_chunk_i32);
        data_buffer_str.reserve(chunk_config.data_chunk_str);
        data_buffer_c128.reserve(chunk_config.data_chunk_c128);
        
        disk_size_f64 = 0;
        disk_size_i32 = 0;
        disk_size_c128 = 0;
        disk_size_str = 0;
        
    } else if (mode == OpenMode::APPEND) {
        // Open existing datasets
        if (H5Lexists(file_id, "index", H5P_DEFAULT) > 0) index_dset = H5Dopen2(file_id, "index", dapl);
        // APPEND: continue row ids from the on-disk /index row count so new parent
        // references stay aligned with the order a later READ observes.
        next_row_id = 0;
        if (index_dset >= 0) {
            hid_t ispace = H5Dget_space(index_dset);
            hsize_t idims[2] = {0, 0};
            H5Sget_simple_extent_dims(ispace, idims, nullptr);
            H5Sclose(ispace);
            next_row_id = idims[0];
        }
        if (H5Lexists(file_id, "data_raw_f64", H5P_DEFAULT) > 0) data_dset_f64 = H5Dopen2(file_id, "data_raw_f64", dapl);
        if (H5Lexists(file_id, "data_raw_i32", H5P_DEFAULT) > 0) data_dset_i32 = H5Dopen2(file_id, "data_raw_i32", dapl);
        if (H5Lexists(file_id, "paths", H5P_DEFAULT) > 0) paths_dset = H5Dopen2(file_id, "paths", dapl);
        if (H5Lexists(file_id, "data_raw_str", H5P_DEFAULT) > 0) data_dset_str = H5Dopen2(file_id, "data_raw_str", dapl);
        if (H5Lexists(file_id, "data_raw_c128", H5P_DEFAULT) > 0) data_dset_c128 = H5Dopen2(file_id, "data_raw_c128", dapl);
        // M1: /parent_paths is reconstructed on read; an orphaned one is ignored.
        parent_paths_dset = H5I_INVALID_HID;
        leaves_cache_valid = false;
        
        // NOUVEAU: Lire la configuration de chunking du fichier existant
        readChunkingConfig();
        updateDiskSizes();
        
        // Optimized buffers
        index_buffer.reserve(chunk_config.index_chunk_rows * 14);
        data_buffer_f64.reserve(chunk_config.data_chunk_f64);
        data_buffer_i32.reserve(chunk_config.data_chunk_i32);
        data_buffer_str.reserve(chunk_config.data_chunk_str);
        data_buffer_c128.reserve(chunk_config.data_chunk_c128);
        
    } else { // OpenMode::READ
        if (file_id < 0) throw std::runtime_error("Failed to open file for reading");
        if (H5Lexists(file_id, "index", H5P_DEFAULT) > 0) index_dset = H5Dopen2(file_id, "index", H5P_DEFAULT);
        if (H5Lexists(file_id, "data_raw_f64", H5P_DEFAULT) > 0) data_dset_f64 = H5Dopen2(file_id, "data_raw_f64", H5P_DEFAULT);
        if (H5Lexists(file_id, "data_raw_i32", H5P_DEFAULT) > 0) data_dset_i32 = H5Dopen2(file_id, "data_raw_i32", H5P_DEFAULT);
        if (H5Lexists(file_id, "paths", H5P_DEFAULT) > 0) paths_dset = H5Dopen2(file_id, "paths", H5P_DEFAULT);
        if (H5Lexists(file_id, "data_raw_c128", H5P_DEFAULT) > 0) data_dset_c128 = H5Dopen2(file_id, "data_raw_c128", H5P_DEFAULT);
        if (H5Lexists(file_id, "data_raw_str", H5P_DEFAULT) > 0) data_dset_str = H5Dopen2(file_id, "data_raw_str", H5P_DEFAULT);
        // M1: /parent_paths is reconstructed on read; an orphaned one is ignored.
        parent_paths_dset = H5I_INVALID_HID;
        leaves_cache_valid = false;
        
        // NOUVEAU: Lire la configuration de chunking
        readChunkingConfig();
        
        // NOUVEAU: Configurer le cache HDF5 pour lectures optimales
        configureReadCache();

    }

    H5Pclose(dapl);
    restoreTimeContext();
}

void PanzerDB::updateDiskSizes() {
    if (data_dset_f64 >= 0) {
        hid_t space = H5Dget_space(data_dset_f64);
        H5Sget_simple_extent_dims(space, &disk_size_f64, NULL);
        H5Sclose(space);
    }
    if (data_dset_i32 >= 0) {
        hid_t space = H5Dget_space(data_dset_i32);
        H5Sget_simple_extent_dims(space, &disk_size_i32, NULL);
        H5Sclose(space);
    }
    if (data_dset_c128 >= 0) {
        hid_t space = H5Dget_space(data_dset_c128);
        H5Sget_simple_extent_dims(space, &disk_size_c128, NULL);
        H5Sclose(space);
    }
    if (data_dset_str >= 0) {
        hid_t space = H5Dget_space(data_dset_str);
        H5Sget_simple_extent_dims(space, &disk_size_str, NULL);
        H5Sclose(space);
    }
}

// ============================================================================
// READING THE PERSISTED CHUNKING CONFIGURATION
// ============================================================================

void PanzerDB::readChunkingConfig() {
    if (data_dset_f64 < 0) {
        // Can't read config if the primary dataset doesn't exist.
        // Default config will be used.
        std::cout << "[PanzerDB] Warning: data_raw_f64 not found. Using default chunking config for read." << std::endl;
        return;
    }
    // Lire la configuration de chunking depuis le dataset existant
    hid_t dcpl = H5Dget_create_plist(data_dset_f64);
    
    if (H5Pget_layout(dcpl) == H5D_CHUNKED) {
        hsize_t chunk_dims[1];
        H5Pget_chunk(dcpl, 1, chunk_dims);
        chunk_config.data_chunk_f64 = chunk_dims[0];
        
        // Check whether compression is active
        if (H5Pget_nfilters(dcpl) > 0) {
            chunk_config.enable_compression = true;
        }
    }
    
    H5Pclose(dcpl);
    
    /*std::cout << "[PanzerDB] Read chunking config from file:" << std::endl;
    std::cout << "  Data chunk (f64): " << chunk_config.data_chunk_f64 << " elements" << std::endl;
    std::cout << "  Compression: " << (chunk_config.enable_compression ? "enabled" : "disabled") << std::endl;*/
}

// ============================================================================
// CONFIGURING THE READ (CHUNK) CACHE
// ============================================================================

void PanzerDB::configureReadCache() {
    // Configurer le cache HDF5 pour optimiser les lectures via DAPL (Dataset Access Property List)
    // This applies the cache to datasets already open or about to be opened.

    hid_t dapl = H5Pcreate(H5P_DATASET_ACCESS);
    if (dapl < 0) {
        std::cerr << "[PanzerDB] Warning: H5Pcreate failed. Read cache not configured." << std::endl;
        return;
    }

    // OPTIMISATION: Cache de chunk (Raw Data Chunk Cache)
    if (H5Pset_chunk_cache(dapl, 
                           chunk_config.chunk_cache_nslots, 
                           chunk_config.chunk_cache_size, 
                           0.75) < 0) {
        std::cerr << "[PanzerDB] Warning: H5Pset_chunk_cache failed." << std::endl;
        H5Pclose(dapl);
        return;
    }
    
    // Helper to reopen a dataset with the new DAPL
    auto reopen_dataset = [&](hid_t& dset_id, const char* name) {
        if (dset_id >= 0) {
            H5Dclose(dset_id);
            dset_id = -1;
        }
        if (H5Lexists(file_id, name, H5P_DEFAULT) > 0) {
            dset_id = H5Dopen2(file_id, name, dapl);
        }
    };

    // Appliquer aux datasets principaux
    reopen_dataset(index_dset, "index");
    reopen_dataset(data_dset_f64, "data_raw_f64");
    reopen_dataset(data_dset_i32, "data_raw_i32");
    reopen_dataset(data_dset_str, "data_raw_str");
    reopen_dataset(data_dset_c128, "data_raw_c128");
    reopen_dataset(paths_dset, "paths");
    // M1: /parent_paths no longer exists; it is never reopened.

    H5Pclose(dapl);

    /*std::cout << "[PanzerDB] Read cache configured (DAPL applied to datasets):" << std::endl;
    std::cout << "  Chunk cache: " << (chunk_config.chunk_cache_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Cache slots: " << chunk_config.chunk_cache_nslots << std::endl;*/
}

// ============================================================================
// AUTOMATIC USAGE-PATTERN DETECTION
// ============================================================================

void PanzerDB::configureChunking(const std::string& usage_hint) {
    // Patterns d'usage typiques:
    // - "time_series": many sequential writes, time-slice reads
    // - "array_of_structures": many nested AoS, structure-based access
    // - "bulk_write": single massive write, rare reads
    // - "interactive": frequent, small reads/writes
    
    if (usage_hint == "time_series") {
        // Optimized for sequential writes and time-slice reads
        chunk_config.index_chunk_rows = 4096;      // smaller chunks for better access
        chunk_config.data_chunk_f64 = 65536;       // 512 KB par chunk
        chunk_config.data_chunk_i32 = 131072;      // 512 KB par chunk
        chunk_config.data_chunk_c128 = 32768;      // 512 KB par chunk (16 bytes/elem)
        chunk_config.data_chunk_str = 8192;        // ~64-128 KB (pointeurs)
        chunk_config.path_chunk_entries = 4096;    // ~1 MB (256 bytes/entry)
        chunk_config.enable_compression = true;
        chunk_config.compression_level = 1;        // light compression
        
    } else if (usage_hint == "array_of_structures") {
        // Optimized for AoS with many small structures
        chunk_config.index_chunk_rows = 16384;     // Plus gros index chunks
        chunk_config.data_chunk_f64 = 32768;       // Plus petits data chunks
        chunk_config.data_chunk_i32 = 65536;
        chunk_config.data_chunk_c128 = 16384;      // 256 KB
        chunk_config.data_chunk_str = 4096;
        chunk_config.path_chunk_entries = 4096;
        chunk_config.enable_compression = true;
        chunk_config.compression_level = 1;
        
    } else if (usage_hint == "bulk_write") {
        // Optimized for massive single writes
        chunk_config.index_chunk_rows = 32768;     // very large chunks
        chunk_config.data_chunk_f64 = 524288;      // 4 MB par chunk
        chunk_config.data_chunk_i32 = 1048576;     // 4 MB par chunk
        chunk_config.data_chunk_c128 = 262144;     // 4 MB par chunk
        chunk_config.data_chunk_str = 65536;
        chunk_config.path_chunk_entries = 16384;   // ~4 MB
        chunk_config.enable_compression = true;
        chunk_config.compression_level = 1;        // light compression (speed)
        
    } else if (usage_hint == "interactive") {
        // Optimized for frequent, small accesses
        chunk_config.index_chunk_rows = 1024;      // Petits chunks
        chunk_config.data_chunk_f64 = 8192;        // 64 KB par chunk
        chunk_config.data_chunk_i32 = 16384;       // 64 KB par chunk
        chunk_config.data_chunk_c128 = 4096;       // 64 KB par chunk
        chunk_config.data_chunk_str = 1024;
        chunk_config.path_chunk_entries = 1024;    // ~256 KB
        chunk_config.enable_compression = false;   // Pas de compression
        
    } else {
        // Default configuration (balanced)
        chunk_config.index_chunk_rows = 8192;
        chunk_config.data_chunk_f64 = 131072;
        chunk_config.data_chunk_i32 = 262144;
        chunk_config.data_chunk_c128 = 65536;
        chunk_config.data_chunk_str = 8192;
        chunk_config.path_chunk_entries = 4096;
        chunk_config.enable_compression = true;
        chunk_config.compression_level = 1;
    }
    
    /*std::cout << "[PanzerDB] Chunking configured for: " << usage_hint << std::endl;
    std::cout << "  Index chunk: " << chunk_config.index_chunk_rows << " rows" << std::endl;
    std::cout << "  Data chunk (f64): " << chunk_config.data_chunk_f64 
              << " elements (" << (chunk_config.data_chunk_f64 * 8 / 1024) << " KB)" << std::endl;
    std::cout << "  Compression: " << (chunk_config.enable_compression ? "enabled" : "disabled")
              << " (level " << chunk_config.compression_level << ")" << std::endl; */
}

// ============================================================================
// OPTIMIZED DATASET CREATION
// ============================================================================

hid_t PanzerDB::createOptimizedDataset(const std::string& name,
                                        hid_t type,
                                        size_t chunk_size,
                                        bool enable_compression,
                                        hid_t dapl) {
    // Create the dataspace (1-D, extensible)
    hsize_t dims[1] = {0};
    hsize_t maxdims[1] = {H5S_UNLIMITED};
    hid_t space = H5Screate_simple(1, dims, maxdims);
    
    // Create and configure the property list
    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    
    // OPTIMISATION 1: Chunking
    hsize_t chunk[1] = {chunk_size};
    H5Pset_chunk(plist, 1, chunk);
    
    // OPTIMISATION 2: Compression (GZIP)
    if (enable_compression && chunk_config.enable_compression) {
        H5Pset_deflate(plist, chunk_config.compression_level);
        
        // Shuffle filter (improves compression of numeric data)
        H5T_class_t type_class = H5Tget_class(type);
        if (type_class == H5T_INTEGER || type_class == H5T_FLOAT || type_class == H5T_ARRAY) {
            H5Pset_shuffle(plist);
        }
    }
    
    // OPTIMIZATION 3: fill value (avoids costly initialization)
    if (H5Tget_class(type) == H5T_STRING && H5Tis_variable_str(type) > 0) {
        // For variable-length strings, it's mandatory to set a fill value.
        // Setting it to NULL is the standard way to indicate no fill.
        const char* fill_ptr = nullptr;
        H5Pset_fill_value(plist, type, &fill_ptr);
    } else {
        H5Pset_fill_time(plist, H5D_FILL_TIME_NEVER);
    }
    // OPTIMISATION 4: Allocation strategy
    // Allocate space progressively rather than all at once
    H5Pset_alloc_time(plist, H5D_ALLOC_TIME_INCR);
    
    // Create the dataset
    hid_t dset = H5Dcreate2(file_id, name.c_str(), type, space, 
                            H5P_DEFAULT, plist, dapl);
    
    // Cleanup
    H5Pclose(plist);
    H5Sclose(space);
    
    return dset;
}

// ============================================================================
// PUBLIC CONFIGURATION API
// ============================================================================

void PanzerDB::setChunkingHint(const std::string& hint) {
    if (mode != OpenMode::WRITE) {
        std::cerr << "[PanzerDB] Warning: Chunking hint ignored in non-WRITE mode" << std::endl;
        return;
    }
    configureChunking(hint);
}

void PanzerDB::setCompressionLevel(int level) {
    if (level < 0 || level > 9) {
        throw std::invalid_argument("Compression level must be 0-9");
    }
    chunk_config.compression_level = level;
    chunk_config.enable_compression = (level > 0);
}

void PanzerDB::disableCompression() {
    chunk_config.enable_compression = false;
}


// ============================================================================
// CHUNKING STATISTICS
// ============================================================================

ChunkingStats PanzerDB::getChunkingStats() const {
    ChunkingStats stats;
    
    // Obtenir les statistiques du dataset f64
    hid_t dcpl = H5Dget_create_plist(data_dset_f64);
    
    if (H5Pget_layout(dcpl) == H5D_CHUNKED) {
        hsize_t chunk_dims[1];
        H5Pget_chunk(dcpl, 1, chunk_dims);
        
        // Obtenir la taille du dataset
        hid_t space = H5Dget_space(data_dset_f64);
        hsize_t dims[1];
        H5Sget_simple_extent_dims(space, dims, NULL);
        H5Sclose(space);
        
        // Calculer nombre de chunks
        stats.total_chunks_written = (dims[0] + chunk_dims[0] - 1) / chunk_dims[0];
        stats.total_bytes_written = dims[0] * sizeof(double);
        
        // Compressed size (approximation)
        if (chunk_config.enable_compression) {
            // Typical ratio for numeric data with gzip
            stats.compression_ratio = 40;  // 40% de la taille originale
        }
    }
    
    H5Pclose(dcpl);
    
    return stats;
}

void PanzerDB::printChunkingStats() const {
    auto stats = getChunkingStats();
    
    std::cout << "\n=== PanzerDB Chunking Statistics ===" << std::endl;
    std::cout << "Chunks written: " << stats.total_chunks_written << std::endl;
    std::cout << "Bytes written:  " << stats.total_bytes_written / 1024 / 1024 << " MB" << std::endl;
    
    if (chunk_config.enable_compression) {
        size_t compressed_size = (stats.total_bytes_written * stats.compression_ratio) / 100;
        std::cout << "Compressed to:  " << compressed_size / 1024 / 1024 << " MB" << std::endl;
        std::cout << "Compression:    " << stats.compression_ratio << "%" << std::endl;
        std::cout << "Space saved:    " << (100 - stats.compression_ratio) << "%" << std::endl;
    }
    std::cout << "====================================\n" << std::endl;
}

// Optimized construction of path_prefix from the stack
std::string PanzerDB::buildPathFromStack(const std::vector<ArrayLevel>& stack_vector) const {
    if (stack_vector.empty()) return "";
    
    // OPTIMIZATION 1: precompute the total size for a single allocation
    size_t total_size = 0;
    for (size_t i = 0; i < stack_vector.size(); ++i) {
        total_size += stack_vector[i].name.length();
        if (i > 0) {
            // max length of a uint64_t in decimal = 20 characters
            total_size += 20;  // Pour l'index (ex: "/12345")
        }
        total_size += 1;  // Pour le '/'
    }
    
    // OPTIMISATION 2: Allocation unique
    std::string result;
    result.reserve(total_size + 10); // +10 de marge
    
    // OPTIMIZATION 3: build without reallocation
    for (size_t i = 0; i < stack_vector.size(); ++i) {
        if (i > 0) {
            // Append the previous level's index
            result += '/';
            // use to_string, which is optimized
            result += std::to_string(stack_vector[i-1].current_index);
        }
        if (!result.empty()) {
            result += '/';
        }
        result += stack_vector[i].name;
    }
    
    return result;
}

void PanzerDB::restoreTimeContext() {
    if (mode != OpenMode::APPEND && mode != OpenMode::READ) return;

    auto leaves = getLeaves();
    
    // 1. Initialize counters for explicit Dynamic AoS
    for (const auto& leaf : leaves) {
        if (leaf.flags == 3) {
            aos_time_counters[std::string(leaf.path)] = 0;
        }
    }
    
    // 2. Iterate over DATA leaves
    for (const auto& leaf : leaves) {
        if (leaf.flags == 2 || leaf.flags == 3) continue;

        // Dynamically calculate the number of time steps stored in this leaf
        uint64_t slice_volume = 1;
        for (auto dim : leaf.shape) {
            if (dim > 0) slice_volume *= dim;
        }
        
        if (slice_volume == 0) slice_volume = 1;

        uint64_t n_steps_in_leaf = leaf.count / slice_volume;
        if (n_steps_in_leaf == 0 && leaf.count > 0) n_steps_in_leaf = 1;

        uint64_t next_time = leaf.time_index + n_steps_in_leaf;

        // Find dynamic AOS parent
        std::string parent_aos = "";
        // OPTIMIZATION: Iterate over cached roots instead of all counters to avoid O(N^2)
        for (const auto& aos_path : cached_dynamic_aos_roots) {
            // Check if leaf path starts with AOS path + "/"
            // e.g. leaf="profiles_1d/0/ion/...", aos="profiles_1d"
            std::string prefix = aos_path + "/";
            if (leaf.path.rfind(prefix, 0) == 0) {
                // Take the longest match (most nested)
                if (aos_path.length() > parent_aos.length()) {
                    parent_aos = aos_path;
                }
            }
        }

        // Update counter for standalone data
        std::string leaf_path_str(leaf.path);
        if (next_time > aos_time_counters[leaf_path_str]) {
            aos_time_counters[leaf_path_str] = next_time;
        }

        // Update counter for parent AOS
        if (!parent_aos.empty()) {
            if (next_time > aos_time_counters[parent_aos]) {
                aos_time_counters[parent_aos] = next_time;
            }
        }
    }
}

PanzerDB::~PanzerDB() { close(); }

void PanzerDB::flush() {
    if (index_buffer.empty()) return;

    // Commit ordering (crash-safe): write the payloads (paths, data_raw_*) FIRST
    // and advance /index LAST. A reader discovers rows only through /index, so a
    // crash before the index write leaves only harmless unindexed payload,
    // whereas the old index-first order could leave a dangling index row pointing
    // past the materialised data_raw_* extent (a corrupt read). The final
    // H5Fflush below remains the cross-process SWMR commit barrier.
    hid_t filespace, memspace;

    // --- Flush Paths Buffer ---
    if (!paths_buffer.empty()) {
        hsize_t n_new_paths = paths_buffer.size() / PATH_MAX_LEN;
        filespace = H5Dget_space(paths_dset);
        hsize_t current_path_dims[1];
        H5Sget_simple_extent_dims(filespace, current_path_dims, NULL);
        H5Sclose(filespace);

        hsize_t new_path_dims[1] = {current_path_dims[0] + n_new_paths};
        H5Dset_extent(paths_dset, new_path_dims);
        
        filespace = H5Dget_space(paths_dset);
        hsize_t path_offset[1] = {current_path_dims[0]};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, path_offset, NULL, &n_new_paths, NULL);
        memspace = H5Screate_simple(1, &n_new_paths, NULL);
        H5Dwrite(paths_dset, H5Dget_type(paths_dset), memspace, filespace, H5P_DEFAULT, paths_buffer.data());
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    // M1: the /parent_paths dataset is no longer written; the parent path of each
    // row is reconstructed on read from the row's parent_id + kind + full path
    // (see getLeaves).  /paths still carries the full path text.

    // --- Flush Data Buffers (one for each type) ---
    if (!data_buffer_f64.empty()) {
        hsize_t n_new_data = data_buffer_f64.size();
        filespace = H5Dget_space(data_dset_f64);
        hsize_t current_data_dims[1];
        H5Sget_simple_extent_dims(filespace, current_data_dims, NULL);
        H5Sclose(filespace);

        hsize_t new_data_dims[1] = {current_data_dims[0] + n_new_data};
        H5Dset_extent(data_dset_f64, new_data_dims);
        
        filespace = H5Dget_space(data_dset_f64);
        hsize_t data_offset[1] = {current_data_dims[0]};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, data_offset, NULL, &n_new_data, NULL);
        memspace = H5Screate_simple(1, &n_new_data, NULL);
        H5Dwrite(data_dset_f64, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, data_buffer_f64.data());
        disk_size_f64 = new_data_dims[0];
        H5Sclose(memspace);
        H5Sclose(filespace);
    }
    if (!data_buffer_i32.empty()) {
        hsize_t n_new_data = data_buffer_i32.size();
        filespace = H5Dget_space(data_dset_i32);
        hsize_t current_data_dims[1];
        H5Sget_simple_extent_dims(filespace, current_data_dims, NULL);
        H5Sclose(filespace);

        hsize_t new_data_dims[1] = {current_data_dims[0] + n_new_data};
        H5Dset_extent(data_dset_i32, new_data_dims);
        
        filespace = H5Dget_space(data_dset_i32);
        hsize_t data_offset[1] = {current_data_dims[0]};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, data_offset, NULL, &n_new_data, NULL);
        memspace = H5Screate_simple(1, &n_new_data, NULL);
        H5Dwrite(data_dset_i32, H5T_NATIVE_INT, memspace, filespace, H5P_DEFAULT, data_buffer_i32.data());
        disk_size_i32 = new_data_dims[0];
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    if (!data_buffer_str.empty()) {
        hsize_t n_new_data = data_buffer_str.size();

        // SWMR-safe fixed-width write: validate, then memcpy into a flat NUL-padded
        // buffer (one STRING_MAX_LEN-byte slot per element). This mirrors the `paths`
        // column and avoids the vlen heap (which is not SWMR-safe and is excluded from
        // HDF5's official SWMR feature).
        std::vector<char> flat(n_new_data * STRING_MAX_LEN, 0);
        for (hsize_t i = 0; i < n_new_data; ++i) {
            const std::string& s = data_buffer_str[i];
            // Strings are already validated to <= STRING_MAX_LEN-1 at write time;
            // the min() is a defensive bound so a copy can never overflow the slot.
            size_t n = std::min(s.size(), (size_t)STRING_MAX_LEN);
            if (n)
                memcpy(flat.data() + (size_t)i * STRING_MAX_LEN, s.data(), n);
        }

        filespace = H5Dget_space(data_dset_str);
        hsize_t current_data_dims[1];
        H5Sget_simple_extent_dims(filespace, current_data_dims, NULL);
        H5Sclose(filespace);

        hsize_t new_data_dims[1] = {current_data_dims[0] + n_new_data};
        H5Dset_extent(data_dset_str, new_data_dims);

        filespace = H5Dget_space(data_dset_str);
        hsize_t data_offset[1] = {current_data_dims[0]};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, data_offset, NULL, &n_new_data, NULL);
        memspace = H5Screate_simple(1, &n_new_data, NULL);

        hid_t str_type_fixed = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_type_fixed, STRING_MAX_LEN);
        H5Tset_strpad(str_type_fixed, H5T_STR_NULLPAD);
        H5Tset_cset(str_type_fixed, H5T_CSET_UTF8);
        H5Dwrite(data_dset_str, str_type_fixed, memspace, filespace, H5P_DEFAULT, flat.data());
        H5Tclose(str_type_fixed);
        disk_size_str = new_data_dims[0];
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    if (!data_buffer_c128.empty()) {
        hsize_t n_new_data = data_buffer_c128.size();
        filespace = H5Dget_space(data_dset_c128);
        hsize_t current_data_dims[1];
        H5Sget_simple_extent_dims(filespace, current_data_dims, NULL);
        H5Sclose(filespace);

        hsize_t new_data_dims[1] = {current_data_dims[0] + n_new_data};
        H5Dset_extent(data_dset_c128, new_data_dims);

        filespace = H5Dget_space(data_dset_c128);
        hsize_t data_offset[1] = {current_data_dims[0]};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, data_offset, NULL, &n_new_data, NULL);
        memspace = H5Screate_simple(1, &n_new_data, NULL);

        hsize_t complex_dims[1] = {2};
        hid_t complex_tid = H5Tarray_create2(H5T_IEEE_F64LE, 1, complex_dims);
        H5Dwrite(data_dset_c128, complex_tid, memspace, filespace, H5P_DEFAULT, data_buffer_c128.data());
        disk_size_c128 = new_data_dims[0];
        H5Tclose(complex_tid);
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    // --- Flush Index Buffer (LAST — this is the commit marker) ---
    // Written after every payload so a crash can only leave unindexed slack in
    // data_raw_*, never an index row that points past the written data.
    {
        hsize_t n_new_rows = index_buffer.size() / 14;
        hsize_t current_dims[2];
        filespace = H5Dget_space(index_dset);
        H5Sget_simple_extent_dims(filespace, current_dims, NULL);

        hsize_t new_dims[2] = {current_dims[0] + n_new_rows, 14};
        H5Dset_extent(index_dset, new_dims);

        filespace = H5Dget_space(index_dset);
        hsize_t offset[2] = {current_dims[0], 0};
        hsize_t slab_dims[2] = {n_new_rows, 14};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, slab_dims, NULL);

        hsize_t mem_dims[2] = {n_new_rows, 14};
        memspace = H5Screate_simple(2, mem_dims, NULL);
        H5Dwrite(index_dset, H5T_NATIVE_UINT64, memspace, filespace, H5P_DEFAULT, index_buffer.data());
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    index_buffer.clear();
    data_buffer_f64.clear();
    data_buffer_i32.clear();
    data_buffer_str.clear();
    data_buffer_c128.clear();
    paths_buffer.clear();
    parent_paths_buffer.clear();
    leaves_cache_valid = false;

    // SWMR commit barrier. The H5Dwrite calls above only update HDF5's in-memory file
    // image; they are NOT visible to another process yet. Push the whole file image to
    // the OS (page cache / disk) so this commit becomes visible, as one unit: a reader
    // on the same file sees either the previous committed state or the complete new
    // state, never a torn mix of index row + payload. This single GLOBAL flush is what
    // makes the "index row = commit point" protocol hold across processes.
    if (file_id >= 0) {
        H5Fflush(file_id, H5F_SCOPE_GLOBAL);
    }

}

void PanzerDB::close() {
    if (file_id < 0) return;

    flush();
    if (index_dset >= 0) H5Dclose(index_dset);
    if (data_dset_f64 >= 0) H5Dclose(data_dset_f64);
    if (data_dset_i32 >= 0) H5Dclose(data_dset_i32);
    if (data_dset_str >= 0) H5Dclose(data_dset_str);
    if (data_dset_c128 >= 0) H5Dclose(data_dset_c128);
    if (paths_dset >= 0) H5Dclose(paths_dset);
    if (parent_paths_dset >= 0) H5Dclose(parent_paths_dset);

    if (should_close_loc_id) {
        H5Gclose(file_id); // It is a group
    } else if (H5Iget_type(file_id) == H5I_FILE) {
        H5Fclose(file_id); // It is a file
    }
    file_id = -1;
}


// DIRECT HYPERSLAB READ (no intermediate buffer)
// ============================================================================

template<typename T>
int PanzerDB::readSliceDirect(const Leaf& leaf, 
                               int64_t time_index,
                               T* out_buffer) const {
    
    // Volume of one time slice (recomputed from leaf.shape: already in memory)
    uint64_t slice_volume = 1;
    for (size_t s : leaf.shape) {
        if (s > 0) slice_volume *= s;
    }
    if (slice_volume == 0) slice_volume = 1;

    uint64_t local_step = time_index - leaf.time_index;
    
    // Determine the appropriate raw dataset and HDF5 type
    hid_t dset_id = -1;
    hid_t mem_type = -1;
    
    DataType type = static_cast<DataType>(leaf.flags >> 4);
    if (type == DataType::FLOAT64) {
        dset_id = data_dset_f64;
        mem_type = H5T_NATIVE_DOUBLE;
    } else if (type == DataType::INT32) {
        dset_id = data_dset_i32;
        mem_type = H5T_NATIVE_INT;
    } else if (type == DataType::COMPLEX128) {
        dset_id = data_dset_c128;
        hsize_t complex_dims[1] = {2};
        mem_type = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, complex_dims);
    } else {
        return -1;
    }
    
    // KEY OPTIMIZATION: read the slice directly via a hyperslab
    // Au lieu de lire tout le chunk puis copier
    
    hid_t file_space = H5Dget_space(dset_id);
    
    // Select exactly the desired slice
    hsize_t offset[1] = {leaf.offset + local_step * slice_volume};
    hsize_t count[1] = {slice_volume};
    
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, NULL, count, NULL);
    
    // Create the output memory space
    hid_t mem_space = H5Screate_simple(1, count, NULL);
    
    // Lecture DIRECTE dans le buffer de sortie (pas de copie!)
    herr_t status = H5Dread(dset_id, mem_type, mem_space, file_space, 
                            H5P_DEFAULT, out_buffer);
    
    H5Sclose(mem_space);
    H5Sclose(file_space);
    
    if (type == DataType::COMPLEX128) {
        H5Tclose(mem_type);
    }
    
    return (status >= 0) ? 0 : -1;
}

// Instantiation explicite
template int PanzerDB::readSliceDirect<double>(const Leaf&, int64_t, double*) const;
template int PanzerDB::readSliceDirect<int32_t>(const Leaf&, int64_t, int32_t*) const;
template int PanzerDB::readSliceDirect<std::complex<double>>(const Leaf&, int64_t, 
                                                              std::complex<double>*) const;

int PanzerDB::readLeavesUnion(const std::vector<const Leaf*>& leaves, void* buffer, DataType dtype) const {
    if (leaves.empty()) return 0;

    // 1. Check monotonicity of offsets to determine if we can use Hyperslab Union
    // HDF5 returns data in increasing coordinate order. If our leaves are sorted by time
    // but their offsets are not monotonic (e.g. data written out of order), 
    // a single read would scramble the time steps.
    bool monotonic = true;
    for (size_t i = 0; i < leaves.size() - 1; ++i) {
        if (leaves[i]->offset >= leaves[i+1]->offset) {
            monotonic = false;
            break;
        }
    }

    hid_t dset_id = -1;
    hid_t mem_type = -1;
    size_t element_size = 0;
    bool need_close_type = false;

    switch (dtype) {
        case DataType::FLOAT64:
            dset_id = data_dset_f64;
            mem_type = H5T_NATIVE_DOUBLE;
            element_size = sizeof(double);
            break;
        case DataType::INT32:
            dset_id = data_dset_i32;
            mem_type = H5T_NATIVE_INT;
            element_size = sizeof(int32_t);
            break;
        case DataType::COMPLEX128:
            dset_id = data_dset_c128;
            {
                hsize_t dims[1] = {2};
                mem_type = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, dims);
                need_close_type = true;
            }
            element_size = sizeof(std::complex<double>);
            break;
        default:
            return -1; // Not supported for union read (e.g. strings)
    }

    if (monotonic) {
        // OPTIMIZATION: Hyperslab Union
        hid_t file_space = H5Dget_space(dset_id);
        H5Sselect_none(file_space);

        // OPTIMIZATION: Merge contiguous leaves to reduce hyperslab selections
        hsize_t total_elements = leaves[0]->count;
        hsize_t current_off = leaves[0]->offset;
        hsize_t current_cnt = leaves[0]->count;

        for (size_t i = 1; i < leaves.size(); ++i) {
            total_elements += leaves[i]->count;
            
            if (leaves[i]->offset == current_off + current_cnt) {
                // Contiguous: extend current block
                current_cnt += leaves[i]->count;
            } else {
                // Gap: select current block and start new one
                H5Sselect_hyperslab(file_space, H5S_SELECT_OR, &current_off, NULL, &current_cnt, NULL);
                current_off = leaves[i]->offset;
                current_cnt = leaves[i]->count;
            }
        }
        // Select the last block
        H5Sselect_hyperslab(file_space, H5S_SELECT_OR, &current_off, NULL, &current_cnt, NULL);

        hsize_t mem_dims[1] = {total_elements};
        hid_t mem_space = H5Screate_simple(1, mem_dims, NULL);

        herr_t status = H5Dread(dset_id, mem_type, mem_space, file_space, H5P_DEFAULT, buffer);
        
        H5Sclose(mem_space);
        H5Sclose(file_space);
        if (need_close_type) H5Tclose(mem_type);

        return (status >= 0) ? 0 : -1;
    } else {
        // Fallback: Sequential reads to preserve order
        // This happens if data was written non-chronologically
        char* ptr = static_cast<char*>(buffer);
        for (const auto* leaf : leaves) {
            hsize_t off = leaf->offset;
            hsize_t cnt = leaf->count;
            
            hid_t fspace = H5Dget_space(dset_id);
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &off, NULL, &cnt, NULL);
            
            hsize_t mdims[1] = {cnt};
            hid_t mspace = H5Screate_simple(1, mdims, NULL);
            
            H5Dread(dset_id, mem_type, mspace, fspace, H5P_DEFAULT, ptr);
            
            H5Sclose(mspace);
            H5Sclose(fspace);
            
            ptr += cnt * element_size;
        }
        if (need_close_type) H5Tclose(mem_type);
        return 0;
    }
}

void PanzerDB::beginArray(const std::string& name, size_t size) {
    ArrayLevel level;
    level.name = name;
    level.saved_path_prefix = path_prefix;
    level.declared_size = size;
    
    beginArray(level);

    invalidateDynamicAOSCache();
}

void PanzerDB::beginArray(const std::string& name, const std::string& timebase) {
    ArrayLevel level;
    level.name = name;
    level.saved_path_prefix = path_prefix;
    level.timebase = timebase;
    level.is_dynamic = !timebase.empty();
    
    // Calculate path of this AOS
    std::string new_aos_path = path_prefix;
    if (!array_stack.empty()) {
        if (!new_aos_path.empty()) new_aos_path += "/";
        new_aos_path += std::to_string(array_stack.back().current_index);
    }
    if (!new_aos_path.empty()) new_aos_path += "/";
    new_aos_path += name;
    
    if (level.is_dynamic) {
        // ✅ Only one dynamic AOS allowed
        if (!dynamic_level.empty()) {
            throw std::runtime_error("Only one dynamic AOS allowed in hierarchy");
        }
        dynamic_level = name;
        
        // ✅ Initialize time counter for this AOS
        if (aos_time_counters.find(new_aos_path) == aos_time_counters.end()) {
            aos_time_counters[new_aos_path] = 0;
        }
        
        level.aos_full_path = new_aos_path;

        // FIX: In APPEND mode, set index to end of array
        if (mode == OpenMode::APPEND) {
            level.current_index = aos_time_counters[new_aos_path];
        }
    }
    
    beginArray(level);
    // Cache invalidation: the dynamic-AoS set can change (e.g. opening a dynamic
    // AoS after a root time write). The static-path beginArray already does this;
    // the dynamic path must as well, otherwise a stale cache (empty dyn path from
    // an earlier root write) poisons every subsequent in-AoS signal write and
    // misroutes it to CASE 2. Mirrors the static-path beginArray (line ~1263).
    invalidateDynamicAOSCache();
}

void PanzerDB::beginArray(ArrayLevel& level) {
    const auto& name = level.name;
    if (level.is_dynamic) {
        if (!dynamic_level.empty() && dynamic_level != name) throw std::runtime_error("Only one dynamic level allowed");
        dynamic_level = name;
    }

    std::string parent_path = path_prefix;
    std::string new_node_path = path_prefix;
    // CORRECTION: If inside an AoS, the new AoS path must include
    // the parent AoS instance index.
    if (!array_stack.empty()) {
        if (!new_node_path.empty()) new_node_path += "/";
        new_node_path += std::to_string(array_stack.back().current_index);
    }

    if (!new_node_path.empty()) new_node_path += "/";
    new_node_path += name;

    // ✅ Check if AoS meta-node already exists to avoid duplicates in Index Table
    bool already_exists = false;
    if (mode == OpenMode::APPEND || mode == OpenMode::READ) {
         // OPTIMIZATION: Use leaf_lookup instead of linear scan
         getLeaves(); // Ensure cache is built
         auto it = leaf_lookup.find(new_node_path);
         if (it != leaf_lookup.end()) already_exists = true;
    }

    // Anchor this meta-node to the enclosing AoS (if any) so its children can
    // record a numeric parent_id instead of a duplicated parent text.
    const uint64_t parent_row = array_stack.empty() ? PANZER_NO_PARENT_ROW
                                                    : array_stack.back().container_row_id;
    const uint64_t inst = array_stack.empty() ? 0 : array_stack.back().current_index;

    if (!already_exists) {
        uint64_t aos_flags = level.is_dynamic ? 3 : 2; // 3 for dynamic AoS, 2 for static AoS
        level.container_row_id = append_index_row(
            new_node_path, parent_path, {(uint64_t)level.declared_size}, 0, 0, 0, 0, aos_flags,
            parent_row, inst);
    } else {
        // APPEND/READ dedup: reuse the existing meta-node's row id as the anchor for
        // the children about to be appended.  In parent-first order the leaf index is
        // the stable row id.
        if (auto it = leaf_lookup.find(new_node_path); it != leaf_lookup.end() && !it->second.empty()) {
            level.container_row_id = it->second.front();
        }
    }

    path_prefix = new_node_path;
    current_path.push_back(name);
    array_stack.push_back(level);
}

const std::vector<PanzerDB::Leaf>& PanzerDB::getLeaves() const { // NOLINT(readability-const-return-type)
    if (leaves_cache_valid) {
        return cached_leaves;
    }

    // Cache is invalid, rebuild it directly in class member.
    cached_leaves.clear();
    leaf_lookup.clear();
    parent_lookup.clear();
    max_time_at_dynamic_root.clear();

    // 1. Read size of /index
    hsize_t dims[2];
    hid_t space = H5Dget_space(index_dset);
    if (space < 0) {
        leaves_cache_valid = true; // Le cache est maintenant valide (mais vide).
        return cached_leaves;
    }
    H5Sget_simple_extent_dims(space, dims, nullptr);
    H5Sclose(space);
    uint64_t n_rows = dims[0];

    if (n_rows == 0) {
        max_time_at_dynamic_root.clear();
        leaves_cache_valid = true; // Cache is now valid (but empty).
        return cached_leaves;
    }

    uint64_t read_count = n_rows;

    cached_dynamic_aos_roots.clear();
    cached_paths_blocks.clear();
    cached_parent_paths_blocks.clear();

    // 2. Read index and paths (full reload)
    std::vector<uint64_t> idx(read_count * 14);
    H5Dread(index_dset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, idx.data());

    cached_paths_blocks.emplace_back(read_count * PATH_MAX_LEN);
    cached_parent_paths_blocks.emplace_back(read_count * PATH_MAX_LEN);
    std::vector<char>& paths_data = cached_paths_blocks.back();
    std::vector<char>& parent_paths_data = cached_parent_paths_blocks.back();

    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, PATH_MAX_LEN);
    H5Tset_strpad(str_type, H5T_STR_NULLPAD);
    if (paths_dset >= 0) {
        hid_t paths_space = H5Dget_space(paths_dset);
        hsize_t paths_dims[1];
        H5Sget_simple_extent_dims(paths_space, paths_dims, nullptr);
        
        if (paths_dims[0] >= n_rows) { // Ensure data exists
            H5Dread(paths_dset, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, paths_data.data());
        }
        H5Sclose(paths_space);
    }
    H5Tclose(str_type);

    // M1: /parent_paths is no longer read.  parent_paths_data (already allocated and
    // zero-initialised above) simply backs the reconstructed parent paths, which the
    // loop below fills from each row's (parent_id, kind, full path).

    // 3. Iterate over each row and build Leaf objects
    // Only reserve if we are starting from scratch or growing significantly
    if (cached_leaves.capacity() < n_rows) {
        cached_leaves.reserve(n_rows);
        // Note: reserving map/lookup is not standard but we can hint if needed,
        // but standard containers manage this.
    }

    // M1: reconstruct each row's parent_path without a /parent_paths dataset.  The
    // writer builds a DATA leaf's full path as  parent_path + "/" + name  and an AoS
    // META row's full path as  parent_path + "/" + instance + "/" + name,  storing
    // parent_path = path_prefix respectively.  So the stored parent_path is exactly:
    //   - ""                         for a root row          (parent_id == NO_PARENT)
    //   - full_path minus  1 segment (last name)             for a data leaf (kind 0/1)
    //   - full_path minus  2 segments (instance + name)      for an AoS meta (kind 2/3)
    auto parent_path_of = [&](uint64_t i) -> std::string {
        if (idx[i * 14 + 12] == PANZER_NO_PARENT_ROW) return std::string();
        const uint64_t leaf_kind = idx[i * 14 + 11] & 0xFULL;
        const std::string full(static_cast<const char*>(paths_data.data() + i * PATH_MAX_LEN));
        const size_t n_strip = (leaf_kind == 2 || leaf_kind == 3) ? 2 : 1;
        size_t end = full.size();
        for (size_t k = 0; k < n_strip && end > 0; ++k) {
            const size_t slash = full.rfind('/', end - 1);
            end = (slash == std::string::npos) ? 0 : slash;
        }
        return full.substr(0, end);
    };

    // --- Crash-safe guardrail: snapshot data_raw_* extents -----------------
    // Materialised on-disk extents of the four data_raw_* datasets at the
    // moment of this read. A DATA row whose (offset + count) exceeds its
    // dataset's extent is dangling (the old index-first ordering could produce
    // one if a crash interrupted flush()); we skip it rather than serving a
    // corrupt read.
    auto dset_extent = [](hid_t dset_id) -> uint64_t {
        if (dset_id < 0) return 0;
        hid_t space = H5Dget_space(dset_id);
        if (space < 0) return 0;
        hsize_t ext[1] = {0};
        H5Sget_simple_extent_dims(space, ext, nullptr);
        H5Sclose(space);
        return ext[0];
    };
    const uint64_t ext_f64  = dset_extent(data_dset_f64);
    const uint64_t ext_i32  = dset_extent(data_dset_i32);
    const uint64_t ext_c128 = dset_extent(data_dset_c128);
    const uint64_t ext_str  = dset_extent(data_dset_str);
    // -----------------------------------------------------------------------

    for (uint64_t i = 0; i < read_count; ++i) {
        const uint64_t* row = &idx[i * 14];

        // --- Crash-safe guardrail (per row, before emplace) -----------------
        // A DATA row (kind 0) whose payload extends past the materialised
        // data_raw_* extent is dangling: only a torn pre-change crash could
        // produce it. Skip it. AoS meta rows (kind 2/3) and empty-data rows
        // (is_empty) are not backed by data_raw_*, so the check only applies
        // to kind 0 data.
        {
            const uint64_t leaf_kind = row[11] & 0xFULL;
            if (leaf_kind == 0) {
                const DataType dtype = static_cast<DataType>(row[11] >> 4);
                uint64_t extent = 0;
                if      (dtype == DataType::FLOAT64)    extent = ext_f64;
                else if (dtype == DataType::INT32)      extent = ext_i32;
                else if (dtype == DataType::COMPLEX128) extent = ext_c128;
                else if (dtype == DataType::STRING)     extent = ext_str;
                if (row[9] + row[10] > extent) continue;   // dangling → skip
            }
        }
        // -------------------------------------------------------------------

        // OPTIMIZATION: build in place to avoid copying 'leaf' and its 'shape' vector
        cached_leaves.emplace_back();
        Leaf& leaf = cached_leaves.back();

        leaf.path         = std::string_view(paths_data.data() + i * PATH_MAX_LEN);

        // Materialise the reconstructed parent path into the persistent backing
        // buffer so the std::string_view stays valid for the life of the leaf cache.
        const std::string recon_parent = parent_path_of(i);
        char* pp_slot = parent_paths_data.data() + i * PATH_MAX_LEN;
        const size_t copy_len = std::min(recon_parent.size(), (size_t)(PATH_MAX_LEN - 1));
        std::memcpy(pp_slot, recon_parent.data(), copy_len);
        pp_slot[copy_len] = '\0';
        leaf.parent_path  = std::string_view(pp_slot);

        leaf.time_index   = row[8];
        leaf.offset       = row[9];
        leaf.count        = row[10];
        leaf.flags        = row[11];
        leaf.is_empty     = (row[11] == 1);

        // Shape
        uint64_t ndim = row[1];
        leaf.shape.reserve(ndim);
        for (uint64_t d = 0; d < ndim && d < 6; ++d) leaf.shape.push_back(row[2 + d]);
        
        // Populate lookups
        leaf_lookup[leaf.path].push_back(cached_leaves.size() - 1);
        parent_lookup[leaf.parent_path].push_back(cached_leaves.size() - 1);
        
        if (leaf.flags == 3) {
            cached_dynamic_aos_roots.emplace_back(leaf.path);
        }
    }

    rebuildDynamicRootTimeIndex();

    leaves_cache_valid = true;
    return cached_leaves;
}

void PanzerDB::rebuildDynamicRootTimeIndex() const {
    max_time_at_dynamic_root.clear();
    if (cached_dynamic_aos_roots.empty() || cached_leaves.empty()) return;

    // For each dynamic AoS root, track the highest time_index among ALL of its
    // descendant DATA leaves (any depth). This makes getAOSShape correct for
    // dynamic AoS that only contain nested static AoS (e.g. time_slice/ggd/
    // theta/values) where none of the direct children carry a time index.
    for (const auto& root : cached_dynamic_aos_roots) {
        const std::string prefix = root + "/";
        uint64_t max_t = 0;
        bool found = false;
        for (const auto& leaf : cached_leaves) {
            if ((leaf.flags & 0xF) != 0) continue; // data leaves only
            if (leaf.path.compare(0, prefix.size(), prefix) != 0) continue;
            if (!found || leaf.time_index > max_t) {
                max_t = leaf.time_index;
                found = true;
            }
        }
        if (found) {
            max_time_at_dynamic_root[std::string_view(root)] = max_t;
        }
    }
}

template<typename T>
void PanzerDB::readTensor(const Leaf& leaf, T* out_buffer) const {
    if (leaf.is_empty) throw std::runtime_error("Leaf is empty");

    DataType type = static_cast<DataType>(leaf.flags >> 4); // Type is encoded in upper bits of flag
    hid_t dset_id = -1;
    hid_t mem_type = -1;

    if (type == DataType::FLOAT64) {
        dset_id = data_dset_f64;
        mem_type = H5T_NATIVE_DOUBLE;
    } else if (type == DataType::INT32) {
        dset_id = data_dset_i32;
        mem_type = H5T_NATIVE_INT;
    } else if (type == DataType::COMPLEX128) {
        dset_id = data_dset_c128;
        hsize_t complex_dims[1] = {2};
        mem_type = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, complex_dims);
    } else {
        throw std::runtime_error("Unsupported data type for readTensor");
    }

    if (dset_id < 0) throw std::runtime_error("Dataset for this type is not open");

    hsize_t hoffset = leaf.offset;
    hsize_t hcount = leaf.count;
    hid_t space = H5Dget_space(dset_id);
    H5Sselect_hyperslab(space, H5S_SELECT_SET, &hoffset, NULL, &hcount, NULL);
    hid_t memspace = H5Screate_simple(1, &hcount, NULL);
    herr_t status = H5Dread(dset_id, mem_type, memspace, space, H5P_DEFAULT, out_buffer);
    if (status < 0) {
        // Handle HDF5 read error
    }
    H5Sclose(memspace); H5Sclose(space);
    if (type == DataType::COMPLEX128) {
        H5Tclose(mem_type);
    }
}

// Explicit template instantiation
template void PanzerDB::readTensor<double>(const Leaf& leaf, double* out_buffer) const;
template void PanzerDB::readTensor<int32_t>(const Leaf& leaf, int32_t* out_buffer) const;
template void PanzerDB::readTensor<std::complex<double>>(const Leaf& leaf, std::complex<double>* out_buffer) const;

template<>
void PanzerDB::readTensor<std::string>(const Leaf& leaf, std::string* out_buffer) const {
    if (leaf.is_empty) throw std::runtime_error("Leaf is empty");
    if (static_cast<DataType>(leaf.flags >> 4) != DataType::STRING) {
        throw ALBackendException("Mismatched data type for readTensor<std::string>", LOG);
    }

    // Check validity and attempt recovery if needed
    H5I_type_t id_type = H5Iget_type(data_dset_str);
    if (id_type != H5I_DATASET) {
        throw ALBackendException("Dataset for string type is not open", LOG);
    }
    
    hid_t dset_id = data_dset_str;

    hsize_t hoffset = leaf.offset;
    hsize_t hcount = leaf.count;
    hid_t space = H5Dget_space(dset_id);
    if (space < 0) throw ALBackendException("H5Dget_space failed", LOG);
    if (H5Sselect_hyperslab(space, H5S_SELECT_SET, &hoffset, NULL, &hcount, NULL) < 0) {
        H5Sclose(space);
        throw ALBackendException("H5Sselect_hyperslab failed", LOG);
    }
    hid_t memspace = H5Screate_simple(1, &hcount, NULL);

    // data_raw_str is now a FIXED-WIDTH, NUL-padded C1 column (SWMR-safe).
    // Read hcount elements of STRING_MAX_LEN bytes, then copy each to a
    // std::string trimmed at its NUL terminator.
    hid_t str_type_fixed = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type_fixed, STRING_MAX_LEN);
    H5Tset_strpad(str_type_fixed, H5T_STR_NULLPAD);
    H5Tset_cset(str_type_fixed, H5T_CSET_UTF8);

    std::vector<char> buf((size_t)hcount * STRING_MAX_LEN, 0);

    herr_t status = H5Dread(dset_id, str_type_fixed, memspace, space, H5P_DEFAULT, buf.data());
    if (status < 0) {
        H5Tclose(str_type_fixed);
        H5Sclose(memspace);
        H5Sclose(space);
        throw ALBackendException("H5Dread failed for string tensor", LOG);
    }

    for (size_t i = 0; i < (size_t)hcount; ++i) {
        const char* slot = buf.data() + (size_t)i * STRING_MAX_LEN;
        out_buffer[i] = std::string(slot, strnlen(slot, STRING_MAX_LEN));
    }

    H5Tclose(str_type_fixed);
    H5Sclose(memspace);
    H5Sclose(space);
}

template void PanzerDB::readTensor<std::string>(const Leaf& leaf, std::string* out_buffer) const;

// Private helper function to avoid duplication
template<typename T>
void PanzerDB::writeDataImpl(const std::string& name,
                              const std::vector<size_t>& shape,
                              const T* data,
                              size_t count,
                              const std::string& timebase,
                              DataType dtype,
                              hid_t dataset_id,
                              std::vector<T>& buffer) {
    // ✅ ASSERT: timebase must be empty (static data only)
    if (!timebase.empty()) {
        throw ALBackendException("writeData with timebase not allowed, use writeDataSlices", LOG);
    }
    
    last_level_had_write = true;
    
    if (!array_stack.empty()) { 
        array_stack.back().had_write = true;
        array_stack.back().actual_count++;
    }

    // OPTIMIZATION 1: build parent_path efficiently
    std::string parent_path;
    if (!array_stack.empty()) {
        // precompute the required size
        size_t estimated_size = path_prefix.length() + 25; // +25 pour "/12345"
        parent_path.reserve(estimated_size);
        
        parent_path = path_prefix;
        parent_path += '/';
        parent_path += std::to_string(array_stack.back().current_index);
    } else {
        parent_path = path_prefix;
    }

    // OPTIMIZATION 2: build full_path with reserved capacity
    std::string full_path;
    full_path.reserve(parent_path.length() + name.length() + 1);
    
    full_path = parent_path;
    if (!full_path.empty()) full_path += '/';
    full_path += name;


    // ✅ Static data: time_idx = 0 always
    uint64_t time_idx = 0;
    
    // OPTIMISATION: Use cached disk size
    uint64_t current_disk_size = 0;
    if (dtype == DataType::FLOAT64) current_disk_size = disk_size_f64;
    else if (dtype == DataType::INT32) current_disk_size = disk_size_i32;
    else if (dtype == DataType::COMPLEX128) current_disk_size = disk_size_c128;
    
    uint64_t offset = current_disk_size + buffer.size();

    if (buffer.capacity() < buffer.size() + count) {
        size_t new_capacity = std::max(buffer.size() + count, 
                                       buffer.capacity() * BUFFER_GROWTH_FACTOR);
        buffer.reserve(new_capacity);
    }

    buffer.insert(buffer.end(), data, data + count);

    uint64_t flags = (static_cast<uint64_t>(dtype) << 4);
    const uint64_t parent_row = array_stack.empty() ? PANZER_NO_PARENT_ROW
                                                    : array_stack.back().container_row_id;
    const uint64_t inst = array_stack.empty() ? 0 : array_stack.back().current_index;
    append_index_row(full_path, parent_path, shape, 0, time_idx, offset, count, flags, parent_row, inst);
}

// OPTIMIZED PATH PARSING (for substitution in readDataByIndex)
// ============================================================================



PathComponents PanzerDB::parsePath(const std::string& path, 
                                   const std::string& aos_path) const {
    PathComponents result;
    
    if (path.length() <= aos_path.length()) {
        return result;
    }
    
    // check that path starts with aos_path + '/'
    std::string expected_prefix = aos_path + "/";
    if (path.rfind(expected_prefix, 0) != 0) {
        return result;
    }
    
    result.prefix = aos_path;
    
    // Extraire ce qui suit
    std::string remainder = path.substr(expected_prefix.length());
    
    // OPTIMISATION: Trouver le premier '/' sans copie de string
    size_t first_slash = remainder.find('/');
    
    if (first_slash != std::string::npos) {
        result.index_str = remainder.substr(0, first_slash);
        result.suffix = remainder.substr(first_slash);
        result.has_index = true;
    } else {
        result.index_str = remainder;
        result.suffix = "";
        result.has_index = true;
    }
    
    return result;
}

// double
template<>
void PanzerDB::writeData<double>(const std::string& name,
                         const std::vector<size_t>& shape,
                         const double* data, size_t count,
                         const std::string& timebase) {
    writeDataImpl(name, shape, data, count, timebase,
                  DataType::FLOAT64, data_dset_f64, data_buffer_f64);
}

// int32_t
template<>
void PanzerDB::writeData<int32_t>(const std::string& name,
                         const std::vector<size_t>& shape,
                         const int32_t* data, size_t count,
                         const std::string& timebase) {
    writeDataImpl(name, shape, data, count, timebase,
                  DataType::INT32, data_dset_i32, data_buffer_i32);
}

// complex
template<>
void PanzerDB::writeData<std::complex<double>>(const std::string& name,
                         const std::vector<size_t>& shape,
                         const std::complex<double>* data, size_t count,
                         const std::string& timebase) {
    writeDataImpl(name, shape, data, count, timebase,
                  DataType::COMPLEX128, data_dset_c128, data_buffer_c128);
}

// New specific signature for raw strings
template<>
void PanzerDB::writeData<char>(const std::string& name,
                         const std::vector<size_t>& shape,
                         const char* data,  // ✅ Simple pointer to buffer
                         size_t count,      // count should be == shape[0]
                         const std::string& timebase) {
    if (!timebase.empty()) {
        throw std::runtime_error("writeData with timebase not allowed, use writeDataSlices");
    }
    
    last_level_had_write = true;
    
    if (!array_stack.empty()) { 
        array_stack.back().had_write = true;
        array_stack.back().actual_count++;
    }

    std::string parent_path = path_prefix;
    if (!array_stack.empty()) {
        parent_path += "/" + std::to_string(array_stack.back().current_index);
    }

    std::string full_path = parent_path;
    if (!full_path.empty()) full_path += "/";
    full_path += name;

    uint64_t time_idx = 0;
    
    uint64_t offset = disk_size_str + data_buffer_str.size();

    // ✅ Construct std::string from raw buffer
    size_t str_length = shape.empty() ? count : shape[0];
    if (str_length > STRING_MAX_LEN - 1) {
        throw ALBackendException(
            "writeStrings: string of " + std::to_string(str_length) +
            " bytes exceeds fixed column width STRING_MAX_LEN-1 = " +
            std::to_string(STRING_MAX_LEN - 1) + " bytes", LOG);
    }
    data_buffer_str.emplace_back(data, str_length);  // std::string(ptr, length)

    uint64_t flags = (static_cast<uint64_t>(DataType::STRING) << 4);
    const uint64_t parent_row = array_stack.empty() ? PANZER_NO_PARENT_ROW
                                                    : array_stack.back().container_row_id;
    const uint64_t inst = array_stack.empty() ? 0 : array_stack.back().current_index;
    append_index_row(full_path, parent_path, shape, 0, time_idx, offset, 1, flags, parent_row, inst);
    //                                                                    ↑ count = 1 (single string)
}

template<>
void PanzerDB::writeData<const char*>(const std::string& name,
                         const std::vector<size_t>& shape,
                         const char* const* data,
                         size_t count,
                         const std::string& timebase) {
    // This is for writing a single string scalar or a list of strings as a static entry.
    // We can reuse writeDataSlices with n_slices = 1 for a single entry.
    writeDataSlices(name, shape, data, 1, timebase);
}
// Private helper function to avoid duplication
template<typename T>
void PanzerDB::writeDataSlicesImpl(const std::string& name,
                                    const std::vector<size_t>& base_shape,
                                    const T* data,
                                    size_t n_slices,
                                    const std::string& timebase,
                                    DataType dtype,
                                    hid_t dataset_id,
                                    std::vector<T>& buffer) {
    std::string dynamic_aos_path = getDynamicAOSPath();

    // ✅ Construct data path to serve as time key
    std::string parent_path = path_prefix;
    if (!array_stack.empty() && !array_stack.back().is_dynamic) {
        parent_path += "/" + std::to_string(array_stack.back().current_index);
    }
    std::string full_path = parent_path;
    if (!full_path.empty()) full_path += "/";
    full_path += name;
    
    uint64_t base_time = 0;

    // ✅ CASE 1: Data in dynamic AOS
    if (!dynamic_aos_path.empty()) {
        // Auto-sync avec l'index de l'AOS dynamique
        size_t dynamic_aos_current_iteration = 0;
        std::string dynamic_aos_timebase;
        // Parcours inverse pour trouver le plus proche parent dynamique
        for (auto it = array_stack.rbegin(); it != array_stack.rend(); ++it) {
            if (it->is_dynamic) {
                dynamic_aos_current_iteration = it->current_index;
                dynamic_aos_timebase = it->timebase;
                break;
            }
        }

        // FIX: Use max between current iteration and known signal end
        // This allows:
        // 1. Alignment of different signals (test_magnetics) via dynamic_aos_current_iteration
        // 2. Chunked writing of the SAME signal via aos_time_counters[full_path]
        
        uint64_t signal_next_time = aos_time_counters[full_path];
        base_time = std::max((uint64_t)dynamic_aos_current_iteration, signal_next_time);
        
        // FIX: If signal is lagging (gap) relative to AoS timebase, align it.
        // Assume timebase has already been written for this slice (standard case).
        if (!dynamic_aos_timebase.empty()) {
             std::string time_path = dynamic_aos_path + "/" + dynamic_aos_timebase;
             if (aos_time_counters.find(time_path) != aos_time_counters.end()) {
                 uint64_t time_next = aos_time_counters[time_path];
                 if (time_next >= n_slices) {
                     base_time = std::max(base_time, time_next - n_slices);
                 }
             }
        }
    } 
    // ✅ CASE 2: Standalone dynamic data (no dynamic AOS parent)
    else {
        // Use full data path as time key
        std::string time_key = full_path;
        
        // Initialize counter if necessary
        if (aos_time_counters.find(time_key) == aos_time_counters.end()) {
            aos_time_counters[time_key] = 0;
        }
        
        base_time = aos_time_counters[time_key];

        // FIX (gaps, homogeneous_time=1): for a standalone dynamic signal on the
        // master timebase, anchor the slice number to the timebase's written
        // position (like CASE 1 does with the dynamic AoS timebase). Without this,
        // skipping the signal for a time step would compress the signal's timeline
        // (its next write landing on the next slice) instead of leaving a resolvable
        // gap aligned to the master timebase. max() preserves chunked writes.
        // Guarded by timebase so it never fires for the timebase write itself.
        if (!timebase.empty() && aos_time_counters.find(timebase) != aos_time_counters.end()) {
            uint64_t time_next = aos_time_counters[timebase];
            if (time_next >= n_slices) {
                base_time = std::max(base_time, time_next - n_slices);
            }
        }
    }

    // --- Common writing ---
    size_t slice_size = 1;
    for (auto s : base_shape) if (s > 0) slice_size *= s;

    // Raw data writing
    // OPTIMISATION: Use cached disk size
    uint64_t current_disk_size = 0;
    if (dtype == DataType::FLOAT64) current_disk_size = disk_size_f64;
    else if (dtype == DataType::INT32) current_disk_size = disk_size_i32;
    else if (dtype == DataType::COMPLEX128) current_disk_size = disk_size_c128;
    
    uint64_t start_offset = current_disk_size + buffer.size();
    buffer.insert(buffer.end(), data, data + slice_size * n_slices);
    
    uint64_t flags = (static_cast<uint64_t>(dtype) << 4);
    const uint64_t parent_row = array_stack.empty() ? PANZER_NO_PARENT_ROW
                                                    : array_stack.back().container_row_id;
    const uint64_t inst = array_stack.empty() ? 0 : array_stack.back().current_index;

    append_index_row(full_path, parent_path, base_shape, 0,
                    base_time,                  // Start time
                    start_offset,               // Start offset
                    slice_size * n_slices,      // TOTAL Count
                    flags, parent_row, inst);

    last_level_had_write = true;
    
    // Update counters
    uint64_t next_time = base_time + n_slices;
    aos_time_counters[full_path] = next_time;

    if (!dynamic_aos_path.empty()) {
        if (next_time > aos_time_counters[dynamic_aos_path]) {
            aos_time_counters[dynamic_aos_path] = next_time;
        }
    }

}

// double - uses data_dset_f64 and data_buffer_f64
void PanzerDB::writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const double* data, size_t n_slices,
                               const std::string& timebase) {
    writeDataSlicesImpl(name, base_shape, data, n_slices, timebase,
                        DataType::FLOAT64, data_dset_f64, data_buffer_f64);
}

// int32_t - uses data_dset_i32 and data_buffer_i32
void PanzerDB::writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const int32_t* data, size_t n_slices,
                               const std::string& timebase) {
    writeDataSlicesImpl(name, base_shape, data, n_slices, timebase,
                        DataType::INT32, data_dset_i32, data_buffer_i32);
}

// complex - uses data_dset_c128 and data_buffer_c128
void PanzerDB::writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const std::complex<double>* data, size_t n_slices,
                               const std::string& timebase) {
    writeDataSlicesImpl(name, base_shape, data, n_slices, timebase,
                        DataType::COMPLEX128, data_dset_c128, data_buffer_c128);
}

void PanzerDB::writeDataSlices(const std::string& name,
                               const std::vector<size_t>& base_shape,
                               const char* const* data, size_t n_slices,
                               const std::string& timebase) {
    std::string dynamic_aos_path = getDynamicAOSPath();

    // Auto-sync (same code as others)
    size_t dynamic_aos_current_iteration = 0;
    std::string dynamic_aos_timebase;
    for (auto it = array_stack.rbegin(); it != array_stack.rend(); ++it) {
        if (it->is_dynamic) {
            dynamic_aos_current_iteration = it->current_index;
            dynamic_aos_timebase = it->timebase;
            break;
        }
    }
    
    uint64_t base_time = 0;
    std::string time_key;
    
    std::string parent_path = path_prefix;
    if (!array_stack.empty() && !array_stack.back().is_dynamic) {
        parent_path += "/" + std::to_string(array_stack.back().current_index);
    }
    std::string full_path = parent_path;
    if (!full_path.empty()) full_path += "/";
    full_path += name;

    if (!dynamic_aos_path.empty()) {
        // FIX: Use max between current iteration and known signal end (like numerics)
        uint64_t signal_next_time = aos_time_counters[full_path];
        base_time = std::max((uint64_t)dynamic_aos_current_iteration, signal_next_time);
        
        // FIX: Alignment with timebase in case of gap (like numerics)
        if (!dynamic_aos_timebase.empty()) {
             std::string time_path = dynamic_aos_path + "/" + dynamic_aos_timebase;
             if (aos_time_counters.find(time_path) != aos_time_counters.end()) {
                 uint64_t time_next = aos_time_counters[time_path];
                 if (time_next >= n_slices) {
                     base_time = std::max(base_time, time_next - n_slices);
                 }
             }
        }
    } else {
        time_key = full_path;
        if (aos_time_counters.find(time_key) == aos_time_counters.end()) {
            aos_time_counters[time_key] = 0;
        }
        base_time = aos_time_counters[time_key];

        // FIX (gaps, homogeneous_time=1): same timebase anchoring as the numeric
        // writeDataSlicesImpl CASE 2, so a skipped string slice leaves a
        // resolvable gap aligned to the master timebase instead of compressing.
        if (!timebase.empty() && aos_time_counters.find(timebase) != aos_time_counters.end()) {
            uint64_t time_next = aos_time_counters[timebase];
            if (time_next >= n_slices) {
                base_time = std::max(base_time, time_next - n_slices);
            }
        }
    }

    uint64_t start_offset = disk_size_str + data_buffer_str.size();

    // ✅ DIFFERENCE: count_per_slice for strings
    size_t count_per_slice = base_shape.empty() ? 1 : base_shape[0];

    // Enclosing AoS is constant across the loop: one parent anchor for all slices.
    const uint64_t parent_row = array_stack.empty() ? PANZER_NO_PARENT_ROW
                                                    : array_stack.back().container_row_id;
    const uint64_t inst = array_stack.empty() ? 0 : array_stack.back().current_index;

    for (size_t i = 0; i < n_slices; ++i) {
        const char* const* slice_data = data + (i * count_per_slice);
        for (size_t j = 0; j < count_per_slice; ++j) {
            size_t slen = std::strlen(slice_data[j]);
            if (slen > STRING_MAX_LEN - 1) {
                throw ALBackendException(
                    "writeStrings: string of " + std::to_string(slen) +
                    " bytes exceeds fixed column width STRING_MAX_LEN-1 = " +
                    std::to_string(STRING_MAX_LEN - 1) + " bytes", LOG);
            }
            data_buffer_str.emplace_back(slice_data[j]);
        }

        uint64_t flags = (static_cast<uint64_t>(DataType::STRING) << 4);
        append_index_row(full_path, parent_path, base_shape, 0, base_time + i,
                        start_offset + (i * count_per_slice), count_per_slice, flags,
                        parent_row, inst);
    }
    
    last_level_had_write = true;
    
    if (!dynamic_aos_path.empty()) {
        uint64_t next_time = base_time + n_slices;
        
        aos_time_counters[full_path] = next_time; // Update signal counter
        
        if (next_time > aos_time_counters[dynamic_aos_path]) {
            aos_time_counters[dynamic_aos_path] = next_time;
        }
    } else {
        aos_time_counters[time_key] += n_slices;
    }
}

void PanzerDB::incrementArrayIndex() {
    if (!array_stack.empty()) {
        array_stack.back().current_index++;
    }
}

void PanzerDB::setCurrentArrayIndex(size_t new_index) {
    if (!array_stack.empty()) {
        array_stack.back().current_index = new_index;
    }
}

// APPEND_INDEX_ROW: AVOID STRING COPIES
// ============================================================================

uint64_t PanzerDB::append_index_row(const std::string& full_path,
                                    const std::string& parent_path,
                                    const std::vector<size_t>& shape,
                                    uint64_t type,
                                    uint64_t time_idx,
                                    uint64_t offset,
                                    uint64_t count,
                                    uint64_t flags,
                                    uint64_t parent_id, uint64_t index_value) {

    const uint64_t assigned_row_id = next_row_id++;
    // M1: parent_path is no longer stored on disk; it is reconstructed in
    // getLeaves() from (parent_id, kind, full path).
    (void)parent_path;

    // OPTIMIZATION 1: preallocate exactly what is needed for the /paths entry
    size_t required_paths = paths_buffer.size() + PATH_MAX_LEN;
    if (paths_buffer.capacity() < required_paths) {
        size_t new_cap = std::max(required_paths, paths_buffer.capacity() * BUFFER_GROWTH_FACTOR);
        paths_buffer.reserve(new_cap);
    }

    size_t current_path_size = paths_buffer.size();
    paths_buffer.resize(current_path_size + PATH_MAX_LEN, 0);

    // optimized copy (strncpy is fast for fixed-size buffers)
    strncpy(paths_buffer.data() + current_path_size, full_path.c_str(), PATH_MAX_LEN - 1);

    // build the row (layout: row[0]=type, row[12]=parent_id, row[13]=index_value)
    uint64_t row[14] = {0};
    row[0] = type;
    row[1] = shape.size();
    for (size_t i = 0; i < shape.size() && i < 6; ++i) {
        row[2+i] = shape[i];
    }
    row[8] = time_idx;
    row[9] = offset;
    row[10] = count;
    row[11] = flags;
    row[12] = parent_id;     // enclosing AoS meta-node row id (PANZER_NO_PARENT_ROW for root)
    row[13] = index_value;   // instance index within that parent AoS

    index_buffer.insert(index_buffer.end(), row, row + 14);
    return assigned_row_id;
}

bool PanzerDB::isInsideDynamicAOS(std::string* timebase) const {
    if (array_stack.empty()) {
        return false;
    }
    // Data is dynamic if it has at least one dynamic AoS parent.
    // So we must traverse the entire stack.
    for (auto it = array_stack.rbegin(); it != array_stack.rend(); ++it) {
        const ArrayLevel& level = *it;
        if (level.is_dynamic) {
            if (timebase) {
                *timebase = level.timebase;
            }
            return true; // Found dynamic parent
        }
    }
    if (timebase) timebase->clear();
    return false; // No dynamic parent found
}


void PanzerDB::dumpLeafIndex() const {
    const auto& leaves = getLeaves();
    std::cerr << "=== PanzerDB Index Table (" << leaves.size() << " entries) ===" << std::endl;
    std::cerr << std::left << std::setw(60) << "Path" 
              << std::setw(10) << "TimeIdx" 
              << std::setw(10) << "Count" 
              << std::setw(15) << "Offset" 
              << "Shape" << std::endl;
    
    for (const auto& leaf : leaves) {
        std::cerr << std::left << std::setw(60) << leaf.path 
                  << std::setw(10) << leaf.time_index 
                  << std::setw(10) << leaf.count 
                  << std::setw(15) << leaf.offset << " [";
        for (size_t i = 0; i < leaf.shape.size(); ++i) {
            std::cerr << leaf.shape[i] << (i < leaf.shape.size() - 1 ? "," : "");
        }
        std::cerr << "]" << std::endl;
    }
    std::cerr << "============================================" << std::endl;
}

void PanzerDB::endArray() {
   if (current_path.empty() || array_stack.empty()) {
        return;
    }

    ArrayLevel level = array_stack.back();
    array_stack.pop_back();
    current_path.pop_back();

    path_prefix = level.saved_path_prefix;

    // ✅ If exiting dynamic AOS, signal it by clearing dynamic_level
    if (level.is_dynamic && level.name == dynamic_level) {
        dynamic_level = "";
    }

    if (preserve_empty_nodes && !level.had_write) {
        // Logic for empty nodes to be reviewed
    }

    if (!array_stack.empty()) {
        array_stack.back().actual_count += level.actual_count ? level.actual_count : 1;
    }

    invalidateDynamicAOSCache();
}

std::vector<size_t> PanzerDB::getAOSShape(const std::string& level_name) const {
  const auto &leaves = getLeaves();
  std::vector<size_t> shapes;

  // 1. Find root AoS meta-node
  const Leaf *aos_root_leaf = nullptr;


  
  // OPTIMISATION: Utiliser leaf_lookup
  auto it_root = leaf_lookup.find(std::string_view(level_name));
  if (it_root != leaf_lookup.end() && !it_root->second.empty()) {
      aos_root_leaf = &leaves[it_root->second[0]];
  }
  // No linear fallback is needed if the index is consistent

  if (!aos_root_leaf) {
    return shapes; // Return empty vector if AoS not found
  }

  // NEW LOGIC FOR DYNAMIC AoS
  if (aos_root_leaf->flags == 3) { // flags == 3 indicates dynamic AoS
    // Use the max time_index computed over ALL descendant data leaves (any
    // depth) at cache build time. Looking at direct children only would miss
    // leaves behind nested static AoS (e.g. time_slice/ggd/theta/values)
    // where no direct child carries a time index.
    auto it = max_time_at_dynamic_root.find(std::string_view(level_name));
    if (it != max_time_at_dynamic_root.end()) {
      shapes.push_back(static_cast<size_t>(it->second + 1));
    } else {
      // No data leaves found under this dynamic AoS. Size is 0.
      shapes.push_back(0);
    }

    return shapes;
  }

  // OLD LOGIC (kept for static AoS)
  // 2. Scan children to find maximum index of real instances
  long long max_index = -1;
  
  // OPTIMISATION: Utiliser parent_lookup pour ne parcourir que les enfants directs
  auto it_children = parent_lookup.find(std::string_view(level_name));
  if (it_children != parent_lookup.end()) {
    for (size_t idx : it_children->second) {
      const auto& leaf = leaves[idx];
      // leaf.path est de la forme "level_name/suffix"
      std::string_view suffix = leaf.path.substr(level_name.length() + 1); // +1 pour '/'
      // Suffix must start with a digit.
      if (!suffix.empty() && isdigit(suffix[0])) {
        size_t slash_pos = suffix.find('/');
        std::string index_str(suffix.substr(0, slash_pos));
        try {
          long long idx = std::stoll(index_str);
          if (idx > max_index) {
            max_index = idx;
          }
        } catch (...) {
          // Ignore if not a number
        }
      }
    }
  }

  // If instances found (max_index >= 0), size is max_index + 1.
  // Otherwise, use size declared in AoS meta-node.
  if (max_index >= 0) {
    shapes.push_back(static_cast<size_t>(max_index + 1));
  } else {
    // If no indexed child found, rely on declared size
    // in the AoS meta-node itself.
    if (!aos_root_leaf->shape.empty()) {
      shapes.push_back(aos_root_leaf->shape[0]);
    } else {
      shapes.push_back(0); // Case where even meta-node has no shape (should not happen for an AoS)
    }
  }
  return shapes;
}

size_t PanzerDB::getDynamicAOSSize(const std::string& aos_path) const {
    auto it = aos_time_counters.find(aos_path);
    if (it != aos_time_counters.end()) {
        return static_cast<size_t>(it->second);
    }
    return 0;
}

int64_t PanzerDB::getTimeIndex(const std::string& timebase_path, double requested_time, int interp_mode) const {
    // 1. Retrieve all index leaves.
    const auto& leaves = getLeaves();

    // 2. Filter to keep only leaves of requested timebase.
    //    Path can be complex (e.g. "A/0/B/time").
    std::vector<const Leaf*> timebase_leaves;
    
    // OPTIMIZATION: Use leaf_lookup
    auto it = leaf_lookup.find(timebase_path);
    if (it != leaf_lookup.end()) {
        for (size_t idx : it->second) {
            timebase_leaves.push_back(&leaves[idx]);
        }
    }

    // 3. Ensure leaves are sorted by time_index.
    std::sort(timebase_leaves.begin(), timebase_leaves.end(), 
              [](const Leaf* a, const Leaf* b) {
                  return a->time_index < b->time_index;
              });

    if (timebase_leaves.empty()) {
        return -1; // Timebase not found.
    }

    // 4. Read all time values into a single vector.
    std::vector<double> time_values;
    for (const auto* leaf : timebase_leaves) {
        if (leaf->count > 0) {
            size_t current_size = time_values.size();
            time_values.resize(current_size + leaf->count);
            readTensor(*leaf, time_values.data() + current_size);
        }
    }

    if (time_values.empty()) {
        return -1; // Timebase exists but is empty.
    }

    // 5. Use DataInterpolation to get the correct index based on interpolation mode.
    DataInterpolation interpolator;
    std::map<std::string, int> times_indices;
    try {
        return interpolator.getSlicesTimesIndices(requested_time, time_values, times_indices, interp_mode);
    } catch (const ALBackendException& e) {
        // Handle cases where time vector is empty or other issues.
        std::cerr << "[PanzerDB::getTimeIndex] Warning: " << e.what() << std::endl;
        return -1;
    }
}

// In panzerdb.cpp, replace lambda is_time_in_leaf with a method:
bool PanzerDB::isTimeInLeaf(const Leaf& leaf, int64_t time_index) const {
    if ((leaf.flags & 0xF) != 0) return false; // Not data

    // Spatial volume calculation
    size_t slice_volume = 1;
    for (auto s : leaf.shape) if(s > 0) slice_volume *= s;
    
    // Time steps count calculation
    size_t n_steps = (slice_volume > 0) ? (leaf.count / slice_volume) : 1;
    if (n_steps == 0) n_steps = 1;

    return (static_cast<uint64_t>(time_index) >= leaf.time_index && 
            static_cast<uint64_t>(time_index) < (leaf.time_index + n_steps));
}

int64_t PanzerDB::nearestAvailableSliceIndex(const char* full_path, int64_t requested_idx,
                                             int64_t prefer_direction,
                                             const std::vector<double>& time_basis,
                                             double requested_time) const {
    std::string target_path(full_path);

    // The dynamic AoS prefix of the target (if any), mirroring the path
    // resolution strategies of the pz_read*Data_by_index family:
    //   direct path, generic path (index dropped), substituted path
    //   (index replaced by the requested one).
    std::string dynamic_aos;
    std::string remainder_suffix; // e.g. "/ion/0/signal_1d" (leading '/')
    for (const auto& aos_path : cached_dynamic_aos_roots) {
        if (target_path.size() > aos_path.size() &&
            target_path[aos_path.size()] == '/' &&
            target_path.compare(0, aos_path.size(), aos_path) == 0) {
            dynamic_aos = aos_path;
            const std::string::size_type dyn_len = dynamic_aos.size();
            // remainder = target_path[dyn_len+1 .. end]  (skips trailing '/')
            const std::string::size_type rem_start = dyn_len + 1;
            std::string remainder = (rem_start < target_path.size())
                                            ? target_path.substr(rem_start)
                                            : std::string();
            std::string::size_type first_slash = remainder.find('/');
            if (first_slash != std::string::npos) {
                remainder_suffix = remainder.substr(first_slash);
            }
            break;
        }
    }

    auto path_covers = [&](const std::string& path, int64_t idx) -> bool {
        auto it = leaf_lookup.find(path);
        if (it == leaf_lookup.end()) return false;
        for (size_t lidx : it->second) {
            if (isTimeInLeaf(getLeaves()[lidx], idx)) return true;
        }
        return false;
    };

    auto available = [&](int64_t idx) -> bool {
        if (path_covers(target_path, idx)) return true;
        if (dynamic_aos.empty()) return false;
        if (path_covers(dynamic_aos + remainder_suffix, idx)) return true; // generic
        std::string substituted;
        substituted.reserve(dynamic_aos.size() + remainder_suffix.size() + 12);
        substituted.append(dynamic_aos);
        substituted.push_back('/');
        substituted.append(std::to_string(idx));
        substituted.append(remainder_suffix);
        return path_covers(substituted, idx);
    };

    long long nb = static_cast<long long>(time_basis.size());
    int64_t lo = 0;
    int64_t hi = (nb > 0) ? nb - 1 : 0;

    if (available(requested_idx)) return requested_idx;

    // Highest available index <= requested (linear search, few slices expected)
    int64_t highest_below = -1;
    for (int64_t i = requested_idx; i >= lo; --i) {
        if (available(i)) { highest_below = i; break; }
    }
    // Lowest available index >= requested
    int64_t lowest_above = -1;
    if (requested_idx <= hi) {
        for (int64_t i = requested_idx; i <= hi; ++i) {
            if (available(i)) { lowest_above = i; break; }
        }
    }

    if (prefer_direction < 0) {
        return highest_below != -1 ? highest_below : lowest_above;
    }
    if (prefer_direction > 0) {
        return lowest_above != -1 ? lowest_above : highest_below;
    }

    // Nearest in time; ties resolved to the smaller index
    auto distance = [&](int64_t idx) -> double {
        if (requested_time >= 0 && idx >= 0 && idx < static_cast<int64_t>(time_basis.size())) {
            return std::abs(time_basis[idx] - requested_time);
        }
        return std::abs(static_cast<double>(idx - requested_idx));
    };

    if (highest_below == -1) return lowest_above;
    if (lowest_above == -1) return highest_below;
    double d_below = distance(highest_below);
    double d_above = distance(lowest_above);
    if (d_below < d_above) return highest_below;
    if (d_above < d_below) return lowest_above;
    return highest_below; // tie -> smaller index
}

int PanzerDB::readDataByIndex(
                         const char* full_data_path,
                         int64_t time_index,
                         uint64_t* ndim_out,
                         uint64_t shape_out[6],
                         double** data_out) {


    const auto& leaves = getLeaves();
    if (leaves.empty()) return -1;

    std::string_view target_path(full_data_path);
    

    if (time_index == -1) {
        // --- Static or Global Search (Aggregation) ---
        std::vector<const Leaf*> matches;
        uint64_t total_count = 0;

        // Note: In C++17/20, find with string_view works if heterogeneous lookup is enabled or via temp string
        auto it = leaf_lookup.find(target_path);
        if (it != leaf_lookup.end()) {
            matches.reserve(it->second.size());
            for (size_t idx : it->second) {
                matches.push_back(&leaves[idx]);
                total_count += leaves[idx].count;
            }
        }

        if (matches.empty()) return -1;

        if (total_count == 0) {
            *data_out = nullptr;
            *ndim_out = 0;
            // shape_out can be left uninitialized
            return 0; // Success, empty dataset
        }

        // Simple case: single leaf
        if (matches.size() == 1) {
            const Leaf& leaf = *matches[0];
            
            size_t slice_vol = 1;
            for(auto s : leaf.shape) if(s > 0) slice_vol *= s;
            
            if (leaf.count > slice_vol) {
                *ndim_out = leaf.shape.size() + 1;
                for (size_t i = 0; i < leaf.shape.size(); ++i) shape_out[i] = leaf.shape[i];
                shape_out[*ndim_out - 1] = leaf.count / slice_vol;
            } else {
                *ndim_out = leaf.shape.size();
                for (size_t i = 0; i < *ndim_out && i < 6; ++i) shape_out[i] = leaf.shape[i];
            }
            
            *data_out = (double*)malloc(leaf.count * sizeof(double));
            this->readTensor(leaf, *data_out);
            return 0;
        }

        // Complex case: aggregation of time slices
        std::sort(matches.begin(), matches.end(), [](const Leaf* a, const Leaf* b){
            return a->time_index < b->time_index;
        });

        const Leaf& first = *matches[0];
        *data_out = (double*)malloc(total_count * sizeof(double));
        // OPTIMISATION: Use readLeavesUnion for batched read
        if (readLeavesUnion(matches, *data_out, DataType::FLOAT64) < 0) {
            size_t offset = 0;
            for(const auto* leaf : matches) {
                this->readTensor(*leaf, (*data_out) + offset);
                offset += leaf->count;
            }
        }

        *ndim_out = first.shape.size() + 1;
        for(size_t i=0; i<first.shape.size(); ++i) shape_out[i] = first.shape[i];
        
        size_t slice_vol = 1;
        for(auto s : first.shape) if(s > 0) slice_vol *= s;
        shape_out[*ndim_out - 1] = total_count / slice_vol;


        return 0;
    } else {
        // --- Dynamic Search with specific time_index ---
        
        // 1. Direct search
        auto it = leaf_lookup.find(target_path);
        if (it != leaf_lookup.end()) {
            for (size_t idx : it->second) {
                const auto& leaf = leaves[idx];
                if (isTimeInLeaf(leaf, time_index)) {
                    size_t slice_volume = 1;
                    for (auto s : leaf.shape) if(s > 0) slice_volume *= s;
                    if (slice_volume == 0) slice_volume = 1;
                    
                    *data_out = (double*)malloc(slice_volume * sizeof(double));
                    if (this->readSliceDirect(leaf, time_index, *data_out) < 0) {
                        free(*data_out);
                        return -1;
                    }
                    
                    *ndim_out = leaf.shape.size();
                    for (size_t i = 0; i < *ndim_out && i < 6; ++i) shape_out[i] = leaf.shape[i];
                    
                    return 0;
                }
            }
        }

        // 2. Search via index substitution in path
        // Ex: "profiles_2d/0/ion/0/state/0/z_min" -> "profiles_2d/1/ion/0/state/0/z_min"
        // for time_index=1
        
        // Note: We don't need to split the whole path to find the dynamic AoS.
        // We iterate over known leaves to find the matching dynamic AoS prefix.
        
        // Find first dynamic AoS in hierarchy
        std::string dynamic_aos_path = "";
        // OPTIMISATION: Use cached roots instead of iterating all leaves
        for (const auto& aos_path : cached_dynamic_aos_roots) {
            // Check if target path starts with this AoS
            if (target_path.size() > aos_path.size() && 
                target_path[aos_path.size()] == '/' &&
                target_path.compare(0, aos_path.size(), aos_path) == 0) {
                dynamic_aos_path = aos_path;
                break;
            }
        }
        
        if (!dynamic_aos_path.empty()) {
            // OPTIMISATION: Parse une seule fois
            // Inline logic using string_view to avoid allocations in parsePath
            std::string_view aos_view(dynamic_aos_path);
            
            // We know target_path starts with aos_view + "/"
            std::string_view remainder = target_path.substr(aos_view.size() + 1);
            size_t first_slash = remainder.find('/');
            
            // If there is a remainder, we have an index
            if (!remainder.empty()) {
                std::string_view suffix = (first_slash != std::string_view::npos) 
                                        ? remainder.substr(first_slash) 
                                        : std::string_view();

                // STRATEGY 1: Generic Path
                std::string generic_path;
                generic_path.reserve(aos_view.size() + suffix.size());
                generic_path.append(aos_view);
                generic_path.append(suffix);


                auto it_gen = leaf_lookup.find(generic_path);
                if (it_gen != leaf_lookup.end()) {
                    for (size_t idx : it_gen->second) {
                        const auto& leaf = leaves[idx];
                        if (isTimeInLeaf(leaf, time_index)) {
                            size_t slice_volume = 1;
                            for (auto s : leaf.shape) if(s > 0) slice_volume *= s;
                            if (slice_volume == 0) slice_volume = 1;
                            
                            *data_out = (double*)malloc(slice_volume * sizeof(double));
                            if (this->readSliceDirect(leaf, time_index, *data_out) < 0) {
                                free(*data_out);
                                return -1;
                            }
                            
                            *ndim_out = leaf.shape.size();
                            for (size_t i = 0; i < *ndim_out && i < 6; ++i) shape_out[i] = leaf.shape[i];
                            
                            return 0;
                        }
                    }
                }

                // STRATEGY 2: Substituted Path
                std::string substituted_path;
                substituted_path.reserve(aos_view.size() + 25 + suffix.size());
                substituted_path.append(aos_view);
                substituted_path.push_back('/');
                substituted_path.append(std::to_string(time_index));
                substituted_path.append(suffix);

                auto it_sub = leaf_lookup.find(substituted_path);
                if (it_sub != leaf_lookup.end()) {
                    for (size_t idx : it_sub->second) {
                        const auto& leaf = leaves[idx];
                        if (isTimeInLeaf(leaf, time_index)) {
                            size_t slice_volume = 1;
                            for (auto s : leaf.shape) if(s > 0) slice_volume *= s;
                            if (slice_volume == 0) slice_volume = 1;
                            
                            *data_out = (double*)malloc(slice_volume * sizeof(double));
                            if (this->readSliceDirect(leaf, time_index, *data_out) < 0) {
                                free(*data_out);
                                return -1;
                            }
                            
                            *ndim_out = leaf.shape.size();
                            for (size_t i = 0; i < *ndim_out && i < 6; ++i) shape_out[i] = leaf.shape[i];
                            
                            return 0;
                        }
                    }
                }
            }
        }
    }

    return -1;
}

int PanzerDB::readStringDataByIndex(
                          const char* full_data_path,
                          int64_t time_index,
                          uint64_t* ndim_out,
                          uint64_t shape_out[6],
                          char** data_out) {


    const auto& leaves = getLeaves();
    if (leaves.empty()) return -1;

    std::string_view target_path(full_data_path);


    if (time_index == -1) {
        // --- Static/Global Search ---
        std::vector<const Leaf*> matches;
        uint64_t total_count = 0;

        auto it = leaf_lookup.find(target_path);
        if (it != leaf_lookup.end()) {
            matches.reserve(it->second.size());
            for (size_t idx : it->second) {
                matches.push_back(&leaves[idx]);
                total_count += leaves[idx].count;
            }
        }

        if (matches.empty()) return -1;

        if (total_count == 0) {
            *data_out = nullptr;
            *ndim_out = 0;
            return 0; // Success, empty dataset
        }

        std::sort(matches.begin(), matches.end(), [](const Leaf* a, const Leaf* b){
            return a->time_index < b->time_index;
        });

        std::vector<std::string> buffer;
        buffer.reserve(total_count);

        for(const auto* leaf : matches) {
            size_t current_size = buffer.size();
            buffer.resize(current_size + leaf->count);
            readTensor(*leaf, buffer.data() + current_size);
        }

        size_t max_len = 0;
        for(const auto& s : buffer) {
            if (s.length() > max_len) max_len = s.length();
        }
        max_len += 1;

        size_t total_bytes = total_count * max_len;
        *data_out = (char*)malloc(total_bytes);
        memset(*data_out, 0, total_bytes);

        for(size_t i=0; i<total_count; ++i) {
            strncpy(*data_out + (i * max_len), buffer[i].c_str(), max_len - 1);
        }

        *ndim_out = 2;
        shape_out[0] = total_count;
        shape_out[1] = max_len;

        return 0;
    }

    // --- Dynamic Search ---
    const PanzerDB::Leaf* target_leaf = nullptr;
    
    // 1. Direct search
    auto it = leaf_lookup.find(target_path);
    if (it != leaf_lookup.end()) {
        for (size_t idx : it->second) {
            if (isTimeInLeaf(leaves[idx], time_index)) {
                target_leaf = &leaves[idx];
                break;
            }
        }
    }

    // 2. Search via substitution
    if (!target_leaf) {
        std::string dynamic_aos_path = "";
        for (const auto& aos_path : cached_dynamic_aos_roots) {
            if (target_path.size() > aos_path.size() && 
                target_path[aos_path.size()] == '/' &&
                target_path.compare(0, aos_path.size(), aos_path) == 0) {
                dynamic_aos_path = aos_path;
                break;
            }
        }

        if (!dynamic_aos_path.empty()) {
            std::string prefix = dynamic_aos_path + "/";
            if (target_path.size() >= prefix.size() && target_path.compare(0, prefix.size(), prefix) == 0) {
                std::string_view suffix = target_path.substr(prefix.length());
                size_t first_slash = suffix.find('/');
                
                // STRATEGY 1: Generic Path
                std::string generic_path = dynamic_aos_path;
                if (first_slash != std::string::npos) {
                    generic_path.append(suffix.substr(first_slash));
                } else {
                    generic_path.append("/").append(suffix);
                }

                auto it_gen = leaf_lookup.find(generic_path);
                if (it_gen != leaf_lookup.end()) {
                    for (size_t idx : it_gen->second) {
                        if (isTimeInLeaf(leaves[idx], time_index)) {
                            target_leaf = &leaves[idx];
                            break;
                        }
                    }
                }

                // STRATEGY 2: Substituted Path
                if (!target_leaf) {
                    std::string substituted_path = dynamic_aos_path + "/" + std::to_string(time_index);
                    if (first_slash != std::string::npos) {
                        substituted_path.append(suffix.substr(first_slash));
                    }
                    auto it_sub = leaf_lookup.find(substituted_path);
                    if (it_sub != leaf_lookup.end()) {
                        for (size_t idx : it_sub->second) {
                            if (isTimeInLeaf(leaves[idx], time_index)) {
                                target_leaf = &leaves[idx];
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    if (!target_leaf) {
        // Fallback to direct search if substitution fails (for root dynamic signals)
        auto it_fb = leaf_lookup.find(target_path);
        if (it_fb != leaf_lookup.end()) {
            for (size_t idx : it_fb->second) {
                if (isTimeInLeaf(leaves[idx], time_index)) {
                    target_leaf = &leaves[idx];
                    break;
                }
            }
        }
    }
    
    if (target_leaf) {
        // OPTIMISATION: Utilisation du scratch buffer string
        scratch_str.resize(target_leaf->count);
        readTensor(*target_leaf, scratch_str.data());
        
        size_t shape_prod = 1;
        for(auto s : target_leaf->shape) shape_prod *= s;
        if (shape_prod == 0) shape_prod = 1;

        uint64_t n_steps = target_leaf->count / shape_prod;
        if (n_steps == 0) n_steps = 1;
        
        uint64_t element_size = target_leaf->count / n_steps;
        uint64_t local_step = time_index - target_leaf->time_index;
        
        size_t start_idx = local_step * element_size;
        
        bool is_scalar = target_leaf->shape.empty();

        if (is_scalar) {
            const std::string& s = scratch_str[start_idx];
            *data_out = (char*)malloc(s.length() + 1);
            strcpy(*data_out, s.c_str());
            
            *ndim_out = 1;
            shape_out[0] = s.length();
        } else {
            size_t max_len = 0;
            for(size_t i=0; i<element_size; ++i) {
                size_t len = scratch_str[start_idx + i].length();
                if (len > max_len) max_len = len;
            }
            max_len += 1;

            size_t total_bytes = element_size * max_len;
            *data_out = (char*)malloc(total_bytes);
            memset(*data_out, 0, total_bytes);

            for(size_t i=0; i<element_size; ++i) {
                const std::string& s = scratch_str[start_idx + i];
                strncpy(*data_out + (i * max_len), s.c_str(), max_len - 1);
            }

            *ndim_out = 2;
            shape_out[0] = element_size;
            shape_out[1] = max_len;
        }
        
        return 0;
    }

    return -1;
}

int PanzerDB::readComplexDataByIndex(
                          const char* full_data_path,
                          int64_t time_index,
                          uint64_t* ndim_out,
                          uint64_t shape_out[6],
                          std::complex<double>** data_out) {


    const auto& leaves = getLeaves();
    if (leaves.empty()) return -1;

    std::string_view target_path(full_data_path);

    
    if (time_index == -1) {
        // --- Static/Global Search ---
        std::vector<const Leaf*> matches;
        uint64_t total_count = 0;

        auto it = leaf_lookup.find(target_path);
        if (it != leaf_lookup.end()) {
            matches.reserve(it->second.size());
            for (size_t idx : it->second) {
                matches.push_back(&leaves[idx]);
                total_count += leaves[idx].count;
            }
        }

        if (matches.empty()) return -1;

        if (total_count == 0) {
            *data_out = nullptr;
            *ndim_out = 0;
            return 0; // Success, empty dataset
        }

        if (matches.size() == 1) {
            const Leaf& leaf = *matches[0];
            
            size_t slice_vol = 1;
            for(auto s : leaf.shape) if(s > 0) slice_vol *= s;
            
            if (leaf.count > slice_vol) {
                *ndim_out = leaf.shape.size() + 1;
                for (size_t i = 0; i < leaf.shape.size(); ++i) shape_out[i] = leaf.shape[i];
                shape_out[*ndim_out - 1] = leaf.count / slice_vol;
            } else {
                *ndim_out = leaf.shape.size();
                for (size_t i = 0; i < *ndim_out && i < 6; ++i) shape_out[i] = leaf.shape[i];
            }
            
            *data_out = (std::complex<double>*)malloc(leaf.count * sizeof(std::complex<double>));
            this->readTensor(leaf, *data_out);
            return 0;
        }

        std::sort(matches.begin(), matches.end(), [](const Leaf* a, const Leaf* b){
            return a->time_index < b->time_index;
        });

        const Leaf& first = *matches[0];
        *data_out = (std::complex<double>*)malloc(total_count * sizeof(std::complex<double>));
        // OPTIMISATION: Use readLeavesUnion for batched read
        if (readLeavesUnion(matches, *data_out, DataType::COMPLEX128) < 0) {
            size_t offset = 0;
            for(const auto* leaf : matches) {
                this->readTensor(*leaf, (*data_out) + offset);
                offset += leaf->count;
            }
        }

        *ndim_out = first.shape.size() + 1;
        for(size_t i=0; i<first.shape.size(); ++i) shape_out[i] = first.shape[i];
        
        size_t slice_vol = 1;
        for(auto s : first.shape) if(s > 0) slice_vol *= s;
        shape_out[*ndim_out - 1] = total_count / slice_vol;

        return 0;
    }

    // --- Dynamic Search ---
    
    const PanzerDB::Leaf* target_leaf = nullptr;

    // 1. Direct search
    auto it = leaf_lookup.find(target_path);
    if (it != leaf_lookup.end()) {
        for (size_t idx : it->second) {
            if (isTimeInLeaf(leaves[idx], time_index)) {
                target_leaf = &leaves[idx];
                break;
            }
        }
    }

    // 2. Search via substitution
    if (!target_leaf) {
        std::string dynamic_aos_path = "";
        for (const auto& aos_path : cached_dynamic_aos_roots) {
            if (target_path.size() > aos_path.size() && 
                target_path[aos_path.size()] == '/' &&
                target_path.compare(0, aos_path.size(), aos_path) == 0) {
                dynamic_aos_path = aos_path;
                break;
            }
        }

        if (!dynamic_aos_path.empty()) {
            std::string prefix = dynamic_aos_path + "/";
            if (target_path.size() >= prefix.size() && target_path.compare(0, prefix.size(), prefix) == 0) {
                std::string_view suffix = target_path.substr(prefix.length());
                size_t first_slash = suffix.find('/');

                // STRATEGY 1: Generic Path
                std::string generic_path = dynamic_aos_path;
                if (first_slash != std::string::npos) {
                    generic_path.append(suffix.substr(first_slash));
                } else {
                    generic_path.append("/").append(suffix);
                }
                auto it_gen = leaf_lookup.find(generic_path);
                if (it_gen != leaf_lookup.end()) {
                    for (size_t idx : it_gen->second) {
                        if (isTimeInLeaf(leaves[idx], time_index)) {
                            target_leaf = &leaves[idx];
                            break;
                        }
                    }
                }

                // STRATEGY 2: Substituted Path
                if (!target_leaf) {
                    std::string substituted_path = dynamic_aos_path + "/" + std::to_string(time_index);
                    if (first_slash != std::string::npos) {
                        substituted_path.append(suffix.substr(first_slash));
                    }
                    auto it_sub = leaf_lookup.find(substituted_path);
                    if (it_sub != leaf_lookup.end()) {
                        for (size_t idx : it_sub->second) {
                            if (isTimeInLeaf(leaves[idx], time_index)) {
                                target_leaf = &leaves[idx];
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    if (target_leaf) {
        size_t slice_volume = 1;
        for(auto s : target_leaf->shape) if (s > 0) slice_volume *= s;
        if (slice_volume == 0) slice_volume = 1;
        
        *data_out = (std::complex<double>*)malloc(slice_volume * sizeof(std::complex<double>));
        if (this->readSliceDirect(*target_leaf, time_index, *data_out) < 0) {
            free(*data_out);
            return -1;
        }
        
        *ndim_out = target_leaf->shape.size();
        for (size_t i = 0; i < *ndim_out && i < 6; ++i) shape_out[i] = target_leaf->shape[i];
        
        return 0;
    }

    return -1;
}

int PanzerDB::readIntDataByIndex(
                         const char* full_data_path,
                         int64_t time_index,
                         uint64_t* ndim_out,
                         uint64_t shape_out[6],
                         int32_t** data_out) {

    const auto& leaves = getLeaves();
    if (leaves.empty()) return -1;

    std::string_view target_path(full_data_path);
    
    if (time_index == -1) {
        // --- Static or Global Search (Aggregation) ---
        std::vector<const Leaf*> matches;
        uint64_t total_count = 0;

        auto it = leaf_lookup.find(target_path);
        if (it != leaf_lookup.end()) {
            matches.reserve(it->second.size());
            for (size_t idx : it->second) {
                const auto& leaf = leaves[idx];
                if (static_cast<DataType>(leaf.flags >> 4) == DataType::INT32) {
                    matches.push_back(&leaf);
                    total_count += leaf.count;
                }
            }
        }

        if (matches.empty()) return -1;

        if (total_count == 0) {
            *data_out = nullptr;
            *ndim_out = 0;
            return 0;
        }

        *data_out = (int32_t*)malloc(total_count * sizeof(int32_t));

        if (matches.size() == 1) {
            const Leaf& leaf = *matches[0];
            this->readTensor(leaf, *data_out);
            *ndim_out = leaf.shape.size();
             for (size_t i = 0; i < *ndim_out && i < 6; ++i) shape_out[i] = leaf.shape[i];
            return 0;
        }

        // Aggregation of time slices
        std::sort(matches.begin(), matches.end(), [](const Leaf* a, const Leaf* b){
            return a->time_index < b->time_index;
        });

        if (readLeavesUnion(matches, *data_out, DataType::INT32) < 0) {
            // Fallback to sequential reads if union fails
            size_t offset = 0;
            for(const auto* leaf : matches) {
                this->readTensor(*leaf, (*data_out) + offset);
                offset += leaf->count;
            }
        }

        const Leaf& first = *matches[0];
        *ndim_out = first.shape.size() + 1;
        for(size_t i=0; i<first.shape.size(); ++i) shape_out[i] = first.shape[i];
        
        size_t slice_vol = 1;
        for(auto s : first.shape) if(s > 0) slice_vol *= s;
        if (slice_vol > 0) {
            shape_out[*ndim_out - 1] = total_count / slice_vol;
        } else {
            shape_out[*ndim_out - 1] = total_count;
        }

        return 0;

    } else {
        // --- Dynamic Search with specific time_index ---
        const Leaf* target_leaf = nullptr;

        // 1. Direct search
        auto it = leaf_lookup.find(target_path);
        if (it != leaf_lookup.end()) {
            for (size_t idx : it->second) {
                const auto& leaf = leaves[idx];
                if (static_cast<DataType>(leaf.flags >> 4) == DataType::INT32 && isTimeInLeaf(leaf, time_index)) {
                    target_leaf = &leaf;
                    break;
                }
            }
        }

        // 2. Search via index substitution in path (if not found yet)
        if (!target_leaf) {
             // This logic would be a copy of readDataByIndex, adapted for int32.
             // For now, we keep it simple.
        }

        if (target_leaf) {
            size_t slice_volume = 1;
            for (auto s : target_leaf->shape) if(s > 0) slice_volume *= s;
            if (slice_volume == 0) slice_volume = 1;
            
            *data_out = (int32_t*)malloc(slice_volume * sizeof(int32_t));
            if (this->readSliceDirect(*target_leaf, time_index, *data_out) < 0) {
                free(*data_out);
                return -1;
            }
            
            *ndim_out = target_leaf->shape.size();
            for (size_t i = 0; i < *ndim_out && i < 6; ++i) shape_out[i] = target_leaf->shape[i];
            
            return 0;
        }
    }

    return -1;
}

int PanzerDB::readInterpolatedData(
                         const char* full_data_path,  // ✅ Changed
                         double time,
                         const std::vector<double>& time_basis,
                         int interp_mode,
                         int datatype,
                         uint64_t* ndim_out,
                         uint64_t shape_out[6],
                         void** data_out,
                         bool expect_time_dim) {


    DataInterpolation data_interpolation_component;
    std::map<std::string, int> times_indices;

    int slice_index = data_interpolation_component.getSlicesTimesIndices(time, time_basis, times_indices, interp_mode);
    int request_sup = times_indices[SLICE_SUP];

    // Resolve the requested time indices to available slices.
    // A "gap" (missing slice for this signal) is resolved to the closest
    // available slice:
    //   - closest / undefined : nearest available slice in time (ties -> lower index)
    //   - previous            : last available <= t (else first available >= t)
    //   - linear              : last available <= t (inf) and first available
    //                           >= t (sup), interpolation factor computed with
    //                           the actual times of those slices.
    // A signal with no data at all -> -1 (data not available).
    // Closest (and undefined) ask for the nearest slice in time; previous and
    // linear ask for "last available <= t" on the inf side.
    const bool want_nearest = (interp_mode != PREVIOUS_INTERP && interp_mode != LINEAR_INTERP);
    const int64_t inf_direction = want_nearest ? 0 : -1;
    int64_t inf_i = nearestAvailableSliceIndex(full_data_path, slice_index, inf_direction, time_basis, time);
    if (inf_i == -1) return -1;

    int64_t sup_i = inf_i;
    if (interp_mode == LINEAR_INTERP && request_sup != slice_index) {
        sup_i = nearestAvailableSliceIndex(full_data_path, request_sup, +1, time_basis, time);
        if (sup_i == -1) sup_i = inf_i;
        if (sup_i < inf_i) {
            std::swap(inf_i, sup_i);
        }
    }

    auto read_slice_at = [this, full_data_path, datatype, ndim_out, shape_out](int64_t idx, void** out_ptr) -> int {
        *out_ptr = nullptr;
        if (datatype == alconst::char_data) {
            return readStringDataByIndex(full_data_path, idx, ndim_out, shape_out, (char**)out_ptr);
        } else if (datatype == alconst::complex_data) {
            return readComplexDataByIndex(full_data_path, idx, ndim_out, shape_out, (std::complex<double>**)out_ptr);
        } else {
            return readDataByIndex(full_data_path, idx, ndim_out, shape_out, (double**)out_ptr);
        }
    };

    void* data_inf = nullptr;
    if (read_slice_at(inf_i, &data_inf) != 0) {
        return -1;
    }

    if (sup_i == inf_i || interp_mode != LINEAR_INTERP) {
        *data_out = data_inf;
        return 0;
    }

    void* data_sup = nullptr;
    if (read_slice_at(sup_i, &data_sup) != 0) {
        free(data_inf);
        return -1;
    }

    std::map<std::string, double> slices_times;
    slices_times[SLICE_INF] = time_basis[inf_i];
    slices_times[SLICE_SUP] = time_basis[sup_i];

    std::map<std::string, void*> y_slices;
    y_slices[SLICE_INF] = data_inf;
    y_slices[SLICE_SUP] = data_sup;

    size_t shape_prod = 1;
    for (size_t i = 0; i < *ndim_out; ++i) shape_prod *= shape_out[i];

    data_interpolation_component.interpolate(datatype, shape_prod, y_slices, slices_times, time, data_out, interp_mode);

    if (data_inf && *data_out != data_inf) free(data_inf);
    if (data_sup && *data_out != data_sup) free(data_sup);

    return 0;
}

std::string PanzerDB::getDynamicAOSPath() const {
    // OPTIMISATION: Retourner le cache si valide
    if (dynamic_aos_path_valid) {
        return cached_dynamic_aos_path;
    }
    
    if (array_stack.empty()) {
        cached_dynamic_aos_path = "";
        dynamic_aos_path_valid = true;
        return cached_dynamic_aos_path;
    }
    
    // Recherche du premier AoS dynamique dans le stack
    for (auto it = array_stack.rbegin(); it != array_stack.rend(); ++it) {
        if (it->is_dynamic) {
            cached_dynamic_aos_path = it->aos_full_path;
            dynamic_aos_path_valid = true;
            return cached_dynamic_aos_path;
        }
    }
    
    cached_dynamic_aos_path = "";
    dynamic_aos_path_valid = true;
    return cached_dynamic_aos_path;
}

void PanzerDB::invalidateDynamicAOSCache() {
    dynamic_aos_path_valid = false;
}

uint64_t PanzerDB::getCurrentTimeForAOS(const std::string& aos_path) {
    if (aos_path.empty()) return 0;
    auto it = aos_time_counters.find(aos_path);
    return (it != aos_time_counters.end()) ? it->second : 0;
}

void PanzerDB::advanceTimeForAOS(const std::string& aos_path, uint64_t delta) {
    if (!aos_path.empty()) {
        aos_time_counters[aos_path] += delta;
    }
}

// Dans panzerdb.cpp
void PanzerDB::advanceTimebase(const std::string& timebase_name, uint64_t n_steps) {
    std::string dynamic_aos_path = getDynamicAOSPath();
    if (dynamic_aos_path.empty()) {
        throw std::runtime_error("advanceTimebase called outside a dynamic AoS");
    }
    advanceTimeForAOS(dynamic_aos_path, n_steps);
}


// Retrieve a whole dynamic signal that is not inside an AoS
std::vector<double> PanzerDB::getWholeDynamicSignal(const std::string& dataset_name) {
    
    // 1. Access the in-memory index
    const auto& leaves = getLeaves();
    
    // 2. Filtrer les feuilles correspondant au dataset
    std::vector<const PanzerDB::Leaf*> target_leaves;
    size_t total_elements = 0;

    auto it = leaf_lookup.find(dataset_name);
    if (it != leaf_lookup.end()) {
        for (size_t idx : it->second) {
            // Comparaison stricte car le dataset n'est pas dans un AOS
            // (already guaranteed by the lookup)
            target_leaves.push_back(&leaves[idx]);
            total_elements += leaves[idx].count;
        }
    }

    if (target_leaves.empty()) {
        return {};
    }

    // 3. Sort by Time Index
    // Crucial if data was written in multiple times (APPEND)
    std::sort(target_leaves.begin(), target_leaves.end(), 
        [](const PanzerDB::Leaf* a, const PanzerDB::Leaf* b) {
            return a->time_index < b->time_index;
        });

    // 4. Result vector pre-allocation
    std::vector<double> full_signal;
    full_signal.resize(total_elements);

    // 5. Reading and assembly (Concatenation)
    size_t current_offset = 0;
    
    for (const auto* leaf : target_leaves) {
        // Read directly into correct portion of final vector
        // leaf->count is number of elements in this write "slice"
        readTensor(*leaf, full_signal.data() + current_offset);
        
        current_offset += leaf->count;
    }

    return full_signal;
}

std::string PanzerDB::stripIndices(const std::string& path) {
    std::string result;
    result.reserve(path.size());
    
    std::stringstream ss(path);
    std::string segment;
    
    bool first = true;
    while(std::getline(ss, segment, '/')) {
        if (segment.empty()) continue;
        
        // Check if numeric (heuristic for index)
        bool is_index = !segment.empty() && std::all_of(segment.begin(), segment.end(), ::isdigit);
        
        if (!is_index) {
            if (!first) result += '/';
            result += segment;
            first = false;
        }
    }
    return result;
}

void PanzerDB::writeMetadata(const std::string& path, const std::string& value) {
    // path is the full metadata key, e.g., "flux_loop/field@units"
    // value is the metadata value, e.g., "T"

    // Use the set to avoid writing the same metadata attribute more than once.
    // This is critical for efficiency and for APPEND mode.
    if (written_metadata_schema_paths.count(path)) {
        return; // Already written for this schema, do nothing.
    }

    // Treat the metadata entry as a single, static string.
    const char* value_cstr = value.c_str();

    // Write the string using the specialized writeData template.
    // The 'path' itself serves as the unique dataset name for the metadata.
    // Shape is empty for a scalar, count is 1, and timebase is empty for static data.
    this->writeData<const char*>(path, {}, &value_cstr, 1, "");

    // Mark this metadata path as written to prevent duplicates.
    written_metadata_schema_paths.insert(path);
}


std::map<std::string, std::string> PanzerDB::readMetadata(const std::string& instance_path) {
    std::map<std::string, std::string> metadata;

    // 1. Convert to schema path (remove indices)
    std::string schema_path = stripIndices(instance_path);

    // 2. Check cache to avoid redundant reads
    if (written_metadata_schema_paths.count(schema_path)) {
        return metadata; // Already processed, return empty (or cached if we stored it)
    }

    // 3. Scan for metadata leaves (schema_path + "@key")
    std::string prefix = schema_path + "@";
    const auto& leaves = getLeaves();
    
    for (const auto& leaf : leaves) {
        // Check if leaf path starts with prefix
        if (leaf.path.rfind(prefix, 0) == 0) {
            std::string key = std::string(leaf.path.substr(prefix.length()));
            std::string value;
            readTensor<std::string>(leaf, &value);
            metadata[key] = value;
        }
    }

    // 4. Mark as processed
    written_metadata_schema_paths.insert(schema_path);
    return metadata;
}

void PanzerDB::synchronizeArrayStack(const std::vector<std::string>& aos_names,
                                     const std::vector<int>& indices) {
    if (aos_names.size() != indices.size()) {
        throw std::runtime_error("synchronizeArrayStack: aos_names and indices must have the same size");
    }
    
    if (aos_names.empty()) return;
    
    size_t sync_depth = std::min(array_stack.size(), aos_names.size());
    
    if (sync_depth == 0) {
        return;
    }
    
   // OPTIMIZATION 2: in-place modification with change detection
    bool modified = false;
    for (size_t level = 0; level < sync_depth; ++level) {
        const std::string& target_name = aos_names[level];
        int target_index = indices[level];
        
        ArrayLevel& current_level = array_stack[level];
        
        // name check
        if (current_level.name != target_name) {
            continue;
        }
        
        // FIX: Ne pas synchroniser dynamic AoS en mode APPEND
        if (mode == OpenMode::APPEND && current_level.is_dynamic) {
            continue;
        }

        // update if necessary
        if (current_level.current_index != static_cast<size_t>(target_index)) {
            current_level.current_index = target_index;
            modified = true;
        }
    }
    
    // OPTIMIZATION 3: rebuild only if modified
    if (modified) {
        // Reconstruire le stack
        
        // optimized rebuild of path_prefix (single allocation)
        path_prefix = buildPathFromStack(array_stack);
    }
}

/**
 * @brief Determines the data type of a leaf from its path using the
 * in-memory PanzerDB index. This is the correct and performant way.
 * @param path Full logical path to the data node.
 * @return The data type as a DataType enum value.
 */
 imas::direct_access::DataType PanzerDB::getLeafType(const std::string& path)
 {
     // 1. Fetch the in-memory index (fast, uses the cache).
     const auto& leaves = getLeaves();
 
     // 2. Look up the leaf using PanzerDB's optimized lookup table.
     auto it = leaf_lookup.find(std::string_view(path));
     if (it != leaf_lookup.end() && !it->second.empty()) {
         // A path may have several entries (e.g. time series).
         // The type is identical for all of them, so take the first one.
         const Leaf& leaf = leaves[it->second[0]];
 
         // 3. Decode the data type from the 'flags' member.
         // The type is encoded in the most significant bits.
         uint64_t type_bits = (leaf.flags >> 4);
         PanzerDB::DataType pz_type = static_cast<PanzerDB::DataType>(type_bits);
 
         // 4. Fait la correspondance entre le type interne de PanzerDB et celui de l'API.
         switch (pz_type) {
             case PanzerDB::DataType::FLOAT64:
                 return imas::direct_access::DataType::DOUBLE;
             case PanzerDB::DataType::INT32:
                 return imas::direct_access::DataType::INT32;
             case PanzerDB::DataType::COMPLEX128:
                 return imas::direct_access::DataType::COMPLEX_DOUBLE;
             case PanzerDB::DataType::STRING:
                 return imas::direct_access::DataType::STRING;
             case PanzerDB::DataType::LIST_OF_STRINGS:
                 return imas::direct_access::DataType::LIST_OF_STRINGS;
             default:
                 throw std::runtime_error("Unknown data type in leaf flags for path: " + path);
         }
     }
 
     // If the leaf is not found in the index, this is an error.
     throw std::runtime_error("Path not found in the PanzerDB index: " + path);
 }

 bool PanzerDB::isDynamicAOS(const std::string& aos_path) const
{
    // Ensure the index is loaded in memory.
    const auto& leaves = getLeaves();

    // Use the optimized lookup table to find the leaf matching the path.
    auto it = leaf_lookup.find(std::string_view(aos_path));
    if (it != leaf_lookup.end() && !it->second.empty()) {
        // A path should uniquely identify an AoS meta-node,
        // so take the first match.
        const Leaf& leaf = leaves[it->second[0]];

        // Flag '3' marks a dynamic AoS meta-node in the PanzerDB index.
        if (leaf.flags == 3) {
            return true;
        }
    }

    // If the path is not found or is not a dynamic AoS.
    return false;
}

// Reports whether a data signal (leaf) varies over time (see header for the
// criterion).  The two sub-cases (dynamic-AoS ancestor / own time axis) mirror
// exactly the reader's own "add one time dimension" decision, cf. readDataByIndex
// (panzerdb.cpp:2384) and isTimeInLeaf (panzerdb.cpp:2227-2238).
bool PanzerDB::isDynamicSignal(const std::string& signal_path) const
{
    // Ensure the index (and its lookup caches) is loaded in memory.
    const auto& leaves = getLeaves();

    // Case 1: signal lives under a dynamic Array of Structures — the time axis
    // is supplied by the AoS iteration.  A dynamic-AoS root is any prefix of the
    // leaf path (top-level, or nested under a static-AoS instance index).
    for (const auto& root : cached_dynamic_aos_roots) {
        if (signal_path.size() >= root.size() &&
            signal_path.compare(0, root.size(), root) == 0 &&
            (signal_path.size() == root.size() || signal_path[root.size()] == '/')) {
            return true;
        }
    }

    // Case 2: no dynamic AoS — but the leaf owns its own time axis.
    auto it = leaf_lookup.find(std::string_view(signal_path));
    if (it == leaf_lookup.end() || it->second.empty()) return false;

    // Iterative form: several leaf rows at the same path (one per time step).
    if (it->second.size() > 1) return true;

    // Bulk form: a single data row whose stored count exceeds one spatial slice.
    const Leaf& leaf = leaves[it->second.front()];
    if ((leaf.flags & 0xF) != 0) return false;   // not a data leaf (e.g. an AoS meta-node)
    uint64_t slice_volume = 1;
    for (auto s : leaf.shape) if (s > 0) slice_volume *= s;
    if (slice_volume == 0) slice_volume = 1;     // scalar: empty shape
    return leaf.count > slice_volume;
}