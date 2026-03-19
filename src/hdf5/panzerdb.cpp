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
 *       `buildTimeIndex` creates an O(log n) lookup structure for fast time-based queries.
 *
 * 4.  **Path Substitution**:
 *     - When reading dynamic data (e.g., `pz_readData_by_index`), the code must often
 *       translate a logical path like `A/0/B` (requested at time `t=1`) into the
 *       actual stored path `A/1/B`. This is handled by finding the "Dynamic AoS Root"
 *       and substituting the index.
 */

constexpr size_t PATH_MAX_LEN = 256;

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

    std::string usage_hint = "time_series";  // Défaut
    if (mode == OpenMode::READ) {
        usage_hint = "interactive";  // Mode lecture privilégie accès rapide
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
        
         // String dataset (compression less effective, but still useful)
        hid_t str_type_vl = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_type_vl, H5T_VARIABLE);
        H5Tset_cset(str_type_vl, H5T_CSET_UTF8);
        data_dset_str = createOptimizedDataset("data_raw_str", str_type_vl, 
                                                chunk_config.data_chunk_str, false, dapl);
        H5Tclose(str_type_vl);
        
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
        parent_paths_dset = createOptimizedDataset("parent_paths", str_type, 
                                                   chunk_config.path_chunk_entries, true, dapl);
        H5Tclose(str_type);
        
        // Buffers optimisés
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
        if (H5Lexists(file_id, "data_raw_f64", H5P_DEFAULT) > 0) data_dset_f64 = H5Dopen2(file_id, "data_raw_f64", dapl);
        if (H5Lexists(file_id, "data_raw_i32", H5P_DEFAULT) > 0) data_dset_i32 = H5Dopen2(file_id, "data_raw_i32", dapl);
        if (H5Lexists(file_id, "paths", H5P_DEFAULT) > 0) paths_dset = H5Dopen2(file_id, "paths", dapl);
        if (H5Lexists(file_id, "data_raw_str", H5P_DEFAULT) > 0) data_dset_str = H5Dopen2(file_id, "data_raw_str", dapl);
        if (H5Lexists(file_id, "data_raw_c128", H5P_DEFAULT) > 0) data_dset_c128 = H5Dopen2(file_id, "data_raw_c128", dapl);
        if (H5Lexists(file_id, "parent_paths", H5P_DEFAULT) > 0) parent_paths_dset = H5Dopen2(file_id, "parent_paths", dapl);
        leaves_cache_valid = false;
        
        // NOUVEAU: Lire la configuration de chunking du fichier existant
        readChunkingConfig();
        updateDiskSizes();
        
        // Buffers optimisés
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
        if (H5Lexists(file_id, "parent_paths", H5P_DEFAULT) > 0) parent_paths_dset = H5Dopen2(file_id, "parent_paths", H5P_DEFAULT);
        leaves_cache_valid = false;
        
        // NOUVEAU: Lire la configuration de chunking
        readChunkingConfig();
        
        // NOUVEAU: Configurer le cache HDF5 pour lectures optimales
        configureReadCache();

        //dumpLeavesCache();
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
// 5. LECTURE DE LA CONFIGURATION DE CHUNKING
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
        
        // Vérifier si compression est active
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
// 6. CONFIGURATION DU CACHE DE LECTURE
// ============================================================================

void PanzerDB::configureReadCache() {
    // Configurer le cache HDF5 pour optimiser les lectures via DAPL (Dataset Access Property List)
    // Cela permet d'appliquer le cache sur les datasets déjà ouverts ou à ouvrir.

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
    
    // Fonction helper pour réouvrir un dataset avec le nouveau DAPL
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
    reopen_dataset(parent_paths_dset, "parent_paths");

    H5Pclose(dapl);

    /*std::cout << "[PanzerDB] Read cache configured (DAPL applied to datasets):" << std::endl;
    std::cout << "  Chunk cache: " << (chunk_config.chunk_cache_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Cache slots: " << chunk_config.chunk_cache_nslots << std::endl;*/
}

// ============================================================================
// 2. DÉTECTION AUTOMATIQUE DU PATTERN D'USAGE
// ============================================================================

void PanzerDB::configureChunking(const std::string& usage_hint) {
    // Patterns d'usage typiques:
    // - "time_series": Beaucoup d'écritures séquentielles, lectures par slice temporel
    // - "array_of_structures": Beaucoup d'AoS imbriqués, accès par structure
    // - "bulk_write": Écriture massive unique, lecture rare
    // - "interactive": Lectures/écritures fréquentes et petites
    
    if (usage_hint == "time_series") {
        // Optimisé pour écriture séquentielle et lecture par slice temporel
        chunk_config.index_chunk_rows = 4096;      // Plus petit pour meilleur accès
        chunk_config.data_chunk_f64 = 65536;       // 512 KB par chunk
        chunk_config.data_chunk_i32 = 131072;      // 512 KB par chunk
        chunk_config.data_chunk_c128 = 32768;      // 512 KB par chunk (16 bytes/elem)
        chunk_config.data_chunk_str = 8192;        // ~64-128 KB (pointeurs)
        chunk_config.path_chunk_entries = 4096;    // ~1 MB (256 bytes/entry)
        chunk_config.enable_compression = true;
        chunk_config.compression_level = 1;        // Compression légère
        
    } else if (usage_hint == "array_of_structures") {
        // Optimisé pour AoS avec beaucoup de petites structures
        chunk_config.index_chunk_rows = 16384;     // Plus gros index chunks
        chunk_config.data_chunk_f64 = 32768;       // Plus petits data chunks
        chunk_config.data_chunk_i32 = 65536;
        chunk_config.data_chunk_c128 = 16384;      // 256 KB
        chunk_config.data_chunk_str = 4096;
        chunk_config.path_chunk_entries = 4096;
        chunk_config.enable_compression = true;
        chunk_config.compression_level = 1;
        
    } else if (usage_hint == "bulk_write") {
        // Optimisé pour écriture massive
        chunk_config.index_chunk_rows = 32768;     // Très gros chunks
        chunk_config.data_chunk_f64 = 524288;      // 4 MB par chunk
        chunk_config.data_chunk_i32 = 1048576;     // 4 MB par chunk
        chunk_config.data_chunk_c128 = 262144;     // 4 MB par chunk
        chunk_config.data_chunk_str = 65536;
        chunk_config.path_chunk_entries = 16384;   // ~4 MB
        chunk_config.enable_compression = true;
        chunk_config.compression_level = 1;        // Compression légère (vitesse)
        
    } else if (usage_hint == "interactive") {
        // Optimisé pour accès fréquents et petits
        chunk_config.index_chunk_rows = 1024;      // Petits chunks
        chunk_config.data_chunk_f64 = 8192;        // 64 KB par chunk
        chunk_config.data_chunk_i32 = 16384;       // 64 KB par chunk
        chunk_config.data_chunk_c128 = 4096;       // 64 KB par chunk
        chunk_config.data_chunk_str = 1024;
        chunk_config.path_chunk_entries = 1024;    // ~256 KB
        chunk_config.enable_compression = false;   // Pas de compression
        
    } else {
        // Configuration par défaut (équilibrée)
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
// 3. CRÉATION OPTIMISÉE DES DATASETS
// ============================================================================

hid_t PanzerDB::createOptimizedDataset(const std::string& name,
                                        hid_t type,
                                        size_t chunk_size,
                                        bool enable_compression,
                                        hid_t dapl) {
    // Créer le dataspace (1D, extensible)
    hsize_t dims[1] = {0};
    hsize_t maxdims[1] = {H5S_UNLIMITED};
    hid_t space = H5Screate_simple(1, dims, maxdims);
    
    // Créer et configurer le property list
    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    
    // OPTIMISATION 1: Chunking
    hsize_t chunk[1] = {chunk_size};
    H5Pset_chunk(plist, 1, chunk);
    
    // OPTIMISATION 2: Compression (GZIP)
    if (enable_compression && chunk_config.enable_compression) {
        H5Pset_deflate(plist, chunk_config.compression_level);
        
        // Shuffle filter (améliore compression pour données numériques)
        H5T_class_t type_class = H5Tget_class(type);
        if (type_class == H5T_INTEGER || type_class == H5T_FLOAT || type_class == H5T_ARRAY) {
            H5Pset_shuffle(plist);
        }
    }
    
    // OPTIMISATION 3: Fillvalue (évite initialisation coûteuse)
    if (H5Tget_class(type) == H5T_STRING && H5Tis_variable_str(type) > 0) {
        // For variable-length strings, it's mandatory to set a fill value.
        // Setting it to NULL is the standard way to indicate no fill.
        const char* fill_ptr = nullptr;
        H5Pset_fill_value(plist, type, &fill_ptr);
    } else {
        H5Pset_fill_time(plist, H5D_FILL_TIME_NEVER);
    }
    // OPTIMISATION 4: Allocation strategy
    // Allouer l'espace progressivement plutôt que tout d'un coup
    H5Pset_alloc_time(plist, H5D_ALLOC_TIME_INCR);
    
    // Créer le dataset
    hid_t dset = H5Dcreate2(file_id, name.c_str(), type, space, 
                            H5P_DEFAULT, plist, dapl);
    
    // Cleanup
    H5Pclose(plist);
    H5Sclose(space);
    
    return dset;
}

// ============================================================================
// 7. API PUBLIQUE POUR CONFIGURATION
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
// 8. STATISTIQUES DE CHUNKING
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
        
        // Taille compressée (approximation)
        if (chunk_config.enable_compression) {
            // Ratio typique pour données numériques avec gzip
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

// Construction optimisée du path_prefix depuis le stack
std::string PanzerDB::buildPathFromStack(const std::vector<ArrayLevel>& stack_vector) const {
    if (stack_vector.empty()) return "";
    
    // OPTIMISATION 1: Pré-calculer la taille totale pour une seule allocation
    size_t total_size = 0;
    for (size_t i = 0; i < stack_vector.size(); ++i) {
        total_size += stack_vector[i].name.length();
        if (i > 0) {
            // Longueur max d'un uint64_t en décimal = 20 caractères
            total_size += 20;  // Pour l'index (ex: "/12345")
        }
        total_size += 1;  // Pour le '/'
    }
    
    // OPTIMISATION 2: Allocation unique
    std::string result;
    result.reserve(total_size + 10); // +10 de marge
    
    // OPTIMISATION 3: Construction sans réallocation
    for (size_t i = 0; i < stack_vector.size(); ++i) {
        if (i > 0) {
            // Ajouter l'index du niveau précédent
            result += '/';
            // Utiliser to_string qui est optimisé
            result += std::to_string(stack_vector[i-1].current_index);
        }
        if (!result.empty()) {
            result += '/';
        }
        result += stack_vector[i].name;
    }
    
    return result;
}

void PanzerDB::rebuildPathPrefix() {
    if (!path_prefix_dirty) return;
    
    // Avec vector, array_stack est déjà dans le bon ordre (racine -> feuille)
    path_prefix = buildPathFromStack(array_stack);
    path_prefix_dirty = false;
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

// Helper to find dynamic AOS parent of a leaf
std::string PanzerDB::findDynamicAOSParent(const std::string& parent_path,
                                           const std::vector<Leaf>& leaves) {
    // OPTIMISATION: Utiliser le cache des racines dynamiques au lieu de parcourir toutes les feuilles
    std::string best_match = "";
    for (const auto& aos_path : cached_dynamic_aos_roots) {
        if (parent_path.find(aos_path) == 0) {
            // Vérifier frontières (ex: "A/B" match "A" mais pas "A_suffix")
            if (parent_path.length() == aos_path.length() || parent_path[aos_path.length()] == '/') {
                if (aos_path.length() > best_match.length()) best_match = aos_path;
            }
        }
    }
    return best_match;
}

PanzerDB::~PanzerDB() { close(); }

void PanzerDB::flush() {
    if (index_buffer.empty()) return;

    // --- Flush Index Buffer ---
    hsize_t n_new_rows = index_buffer.size() / 14;
    hid_t filespace = H5Dget_space(index_dset);
    hsize_t current_dims[2];
    H5Sget_simple_extent_dims(filespace, current_dims, NULL);
    H5Sclose(filespace);

    // IMPORTANT: Extend dimensions, do not replace them
    hsize_t new_dims[2] = {current_dims[0] + n_new_rows, 14};
    H5Dset_extent(index_dset, new_dims);

    filespace = H5Dget_space(index_dset);
    hsize_t offset[2] = {current_dims[0], 0};
    hsize_t slab_dims[2] = {n_new_rows, 14};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, slab_dims, NULL);

    hsize_t mem_dims[2] = {n_new_rows, 14};
    hid_t memspace = H5Screate_simple(2, mem_dims, NULL);
    H5Dwrite(index_dset, H5T_NATIVE_UINT64, memspace, filespace, H5P_DEFAULT, index_buffer.data());
    H5Sclose(memspace);
    H5Sclose(filespace);

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

    // --- Flush Parent Paths Buffer ---
    if (!parent_paths_buffer.empty()) {
        hsize_t n_new_paths = parent_paths_buffer.size() / PATH_MAX_LEN;
        filespace = H5Dget_space(parent_paths_dset);
        hsize_t current_parent_dims[1];
        H5Sget_simple_extent_dims(filespace, current_parent_dims, NULL);
        H5Sclose(filespace);

        hsize_t new_parent_dims[1] = {current_parent_dims[0] + n_new_paths};
        H5Dset_extent(parent_paths_dset, new_parent_dims);
        
        filespace = H5Dget_space(parent_paths_dset);
        hsize_t parent_offset[1] = {current_parent_dims[0]};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, parent_offset, NULL, &n_new_paths, NULL);
        memspace = H5Screate_simple(1, &n_new_paths, NULL);
        H5Dwrite(parent_paths_dset, H5Dget_type(parent_paths_dset), memspace, filespace, H5P_DEFAULT, parent_paths_buffer.data());
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

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

        // Conversion de std::vector<std::string> en char*[] pour HDF5
        std::vector<const char*> c_str_vector;
        c_str_vector.reserve(data_buffer_str.size());
        for (const auto& s : data_buffer_str) {
            c_str_vector.push_back(s.c_str());
        }

        hid_t str_type_vl = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_type_vl, H5T_VARIABLE);
        H5Tset_cset(str_type_vl, H5T_CSET_UTF8);
        H5Dwrite(data_dset_str, str_type_vl, memspace, filespace, H5P_DEFAULT, c_str_vector.data());
        H5Tclose(str_type_vl);
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

    // OPTIMISATION: Ne pas forcer le flush disque à chaque appel.
    // Cela permet à l'OS et à HDF5 d'optimiser les écritures en cache.
    // H5Fflush(file_id, H5F_SCOPE_GLOBAL);
    
    index_buffer.clear();
    data_buffer_f64.clear();
    data_buffer_i32.clear();
    data_buffer_str.clear();
    data_buffer_c128.clear();
    paths_buffer.clear();
    parent_paths_buffer.clear();
    leaves_cache_valid = false;
    
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


// 2. CONSTRUCTION DE L'INDEX TEMPOREL (une seule fois)
// ============================================================================

void PanzerDB::buildTimeIndex() const {
    if (time_index_valid) return;
    
    time_range_index.clear();
    leaf_metadata_cache.clear();
    
    const auto& leaves = getLeaves();
    leaf_metadata_cache.resize(leaves.size());
    
    for (size_t i = 0; i < leaves.size(); ++i) {
        const auto& leaf = leaves[i];
        
        // Skip meta-nodes (AoS)
        if ((leaf.flags & 0xF) != 0) continue;
        
        // Calculer métadonnées UNE SEULE FOIS
        size_t slice_volume = 1;
        for (auto s : leaf.shape) {
            if (s > 0) slice_volume *= s;
        }
        if (slice_volume == 0) slice_volume = 1;
        
        size_t n_steps = leaf.count / slice_volume;
        if (n_steps == 0 && leaf.count > 0) n_steps = 1;
        
        TimeRangeP range{leaf.time_index, leaf.time_index + n_steps};
        
        // Stocker métadonnées
        leaf_metadata_cache[i] = {slice_volume, n_steps, range};
        
        // Indexer par time range
        time_range_index.insert({range, i});
    }
    
    time_index_valid = true;
}

// 3. RECHERCHE OPTIMISÉE O(log n) AU LIEU DE O(n)
// ============================================================================

const PanzerDB::Leaf* PanzerDB::findLeafByTime(const std::string& path, 
                                                 int64_t time_index) const {
    buildTimeIndex(); // Construit l'index si nécessaire (une seule fois)
    
    const auto& leaves = getLeaves();
    
    // Recherche par path
    auto path_it = leaf_lookup.find(path);
    if (path_it == leaf_lookup.end()) return nullptr;
    
    // Recherche optimisée par time range
    TimeRangeP query{static_cast<uint64_t>(time_index), 
                    static_cast<uint64_t>(time_index + 1)};
    
    auto range_it = time_range_index.lower_bound(query);
    
    // Parcourir les candidats (très peu, généralement 1-3)
    for (; range_it != time_range_index.end(); ++range_it) {
        size_t leaf_idx = range_it->second;
        
        // Vérifier que c'est le bon path
        if (leaves[leaf_idx].path != path) continue;
        
        // Vérifier que le time_index est dans le range
        if (range_it->first.contains(time_index)) {
            return &leaves[leaf_idx];
        }
        
        // Si on a dépassé, arrêter
        if (range_it->first.start > static_cast<uint64_t>(time_index)) break;
    }
    
    return nullptr;
}

// 4. LECTURE DIRECTE VIA HYPERSLAB (sans buffer intermédiaire)
// ============================================================================

template<typename T>
int PanzerDB::readSliceDirect(const Leaf& leaf, 
                               int64_t time_index,
                               T* out_buffer) const {
    
    // Utiliser les métadonnées en cache
    buildTimeIndex();
    size_t leaf_idx = &leaf - &getLeaves()[0]; // Index du leaf
    const auto& meta = leaf_metadata_cache[leaf_idx];
    
    uint64_t local_step = time_index - leaf.time_index;
    
    // Déterminer le dataset et type appropriés
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
    
    // OPTIMISATION CLÉE: Lecture directe du slice via hyperslab
    // Au lieu de lire tout le chunk puis copier
    
    hid_t file_space = H5Dget_space(dset_id);
    
    // Sélectionner exactement le slice voulu
    hsize_t offset[1] = {leaf.offset + local_step * meta.slice_volume};
    hsize_t count[1] = {meta.slice_volume};
    
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, NULL, count, NULL);
    
    // Créer memspace pour la sortie
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

// 5. LECTURE GROUPÉE POUR SIGNAUX CONTIGUS
// ============================================================================

template<typename T>
int PanzerDB::readMultipleSlices(const std::vector<const Leaf*>& leaves,
                                  T* output) const {
    if (leaves.empty()) return -1;
    
    // Trier par offset pour détecter contiguïté
    std::vector<const Leaf*> sorted_leaves = leaves;
    std::sort(sorted_leaves.begin(), sorted_leaves.end(),
              [](const Leaf* a, const Leaf* b) {
                  return a->offset < b->offset;
              });
    
    // Grouper les leaves contigus
    std::vector<std::vector<const Leaf*>> groups;
    std::vector<const Leaf*> current_group;
    
    for (size_t i = 0; i < sorted_leaves.size(); ++i) {
        if (current_group.empty()) {
            current_group.push_back(sorted_leaves[i]);
        } else {
            const Leaf* prev = current_group.back();
            const Leaf* curr = sorted_leaves[i];
            
            // Vérifier contiguïté
            if (prev->offset + prev->count == curr->offset) {
                current_group.push_back(curr);
            } else {
                groups.push_back(current_group);
                current_group.clear();
                current_group.push_back(curr);
            }
        }
    }
    if (!current_group.empty()) {
        groups.push_back(current_group);
    }
    
    // Lire chaque groupe en une seule opération HDF5
    size_t total_offset = 0;
    
    for (const auto& group : groups) {
        uint64_t group_offset = group[0]->offset;
        uint64_t group_count = 0;
        for (const auto* leaf : group) {
            group_count += leaf->count;
        }
        
        // Déterminer dataset
        DataType type = static_cast<DataType>(group[0]->flags >> 4);
        hid_t dset_id = -1;
        hid_t mem_type = -1;
        
        if (type == DataType::FLOAT64) {
            dset_id = data_dset_f64;
            mem_type = H5T_NATIVE_DOUBLE;
        } else if (type == DataType::INT32) {
            dset_id = data_dset_i32;
            mem_type = H5T_NATIVE_INT;
        } else {
            continue; // Skip unsupported types for grouped read
        }
        
        // Lecture groupée
        hid_t file_space = H5Dget_space(dset_id);
        hsize_t offset[1] = {group_offset};
        hsize_t count[1] = {group_count};
        
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, NULL, count, NULL);
        hid_t mem_space = H5Screate_simple(1, count, NULL);
        
        H5Dread(dset_id, mem_type, mem_space, file_space, H5P_DEFAULT, 
                output + total_offset);
        
        H5Sclose(mem_space);
        H5Sclose(file_space);
        
        total_offset += group_count;
    }
    
    return 0;
}

template int PanzerDB::readMultipleSlices<double>(const std::vector<const Leaf*>&, double*) const;
template int PanzerDB::readMultipleSlices<int32_t>(const std::vector<const Leaf*>&, int32_t*) const;
template int PanzerDB::readMultipleSlices<std::complex<double>>(const std::vector<const Leaf*>&, std::complex<double>*) const;

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
    
    //printf("[DEBUG beginArray BEFORE] array_stack.size() = %zu\n", array_stack.size());
    beginArray(level);
    //printf("[DEBUG beginArray AFTER] array_stack.size() = %zu\n", array_stack.size());

    path_prefix_dirty = true;
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

    if (!already_exists) {
        uint64_t aos_flags = level.is_dynamic ? 3 : 2; // 3 for dynamic AoS, 2 for static AoS
        append_index_row(new_node_path, parent_path, {(uint64_t)level.declared_size}, 0, 0, 0, 0, aos_flags);
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

    uint64_t start_row = 0;
    if (!cached_leaves.empty()) {
        if (n_rows == cached_leaves.size()) {
            leaves_cache_valid = true;
            return cached_leaves;
        }
        if (n_rows > cached_leaves.size()) {
            start_row = cached_leaves.size();
        } else {
            // File shrank or reset? Full reload.
            cached_leaves.clear();
            leaf_lookup.clear();
            parent_lookup.clear();
            cached_dynamic_aos_roots.clear();
            cached_paths_blocks.clear();
            cached_parent_paths_blocks.clear();
        }
    }

    if (n_rows == 0) {
        leaves_cache_valid = true; // Cache is now valid (but empty).
        return cached_leaves;
    }

    uint64_t read_count = n_rows - start_row;
    
    if (start_row == 0) {
        cached_dynamic_aos_roots.clear();
        cached_paths_blocks.clear();
        cached_parent_paths_blocks.clear();
    }

    // 2. Read index and paths (Incremental or Full)
    std::vector<uint64_t> idx(read_count * 14);
    
    if (start_row == 0) {
        H5Dread(index_dset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, idx.data());
    } else {
        hsize_t offset[2] = {start_row, 0};
        hsize_t count[2] = {read_count, 14};
        hid_t memspace = H5Screate_simple(2, count, NULL);
        hid_t filespace = H5Dget_space(index_dset);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
        H5Dread(index_dset, H5T_NATIVE_UINT64, memspace, filespace, H5P_DEFAULT, idx.data());
        H5Sclose(filespace);
        H5Sclose(memspace);
    }

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
            if (start_row == 0) {
                H5Dread(paths_dset, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, paths_data.data());
            } else {
                hsize_t offset[1] = {start_row};
                hsize_t count[1] = {read_count};
                hid_t memspace = H5Screate_simple(1, count, NULL);
                H5Sselect_hyperslab(paths_space, H5S_SELECT_SET, offset, NULL, count, NULL);
                H5Dread(paths_dset, str_type, memspace, paths_space, H5P_DEFAULT, paths_data.data());
                H5Sclose(memspace);
            }
        }
        H5Sclose(paths_space);

        if (parent_paths_dset >= 0) {
            // Same logic for parent_paths
            hid_t ppaths_space = H5Dget_space(parent_paths_dset);
            hsize_t ppaths_dims[1];
            H5Sget_simple_extent_dims(ppaths_space, ppaths_dims, nullptr);
            
            if (ppaths_dims[0] >= n_rows) {
                if (start_row == 0) {
                    H5Dread(parent_paths_dset, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, parent_paths_data.data());
                } else {
                    hsize_t offset[1] = {start_row};
                    hsize_t count[1] = {read_count};
                    hid_t memspace = H5Screate_simple(1, count, NULL);
                    H5Sselect_hyperslab(ppaths_space, H5S_SELECT_SET, offset, NULL, count, NULL);
                    H5Dread(parent_paths_dset, str_type, memspace, ppaths_space, H5P_DEFAULT, parent_paths_data.data());
                    H5Sclose(memspace);
                }
            }
            H5Sclose(ppaths_space);
        }
    }
    H5Tclose(str_type);

    // 3. Iterate over each row and build Leaf objects
    // Only reserve if we are starting from scratch or growing significantly
    if (cached_leaves.capacity() < n_rows) {
        cached_leaves.reserve(n_rows);
        // Note: reserving map/lookup is not standard but we can hint if needed, 
        // but standard containers manage this.
    }

    for (uint64_t i = 0; i < read_count; ++i) {
        const uint64_t* row = &idx[i * 14];

        // OPTIMISATION: Construction en place pour éviter la copie de 'leaf' et de son vecteur 'shape'
        cached_leaves.emplace_back();
        Leaf& leaf = cached_leaves.back();

        leaf.path         = std::string_view(paths_data.data() + i * PATH_MAX_LEN);
        leaf.parent_path  = std::string_view(parent_paths_data.data() + i * PATH_MAX_LEN);
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

    leaves_cache_valid = true;
    return cached_leaves;
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
        /*std::cerr << "[PanzerDB] WARNING: data_dset_str (" << data_dset_str << ") is invalid (Type: " << id_type << "). Attempting to reopen..." << std::endl;
        
        if (H5Lexists(file_id, "data_raw_str", H5P_DEFAULT) > 0) {
            data_dset_str = H5Dopen2(file_id, "data_raw_str", H5P_DEFAULT);
            if (data_dset_str < 0) {
                 throw std::runtime_error("Failed to recover data_dset_str");
            }
            std::cerr << "[PanzerDB] Successfully reopened data_dset_str: " << data_dset_str << std::endl;
        } else {
            throw std::runtime_error("Dataset for string type is not open and does not exist in file");
        }*/
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

    // HDF5 reads variable-length strings into a char* array
    char** rdata = (char**)calloc(hcount, sizeof(char*)); 
    hid_t str_type_vl = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type_vl, H5T_VARIABLE);
    H5Tset_cset(str_type_vl, H5T_CSET_UTF8);

    herr_t status = H5Dread(dset_id, str_type_vl, memspace, space, H5P_DEFAULT, rdata);
    if (status < 0) {
        //std::cerr << "[PanzerDB] Error: H5Dread failed for string tensor at offset " << leaf.offset << " count " << leaf.count << std::endl;
        //H5Eprint2(H5E_DEFAULT, stderr);
        throw ALBackendException("H5Dread failed for string tensor", LOG);
    }
    /*else{
        printf("[PanzerDB] Successfully read string tensor: offset = %llu, count = %llu\n", leaf.offset, leaf.count);
    } */ 

    for (size_t i = 0; i < hcount; ++i) {
        if (rdata[i] != nullptr) {
            out_buffer[i] = rdata[i]; 
        } else {
            out_buffer[i] = "";
        }
        free(rdata[i]); // Free individual strings allocated by HDF5
    }
    free(rdata); // Free array of pointers

    H5Tclose(str_type_vl);
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

    // OPTIMISATION 1: Construire parent_path de manière optimisée
    std::string parent_path;
    if (!array_stack.empty()) {
        // Pré-calculer taille nécessaire
        size_t estimated_size = path_prefix.length() + 25; // +25 pour "/12345"
        parent_path.reserve(estimated_size);
        
        parent_path = path_prefix;
        parent_path += '/';
        parent_path += std::to_string(array_stack.back().current_index);
    } else {
        parent_path = path_prefix;
    }

    // OPTIMISATION 2: Construire full_path avec réservation
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
    append_index_row(full_path, parent_path, shape, 0, time_idx, offset, count, flags);

    //autoFlushIfNeeded();
}

// 6. PATH PARSING OPTIMISÉ (pour substitution dans pz_readData_by_index)
// ============================================================================



PathComponents PanzerDB::parsePath(const std::string& path, 
                                   const std::string& aos_path) const {
    PathComponents result;
    
    if (path.length() <= aos_path.length()) {
        return result;
    }
    
    // Vérifier que path commence par aos_path + "/"
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
                         size_t count,      // count devrait être == shape[0]
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
    data_buffer_str.emplace_back(data, str_length);  // std::string(ptr, length)

    uint64_t flags = (static_cast<uint64_t>(DataType::STRING) << 4);
    append_index_row(full_path, parent_path, shape, 0, time_idx, offset, 1, flags);
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
    //if (timebase.empty()) throw std::runtime_error("timebase required");

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
    
    append_index_row(full_path, parent_path, base_shape, 0, 
                    base_time,                  // Start time
                    start_offset,               // Start offset
                    slice_size * n_slices,      // TOTAL Count
                    flags);

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
    //if (timebase.empty()) throw std::runtime_error("timebase required");
                                
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
    }

    uint64_t start_offset = disk_size_str + data_buffer_str.size();
    
    // ✅ DIFFERENCE: count_per_slice for strings
    size_t count_per_slice = base_shape.empty() ? 1 : base_shape[0];

    for (size_t i = 0; i < n_slices; ++i) {
        const char* const* slice_data = data + (i * count_per_slice);
        for (size_t j = 0; j < count_per_slice; ++j) {
            data_buffer_str.emplace_back(slice_data[j]);
        }
        
        uint64_t flags = (static_cast<uint64_t>(DataType::STRING) << 4);
        append_index_row(full_path, parent_path, base_shape, 0, base_time + i,
                        start_offset + (i * count_per_slice), count_per_slice, flags);
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

// 3. APPEND_INDEX_ROW: ÉVITER COPIES DE STRINGS
// ============================================================================

void PanzerDB::append_index_row(const std::string& full_path, 
                                const std::string& parent_path,
                                const std::vector<size_t>& shape, 
                                uint64_t type,
                                uint64_t time_idx, 
                                uint64_t offset, 
                                uint64_t count, 
                                uint64_t flags) {
    
    // OPTIMISATION 1: Pré-allouer exactement ce qu'il faut
    size_t required_paths = paths_buffer.size() + PATH_MAX_LEN;
    if (paths_buffer.capacity() < required_paths) {
        size_t new_cap = std::max(required_paths, paths_buffer.capacity() * BUFFER_GROWTH_FACTOR);
        paths_buffer.reserve(new_cap);
    }

    size_t required_parent = parent_paths_buffer.size() + PATH_MAX_LEN;
    if (parent_paths_buffer.capacity() < required_parent) {
        size_t new_cap = std::max(required_parent, parent_paths_buffer.capacity() * BUFFER_GROWTH_FACTOR);
        parent_paths_buffer.reserve(new_cap);
    }
    
    // OPTIMISATION 2: Construction directe sans buffer temporaire
    size_t current_path_size = paths_buffer.size();
    paths_buffer.resize(current_path_size + PATH_MAX_LEN, 0);
    
    // Copie optimisée (strncpy est rapide pour buffers fixes)
    strncpy(paths_buffer.data() + current_path_size, 
            full_path.c_str(), 
            PATH_MAX_LEN - 1);
    
    size_t current_parent_size = parent_paths_buffer.size();
    parent_paths_buffer.resize(current_parent_size + PATH_MAX_LEN, 0);
    
    strncpy(parent_paths_buffer.data() + current_parent_size, 
            parent_path.c_str(), 
            PATH_MAX_LEN - 1);
    
    // Construction de la row (inchangé)
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

    index_buffer.insert(index_buffer.end(), row, row + 14);
}

uint64_t PanzerDB::getTimeBaseLength(const std::string& timebase_name) const {
    const auto& leaves = getLeaves();
    uint64_t count = 0;
    for (const auto& leaf : leaves) {
        // Assume timebase is a 1D signal
        if (leaf.path == timebase_name) {
            count += leaf.count;
        }
    }
    return count;
}

uint64_t PanzerDB::getLastTimeIndex(const std::string& data_path) const {
    const auto& leaves = getLeaves();
    int64_t max_ti = -1;
    for (const auto& leaf : leaves) {
        if (leaf.path == data_path) {
            if (static_cast<int64_t>(leaf.time_index) > max_ti) {
                max_ti = leaf.time_index;
            }
        }
    }
    return (max_ti == -1) ? 0 : max_ti;
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


void PanzerDB::dumpLeavesCache() const {
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
        //printf("[WARN endArray] Stack is empty, ignoring endArray() call\n");
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

    path_prefix_dirty = true;
    invalidateDynamicAOSCache();
}

std::vector<size_t> PanzerDB::getAOSShape(const std::string& level_name) const {
  const auto &leaves = getLeaves();
  std::vector<size_t> shapes;

  // 1. Find root AoS meta-node
  const Leaf *aos_root_leaf = nullptr;

  //printf("level_name: '%s'\n", level_name.c_str());

  
  // OPTIMISATION: Utiliser leaf_lookup
  auto it_root = leaf_lookup.find(std::string_view(level_name));
  if (it_root != leaf_lookup.end() && !it_root->second.empty()) {
      aos_root_leaf = &leaves[it_root->second[0]];
  }
  // Pas de fallback linéaire nécessaire si l'index est cohérent

  if (!aos_root_leaf) {
    return shapes; // Return empty vector if AoS not found
  }

  // NEW LOGIC FOR DYNAMIC AoS
  if (aos_root_leaf->flags == 3) { // flags == 3 indicates dynamic AoS
    long long max_time_index = -1;
    
    // OPTIMIZATION: Use parent_lookup
    auto it = parent_lookup.find(aos_root_leaf->path);
    if (it != parent_lookup.end()) {
      for (size_t idx : it->second) {
         if (static_cast<long long>(leaves[idx].time_index) > max_time_index) {
             max_time_index = leaves[idx].time_index;
         }
      }
    }

    if (max_time_index >= 0) {
      shapes.push_back(static_cast<size_t>(max_time_index + 1));
    } else {
      // No data children found, but AoS exists. Size is 0.
      shapes.push_back(0);
    }
    //printf("Dynamic AoS detected. max_time_index: %lld, inferred size: %zu\n", max_time_index, shapes[0]);

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
  //printf("Static AoS detected. max_index: %lld, inferred size: %zu\n", max_index, shapes[0]);
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

int PanzerDB::pz_readData_by_index(
                         const char* full_data_path,
                         int64_t time_index,
                         uint64_t* ndim_out,
                         uint64_t shape_out[6],
                         double** data_out) {

    //printf("    [pz_readData_by_index] ENTER for full_data_path: %s, time_index: %lld\n", 
    //       full_data_path, (long long)time_index);

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

        //printf("    [pz_readData_by_index] Aggregated %zu matches, ndim_out=%llu\n", 
        //       matches.size(), *ndim_out);

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

                //printf("    [pz_readData_by_index] Trying generic path: '%s' with time_index: %lld\n", 
                //       generic_path.c_str(), (long long)time_index);

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

    //printf("    [pz_readData_by_index] Data not found\n");
    return -1;
}

int PanzerDB::pz_readStringData_by_index(
                          const char* full_data_path,
                          int64_t time_index,
                          uint64_t* ndim_out,
                          uint64_t shape_out[6],
                          char** data_out) {

    //printf("    [pz_readStringData_by_index] ENTER for full_data_path: %s, time_index: %lld\n", 
    //       full_data_path, (long long)time_index);

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
                //printf("    [pz_readStringData_by_index] Trying generic path: '%s' with time_index: %lld\n", generic_path.c_str(), (long long)time_index);

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
                    //printf("    [pz_readStringData_by_index] Trying substituted path: '%s' with time_index: %lld\n", substituted_path.c_str(), (long long)time_index);
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

int PanzerDB::pz_readComplexData_by_index(
                          const char* full_data_path,
                          int64_t time_index,
                          uint64_t* ndim_out,
                          uint64_t shape_out[6],
                          std::complex<double>** data_out) {

    //printf("    [pz_readComplexData_by_index] ENTER for full_data_path: %s, time_index: %lld\n", 
    //       full_data_path, (long long)time_index);

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
                //printf("    [pz_readComplexData_by_index] Trying generic path: '%s' with time_index: %lld\n", generic_path.c_str(), (long long)time_index);
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
                    //printf("    [pz_readComplexData_by_index] Trying substituted path: '%s' with time_index: %lld\n", substituted_path.c_str(), (long long)time_index);
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

    //printf("[PanzerDB::readInterpolatedData] ENTER for path: %s, time: %f\n", 
    //       full_data_path, time);

    DataInterpolation data_interpolation_component;
    std::map<std::string, int> times_indices;

    //printf("--> Time basis size: %zu\n", time_basis.size());
    int slice_index = data_interpolation_component.getSlicesTimesIndices(time, time_basis, times_indices, interp_mode);

    auto read_slice_with_fallback = [&](int64_t idx, void** out_ptr) -> int {
        //printf("    [read_slice_with_fallback] Trying to read index: %lld\n", (long long)idx);
        
        int res = -1;
        if (datatype == alconst::char_data) {
             res = pz_readStringData_by_index(full_data_path, idx, ndim_out, shape_out, (char**)out_ptr);
        } else if (datatype == alconst::complex_data) {
             res = pz_readComplexData_by_index(full_data_path, idx, ndim_out, shape_out, (std::complex<double>**)out_ptr);
        } else {
             res = pz_readData_by_index(full_data_path, idx, ndim_out, shape_out, (double**)out_ptr);
        }
        
        //printf("    [read_slice_with_fallback] Result: %d\n", res);

        if (res != 0 && datatype != alconst::char_data && datatype != alconst::complex_data) {
            //printf("    [read_slice_with_fallback] Trying fallback to time_index=0\n");
            
            void* fallback_ptr = nullptr;
            int res_fallback = pz_readData_by_index(full_data_path, 0, ndim_out, shape_out, (double**)&fallback_ptr);
            
            if (res_fallback == 0 && fallback_ptr != nullptr) {
                if (*ndim_out == 0 || !expect_time_dim) {
                    size_t element_size = sizeof(double);
                    if (datatype == (int)DataType::INT32) element_size = sizeof(int);
                    else if (datatype == (int)DataType::COMPLEX128) element_size = sizeof(std::complex<double>);
                    
                    size_t total_elements = 1;
                    for(size_t i=0; i<*ndim_out; ++i) total_elements *= shape_out[i];

                    *out_ptr = malloc(total_elements * element_size);
                    memcpy(*out_ptr, fallback_ptr, total_elements * element_size);
                    free(fallback_ptr);
                    return 0;
                }

                size_t num_slices = shape_out[*ndim_out - 1];
                if (idx >= 0 && (size_t)idx < num_slices) {
                    size_t total_elements = 1;
                    for(size_t i=0; i<*ndim_out; ++i) total_elements *= shape_out[i];
                    
                    size_t element_size = sizeof(double); 
                    if (datatype == (int)DataType::INT32) element_size = sizeof(int);

                    size_t single_slice_elements = total_elements / num_slices;
                    
                    *out_ptr = malloc(single_slice_elements * element_size);
                    memcpy(*out_ptr, (double*)fallback_ptr + (idx * single_slice_elements), 
                           single_slice_elements * element_size);
                    
                    free(fallback_ptr);
                    *ndim_out -= 1;
                    return 0;
                }
                free(fallback_ptr);
            }
        }
        return res;
    };

    void* data_inf = nullptr;
    int res_inf = read_slice_with_fallback(slice_index, &data_inf);
    
    if (res_inf != 0) {
        //printf("[PanzerDB::readInterpolatedData] EXIT with error\n");
        return -1;
    }

    // Interpolation if necessary
    if (times_indices["slice_inf"] != times_indices["slice_sup"]) {
        void* data_sup = nullptr;
        int res_sup = read_slice_with_fallback(times_indices["slice_sup"], &data_sup);

        if (res_sup != 0) {
            if (data_inf) free(data_inf);
            return -1;
        }

        std::map<std::string, double> slices_times;
        slices_times["slice_inf"] = time_basis[times_indices["slice_inf"]];
        slices_times["slice_sup"] = time_basis[times_indices["slice_sup"]];
        
        std::map<std::string, void*> y_slices;
        y_slices["slice_inf"] = data_inf;
        y_slices["slice_sup"] = data_sup;
        
        size_t shape_prod = 1;
        for (size_t i = 0; i < *ndim_out; ++i) shape_prod *= shape_out[i];
        
        data_interpolation_component.interpolate(datatype, shape_prod, y_slices, slices_times, time, data_out, interp_mode);

        if (data_inf && *data_out != data_inf) free(data_inf);
        if (data_sup && *data_out != data_sup) free(data_sup);

        return 0;
    }

    if (data_inf != nullptr) {
        *data_out = data_inf;
        return 0;
    }

    return -1;
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
        throw std::runtime_error("advanceTimebase appelé hors d'un AOS dynamique");
    }
    advanceTimeForAOS(dynamic_aos_path, n_steps);
}


// Fonction pour récupérer l'intégralité d'un signal dynamique hors-AOS
std::vector<double> PanzerDB::getWholeDynamicSignal(const std::string& dataset_name) {
    
    // 1. Accéder à l'index en mémoire
    const auto& leaves = getLeaves();
    
    // 2. Filtrer les feuilles correspondant au dataset
    std::vector<const PanzerDB::Leaf*> target_leaves;
    size_t total_elements = 0;

    auto it = leaf_lookup.find(dataset_name);
    if (it != leaf_lookup.end()) {
        for (size_t idx : it->second) {
            // Comparaison stricte car le dataset n'est pas dans un AOS
            // (déjà garanti par le lookup)
            target_leaves.push_back(&leaves[idx]);
            total_elements += leaves[idx].count;
        }
    }

    if (target_leaves.empty()) {
        //std::cerr << "[WARN] Dynamic dataset not found: " << dataset_name << std::endl;
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

void PanzerDB::writeMetadata(const std::string& path, const std::map<std::string, std::string>& metadata_map) {
    // The 'path' argument is the leaf name, e.g., "t_e"
    const std::string& name = path;

    // 1. Construct the schema path.
    // The internal `path_prefix` is the current AoS path, e.g., "profiles_1d/0/ion/1".
    // We strip the numeric indices to get the schema prefix.
    std::string schema_path_prefix = stripIndices(path_prefix);
    std::string schema_path = schema_path_prefix;
    if (!schema_path.empty()) schema_path += "/";
    schema_path += name;
    
    // 2. Check if we have already written metadata for this schema path.
    if ( written_metadata_schema_paths.count(schema_path)) {
        return; // Already written, do nothing.
    }

    // 3. Write metadata and record it.
    for (auto const& [key, value] : metadata_map) {
        // The metadata path itself is a schema path.
        std::string metadata_path = schema_path + "@" + key;
        const char* valueStr = value.c_str();
        // writeData for a static scalar string.
        this->writeData(metadata_path.c_str(), {}, &valueStr, 1);
    }

    written_metadata_schema_paths.insert(schema_path);

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
            // std::cout << "[PanzerDB] Read metadata for " << schema_path << ": " << key << " = " << value << std::endl;
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
        //printf("[DEBUG sync]   Stack is empty, nothing to synchronize\n");
        return;
    }
    
   // OPTIMISATION 2: Modification in-place avec détection de changement
    bool modified = false;
    for (size_t level = 0; level < sync_depth; ++level) {
        const std::string& target_name = aos_names[level];
        int target_index = indices[level];
        
        ArrayLevel& current_level = array_stack[level];
        
        // Vérification nom
        if (current_level.name != target_name) {
            continue;
        }
        
        // FIX: Ne pas synchroniser dynamic AoS en mode APPEND
        if (mode == OpenMode::APPEND && current_level.is_dynamic) {
            continue;
        }

        // Mise à jour si nécessaire
        if (current_level.current_index != static_cast<size_t>(target_index)) {
            current_level.current_index = target_index;
            modified = true;
        }
    }
    
    // OPTIMISATION 3: Reconstruction seulement si modifié
    if (modified) {
        // Reconstruire le stack
        
        // Reconstruction optimisée du path_prefix (une seule allocation)
        path_prefix = buildPathFromStack(array_stack);
    }
}
