#include "hdf5_writer_v2.h"
#include "al_backend.h"
#include "al_defs.h"

#include <algorithm>
#include <boost/filesystem.hpp>
#include <hdf5.h> // Explicitly include hdf5.h here
#include <iomanip>
#include <string.h>

#include "hdf5_utils.h"


#ifdef DEBUG_HDF5_WRITER_V2
#define DEBUG_PRINT(msg)                                                       \
  std::cerr << "[DEBUG " << __func__ << "] " << msg << std::endl
#else
#define DEBUG_PRINT(msg)                                                       \
  do {                                                                         \
  } while (0)
#endif
// #endif


using namespace boost::filesystem;

HDF5Writer_v2::HDF5Writer_v2(std::pair<int,int> backend_version_)
    : HDF5Writer(backend_version_)
       {
}

HDF5Writer_v2::~HDF5Writer_v2() {}

bool HDF5Writer_v2::compression_enabled = true;
size_t HDF5Writer_v2::read_chunk_cache_size = READ_CHUNK_CACHE_SIZE;
size_t HDF5Writer_v2::write_chunk_cache_size = WRITE_CHUNK_CACHE_SIZE;

void HDF5Writer_v2::read_homogeneous_time(int *homogenenous_time, hid_t gid) {

  if (gid == -1) {
    *homogenenous_time = -1;
    return;
  }
  const char *dataset_name = "ids_properties&homogeneous_time";
  PanzerDB panzer_db(gid, PanzerDB::OpenMode::READ, true);
   int status = -1;
   int temp = panzer_db.readScalar<int>(dataset_name, &status);
   if (status == 0) {
       *homogenenous_time = temp;
   } else {
       *homogenenous_time = -1;
   }
   panzer_db.close();
}

void HDF5Writer_v2::setWriteStrategy(OperationContext * ctx, int write_mode, hid_t loc_id) {
  DEBUG_PRINT("Setting write strategy to PanzerDB (mode: " << write_mode << ")");
  
  if (write_mode == GLOBAL_OP) {
    panzer_db_ptr = std::make_unique<PanzerDB>(loc_id, PanzerDB::OpenMode::WRITE, true, false);
    const char* imas_prefix = std::getenv("IMAS_PREFIX");
    if (imas_prefix) {
        std::string xml_path = std::string(imas_prefix) + "/include/IDSDef.xml";
        MetadataExtractor extractor(xml_path);
        metadata_map = extractor.extract_metadata(ctx->getDataobjectName());
    }
  } else if (write_mode == SLICE_OP) {
      DEBUG_PRINT("Write mode is SLICE_OP");
      panzer_db_ptr = std::make_unique<PanzerDB>(loc_id, PanzerDB::OpenMode::APPEND, true, false);
  }
}

void HDF5Writer_v2::beginWriteArraystructAction(ArraystructContext *ctx,
                                             int *size) {
  HDF5Utils hdf5_utils;
  OperationContext *opctx = ctx->getOperationContext();
  hid_t gid = -1;
  auto got_gid = IDS_group_id.find(opctx->getDataobjectName());
  if (got_gid != IDS_group_id.end()) {
    gid = got_gid->second;
  }

  if (*size == 0 ) {
      return;
  }


  DEBUG_PRINT("[TRACE_ID] Timebase path: " << ctx->getTimebasePath().c_str());

  if (!(gid >= 0))
    throw ALBackendException("HDF5Backend: unexpected value for gid in "
                             "HDF5Writer_v2::beginWriteArraystructAction()",
                             LOG);

  // SYNCHRONIZATION: Necessary for nested AOS
  // If we open an AOS B inside an AOS A, PanzerDB needs to
  // know the current index of A before creating B
  if (panzer_db_ptr) {
      std::vector<std::string> aos_names;
      std::vector<int> indices;
      
      // Go up to the PARENT to collect the indices
      Context* curr = ctx->getParent();
      while (curr && curr->getType() == CTX_ARRAYSTRUCT_TYPE) {
          ArraystructContext* arr = static_cast<ArraystructContext*>(curr);
          
          std::string full_path = arr->getPath();
          std::string aos_name;

          Context* parent = arr->getParent();
          if (parent && parent->getType() == CTX_ARRAYSTRUCT_TYPE) {
              ArraystructContext* parent_arr = static_cast<ArraystructContext*>(parent);
              std::string parent_path = parent_arr->getPath();
              if (full_path.size() > parent_path.size() && full_path.rfind(parent_path + "/", 0) == 0) {
                  aos_name = full_path.substr(parent_path.size() + 1);
              } else {
                  aos_name = full_path; // Fallback
              }
          } else {
              aos_name = full_path; // Root AoS
          }
          
          std::replace(aos_name.begin(), aos_name.end(), '/', '&');

          aos_names.insert(aos_names.begin(), aos_name);
          indices.insert(indices.begin(), arr->getIndex());
          
          curr = arr->getParent();
      }
      
      // Synchronize PanzerDB with the parent's state
      if (!aos_names.empty()) {
          panzer_db_ptr->synchronizeArrayStack(aos_names, indices);
      }
  }

  std::string full_path = ctx->getPath();
  std::string aos_name;
  Context* parent = ctx->getParent();
  if (parent && parent->getType() == CTX_ARRAYSTRUCT_TYPE) {
      ArraystructContext* parent_arr = static_cast<ArraystructContext*>(parent);
      std::string parent_path = parent_arr->getPath();
      if (full_path.size() > parent_path.size() && full_path.rfind(parent_path + "/", 0) == 0) {
          aos_name = full_path.substr(parent_path.size() + 1);
      } else {
          aos_name = full_path; // Fallback
      }
  } else {
      // Root AoS
      aos_name = full_path;
  }
  
  // Replace '/' with '&' in AoS name to handle nested names like "constraints/x_point" as a single level
  std::replace(aos_name.begin(), aos_name.end(), '/', '&');
  
  // Use the correct overload depending on the AOS type
  if (ctx->getTimed() && !ctx->getTimebasePath().empty()) {
      panzer_db_ptr->beginArray(aos_name, ctx->getTimebasePath());
  } else {
        panzer_db_ptr->beginArray(aos_name, static_cast<size_t>(*size));
  }


  initialized_aos.insert(ctx);
  
}

ArraystructContext *HDF5Writer_v2::getDynamicAOS(Context *ctx) {
  ArraystructContext *timed_ctx = dynamic_cast<ArraystructContext *>(ctx);
  while (timed_ctx != nullptr) {
    if (timed_ctx->getTimed())
      return timed_ctx;
    timed_ctx = timed_ctx->getParent();
  }
  return nullptr;
}

void HDF5Writer_v2::write_ND_Data(Context *ctx, const std::string &dataset_name,
                               const std::string &timebasename, int datatype,
                               int dim, int *size, void *data) {

  bool is_metadata = dataset_name.find('@') != std::string::npos;

  std::string dataset_name_copy = dataset_name;
  std::string timebasename_copy = timebasename;

  std::replace(dataset_name_copy.begin(), dataset_name_copy.end(), '/', '&');
  std::replace(timebasename_copy.begin(), timebasename_copy.end(), '/', '&');

  OperationContext *opctx = nullptr;
  if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
    opctx = (static_cast<ArraystructContext *>(ctx))->getOperationContext();
  } else {
    opctx = static_cast<OperationContext *>(ctx);
  }


  // FULL SYNCHRONIZATION: Rebuild PanzerDB's state from the AL context
  if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE && panzer_db_ptr) {
      std::vector<std::string> aos_names;
      std::vector<int> indices;
      
      // Go up the hierarchy to collect all AoS and their indices
      Context* curr = ctx;
      while (curr && curr->getType() == CTX_ARRAYSTRUCT_TYPE) {
          ArraystructContext* arr = static_cast<ArraystructContext*>(curr);
          
          std::string full_path = arr->getPath();
          std::string aos_name;

          Context* parent = arr->getParent();
          if (parent && parent->getType() == CTX_ARRAYSTRUCT_TYPE) {
              ArraystructContext* parent_arr = static_cast<ArraystructContext*>(parent);
              std::string parent_path = parent_arr->getPath();
              if (full_path.size() > parent_path.size() && full_path.rfind(parent_path + "/", 0) == 0) {
                  aos_name = full_path.substr(parent_path.size() + 1);
              } else {
                  aos_name = full_path; // Fallback
              }
          } else {
              // Root AoS
              aos_name = full_path;
          }
          
          std::replace(aos_name.begin(), aos_name.end(), '/', '&');

          aos_names.insert(aos_names.begin(), aos_name);
          indices.insert(indices.begin(), arr->getIndex());
          
          curr = arr->getParent();
      }
      
      // Synchronize PanzerDB with these indices
      if (!aos_names.empty()) {
          panzer_db_ptr->synchronizeArrayStack(aos_names, indices);
      }
  }

  DataEntryContext *dec = opctx->getDataEntryContext();
  hid_t gid = -1;
  auto got_gid = IDS_group_id.find(opctx->getDataobjectName());
  if (got_gid != IDS_group_id.end())
    gid = got_gid->second;
  else {
      throw ALBackendException("HDF5Backend: Dataobject group not opened in "
                               "HDF5Writer_v2::write_ND_Data()",
                               LOG);
    }

  if (!(gid >= 0))
    throw ALBackendException(
        "HDF5Backend: unexpected value for gid in HDF5Writer_v2::write_ND_Data()",
        LOG);

  if (dataset_name_copy == "ids_properties&homogeneous_time") {
    int *v = (int *)data;
    homogeneous_time = v[0];
  }

  // --- START: NEW METADATA HANDLING LOGIC ---

  // Only write metadata for actual data nodes, not for other metadata attributes.
  if (!is_metadata) {
    // Convert the instance path (e.g., "flux_loop/0/field") to a schema path ("flux_loop/field").
    // The schema path is used to find all associated metadata attributes.
    std::string schema_path = PanzerDB::stripIndices(dataset_name_copy);

    // Search the full metadata map for attributes related to this specific schema path.
    for (const auto& meta_entry : metadata_map) {
        const std::string& full_meta_path = meta_entry.first;  // e.g., "flux_loop/field@units"
        const std::string& meta_value = meta_entry.second;     // e.g., "T"

        size_t at_pos = full_meta_path.find('@');
        if (at_pos != std::string::npos) {
            // Extract the base path from the metadata key.
            std::string meta_base_path = full_meta_path.substr(0, at_pos);

            // If the metadata's base path matches the current data's schema path, write it.
            if (meta_base_path == schema_path) {
                panzer_db_ptr->writeMetadata(full_meta_path, meta_value);
            }
        }
    }
  }
// --- END: NEW METADATA HANDLING LOGIC ---
  
  DEBUG_PRINT("\n**************************************************************"
              "******************");
  DEBUG_PRINT("***** DATASET (DATA) : " << dataset_name_copy.c_str() << " *****");
  DEBUG_PRINT("****************************************************************"
              "****************");

  // 1. Calculate the total number of elements
  size_t n_slices;
  int count = 1;
  std::vector<size_t> shape(dim);
  if (dim >= 1) {
    n_slices = size[dim - 1];
    for(int i=0; i< dim; ++i) {
      shape[i] = size[i];
      count *= size[i];
    }
  } 
  else if (dim == 0) { // Static scalar or scalar in dynamic AOS
    n_slices = 1;
    count = 1;
    shape = {};
  } 

  std::string aos_timebase;
  bool is_in_dynamic_aos = panzer_db_ptr ? panzer_db_ptr->isInsideDynamicAOS(&aos_timebase) : false;

  // RAII containers to manage memory for string data conversion
  std::vector<char *> string_pointers;
  std::vector<std::vector<char>> string_data_copies;

  // Validation for APPEND mode
  if (panzer_db_ptr && panzer_db_ptr->getOpenMode() == PanzerDB::OpenMode::APPEND) {
      if (aos_timebase.empty() && timebasename_copy.empty()) {
          throw ALBackendException("In APPEND mode, data must have a timebase, either from a dynamic AoS parent or directly.", LOG);
      }
  }

    // Writing metadata for this node
    if (is_metadata) {
        DEBUG_PRINT("Writing metadata: " << dataset_name_copy.c_str() << " = " 
                    << (datatype == alconst::char_data ? std::string(static_cast<const char*>(data), size[0]) : "<non-string data>"));
        panzer_db_ptr->writeMetadata(dataset_name_copy, std::string(static_cast<const char*>(data), size[0]));
        return;
    }

  // ========== NUMERICAL TYPES (DOUBLE) ==========
  if (datatype == alconst::double_data) {
    
    // CASE 1: Data with explicit timebase
    if (!timebasename_copy.empty()) {
        size_t n_slices_dyn = 1;
        std::vector<size_t> base_shape;

        // AUTOMATIC mode DETECTION:
        // If we are in a dynamic AOS, then the iteration is EXTERNAL
        // → dim ONLY contains the spatial dimensions
        // → n_slices = 1 (we write the current slice)
        
        if (is_in_dynamic_aos) {
            // Iterative mode: dim = spatial dimensions only
            base_shape = shape; // All dimensions are spatial
            n_slices_dyn = 1;   // A single slice
            
            DEBUG_PRINT("Writing in dynamic AoS context: 1 slice of shape [" 
                       << (shape.empty() ? "scalar" : std::to_string(shape[0])) << "]");
        } 
        else {
            // Bulk mode: last dimension = time
            if (dim >= 1) {
                n_slices_dyn = shape.back(); 
                base_shape.assign(shape.begin(), shape.end() - 1);
                
                DEBUG_PRINT("Writing in bulk mode: " << n_slices_dyn 
                           << " slices of shape [" << (base_shape.empty() ? "scalar" : std::to_string(base_shape[0])) << "]");
            }
        }
        
        panzer_db_ptr->writeDataSlices(dataset_name_copy, base_shape, 
                                       static_cast<const double*>(data), 
                                       n_slices_dyn, timebasename_copy);
    } 
    // CASE 2: Data in a dynamic AOS WITHOUT an explicit timebase
    else if (is_in_dynamic_aos) {
        DEBUG_PRINT("Data is inside dynamic AoS, writing as a single slice.");
        
        // shape contains ALL spatial dimensions
        panzer_db_ptr->writeDataSlices(dataset_name_copy, shape, 
                                       static_cast<const double*>(data), 
                                       1, aos_timebase);
    }
    // CASE 3: Pure static data
    else {
        size_t total_count = 1;
        for (auto s : shape) total_count *= s;
        panzer_db_ptr->writeData(dataset_name_copy, shape, 
                                static_cast<const double*>(data), 
                                total_count);
    }
 }
  
  // ========== NUMERICAL TYPES (INTEGER) ==========
  else if (datatype == alconst::integer_data) {
    if (!timebasename_copy.empty()) {
      size_t n_slices_dyn = 1;
      std::vector<size_t> base_shape;
      
      if (is_in_dynamic_aos) {
            // Iterative mode: dim = spatial dimensions only
            base_shape = shape; // All dimensions are spatial
            n_slices_dyn = 1;   // A single slice
            
            DEBUG_PRINT("Writing in dynamic AoS context: 1 slice of shape [" 
                       << (shape.empty() ? "scalar" : std::to_string(shape[0])) << "]");
        } 
        else {
            // Bulk mode: last dimension = time
            if (dim >= 1) {
                n_slices_dyn = shape.back(); 
                base_shape.assign(shape.begin(), shape.end() - 1);
                
                DEBUG_PRINT("Writing in bulk mode: " << n_slices_dyn 
                           << " slices of shape [" << (base_shape.empty() ? "scalar" : std::to_string(base_shape[0])) << "]");
            }
        }
      
      panzer_db_ptr->writeDataSlices(dataset_name_copy, base_shape, 
                                     static_cast<const int32_t*>(data), 
                                     n_slices_dyn, timebasename_copy);
    } 
    else if (is_in_dynamic_aos) {
      DEBUG_PRINT("Data is inside dynamic AoS, writing as a single slice.");
      
      // 1 single slice with full shape
      panzer_db_ptr->writeDataSlices(dataset_name_copy, shape, 
                                     static_cast<const int32_t*>(data), 
                                     1, aos_timebase);
    }
    else {
      size_t total_count = 1;
      for (auto s : shape) total_count *= s;
      panzer_db_ptr->writeData(dataset_name_copy, shape, 
                              static_cast<const int32_t*>(data), 
                              total_count, "");
    }
  } 
  
  // ========== STRINGS ==========
  else if (datatype == alconst::char_data) {
    
    // CASE 1: String scalar (dim=1, buffer size)
    if (dim == 1) {
        std::string temp_str(static_cast<const char*>(data), size[0]);
        const char* c_str_data = temp_str.c_str();
        
        if (!timebasename_copy.empty()) {
            // Dynamic scalar signal with explicit timebase
            panzer_db_ptr->writeDataSlices(dataset_name_copy, {}, 
                                          &c_str_data, 1, timebasename_copy);
        } 
        else if (is_in_dynamic_aos) {
            DEBUG_PRINT("String scalar in dynamic AoS, writing as a single slice.");
            
            // 1 single slice (scalar)
            panzer_db_ptr->writeDataSlices(dataset_name_copy, {}, 
                                          &c_str_data, 1, aos_timebase);
        } 
        else {
            // Static
            panzer_db_ptr->writeData(dataset_name_copy, {}, 
                                    &c_str_data, 1, "");
        }
    } 
    // CASE 2: List of strings (dim=2)
    else if (dim == 2) {
        // Flat buffer conversion → char**
        string_pointers.reserve(size[0]);
        string_data_copies.reserve(size[0]);
        char *input_data_ptr = static_cast<char *>(data);
        
        for (int i = 0; i < size[0]; ++i) {
            string_data_copies.emplace_back(size[1] + 1, '\0');
            strncpy(string_data_copies.back().data(), 
                   input_data_ptr + i * size[1], size[1]);
            string_pointers.push_back(string_data_copies.back().data());
        }
        const char** converted_data = (const char**)string_pointers.data();

        if (!timebasename_copy.empty()) {
            // Temporal signal with explicit timebase
            // If dim=2 with timebase, it's probably [count, time]
            // We need to determine if size[1] is the time dimension or the max length
            // HEURISTIC: If size[1] looks like a buffer length (>20),
            // it's a list of strings over 1 time step
            if (size[1] > 20) {
                // It's [count_strings, max_len] → 1 slice of count_strings elements
                panzer_db_ptr->writeDataSlices(dataset_name_copy, {(size_t)size[0]}, 
                                              converted_data, 1, timebasename_copy);
            } else {
                // It's [spatial, time] → size[1] slices
                // BUT for strings this is rare, we keep the logic simple
                panzer_db_ptr->writeDataSlices(dataset_name_copy, {(size_t)size[0]}, 
                                              converted_data, 1, timebasename_copy);
            }
        } 
        else if (is_in_dynamic_aos) {
            DEBUG_PRINT("String list in dynamic AoS, writing as a single slice.");
            
            // 1 single slice of size[0] strings
            panzer_db_ptr->writeDataSlices(dataset_name_copy, {(size_t)size[0]}, 
                                          converted_data, 1, aos_timebase);
        } 
        else {
            // Static
            panzer_db_ptr->writeData(dataset_name_copy, {(size_t)size[0]}, 
                                    converted_data, size[0], "");
        }
    } 
    else {
        throw ALBackendException("Writing string arrays with dimension > 2 is not supported.", LOG);
    }
  } 
  
  // ========== COMPLEX ==========
  else if (datatype == alconst::complex_data) {
      if (!timebasename_copy.empty()) {
          size_t n_slices_dyn = 1;
          std::vector<size_t> base_shape;
          
          if (is_in_dynamic_aos) {
            // Iterative mode: dim = spatial dimensions only
            base_shape = shape; // All dimensions are spatial
            n_slices_dyn = 1;   // A single slice
            
            DEBUG_PRINT("Writing in dynamic AoS context: 1 slice of shape [" 
                       << (shape.empty() ? "scalar" : std::to_string(shape[0])) << "]");
        } 
        else {
            // Bulk mode: last dimension = time
            if (dim >= 1) {
                n_slices_dyn = shape.back(); 
                base_shape.assign(shape.begin(), shape.end() - 1);
                
                DEBUG_PRINT("Writing in bulk mode: " << n_slices_dyn 
                           << " slices of shape [" << (base_shape.empty() ? "scalar" : std::to_string(base_shape[0])) << "]");
            }
        }
          
          panzer_db_ptr->writeDataSlices(dataset_name_copy, base_shape, 
                                         static_cast<const std::complex<double>*>(data), 
                                         n_slices_dyn, timebasename_copy);
      } 
      else if (is_in_dynamic_aos) {
          DEBUG_PRINT("Complex data in dynamic AoS, writing as a single slice.");
          
          // 1 single slice with full shape
          panzer_db_ptr->writeDataSlices(dataset_name_copy, shape, 
                                         static_cast<const std::complex<double>*>(data), 
                                         1, aos_timebase);
      }
      else {
          size_t total_count = 1;
          for (auto s : shape) total_count *= s;
          panzer_db_ptr->writeData(dataset_name_copy, shape, 
                                  static_cast<const std::complex<double>*>(data), 
                                  total_count, "");
      }
  } 
  else {
    throw ALBackendException("Data type not supported by HDF5Writer_v2.", LOG);
  }
  
}


void HDF5Writer_v2::endAction(Context *ctx) {
  
  if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
    // IMPORTANT: Do NOT synchronize here!
    // Synchronization has already occurred in the previous write_ND_Data calls
    // and in beginWriteArraystructAction for the children.
    // 
    // For empty AOS, we rely on the fact that:
    // 1. beginArray() has already created the meta-node with the correct size
    // 2. Empty indices do not need entries in the index

    ArraystructContext* arrCtx = static_cast<ArraystructContext*>(ctx);

    // Only close an AoS we actually opened (tracked in initialized_aos).
    if (initialized_aos.count(arrCtx) > 0) {
        if (panzer_db_ptr) panzer_db_ptr->endArray();
        initialized_aos.erase(arrCtx);
    }
  } else if (ctx->getType() == CTX_OPERATION_TYPE) {

    if (panzer_db_ptr) panzer_db_ptr->flush();
    if (panzer_db_ptr) panzer_db_ptr->close();
  }
}