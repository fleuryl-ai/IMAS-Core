#include "hdf5_writer_v2.h"
#include "al_backend.h"
#include "al_defs.h"

#include <algorithm>
#include <boost/filesystem.hpp>
#include <hdf5.h> // Explicitly include hdf5.h here
#include <iomanip>
#include <string.h>

#include "hdf5_utils.h"
#include "aos_path_helpers.h"


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

  // Direct H5Dread of the scalar. This used to build a fresh READ PanzerDB on
  // the group — a full /index load + leaf cache + path index — just to read
  // one int32 on every SLICE_OP (APPEND) open. The datasets live in the IDS
  // group itself; when "index" is not a direct child, mirror PanzerDB::init's
  // one-level descent (child group holding /index).
  hid_t root = gid;
  if (H5Lexists(gid, "index", H5P_DEFAULT) <= 0) {
      H5G_info_t group_info;
      if (H5Gget_info(gid, &group_info) >= 0) {
          H5E_auto2_t old_func;
          void *old_client_data;
          H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
          H5Eset_auto2(H5E_DEFAULT, NULL, NULL);   // silence expected open failures
          for (hsize_t i = 0; i < group_info.nlinks; ++i) {
              char name[256];
              if (H5Lget_name_by_idx(gid, ".", H5_INDEX_NAME, H5_ITER_INC, i, name,
                                     sizeof(name), H5P_DEFAULT) < 0)
                  continue;
              hid_t child = H5Gopen2(gid, name, H5P_DEFAULT);
              if (child >= 0) {
                  if (H5Lexists(child, "index", H5P_DEFAULT) > 0) {
                      root = child;
                      break;
                  }
                  H5Gclose(child);
              }
          }
          H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
      }
  }

  hid_t dataset_id = H5Dopen2(root, dataset_name, H5P_DEFAULT);
  herr_t status = -1;
  if (dataset_id >= 0) {
      status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, homogenenous_time);
      H5Dclose(dataset_id);
  }
  if (status < 0)
      *homogenenous_time = -1;

  if (root != gid)
      H5Gclose(root);   // the child group handle we opened above
}

void HDF5Writer_v2::setWriteStrategy(OperationContext * ctx, int write_mode, hid_t loc_id) {
  DEBUG_PRINT("Setting write strategy to PanzerDB (mode: " << write_mode << ")");
  
  if (write_mode == GLOBAL_OP) {
    panzer_db_ptr = std::make_unique<PanzerDB>(loc_id, PanzerDB::OpenMode::WRITE, true, false);
    metadata_by_path.clear();   // fresh per put(); only (re)populate from this IDS's XML
    const char* imas_prefix = std::getenv("IMAS_PREFIX");
    if (imas_prefix) {
        std::string xml_path = std::string(imas_prefix) + "/include/IDSDef.xml";
        MetadataExtractor extractor(xml_path);
        const std::map<std::string, std::string> flat =
            extractor.extract_metadata(ctx->getDataobjectName());
        // Group the flat "path@attr" -> value entries by the node's schema path, so
        // write_ND_Data() can retrieve a node's @keys with a single O(log) find()
        // (previously each node scanned every entry: O(N*M) string scans + allocs).
        for (const auto& kv : flat) {
            size_t at = kv.first.find('@');
            if (at == std::string::npos) {
                continue;
            }
            std::string schema_path = kv.first.substr(0, at);   // e.g. "flux_loop/field"
            std::string attr_name   = kv.first.substr(at + 1);  // e.g. "units"
            metadata_by_path[schema_path][attr_name] = kv.second;
        }
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
      if (ctx->getParent())
          collectAosChain(ctx->getParent(), aos_names, indices);
      // Synchronize PanzerDB with the parent's state
      if (!aos_names.empty()) {
          panzer_db_ptr->synchronizeArrayStack(aos_names, indices);
      }
  }

  const std::string aos_name = localAosName(ctx);

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
      collectAosChain(ctx, aos_names, indices);
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
    // Resolve this node's schema path in the AL/XML namespace ("/"-separated): strip any
    // array indices from the ORIGINAL name (dataset_name), e.g. "electrons/0/density" ->
    // "electrons/density". metadata_by_path is keyed by these "/" schema paths (built from
    // the IDSDef.xml "path" attributes in setWriteStrategy), so the key must stay "/".
    std::string schema_path = PanzerDB::stripIndices(dataset_name);

    auto it = metadata_by_path.find(schema_path);
    if (it != metadata_by_path.end()) {
        // PanzerDB names datasets with "&" as the separator (see e.g. "ids_properties&
        // homogeneous_time"), so convert "/" -> "&" before handing the key to the engine.
        std::string pzd_schema = schema_path;
        std::replace(pzd_schema.begin(), pzd_schema.end(), '/', '&');
        for (const auto& attr : it->second) {
            panzer_db_ptr->writeMetadata(pzd_schema + "@" + attr.first, attr.second);
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

  // 2. Resolve the time-role of the written dimensions ONCE for every type:
  //   - inside a dynamic AoS: the time iteration is EXTERNAL (the AL gives
  //     spatial dims only) -> one slice, every dim is spatial;
  //   - explicit timebase outside a dynamic AoS: the LAST written dim is the
  //     time axis (bulk mode) -> n_slices = shape.back();
  //   - no timebase: pure static tensor.
  const bool bulk_temporal = !timebasename_copy.empty() && !is_in_dynamic_aos;
  std::vector<size_t> base_shape;
  size_t n_slices_dyn = 1;
  if (bulk_temporal) {
      if (dim >= 1) {
          n_slices_dyn = shape.back();
          base_shape.assign(shape.begin(), shape.end() - 1);
      }
  }
  const std::string eff_timebase = timebasename_copy.empty() ? aos_timebase : timebasename_copy;

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

  // 3. Engine dispatch shared by every datatype: the writeData/writeDataSlices
  //    overloads resolve the element type from the pointer, so the former
  //    per-type copies (double/integer/complex, identical apart from the cast)
  //    collapse into one resolution driven by the shape analysis above.
  if (datatype == alconst::double_data || datatype == alconst::integer_data ||
      datatype == alconst::complex_data) {
    if (bulk_temporal) {
      DEBUG_PRINT("Writing in bulk mode: " << n_slices_dyn
                  << " slices of shape [" << (base_shape.empty() ? "scalar" : std::to_string(base_shape[0])) << "]");
      if (datatype == alconst::double_data)
        panzer_db_ptr->writeDataSlices(dataset_name_copy, base_shape,
                                       static_cast<const double*>(data), n_slices_dyn, timebasename_copy);
      else if (datatype == alconst::integer_data)
        panzer_db_ptr->writeDataSlices(dataset_name_copy, base_shape,
                                       static_cast<const int32_t*>(data), n_slices_dyn, timebasename_copy);
      else
        panzer_db_ptr->writeDataSlices(dataset_name_copy, base_shape,
                                       static_cast<const std::complex<double>*>(data), n_slices_dyn, timebasename_copy);
    }
    // CASE 2: Data in a dynamic AoS (explicit timebase or the enclosing one):
    // every written dimension is spatial, the time step comes from the iteration
    else if (is_in_dynamic_aos) {
        DEBUG_PRINT("Writing in dynamic AoS context: 1 slice of shape ["
                    << (shape.empty() ? "scalar" : std::to_string(shape[0])) << "]");
      if (datatype == alconst::double_data)
        panzer_db_ptr->writeDataSlices(dataset_name_copy, shape,
                                       static_cast<const double*>(data), 1, eff_timebase);
      else if (datatype == alconst::integer_data)
        panzer_db_ptr->writeDataSlices(dataset_name_copy, shape,
                                       static_cast<const int32_t*>(data), 1, eff_timebase);
      else
        panzer_db_ptr->writeDataSlices(dataset_name_copy, shape,
                                       static_cast<const std::complex<double>*>(data), 1, eff_timebase);
    }
    // CASE 3: Pure static data
    else {
      size_t total_count = (size_t)count;
      if (datatype == alconst::double_data)
        panzer_db_ptr->writeData(dataset_name_copy, shape,
                                 static_cast<const double*>(data), total_count);
      else if (datatype == alconst::integer_data)
        panzer_db_ptr->writeData(dataset_name_copy, shape,
                                 static_cast<const int32_t*>(data), total_count);
      else
        panzer_db_ptr->writeData(dataset_name_copy, shape,
                                 static_cast<const std::complex<double>*>(data), total_count);
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
            // Temporal string signal with explicit timebase: the AL gives ONE
            // slice of size[0] strings per call (dim = 2 is the [count,max_len]
            // buffer layout); the time axis comes from the timebase, not from
            // the buffer. The old code branched on "max_len > 20" but both
            // branches performed the same call — a dead heuristic.
            panzer_db_ptr->writeDataSlices(dataset_name_copy, {(size_t)size[0]}, 
                                          converted_data, 1, timebasename_copy);
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