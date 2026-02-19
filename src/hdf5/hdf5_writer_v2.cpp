#include "hdf5_writer_v2.h"
#include "al_backend.h"
#include "al_defs.h"

#include <algorithm>
#include <boost/filesystem.hpp>
#include <hdf5.h> // Explicitly include hdf5.h here
#include <iomanip>
#include <string.h>

#include "hdf5_utils.h"

/* ----------------------------------------------------------------------
 *  Debug macro  define DEBUG_HDF5_WRITER in the build system to enable
 *  the trace.  When undefined the macro expands to a no-op (zero cost).
 * ---------------------------------------------------------------------- */

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

HDF5Writer_v2::HDF5Writer_v2(std::string backend_version_)
    : HDF5Writer(backend_version_), backend_version(backend_version_)
       {
  // H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

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

void HDF5Writer_v2::setWriteStrategy(int write_mode, hid_t loc_id) {
  DEBUG_PRINT("Setting write strategy to PanzerDB (mode: " << write_mode << ")");
  
  // ✅ Toujours recréer (l'ancienne instance sera automatiquement détruite)
  if (write_mode == GLOBAL_OP) {
    panzer_db_ptr = std::make_unique<PanzerDB>(loc_id, PanzerDB::OpenMode::WRITE, true, false);
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
      //printf("[DEBUG beginArray] No data for array of structure '%s', skipping beginArray()\n", 
      //       ctx->getPath().c_str());
      return;
  }


  DEBUG_PRINT("[TRACE_ID] Timebase path: " << ctx->getTimebasePath().c_str());

  if (!(gid >= 0))
    throw ALBackendException("HDF5Backend: unexpected value for gid in "
                             "HDF5Writer_v2::beginWriteArraystructAction()",
                             LOG);

  // ✅ SYNCHRONISATION : Nécessaire pour les AOS imbriqués
  // Si on ouvre un AOS B à l'intérieur d'un AOS A, il faut que PanzerDB
  // connaisse l'index courant de A avant de créer B
  if (panzer_db_ptr) {
      std::vector<std::string> aos_names;
      std::vector<int> indices;
      
      // Remonter jusqu'au PARENT pour collecter les indices
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
      
      // Synchroniser PanzerDB avec l'état du parent
      if (!aos_names.empty()) {
          //printf("[DEBUG beginArray] Synchronizing parent hierarchy before creating '%s'\n", 
          //       ctx->getPath().c_str());
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
  
  // Utiliser la bonne surcharge selon le type d'AOS
  if (ctx->getTimed() && !ctx->getTimebasePath().empty()) {
      //printf("[DEBUG beginArray] Creating DYNAMIC AoS: '%s' with timebase '%s'\n",
      //       aos_name.c_str(), ctx->getTimebasePath().c_str());
      panzer_db_ptr->beginArray(aos_name, ctx->getTimebasePath());
  } else {
      //printf("[DEBUG beginArray] Creating STATIC AoS: '%s' with size %d\n",
      //       aos_name.c_str(), *size);
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


  // ✅ SYNCHRONISATION COMPLÈTE : Reconstruire l'état de PanzerDB depuis le contexte AL
  if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE && panzer_db_ptr) {
      std::vector<std::string> aos_names;
      std::vector<int> indices;
      
      // Remonter la hiérarchie pour collecter tous les AoS et leurs indices
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
      
      // Synchroniser PanzerDB avec ces indices
      if (!aos_names.empty()) {
          /*printf("[DEBUG write_ND_Data] Synchronizing before writing '%s'\n", dataset_name_copy.c_str());
          for (size_t i = 0; i < aos_names.size(); ++i) {
              printf("[DEBUG write_ND_Data]   AoS[%zu]: '%s' at index %d\n", 
                     i, aos_names[i].c_str(), indices[i]);
          }*/
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
  
  DEBUG_PRINT("\n**************************************************************"
              "******************");
  DEBUG_PRINT("***** DATASET (DATA) : " << dataset_name_copy.c_str() << " *****");
  DEBUG_PRINT("****************************************************************"
              "****************");

  // 1. Calculer le nombre total d'éléments
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
  else if (dim == 0) { // Scalaire statique ou scalaire dans AOS dynamique
    n_slices = 1;
    count = 1;
    shape = {};
  } 

  std::string aos_timebase;
  bool is_in_dynamic_aos = panzer_db_ptr ? panzer_db_ptr->isInsideDynamicAOS(&aos_timebase) : false;

  // RAII containers to manage memory for string data conversion
  std::vector<char *> string_pointers;
  std::vector<std::vector<char>> string_data_copies;

  // Validation pour le mode APPEND
  if (panzer_db_ptr && panzer_db_ptr->getOpenMode() == PanzerDB::OpenMode::APPEND) {
      if (aos_timebase.empty() && timebasename_copy.empty()) {
          throw ALBackendException("In APPEND mode, data must have a timebase, either from a dynamic AoS parent or directly.", LOG);
      }
  }

  // ========== TYPES NUMÉRIQUES (DOUBLE) ==========
  if (datatype == alconst::double_data) {
    
    // CAS 1 : Donnée avec timebase explicite
    if (!timebasename_copy.empty()) {
        size_t n_slices_dyn = 1;
        std::vector<size_t> base_shape;

        // ✅ DÉTECTION AUTOMATIQUE du mode :
        // Si on est dans un AOS dynamique, alors l'itération est EXTERNE
        // → dim contient UNIQUEMENT les dimensions spatiales
        // → n_slices = 1 (on écrit la slice courante)
        
        if (is_in_dynamic_aos) {
            // Mode itératif : dim = dimensions spatiales uniquement
            base_shape = shape; // Toutes les dimensions sont spatiales
            n_slices_dyn = 1;   // Une seule slice
            
            DEBUG_PRINT("Writing in dynamic AoS context: 1 slice of shape [" 
                       << (shape.empty() ? "scalar" : std::to_string(shape[0])) << "]");
        } 
        else {
            // Mode bulk : dernière dimension = temps
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
    // CAS 2 : Donnée dans un AOS dynamique SANS timebase explicite
    else if (is_in_dynamic_aos) {
        DEBUG_PRINT("Data is inside dynamic AoS, writing as a single slice.");
        
        // ✅ shape contient TOUTES les dimensions spatiales
        panzer_db_ptr->writeDataSlices(dataset_name_copy, shape, 
                                       static_cast<const double*>(data), 
                                       1, aos_timebase);
    }
    // CAS 3 : Donnée statique pure
    else {
        size_t total_count = 1;
        for (auto s : shape) total_count *= s;
        panzer_db_ptr->writeData(dataset_name_copy, shape, 
                                static_cast<const double*>(data), 
                                total_count);
    }
 }
  
  // ========== TYPES NUMÉRIQUES (INTEGER) ==========
  else if (datatype == alconst::integer_data) {
    if (!timebasename_copy.empty()) {
      size_t n_slices_dyn = 1;
      std::vector<size_t> base_shape;
      
      if (is_in_dynamic_aos) {
            // Mode itératif : dim = dimensions spatiales uniquement
            base_shape = shape; // Toutes les dimensions sont spatiales
            n_slices_dyn = 1;   // Une seule slice
            
            DEBUG_PRINT("Writing in dynamic AoS context: 1 slice of shape [" 
                       << (shape.empty() ? "scalar" : std::to_string(shape[0])) << "]");
        } 
        else {
            // Mode bulk : dernière dimension = temps
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
      
      // ✅ FIX : 1 seule slice avec shape complète
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
  
  // ========== CHAÎNES DE CARACTÈRES ==========
  else if (datatype == alconst::char_data) {
    
    // CAS 1 : Scalaire string (dim=1, taille du buffer)
    if (dim == 1) {
        std::string temp_str(static_cast<const char*>(data), size[0]);
        const char* c_str_data = temp_str.c_str();
        
        if (!timebasename_copy.empty()) {
            // Signal scalaire dynamique avec timebase explicite
            panzer_db_ptr->writeDataSlices(dataset_name_copy, {}, 
                                          &c_str_data, 1, timebasename_copy);
        } 
        else if (is_in_dynamic_aos) {
            DEBUG_PRINT("String scalar in dynamic AoS, writing as a single slice.");
            
            // ✅ FIX : 1 seule slice (scalaire)
            panzer_db_ptr->writeDataSlices(dataset_name_copy, {}, 
                                          &c_str_data, 1, aos_timebase);
        } 
        else {
            // Statique
            panzer_db_ptr->writeData(dataset_name_copy, {}, 
                                    &c_str_data, 1, "");
        }
    } 
    // CAS 2 : Liste de strings (dim=2)
    else if (dim == 2) {
        // Conversion buffer plat → char**
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
            // Signal temporel avec timebase explicite
            // ❌ ANCIEN CODE :
            // panzer_db_ptr->writeDataSlices(dataset_name_copy, {(size_t)size[0]}, 
            //                               converted_data, 1, timebasename_copy);
            
            // ✅ NOUVEAU : Si dim=2 avec timebase, c'est probablement [count, temps]
            // Il faut déterminer si size[1] est la dimension temporelle ou la longueur max
            // HEURISTIQUE : Si size[1] ressemble à une longueur de buffer (>20), 
            // c'est une liste de strings sur 1 pas de temps
            if (size[1] > 20) {
                // C'est [count_strings, max_len] → 1 slice de count_strings éléments
                panzer_db_ptr->writeDataSlices(dataset_name_copy, {(size_t)size[0]}, 
                                              converted_data, 1, timebasename_copy);
            } else {
                // C'est [spatial, temps] → size[1] slices
                // MAIS pour les strings c'est rare, on garde la logique simple
                panzer_db_ptr->writeDataSlices(dataset_name_copy, {(size_t)size[0]}, 
                                              converted_data, 1, timebasename_copy);
            }
        } 
        else if (is_in_dynamic_aos) {
            DEBUG_PRINT("String list in dynamic AoS, writing as a single slice.");
            
            // ✅ FIX : 1 seule slice de size[0] strings
            panzer_db_ptr->writeDataSlices(dataset_name_copy, {(size_t)size[0]}, 
                                          converted_data, 1, aos_timebase);
        } 
        else {
            // Statique
            panzer_db_ptr->writeData(dataset_name_copy, {(size_t)size[0]}, 
                                    converted_data, size[0], "");
        }
    } 
    else {
        throw ALBackendException("L'écriture de tableaux de chaînes de dimension > 2 n'est pas supportée.", LOG);
    }
  } 
  
  // ========== COMPLEXES ==========
  else if (datatype == alconst::complex_data) {
      if (!timebasename_copy.empty()) {
          size_t n_slices_dyn = 1;
          std::vector<size_t> base_shape;
          
          if (is_in_dynamic_aos) {
            // Mode itératif : dim = dimensions spatiales uniquement
            base_shape = shape; // Toutes les dimensions sont spatiales
            n_slices_dyn = 1;   // Une seule slice
            
            DEBUG_PRINT("Writing in dynamic AoS context: 1 slice of shape [" 
                       << (shape.empty() ? "scalar" : std::to_string(shape[0])) << "]");
        } 
        else {
            // Mode bulk : dernière dimension = temps
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
          
          // ✅ FIX : 1 seule slice avec shape complète
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
    throw ALBackendException("Type de données non supporté par HDF5Writer_v2.", LOG);
  }
  
}


void HDF5Writer_v2::endAction(Context *ctx) {
  
  if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
    // ✅ IMPORTANT : Ne PAS synchroniser ici !
    // La synchronisation a déjà eu lieu lors des write_ND_Data précédents
    // et dans beginWriteArraystructAction pour les enfants.
    // 
    // Pour les AOS vides, on s'appuie sur le fait que :
    // 1. beginArray() a déjà créé le méta-nœud avec la bonne taille
    // 2. Les indices vides n'ont pas besoin d'entrées dans l'index
    //printf("HDF5Writer_v2::endAction called for ArraystructContext, calling endArray()\n");
    //if (panzer_db_ptr) {
        //panzer_db_ptr->endArray();
    //}

    ArraystructContext* arrCtx = static_cast<ArraystructContext*>(ctx);
    
    // ✅ FIX : Ne fermer que si on a ouvert
    if (initialized_aos.count(arrCtx) > 0) {
        //printf("[DEBUG endAction] Closing AoS '%s'\n", arrCtx->getPath().c_str());
        if (panzer_db_ptr) panzer_db_ptr->endArray();
        initialized_aos.erase(arrCtx);
    } else {
        //printf("[DEBUG endAction] Skipping endArray() for AoS '%s' (was not initialized)\n", 
        //       arrCtx->getPath().c_str());
    }
    
    
  } else if (ctx->getType() == CTX_OPERATION_TYPE) {
    //printf("HDF5Writer_v2::endAction called for OperationContext, calling flush()\n");

    if (panzer_db_ptr) panzer_db_ptr->flush();
    if (panzer_db_ptr) panzer_db_ptr->dumpLeavesCache();
    if (panzer_db_ptr) panzer_db_ptr->close();
  }
}