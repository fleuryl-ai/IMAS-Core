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

// Forward declarations
class Context;

// Macro de debug
#ifdef DEBUG_HDF5_READER_V2
#define DEBUG_PRINT(msg) \
  std::cerr << "[DEBUG " << __func__ << "] " << msg << std::endl
#else
#define DEBUG_PRINT(msg) \
  do {                   \
  } while (0)
#endif

class IReadStrategy {

protected:
    // OPTIMISATION : Cache pour accès rapide chemin -> feuilles
    // Clé : Chemin complet (string)
    // Valeur : Vecteur de pointeurs vers les feuilles partageant ce chemin (différents time_index)
    std::unordered_map<std::string_view, std::vector<const PanzerDB::Leaf*>> path_cache;
    // ✅ OPTIMISATION CRITIQUE: Cache pour les chemins de contexte
    // Évite de reconstruire le chemin du parent à chaque appel de read_ND_Data
    // Clé: Pointeur vers le contexte
    // Valeur: Chemin pré-calculé
    mutable std::unordered_map<Context*, std::string> context_path_cache;
    std::unique_ptr<PanzerDB> panzer_db_ptr;

public:
    IReadStrategy(hid_t loc_id) {
        panzer_db_ptr = std::make_unique<PanzerDB>(loc_id, PanzerDB::OpenMode::READ);
        build_path_index(); // Construire l'index juste après l'initialisation de PanzerDB
    } 
    virtual void beginReadArraystructAction(ArraystructContext * ctx, int *size) = 0;
    virtual void endAction(Context * ctx) = 0;
    virtual ~IReadStrategy() = default;

    //virtual void build_path_index();
    virtual int read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                             int* datatype, void **data, int *dim, int *size) = 0;

    int getHomogeneousTime() {
        if (!panzer_db_ptr) return -1;
        int homogeneous_time = 1;
        int status = -1;
        int temp = panzer_db_ptr->readScalar<int32_t>("ids_properties&homogeneous_time", &status);
        if (status == 0)
           homogeneous_time = temp;
        return homogeneous_time;
    }

    void build_path_index() {
        path_cache.clear();
        if (!panzer_db_ptr) return;

        const auto& leaves = panzer_db_ptr->getLeaves();
        
        // On pré-réserve pour éviter les réallocations
        path_cache.reserve(leaves.size());

        for (const auto& leaf : leaves) {
            // On stocke le pointeur vers la feuille dans la map
            path_cache[leaf.path].push_back(&leaf);
        }
        
        DEBUG_PRINT("Path index built with " << path_cache.size() << " unique paths.");
    }


    const PanzerDB::Leaf* find_leaf_for_context(Context* ctx, std::string_view dataset_name, std::string_view timebasename, int homogeneous_time) {
    DEBUG_PRINT("Searching for dataset '" << dataset_name << "' with timebasename='" << timebasename << "' and homogeneous_time=" << homogeneous_time);

    // --- Reconstruction du chemin AVEC le chemin parent COMPLET ---
    std::vector<std::string> path_segments;
    std::vector<int> indices;
    Context* curr = ctx;
    
    // ✅ FIX: Remonter jusqu'à OperationContext pour capturer le chemin complet
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

    // ✅ Construire le chemin COMPLET avec indices
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
    
    // ✅ Calculer context_prefix (chemin sans le dataset final)
    std::string context_prefix = strict_target_path;
    if (!path_segments.empty()) context_prefix += "/";
    
    // ✅ Ajouter le dataset pour obtenir le chemin complet
    if (!path_segments.empty()) strict_target_path += "/";
    strict_target_path += clean_ds_name;
    
    // Ex: "core_sources/source/0/species&neutral&state&vibrational_level"

    DEBUG_PRINT("Reconstructed strict_target_path: '" << strict_target_path << "'");
    DEBUG_PRINT("Context prefix: '" << context_prefix << "'");

    int64_t target_time = -1;
    if (!timebasename.empty()) {
        target_time = 0;
    }

    // --- PASSE 1 OPTIMISÉE : Recherche via Hash Map (O(1)) ---
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

    // --- PASSE 2 : Fallback (Recherche linéaire) ---
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

  // Dans slice_read_strategy.cpp, remplacer la fonction getTimeValues par :

std::vector<double> getTimeValues(Context *ctx, int homogeneous_time, const std::string& timebasename = "") {
    std::vector<double> time_values;

    if (homogeneous_time == 1) {
        time_values = panzer_db_ptr->getWholeDynamicSignal("time");
        return time_values;
    }

    // --- Logique pour homogeneous_time == 0 ---

    if (ctx == nullptr) {
        return time_values; // Return empty vector
    }

    ArraystructContext *arrCtx = dynamic_cast<ArraystructContext*>(ctx);
    if (!arrCtx) { // Cas où le contexte est OperationContext
        time_values = panzer_db_ptr->getWholeDynamicSignal("time"); // Fallback pour le temps racine
        return time_values;
    }

    // On cherche le parent "timed" pour construire le chemin de la base de temps
    ArraystructContext* timed_ctx = arrCtx;
    while(timed_ctx != nullptr && !timed_ctx->getTimed()) {
        timed_ctx = timed_ctx->getParent();
    }

    // If no dynamic parent is found, it could be a dynamic signal inside a static AoS.
    // In this case, the time vector is also static relative to the current context.
    if (!timed_ctx) {
        if (!timebasename.empty()) {
            const PanzerDB::Leaf* leaf = find_leaf_for_context(ctx, timebasename, "", 0);
            if (leaf && leaf->count > 0) {
                time_values.resize(leaf->count);
                panzer_db_ptr->readTensor(*leaf, time_values.data());
                return time_values;
            }
        }
        return {}; // No timebase found
    }

    std::string full_timebase_path = getPath(timed_ctx, false); // false = pas d'index final
    if (!full_timebase_path.empty()) {
        full_timebase_path += "/";
    }
    full_timebase_path += timed_ctx->getTimebasePath();

    auto leaves = panzer_db_ptr->getLeaves();

    // ✅ Filtrer pour ne garder QUE les feuilles correspondant au chemin exact
    std::map<uint64_t, const PanzerDB::Leaf*> time_leaves_map;

    for (const auto& leaf : leaves) {
        // The full_timebase_path is a "generic" path like "time_slice/time".
        // The actual leaf paths are "time_slice/0/time", "time_slice/1/time", etc.
        // We need to match the pattern: timed_aos_path + "/" + index + "/" + timebase_name
        
        // 1. Check if the leaf's parent path starts with the AoS path.
        //    e.g., parent_path "time_slice/0" starts with "time_slice"
        if (leaf.parent_path.rfind(getPath(timed_ctx, false), 0) == 0) {
            // 2. Check if the leaf's own name is the timebase name.
            size_t last_slash = leaf.path.find_last_of('/');
            if (last_slash != std::string::npos) {
                std::string_view leaf_name = leaf.path.substr(last_slash + 1);
                if (leaf_name == timed_ctx->getTimebasePath() && !leaf.is_empty) {
                    time_leaves_map[leaf.time_index] = &leaf;
                }
            }
        }
    }

    // Lire les valeurs dans l'ordre des time_index
    for (const auto& [time_idx, leaf_ptr] : time_leaves_map) {
        if (leaf_ptr->count > 0) {
            std::vector<double> temp_data(leaf_ptr->count);
            panzer_db_ptr->readTensor(*leaf_ptr, temp_data.data());
            time_values.insert(time_values.end(), temp_data.begin(), temp_data.end());
        }
    }

    return time_values;
}

   protected: // La méthode est `protected` pour être accessible par les classes filles
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
        // Si un override est fourni ET que le contexte actuel est dynamique, on l'utilise.
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
    return path_stream.str();
}

   /**
    * @brief Construit le chemin hiérarchique complet vers un dataset à l'intérieur d'AoS imbriqués.
    *        Le chemin est formaté comme "AOS1/index1/AOS2/index2/.../dataset".
    * 
    * @param ctx Le contexte actuel, doit être un ArraystructContext ou un de ses enfants.
    * @param dataset_name Le nom du dataset final.
    * @return std::string Le chemin complet, par exemple "A/0/B/0/data".
    */
   std::string buildFullPath(Context* ctx, const std::string& dataset_name) {
       std::vector<std::pair<std::string, int>> segments;
       Context* current_ctx = ctx;

       // Remonte la hiérarchie des contextes pour collecter les noms et indices des AoS.
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
    
   protected: // La méthode est `protected` pour être accessible par les classes filles
    int readLeafData(const PanzerDB::Leaf* leaf, int datatype, void **data, int *dim, int *size) {
        if (!panzer_db_ptr) {
            throw ALBackendException("PanzerDB not initialized", LOG);
        }
        if (!leaf) {
            return 0;
        }

        try {
            if (datatype == alconst::double_data) {
                *data = new double[leaf->count];
                panzer_db_ptr->readTensor<double>(*leaf, static_cast<double*>(*data));
            } else if (datatype == alconst::integer_data) {
                *data = new int32_t[leaf->count];
                panzer_db_ptr->readTensor<int32_t>(*leaf, static_cast<int32_t*>(*data));
            } else if (datatype == alconst::complex_data) {
                *data = new std::complex<double>[leaf->count];
                panzer_db_ptr->readTensor<std::complex<double>>(*leaf, static_cast<std::complex<double>*>(*data));
            } else if (datatype == alconst::char_data) {
                if (leaf->shape.size() <= 1) { // Liste 1D ou Scalaire
                    std::vector<std::string> str_list(leaf->count);
                    panzer_db_ptr->readTensor<std::string>(*leaf, str_list.data());
                    
                    *data = new char*[leaf->count];
                    char** out_ptr = static_cast<char**>(*data);
                    for (size_t i = 0; i < leaf->count; ++i) {
                         size_t len = str_list[i].size() + 1;
                         out_ptr[i] = new char[len];
                         std::memcpy(out_ptr[i], str_list[i].c_str(), len);
                    }
                } else {
                    throw ALBackendException("HDF5Reader_v2: Multidimensional strings not supported", LOG);
                }
            } else {
                throw ALBackendException("HDF5Reader_v2: Unknown datatype", LOG);
            }

            // Mettre à jour les dimensions de sortie
            *dim = leaf->shape.size();
            for (size_t i = 0; i < leaf->shape.size(); ++i) {
                size[i] = leaf->shape[i];
            }

            return 1; // Succès
        } catch (const std::exception& e) {
            throw ALBackendException(std::string("PanzerDB read error: ") + e.what(), LOG);
        }
    }

    // Méthode commune pour lire un dataset entier (toutes les tranches temporelles concaténées)
    // Refactorisé depuis GlobalReadStrategy pour être utilisé par TimeRangeReadStrategy
    int read_dataset_globally(Context *ctx, std::string &dataset_name, int* datatype, void **data, int *dim, int *size) {
        DEBUG_PRINT("--> Entering read_dataset_globally for dataset: " << dataset_name);

        if (!panzer_db_ptr) {
            throw ALBackendException("PanzerDB not initialized", LOG);
        }

        // ✅ OPTIMISATION: Utiliser le cache de chemin de contexte
        std::string context_prefix;
        auto cache_it = context_path_cache.find(ctx);
        if (cache_it != context_path_cache.end()) {
            context_prefix = cache_it->second;
        } else {
            // Le chemin n'est pas en cache, on le construit et on le stocke
            std::stringstream ss_prefix;
            // ... (la logique de construction du chemin reste ici)
            // Après la construction, stocker dans le cache :
            // context_path_cache[ctx] = ss_prefix.str();
            // Pour l'instant, on va intégrer la logique directement ci-dessous.
        }

        // 1. Reconstruction du chemin
        std::vector<std::string> path_segments;
        std::vector<int> indices;
        std::vector<bool> is_dynamic_level;
        int64_t target_time_index = -1;
        Context* curr = ctx;
        while (curr != nullptr) {
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

        // 2. Construction du chemin complet et récupération des feuilles
        // (Intégration de la logique de cache ici)
        std::stringstream ss_specific;
        for (size_t i = 0; i < path_segments.size(); ++i) {
            ss_specific << path_segments[i] << "/" << indices[i] << "/";
        }
        context_prefix = ss_specific.str();
        context_path_cache[ctx] = context_prefix; // Mise en cache

        ss_specific << clean_ds_name;
        std::string specific_path = ss_specific.str();

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
            // Fallback : Recherche linéaire pour les signaux dynamiques lus depuis un parent
            // (ex: lire "profiles_1d/signal" depuis la racine)
            // Note: Ce cas est rare si on utilise correctement les contextes.
            DEBUG_PRINT("Leaf not found for: " << specific_path << " or generic variant");
            return 0;
        }

        std::sort(sorted_leaves.begin(), sorted_leaves.end(), 
            [](const PanzerDB::Leaf* a, const PanzerDB::Leaf* b) {
                return a->time_index < b->time_index;
            });

        size_t total_elements = 0;
        for (const auto* leaf : sorted_leaves) {
            total_elements += leaf->count;
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

        // Déterminer le type réel à partir des flags de la feuille
        PanzerDB::DataType actual_type_enum = static_cast<PanzerDB::DataType>(first_leaf->flags >> 4);
        int actual_datatype = 0;
        switch(actual_type_enum) {
            case PanzerDB::DataType::FLOAT64: actual_datatype = alconst::double_data; break;
            case PanzerDB::DataType::INT32: actual_datatype = alconst::integer_data; break;
            case PanzerDB::DataType::STRING: actual_datatype = alconst::char_data; break;
            case PanzerDB::DataType::COMPLEX128: actual_datatype = alconst::complex_data; break;
            default:
                DEBUG_PRINT("Unknown data type in leaf flags: " << (first_leaf->flags >> 4));
                return 0; // Type inconnu
        }

        size_t leaf_rank = first_leaf->shape.size();

        // 3. Traitement CHAR / STRING
        // On lit toujours avec le type réel du fichier. La conversion sera faite par al_lowlevel.
        if (actual_datatype == alconst::char_data) {
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
             
             bool return_as_scalar = (is_scalar_leaf && total_elements == 1);

             if (return_as_scalar) {
                 *dim = 1;
                 size[0] = (int)temp_buffer[0].size(); // Length excluding null
                 *data = new char[size[0] + 1];
                 memcpy(*data, temp_buffer[0].c_str(), size[0] + 1);
             } else {
                 *dim = 2;
                 size[0] = (int)total_elements;
                 size[1] = (int)max_str_len;
                 
                 size_t buffer_bytes = total_elements * max_str_len;
                 char* char_buffer = new char[buffer_bytes];
                 std::memset(char_buffer, 0, buffer_bytes);
                 for (size_t i = 0; i < total_elements; ++i) {
                     if (!temp_buffer[i].empty()) strncpy(char_buffer + (i * max_str_len), temp_buffer[i].c_str(), max_str_len);
                 }
                 *data = char_buffer;
             }
 
             *datatype = actual_datatype; // On retourne le type qui a été lu
             return 1;
        }

        // 4. Traitement NUMÉRIQUE
        // 4. Traitement NUMÉRIQUE
        if (leaf_rank == 0) {
            // Signal 0D (scalaire) → devient 1D avec dimension temporelle
            if (total_elements > 1) { 
                *dim = 1; 
                size[0] = (int)total_elements; 
            } else { 
                *dim = 0; 
                // size[0] = 1; // Implicite pour un scalaire
            }
        } else {
            // Signal N-D → Ajouter dimension temporelle SI plusieurs tranches
            size_t spatial_product = 1;
            for (size_t i = 0; i < leaf_rank; ++i) { 
                size[i] = (int)first_leaf->shape[i]; 
                spatial_product *= first_leaf->shape[i]; 
            }
            
            size_t n_time_slices = (spatial_product > 0) ? (total_elements / spatial_product) : 1;
            
            if (n_time_slices > 1) {
                *dim = (int)(leaf_rank + 1); // ✅ +1 pour dimension temporelle
                size[leaf_rank] = (int)n_time_slices;
            } else {
                *dim = (int)leaf_rank; // Pas de dimension temporelle si 1 seule slice
            }
        }

        // Allocation
        if (actual_datatype == alconst::double_data) *data = malloc(total_elements * sizeof(double));
        else if (actual_datatype == alconst::integer_data) *data = malloc(total_elements * sizeof(int32_t));
        else if (actual_datatype == alconst::complex_data) *data = malloc(total_elements * sizeof(std::complex<double>));
        else return 0;

        // OPTIMISATION: Lecture groupée (Hyperslab Union)
        int res = panzer_db_ptr->readLeavesUnion(sorted_leaves, *data, actual_type_enum);
        if (res < 0) return 0;

        *datatype = actual_datatype; // On retourne le type qui a été lu
        return 1;
    }
    
};

#endif // I_READ_STRATEGY_H