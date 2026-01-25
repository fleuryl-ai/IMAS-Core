#include "slice_read_strategy.h"
#include "data_interpolation.h"
#include <complex>
/**
 * @brief Constructor for the SliceReadStrategy.
 * @param loc_id The HDF5 location ID (file or group) for PanzerDB.
 */
SliceReadStrategy::SliceReadStrategy(hid_t loc_id)
    : IReadStrategy(loc_id) {}

/**
 * @brief Prepares for reading an Array of Structures (AoS) in slice mode.
 * 
 * For a static AoS, it retrieves the full size.
 * For a timed (dynamic) AoS, the size is always 1 in slice mode, as we are reading a single time slice.
 * 
 * @param ctx The context of the Arraystruct being read.
 * @param size Output pointer to store the calculated size of the AoS for this operation.
 */
void SliceReadStrategy::beginReadArraystructAction(ArraystructContext *ctx, int *size) {
    // Get the path to the AoS meta-node (e.g., "profiles_1d" not "profiles_1d/0")
    std::string path_container = getPath(ctx, false);

    auto shapes = panzer_db_ptr->getAOSShape(path_container);
    *size = shapes.empty() ? 0 : shapes[0]; 

    // In slice mode, a timed AoS is treated as having a size of 1.
    if (ctx->getTimed() && *size > 0) {
        *size = 1;
    }
}

/**
 * @brief Gathers the hierarchical indices from the context stack.
 * 
 * This helper function traverses the context hierarchy upwards from the current context,
 * collecting the index at each level of the Array of Structures.
 * For a timed AoS, it can substitute the context's index with a provided dynamic_index.
 * 
 * @param ctx The current context.
 * @param dynamic_index An optional index to substitute for timed AoS levels. If -1, the context's own index is used.
 * @return A vector of indices representing the path through the nested AoS, from root to current.
 */
std::vector<int> SliceReadStrategy::getIndices(Context *ctx, int dynamic_index) {
    std::vector<int> indices;
    if (ctx->getType() != CTX_ARRAYSTRUCT_TYPE) {
        return indices;
    }

    ArraystructContext *arrCtx = dynamic_cast<ArraystructContext *>(ctx);
    while (arrCtx != nullptr) {
        if (arrCtx->getTimed()) {
            if (dynamic_index == -1) {
                indices.push_back(arrCtx->getIndex());
            } else {
               indices.push_back(dynamic_index);
            }
        } else {
            indices.push_back(arrCtx->getIndex());
        }
        arrCtx = arrCtx->getParent();
    }
    std::reverse(indices.begin(), indices.end());
    return indices;
}

/**
 * @brief Finalizes an action. For an OperationContext, it closes the PanzerDB instance.
 * @param ctx The context to finalize.
 */
void SliceReadStrategy::endAction(Context *ctx) {
    // When the entire read operation for an IDS is finished (end of OperationContext),
    // we can close the PanzerDB instance to release file handles.
    if (ctx->getType() == CTX_OPERATION_TYPE) {
        if (panzer_db_ptr) {
            panzer_db_ptr->close();
        }
    }
}

/**
 * @brief Reads N-Dimensional data for a specific time slice.
 * 
 * This is the core method for the slice strategy.
 * - If the data is static (not time-dependent), it reads the entire dataset using the global reader helper.
 * - If the data is dynamic, it determines the target time and interpolation mode from the context,
 *   retrieves the relevant time basis, and then calls `PanzerDB::readInterpolatedData` to get the
 *   data for the specific time slice, performing interpolation if necessary.
 * 
 * @param ctx Current context.
 * @param dataset_name Name of the dataset to read.
 * @param timebasename Name of the timebase (if the data is dynamic).
 * @param datatype Output pointer for the data's type.
 * @param data Output pointer for the data buffer.
 * @param dim Output pointer for the number of dimensions.
 * @param size Output pointer for the dimensions' sizes.
 * @return 1 on success, 0 on failure.
 */
int SliceReadStrategy::read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                                    int *datatype, void **data, int *dim, int *size) {
    
    int homogeneous_time = getHomogeneousTime();
    
    // If the data is not inside a timed AoS and has no explicit timebase, it's static.
    // Read the entire dataset.
    if (timebasename.empty() && !isTimedContext(ctx)) {                              
        int status = read_dataset_globally(ctx, dataset_name, datatype, data, dim, size);
        return status;  
    }                                  

    // For dynamic data, get the appropriate time vector.
    std::vector<double> time_basis_vector = getTimeValues(ctx, homogeneous_time);

    double time;
    int interp;

    // Extract time and interpolation mode from the operation context.
    if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
        time  = dynamic_cast<ArraystructContext*>(ctx)->getOperationContext()->getTime();
        interp = dynamic_cast<ArraystructContext*>(ctx)->getOperationContext()->getInterpmode();
    } else if (ctx->getType() == CTX_OPERATION_TYPE) {
        time  = dynamic_cast<OperationContext*>(ctx)->getTime();
        interp = dynamic_cast<OperationContext*>(ctx)->getInterpmode();
    } else {
        return 0; // Should not happen
    }
    
    // Build the full, indexed path to the dataset.
    std::string full_path = buildFullPath(ctx, dataset_name);
    const char* c_full_path = full_path.c_str();

    uint64_t ndim_out = 0;
    uint64_t shape_out[6] = {0};
    void* data_out = nullptr;

    // Delegate the core logic of reading and interpolating to PanzerDB.
    int res = panzer_db_ptr->readInterpolatedData(
        c_full_path,
        time,
        time_basis_vector,
        interp,
        *datatype,
        &ndim_out,
        shape_out,
        &data_out,
        !isTimedContext(ctx) // expect_time_dim: if not in a timed context, the data itself should have a time dimension.
    );

    if (res == 0) {
        *data = data_out;
        *dim = ndim_out;
        for (size_t i = 0; i < ndim_out; ++i) {
            size[i] = (int)shape_out[i];
        }

        // AL Convention: For a time-dependent N-D array (N>0), a slice should be returned as an (N+1)-D array
        // with the last dimension of size 1. For a time-dependent scalar (N=0), the behavior is ambiguous.
        // Heuristic: A scalar signal defined at the root or in a static AoS gets its dimension promoted when sliced,
        // but a scalar signal defined inside a dynamic AoS remains a scalar when sliced.
        bool is_dynamic = !timebasename.empty() || isTimedContext(ctx);
        if (is_dynamic && *datatype != alconst::char_data) {
            if (!isTimedContext(ctx)) {
                 size[*dim] = 1;
                 (*dim)++;
            }
        }
        return 1;
    }

    return 0;
}