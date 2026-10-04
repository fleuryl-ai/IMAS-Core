#include "slice_read_strategy.h"
#include "data_interpolation.h"
#include <complex>


SliceReadStrategy::SliceReadStrategy(std::shared_ptr<PanzerDB> panzer_db, std::shared_ptr<ReadIndex> read_index)
    : IReadStrategy(std::move(panzer_db), std::move(read_index)) {}

void SliceReadStrategy::beginReadArraystructAction(ArraystructContext *ctx, int *size) {
    refresh_index_if_needed();
    OperationContext* opCtx = ctx->getOperationContext();

    // Find the closest dynamic parent to identify the correct time base.
    ArraystructContext* timed_parent = nearestTimedContext(ctx);

    int64_t slice_idx = -1;
    if (timed_parent) {
        std::string timebase_name = timed_parent->getTimebasePath();
        if (timebase_name.empty()) timebase_name = "time";

        timebase_name = cleanFlatPath(sanitize_path(timed_parent, timebase_name));

        std::string timebase_path = getPath(timed_parent, false);
        if (!timebase_path.empty()) timebase_path += "/";
        timebase_path += timebase_name;

        slice_idx = panzer_db_ptr->getTimeIndex(timebase_path, opCtx->getTime(), opCtx->getInterpmode());
    }

    std::string path_container = getPath(ctx, false, slice_idx);

    auto shapes = panzer_db_ptr->getAOSShape(path_container);
    *size = shapes.empty() ? 0 : shapes[0]; 
    if (ctx->getTimed() && *size > 0)
        *size = 1;
}

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
            }
            else 
               indices.push_back(dynamic_index);
        } else {
            indices.push_back(arrCtx->getIndex());
        }
        arrCtx = arrCtx->getParent();
    }
    std::reverse(indices.begin(), indices.end());
    return indices;
}

void SliceReadStrategy::endAction(Context *ctx) {
    if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
        // In read mode, panzer_db_ptr->endArray() is not necessary because array_stack is not used.
    } else if (ctx->getType() == CTX_OPERATION_TYPE) {
        // Nothing to close: the shared engine is flushed/closed by RAII when the
        // read session is reset (HDF5Reader_v2) or at its destruction.
    }
  else{
  }
}

int SliceReadStrategy::read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                                    int *datatype, void **data, int *dim, int *size) {
    refresh_index_if_needed();

    int homogeneous_time = getHomogeneousTime();
    
     if (timebasename.empty() && !isTimedContext(ctx)) {                              
        int status = read_dataset_globally(ctx, dataset_name, datatype, data, dim, size);
        // If data is static, we read it globally and we are done.
        return status;  
     }                                  

    std::vector<double> time_basis_vector = getTimeValues(ctx, homogeneous_time, timebasename);

    double time;
    int interp;

    if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE){
        time  = dynamic_cast<ArraystructContext*>(ctx)->getOperationContext()->getTime();
        interp = dynamic_cast<ArraystructContext*>(ctx)->getOperationContext()->getInterpmode();
    } 
    else if (ctx->getType() == CTX_OPERATION_TYPE){
        time  = dynamic_cast<OperationContext*>(ctx)->getTime();
        interp = dynamic_cast<OperationContext*>(ctx)->getInterpmode();
    }
    
    std::vector<int> ctx_indices;
    // We retrieve the indices without forcing the time index (it will be managed by readInterpolatedData)
    ctx_indices = getIndices(ctx, -1);

    std::vector<uint64_t> indices(ctx_indices.begin(), ctx_indices.end());

    uint64_t ndim_out = 0;
    uint64_t shape_out[6] = {0};
    void* data_out = nullptr;

    std::string full_path = buildFullPath(ctx, dataset_name);
    const char* c_full_path = full_path.c_str();

    // Read the node's metadata (also marks it processed to avoid re-reading it later).
    panzer_db_ptr->readMetadata(full_path);

    int res = panzer_db_ptr->readInterpolatedData(
        c_full_path,
        time,
        time_basis_vector,
        interp,
        *datatype,
        &ndim_out,
        shape_out,
        &data_out,
        !isTimedContext(ctx) // expect_time_dim
    );

    if (res == 0) {
        *data = data_out;
        *dim = ndim_out;
        for (size_t i = 0; i < ndim_out; ++i) {
            size[i] = (int)shape_out[i];
        }

        // AL convention, made explicit (to_improve.md point 3): a time-dependent
        // scalar signal outside a dynamic AoS gets a slice dimension of size 1;
        // a scalar inside a dynamic AoS is already selected by that dynamic slice
        // and remains a scalar.
        if (shouldPromoteTimeScalarOnSlice(ctx, timebasename, *datatype)) {
            size[*dim] = 1;
            (*dim)++;
        }
        return 1;
    }
    return 0;
}