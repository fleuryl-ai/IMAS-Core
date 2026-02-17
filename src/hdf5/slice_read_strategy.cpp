#include "slice_read_strategy.h"
#include "data_interpolation.h"
#include <complex>


SliceReadStrategy::SliceReadStrategy(hid_t loc_id)
    : IReadStrategy(loc_id) {}

void SliceReadStrategy::beginReadArraystructAction(ArraystructContext *ctx, int *size) {
    OperationContext* opCtx = ctx->getOperationContext();

    // Trouve le parent dynamique le plus proche pour identifier la bonne base de temps
    ArraystructContext* timed_parent = nullptr;
    Context* p = ctx;
    while(p && p->getType() == CTX_ARRAYSTRUCT_TYPE) {
        ArraystructContext* arr_p = static_cast<ArraystructContext*>(p);
        if (arr_p->getTimed()) {
            timed_parent = arr_p;
            break;
        }
        p = arr_p->getParent();
    }

    int64_t slice_idx = -1;
    if (timed_parent) {
        int homogeneous_time = getHomogeneousTime();
        std::string timebase_path;
        if (homogeneous_time == 1) {
            timebase_path = "time";
        } else {
            timebase_path = getPath(timed_parent, false);
            if (!timebase_path.empty()) timebase_path += "/";
            timebase_path += timed_parent->getTimebasePath();
        }
        slice_idx = panzer_db_ptr->getTimeIndex(timebase_path, opCtx->getTime());
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
        // En mode lecture, panzer_db_ptr->endArray() n'est pas nécessaire car array_stack n'est pas utilisé.
    } else if (ctx->getType() == CTX_OPERATION_TYPE) {
        //printf("GlobalReadStrategy::endAction called for OperationContext\n");
        //if (panzer_db_ptr) panzer_db_ptr->dumpLeavesCache(); // Dump du cache de feuilles pour le debug
        if (panzer_db_ptr) panzer_db_ptr->close(); // If panzer_db_ptr is not null
    }
  else{
  }
}

int SliceReadStrategy::read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                                    int *datatype, void **data, int *dim, int *size) {
    
    //bool is_timed = isTimedContext(ctx);
    //if (!is_timed && timebasename.empty()) return 0;

    int homogeneous_time = getHomogeneousTime();
    
     if (timebasename.empty() && !isTimedContext(ctx)) {                              
        int status = read_dataset_globally(ctx, dataset_name, datatype, data, dim, size);
        // If data is static, we read it globally and we are done.
        return status;  
     }                                  

    std::vector<double> time_basis_vector = getTimeValues(ctx, homogeneous_time, timebasename);

    double time;
    int interp;
    std::string aos_path = "";

    if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE){
        time  = dynamic_cast<ArraystructContext*>(ctx)->getOperationContext()->getTime();
        interp = dynamic_cast<ArraystructContext*>(ctx)->getOperationContext()->getInterpmode();
        aos_path = getPath(dynamic_cast<ArraystructContext*>(ctx));
    } 
    else if (ctx->getType() == CTX_OPERATION_TYPE){
        time  = dynamic_cast<OperationContext*>(ctx)->getTime();
        interp = dynamic_cast<OperationContext*>(ctx)->getInterpmode();
    }
    
    std::vector<int> ctx_indices;
    // On récupère les indices sans forcer l'index temporel (il sera géré par readInterpolatedData)
    ctx_indices = getIndices(ctx, -1);

    std::vector<uint64_t> indices(ctx_indices.begin(), ctx_indices.end());

    // Extraction du nom racine (ex: "A" depuis "A/0/B")
    std::string root_name = aos_path;
    size_t pos = aos_path.find('/');
    if (pos != std::string::npos) {
        root_name = aos_path.substr(0, pos);
    }

    uint64_t ndim_out = 0;
    uint64_t shape_out[6] = {0};
    void* data_out = nullptr;

    //const char* full_path = "A/0/B/0/C/0/D/0/tensor";
    std::string full_path = buildFullPath(ctx, dataset_name);
    const char* c_full_path = full_path.c_str();

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

    //printf("ndim_out = %llu\n", ndim_out);

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
        //printf("[SliceReadStrategy] read_ND_Data succeeded for dataset '%s' with time interpolation at t=%f\n", 
        //       dataset_name.c_str(), time);
        //printf(dim != nullptr ? "dim = %d\n" : "dim is nullptr\n", *dim);
        return 1;
    }
    //printf("[SliceReadStrategy] read_ND_Data failed for dataset '%s' with time interpolation at t=%f\n", 
    //       dataset_name.c_str(), time);
    return 0;
}