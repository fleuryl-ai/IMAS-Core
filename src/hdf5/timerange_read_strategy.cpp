#include "timerange_read_strategy.h"
#include "data_interpolation.h"

TimeRangeReadStrategy::TimeRangeReadStrategy(hid_t loc_id)
    : IReadStrategy(loc_id) {}

void TimeRangeReadStrategy::beginReadArraystructAction(ArraystructContext *ctx, int *size) {

    int homogeneous_time = getHomogeneousTime();
    time_basis_vector = getTimeValues(ctx, homogeneous_time);

    if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
      ArraystructContext* arrctx = static_cast<ArraystructContext*>(ctx);
      //printf("Synchronizing PanzerDB index for AoS path: %s, to index: %d\n", arrctx->getPath().c_str(), arrctx->getIndex());
      if (panzer_db_ptr) panzer_db_ptr->setCurrentArrayIndex(arrctx->getIndex());
    }

    if (ctx->getTimed()) {

        std::map<std::string, int> times_indices;

        OperationContext *opCtx = ctx->getOperationContext();
        DataInterpolation data_interpolation_component;
        if (opCtx->time_range.dtime.size() != 0)
            { // resampling
                
                void *result = nullptr;
                // We want the total number of slices for the new time basis, so we pass -1 as the index
                *size = data_interpolation_component.resample_timebasis(opCtx->time_range.tmin, opCtx->time_range.tmax,
                                                            opCtx->time_range.dtime, -1,
                                                            time_basis_vector, &result);
            }
            else
            {
                double tmin;
                double tmax;
                if (opCtx->time_range.dtime.size() <= 1)
                {
                    tmin = opCtx->time_range.tmin;
                    tmax = opCtx->time_range.tmax;
                }
                else
                {
                    tmin = opCtx->time_range.dtime[0];
                    tmax = opCtx->time_range.dtime.back();
                }
                int time_range_tmin_index;
                int time_range_tmax_index;
                data_interpolation_component.getTimeRangeIndices(tmin, 
                                                                 tmax, 
                                                                 opCtx->time_range.dtime,
                                                                 time_basis_vector, 
                                                                 &time_range_tmin_index, 
                                                                 &time_range_tmax_index, 
                                                                 size, 
                                                                 opCtx->time_range.interpolation_method);
            }
    } else {
        std::string aos_path_token = getPath(ctx, false);
        auto shapes = panzer_db_ptr->getAOSShape(aos_path_token);
        *size = shapes.empty() ? 0 : shapes[0]; 
        //printf("-->Preparing AOS: %s\n", aos_path_token.c_str());
        //printf("size = %d\n", *size);
    }
    
}

void TimeRangeReadStrategy::endAction(Context *ctx) {
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

int TimeRangeReadStrategy::read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                                        int *datatype, void **data, int *dim, int *size) {

    DEBUG_PRINT("--> Entering read_ND_Data for dataset: " << dataset_name);

    OperationContext *opctx = nullptr;
    if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
        opctx = static_cast<ArraystructContext*>(ctx)->getOperationContext();
    } else {
        opctx = static_cast<OperationContext*>(ctx);
    }

    bool is_homogeneous_time_basis_dataset = (dataset_name == "time" || dataset_name == "/time");
    bool is_time_dataset = ends_with(dataset_name, "time"); 
    //bool is_time_basis_dataset = is_homogeneous_time_basis_dataset || is_time_dataset;
    bool is_inhomogeneous_time_basis_dataset = is_time_dataset && !is_homogeneous_time_basis_dataset;
    bool resampling = opctx->time_range.dtime.size() >= 1;
    bool homogeneous_time_basis_dataset_with_resampling = resampling && is_homogeneous_time_basis_dataset;
    bool inhomogeneous_time_basis_dataset_with_resampling = resampling && is_inhomogeneous_time_basis_dataset;
    
    DataInterpolation data_interpolation_component;

    if (dataset_name == "ids_properties/homogeneous_time" && opctx->time_range.dtime.size() != 0)
    {
        int *d = (int *)*data;
        *d = 1;
        return 1;
    }
    
    if (homogeneous_time_basis_dataset_with_resampling)
    {
        DEBUG_PRINT("Handling homogeneous_time_basis_dataset_with_resampling");
        std::vector<int> ctx_indices;
        int dynamic_index = -1;
        ctx_indices = getIndices(ctx, &dynamic_index);
        DEBUG_PRINT("dynamic_index: " << dynamic_index);
        double tmin;
        double tmax;
        if (opctx->time_range.dtime.size() <= 1)
        {
                tmin = opctx->time_range.tmin;
                tmax = opctx->time_range.tmax;
        }
         else
        {
                tmin = opctx->time_range.dtime[0];
                tmax = opctx->time_range.dtime.back();
        }
        DEBUG_PRINT("tmin: " << tmin << ", tmax: " << tmax << ", dtime size: " << opctx->time_range.dtime.size());
        *dim = 1;
        size[*dim - 1] = data_interpolation_component.resample_timebasis(tmin, tmax,
                                                            opctx->time_range.dtime, dynamic_index,
                                                            time_basis_vector, data);
        
        DEBUG_PRINT("resample_timebasis result size: " << size[*dim - 1]);
        
        return 1;
    }
    else if (inhomogeneous_time_basis_dataset_with_resampling)
    { // in this case, no value is returned
        return 0;
    }

    bool is_dynamic = timebasename.compare("") != 0 ? true : false;
    std::vector<int> ctx_indices;
        // On récupère les indices sans forcer l'index temporel (il sera géré par readInterpolatedData)
    int dynamic_index = -1;
    ctx_indices = getIndices(ctx, &dynamic_index);

    DEBUG_PRINT("is_dynamic=" << is_dynamic << ", dynamic_index=" << dynamic_index);

    if (dynamic_index == -1 && !is_dynamic)
    {
        int status = read_dataset_globally(ctx, dataset_name, datatype, data, dim, size);
        // If data is static, we read it globally and we are done.
        return status;
    }

    int time_min_index;
    int time_max_index;
    int range;

    double tmin;
    double tmax;

    std::map<std::string, int> times_indices;
    double requested_time; // requested time for time range with dtime !=-1 (resampling)
    int slice_index;

    if (time_basis_vector.empty()) {
        int homogeneous_time = getHomogeneousTime();
        time_basis_vector = getTimeValues(ctx, homogeneous_time);
        DEBUG_PRINT("Loaded time_basis_vector on demand, size: " << time_basis_vector.size());
    }

    if (opctx->time_range.dtime.size() <= 1)
        {
                tmin = opctx->time_range.tmin;
                tmax = opctx->time_range.tmax;
        }
    else
        {
                tmin = opctx->time_range.dtime[0];
                tmax = opctx->time_range.dtime.back();
        }

        data_interpolation_component.getTimeRangeIndices(tmin, tmax, opctx->time_range.dtime,
                                                             time_basis_vector, &time_min_index, &time_max_index, &range, opctx->time_range.interpolation_method);

        DEBUG_PRINT("Time range params: tmin=" << tmin << ", tmax=" << tmax);
        DEBUG_PRINT("Time basis size: " << time_basis_vector.size());
        DEBUG_PRINT("Calculated indices: time_min_index=" << time_min_index << ", time_max_index=" << time_max_index << ", range=" << range);

        
        
        if (dynamic_index == -1) {
            // No dynamic index found: we read the whole dataset (global behavior)
            // then we slice/resample according to the time range
            void* full_data = nullptr;
            int full_dim = 0;
            int full_size[6] = {0};
            
            int status = read_dataset_globally(ctx, dataset_name, datatype, &full_data, &full_dim, full_size);
            DEBUG_PRINT("read_dataset_globally status: " << status << ", full_dim: " << full_dim << ", full_size[0]: " << (full_dim > 0 ? full_size[0] : -1));
            if (status == 0 || full_data == nullptr) return 0;

            if (opctx->time_range.dtime.size() != 0) {
                 // Resampling

                // Since interpolate_with_resampling frees the data buffer it receives,
                // we must create a new buffer containing only the relevant time range
                // from the full dataset.
                int stop_idx = time_max_index;
                // For linear interpolation, we need the next point as well
                if (opctx->time_range.interpolation_method == alconst::linear_interp && stop_idx < (int)time_basis_vector.size() - 1) {
                    stop_idx++;
                }

                if (time_min_index == -1 || stop_idx == -1 || time_min_index > stop_idx) {
                    free(full_data);
                    return 0;
                }

                size_t element_size_bytes = 0;
                if (*datatype == alconst::double_data) element_size_bytes = sizeof(double);
                else if (*datatype == alconst::integer_data) element_size_bytes = sizeof(int);
                else if (*datatype == alconst::char_data) element_size_bytes = sizeof(char);
                else if (*datatype == alconst::complex_data) element_size_bytes = sizeof(std::complex<double>);

                size_t slice_size_elements = 1;
                for(int i=0; i<full_dim-1; ++i) slice_size_elements *= full_size[i];

                size_t count = stop_idx - time_min_index + 1;
                size_t data_for_interp_bytes = count * slice_size_elements * element_size_bytes;
                void* data_for_interp = malloc(data_for_interp_bytes);
                if (!data_for_interp) { free(full_data); throw ALBackendException("Memory allocation failed", LOG); }

                char* src_ptr = (char*)full_data + (time_min_index * slice_size_elements * element_size_bytes);
                memcpy(data_for_interp, src_ptr, data_for_interp_bytes);
                free(full_data);

                // Also slice the time vector to match the data buffer
                std::vector<double> time_vector_for_interp(time_basis_vector.begin() + time_min_index, time_basis_vector.begin() + stop_idx + 1);

                 void* resampled_data = nullptr;
                 int nb_slices = data_interpolation_component.interpolate_with_resampling(
                     opctx->time_range.tmin, opctx->time_range.tmax, opctx->time_range.dtime,
                     *datatype, full_size, full_dim, data_for_interp, time_vector_for_interp, &resampled_data,
                     opctx->time_range.interpolation_method
                 );
                 
                 *data = resampled_data;
                 *dim = full_dim;
                 for(int i=0; i<full_dim; ++i) size[i] = full_size[i];
                 size[full_dim-1] = nb_slices;
                 return 1;
            } else {
                 // No resampling, just slicing [time_min_index, time_max_index]
                 if (time_min_index == -1 || time_max_index == -1) {
                     DEBUG_PRINT("Invalid time indices: " << time_min_index << ", " << time_max_index);
                     free(full_data);
                     return 0;
                 }
                 int count = time_max_index - time_min_index + 1;
                 DEBUG_PRINT("Slicing data: from index " << time_min_index << " to " << time_max_index << " (count=" << count << ")");
                 if (count <= 0) {
                     free(full_data);
                     return 0;
                 }
                 
                 size_t element_size_bytes = 0;
                 if (*datatype == alconst::double_data) element_size_bytes = sizeof(double);
                 else if (*datatype == alconst::integer_data) element_size_bytes = sizeof(int);
                 else if (*datatype == alconst::char_data) element_size_bytes = sizeof(char);
                 else if (*datatype == alconst::complex_data) element_size_bytes = sizeof(std::complex<double>);
                 
                 size_t slice_size_elements = 1;
                 for(int i=0; i<full_dim-1; ++i) slice_size_elements *= full_size[i];
                 DEBUG_PRINT("Elements per slice: " << slice_size_elements);
                 
                 size_t offset_elements = time_min_index * slice_size_elements;
                 size_t count_elements = count * slice_size_elements;
                 
                 void* sliced_data = malloc(count_elements * element_size_bytes);
                 if (!sliced_data) {
                     free(full_data);
                     throw ALBackendException("Memory allocation failed in TimeRangeReadStrategy", LOG);
                 }
                 
                 char* src_ptr = (char*)full_data;
                 memcpy(sliced_data, src_ptr + offset_elements * element_size_bytes, count_elements * element_size_bytes);
                 
                 free(full_data);
                 *data = sliced_data;
                 *dim = full_dim;
                 for(int i=0; i<full_dim; ++i) size[i] = full_size[i];
                 size[full_dim-1] = count;
                 DEBUG_PRINT("Final size[0] for sliced data: " << size[0]);
                 
                 return 1;
            }
        }
        else {
            DEBUG_PRINT("Dynamic index for time-dependent read: " << dynamic_index);
        
            std::vector<uint64_t> indices(ctx_indices.begin(), ctx_indices.end());                                                     
            int interp_mode;

            if (opctx->time_range.dtime.size() != 0) {
                // Resampling case: calculate the requested time for this specific index
                if (opctx->time_range.dtime.size() == 1) {
                    requested_time = opctx->time_range.tmin + dynamic_index * opctx->time_range.dtime[0];
                } else {
                    requested_time = opctx->time_range.dtime[dynamic_index];
                }
                interp_mode = opctx->time_range.interpolation_method;
            } else {
                // No resampling: get the value at the specific grid point
                slice_index = time_min_index + dynamic_index;
                DEBUG_PRINT("No resampling. slice_index = " << time_min_index << " + " << dynamic_index << " = " << slice_index);
                if (slice_index >= time_basis_vector.size() || slice_index < 0) {
                    return 0; // Index is out of the valid range for the time basis
                }
                requested_time = time_basis_vector[slice_index];
                interp_mode = alconst::closest_interp; // Use closest to pick the exact value at the grid point
            }

            DEBUG_PRINT("Requesting interpolated data at time=" << requested_time << " with interp_mode=" << interp_mode);
            // Get root name of the AoS hierarchy
            std::string aos_path;
            if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
                aos_path = getPath(static_cast<ArraystructContext*>(ctx));
            }
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

            // --- METADATA HANDLING ---
            std::map<std::string, std::string> meta = panzer_db_ptr->readMetadata(full_path);
            if (!meta.empty()) {
                DEBUG_PRINT("Loaded " << meta.size() << " metadata entries for " << full_path);
            }

            int res = panzer_db_ptr->readInterpolatedData(
                c_full_path,
                requested_time,
                time_basis_vector,
                interp_mode,
                *datatype,
                &ndim_out,
                shape_out,
                &data_out,
                !isTimedContext(ctx) // expect_time_dim
            );

            DEBUG_PRINT("readInterpolatedData result: " << res);

            if (res == 0) {
                *data = data_out;
                *dim = ndim_out;
                for (size_t i = 0; i < ndim_out; ++i) size[i] = (int)shape_out[i];
                return 1;
            }
        } 
            
    return 0; 
}

bool TimeRangeReadStrategy::ends_with(const std::string &str, const std::string &suffix)
{
    int pos = str.size() - suffix.size();
    if (pos >= 0)
        return str.compare(pos, suffix.size(), suffix) == 0;
    return false;
}

std::vector<int> TimeRangeReadStrategy::getIndices(Context *ctx, int *dynamic_index) {
    std::vector<int> indices;
    if (ctx->getType() != CTX_ARRAYSTRUCT_TYPE) {
        return indices;
    }

    ArraystructContext *arrCtx = dynamic_cast<ArraystructContext *>(ctx);
    while (arrCtx != nullptr) {
        if (arrCtx->getTimed()) {
            indices.push_back(arrCtx->getIndex());
            *dynamic_index = arrCtx->getIndex();
        } else {
            indices.push_back(arrCtx->getIndex());
        }
        arrCtx = arrCtx->getParent();
    }
    std::reverse(indices.begin(), indices.end());
    return indices;
}