#include "hdf5_reader_v2.h"

#include "al_backend.h"
#include "al_defs.h"
#include <algorithm>
#include <vector>
#include <cstring>
#include <complex>
#include <sstream>
#include "hdf5_utils.h"


// Macro de debug
#ifdef DEBUG_HDF5_READER_V2
#define DEBUG_PRINT(msg) \
  std::cerr << "[DEBUG " << __func__ << "] " << msg << std::endl
#else
#define DEBUG_PRINT(msg) \
  do {                   \
  } while (0)
#endif

HDF5Reader_v2::HDF5Reader_v2(std::pair<int,int> backend_version_)
    : HDF5Reader(backend_version_){} 

HDF5Reader_v2::~HDF5Reader_v2() = default;

std::string HDF5Reader_v2::getVersion()
{
    return "2.0.0";
}

void HDF5Reader_v2::closePulse(DataEntryContext *ctx, int mode, hid_t *file_id, std::unordered_map<std::string, hid_t> &opened_IDS_files, int files_path_strategy, std::string &files_directory, std::string &relative_file_path)
{
    HDF5Reader::closePulse(ctx, mode, file_id, opened_IDS_files, files_path_strategy, files_directory, relative_file_path);
}

void HDF5Reader_v2::open_IDS_group(OperationContext *ctx, hid_t file_id, std::unordered_map<std::string, hid_t> &opened_IDS_files, std::string &files_directory, std::string &relative_file_path)
{
    HDF5Reader::open_IDS_group(ctx, file_id, opened_IDS_files, files_directory, relative_file_path);
    
    global_strategy.reset();
    slice_strategy.reset();
    timerange_strategy.reset();
    read_strategy = nullptr;
}

void HDF5Reader_v2::close_group(OperationContext *ctx)
{
    HDF5Reader::close_group(ctx);
}


void HDF5Reader_v2::beginReadArraystructAction(ArraystructContext *ctx, int *size)
{
    prepare_strategy(ctx);
    read_strategy->beginReadArraystructAction(ctx, size);
}

void HDF5Reader_v2::endAction(Context *ctx)
{
    prepare_strategy(ctx);
    auto op_ctx = static_cast<OperationContext *>(ctx);

    if (read_strategy && IDS_group_id.size() == 1 && IDS_group_id.count(op_ctx)) {
        read_strategy->endAction(ctx);
    }
}

void HDF5Reader_v2::select_strategy(OperationContext *ctx, hid_t gid) {
    if (ctx->getRangemode() == GLOBAL_OP) {
        if (!global_strategy) global_strategy = std::make_unique<GlobalReadStrategy>(gid);
        read_strategy = global_strategy.get();
    } 
    else if (ctx->getRangemode() == SLICE_OP) {
        if (!slice_strategy) slice_strategy = std::make_unique<SliceReadStrategy>(gid);
        read_strategy = slice_strategy.get();
    }
    else if (ctx->getRangemode() == TIMERANGE_OP) {
        if (!timerange_strategy) timerange_strategy = std::make_unique<TimeRangeReadStrategy>(gid);
        read_strategy = timerange_strategy.get();
     }
    else {
        throw ALBackendException("Unknown operation context range mode", LOG);
    }
}

void HDF5Reader_v2::prepare_strategy(Context *ctx) {
    // 1. Extraire l'OperationContext selon le type de contexte
    OperationContext *opctx = nullptr;
    if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
        opctx = static_cast<ArraystructContext *>(ctx)->getOperationContext();
    } else {
        opctx = static_cast<OperationContext *>(ctx);
    }

    // 2. Trouver le GID
    hid_t gid = -1;
    auto it = IDS_group_id.find(opctx);
    if (it != IDS_group_id.end()) {
        gid = it->second;
    }

    if (gid == -1) {
        throw ALBackendException("No HDF5 group ID found for this context", LOG);
    }

    // 3. Sélectionner/Initialiser
    select_strategy(opctx, gid);
}

// Délégation de la méthode read_ND_Data à la stratégie de lecture
int HDF5Reader_v2::read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                                 int *datatype, void **data, int *dim, int *size) {
  
  prepare_strategy(ctx);

  std::string dataset_name_copy = dataset_name;
  std::string timebasename_copy = timebasename;
  std::replace(dataset_name_copy.begin(), dataset_name_copy.end(), '/', '&');
  //printf("--> HDF5Reader_v2::read_ND_Data called for dataset: %s\n", dataset_name_copy.c_str());
  
  int status = read_strategy->read_ND_Data(ctx, dataset_name_copy, timebasename_copy, datatype, data, dim, size);

  return status;
}
