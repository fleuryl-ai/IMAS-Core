#include "hdf5_reader_v2.h"

#include "al_backend.h"
#include "al_defs.h"
#include <algorithm>
#include <vector>
#include <cstring>
#include <complex>
#include <sstream>
#include "hdf5_utils.h"


// Debug macro
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

    // New read session: drop the previous session's engine/index (their
    // destructors flush + close the HDF5 handles they own).
    session_gid = -1;
    session_db.reset();
    session_index.reset();
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
    // One shared engine + index per group: built on the first read that needs it,
    // adopted by every strategy of the session (3x /index load -> 1x).
    if (gid != session_gid) {
        session_gid = gid;
        session_db.reset();
        session_index.reset();
        global_strategy.reset();
        slice_strategy.reset();
        timerange_strategy.reset();
        read_strategy = nullptr;
    }
    if (!session_db) {
        session_db = std::make_shared<PanzerDB>(gid, PanzerDB::OpenMode::READ);
        session_index = std::make_shared<ReadIndex>(session_db);
    }

    if (ctx->getRangemode() == GLOBAL_OP) {
        if (!global_strategy) global_strategy = std::make_unique<GlobalReadStrategy>(session_db, session_index);
        read_strategy = global_strategy.get();
    } 
    else if (ctx->getRangemode() == SLICE_OP) {
        if (!slice_strategy) slice_strategy = std::make_unique<SliceReadStrategy>(session_db, session_index);
        read_strategy = slice_strategy.get();
    }
    else if (ctx->getRangemode() == TIMERANGE_OP) {
        if (!timerange_strategy) timerange_strategy = std::make_unique<TimeRangeReadStrategy>(session_db, session_index);
        read_strategy = timerange_strategy.get();
    }
    else {
        throw ALBackendException("Unknown operation context range mode", LOG);
    }
}

void HDF5Reader_v2::prepare_strategy(Context *ctx) {
    // 1. Extract the OperationContext according to the context type
    OperationContext *opctx = nullptr;
    if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
        opctx = static_cast<ArraystructContext *>(ctx)->getOperationContext();
    } else {
        opctx = static_cast<OperationContext *>(ctx);
    }

    // 2. Find the GID
    hid_t gid = -1;
    auto it = IDS_group_id.find(opctx);
    if (it != IDS_group_id.end()) {
        gid = it->second;
    }

    if (gid == -1) {
        throw ALBackendException("No HDF5 group ID found for this context", LOG);
    }

    // 3. Select/Initialize
    select_strategy(opctx, gid);
}

// Delegation of the read_ND_Data method to the read strategy
int HDF5Reader_v2::read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                                 int *datatype, void **data, int *dim, int *size) {
  
  prepare_strategy(ctx);

  std::string dataset_name_copy = dataset_name;
  std::string timebasename_copy = timebasename;
  std::replace(dataset_name_copy.begin(), dataset_name_copy.end(), '/', '&');

  int status = read_strategy->read_ND_Data(ctx, dataset_name_copy, timebasename_copy, datatype, data, dim, size);

  return status;
}
