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

HDF5Reader_v2::HDF5Reader_v2(std::string backend_version_)
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
    
    hid_t gid = -1;
    auto it = IDS_group_id.find(ctx);
    if (it != IDS_group_id.end()) {
        gid = it->second;
    }

    if (gid < 0)
        throw ALBackendException("HDF5Reader_v2: Could not get a valid group ID to create a read strategy.", LOG);

    if (ctx->getRangemode() == GLOBAL_OP) { 
        read_strategy = std::make_unique<GlobalReadStrategy>(gid); 
    } else if (ctx->getRangemode() == SLICE_OP) { 
        read_strategy = std::make_unique<SliceReadStrategy>(gid); 
    } else if (ctx->getRangemode() == TIMERANGE_OP) { 
        read_strategy = std::make_unique<TimeRangeReadStrategy>(gid); 
    } else {
        throw ALBackendException("Unkwown operation context range mode", LOG);
    }

    // ✅ OPTIMIZATION: Clear context path cache because group/strategy changed
    // read_strategy->clearContextPathCache(); // -> Method to add in IReadStrategy

    // The call to build_path_index() is now in the IReadStrategy constructor.
}

void HDF5Reader_v2::close_group(OperationContext *ctx)
{
    HDF5Reader::close_group(ctx);
}


void HDF5Reader_v2::beginReadArraystructAction(ArraystructContext *ctx, int *size)
{
    read_strategy->beginReadArraystructAction(ctx, size);

}

void HDF5Reader_v2::endAction(Context *ctx)
{
    read_strategy->endAction(ctx);

}

// Delegation of the read_ND_Data method to the read strategy
int HDF5Reader_v2::read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                                 int *datatype, void **data, int *dim, int *size) {
    // The read logic is now delegated to the strategy object
  std::string dataset_name_copy = dataset_name;
  std::string timebasename_copy = timebasename;

  std::replace(dataset_name_copy.begin(), dataset_name_copy.end(), '/', '&');
  std::replace(dataset_name_copy.begin(), dataset_name_copy.end(), '/', '&');

  //printf("HDF5Reader_v2::read_ND_Data called for dataset: %s\n", dataset_name.c_str());
  int status = read_strategy->read_ND_Data(ctx, dataset_name_copy, timebasename_copy, datatype, data, dim, size);
  //printf("done HDF5Reader_v2::read_ND_Data for dataset: %s with status: %d\n", dataset_name.c_str(), status);
  //printf("Data dimension: %d\n", *dim);
  return status;
}
