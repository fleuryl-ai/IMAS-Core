#include "global_read_strategy.h"
#include "al_backend.h"
#include "al_defs.h"
#include <vector>
#include <cstring>
#include <complex>
#include <sstream>

#define HOMOGENEOUS_TIME_FIELD_NAME "ids_properties&homogeneous_time"
#define HOMOGENEOUS_TIME_BASIS_FIELD_NAME "time"

// Debug macro
#ifdef DEBUG_HDF5_READER_V2
#define DEBUG_PRINT(msg) \
  std::cerr << "[DEBUG " << __func__ << "] " << msg << std::endl
#else
#define DEBUG_PRINT(msg) \
  do {                   \
  } while (0)
#endif

GlobalReadStrategy::GlobalReadStrategy(hid_t loc_id)
    : IReadStrategy(loc_id) {}


void GlobalReadStrategy::beginReadArraystructAction(ArraystructContext *ctx, int *size) {

    std::string aos_path_token = getPath(ctx, false);
    DEBUG_PRINT("Preparing AOS for path: " << aos_path_token);
    auto shapes = panzer_db_ptr->getAOSShape(aos_path_token);
    if (!shapes.empty()) {
      DEBUG_PRINT("AOS retrieved with " << shapes[0] << " elements.");
    } 
    *size = shapes.empty() ? 0 : shapes[0]; 

}

void GlobalReadStrategy::endAction(Context *ctx) {
    if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
        // In read mode, panzer_db_ptr->endArray() is not necessary because array_stack is not used.
    } else if (ctx->getType() == CTX_OPERATION_TYPE) {
        if (panzer_db_ptr) panzer_db_ptr->close(); // If panzer_db_ptr is not null
    }
  else{
  }
}

int GlobalReadStrategy::read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                                     int *datatype, void **data, int *dim, int *size) {

    // Use the common method defined in IReadStrategy
    return read_dataset_globally(ctx, dataset_name, datatype, data, dim, size);
}
