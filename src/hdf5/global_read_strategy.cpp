#include "global_read_strategy.h"
#include "al_context.h"

GlobalReadStrategy::GlobalReadStrategy(hid_t loc_id)
    : IReadStrategy(loc_id) {}

void GlobalReadStrategy::beginReadArraystructAction(ArraystructContext *ctx, int *size) {
    // For a global read, we need the full size of the Array of Structures.
    // The getPath method constructs the path to the AoS meta-node.
    // We pass 'false' to exclude the index of the context itself, giving us the path to the array, not an element.
    std::string aos_path_token = getPath(ctx, false);
    auto shapes = panzer_db_ptr->getAOSShape(aos_path_token);
    *size = shapes.empty() ? 0 : shapes[0];
}

void GlobalReadStrategy::endAction(Context *ctx) {
    // The end of an OperationContext signifies the end of the entire read operation for this IDS.
    // We can close the PanzerDB instance to release file handles.
    if (ctx->getType() == CTX_OPERATION_TYPE) {
        if (panzer_db_ptr) {
            panzer_db_ptr->close();
        }
    }
}

int GlobalReadStrategy::read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                                     int* datatype, void **data, int *dim, int *size) {
    // The global strategy is to read the entire dataset, concatenating all time slices.
    // This is exactly what the refactored `read_dataset_globally` helper method does.
    return read_dataset_globally(ctx, dataset_name, datatype, data, dim, size);
}