#ifndef SLICE_READ_STRATEGY_H
#define SLICE_READ_STRATEGY_H 1

#include "iread_strategy.h"

/**
 * @brief READ strategy for SLICE_OP: reads a single requested time step (slice) of a
 *        time-dependent node.
 * @note Inherited pure-virtual semantics are described in IReadStrategy.
 */
class SliceReadStrategy : public IReadStrategy {
public:
    /**
     * @brief Opens the PanzerDB handle in READ mode at the given location.
     * @param loc_id HDF5 location (group or file) holding the PanzerDB root.
     */
    SliceReadStrategy(hid_t loc_id);
    void beginReadArraystructAction(ArraystructContext * ctx, int *size) override;
    void endAction(Context * ctx) override;
    /**
     * @brief Reads the single slice of `dataset_name` matching the operation's requested time.
     */
    int read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                     int *datatype, void **data, int *dim, int *size) override;
private:
    /**
     * @brief Builds the list of time indices to select for a slice read.
     * @param ctx           Current context.
     * @param dynamic_index The requested dynamic index (-1 to determine it from the time).
     * @return The indices to read.
     */
    std::vector<int> getIndices(Context *ctx, int dynamic_index);
};

#endif // SLICE_READ_STRATEGY_H