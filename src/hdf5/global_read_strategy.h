#ifndef GLOBAL_READ_STRATEGY_H
#define GLOBAL_READ_STRATEGY_H 1

#include "iread_strategy.h"

/**
 * @brief READ strategy for GLOBAL_OP: reads nodes with their full extent (all time steps).
 * @note Inherited pure-virtual semantics (beginReadArraystructAction / endAction /
 *       read_ND_Data) are described in IReadStrategy.
 */
class GlobalReadStrategy : public IReadStrategy {
public:
    /**
     * @brief Opens the PanzerDB handle in READ mode at the given location.
     * @param loc_id HDF5 location (group or file) holding the PanzerDB root.
     */
    GlobalReadStrategy(hid_t loc_id);
    void beginReadArraystructAction(ArraystructContext * ctx, int *size) override;
    void endAction(Context * ctx) override;
    /**
     * @brief Reads the whole node (every time slice) and returns it as a single array.
     */
    int read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                     int *datatype, void **data, int *dim, int *size) override;
};

#endif // GLOBAL_READ_STRATEGY_H