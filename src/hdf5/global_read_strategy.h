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
     * @brief Adopts the session's shared read-only engine and shared index.
     * @param panzer_db  The shared PanzerDB opened in READ mode for the session.
     * @param read_index The shared ReadIndex built once over that engine.
     */
    GlobalReadStrategy(std::shared_ptr<PanzerDB> panzer_db, std::shared_ptr<ReadIndex> read_index);
    void beginReadArraystructAction(ArraystructContext * ctx, int *size) override;
    void endAction(Context * ctx) override;
    /**
     * @brief Reads the whole node (every time slice) and returns it as a single array.
     */
    int read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                     int *datatype, void **data, int *dim, int *size) override;
};

#endif // GLOBAL_READ_STRATEGY_H