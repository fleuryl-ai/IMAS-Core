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
     * @brief Adopts the session's shared read-only engine and shared index.
     * @param panzer_db  The shared PanzerDB opened in READ mode for the session.
     * @param read_index The shared ReadIndex built once over that engine.
     */
    SliceReadStrategy(std::shared_ptr<PanzerDB> panzer_db, std::shared_ptr<ReadIndex> read_index);
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