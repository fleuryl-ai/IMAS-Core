#ifndef TIMERANGE_READ_STRATEGY_H
#define TIMERANGE_READ_STRATEGY_H 1

#include "iread_strategy.h"


/**
 * @brief READ strategy for TIMERANGE_OP: reads a node resampled over a requested time range
 *        ([tmin, tmax] with the requested time grid), interpolating as needed.
 * @note Inherited pure-virtual semantics are described in IReadStrategy.
 */
class TimeRangeReadStrategy : public IReadStrategy {
public:
    /**
     * @brief Adopts the session's shared read-only engine and shared index.
     * @param panzer_db  The shared PanzerDB opened in READ mode for the session.
     * @param read_index The shared ReadIndex built once over that engine.
     */
    TimeRangeReadStrategy(std::shared_ptr<PanzerDB> panzer_db, std::shared_ptr<ReadIndex> read_index);
    void beginReadArraystructAction(ArraystructContext * ctx, int *size) override;
    void endAction(Context * ctx) override;
    /**
     * @brief Reads and resamples the node over the operation's requested time range.
     */
    int read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                     int *datatype, void **data, int *dim, int *size) override;
private:
    std::vector<double> time_basis_vector;   // The source time basis used for resampling.
    bool ends_with(const std::string &str, const std::string &suffix);
    /**
     * @brief Maps the requested time range to the concrete time indices to read.
     * @param ctx           Current context.
     * @param dynamic_index [out] The dynamic index resolved for the operation.
     * @return The indices to read.
     */
    std::vector<int> getIndices(Context *ctx, int *dynamic_index);
};

#endif // TIMERANGE_READ_STRATEGY_H