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
     * @brief Opens the PanzerDB handle in READ mode at the given location.
     * @param loc_id HDF5 location (group or file) holding the PanzerDB root.
     */
    TimeRangeReadStrategy(hid_t loc_id);
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