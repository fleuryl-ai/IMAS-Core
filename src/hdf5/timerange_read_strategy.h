#ifndef TIMERANGE_READ_STRATEGY_H
#define TIMERANGE_READ_STRATEGY_H 1

#include "iread_strategy.h"


class TimeRangeReadStrategy : public IReadStrategy {
public:
    TimeRangeReadStrategy(hid_t loc_id);
    void beginReadArraystructAction(ArraystructContext * ctx, int *size) override;
    void endAction(Context * ctx) override;
    int read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                     int *datatype, void **data, int *dim, int *size) override;
private:
    std::vector<double> time_basis_vector;
    bool ends_with(const std::string &str, const std::string &suffix);
    std::vector<int> getIndices(Context *ctx, int *dynamic_index);
};

#endif // TIMERANGE_READ_STRATEGY_H