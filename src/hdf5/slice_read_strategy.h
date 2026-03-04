#ifndef SLICE_READ_STRATEGY_H
#define SLICE_READ_STRATEGY_H 1

#include "iread_strategy.h"

class SliceReadStrategy : public IReadStrategy {
public:
    SliceReadStrategy(hid_t loc_id);
    void beginReadArraystructAction(ArraystructContext * ctx, int *size) override;
    void endAction(Context * ctx) override;
    int read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                     int *datatype, void **data, int *dim, int *size) override;
private:
    std::vector<int> getIndices(Context *ctx, int dynamic_index);
};

#endif // SLICE_READ_STRATEGY_H