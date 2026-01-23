#ifndef GLOBAL_READ_STRATEGY_H
#define GLOBAL_READ_STRATEGY_H 1

#include "iread_strategy.h"

class GlobalReadStrategy : public IReadStrategy {
public:
    GlobalReadStrategy(hid_t loc_id);
    void beginReadArraystructAction(ArraystructContext * ctx, int *size) override;
    void endAction(Context * ctx) override;
    int read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                     int *datatype, void **data, int *dim, int *size) override;
};

#endif // GLOBAL_READ_STRATEGY_H