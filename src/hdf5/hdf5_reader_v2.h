#ifndef HDF5_READER_V2_H
#define HDF5_READER_V2_H 1

#include "hdf5_reader.h" // Ajout de l'inclusion de la classe de base
#include <hdf5.h>
#include "iread_strategy.h" // Inclure l'interface de la stratégie
#include "global_read_strategy.h" // Inclure la stratégie par défaut
#include "slice_read_strategy.h" 
#include "timerange_read_strategy.h" 

#include "al_backend.h"
#include "data_interpolation.h"

#include <memory>
#include <vector>
#include <list>
#include <unordered_map>


class HDF5Reader_v2 : public HDF5Reader {

  private:

    void read_homogeneous_time(int* homogenenous_time, hid_t gid);

    void build_path_index();
    
    std::unique_ptr<IReadStrategy> read_strategy; // Stratégie de lecture

  public:

     HDF5Reader_v2(std::pair<int,int> backend_version_);
    ~HDF5Reader_v2() override;

    void open_IDS_group(OperationContext * ctx, hid_t file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, std::string & files_directory, std::string & relative_file_path) override;
    void close_group(OperationContext *ctx) override; 
    void closePulse(DataEntryContext * ctx, int mode, hid_t *file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, int files_path_strategy, std::string & files_directory, std::string & relative_file_path) override;
    int read_ND_Data(Context * ctx, std::string & att_name, std::string & timebasename, int *datatype, void **data, int *dim, int *size) override;
    void beginReadArraystructAction(ArraystructContext * ctx, int *size) override; 

    void endAction(Context * ctx) override; 

    std::string getVersion() override;

};

#endif
