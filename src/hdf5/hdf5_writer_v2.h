#ifndef HDF5_WRITER_V2_H
#define HDF5_WRITER_V2_H 1

#include "al_backend.h"
#include <hdf5.h>
#include "hdf5_writer.h"


#include <memory>
#include <unordered_map>
#include <vector>
#include <map>

#include "panzerdb.h"



class HDF5Writer_v2 : public HDF5Writer {

private:
   std::unique_ptr<PanzerDB> panzer_db_ptr;
   ArraystructContext *getDynamicAOS(Context *ctx);
   std::unordered_set<ArraystructContext*> initialized_aos;

   const std::map<std::string, std::string> metadata_map;

public:
  HDF5Writer_v2(std::pair<int,int> backend_version_);
  ~HDF5Writer_v2();


  static bool compression_enabled;
  static size_t read_chunk_cache_size;
  static size_t write_chunk_cache_size;

  void setWriteStrategy(int write_mode, hid_t loc_id) override;

  void write_ND_Data(Context *ctx, const std::string &att_name,
                     const std::string &timebasename, int datatype, int dim,
                     int *size, void *data) override;

  void beginWriteArraystructAction(ArraystructContext *ctx, int *size) override;

  void read_homogeneous_time(int *homogenenous_time, hid_t gid);

  void endAction(Context *ctx) override;
};

#endif
