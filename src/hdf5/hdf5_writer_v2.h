#ifndef HDF5_WRITER_V2_H
#define HDF5_WRITER_V2_H 1

#include "al_backend.h"
#include <hdf5.h>
#include "hdf5_writer.h"


#include <memory>
#include <unordered_map>
#include <vector>
#include "panzerdb.h"



class HDF5Writer_v2 : public HDF5Writer {
public:
  std::string backend_version;

private:
   std::unique_ptr<PanzerDB> panzer_db_ptr;
   ArraystructContext *getDynamicAOS(Context *ctx);
   std::unordered_set<ArraystructContext*> initialized_aos;

public:
  /**
   * @brief Constructor.
   * @param backend_version_ The version of the backend.
   */
  HDF5Writer_v2(std::string backend_version_);

  /**
   * @brief Destructor.
   */
  ~HDF5Writer_v2();


  static bool compression_enabled;
  static size_t read_chunk_cache_size;
  static size_t write_chunk_cache_size;

  /**
   * @brief Sets the write strategy (Global or Slice) and initializes PanzerDB.
   * @param write_mode The write mode (GLOBAL_OP or SLICE_OP).
   * @param loc_id The HDF5 location ID (group or file).
   */
  void setWriteStrategy(int write_mode, hid_t loc_id) override;

  /**
   * @brief Writes N-Dimensional data to the IDS.
   * @param ctx The context (Operation or Arraystruct).
   * @param att_name The name of the attribute/dataset.
   * @param timebasename The name of the timebase (if any).
   * @param datatype The data type (integer, double, etc.).
   * @param dim The number of dimensions.
   * @param size Array containing the size of each dimension.
   * @param data Pointer to the data buffer.
   */
  void write_ND_Data(Context *ctx, const std::string &att_name,
                     const std::string &timebasename, int datatype, int dim,
                     int *size, void *data) override;

  /**
   * @brief Starts writing an Array of Structures (AoS).
   * Initializes the AoS in PanzerDB.
   * @param ctx The Arraystruct context.
   * @param size Pointer to the size of the AoS.
   */
  void beginWriteArraystructAction(ArraystructContext *ctx, int *size) override;

  /**
   * @brief Reads the homogeneous time property from the IDS.
   * @param homogenenous_time Pointer to store the result (1 if homogeneous, 0 otherwise).
   * @param gid The HDF5 group ID.
   */
  void read_homogeneous_time(int *homogenenous_time, hid_t gid);

  /**
   * @brief Finalizes the action on the current context.
   * Flushes data or closes groups/files as needed.
   * @param ctx The context to end.
   */
  void endAction(Context *ctx) override;
};

#endif
