#ifndef HDF5_WRITER_V2_H
#define HDF5_WRITER_V2_H 1

#include "al_backend.h"
#include <hdf5.h>
#include "hdf5_writer.h"

#include <memory>
#include <unordered_map>
#include <vector>
#include <map>
#include <string>

#include "panzerdb.h"
#include "metadata/metadata_extractor.h"

/**
 * @brief Concrete HDF5 write backend (v2) built on top of the PanzerDB engine.
 *
 * Implements the AL low-level writer interface (HDF5Writer) by delegating every
 * operation to a PanzerDB instance:
 *  - GLOBAL_OP  -> a fresh PanzerDB opened in WRITE mode;
 *  - SLICE_OP   -> a PanzerDB opened in APPEND mode.
 *
 * The write path is: beginWriteArraystructAction -> write_ND_Data (repeatedly)
 * -> endAction. Data is buffered by PanzerDB and physically flushed in endAction
 * (OperationContext) or at destruction.
 */
class HDF5Writer_v2 : public HDF5Writer {

private:
   std::unique_ptr<PanzerDB> panzer_db_ptr;     // The storage engine for the current write session.
   ArraystructContext *getDynamicAOS(Context *ctx);
   std::unordered_set<ArraystructContext*> initialized_aos;  // AoS contexts opened via beginWriteArraystructAction (closed in endAction).
   std::map<std::string, std::string> metadata_map;          // Schema metadata extracted from IDSDef.xml, replayed as @key attributes.

public:
   HDF5Writer_v2(std::pair<int,int> backend_version_);
   ~HDF5Writer_v2();

   static bool compression_enabled;         // Global flag: apply GZIP compression to new datasets.
   static size_t read_chunk_cache_size;     // Raw-data (chunk) cache size used when reading.
   static size_t write_chunk_cache_size;    // Raw-data (chunk) cache size used when writing.

   /**
    * @brief Creates the PanzerDB backend for the current write operation.
    * @param ctx        The OperationContext (provides the dataobject name).
    * @param write_mode TARGET mode: GLOBAL_OP (WRITE) or SLICE_OP (APPEND).
    * @param loc_id     Location (file or group) where the PanzerDB root lives.
    * @note In GLOBAL_OP it also extracts the schema metadata map from the
    *       IDS definition (for replaying IMAS @key attributes).
    */
   void setWriteStrategy(OperationContext * ctx, int write_mode, hid_t loc_id) override;

   /**
    * @brief Writes one data node (or one metadata @key) to the PanzerDB engine.
    * @param ctx          The current context (data entry or array struct).
    * @param att_name     Data node path ('/' replaced by '&' internally).
    * @param timebasename Time base path, if the node is time dependent (may be empty).
    * @param datatype     AL data type (double/integer/char/complex data).
    * @param dim          Number of dimensions.
    * @param size         Dimension sizes.
    * @param data         Pointer to the element buffer.
    */
   void write_ND_Data(Context *ctx, const std::string &att_name,
                      const std::string &timebasename, int datatype, int dim,
                      int *size, void *data) override;

   /**
    * @brief Opens an Array-of-Structures level (static or dynamic) in PanzerDB.
    * @param ctx  The array-struct context to open.
    * @param size Number of elements of the AoS.
    */
   void beginWriteArraystructAction(ArraystructContext *ctx, int *size) override;

   /**
    * @brief Reads the 'ids_properties&homogeneous_time' scalar from an IDS.
    * @param homogenenous_time [out] -1 if the key is absent, else its value.
    * @param gid               The IDS group id (-1 means not set, returns -1).
    */
   void read_homogeneous_time(int *homogenenous_time, hid_t gid);

   /**
    * @brief Closes the current context: endArray() for AoS, flush()+close() for the operation.
    * @param ctx The context being closed.
    */
   void endAction(Context *ctx) override;
};

#endif