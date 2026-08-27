#ifndef HDF5_READER_V2_H
#define HDF5_READER_V2_H 1

#include "hdf5_reader.h"   // base class
#include <hdf5.h>
#include "iread_strategy.h"          // read strategy interface
#include "global_read_strategy.h"    // GLOBAL_OP strategy
#include "slice_read_strategy.h"     // SLICE_OP strategy
#include "timerange_read_strategy.h" // TIMERANGE_OP strategy

#include "al_backend.h"
#include "data_interpolation.h"

#include <memory>
#include <vector>
#include <list>
#include <unordered_map>


/**
 * @brief Concrete HDF5 read backend (v2) for the PanzerDB layout.
 *
 * Implements the AL low-level reader interface (HDF5Reader) by delegating to one
 * of three strategies, selected by the operation's range mode:
 *  - GLOBAL_OP   -> GlobalReadStrategy
 *  - SLICE_OP    -> SliceReadStrategy
 *  - TIMERANGE_OP-> TimeRangeReadStrategy
 *
 * The active strategy is chosen lazily (and cached) by prepare_strategy()
 * before every read entry point.
 */
class HDF5Reader_v2 : public HDF5Reader {

  private:

    void read_homogeneous_time(int* homogenenous_time, hid_t gid);   // Read ids_properties&homogeneous_time, if present.
    void build_path_index();                                          // Build the path -> leaves index used by the strategies.
    void select_strategy(OperationContext *ctx, hid_t gid);          // Pick + activate the strategy for the range mode.
    void prepare_strategy(Context *ctx);                              // Resolve gid + strategy for the given context.

    std::unique_ptr<IReadStrategy> global_strategy;   // Lazily created, shared across the session.
    std::unique_ptr<IReadStrategy> slice_strategy;
    std::unique_ptr<IReadStrategy> timerange_strategy;

    IReadStrategy* read_strategy = nullptr;   // Pointer to the currently active strategy.

  public:

     HDF5Reader_v2(std::pair<int,int> backend_version_);
    ~HDF5Reader_v2() override;

    /**
     * @brief Opens an IDS group and resets any cached read strategy for the new session.
     * @param ctx The OperationContext for the read.
     * @see open group handling of the base HDF5Reader class.
     */
    void open_IDS_group(OperationContext * ctx, hid_t file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, std::string & files_directory, std::string & relative_file_path) override;

    /**
     * @brief Closes the group for the given operation context (delegates to the base class).
     */
    void close_group(OperationContext *ctx) override;

    /**
     * @brief Closes a pulse read (delegates to the base class).
     */
    void closePulse(DataEntryContext * ctx, int mode, hid_t *file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, int files_path_strategy, std::string & files_directory, std::string & relative_file_path) override;

    /**
     * @brief Reads one node, delegating to the active read strategy.
     * @param ctx          The current context.
     * @param att_name     Node path (normalized internally: '/' -> '&').
     * @param timebasename Time base path, if any.
     * @param datatype     [in/out] AL data type.
     * @param data         [out] Allocated output buffer (caller frees).
     * @param dim          [out] Number of output dimensions.
     * @param size         [out] Output shape.
     * @return 0 on success, non-zero on failure.
     */
    int read_ND_Data(Context * ctx, std::string & att_name, std::string & timebasename, int *datatype, void **data, int *dim, int *size) override;

    /**
     * @brief Opens an AoS read; the delegated strategy fills `size` with the effective AoS size.
     */
    void beginReadArraystructAction(ArraystructContext * ctx, int *size) override;

    /**
     * @brief Closes the current read (delegates endAction of the active strategy).
     */
    void endAction(Context * ctx) override;

    /**
     * @brief Returns the backend version identifier.
     */
    std::string getVersion() override;

};

#endif
