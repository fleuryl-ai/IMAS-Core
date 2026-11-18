#ifndef HDF5_EVENTS_HANDLER_H
#define HDF5_EVENTS_HANDLER_H 1

#include <hdf5.h>
#include "al_backend.h"
#include "hdf5_reader.h"
#include "hdf5_writer.h"

#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#include <string>

/**
 * @file hdf5_events_handler.h
 * @brief Sequences the reader/writer calls implied by the AL low-level
 *        beginAction / endAction events, so the backend stays a thin dispatcher.
 *
 * The AL opens an IDS group, picks a storage strategy and closes everything again
 * around every operation. Which calls run in which order depends on the
 * operation's access mode (READ/WRITE) and range mode (GLOBAL/SLICE); that policy
 * lives here rather than in HDF5Backend. It is version-independent: the same
 * handler drives the v1 and the v2 reader/writer, which is why it only ever sees
 * their base classes.
 */

/**
 * @class HDF5EventsHandler
 * @brief Turns AL action events into the matching reader/writer call sequences.
 *
 * Stateless apart from `strategy_set`, which records whether a write strategy has
 * already been installed for the action in progress: an OperationContext action
 * opens the group once, and the AoS actions nested inside it must not reinstall it.
 */
class HDF5EventsHandler {
  private:



  public:

    HDF5EventsHandler();
    ~HDF5EventsHandler();

  /**
   * @brief Handles a low-level beginAction event: opens the IDS group and installs
   *        the storage strategy matching the operation's access + range mode.
   * @param ctx                The operation context that opened the action.
   * @param file_id            Master pulse file handle.
   * @param opened_IDS_files   [in/out] Cache of IDS files reached through external links.
   * @param writer             Writer instance for this backend version.
   * @param reader             Reader instance for this backend version.
   * @param files_directory    Root directory of the pulse layout.
   * @param relative_file_path Pulse path relative to that root.
   * @param access_mode        Backend access mode to open the IDS file with.
   * @note WRITE + GLOBAL: creates the IDS group and sets the GLOBAL (whole-IDS) strategy.
   * @note WRITE + SLICE: opens the existing group and sets the SLICE (append) strategy.
   *       Falls back to creating the group with the GLOBAL strategy when the IDS file
   *       does not exist, the group cannot be opened, or the group carries no
   *       ids_properties&homogeneous_time flag — an append target must already be a
   *       valid IDS, otherwise this run has to write the whole IDS.
   * @note READ: opens the IDS group for reading and sets the reader's slice mode.
   */
    virtual void beginAction(OperationContext * ctx, hid_t file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, HDF5Writer & writer, HDF5Reader & reader, std::string & files_directory, std::string & relative_file_path, int access_mode);

  /**
   * @brief Handles a low-level endAction event: closes whatever level just ended.
   * @param ctx                The context being closed (AoS or operation).
   * @param file_id            Master pulse file handle (flushed on a write).
   * @param writer             Writer instance for this backend version.
   * @param reader             Reader instance for this backend version.
   * @param opened_IDS_files   [in/out] Cache of IDS files reached through external links.
   * @note An ArraystructContext closes one AoS level (endArray on the writer, the
   *       matching bookkeeping on the reader).
   * @note An OperationContext closes the whole operation. On a write: the buffers are
   *       flushed for GLOBAL_OP, the file is flushed with H5Fflush, then the datasets,
   *       the group and the file handle are released. On a read: the same resources
   *       are released without flushing. Either way `strategy_set` is reset so the
   *       next action starts from a clean state.
   */
    virtual void endAction(Context * ctx, hid_t file_id, HDF5Writer & writer, HDF5Reader & reader, std::unordered_map < std::string, hid_t > &opened_IDS_files);

    private:
    bool strategy_set;   // True once setWriteStrategy() ran for the current action (set in beginAction, cleared in endAction).

};

#endif
