/**
 * @file hdf5_writer.h
 * @brief v1 writer: one extensible HDF5 dataset per IMAS node, addressed by AoS index.
 *
 * The v1 layout writes each node into "<AoS path>&<node path>" whose leading
 * dimensions are the open AoS levels; two kinds of companion dataset record what
 * the file cannot otherwise tell a reader:
 *  - "<field path>_SHAPE": the array extents of one field for one AoS instance
 *    (needed because a dynamic signal stores its time steps along its first axis);
 *  - "<AoS path>&AOS_SHAPE": how many slices each dynamic AoS level holds at one
 *    instance, which is what tells a reader where to append next.
 * HDF5DataSetHandler performs the actual I/O; this class keeps the per-context
 * bookkeeping (tensorized paths, AoS shape chains, slice offsets) that lets a
 * low-level put_slice find its place in the file.
 *
 * @see HDF5Writer_v2 for the PanzerDB (v2) implementation of the same interface.
 */
#ifndef HDF5_WRITER_H
#define HDF5_WRITER_H 1

#include <hdf5.h>
#include "al_backend.h"
#include "hdf5_dataset_handler.h"

#include <memory>
#include <vector>
#include <unordered_map>

/**
 * @class HDF5Writer
 * @brief Writes the v1 layout: datasets, shape companions, AoS extents and pulse files.
 *
 * Virtual so HDF5Writer_v2 can override the operation-level entry points while the
 * file/group helpers stay shared.
 */
class HDF5Writer {
  private:

    std::pair<int,int> backend_version;   // (major, minor) recorded in every IDS file this writer creates.

    // Handlers kept open between low-level calls of one operation, keyed by tensorized path.
    // A handler is moved out of this map for the duration of one write and put back at the end,
    // so an entry here is one that was opened but not yet reused.
    std::unordered_map < std::string, std::unique_ptr < HDF5DataSetHandler > > opened_data_sets;
    std::unordered_map < std::string, hid_t > existing_data_sets;   // Existence memo: 1/0 per tensorized path, avoiding repeated H5Lexists calls.
    std::unordered_map < ArraystructContext *, std::vector < std::string >> tensorized_paths_per_context;   // Cumulative "&"-joined path per open AoS level.
    std::unordered_map < ArraystructContext *,  std::vector<int>> arrctx_shapes_per_context;                // Declared size of each open AoS level (outermost first).
    std::unordered_map < ArraystructContext *, int> dynamic_AOS_slices_extension;   // Slices a put_slice appends at a timed AoS level.

    int slice_mode;   // GLOBAL_OP (rewrite the node) or SLICE_OP (append time steps) for the operation in progress.

    /**
     * @brief Writes the "<field>_SHAPE" companion of a field, creating it when needed.
     * @param ctx                  Context of the field write.
     * @param loc_id               Group the dataset lives in.
     * @param field_tensorized_path Tensorized path of the field (the companion adds "_SHAPE").
     * @param fieldHandler         Handler of the field just written; supplies the extents to record.
     * @param timebasename         Time base path of the field (flattened), kept for the companion's context.
     * @param timed_AOS_index      Dynamic AoS level index, or -1.
     * @param arrctx_indices       AoS indices of the instance being recorded.
     * @param arrctx_shapes        Declared extent of each AoS level.
     * @return The companion dataset handle, or -1 when there is nothing to record (a node
     *         with no AoS levels, or with no array dimensions of its own).
     * @note Under SLICE_OP the first recorded extent includes the offset just appended, so a
     *       reader sees the full grown length rather than the chunk added by this call.
     */
    hid_t createOrUpdateShapesDataSet(Context * ctx, hid_t loc_id, const std::string & field_tensorized_path, HDF5DataSetHandler & fieldHandler,
				      const std::string & timebasename, int timed_AOS_index, const std::vector < int > &arrctx_indices, const std::vector < int > &arrctx_shapes);
    /**
     * @brief Writes the "<AoS path>&AOS_SHAPE" companion that records a dynamic AoS extent.
     * @param ctx           The AoS context being opened.
     * @param loc_id        Group the dataset lives in.
     * @param timedAOS_shape Slices already stored at that level (0 when untimed).
     * @param arrctx_indices AoS indices of the enclosing levels (the level's own index is dropped).
     * @param arrctx_shapes  Declared extent of each enclosing AoS level.
     * @note Creating this dataset under SLICE_OP reads the existing extent back first: an
     *       append must not invent the slice count of a level whose dataset is new.
     */
    void createOrUpdateAOSShapesDataSet(ArraystructContext * ctx, hid_t loc_id, int timedAOS_shape, const std::vector < int > &arrctx_indices, const std::vector < int > &arrctx_shapes);
    /**
     * @brief Reads back how many slices a dynamic AoS level holds at one instance.
     * @param ctx                    Context inside that AoS.
     * @param loc_id                 Group to look in.
     * @param current_arrctx_indices AoS indices selecting the instance.
     * @return The stored slice count; 0 when the level has no &AOS_SHAPE dataset (the
     *         absence is memoised in existing_data_sets).
     */
    int readTimedAOSShape(Context * ctx, hid_t loc_id, const std::vector < int > &current_arrctx_indices);
    /**
     * @brief Reads the first value of an &AOS_SHAPE dataset.
     * @param loc_id                 Group to look in.
     * @param tensorized_path        Dataset name.
     * @param current_arrctx_indices AoS indices selecting the instance.
     * @param uri                    Backend URI for the temporary dataset handler.
     * @return The stored slice count.
     * @throws ALBackendException when the read or the close fails.
     */
    int readTimedAOSShape(hid_t loc_id, std::string &tensorized_path, const std::vector < int > &current_arrctx_indices, uri::Uri uri);
    /**
     * @brief Slices a put_slice appends at a timed AoS level (1 when the level is unknown).
     * @param ctx Context inside the timed AoS.
     */
    int getDynamic_AOS_slices_extension(Context *ctx);
    /**
     * @brief Steps a write appends along whichever axis carries the time.
     * @param ctx                Context of the write.
     * @param timed_AOS_index    Dynamic AoS level index, or -1 when the signal carries time itself.
     * @param time_vector_length Requested time extent (used when timed_AOS_index is -1).
     */
    int getDynamic_slices_extension(Context *ctx, int timed_AOS_index, int time_vector_length);
    /**
     * @brief Nearest timed (time-evolving) AoS context at or above `ctx`, nullptr when none.
     * @param ctx Context to start from.
     */
    ArraystructContext* getDynamicAOS(Context * ctx);

  public:

    /**
     * @brief Binds the writer to one backend format version.
     * @param backend_version_ Version pair recorded in created files (see HDF5BackendFactory).
     */
     HDF5Writer(std::pair<int,int> backend_version_);
    /**
     * @brief Virtual destructor; dataset and file handles are released by close_datasets()/closePulse().
     */
    virtual ~HDF5Writer();

     /// Cached value of ids_properties&homogeneous_time seen during this run (1 homogeneous, 2 inhomogeneous, -1 unknown).
     int homogeneous_time;

    std::unordered_map < std::string,  hid_t> IDS_group_id;   // Open IDS group, keyed by IDS name.
    // Process-wide defaults exposed for callers that inspect the writer's configuration.
    // The effective values for a given dataset are re-read from the URI by each
    // HDF5DataSetHandler (HDF5Utils::setDefaultOptions then readOptions), so overriding
    // these statics alone does not change how a dataset is created.
    static bool compression_enabled;              // Shuffle+deflate on created datasets.
    static bool useBuffering;                     // Deferred merge-on-flush path for small writes.
    static size_t read_chunk_cache_size;          // HDF5 raw-data cache for reads, in bytes.
    static size_t write_chunk_cache_size;         // HDF5 raw-data cache for writes, in bytes.

    /**
     * @brief Ends a write pulse: closes datasets, IDS files and the master file.
     * @param ctx                 The data-entry context.
     * @param mode                Pulse close mode (unused; the caches decide what to close).
     * @param file_id             [in/out] Master file handle (reset to -1).
     * @param opened_IDS_files    [in/out] Cache of IDS file handles.
     * @param files_path_strategy Path layout selector (unused here).
     * @param files_directory     Pulse directory (unused here).
     * @param relative_file_path  Master file name (unused here).
     */
    virtual void closePulse(DataEntryContext * ctx, int mode, hid_t *file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, int files_path_strategy, std::string & files_directory, std::string & relative_file_path);
    /**
     * @brief Deletes one IDS: its file, its link in the master file, and its open handles.
     * @param ctx              The operation context naming the IDS.
     * @param file_id          Master file handle (must already be open).
     * @param opened_IDS_files [in/out] Cache of IDS file handles.
     * @param files_directory  Pulse directory.
     * @param relative_file_path Master file name inside it.
     * @note A no-op when the IDS group is not open in this session — there is nothing of
     *       this IDS in the master file to unlink.
     * @throws ALBackendException when the master file is not open.
     */
    virtual void deleteData(OperationContext * ctx, hid_t file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, std::string & files_directory, std::string & relative_file_path);
    /**
     * @brief Writes one node's values for one AoS instance (the AL put / put_slice entry point).
     * @param ctx          Context of the write (an AoS context, or the operation context at top level).
     * @param att_name     IMAS path of the node; '/' is flattened to '&' for the dataset name.
     * @param timebasename Time base path from the dictionary (flattened; used for companions).
     * @param datatype     AL datatype of the values.
     * @param dim          Dimensionality of the request (0 for a scalar).
     * @param size         Shape of the request (copied locally before it is used).
     * @param data         The values. Strings are copied here into NUL-terminated C strings,
     *                     and the copies are freed before returning.
     * @note GLOBAL_OP creates the dataset on first write and reuses/extends it afterwards;
     *       SLICE_OP opens (or creates) it, snapshots the dims at the start of the slice,
     *       grows the extent and records where this slice starts.
     * @note The value write goes through the handler's buffer when buffering applies, and
     *       straight through a hyperslab otherwise. The "_SHAPE" companion is written after
     *       the values, for any node that has array dimensions.
     * @throws ALBackendException when the IDS group is not open or the underlying write fails.
     */
    virtual void write_ND_Data(Context * ctx, const std::string & att_name, const std::string & timebasename, int datatype, int dim, int *size, void *data);
    /**
     * @brief Opens one AoS level for writing: builds its path/shape chain and updates its extent dataset.
     * @param ctx  The AoS context being opened.
     * @param size [in/out] Declared size of this level; under SLICE_OP on a timed level it is
     *             replaced by the extent read back from the file, and the appended count is
     *             recorded for the writes that follow.
     * @throws ALBackendException when the IDS group is not open.
     */
    virtual void beginWriteArraystructAction(ArraystructContext * ctx, int *size);
    /**
     * @brief Closes one AoS level: drops its path, shape and slice-extension bookkeeping.
     * @param ctx The context being closed (a no-op for non-AoS contexts).
     */
    virtual void endAction(Context * ctx);

    /**
     * @brief Selects the write strategy for the operation (stores write_mode as slice_mode).
     * @param ctx        The operation context.
     * @param write_mode GLOBAL_OP or SLICE_OP.
     * @param loc_id     Group the strategy applies to (unused here).
     */
    virtual void setWriteStrategy(OperationContext * ctx, int write_mode, hid_t loc_id);
	  /**
	   * @brief Sets the slice mode directly (used by the event handler before the operation starts).
	   * @param slice_mode GLOBAL_OP or SLICE_OP.
	   */
	  void setSliceMode(int slice_mode);
    /**
     * @brief Flushes every handler that accumulated buffered writes.
     */
    void write_buffers();
    /**
     * @brief Reads ids_properties&homogeneous_time out of an open IDS group.
     * @param homogenenous_time [out] 1 homogeneous, 2 inhomogeneous, -1 when absent or unreadable.
     * @param gid               IDS group handle (-1 yields -1 immediately).
     * @note Used to decide whether a SLICE_OP may append: an IDS whose flag cannot be read
     *       has to be rewritten as a whole.
     */
    void read_homogeneous_time(int* homogenenous_time, hid_t gid);
    /**
     * @brief Closes one IDS file from the cache and resets its entry to -1.
     * @param external_link_name IDS name ('/' accepted; normalised to '_').
     * @param opened_IDS_files   [in/out] Cache of IDS file handles.
     */
    void close_file_handler(std::string external_link_name, std::unordered_map < std::string, hid_t > &opened_IDS_files);
    /**
     * @brief Creates (or opens) the IDS file and (re)creates the group the master file points at.
     * @param ctx              The operation context naming the IDS.
     * @param file_id          Master file handle to attach the external link to.
     * @param opened_IDS_files [in/out] Cache of IDS file handles.
     * @param files_directory  Pulse directory.
     * @param relative_file_path Master file name inside it.
     * @param access_mode      Requested access mode (the file is opened read/write regardless).
     * @param loc_id           [out] Group handle to write the IDS through.
     * @note The external link is created only when absent, so re-creating an IDS group does
     *       not duplicate it. Any previous group handle is closed first.
     * @throws ALBackendException when the IDS file or the group cannot be created.
     */
    void create_IDS_group(OperationContext *ctx, hid_t file_id,
                   std::unordered_map<std::string, hid_t> &opened_IDS_files,
                   std::string &files_directory,
                   std::string &relative_file_path, int access_mode, hid_t *loc_id);
    /**
     * @brief Opens the existing IDS file and group for writing.
     * @param ctx              The operation context naming the IDS.
     * @param file_id          Master file handle.
     * @param opened_IDS_files [in/out] Cache of IDS file handles.
     * @param files_directory  Pulse directory.
     * @param relative_file_path Master file name inside it.
     * @param loc_id           [out] Group handle (also cached by IDS name).
     */
    void open_IDS_group(OperationContext * ctx, hid_t file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, std::string & files_directory, std::string & relative_file_path, hid_t *loc_id);
    /**
     * @brief Closes every open dataset handler and clears the handler and existence caches.
     */
    void close_datasets();
    /**
     * @brief Closes the IDS group of an operation and removes it from the cache.
     * @param ctx The operation context whose group is closed (a no-op when none is open).
     * @throws ALBackendException when H5Gclose fails.
     */
    void close_group(OperationContext *ctx);
    
};

#endif
