/**
 * @file hdf5_reader.h
 * @brief v1 reader: reads one-dataset-per-node layouts, with shape companions,
 *        time-base resolution, slice selection and interpolation.
 *
 * The v1 layout keeps, besides the value dataset of each node:
 *  - "<field path>_SHAPE": the array extents of one field for one AoS instance — the
 *    reader needs it because a dynamic signal grows along its first stored axis, so
 *    the extent of one slice is not derivable from the dataset alone;
 *  - "<AoS path>&AOS_SHAPE": the slice count of a dynamic AoS level at one instance,
 *    which is what bounds a read and what maps a requested time to a slice index;
 *  - the time base itself: "time" at the IDS root for a homogeneous-time IDS, or
 *    "<AoS path>&time" for a dynamic AoS (one time per slice, and per instance when
 *    the IDS has inhomogeneous time).
 * Range modes are handled here: GLOBAL_OP returns the whole node, SLICE_OP resolves
 * the requested time to one slice (nearest, or linear interpolation between two), and
 * TIMERANGE_OP returns a window, optionally resampled onto a caller-provided grid.
 *
 * @see HDF5Reader_v2 for the PanzerDB (v2) implementation of the same interface.
 */
#ifndef HDF5_READER_H
#define HDF5_READER_H 1

#include <hdf5.h>
#include "al_backend.h"
#include "hdf5_hs_selection_reader.h"
#include "hdf5_dataset_handler.h"
#include "data_interpolation.h"

#include <memory>
#include <vector>
#include <list>
#include <unordered_map>


/**
 * @class HDF5Reader
 * @brief Reads the v1 layout: values, shapes, AoS extents, time bases and occurrences.
 *
 * Virtual so HDF5Reader_v2 can override the operation-level entry points while the
 * file/group helpers stay shared. Handlers stay open between low-level calls of one
 * operation (see exit_request) so that reading every AoS instance of a node costs one
 * open and, where buffering applies, one whole-dataset read.
 */
class HDF5Reader {
  private:

    typedef herr_t(* H5L_iterate1_t) (hid_t group, const char *name, const H5L_info_t *info, void *op_data);   // Cast type H5Literate expects for a callback.
    static herr_t iterate_callback (hid_t loc_id, const char *name, const H5L_info_t *info, void *callback_data);   // Collects one link name into the std::vector passed as callback_data.

    std::pair<int,int> backend_version;   // (major, minor) of the format this reader was created for.

    // Handlers kept open across the low-level calls of one operation.
    std::unordered_map < std::string, std::unique_ptr < HDF5DataSetHandler > > opened_data_sets;             // Value datasets and time bases.
    std::unordered_map < std::string, std::unique_ptr < HDF5DataSetHandler > > opened_shapes_data_sets;      // "<field>_SHAPE" companions.
    std::unordered_map < std::string, std::unique_ptr < HDF5DataSetHandler > > aos_opened_shapes_data_sets;  // "<AoS path>&AOS_SHAPE" companions.
    std::unordered_map < std::string, hid_t > existing_data_sets;   // Existence memo: 1/0 per tensorized path, so a missing node costs one H5Lexists for the whole operation.
    std::unordered_map < ArraystructContext *,  std::vector<std::string>> tensorized_paths_per_context;      // Cumulative "&"-joined path per open AoS level.

    std::unordered_map < OperationContext *,  std::vector<std::string>> tensorized_paths_per_op_context;     // Same chain, kept per operation to resolve a time base path.

    std::unordered_map < ArraystructContext *,  std::vector<int>> arrctx_shapes_per_context;                 // Declared size of each open AoS level (outermost first).

    int slice_mode;   // GLOBAL_OP or SLICE_OP for the operation in progress (TIMERANGE_OP reads as GLOBAL_OP here).

    DataInterpolation data_interpolation_component;   // Time-index lookup plus the interpolation/resampling arithmetic.

    /**
     * @brief Translates a TIMERANGE_OP request into the selector's time window.
     * @param opctx                 The operation context carrying the time range.
     * @param timebasename         Time base path of the node (flattened).
     * @param hsSelectionReader    Selector to configure (its time_range is filled).
     * @param current_arrctx_indices AoS indices of the instance being read.
     * @param gid                  IDS group handle.
     * @param is_dynamic           Whether the node carries the time axis itself.
     * @param isTimed              Whether an enclosing dynamic AoS supplies it.
     * @param timed_AOS_index      Dynamic AoS level index, or -1.
     * @param time_basis_vector    The time base already read for this request.
     * @param size                 [out] Number of slices inside the window (0 when the request
     *                             is not a time range or matches nothing).
     * @param dim                  Requested dimensionality (its last entry is the time axis).
     * @note With LINEAR_INTERP the upper index is extended by one so the bracketing slice the
     *       interpolation needs is included in the read.
     */
    void configureTimeRange(OperationContext *opctx, const std::string &timebasename,
      HDF5HsSelectionReader &hsSelectionReader, const std::vector <int> &current_arrctx_indices, hid_t gid,
      bool is_dynamic, bool isTimed, int timed_AOS_index, const std::vector<double> &time_basis_vector, int *size, int dim);

    /**
     * @brief Fills the time base for the node being read.
     * @param opCtx               The operation context (its time range drives resampling).
     * @param data_set            Handler of the time-base dataset.
     * @param timebasename        Unused here; kept for call-site symmetry.
     * @param timed_AOS_index     Dynamic AoS level index, or -1.
     * @param current_arrctx_indices AoS indices selecting the instance.
     * @param time_basis_vector   [in/out] The resulting time base (cached across calls when homogeneous).
     * @param resampling          Synthesise a uniform grid over the requested window instead of
     *                            reading the stored times.
     * @note Homogeneous IDS: the whole "time" dataset is read once and reused.
     * @note Inhomogeneous IDS: the per-instance bases are read along the dynamic axis and
     *       truncated at the first IMAS invalid value (-9.0E40), which marks where a shorter
     *       instance's base stops.
     * @throws ALBackendException when the basis cannot be read, or when resampling is asked
     *         with a vector that is not empty.
     */
    void getTimeVector(OperationContext * opCtx, std::unique_ptr < HDF5DataSetHandler > &data_set, const std::string & timebasename, int timed_AOS_index,
    const std::vector < int > &current_arrctx_indices, std::vector<double> &time_basis_vector, bool resampling = false);
    /**
     * @brief Checks that a linear interpolation between two slices is well defined.
     * @param ctx             Context of the read.
     * @param gid             IDS group handle.
     * @param tensorized_path Node being interpolated (reported in warnings).
     * @param datatype        AL datatype (strings are never interpolated).
     * @param slice_mode      GLOBAL_OP or SLICE_OP.
     * @param is_dynamic      Whether the node carries the time axis itself.
     * @param slice_min       Lower slice index of the pair.
     * @param slice_sup       Upper slice index of the pair.
     * @param dim             Requested dimensionality.
     * @param timed_AOS_index Dynamic AoS level index, or -1.
     * @param dataset_id_shapes [out] Shape-companion handle.
     * @param current_arrctx_indices AoS indices of the instance.
     * @param dataset_id      Value dataset handle.
     * @param isOpenedShapesDataSet Whether the shape companion is already open.
     * @param hsSelectionReader Selector used for the probe reads.
     * @return 1 when the interpolation may proceed, 0 when either slice is missing or the two
     *         slices have different shapes (the read then returns nothing).
     * @note A 0D node only needs the lower slice to be readable; a shaped node needs both
     *       extents recorded and equal.
     */
    int checkSlicesShapes(Context *ctx, hid_t gid, const std::string &tensorized_path, int datatype, int slice_mode,
                                    bool is_dynamic, int slice_min, int slice_sup, int dim, int timed_AOS_index,
                                    hid_t *dataset_id_shapes, const std::vector<int> &current_arrctx_indices, hid_t dataset_id, bool isOpenedShapesDataSet, HDF5HsSelectionReader &hsSelectionReader);
    /**
     * @brief Reads a node's stored slice shape and turns it into the size the AL expects.
     * @param ctx             Context of the read.
     * @param gid             IDS group handle.
     * @param tensorized_path Node path (the companion adds "_SHAPE").
     * @param datatype        AL datatype.
     * @param slice_mode      GLOBAL_OP or SLICE_OP (selects the companion read variant).
     * @param is_dynamic      Whether the node carries the time axis itself.
     * @param isTimed         Whether an enclosing dynamic AoS supplies it.
     * @param slice_index     Slice whose shape is wanted (-1 for the whole node).
     * @param dim             Requested dimensionality.
     * @param size            [out] Extents in AL order (reversed from the stored order).
     * @param timed_AOS_index Dynamic AoS level index, or -1.
     * @param zero_shape      [out] True when the companion records a zero extent, i.e. the
     *                      instance was never written — the caller then returns nothing.
     * @param dataset_id_shapes [out] Shape-companion handle.
     * @param isOpenedShapesDataSet Whether the companion is already open.
     * @param current_arrctx_indices AoS indices of the instance.
     * @return 1 when a companion dataset exists, 0 when there is none (or the node has no AoS
     *         levels, or its shape needs no companion).
     */
    int getPersistentShapes(Context * ctx, hid_t gid, const std::string & tensorized_path, int datatype, int slice_mode, bool is_dynamic, bool isTimed,
			    int slice_index, int dim, int *size, int timed_AOS_index, bool * zero_shape, hid_t * dataset_id_shapes,
			    bool isOpenedShapesDataSet, const std::vector < int > &current_arrctx_indices);
    /**
     * @brief Reads the "_SHAPE" companion, dispatching on the range mode.
     * @param ctx             Context of the read.
     * @param gid             IDS group handle.
     * @param field_tensorized_path Node path (the companion adds "_SHAPE").
     * @param shapes          [out] Newly allocated extents (caller frees).
     * @param slice_mode      GLOBAL_OP or SLICE_OP.
     * @param is_dynamic      Whether the node carries the time axis itself.
     * @param isTimed         Whether an enclosing dynamic AoS supplies it.
     * @param slice_index     Slice whose shape is wanted.
     * @param timed_AOS_index Dynamic AoS level index, or -1.
     * @param zero_shape      [out] True when the recorded extent is zero.
     * @param dataset_id_shapes [out] Shape-companion handle.
     * @param isOpenedShapesDataSet Whether the companion is already open.
     * @param current_arrctx_indices AoS indices of the instance.
     * @return 1 when the companion exists, 0 otherwise.
     */
    int readPersistentShapes(Context * ctx, hid_t gid, const std::string & field_tensorized_path, void **shapes, int slice_mode, bool is_dynamic, bool isTimed,
			     int slice_index, int timed_AOS_index, bool *zero_shape, hid_t *dataset_id_shapes, bool isOpenedShapesDataSet,
			     const std::vector < int > &current_arrctx_indices
			    );
    /**
     * @brief Reads the "_SHAPE" companion for a whole node (GLOBAL_OP variant).
     * @param ctx             Context of the read.
     * @param gid             IDS group handle.
     * @param field_tensorized_path Node path (the companion adds "_SHAPE").
     * @param shapes          [out] Newly allocated extents (caller frees).
     * @param slice_mode      GLOBAL_OP.
     * @param is_dynamic      Whether the node carries the time axis itself.
     * @param isTimed         Whether an enclosing dynamic AoS supplies it.
     * @param timed_AOS_index Dynamic AoS level index, or -1.
     * @param slice_index     Slice index carried through the dataset read (-1 for the whole node).
     * @param zero_shape      [out] True when the recorded extent is zero.
     * @param dataset_id_shapes [out] Shape-companion handle.
     * @param isOpenedShapesDataSet Whether the companion is already open.
     * @param current_arrctx_indices AoS indices of the instance.
     * @return Always 1 once the companion has been opened and read.
     * @throws ALBackendException when the companion cannot be opened.
     */
    int readPersistentShapes_Get(Context * ctx, hid_t gid, const std::string & field_tensorized_path, void **shapes, int slice_mode, bool is_dynamic,
				 bool isTimed, int timed_AOS_index, int slice_index, bool *zero_shape, hid_t *dataset_id_shapes, bool isOpenedShapesDataSet,
				 const std::vector < int > &current_arrctx_indices
				);
    /**
     * @brief Reads the "_SHAPE" companion of one slice (SLICE_OP variant).
     * @param ctx             Context of the read.
     * @param gid             IDS group handle.
     * @param field_tensorized_path Node path (the companion adds "_SHAPE").
     * @param shapes          [out] Newly allocated extents (caller frees).
     * @param slice_mode      SLICE_OP.
     * @param is_dynamic      Whether the node carries the time axis itself.
     * @param isTimed         Whether an enclosing dynamic AoS supplies it.
     * @param slice_index     Slice whose extent is wanted (substituted for the dynamic AoS index).
     * @param timed_AOS_index Dynamic AoS level index, or -1.
     * @param zero_shape      [out] True when the recorded extent is zero.
     * @param dataset_id_shapes [out] Shape-companion handle.
     * @param isOpenedShapesDataSet Whether the companion is already open.
     * @param current_arrctx_indices AoS indices of the instance.
     * @return 1 when the shape was read, 0 when the underlying read failed.
     * @note For a dynamic signal read outside a dynamic AoS, the time extent is reported as
     *       1: exactly one slice is being returned.
     */
    int readPersistentShapes_GetSlice(Context * ctx, hid_t gid, const std::string & field_tensorized_path, void **shapes, int slice_mode, bool is_dynamic, bool isTimed,
				      int slice_index, int timed_AOS_index, bool * zero_shape, hid_t * dataset_id_shapes, bool isOpenedShapesDataSet,
				      const std::vector < int > &current_arrctx_indices
 				    );
    /**
     * @brief Reads how many slices a dynamic AoS level holds at one instance.
     * @param ctx             Context inside that AoS.
     * @param gid             IDS group handle.
     * @param tensorized_path The "&AOS_SHAPE" dataset name.
     * @param timed_AOS_index Dynamic AoS level index, or -1.
     * @param slice_index     Slice whose extent is wanted (used under SLICE_OP).
     * @param shapes          [out] Newly allocated count (caller frees).
     * @param current_arrctx_indices AoS indices of the enclosing levels (the level's own index is dropped).
     * @return 1 when the dataset exists and the request is inside its extent, else 0.
     */
    int readAOSPersistentShapes(Context * ctx, hid_t gid, const std::string & tensorized_path, int timed_AOS_index, int slice_index, void **shapes, const std::vector < int > &current_arrctx_indices);

    /**
     * @brief Resolves the dataset that holds the time base for a request.
     * @param homogeneous_time The IDS flag (1 homogeneous, 0/2 otherwise).
     * @param timebasePath     Time base path from the dictionary (flattened here).
     * @param timed_AOS_index  Dynamic AoS level index, or -1.
     * @param tensorized_paths Cumulative AoS paths of the operation.
     * @return Dataset name: "time" for a homogeneous IDS or an explicit time path, otherwise
     *         "<AoS path>&time".
     * @throws ALBackendException when a dynamic AoS is expected but timed_AOS_index is -1.
     */
    std::string getTimeVectorDataSetName(int homogeneous_time, const std::string &timebasePath, int timed_AOS_index, std::vector < std::string > &tensorized_paths);
    /**
     * @brief Same resolution, taking the AoS path chain from the per-operation cache.
     * @param opCtx          The operation context owning the path chain.
     * @param timebasename   Time base path from the dictionary ('/' flattened in place).
     * @param timed_AOS_index Dynamic AoS level index, or -1.
     * @return Dataset name of the time base.
     */
    std::string getTimeVectorDataSetName(OperationContext * opCtx, std::string timebasename, int timed_AOS_index);
    /**
     * @brief Opens a time-base dataset.
     * @param opCtx           The operation context (supplies the URI).
     * @param gid             IDS group handle.
     * @param dataset_name    Time-base dataset name.
     * @param time_vector_dim Dimensionality as seen by the caller (0 when a dynamic AoS carries the axis).
     * @return An open handler for the time base (caller keeps or releases it).
     */
    std::unique_ptr < HDF5DataSetHandler > getTimeVectorDataSet(OperationContext *opCtx, hid_t gid, const std::string & dataset_name, int time_vector_dim);

    /**
     * @brief Puts the handler back into the cache before leaving read_ND_Data, then returns the status.
     * @param data_set    Handler to keep open (moved away from the caller).
     * @param exit_status 1 when data was returned, 0 when the read produced nothing.
     * @return `exit_status`.
     */
    int exit_request(std::unique_ptr < HDF5DataSetHandler > &data_set, int exit_status);
    /**
     * @brief The data-entry context behind any context kind.
     * @param ctx Operation or AoS context.
     */
    DataEntryContext* getDataEntryContext(Context * ctx);

    /**
     * @brief Normalises a string result: the reported size becomes the stored length, and a
     *        null result becomes a single empty string.
     * @param datatype Must be char_data for anything to happen.
     * @param data     [in/out] The string block.
     * @param size     [in/out] Reported size.
     * @param dim      Requested dimensionality (1 for a scalar string, 2 for a list).
     * @return 0 when a list has no value to return (the caller returns nothing), else 1.
     */
    int fixCharDataSet(int datatype, void **data, int *size, int dim);
    /// True when a stored time equals the IMAS invalid value -9.0E40 (an unwritten slot).
    static bool isMatchingDefaultValue(double value);
    /// True when `str` ends with `suffix`.
    static bool ends_with(const std::string &str, const std::string &suffix);

    /**
     * @brief Reads ids_properties&homogeneous_time out of an open IDS group.
     * @param homogenenous_time [out] The flag, or -1 when absent/unreadable.
     * @param gid               IDS group handle.
     */
    void read_homogeneous_time(int* homogenenous_time, hid_t gid);

    /// Whether an impossible interpolation is reported on stdout (false when IMAS_AL_DISABLE_INTERPOLATION_WARNING is set).
    bool INTERPOLATION_WARNING;
    std::vector<double> time_basis_vector;   // Time base for the request in progress; reused across the AoS instances of one read.

  public:

    /**
     * @brief Binds the reader to one backend format version.
     * @param backend_version_ Version pair of the file being read (see HDF5BackendFactory).
     */
     HDF5Reader(std::pair<int,int> backend_version_);
    /**
     * @brief Virtual destructor; dataset and file handles are released by close_datasets()/closePulse().
     */
    virtual ~HDF5Reader();
    /**
     * @brief Version string of this backend implementation.
     * @return "2.0.0".
     */
    virtual std::string getVersion();

    /// Cached ids_properties&homogeneous_time of the IDS being read (1 homogeneous, 2 inhomogeneous, -1 unknown).
    int homogeneous_time;
    std::unordered_map < OperationContext *,  hid_t> IDS_group_id;   // Open IDS group, keyed by operation context.

    /**
     * @brief Ends a read pulse: closes datasets, IDS files and the master file.
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
     * @brief Reads one node's values for one AoS instance (the AL get / get_slice / get_time_range entry point).
     * @param ctx          Context of the read (an AoS context, or the operation context at top level).
     * @param att_name     IMAS path of the node ('/' is flattened to '&' in place).
     * @param timebasename Time base path from the dictionary ('/' flattened in place); empty means
     *                     the node is not dynamic.
     * @param datatype     [in/out] AL datatype of the values.
     * @param data         [out] Newly allocated block holding the values (caller frees).
     * @param dim          [in/out] Requested dimensionality; corrected from the stored shape when
     *                     a shape companion is available.
     * @param size         [in/out] Requested extents, updated to the extents actually returned.
     * @return 1 when values were returned, 0 when the node, the instance or the requested time
     *         window has no data — the AL then reports an empty result.
     * @note The path is memoised: a node known to be absent short-circuits for the rest of the
     *       operation.
     * @note Under SLICE_OP the requested time resolves to one slice, or to the two slices
     *       bracketing it when linear interpolation is asked; under TIMERANGE_OP the window is
     *       read in one selection, and a caller-supplied dtime grid resamples it.
     * @note Reading ids_properties&homogeneous_time as a normal node also refreshes this
     *       reader's cached flag, and reports 1 when a resampling makes the result homogeneous.
     */
    virtual int read_ND_Data(Context * ctx, std::string & att_name, std::string & timebasename, int *datatype, void **data, int *dim, int *size);
    /**
     * @brief Opens one AoS level for reading: reports how many instances exist and resolves the time slice.
     * @param ctx  The AoS context being opened.
     * @param size [out] Instance count: the stored &AOS_SHAPE extent, narrowed to 1 for a
     *             requested slice or to the time-range count for a requested window; 0 when
     *             the IDS, the level or the instance does not exist.
     * @note One H5Lexists plus one shape read per level, and the answer is memoised, so an
     *       absent subtree costs almost nothing on the second visit.
     */
    virtual void beginReadArraystructAction(ArraystructContext * ctx, int *size);
    /**
     * @brief Lists the occurrences of an IDS recorded as links in the master file.
     * @param ids_name         IDS name to look for (link names are "<ids_name>_<occurrence>",
     *                         with '_' inside the name allowed).
     * @param occurrences_list [out] malloc'ed sorted occurrence numbers (caller frees).
     * @param size             [out] Number of occurrences.
     * @param master_file_id   Master file handle (must be open).
     * @throws ALBackendException when the master file is not open or H5Literate fails.
     */
    virtual void get_occurrences(const char* ids_name, int** occurrences_list, int* size, hid_t master_file_id);

    /**
     * @brief Opens the IDS file and group for reading, caching the handle per operation.
     * @param ctx              The operation context naming the IDS.
     * @param file_id          Master file handle.
     * @param opened_IDS_files [in/out] Cache of IDS file handles.
     * @param files_directory  Pulse directory.
     * @param relative_file_path Master file name inside it.
     */
    virtual void open_IDS_group(OperationContext * ctx, hid_t file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, std::string & files_directory, std::string & relative_file_path);
     /**
      * @brief Closes the IDS group of an operation and removes it from the cache.
      * @param ctx The operation context whose group is closed (a no-op when none is open).
      * @throws ALBackendException when H5Gclose fails.
      */
     virtual void close_group(OperationContext *ctx);
    /**
     * @brief Closes one AoS level: drops its path and shape bookkeeping and pops the
     *        operation-level path chain back to the parent level.
     * @param ctx The context being closed (a no-op for non-AoS contexts).
     */
    virtual void endAction(Context * ctx);

    /**
     * @brief Closes one IDS file from the cache and resets its entry to -1.
     * @param external_link_name IDS name ('/' accepted; normalised to '_').
     * @param opened_IDS_files   [in/out] Cache of IDS file handles.
     */
    void close_file_handler(std::string external_link_name, std::unordered_map < std::string, hid_t > &opened_IDS_files);
    /**
     * @brief Closes every open handler (values, field shapes, AoS shapes) and clears the caches.
     */
    void close_datasets();
    /**
     * @brief Sets the slice mode from the operation's range mode (TIMERANGE_OP counts as GLOBAL_OP here).
     * @param ctx The operation context.
     */
    void setSliceMode(OperationContext *ctx);
};

#endif
