/**
 * @file hdf5_utils.h
 * @brief Pulse-layout and file/group lifecycle helpers shared by the v1 and v2 backends.
 *
 * A pulse on disk is a directory holding a master file (master.h5) plus one file per
 * IDS, joined by external links whose name is the IDS name with '/' replaced by '_'.
 * This class resolves those paths, creates/opens/closes/deletes the two kinds of file,
 * writes the version attribute and the user block, and exposes the small context
 * helpers (AoS index chain, timed level, tensorized names) that the readers and
 * writers use to address one instance of a node.
 *
 * Tuning is reachable without recompiling: the URI query options handled by
 * readOptions() and the environment variables handled by setDefaultOptions() and by
 * the file-access property list built inside hdf5_utils.cpp.
 */
#ifndef HDF5_UTILS_H
#define HDF5_UTILS_H 1

#include <hdf5.h>
#include "al_backend.h"
#include <memory>
#include <vector>
#include <list>
#include <unordered_map>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#include <Shlobj.h>
#include "fix_include_windows.h"
#else
#include <pwd.h>
#endif

/**
 * @brief H5Literate callback recording one external link of the master file.
 * @param loc_id Master file handle being iterated.
 * @param name   Link name (the IDS name with '/' replaced by '_').
 * @param linfo  Link information (unused).
 * @param opdata Accumulator, expected to point at an `opdata`.
 * @return 0 to continue the iteration.
 * @note As a side effect it opens each existing IDS file in read/write mode and closes
 *       it again, which is how a stale external link is dropped before deletion.
 */
extern "C" herr_t file_info(hid_t loc_id, const char *name, const H5L_info_t * linfo, void *opdata);

/**
 * @struct opdata
 * @brief Accumulator for the master-file link iteration (fixed capacity: 500 links).
 *
 * `link_names` entries are malloc'ed by the callback and freed by the caller.
 */
struct opdata {
    bool mode;                  //0=writing, 1=reading
     std::string files_directory;     //Pulse directory, used to resolve each link to a file.
     std::string relative_file_path;  //Master file name inside that directory.
    int count;                  //Number of links collected so far.
    hid_t file_ids[500];        //Unused by the current callback (kept for the interface).
    char *link_names[500];      //Collected link names (one malloc each).
};

/**
 * @class HDF5Utils
 * @brief Path resolution, file/group lifecycle and context helpers for the HDF5 backend.
 *
 * Most methods are stateless; the class is constructed on demand at each call site.
 * Only `debug` carries state, and it is process-wide.
 */
class HDF5Utils {
  private:
    /**
     * @brief Builds the master-file path for a pulse and creates the directory when needed.
     * @param ctx                The data-entry context (its URI "path" query is the pulse directory).
     * @param mode               Pulse open/create mode (directories are created for the create
     *                           and force-open modes).
     * @param strategy           Path layout selector (see Files_paths_strategies).
     * @param files_directory    [out] Pulse directory taken from the URI.
     * @param relative_file_path [out] Master file name inside it.
     * @return The full master-file path.
     * @throws ALBackendException when the directory cannot be created.
     */
    std::string getPulseFilePath(DataEntryContext * ctx, int mode, int strategy, std::string & files_directory, std::string & relative_file_path);
    /**
     * @brief Deletes every IDS file referenced by the opened-links map.
     * @param opened_IDS_files   Map of link name -> file handle (the names give the file names).
     * @param files_directory    Pulse directory.
     * @param relative_file_path Unused here; kept for symmetry with the other helpers.
     */
    void deleteIDSFiles(std::unordered_map < std::string, hid_t > &opened_IDS_files, std::string & files_directory, std::string & relative_file_path);
    /**
     * @brief Creates a master file with a 1024-byte user block and writes its header.
     * @param ctx            The data-entry context (the URI path is stored in the user block).
     * @param filePath       Master-file path to create.
     * @param file_id        [out] Handle of the new file.
     * @param backend_version Version string to record in the HDF5_BACKEND_VERSION attribute.
     * @throws ALBackendException when the user block or the file cannot be created.
     */
    void createMasterFile(DataEntryContext * ctx, std::string &filePath, hid_t *file_id, std::string &backend_version);
    /**
     * @brief Opens a master file read/write, falling back to read-only.
     * @param file_id      [in/out] Skipped when a file is already open; set to the new handle.
     * @param filePath     Master-file path.
     * @param for_deletion When true and the file cannot be opened at all, it is removed and
     *                     `file_id` is left at -1 so the caller can proceed with deletion.
     * @throws ALBackendException when the file is missing, or unreadable and deletion is not intended.
     */
    void openMasterFile(hid_t *file_id, const std::string &filePath, bool for_deletion = false);
    /**
     * @brief Enumerates the master file's external links into the opened-IDS-files map.
     * @param file_id          Master file handle (must be open).
     * @param opened_IDS_files [out] Map link name -> -1 (not opened yet).
     * @throws ALBackendException when the file handle is not usable.
     */
    void initExternalLinks(hid_t *file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, std::string &files_directory, std::string &relative_file_path);

  public:

    /**
     * @brief Stateless helper; constructed per call site.
     */
     HDF5Utils();
    /**
     * @brief Destructor (nothing to release).
     */
    ~HDF5Utils();

    /// Process-wide debug flag: when false the HDF5 error stack is silenced on open/create.
    static bool debug;

    /**
     * @brief Opens a pulse and reports the backend format its files were written with.
     * @param ctx                 The data-entry context.
     * @param mode                OPEN_PULSE or FORCE_OPEN_PULSE.
     * @param backend_version     [out] The HDF5_BACKEND_VERSION attribute read from the master file.
     * @param file_id             [in/out] Master file handle (an already open one is closed first).
     * @param opened_IDS_files    [in/out] Cache of IDS files reached through external links.
     * @param files_paths_strategy Path layout selector (see Files_paths_strategies).
     * @param files_directory     [out] Pulse directory.
     * @param relative_file_path  [out] Master file name.
     * @param pulseFilePath       [out] Full master-file path.
     * @return 0 on success; -1 for FORCE_OPEN_PULSE when the pulse does not exist yet
     *         (the caller then creates it).
     * @throws ALBackendException for an unsupported mode, or when the master file exists but
     *         carries no HDF5_BACKEND_VERSION attribute — i.e. it is not an IMAS pulse file.
     */
    static int openPulse(DataEntryContext * ctx, int mode, std::string & backend_version, hid_t * file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, int files_paths_strategy, std::string & files_directory, std::string & relative_file_path, std::string &pulseFilePath);

    /**
     * @brief Creates a pulse (master file + version header).
     * @param ctx                 The data-entry context.
     * @param mode                CREATE_PULSE / FORCE_OPEN_PULSE (refuses an existing pulse) or
     *                            FORCE_CREATE_PULSE (deletes the pulse first, IDS files included).
     * @param backend_version     Version string to record.
     * @param file_id             [in/out] Master file handle.
     * @param opened_IDS_files    [in/out] Cache of IDS files reached through external links.
     * @param files_paths_strategy Path layout selector.
     * @param files_directory     [out] Pulse directory.
     * @param relative_file_path  [out] Master file name.
     * @param pulseFilePath       [out] Full master-file path.
     * @throws ALBackendException when the pulse already exists in a non-forcing mode, or for
     *         an unsupported mode.
     */
    static void createPulse(DataEntryContext * ctx, int mode, std::string backend_version, hid_t * file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, int files_paths_strategy, std::string & files_directory, std::string & relative_file_path, std::string &pulseFilePath);

    /**
     * @brief Dispatches the master-file path onto the selected layout strategy.
     * @param ctx                The data-entry context.
     * @param mode               Pulse open/create mode.
     * @param strategy           Layout selector (see Files_paths_strategies).
     * @param files_directory    [out] Pulse directory.
     * @param relative_file_path [out] Master file name.
     * @return The full master-file path.
     * @throws ALBackendException for an unknown strategy, or for FREE_PATH_STRATEGY,
     *         which is declared but not implemented.
     */
     std::string pulseFilePathFactory(DataEntryContext * ctx, int mode, int strategy, std::string & files_directory, std::string & relative_file_path);
    /**
     * @brief Path of the file holding one IDS inside a pulse directory.
     * @param files_directory    Pulse directory.
     * @param relative_file_name Unused here; kept for symmetry with the master-file helper.
     * @param ids_name           IDS name with '/' already replaced by '_'.
     * @return "<files_directory>/<ids_name>.h5".
     */
     std::string getIDSPulseFilePath(const std::string & files_directory, const std::string & relative_file_name, const std::string & ids_name);
    /**
     * @brief True when an IDS file exists on disk.
     * @param IDS_pulse_file Path of the IDS file.
     */
    bool pulseFileExists(const std::string &IDS_pulse_file);
    /**
     * @brief Closes the master file handle and resets it to -1.
     * @param file_id [in/out] Master file handle (a no-op when already closed).
     * @throws ALBackendException when H5Fclose fails.
     */
    void closeMasterFile(hid_t *file_id);
    /**
     * @brief Removes one external link from the master file.
     * @param file_id   Master file handle.
     * @param link_name Link to remove.
     * @throws ALBackendException when H5Ldelete fails.
     */
    void removeLinkFromMasterPulseFile(hid_t &file_id, const std::string &link_name);
    /**
     * @brief Opens an IDS file, creating it when the existing one cannot be opened.
     * @param ctx             The operation context (names the IDS in error messages).
     * @param IDSpulseFile    Path of the IDS file.
     * @param IDS_file_id     [out] File handle; -1 when the file does not exist.
     * @param try_read_only   When a read/write open fails, retry read-only instead of failing.
     * @param backend_version Version to record if the file has to be recreated.
     * @note A read/write open fails either because another process holds the file (errno
     *       EAGAIN/EACCES/EBUSY — reported as such) or because the file is unusable, in which
     *       case it is removed and recreated; the intermediate -2 marks that recreation path.
     * @note The file-access property list is taken from the IMAS_HDF5_FAPL environment
     *       variable when set (cache/alignment/meta-block/sieve-buffer tuning); unset means
     *       HDF5 defaults, i.e. no change in production.
     * @throws ALBackendException when the file can be opened by neither route requested.
     */
    void openIDSFile(OperationContext * ctx, std::string &IDSpulseFile, hid_t *IDS_file_id, bool try_read_only, std::string backend_version = "");
    /**
     * @brief Closes an IDS file handle.
     * @param pulse_file_id      File handle (-1 is ignored).
     * @param external_link_name IDS name, used for the error message.
     * @throws ALBackendException when H5Fclose fails.
     */
    void closeIDSFile(hid_t pulse_file_id, const std::string &external_link_name) ;
    /**
     * @brief Creates an IDS file with a 1024-byte user block and its version header.
     * @param ctx             The operation context (names the IDS in error messages).
     * @param IDSpulseFile    Path to create.
     * @param backend_version Version string to record.
     * @param IDS_file_id     [out] Handle of the new file.
     * @throws ALBackendException when the user block or the file cannot be created.
     */
    void createIDSFile(OperationContext * ctx, std::string &IDSpulseFile, std::string &backend_version, hid_t *IDS_file_id);
    /**
     * @brief Removes a link from an IDS file, when present.
     * @param IDS_file_id    IDS file handle.
     * @param IDS_link_name  Link to remove.
     * @throws ALBackendException when H5Ldelete fails.
     */
    void removeLinkFromIDSPulseFile(hid_t &IDS_file_id, const std::string &IDS_link_name);
    /**
     * @brief Writes the URI path into the file's 1024-byte user block.
     * @param filePath File to patch (opened in binary mode, written at offset 0).
     * @param ctx      The data-entry context providing the URI "path" query.
     * @note The file is edited outside HDF5, so this must happen before the file is
     *       written through the library, and the user block must be reserved at creation.
     * @throws ALBackendException when the file cannot be opened for writing.
     */
    void writeUserBlock(const std::string & filePath, DataEntryContext * ctx);
    /**
     * @brief Writes the HDF5_BACKEND_VERSION attribute on a file root and its user block.
     * @param ctx             The data-entry context (forwarded to writeUserBlock()).
     * @param file_id         File handle to annotate.
     * @param filePath        File whose user block is written.
     * @param backend_version Version text ("1.0" / "2.0") recorded on the root.
     * @note This attribute is what openPulse() reads back to pick the reader/writer version,
     *       so a file without it is rejected as "not an IMAS HDF5 pulse file".
     * @throws ALBackendException when the attribute cannot be created or written.
     */
    void writeHeader(DataEntryContext * ctx, hid_t file_id, std::string & filePath, std::string backend_version);
    /**
     * @brief Removes one IDS file, if it exists.
     * @param filePath File to remove.
     * @throws ALBackendException when the file is still present after remove().
     */
    void deleteIDSFile(const std::string &filePath);
    /**
     * @brief Deletes a whole pulse: every linked IDS file, then the master file.
     * @param filePath         Master-file path.
     * @param file_id          [in/out] Master file handle (opened and closed here).
     * @param opened_IDS_files [out] Filled with the links found in the master file.
     * @param files_directory  Pulse directory.
     * @param relative_file_path Master file name inside it.
     * @note A master file that cannot be opened at all is removed directly; its IDS files
     *       may then survive and are reported on stdout for manual cleanup.
     * @throws ALBackendException when the master file is still present after remove().
     */
    void deleteMasterFile(const std::string &filePath, hid_t *file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, std::string &files_directory, std::string &relative_file_path);

    /**
     * @brief Opens the IDS file and the group the master file uses for this IDS.
     * @param ctx              The operation context naming the IDS.
     * @param file_id          Master file handle (the group lives here).
     * @param opened_IDS_files [in/out] Cache of IDS file handles, keyed by link name.
     * @param files_directory  Pulse directory.
     * @param relative_file_path Master file name inside it.
     * @param IDS_group_id     [out] Group handle to read/write the IDS through.
     * @throws ALBackendException when the IDS file or its group cannot be opened.
     */
    void open_IDS_group(OperationContext * ctx, hid_t file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, std::string & files_directory, std::string & relative_file_path, hid_t *IDS_group_id);
    /**
     * @brief Opens a group under `parent_loc_id`, creating it when absent.
     * @param path          IMAS path; the stored name replaces '/' with '_'.
     * @param parent_loc_id Parent group or file handle.
     * @return The group handle (caller closes).
     * @throws ALBackendException when the group can be neither created nor opened.
     */
    hid_t createOrOpenHDF5Group(const std::string & path, const hid_t & parent_loc_id);
    /**
     * @brief Creates a group under `parent_loc_id`, reporting whether it already existed.
     * @param path                IMAS path; the stored name replaces '/' with '_'.
     * @param parent_loc_id       Parent group or file handle.
     * @param group_already_exists [out] True when the group was left untouched.
     * @return The new group handle, or -1 when the group already existed.
     * @throws ALBackendException when creation fails.
     */
    hid_t createHDF5Group(const std::string & path, const hid_t & parent_loc_id, bool * group_already_exists);
    /**
     * @brief Opens an existing group.
     * @param path          IMAS path; the stored name replaces '/' with '_'.
     * @param parent_loc_id Parent group or file handle.
     * @return The group handle, or -1 when the group is absent.
     * @throws ALBackendException when the group exists but cannot be opened.
     */
    hid_t openHDF5Group(const std::string & path, const hid_t & parent_loc_id);
    /**
     * @brief Larger of two values.
     * @param num1 First value.
     * @param num2 Second value.
     */
    int max(int num1, int num2);
	/**
	 * @brief Smaller of two values.
	 * @param num1 First value.
	 * @param num2 Second value.
	 */
	int min(int num1, int num2);
    /**
     * @brief Compares two slice shapes dimension by dimension.
     * @param first_slice_shape  First shape.
     * @param second_slice_shape Second shape.
     * @param dim                Number of dimensions to compare.
     * @return 0 when identical, 1 when any dimension differs.
     */
    int compareShapes(int *first_slice_shape, int *second_slice_shape, int dim);
    /**
     * @brief Row-major flat index of a multi-dimensional position.
     * @param indices Index along each dimension, in the same order as `shapes`.
     * @param shapes  Extent along each dimension.
     * @return The element offset, i.e. ((i0 * n1) + i1) * n2 + i2 ...
     */
    int indices_to_flat_index(const std::vector<int>& indices, const hsize_t* shapes);
    /**
     * @brief Finds the dynamic (time-evolving) AoS level above a context.
     * @param ctx             Context to inspect (only CTX_ARRAYSTRUCT_TYPE carries a chain).
     * @param timed_AOS_index [out] Position of the timed level counted from the OUTERMOST level
     *                              (matching the dataset dimension order), or -1 when untouched.
     * @return True when some level in the chain is timed.
     */
    bool isTimed(Context * ctx, int *timed_AOS_index);
    /**
     * @brief Collects the AoS index chain of a context, outermost first.
     * @param ctx            Context to walk (a no-op for a non-AoS context).
     * @param indices        [out] getIndex() of each open AoS level, outermost first.
     * @param timedAOS_index [out] Position of the timed level in `indices`, or -1 when none.
     */
    void getAOSIndices(Context * ctx, std::vector < int >&indices, int *timedAOS_index);
    /**
     * @brief Number of AoS levels open above a context (0 for a non-AoS context).
     * @param ctx Context to inspect.
     */
    int getAOSIndicesSize(Context * ctx);
    /**
     * @brief Appends one AoS level to the tensorized path chain of a context.
     * @param ctx             The AoS level to append.
     * @param tensorized_paths [in/out] Cumulative paths, one per open level; each entry is
     *                       the parent entry plus "name[]", with '/' flattened to '&'.
     * @note The "[]" placeholder marks where the instance index is substituted when the
     *       dataset name is finalised.
     */
    void setTensorizedPaths(ArraystructContext * ctx, std::vector < std::string > &tensorized_paths);
    /**
     * @brief Prints how many groups, datasets, files and objects are open in a file (debug aid).
     * @param file_id File to query.
     */
    void showStatus(hid_t file_id);
    /**
     * @enum Files_paths_strategies
     * @brief Layout of a pulse directory.
     */
    enum Files_paths_strategies { FULL_MDSPLUS_STRATEGY = 1,   ///< MDSplus-style tree (currently resolved like the modified variant).
                                 MODIFIED_MDSPLUS_STRATEGY = 2, ///< URI "path" query used as the pulse directory (the backend default).
                                 FREE_PATH_STRATEGY = 3 };      ///< Declared only; not implemented.
    /**
     * @brief Loads the environment-level defaults for caches and buffering.
     * @param read_cache     [in/out] Read chunk cache in bytes (HDF5_BACKEND_READ_CACHE, in MB).
     * @param write_cache    [in/out] Write chunk cache in bytes (HDF5_BACKEND_WRITE_CACHE, in MB).
     * @param readBuffering  [out] Always true here; URI options may turn it off afterwards.
     * @param writeBuffering [out] Always true here; URI options may turn it off afterwards.
     */
    void setDefaultOptions(size_t *read_cache, size_t *write_cache, bool *readBuffering, bool *writeBuffering);
    /**
     * @brief Applies the per-call URI query options, overriding the environment defaults.
     * @param uri                The backend URI.
     * @param compression_enabled [out] hdf5_compression (on unless "no"/"n").
     * @param readBuffering      [out] hdf5_read_buffering (on unless "no"/"n").
     * @param read_cache         [in/out] hdf5_read_cache, in MB.
     * @param writeBuffering     [out] hdf5_write_buffering (on unless "no"/"n").
     * @param write_cache        [in/out] hdf5_write_cache, in MB.
     * @param debug              [out] hdf5_debug ("yes"/"y" turns the HDF5 error stack on).
     */
    void readOptions(uri::Uri uri, bool *compression_enabled, bool *readBuffering, size_t *read_cache, bool *writeBuffering, size_t *write_cache, bool *debug);
    /**
     * @brief Finds the nearest timed (time-evolving) AoS context at or above `ctx`.
     * @param ctx          The AoS context to start from.
     * @param timedContext [out] The timed context, or nullptr when the chain has none.
     */
    void timedContext(ArraystructContext * ctx, ArraystructContext *&timedContext);

};

#endif
