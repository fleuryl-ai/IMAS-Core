/**
 * @file hdf5_backend.h
 * @brief The HDF5 backend: the AL Backend implementation for IMAS pulse files.
 *
 * This class is deliberately thin. It owns the pulse session (master-file handle, the
 * IDS-file cache, the access mode) and the version decision that selects the storage
 * generation, then forwards every operation to a writer, a reader and an action-event
 * handler created by HDF5BackendFactory. Two on-disk generations are supported side by
 * side: v1 (one dataset per IMAS node, see HDF5Writer/HDF5Reader) and v2 (PanzerDB:
 * columnar raw datasets plus a central /index table, see HDF5Writer_v2/HDF5Reader_v2).
 * A file's HDF5_BACKEND_VERSION attribute picks the reader; the target version picks
 * the writer, and the two may differ only where an upgrade is allowed (see getVersion()).
 */
#ifndef HDF5_BACKEND_H
#define HDF5_BACKEND_H 1

#include <hdf5.h>
#include "al_backend.h"
#include "hdf5_reader.h"
#include "hdf5_writer.h"
#include "hdf5_events_handler.h"
#include <memory>
#include <vector>
#include <list>
#include <set>
#include <string>

#include "data_interpolation.h"

/**
 * @class HDF5Backend
 * @brief AL backend for the HDF5 pulse layout; dispatcher over the storage generations.
 *
 * Lifetime: constructed per data entry, then openPulse() binds it to a pulse directory
 * and fixes access_mode; the AL drives it through beginAction/endAction and the
 * read/write/delete calls; closePulse() releases the file handles.
 */
class HDF5Backend:public Backend {
  private:

    // Highest format generation this build can write. Reported by the backend and used as
    // the fallback when the environment names no target version.
    static const int HDF5_BACKEND_VERSION_MAJOR;
    static const int HDF5_BACKEND_VERSION_MINOR;

    std::string files_directory; //Pulse directory, taken from the URI "path" query.
    std::string relative_file_path;   // Master file name inside it ("master.h5").

    hid_t file_id;   // Master file handle; -1 when no pulse is open.
     std::string pulseFilePath;      // Full path of the master file.
     std::unordered_map < std::string, hid_t > opened_IDS_files;   // IDS file handles, keyed by external-link name; -1 for a link not opened yet.

     std::unique_ptr < HDF5Writer > hdf5Writer;   // Writer for the version chosen by getVersion().
     std::unique_ptr < HDF5Reader > hdf5Reader;   // Reader for the version recorded in the file.

     std::unique_ptr < HDF5EventsHandler > eventsHandler;   // Turns AL action events into reader/writer call sequences.

    int access_mode;   // The mode passed to openPulse(); selects reader vs writer paths.
    int files_path_strategy;   // Pulse directory layout (always MODIFIED_MDSPLUS_STRATEGY here).

    // A session whose target version does not match the layout on disk is a rewrite, not an
    // edit: one layout has to be erased before the other is laid down. conversion_to_v1_ says a
    // v2 pulse was accepted for a 1.0 rewrite, upgrade_to_v2_ says a v1 pulse was accepted for a
    // 2.0 one -- getVersion decides both, each under its own environment switch.
    // storage_rewritten_ then says the master file has to carry the new label at closePulse():
    // an IDS file is annotated by whoever creates it, but only the master's label picks the
    // reader of the next open.
    bool conversion_to_v1_ = false;
    bool upgrade_to_v2_ = false;
    bool storage_rewritten_ = false;
    // The drop is per IDS file, not per session: a session may rewrite several IDS of one pulse,
    // and each of their files has to be erased once. Doing it once in total would leave the
    // second file holding both layouts.
    std::set < std::string > rewritten_IDS_;

    /**
     * @brief Erases one IDS file so the writer can lay the other layout down from scratch.
     * @param ctx The write operation about to start (names the IDS).
     * @note Called for the first write of an IDS in a session whose layout differs from the one
     *       on disk, in either direction (v2 -> v1 and v1 -> v2). Runs before the event handler
     *       opens the IDS file for that operation, so the writer -- v1 or v2 -- creates the file,
     *       its external link and its own version header itself
     *       (HDF5Writer::create_IDS_group) instead of adding its datasets next to the previous
     *       layout's, which would leave a file holding both.
     * @note Deleting the file rather than unlinking its objects is deliberate: HDF5 hands no
     *       space back, so an emptied file would keep the size of the layout it is leaving,
     *       which is exactly what a conversion is meant to measure.
     * @note Releases the read session first -- the v2 one keeps an engine and its handles alive
     *       across operations, and they point at the objects about to go away.
     */
    void dropIDSStorageForRewrite(OperationContext *ctx);
    /**
     * @brief Parses a "major.minor" version string.
     * @param versionStr Text to parse.
     * @return The version pair, or the compiled default when the string has no dot.
     */
    std::pair<int, int> parseVersion(const std::string& versionStr);
    /**
     * @brief The version this process writes by default (BACKEND_DEFAULT_WRITE_VERSION, else 2.0).
     * @return Version pair.
     */
    std::pair<int, int> getTargetVersion();
    /**
     * @brief Whether rewriting a v1 file as v2 is permitted (BACKEND_ALLOW_AUTO_UPGRADE == "true").
     * @return True when the environment allows the upgrade.
     */
    bool allowUpgrade();
    /**
     * @brief Whether a v2 file may be rewritten as v1 (BACKEND_HDF5_FORCE_CONVERSION_TO_V1 == "true").
     * @return True when the environment asks for the conversion.
     * @note A testing switch, not a production path: converting means deleting the v2 tables
     *       of the pulse and re-annotating it, which is destructive and bypasses the SWMR
     *       discipline (a reader in another process may be holding those very tables).
     */
    bool forceConversionToV1();
    /**
     * @brief Instantiates the writer, reader and event handler for one format version.
     * @param backend_version Version pair to build components for.
     * @note Replacing the components is safe only between operations: open handles belong
     *       to the previous generation.
     */
    void createBackendComponents(std::pair<int,int> backend_version);
    /**
     * @brief Target version of this instance (equivalent to getVersion(nullptr)).
     * @return Version pair.
     */
    std::pair<int,int> getVersion();

    // Declared for the pulse-path helpers now provided by HDF5Utils; no definition exists
    // in hdf5_backend.cpp and nothing calls it.
    std::string getPulseFilePath(DataEntryContext * ctx);

    /**
     * @brief Opens one AoS level for writing (skipped entirely when the level is empty).
     * @param ctx  The AoS context.
     * @param size [in/out] Declared instance count.
     */
    void beginWriteArraystructAction(ArraystructContext * ctx, int *size);
    /**
     * @brief Opens one AoS level for reading, reporting how many instances exist.
     * @param ctx  The AoS context.
     * @param size [out] Instance count (0 when the subtree is absent).
     */
    void beginReadArraystructAction(ArraystructContext * ctx, int *size);

  public:

    /**
     * @brief Creates a backend for the target format version and instantiates its components.
     */
     HDF5Backend();
    /**
     * @brief Copy-style constructor reserved for the factory interface; it copies nothing
     *        and leaves the object uninitialised, so it must not be used.
     * @param targetB Backend to copy from (ignored).
     */
     HDF5Backend(Backend * targetB);



    /**
     * @brief Destructor; the pulse must already be closed by closePulse().
     */
     virtual ~ HDF5Backend();

    /**
     * @brief Reports the format version this session reads and writes, and installs the
     *        matching reader and writer.
     * @param ctx The pulse context, or nullptr to ask only for the target version.
     * @return The version the AL should treat this pulse as.
     * @note The reader always follows the version physically recorded in the master file,
     *       so a v1 file is always read with the v1 reader.
     * @note Under OPEN_PULSE/FORCE_OPEN_PULSE a file older than the target version is only
     *       accepted when BACKEND_ALLOW_AUTO_UPGRADE is true (otherwise an upgrade to 2.0 is
     *       refused), and a file newer than the target is refused unless
     *       BACKEND_HDF5_FORCE_CONVERSION_TO_V1 is true -- which arms this session as a
     *       v2 to v1 conversion: the next write drops the v2 storage and closePulse()
     *       re-annotates the pulse as 1.0.
     * @note Reading stays on the physical version either way, so a conversion session reads
     *       the v2 tables until it writes, and finds nothing afterwards.
     * @note The open/force-open branches fall through to the create branch, which returns the
     *       target version; the writerVersion/versionToReturn path below is therefore only
     *       reached when access_mode is none of the four pulse modes (in which case those
     *       two locals are returned without having been set).
     * @throws ALBackendException when the file's version would require an unauthorised
     *         upgrade or a downgrade.
     */
     std::pair<int,int> getVersion(DataEntryContext *ctx) override;

        /**
     * @brief Opens a pulse, creating it when the mode and the target version call for it.
     * @param ctx  The pulse context whose URI locates the pulse directory.
     * @param mode Opening option:
     * - OPEN_PULSE = open an existing pulse (only if it exists)
     * - FORCE_OPEN_PULSE = open a pulse (create it if it does not exist)
     * - CREATE_PULSE = create a new pulse (do not overwrite if it already exists)
     * - FORCE_CREATE_PULSE = create a new pulse (erase the old one if it already exists)
     * @note Records the mode as access_mode, which every later read/write/delete call uses
     *       to choose between the reader and the writer.
     * @note A missing master file under FORCE_OPEN_PULSE is created with the target version,
     *       which is why that mode can reach the writer path.
     * @throws ALBackendException for an unsupported mode, or when the version decision fails.
     */
    void openPulse(DataEntryContext * ctx, int mode) override;

        /**
     * @brief Closes a pulse: releases datasets, IDS files and the master file.
     * @param ctx  The pulse context.
     * @param mode Closing option:
     * - CLOSE_PULSE = close the pulse
     * - ERASE_PULSE = close and remove the pulse
     * @note The reader or the writer closes the files according to access_mode, then both
     *       caches are cleared so no handle survives the session.
     * @throws ALBackendException when the context is null.
     */
    void closePulse(DataEntryContext * ctx, int mode) override;

        /**
     * @brief Writes one signal for the AoS instance addressed by the context.
     * @param ctx          Operation or AoS context addressing the node.
     * @param fieldname    Field path within the IDS.
     * @param timebasename Time base field name from the dictionary (empty when the node is
     *                     not dynamic; it is what makes a read time-aware).
     * @param data         Values to write.
     * @param datatype     Type of the values:
     * - CHAR_DATA strings
     * - INTEGER_DATA integers
     * - DOUBLE_DATA double precision floating points
     * - COMPLEX_DATA complex numbers
     * @param dim          Dimensionality of the data (0=scalar, 1=1D vector, ... up to MAXDIM).
     * @param size         Extent of each dimension (ignored when dim is 0).
     * @throws ALBackendException when the node cannot be created or written.
     */
    void writeData(Context * ctx, std::string fieldname, std::string timebasename, void *data, int datatype, int dim, int *size) override;

        /**
     * @brief Reads one signal for the AoS instance addressed by the context.
     * @param ctx          Operation or AoS context addressing the node.
     * @param fieldname    Field path within the IDS.
     * @param timebase     Time base field name from the dictionary (empty when the node is
     *                     not dynamic).
     * @param data         [out] Newly allocated block holding the values (caller frees).
     * @param datatype     [out] Type of the returned data:
     * - CHAR_DATA strings
     * - INTEGER_DATA integers
     * - DOUBLE_DATA double precision floating points
     * - COMPLEX_DATA complex numbers
     * @param dim          [out] Dimensionality of the returned data.
     * @param size         [out] Extent of each returned dimension.
     * @return 1 when values were returned, 0 when the node or the requested instance/window
     *         has no data (the AL then reports an empty result rather than an error).
     * @throws ALBackendException when the read itself fails.
     */
    int readData(Context * ctx, std::string fieldname, std::string timebase, void **data, int *datatype, int *dim, int *size) override;

        /**
    * @brief Removes an IDS from the pulse: its file and its link in the master file.
    * @param ctx  The operation context naming the IDS.
    * @param path Path of the element to delete; the HDF5 backend deletes at IDS granularity,
    *             so this argument does not narrow the scope.
    * @note A no-op when no master file is open.
    */
    void deleteData(OperationContext * ctx, std::string path) override;

    /**
     * @brief Opens one AoS level, dispatching on the operation's access mode.
     * @param ctx  The AoS context.
     * @param size [in/out] Instance count: declared for a write, resolved for a read.
     */
    void beginArraystructAction(ArraystructContext * ctx, int *size) override {
        if (ctx->getOperationContext()->getAccessmode() == READ_OP)
            beginReadArraystructAction(ctx, size);
        else
            beginWriteArraystructAction(ctx, size);
    }
    /**
     * @brief Closes the level that just ended (an AoS level or the whole operation).
     * @param ctx The context being closed.
     */
    void endAction(Context * ctx) override;

    /**
     * @brief Starts an operation: opens the IDS group and installs the storage strategy.
     * @param ctx The operation context.
     */
    void beginAction(OperationContext * ctx) override;

    /**
     * @brief Lists the occurrences of an IDS recorded in the master file.
     * @param ctx              The context requesting the list (unused; the master file is the source).
     * @param ids_name         IDS name to look for.
     * @param occurrences_list [out] malloc'ed sorted occurrence numbers (caller frees).
     * @param size             [out] Number of occurrences.
     * @throws ALBackendException when no master file is open.
     */
    void get_occurrences(Context* ctx, const char* ids_name, int** occurrences_list, int* size) override;

    /**
     * @brief Whether the backend interpolates in time itself (it does, for both generations).
     */
    bool supportsTimeDataInterpolation() override {
      return true;
    }

    /**
     * @brief Nothing to prepare: each reader owns its interpolation component.
     */
    void initDataInterpolationComponent() override {
    }

    /**
     * @brief Whether get_time_range requests are supported (they are, with resampling).
     */
    bool supportsTimeRangeOperation() override {
		return true;
    }

};

#endif
