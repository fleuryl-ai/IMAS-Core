/**
 * @file hdf5_dataset_handler.h
 * @brief Owns one open v1 HDF5 dataset: creation with chunking/compression, extent
 *        growth along the time axis, buffered writes and cached reads.
 *
 * The v1 layout stores each IMAS node in its own extensible chunked dataset whose
 * leading dimensions are the open AoS levels (one per AoS level, outermost first)
 * and whose trailing dimensions are the array itself — with the time steps appended
 * along the FIRST array dimension when the signal is dynamic without an enclosing
 * dynamic AoS. This class is where that dataset is created, resized, written and
 * read; HDF5Writer/HDF5Reader keep one handler per opened dataset and drive it with
 * the AoS indices of the current context.
 *
 * @note Two optimisations live here: writes of small (0D/1D, GLOBAL_OP) requests are
 *       accumulated in memory and merged into the dataset at flush time, and reads of
 *       the same kind load the whole dataset once into a full buffer and serve every
 *       later AoS instance from it. Both are skipped whenever a time range or a slice
 *       operation is in play, which goes through hyperslabs instead.
 * @note The v2/PanzerDB backend does not use this class: it appends to a handful of
 *       flat columnar datasets and records positions in its /index table.
 */
#ifndef HDF5_DATASETHANDLER_H
#define HDF5_DATASETHANDLER_H 1

#include <hdf5.h>
#include "al_backend.h"
#include "hdf5_hs_selection_reader.h"

#include <vector>
#include <string>
#include <set>
#include <unordered_map>

// Legacy: an unsigned-integer AL datatype value kept for compatibility with old
// callers. No code path in this backend maps a node onto it, so datasets are never
// created or read with it.
#define UINT_DATA   61

// Defaults, overridable per call through the backend URI (see HDF5Utils::readOptions).
#define READ_CHUNK_CACHE_SIZE 5*1024*1024      // HDF5 raw-data cache for reads.
#define WRITE_CHUNK_CACHE_SIZE 100*1024*1024   // HDF5 raw-data cache for writes.

/**
 * @struct complex_t
 * @brief In-memory layout of one complex value, mirrored by an HDF5 compound type
 *        with the members "real" and "imaginary" (see setType()).
 */
typedef struct {
   double re;   /*real part */
   double im;   /*imaginary part */
} complex_t;

/**
 * @class HDF5DataSetHandler
 * @brief One open v1 dataset plus the extent, chunking and buffering state around it.
 *
 * Handles are owned: dataset_id and dataspace_id are closed by close()/the
 * destructor, and dtype_id is closed only when it was created or copied here
 * (`immutable` false: complex and string types).
 */
class HDF5DataSetHandler {
  private:

    std::string tensorized_path;        //Dataset name as stored: the IMAS path with '/' flattened to '&'.

    int dataset_rank;     //Total rank = AOSRank + the array dimensions carried by the node.
    int AOSRank;          //Number of leading AoS dimensions (0 when the node sits outside any AoS).
    bool immutable;       //True when dtype_id is a builtin native type (never closed); false when it was created/copied here and must be closed.
	bool shape_dataset;   //True for the companion *_AOS_SHAPE datasets that record AoS sizes.

    bool slice_mode;      //True for a SLICE_OP session: the operation appends time steps instead of rewriting the node.
	int slices_extension; //Time extent of the incoming request (request size on its last dimension): the steps this write adds.
    int timed_AOS_index;  //AOS level that is dynamic (time-evolving), or -1 when none.
    bool isTimed;         //True when an enclosing dynamic AoS supplies the time axis.
    int timedAOS_shape;   //Slice count of that dynamic AoS (slice mode only; see setTimedAOSShape()).
    int timeWriteOffset;  //Element offset along the first array dimension where the appended slices start.

    hsize_t dims[H5S_MAX_RANK];         //Dimensions of the dataset as currently tracked by this handler.
    hsize_t maxdims[H5S_MAX_RANK];      //Maximum dimensions declared at creation (H5S_UNLIMITED for extensible axes).
    hsize_t largest_dims[H5S_MAX_RANK]; //Highest extent reached so far: what H5Dset_extent() is asked for, and the layout used for flat-index lookups.
    hsize_t chunk_dims[H5S_MAX_RANK];   //Chunk dimensions chosen by the create() heuristic.

    hsize_t initial_dims[H5S_MAX_RANK]; //dims as they were at the beginning of a put_slice (storeInitialDims()); the reference the time offsets are computed from.

    /**
     * @brief Maps the AL datatype onto an HDF5 type id and returns its size in bytes.
     * @return Element size (for strings: the stored width, i.e. times the number of
     *         strings for a list).
     * @note Complex data becomes a compound type {real, imaginary}; strings are
     *       variable-length C_S1 in UTF-8. Both set `immutable` false, so the type id
     *       is closed by the destructor.
     * @throws ALBackendException for an unsupported datatype, or when the UTF-8
     *         character set cannot be set.
     */
    int setType();
	/**
	 * @brief Writes the whole `buffer` to the dataset in one shot (shape datasets only),
	 *        then frees it.
	 * @throws ALBackendException when H5Dwrite fails.
	 */
	void writeBuffer();
    /**
     * @brief Enables the merge-on-flush optimisation for the request kinds it supports.
     * @param useBufferingOption Requested setting (from the URI options).
     * @note Buffering applies only outside slice mode and to scalars, 1D numerics and
     *       1D strings; anything else keeps the direct hyperslab path.
     */
    void setBuffering(bool useBufferingOption);
    /**
     * @brief Turns buffering off for requests it cannot serve (dim > 1).
     * @param datatype AL datatype of the request.
     * @param dim      Requested dimensionality.
     */
    void disableBufferingIfNotSupported(int datatype, int dim);
    /**
     * @brief Queues one string value (copied with strdup) for the deferred flush.
     * @param current_arrctx_indices AoS indices of this request (recorded separately in
     *                               requests_arrctx_indices).
     * @param data Pointer to the char* holding the value.
     */
    void appendStringToBuffer(const std::vector < int >&current_arrctx_indices, char **data);
    /**
     * @brief Queues one integer scalar for the deferred flush.
     * @param current_arrctx_indices AoS indices of this request.
     * @param data Pointer to the value.
     */
    void appendInt0DToBuffer(const std::vector < int >&current_arrctx_indices, void *data);
    /**
     * @brief Queues a 1D integer array (copied) for the deferred flush.
     * @param current_arrctx_indices AoS indices of this request.
     * @param data Pointer to the elements.
     * @param dim  Requested dimensionality (the element count comes from the stored shapes).
     */
    void appendIntNDToBuffer(const std::vector < int >&current_arrctx_indices, void *data, int dim);
    /**
     * @brief Queues one double scalar for the deferred flush.
     * @param current_arrctx_indices AoS indices of this request.
     * @param data Pointer to the value.
     */
    void appendDouble0DToBuffer(const std::vector < int >&current_arrctx_indices, void *data);
    /**
     * @brief Queues a 1D double array (copied) for the deferred flush.
     * @param current_arrctx_indices AoS indices of this request.
     * @param data Pointer to the elements.
     * @param dim  Requested dimensionality (the element count comes from the stored shapes).
     */
    void appendDoubleNDToBuffer(const std::vector < int >&current_arrctx_indices, void *data, int dim);

    //read operations -- all of these serve a request out of the whole-dataset cache
    //created by the matching create*Buffer() call, so one H5Dread covers every AoS
    //instance of the dataset; the flat index comes from HDF5Utils::indices_to_flat_index.
    /**
     * @brief Returns one cached string (an empty string where none was stored).
     * @param hsSelectionReader      Reader describing the dataset layout.
     * @param current_arrctx_indices AoS indices selecting the instance.
     * @param data [out] Newly allocated copy of the string (caller frees).
     */
    void read0DStringsFromBuffer(HDF5HsSelectionReader & hsSelectionReader, const std::vector < int >&current_arrctx_indices, void **data);
    /**
     * @brief Reads the whole string dataset into full_data_sets_buffers (one H5Dread).
     * @param hsSelectionReader      Reader describing the dataset layout.
     * @param current_arrctx_indices Unused; kept for the common helper signature.
     * @param data Unused here; values are served later by read0DStringsFromBuffer().
     * @throws ALBackendException when H5Dread fails.
     */
    void create0DStringsBuffer(HDF5HsSelectionReader & hsSelectionReader, const std::vector < int >&current_arrctx_indices, void **data);
    /**
     * @brief Returns one integer scalar out of full_int_data_set_buffer.
     * @param hsSelectionReader      Reader describing the dataset layout.
     * @param current_arrctx_indices AoS indices selecting the instance.
     * @param data [out] Newly allocated int (caller frees).
     * @throws ALBackendException when the cache has not been created.
     */
    void readInt0DFromBuffer(HDF5HsSelectionReader & hsSelectionReader, const std::vector < int >&current_arrctx_indices, void **data);
    /**
     * @brief Returns one 1D integer array out of full_int_data_set_buffer.
     * @param hsSelectionReader      Reader describing the dataset layout.
     * @param current_arrctx_indices AoS indices selecting the instance.
     * @param data [out] Newly allocated block (caller frees).
     * @throws ALBackendException when the cache has not been created.
     */
    void readIntNDFromBuffer(HDF5HsSelectionReader & hsSelectionReader, const std::vector < int >&current_arrctx_indices, void **data);
    /**
     * @brief Returns one 1D double array out of full_double_data_set_buffer.
     * @param hsSelectionReader      Reader describing the dataset layout.
     * @param current_arrctx_indices AoS indices selecting the instance.
     * @param data [out] Newly allocated block (caller frees).
     * @throws ALBackendException when the cache has not been created.
     */
    void readDoubleNDFromBuffer(HDF5HsSelectionReader & hsSelectionReader, const std::vector < int >&current_arrctx_indices, void **data);
    /**
     * @brief Reads the whole integer dataset into full_int_data_set_buffer (one H5Dread).
     * @param hsSelectionReader      Reader describing the dataset layout.
     * @param current_arrctx_indices Unused; kept for the common helper signature.
     * @param data Unused here; values are served later by the read*FromBuffer helpers.
     * @throws ALBackendException when H5Dread fails.
     */
    void createIntBuffer(HDF5HsSelectionReader & hsSelectionReader, const std::vector < int >&current_arrctx_indices, void **data);
    /**
     * @brief Returns one double scalar out of full_double_data_set_buffer.
     * @param hsSelectionReader      Reader describing the dataset layout.
     * @param current_arrctx_indices AoS indices selecting the instance.
     * @param data [out] Newly allocated double (caller frees).
     */
    void readDouble0DFromBuffer(HDF5HsSelectionReader & hsSelectionReader, const std::vector < int >&current_arrctx_indices, void **data);
    /**
     * @brief Reads the whole double dataset into full_double_data_set_buffer (one H5Dread).
     * @param hsSelectionReader      Reader describing the dataset layout.
     * @param current_arrctx_indices Unused; kept for the common helper signature.
     * @param data Unused here; values are served later by the read*FromBuffer helpers.
     * @throws ALBackendException when H5Dread fails.
     */
    void createDoubleBuffer(HDF5HsSelectionReader & hsSelectionReader, const std::vector < int >&current_arrctx_indices, void **data);
    /**
     * @brief Reads the requested slab straight from the file with one hyperslab.
     * @param current_arrctx_indices AoS indices of the instance to read.
     * @param slice_mode             GLOBAL_OP (whole node) or SLICE_OP (one time step).
     * @param is_dynamic             Whether the time axis is one of the array dimensions.
     * @param isTimed                Whether an enclosing dynamic AoS supplies the time axis.
     * @param timed_AOS_index        Dynamic AoS level index, or -1.
     * @param slice_index            Time index to read (-1 when not slicing).
     * @param data                   [out] Newly allocated buffer (caller frees).
     * @param read_strings           True to decode variable-length strings: the values are
     *                               read as pointers, then repacked into a fixed-width
     *                               block of `count` x max-length bytes, and the reader's
     *                               reported shape is updated to that width.
     * @throws ALBackendException when H5Dread fails.
     */
    void readUsingHyperslabs(const std::vector < int >&current_arrctx_indices, int slice_mode, bool is_dynamic, bool isTimed, int timed_AOS_index, int slice_index, void **data, bool read_strings);

  public:

    /**
     * @brief Prepares a handler for one dataset, loading the storage options from the URI.
     * @param writing_mode_ True for a write session (selects the write buffering/cache options).
     * @param uri           Backend URI whose query options override the defaults
     *                      (compression, buffering, chunk-cache sizes, debug).
     */
     HDF5DataSetHandler(bool writing_mode_, uri::Uri uri);
    /**
     * @brief Closes the dataspace and, for complex/string types, the type id created here.
     * @note The dataset itself is closed by close(); the destructor does not close it.
     */
    ~HDF5DataSetHandler();

    int datatype;                     // AL datatype of the stored elements.
    hid_t dataset_id;                 // The open dataset (owned; closed by close()).
    hid_t dtype_id;                   // HDF5 type of the elements (see setType()).
    int request_dim;                  // Dimensionality of the AL request that opened this dataset (0 for a scalar).
    hid_t dataspace_id;               // Current dataspace of the dataset (owned; refreshed by setExtent()).
    hid_t IDS_group_id;               // Enclosing IDS group handle, set by the reader/writer that owns this handler.
    void *buffer;                     // Pending whole-dataset image for the shape-dataset write path (writeBuffer()).
    int buffer_length;                // Number of elements held in `buffer`.
    bool compression_enabled;         // Apply shuffle+deflate to datasets created by this handler.
    bool useBuffering;                // Use the deferred merge/cache paths instead of direct hyperslab I/O.
    size_t chunk_cache_size;          // HDF5 raw-data cache (bytes) used when opening datasets for reading.
    size_t write_chunk_cache_size;    // HDF5 raw-data cache (bytes) declared on created datasets.
    std::vector <std::vector<int> > requests_arrctx_indices;  // AoS indices of each buffered write, one per pending value.
    std::vector <std::vector<int> > requests_shapes;          // Shape of each buffered write (parallel to requests_arrctx_indices).

    // Values queued by the buffered WRITE path, one entry per pending request; consumed and
    // freed by fillFullBuffers(), and freed without writing by close().
    std::vector<char *> data_sets_buffers;
    std::vector<int *> int_data_set_buffer;
    std::vector<double *> double_data_set_buffer;

    // Whole-dataset caches built by the buffered READ path (one H5Dread) and reused for every
    // later AoS instance of the same dataset.
    std::vector<char *> full_data_sets_buffers;
    int * full_int_data_set_buffer;
    double * full_double_data_set_buffer;

    std::unique_ptr < HDF5HsSelectionReader > selection_reader;   // Selector used by readData(); created by the owning reader.

    /**
     * @brief True when both handlers refer to the same open dataset.
     * @param other Handler to compare against.
     * @return False when this handler has no dataset open.
     */
    bool operator==(const HDF5DataSetHandler &other) const;

	/**
	 * @brief Prints the dataset name, rank, AOSRank, slices_extension and per-dimension
	 *        dims/largest_dims to stdout (debug aid).
	 * @param context Prefix written on every line.
	 */
	void showDims(std::string context);
	/**
	 * @brief Prints the AoS indices of a request next to the dataset name (debug aid).
	 * @param context    Prefix written on every line.
	 * @param AOS_indices One index per AoS dimension.
	 */
	void showAOSIndices(std::string context, const std::vector<int> &AOS_indices);
	/**
	 * @brief Prints the AoS shapes of a request next to the dataset name (debug aid).
	 * @param context    Prefix written on every line.
	 * @param AOS_shapes One extent per AoS dimension.
	 */
	void showAOSShapes(std::string context, const std::vector<hsize_t> &AOS_shapes);

	/**
	 * @brief Creates the dataset on disk and opens it.
	 * @param dataset_name     Object name under `loc_id` (the tensorized path).
	 * @param dataset_id       [out] Handle of the new dataset (also stored in this handler).
	 * @param datatype         AL datatype of the elements.
	 * @param loc_id           Group the dataset is created in.
	 * @param dim              Array dimensionality of the node (0 for a scalar).
	 * @param size             Array shape, in AL order (innermost dimension last).
	 * @param AOSRank          Number of leading AoS dimensions to reserve.
	 * @param AOSSize          Extent of each of those AoS dimensions.
	 * @param shape_dataset    True for the *_AOS_SHAPE bookkeeping datasets.
	 * @param create_chunk_cache Whether to attach a write chunk cache to the dataset.
	 * @note Every axis is created unlimited, so a later append only calls H5Dset_extent().
	 * @note Chunking is sized here: the target volume sits between ~10 KB and ~2 MB per chunk.
	 *       The AoS axes are kept whole first and shrunk by halves if the product overshoots;
	 *       failing that the array axes are shrunk; an undersized chunk is grown back up.
	 * @note Filters are added shuffle-first then deflate (HDF5 applies them in add order);
	 *       deflation uses level 1 in v1, trading ratio for speed.
	 * @note Unwritten elements carry the IMAS invalid value as HDF5 fill: -999999999 for
	 *       integers and -9.0E40 for doubles (shape datasets use 0).
	 * @throws ALBackendException when the dataset cannot be created.
	 */
	void create(const char *dataset_name, hid_t * dataset_id, int datatype, hid_t loc_id, int dim, int *size, int AOSRank, int *AOSSize, bool shape_dataset, bool create_chunk_cache);
	/**
	 * @brief Opens an existing dataset and adopts its layout.
	 * @param dataset_name     Object name under `loc_id`.
	 * @param loc_id           Group the dataset lives in.
	 * @param dataset_id       [out] Handle of the opened dataset.
	 * @param dim              Array dimensionality of the request (used to split AoS vs array axes).
	 * @param size             Request shape; its last entry sizes `slices_extension` for a
	 *                         numeric, non-timed signal.
	 * @param datatype         AL datatype of the elements.
	 * @param shape_dataset    True for the *_AOS_SHAPE bookkeeping datasets.
	 * @param create_chunk_cache Whether to open with a private read chunk cache.
	 * @param uri              Backend URI (only used if the dataset has to be created).
	 * @param AOSRank          AoS dimensions to reserve, required when the dataset is missing.
	 * @param AOSSize          Extent of each AoS dimension, required when the dataset is missing.
	 * @note AOSRank is derived from the stored rank and the request: rank - dim for numerics;
	 *       for strings, rank when dim == 1 (a scalar string) and rank - 1 otherwise (a list,
	 *       whose count is a stored axis).
	 * @note Under slice mode a missing dataset is created rather than reported as an error —
	 *       the first slice of a new signal legitimately arrives before its dataset exists.
	 * @throws ALBackendException when the dataset cannot be opened and is not creatable, or
	 *         when the creation path lacks AOSRank/AOSSize.
	 */
	void open(const char *dataset_name, hid_t loc_id, hid_t * dataset_id, int dim, int *size, int datatype, bool shape_dataset, bool create_chunk_cache, uri::Uri uri, int AOSRank=-1, int *AOSSize = NULL);
	/**
	 * @brief Adopts new AoS/array shapes and immediately re-applies the dataset extent.
	 * @param size      Array shape of the request, in AL order.
	 * @param AOSShapes Current extent of each AoS dimension.
	 */
	void setCurrentShapesAndExtend(int *size, std::vector<int> &AOSShapes);
	/**
	 * @brief Adopts new AoS/array shapes into dims, raising largest_dims where they grow.
	 * @param size      Array shape of the request, in AL order.
	 * @param AOSShapes Current extent of each AoS dimension.
	 * @note The time axis is deliberately left alone here: under slice mode the first array
	 *       dimension of a non-timed signal, and the dynamic AoS dimension of a timed signal,
	 *       are driven by setTimeAxisOffset()/extendDataSpaceForTimeSlices() instead.
	 */
	void setCurrentShapes(int *size, std::vector<int> &AOSShapes);
	/**
	 * @brief Pushes largest_dims to the file with H5Dset_extent() and refreshes the dataspace.
	 * @throws ALBackendException when the dataset has no extent (rank 0) or the resize fails.
	 */
    void setExtent();
    /**
     * @brief Snapshots the current dims as the reference for this slice's offsets.
     * @pre The handler is in slice mode.
     * @throws ALBackendException outside slice mode.
     */
    void storeInitialDims();
	/**
	 * @brief Grows the stored extent to make room for the time steps this slice appends.
	 * @param size                        Array shape of the request, in AL order.
	 * @param AOSShapes                   Current extent of each AoS dimension.
	 * @param dynamic_AOS_slices_extension Steps added to a timed AoS by this slice.
	 * @note Applies to the first array dimension for a numeric, non-timed signal and to the
	 *       dynamic AoS dimension for a timed one; shape datasets and strings take the timed
	 *       branch only. When nothing matches, the extent is left as it is.
	 * @pre The handler is in slice mode.
	 * @throws ALBackendException outside slice mode.
	 */
	void extendDataSpaceForTimeSlices(int *size, std::vector<int> &AOSShapes, int dynamic_AOS_slices_extension);
	/**
	 * @brief Records where this slice's data starts and moves dims along the time axis.
	 * @param current_arrctx_indices     Index of each open AoS level, outermost first.
	 * @param dynamic_AOS_slices_extension Steps added to a timed AoS (0 selects the
	 *                                     request-sized branch).
	 * @note For a numeric, non-timed signal: timeWriteOffset = initial_dims on the first
	 *       array dimension, and that dimension advances by the appended step count. For a
	 *       timed signal: the dynamic AoS dimension advances to initial_dims + slice + 1.
	 * @pre The handler is in slice mode with an open dataset.
	 * @throws ALBackendException outside slice mode or with no dataset open.
	 */
	void setTimeAxisOffset(const std::vector < int > &current_arrctx_indices, int dynamic_AOS_slices_extension);
    /**
     * @brief Re-applies the timed-AoS time advance after the slice index moved.
     * @param current_arrctx_indices Index of each open AoS level, outermost first.
     * @pre The handler is in slice mode with an open dataset.
     * @throws ALBackendException outside slice mode or with no dataset open.
     */
    void updateTimeAxisOffset(const std::vector < int > &current_arrctx_indices);

	//AOS Dataset management -- the same operations for the *_AOS_SHAPE datasets, which
	//record the AoS extents themselves and therefore have no array dimensions of their own.
	/**
	 * @brief AoS-shape variant of setCurrentShapesAndExtend(): adopt shapes, then setExtent().
	 * @param size      Array shape of the request, in AL order.
	 * @param AOSShapes Current extent of each AoS dimension.
	 */
	void setCurrentShapesAndExtendForAOSDataSet(int *size, std::vector<int> &AOSShapes);
	/**
	 * @brief AoS-shape variant of setCurrentShapes(); every AoS axis is updatable here.
	 * @param size      Array shape of the request, in AL order.
	 * @param AOSShapes Current extent of each AoS dimension.
	 */
	void setCurrentShapesForAOSDataSet(int *size, std::vector<int> &AOSShapes);
    /**
     * @brief AoS-shape variant of extendDataSpaceForTimeSlices(): grows the dynamic AoS axis only.
     * @param size                         Array shape of the request, in AL order.
     * @param AOSShapes                    Current extent of each AoS dimension.
     * @param dynamic_AOS_slices_extension Steps added to the timed AoS by this slice.
     * @pre The handler is in slice mode.
     * @throws ALBackendException outside slice mode.
     */
    void extendDataSpaceForTimeSlicesForAOSDataSet(int *size, std::vector<int> &AOSShapes, int dynamic_AOS_slices_extension);
	/**
	 * @brief AoS-shape variant of setTimeAxisOffset(): advances the dynamic AoS axis by one.
	 * @pre The handler is in slice mode with an open dataset.
	 * @throws ALBackendException outside slice mode or with no dataset open.
	 */
	void setTimeAxisOffsetForAOSDataSet();
    /**
     * @brief Records the slice count of the dynamic AoS for this operation.
     * @param timedAOS_shape Number of slices written so far.
     * @pre The handler is in slice mode.
     * @throws ALBackendException outside slice mode.
     */
    void setTimedAOSShape(int timedAOS_shape);
    /**
     * @brief Slice count of the dynamic AoS recorded for this operation.
     * @pre The handler is in slice mode.
     * @throws ALBackendException outside slice mode.
     */
    int getTimedAOSShape() const;
    /**
     * @brief Element offset along the first array dimension where this slice's data starts.
     * @pre The handler is in slice mode.
     * @throws ALBackendException outside slice mode.
     */
    int getTimeWriteOffset() const;

	/** @brief The dataset's tensorized name (IMAS path with '/' flattened to '&'). */
	std::string getName() const;
	/** @brief Time extent of the incoming request, i.e. the steps appended along the first array dimension. */
	int getSlicesExtension() const;
	/** @brief True for a *_AOS_SHAPE bookkeeping dataset. */
	bool isShapeDataset() const;
    /** @brief Total dataset rank (AoS dimensions + array dimensions). */
    int getRank() const;
    /** @brief Number of leading AoS dimensions. */
    int getAOSRank() const;
    /** @brief Current dims as tracked by this handler (AoS dimensions first). */
    const hsize_t * getDims();
    /**
     * @brief Current dims as a vector; empty for a node whose data axis is absent
     *        (a numeric scalar, or a string with no list axis).
     */
    std::vector<int> getDimsAsVector() const;
    /**
     * @brief Product of the `request_dim` innermost entries of `v` (the element count of one instance).
     * @param v Dimensions in stored order (innermost first); read back-to-front.
     */
    int computeShapeFromDimsVector(std::vector<int> &v);
    /** @brief Highest extent reached per dimension: the layout flat-index lookups must use. */
    hsize_t * getLargestDims();
    // Declared for symmetry with the other offset accessors; no definition exists in
    // hdf5_dataset_handler.cpp and no caller uses it.
    int getInitialOffset() const;
	/** @brief Element count over largest_dims (the size of a whole-dataset buffer). */
	int getSize() const;
    /**
     * @brief Maximum extent of one axis.
     * @param axis 0-based axis index.
     * @throws ALBackendException when `axis` is out of range.
     */
    size_t getMaxShape(int axis) const;
	/**
	 * @brief Current extent of one axis.
	 * @param axis 0-based axis index.
	 * @throws ALBackendException when `axis` is out of range.
	 */
	int getShape(int axis) const;
    /** @brief The handler's current dataspace handle (not a copy; do not close). */
    hid_t getDataSpace() const;
    /**
     * @brief Switches out of slice mode: no time axis, whole-node semantics.
     */
    void setNonSliceMode();
    /**
     * @brief Switches into slice mode, resolving the dynamic AoS level from the context.
     * @param ctx Current context; queried through HDF5Utils::isTimed().
     */
    void setSliceMode(Context * ctx);
    /**
     * @brief Current extent of the dynamic AoS axis, if there is one.
     * @param timed_AOS_index_ [out] Level index of the dynamic AoS (untouched when there is none).
     * @return That axis extent, or -1 when the handler has no timed AoS.
     */
    int getTimedShape(int *timed_AOS_index_);
    /**
     * @brief Renames the dataset this handler describes (used when the tensorized path is
     *        computed after the handler is created).
     * @param p New tensorized path.
     */
    void setTensorizedPath(std::string p) {
        tensorized_path = p;
    }
    /**
     * @brief Frees the buffered/cached values this handler owns and closes the dataset.
     * @throws ALBackendException when H5Dclose fails.
     */
    void close();
    /**
     * @brief True when every AoS index of the request exists in the stored extent.
     * @param current_arrctx_indices Index of each open AoS level, outermost first.
     * @return True for an empty request, or when all indices are within largest_dims.
     */
    bool isRequestInExtent(const std::vector < int >&current_arrctx_indices);
    /**
     * @brief Flushes the buffered writes: merges them into a whole-dataset image and writes it.
     * @note A no-op when buffering is off. fillFullBuffers() reads the current dataset,
     *       scatters the queued values at their flat indices, then one H5Dwrite with
     *       H5S_ALL replaces the whole dataset.
     * @throws ALBackendException when H5Dwrite fails.
     */
    void write_buffers();
    /**
     * @brief Builds the whole-dataset image from the current file content plus the queued values.
     * @note Frees each queued per-request buffer as it is merged. Numeric and string paths
     *       differ: numerics are scattered over a raw block, strings over a pointer array
     *       pre-filled with empty strings.
     * @throws ALBackendException when the counts disagree or the initial read fails.
     */
    void fillFullBuffers() ;

    /**
     * @brief Writes one instance's values directly through a hyperslab (no buffering).
     * @param current_arrctx_indices     Index of each open AoS level, outermost first.
     * @param slice_mode                 GLOBAL_OP (whole node) or SLICE_OP (append a slice).
     * @param dynamic_AOS_slices_extension Steps the dynamic axis grew by.
     * @param data                       The elements to write.
     * @throws ALBackendException when H5Dwrite fails.
     */
    void writeUsingHyperslabs(const std::vector < int >&current_arrctx_indices, int slice_mode, int dynamic_AOS_slices_extension, void *data);
    /**
     * @brief Records a write, buffering it when the request kind allows, else writing it now.
     * @param current_arrctx_indices AoS indices of this request (always recorded).
     * @param dataSetAlreadyOpened   Whether the dataset was already open for this context.
     * @param datatype               AL datatype of the values.
     * @param dim                    Requested dimensionality (0 or 1 are the buffered kinds).
     * @param slice_mode             GLOBAL_OP or SLICE_OP (slice writes are never buffered).
     * @param dynamic_AOS_slices_extension Steps the dynamic axis grew by (hyperslab path only).
     * @param p                      For char data: pointer to the string value.
     * @param data                   The values (numeric path).
     */
    void appendToBuffer(const std::vector < int >&current_arrctx_indices, bool dataSetAlreadyOpened, int datatype, int dim, int slice_mode, int dynamic_AOS_slices_extension, char**p, void *data);

    //Reading operation
    /**
     * @brief Reads one request's values, choosing between the whole-dataset cache and a hyperslab.
     * @param current_arrctx_indices Index of each open AoS level, outermost first.
     * @param datatype               AL datatype of the values to return.
     * @param dim                    Requested dimensionality (0 or 1 select the cached path).
     * @param slice_mode             GLOBAL_OP or SLICE_OP (slice reads always use a hyperslab).
     * @param is_dynamic             Whether the time axis is one of the array dimensions.
     * @param isTimed                Whether an enclosing dynamic AoS supplies the time axis.
     * @param timed_AOS_index        Dynamic AoS level index, or -1.
     * @param slice_index            Time index to read (-1 when not slicing).
     * @param data                   [out] Newly allocated buffer (caller frees).
     * @note An enabled time range forces the hyperslab path: the whole-dataset cache cannot
     *       express a window. Otherwise the first read of a kind loads the dataset once and
     *       every later instance of the same dataset is served from memory.
     * @throws ALBackendException when the underlying read fails.
     */
    void readData(const std::vector < int >&current_arrctx_indices, int datatype, int dim, int slice_mode, bool is_dynamic, bool isTimed, int timed_AOS_index, int slice_index, void **data);

};

#endif
