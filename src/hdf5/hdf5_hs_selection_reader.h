/**
 * @file hdf5_hs_selection_reader.h
 * @brief Hyperslab selection and output-buffer sizing for v1 reads.
 *
 * Companion of HDF5HsSelectionWriter on the read side: given one v1 dataset (whose
 * leading dimensions are the open AoS levels and whose trailing dimensions are the
 * array), it derives the logical shape the AL expects, sizes the output buffer, and
 * selects the slab matching either the whole node (GLOBAL_OP), one time slice
 * (SLICE_OP), or a requested time range. The v2/PanzerDB read path does not use it.
 */
#ifndef HDF5_HSSELECTIONREADER_H
#define HDF5_HSSELECTIONREADER_H 1

#include <hdf5.h>
#include "al_backend.h"

#include <memory>
#include <vector>
#include <list>
#include <map>

/**
 * @struct TimeRange
 * @brief A requested time window applied along the time axis of a read.
 *
 * Filled by the reader before setHyperSlabs(): when `enabled`, the selection starts
 * at `tmin_index` instead of 0 (only when the time axis is a dataset dimension, i.e.
 * there is no enclosing dynamic AoS — see setHyperSlabs).
 */
struct TimeRange {
    bool enabled;                 // False -> no time filtering.
    std::vector<double> dtime;    // Requested time values; non-empty is the precondition the reader checks.
    int tmin_index;               // First time index inside the window.
    int tmax_index;               // One past the last time index inside the window.
};

/**
 * @class HDF5HsSelectionReader
 * @brief Shapes, sizes and selects the data a v1 read pulls out of one dataset.
 *
 * @note Dimension order: `size` is in AL/IMAS order (outermost first) while the
 *       dataset stores the innermost dimension first. init() maps them with
 *       size[dim-1-i] = dataspace_dims[AOSRank + i], so the dataset's first array
 *       dimension is the LAST output dimension. Keep that reversal in mind when
 *       indexing either array.
 */
class HDF5HsSelectionReader {
  private:
    hid_t dataset_id;         // Dataset this selector was built on (not owned).
    bool immutable;           // True when dtype_id is a builtin native type (never closed); false when it came from H5Dget_type and must be closed.
    int datatype;             // AL data type (integer/double/complex/char) driving type and sizing rules.
    int dataset_rank;         // Total dataset rank = AoS dimensions + array dimensions.

    int AOSRank;              // Number of leading AoS dimensions (0 when the signal is outside any AoS).

    hsize_t size[MAXDIM];     // Logical output shape in AL order (see the class note on the reversal).
    int dim;                  // Number of meaningful entries in `size`.

    hsize_t dataspace_dims[H5S_MAX_RANK];   // Current extent of every dataset dimension.
    hsize_t offset[H5S_MAX_RANK];           // File-side hyperslab start.
    hsize_t count[H5S_MAX_RANK];            // File-side hyperslab extent.
    hsize_t memspace_dims_copy[H5S_MAX_RANK];   // Dimensions the cached `memspace` was built with.

    hsize_t offset_out[H5S_MAX_RANK];       // Memory-side hyperslab start.
    hsize_t count_out[H5S_MAX_RANK];        // Memory-side hyperslab extent.

    /**
     * @brief Recomputes buffer_size from the current `size` and the element size.
     * Numeric: element size times the product of all array dimensions. Strings:
     * element size times the number of strings (size[0]).
     */
    void setBufferSize();
    /**
     * @brief Resolves the HDF5 type id/size and derives `dim` + `size` from the dataset.
     * @param dataset_id  Dataset handle (queried for its type for complex/char data).
     * @param datatype_   AL data type selector (the member value is what is switched on).
     * @param dim         [out] Number of output dimensions the caller gets back.
     * @note Strings are reported as dim 1 (a scalar string, dataset_rank == AOSRank) or
     *       dim 2 (a list: size[0] = number of strings, size[1] = stored string width).
     */
    void init(hid_t dataset_id, int datatype_, int *dim);
    /**
     * @brief True when `dims` differs from the dimensions the cached memspace was built with.
     * @param dims Candidate memory dimensions; the first dataset_rank entries are compared.
     */
    bool memSpaceHasChanged(hsize_t * dims);

  public:
    /**
     * @brief Builds a selector over an open dataset.
     * @param dataset_rank_ Total dataset rank (AoS dimensions + array dimensions).
     * @param dataset_id_   Open dataset handle (not owned).
     * @param dataspace_    Dataset dataspace; selected in place by setHyperSlabs().
     * @param dims          Current extent of each dataset dimension.
     * @param datatype      AL data type of the caller's request.
     * @param AOSRank_      Number of leading AoS dimensions.
     * @param dim           [out] Output dimensionality derived from the dataset (see init()).
     */
     HDF5HsSelectionReader(int dataset_rank_, hid_t dataset_id_, hid_t dataspace_, hsize_t *dims, int datatype, int AOSRank_, int *dim);
    /**
     * @brief Closes the memory dataspace and, for complex/char data, the borrowed type id.
     */
    ~HDF5HsSelectionReader();

    hid_t dtype_id;            // HDF5 type used for the read (builtin native, or the dataset's own type).
    size_t dtype_size;         // Size in bytes of one element of dtype_id (string width for char data).
    hid_t dataspace;           // File dataspace the selection is applied to (H5S_ALL for rank 0).
    hid_t memspace;            // Memory dataspace matching the last selection (owned; -1 until one is built).
    size_t buffer_size;        // Bytes a full-shape read of this node needs.
    TimeRange time_range;      // Time window to apply, when enabled.

    /**
     * @brief Total element count over the whole dataset (all dimensions multiplied).
     * @return Element count, ignoring the caller's selection.
     */
    size_t getSize2();
    /**
     * @brief Overrides the logical output shape (e.g. after a shape check) and resizes the buffer.
     * @param size_ New shape in AL order.
     * @param dim   Number of entries to copy from size_.
     */
    void setSize(int *size_, int dim);
    /**
     * @brief Shrinks one output dimension to a time range and resizes the buffer.
     * @param dim       1-based dimensionality whose LAST entry is the time axis.
     * @param timeRange Number of time steps the read will return.
     */
    void setTimeRange(int dim, int timeRange);
    /**
     * @brief Number of output dimensions (0 for a pure scalar).
     */
    int getDim() const;
    /**
     * @brief Total dataset rank (AoS dimensions + array dimensions).
     */
    int getRank() const;
    /**
     * @brief Fills `size` with the shape this read returns.
     * @param size       [out] Shape in AL order.
     * @param slice_mode GLOBAL_OP returns the full shape; SLICE_OP returns it with the
     *                   last dimension reduced to 1.
     * @param is_dynamic Whether the node carries a time dimension.
     */
    void getSize(int *size, int slice_mode, bool is_dynamic) const;
    /**
     * @brief The dataset's current per-dimension extents (AoS dimensions first).
     * @return Pointer to the internal array (valid for this object's lifetime).
     */
    const hsize_t* getDataSpaceDims();
    /**
     * @brief Extent of one dataset dimension.
     * @param axis_index 0-based dataset dimension index (AoS dimensions included).
     */
    int getShape(int axis_index) const;
    /**
     * @brief True when every AoS index exists in the dataset (no out-of-range request).
     * @param current_arrctx_indices Index of each open AoS level, outermost first.
     * @return True when all indices are within the stored extent (an empty request is
     *         trivially in extent).
     */
    bool isRequestInExtent(const std::vector < int >&current_arrctx_indices);
    /**
     * @brief Allocates a buffer for a whole-node (GLOBAL_OP) read.
     * @param data [out] Newly allocated buffer (caller frees).
     */
    void allocateGlobalOpBuffer(void **data);
    /**
     * @brief Allocates the buffer a read of the given mode needs.
     * @param data       [out] Newly allocated buffer (caller frees).
     * @param slice_mode GLOBAL_OP (whole node) or SLICE_OP (one time step).
     * @param is_dynamic Whether the node carries a time dimension as a dataset dimension.
     * @param isTimed    Whether an enclosing dynamic AoS supplies the time axis instead.
     * @param slice_index Requested slice index (-1 when not slicing).
     * @return The allocated size in bytes.
     * @note For a signal whose time axis is a dataset dimension (dynamic, not timed),
     *       a slice read divides the full-shape size by the time extent.
     * @note The char branch assigns the allocation to the local parameter instead of
     *       `*data`, so no buffer reaches the caller through it; string reads go
     *       through the buffered path in HDF5DataSetHandler.
     * @throws ALBackendException when the allocation fails.
     */
    int allocateBuffer(void **data, int slice_mode, bool is_dynamic, bool isTimed, int slice_index);
    /**
     * @brief Allocates a buffer big enough for the entire dataset, AoS dimensions included.
     * @param data [out] Newly allocated buffer (caller frees).
     * @return The number of elements the buffer holds.
     */
    int allocateFullBuffer(void **data);
    /**
     * @brief Allocates a per-instance time base for an IDS with inhomogeneous time.
     * @param data            [out] Newly allocated buffer (caller frees).
     * @param timed_AOS_index The dynamic AoS level whose extent counts the instances;
     *                        -1 falls back to a whole-node buffer.
     */
    void allocateInhomogeneousTimeDataSet(void **data, int timed_AOS_index);
    /**
     * @brief Selects the whole node (GLOBAL_OP convenience wrapper around setHyperSlabs).
     * @param current_arrctx_indices Index of each open AoS level, outermost first.
     * @param timed_AOS_index        Dynamic AoS level index, or -1.
     * @param count_along_dynamic_aos Read the entire dynamic axis at once
     *                               (used for the per-instance time bases).
     */
    void setHyperSlabsGlobalOp(std::vector < int >current_arrctx_indices, int timed_AOS_index = -1, bool count_along_dynamic_aos = false);
    /**
     * @brief Selects the slab this read must return, in both the file and the memory dataspace.
     * @param slice_mode       GLOBAL_OP (whole node) or SLICE_OP (one time step).
     * @param is_dynamic       Whether the time axis is one of the dataset's array dimensions.
     * @param isTimed          Whether an enclosing dynamic AoS supplies the time axis.
     * @param slice_index      Time index to read; substituted for the dynamic AoS level's
     *                         index when timed, and used along the first array dimension
     *                         when the time axis is a dataset dimension.
     * @param timed_AOS_index  AoS level index of the dynamic level, or -1.
     * @param current_arrctx_indices Index of each open AoS level, outermost first.
     * @param count_along_dynamic_aos Take the whole dynamic axis (count = its extent)
     *                                instead of one instance — reads the per-instance
     *                                time base of an inhomogeneous-time IDS in one go.
     * @note A time range (time_range.enabled with a non-empty dtime) offsets the first
     *       array dimension by tmin_index only when there is no timed AoS; with a timed
     *       AoS the slice index moves along the AoS dimension instead.
     * @note Counts are clamped to the stored extent, so a request beyond what was
     *       written reads what exists rather than failing.
     * @note A rank-0 dataset selects H5S_ALL for both spaces; the memory dataspace is
     *       reused while its dimensions are unchanged, and declares H5S_UNLIMITED for
     *       zero-length dimensions (HDF5 workaround).
     * @throws ALBackendException when either H5Sselect_hyperslab call fails.
     */
    void setHyperSlabs(int slice_mode, bool is_dynamic, bool isTimed, int slice_index, int timed_AOS_index, std::vector < int >current_arrctx_indices, bool count_along_dynamic_aos = false);
};

#endif
