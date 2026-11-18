/**
 * @file hdf5_hs_selection_writer.h
 * @brief Builds the file-side and memory-side hyperslab selection for a v1 write.
 *
 * In the v1 layout a signal lives in ONE extensible HDF5 dataset whose leading
 * dimensions are the open AoS levels and whose trailing dimensions are the array
 * itself (plus the growing time axis for a dynamic AoS). A write therefore selects
 * a thin slab: count 1 along each AoS dimension, the full extent along the array
 * dimensions. This class computes that pair of selections (file dataspace + a
 * matching memory dataspace) and caches the memory dataspace across calls.
 * The v2/PanzerDB path does not use it: it appends to flat columnar datasets.
 */
#ifndef HDF5_HSSELECTIONWRITER_H
#define HDF5_HSSELECTIONWRITER_H 1

#include <hdf5.h>
#include <vector>
#include <string>
#include "hdf5_dataset_handler.h"


/**
 * @class HDF5HsSelectionWriter
 * @brief Hyperslab selector for writing one AoS instance of one slice into a v1 dataset.
 *
 * The memory dataspace is reused whenever its dimensions are unchanged
 * (memSpaceHasChanged), which is the common case: only the file-side offset moves
 * from one write to the next.
 */
class HDF5HsSelectionWriter {
  private:

    int dataset_rank;   // Rank in effect for the last setHyperSlabs() (-1 before any).

    hsize_t offset[H5S_MAX_RANK];      // File-side hyperslab start, per dimension.
    hsize_t count[H5S_MAX_RANK];       // File-side hyperslab extent, per dimension.

    hsize_t dims[H5S_MAX_RANK];        // Memory dataspace dimensions created for the last selection.
    hsize_t offset_out[H5S_MAX_RANK];  // Memory-side hyperslab start (always 0: the buffer is packed).
    hsize_t count_out[H5S_MAX_RANK];   // Memory-side hyperslab extent.

    hsize_t memspace_dims_copy[H5S_MAX_RANK];   // Dimensions the current `memspace` was built with (reuse test).

    /**
     * @brief True when `dims` differs from the dimensions the cached memspace was built with.
     * @param dims Candidate memory dimensions; the first dataset_rank entries are compared.
     * @return True when a new memory dataspace must be created.
     */
    bool memSpaceHasChanged(hsize_t * dims);

  public:

    /**
     * @brief Creates a selector with no memory dataspace yet (memspace == -1).
     */
     HDF5HsSelectionWriter();
    /**
     * @brief Closes the cached memory dataspace, if one was created.
     */
    ~HDF5HsSelectionWriter();

    /**
     * @brief The memory dataspace matching the last selection (owned by this object).
     * H5S_ALL for a rank-0 dataset; otherwise a simple dataspace of rank dataset_rank,
     * reused across calls until its dimensions change. Closed by the destructor.
     */
    hid_t memspace;

    /**
     * @brief Selects, in one dataset, the slab holding the current AoS instance of a write.
     * @param dataset_id                   Dataset being written (reported in error messages).
     * @param current_arrctx_indices       Index of each open AoS level, outermost first;
     *                                     its size is the number of AoS dimensions.
     * @param slice_mode                   SLICE_OP when the operation appends a time slice,
     *                                     GLOBAL_OP when it rewrites the whole node.
     * @param dataspace                    File dataspace to select in (modified in place).
     * @param rank                         Total dataset rank = AoS levels + array dimensions.
     * @param isShapeDataSet               True for the companion *_AOS_SHAPE datasets, whose
     *                                     array dimensions are never sliced.
     * @param isTimed                      True when an enclosing dynamic (time-evolving) AoS exists.
     * @param timed_AOS_index              AoS level index of that dynamic level, or -1.
     * @param timeWriteOffset              Element offset of the new slice along the first array
     *                                     dimension (used only when slicing OUTSIDE a dynamic AoS).
     * @param dataspace_dims               Current extent of every dataset dimension.
     * @param dynamic_AOS_slices_extension Number of slices the dynamic axis was extended by
     *                                     (accepted for interface symmetry; the offset comes
     *                                     through timeWriteOffset).
     * @note Selection rules: every AoS dimension gets count 1 at its instance index, except the
     *       dynamic level under SLICE_OP, which selects the slot just appended
     *       (dataspace_dims - 1). Array dimensions are taken whole, except the first one when
     *       SLICE_OP appends outside a dynamic AoS, which starts at `timeWriteOffset`.
     * @note A rank-0 dataset selects H5S_ALL and creates no memory dataspace.
     * @note The memory dataspace declares H5S_UNLIMITED as the maximum wherever a dimension is
     *       0 — a workaround for HDF5 rejecting zero-length dimensions.
     * @throws ALBackendException when either H5Sselect_hyperslab call fails.
     */
    void setHyperSlabs(hid_t dataset_id, const std::vector < int >&current_arrctx_indices, int slice_mode, hid_t dataspace, int rank, bool isShapeDataSet, bool isTimed, int timed_AOS_index, int timeWriteOffset, const hsize_t *dataspace_dims, int dynamic_AOS_slices_extension);

};

#endif
