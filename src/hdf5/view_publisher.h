// view_publisher.h
// Builds a user-friendly Virtual Dataset (VDS) view on top of a PanzerDB file,
// so standard HDF5 tools (h5ls, h5dump, h5py) can inspect the logical IMAS tree
// without any IMAS-API dependency.
//
// Design (snapshot only — the VDS mapping is static):
//   * One VDS per data leaf at <view_root>/<full-instance-path>.
//       - source = data_raw_*[offset : offset+count] (1 contiguous hyperslab)
//       - dims   = leaf.shape (or [time_dim] + leaf.shape if count > prod(shape))
//   * Intermediate groups are auto-created along each leaf path.
//   * AoS groups (all-integer children) get a @aos_size attribute.
//   * Each VDS gets @datatype, @n_elements, @time_dim, @schema attributes.
//
// No PanzerDB data file (index/paths/data_raw_*) is modified — the view is
// additive and isolated under <view_root>. Safe to re-run (idempotent: the
// entire <view_root> group is deleted and recreated).

#ifndef PANZER_VIEW_PUBLISHER_H
#define PANZER_VIEW_PUBLISHER_H

#include <string>
#include <stdexcept>
#include <vector>

namespace imas::view {

struct PublishStats {
    int n_data_leaves  = 0;   // number of VDS created
    int n_aos_groups   = 0;   // number of groups that received @aos_size
    int n_empty_skipped= 0;   // empty leaves (kind 1) not mirrored
    int n_errors       = 0;   // leaves that failed VDS creation (skipped)
    int n_total        = 0;   // total leaves visited
};

// Publish (idempotent) a VDS view under <root_group> (default "imas_view").
// <filename> is opened for R/W; the VDS source-filename baked into each
// dataset definition is the given <filename> string (HDF5 VDS portability
// caveat: if the file is later renamed/moved, the VDS source reference will
// no longer resolve). The PanzerDB data itself (index/paths/data_raw_*) is
// NOT modified — the operation is purely additive and safe to re-run.
PublishStats publish(const std::string& filename,
                     const std::string& root_group = "imas_view");

}  // namespace imas::view

#endif // PANZER_VIEW_PUBLISHER_H
