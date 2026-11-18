// view_publisher.h
// Builds a user-friendly Virtual Dataset (VDS) view on top of a PanzerDB file,
// so standard HDF5 tools (h5ls, h5dump, h5py) can inspect the logical IMAS tree
// without any IMAS-API dependency.
//
// Design (snapshot only — the VDS mapping is static):
//   * ONE VDS PER SCHEMA PATH (collapse mode, the default).  The Array-of-
//     Structures levels and the time steps are NOT listed as separate HDF5
//     objects: they become LEADING DIMENSIONS of the aggregated dataset.
//       dims = [n_time?, n_aos_0, ..., n_aos_k-1, *leaf_slice_shape]
//     The enclosing AoS levels come first, outermost to innermost; a leading
//     time dimension is added when one logical path is stored as several index
//     rows (one row per time slice).  Every level is a dim, including one of
//     size 1, and a level's length is its DECLARED AoS size (@aos_size) rather
//     than the largest instance index that happens to carry data: a level
//     declared at 15 402 yields a dim of 15 402 even where only 7 968 instances
//     hold a value.  `h5ls` prints no attribute, so the shape it shows is what
//     makes each level's size visible, and it agrees with @aos_size by
//     construction.
//     Unmapped cells (ragged slices, sparse AoS instances) keep the HDF5 fill
//     value, set to the IMAS invalid value: -9E+40 for reals (both components
//     of a complex), -999999 for integers, empty string for text.
//     Each AoS group carries @aos_size (@aos_dynamic=1 when it grows with
//     time); each dataset carries @datatype, @n_elements, @schema, @aos_ndim
//     (how many leading dims are AoS/time) and @dims_label (the name of every
//     dim, the leaf's own dims named last, so [195,10] reads as
//     time_slice|constraints&faraday_angle and @aos_ndim=1 says the leaf shape
//     starts at dim 1).  listView prints the split itself: {195 | 10}.
//     Object names never contain '[' — the HDF5 command line parses brackets
//     as a selection, which makes such an object unreachable by h5dump -d.
//   * --expand (collapse_aos=false) restores the historical view: one VDS per
//     index row, every AoS instance listed under its own subgroup.
//   * Intermediate groups are auto-created along each path — unless
//     no_aos_groups (CLI --no-aos-groups) is set, which publishes every dataset
//     flat under <view_root> under one name whose path segments are joined with
//     ':' (imas_view/grids_ggd:grid:space:objects_per_dimension:object:geometry).
//     A recursive h5ls then shows no AoS group at all, while each level's size
//     stays visible in the shape.  @aos_size/@aos_dynamic belong to the groups,
//     so they are not written in that mode (n_aos_groups stays 0).
//   * listView() reads a published view back and prints one annotated line per
//     object — needed because `h5ls -r` prints NO attribute at all, so the
//     annotations above are invisible in a plain listing.
//
// No PanzerDB data file (index/paths/data_raw_*) is modified — the view is
// additive and isolated under <view_root>. Safe to re-run (idempotent: the
// entire <view_root> group is deleted and recreated).

#ifndef PANZER_VIEW_PUBLISHER_H
#define PANZER_VIEW_PUBLISHER_H

#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

namespace imas::view {

struct PublishStats {
    int n_data_leaves  = 0;   // VDS created (index rows in expand mode, schema paths in collapse mode)
    int n_aos_groups   = 0;   // number of groups that received @aos_size
    int n_empty_skipped= 0;   // empty leaves (kind 1) not mirrored
    int n_errors       = 0;   // leaves that failed VDS creation (skipped)
    int n_total        = 0;   // total leaves visited

    // Collapse-mode gauges (see the --stats option of imas_publish_view):
    int      n_aggregated_datasets       = 0;  // schema paths published
    uint64_t n_mappings                  = 0;  // total H5Pset_virtual calls (DCPL cost driver)
    uint64_t max_mappings_in_one_dataset = 0;  // worst single dataset
    uint64_t n_fill_holes                = 0;  // target cells left to the fill value
    int      n_rank_overflow             = 0;  // datasets skipped: rank > H5S_MAX_RANK
    int      n_mapping_cap_fallback      = 0;  // datasets reduced to the first slice
};

struct PublishOptions {
    std::string root_group = "imas_view";
    bool     collapse_aos = true;    // false == historical per-index-row view
    // Safety valve: when a dataset would need more mappings than this, publish
    // only its first slice and flag it (@mapping_cap=1).  0 == unlimited.
    uint64_t max_mappings_per_dataset = 0;
    // Collapse mode only: publish the datasets flat under <view_root> with ':'
    // joining the path segments, instead of creating one HDF5 group per AoS
    // level.  No effect in --expand mode.
    bool     no_aos_groups = false;
};

// Publish (idempotent) a VDS view under <root_group> (default "imas_view").
// <filename> is opened for R/W; the VDS source-filename baked into each
// dataset definition is the given <filename> string (HDF5 VDS portability
// caveat: if the file is later renamed/moved, the VDS source reference will
// no longer resolve). The PanzerDB data itself (index/paths/data_raw_*) is
// NOT modified — the operation is purely additive and safe to re-run.
PublishStats publish(const std::string& filename,
                     const std::string& root_group = "imas_view");

// Same, with the full option set (collapse mode, mapping cap).
PublishStats publish(const std::string& filename, const PublishOptions& options);

// Read-only listing of an already published view: one line per HDF5 object,
// carrying the annotations that h5ls -r never prints (it shows no attribute at
// all, whatever the HDF5 version).  Nothing is written — the file is opened
// H5F_ACC_RDONLY and the PanzerDB tables are not touched.
//   dataset: /imas_view/<path>  Dataset {195, 519}  @dims_label=...  @ragged  holes=2
//   group:   /imas_view/<path>  Group  @aos_size=195  @aos_dynamic
// `holes` is derived (product of the dims minus @n_elements), so it needs no
// extra attribute and is printed only when non-zero.  Attributes a view does
// not carry (--expand has no @dims_label, @aos_ndim or @n_mappings) are simply
// omitted.  Order is tree order: parent first, children in link-name order.
struct ListViewResult {
    bool found = false;                 // false == <root_group> absent (not an error)
    std::vector<std::string> lines;     // aligned, one entry per object
};

ListViewResult listView(const std::string& filename,
                        const std::string& root_group = "imas_view");

}  // namespace imas::view

#endif // PANZER_VIEW_PUBLISHER_H
