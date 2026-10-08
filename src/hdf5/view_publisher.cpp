// view_publisher.cpp
// Builds a user-friendly VDS view on a PanzerDB snapshot file.
// Design: see view_publisher.h.

#include "view_publisher.h"
#include "panzerdb.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace imas::view {
namespace {

// --------------------------------------------------------------------------
// Small helpers
// --------------------------------------------------------------------------

[[noreturn]] void h5err(const std::string& what) {
    throw std::runtime_error("view_publisher: " + what);
}

std::vector<std::string> splitPath(const std::string& p) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : p) {
        if (c == '/') { if (!cur.empty()) { out.push_back(cur); cur.clear(); } }
        else cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

bool isAllDigits(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    return true;
}

uint64_t productShape(const std::vector<size_t>& shape) {
    uint64_t p = 1;
    for (auto d : shape) p *= (d ? d : 1);
    return p;
}

// Ensure a chain of groups (relative to loc) exists.  Returns an OPEN handle
// to the leaf-most group (which the caller must H5Gclose).  If the segment
// list is empty, returns loc as-is WITHOUT opening a new handle — the caller
// should treat out_gid == loc as "no close needed".
hid_t ensureGroups(hid_t loc, const std::vector<std::string>& segs) {
    if (segs.empty()) return loc;
    hid_t cur = loc;
    for (const auto& s : segs) {
        hid_t nxt;
        if (H5Lexists(cur, s.c_str(), H5P_DEFAULT) > 0)
            nxt = H5Gopen2(cur, s.c_str(), H5P_DEFAULT);
        else
            nxt = H5Gcreate2(cur, s.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (nxt < 0) {
            if (cur != loc) H5Gclose(cur);
            h5err("ensureGroups failed at segment: " + s);
        }
        if (cur != loc) H5Gclose(cur);
        cur = nxt;
    }
    return cur;  // OPEN — caller closes
}

void setStrAttr(hid_t obj, const std::string& name, const std::string& v) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t t  = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, v.size() + 1);
    hid_t a  = H5Acreate2(obj, name.c_str(), t, sp, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, t, v.c_str());
    H5Aclose(a);
    H5Sclose(sp);
    H5Tclose(t);
}

// Store a 64-bit integer as a double attribute (8 bytes, portable HDF5 type,
// exact for values < 2^53 — always true here: n_elements, aos_size).
void setU64Attr(hid_t obj, const std::string& name, uint64_t v) {
    hid_t sp = H5Screate(H5S_SCALAR);
    const double dv = static_cast<double>(v);
    hid_t a  = H5Acreate2(obj, name.c_str(), H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &dv);
    H5Aclose(a);
    H5Sclose(sp);
}

// --------------------------------------------------------------------------
// VDS creation for one data leaf
// --------------------------------------------------------------------------

// Returns the data_raw_* dataset name for a given DataType (global PanzerDB).
const char* dataRawName(PanzerDB::DataType dt) {
    switch (dt) {
        case PanzerDB::DataType::FLOAT64:         return "data_raw_f64";
        case PanzerDB::DataType::INT32:           return "data_raw_i32";
        case PanzerDB::DataType::COMPLEX128:      return "data_raw_c128";
        case PanzerDB::DataType::STRING:
        case PanzerDB::DataType::LIST_OF_STRINGS: return "data_raw_str";
        case PanzerDB::DataType::STRING_CHUNKED:  return "data_raw_str";
        default: return nullptr;
    }
}

const char* dtypeString(PanzerDB::DataType dt) {
    switch (dt) {
        case PanzerDB::DataType::FLOAT64:         return "float64";
        case PanzerDB::DataType::INT32:           return "int32";
        case PanzerDB::DataType::COMPLEX128:      return "complex128";
        case PanzerDB::DataType::STRING:          return "string";
        case PanzerDB::DataType::LIST_OF_STRINGS: return "list_of_strings";
        case PanzerDB::DataType::STRING_CHUNKED:  return "string";
        default: return "unknown";
    }
}

std::vector<hsize_t> computeDims(uint64_t count, const std::vector<size_t>& shape) {
    if (shape.empty()) {
        if (count == 1) return {};               // scalar
        return {(hsize_t)count};
    }
    uint64_t prod = productShape(shape);
    if (count == prod) {
        bool all_one = true;
        for (auto d : shape) if (d != 1) all_one = false;
        if (all_one) return {};                  // [1,1,...,1] → scalar
        return std::vector<hsize_t>(shape.begin(), shape.end());
    }
    if (count > prod && (count % prod) == 0) {
        std::vector<hsize_t> out;
        out.push_back((hsize_t)(count / prod));  // leading time/slice dim
        for (auto d : shape) out.push_back((hsize_t)d);
        return out;
    }
    return {(hsize_t)count};
}

// Create a single VDS at <root_group>/<full_instance_path>.
// raw_prefix: sub-group path (e.g. "magnetics/" or "") that qualifies the
// data_raw_* dataset name, so it resolves correctly even when PanzerDB data
// lives inside a non-root group (discovered by PanzerDB at open time).
bool createVDS(hid_t file_id, hid_t view_root, const PanzerDB::Leaf& leaf,
               const std::string& source_filename,
               const std::string& raw_prefix) {
    std::string fullPath = std::string(leaf.path);
    auto segs = splitPath(fullPath);
    if (segs.empty()) return false;
    std::string name = segs.back();
    segs.pop_back();                             // segs = parent segments
    if (segs.empty() && (name == "index")) name = "index_node";  // collision guard

    if (leaf.count == 0) return false;           // nothing to map

    auto dt = static_cast<PanzerDB::DataType>(leaf.flags >> 4);
    const char* raw = dataRawName(dt);
    if (!raw) return false;

    // 1. Parent group
    hid_t parent = ensureGroups(view_root, segs);
    const bool parent_is_loc = (parent == view_root);

    // 2. Source dataset type
    //    raw_prefix is empty for root-level data, "magnetics/" etc. otherwise.
    std::string full_raw = raw_prefix + raw;
    hid_t src = H5Dopen2(file_id, full_raw.c_str(), H5P_DEFAULT);
    if (src < 0) { if (!parent_is_loc) H5Gclose(parent); return false; }
    hid_t src_type = H5Dget_type(src);
    H5Dclose(src);

    // 3. Target dataspace (the VDS shape)
    std::vector<hsize_t> dims = computeDims(leaf.count, leaf.shape);
    hid_t vspace;
    if (dims.empty()) vspace = H5Screate(H5S_SCALAR);
    else vspace = H5Screate_simple((int)dims.size(), dims.data(), nullptr);
    hid_t vsel = H5Scopy(vspace);

    // 4. Source selection: a single contiguous block [offset, offset+count)
    hsize_t src_extent[1] = { (hsize_t)(leaf.offset + leaf.count) };
    hid_t srcspace = H5Screate_simple(1, src_extent, nullptr);
    hid_t srcsel = H5Scopy(srcspace);
    hsize_t start[1] = { (hsize_t)leaf.offset };
    hsize_t stride[1] = { 1 };
    hsize_t cnt[1]    = { (hsize_t)leaf.count };
    hsize_t blk[1]    = { 1 };
    H5Sselect_hyperslab(srcsel, H5S_SELECT_SET, start, stride, cnt, blk);

    // 5. DCPL with VIRTUAL layout + the one source mapping
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(dcpl, H5D_VIRTUAL);
    H5Pset_virtual(dcpl, vsel, source_filename.c_str(), full_raw.c_str(), srcsel);

    // 6. Create the VDS dataset
    hid_t vds = H5Dcreate2(parent, name.c_str(), src_type, vspace,
                           H5P_DEFAULT, dcpl, H5P_DEFAULT);

    if (vds >= 0) {
        // 7. Attributes (metadata that h5ls cannot infer)
        setStrAttr(vds, "datatype", dtypeString(dt));
        setU64Attr(vds, "n_elements", leaf.count);
        uint64_t prod = productShape(leaf.shape);
        uint64_t tdim = leaf.shape.empty()
                     ? (leaf.count > 1 ? leaf.count : 1)
                     : (leaf.count >= prod && leaf.count % prod == 0 ? leaf.count / prod : 1);
        if (tdim > 1) setU64Attr(vds, "time_dim", tdim);
        setStrAttr(vds, "schema", PanzerDB::stripIndices(fullPath));
        H5Dclose(vds);
    }

    // 8. Cleanup temporary ids
    H5Pclose(dcpl);
    H5Sclose(srcsel);  H5Sclose(srcspace);
    H5Sclose(vsel);    H5Sclose(vspace);
    H5Tclose(src_type);
    if (!parent_is_loc) H5Gclose(parent);
    return vds >= 0;
}

// --------------------------------------------------------------------------
// Empty-node / AoS-group node creation
// --------------------------------------------------------------------------

void createNodeGroup(hid_t view_root, const std::vector<std::string>& segs) {
    if (segs.empty()) return;
    hid_t g = ensureGroups(view_root, segs);
    H5Gclose(g);
}

// --------------------------------------------------------------------------
// Main processing
// --------------------------------------------------------------------------

void processAll(hid_t file_id, hid_t view_root,
                const std::vector<PanzerDB::Leaf>& leaves,
                const std::string& source_filename,
                const std::string& raw_prefix,   // e.g. "magnetics/" or ""
                PublishStats& st)
{
    // Collect per-group max index (for @aos_size).  Key = relative group path
    // under view_root; value = max integer child index seen.
    std::map<std::string, uint64_t> aosMax;

    for (const auto& leaf : leaves) {
        st.n_total++;
        unsigned kind = leaf.flags & 0xFu;
        if (leaf.is_empty || (leaf.flags & 0xF) == 1) { st.n_empty_skipped++; continue; }

        std::string fullPath = std::string(leaf.path);
        auto segs = splitPath(fullPath);

        if (kind == 0) {
            if (createVDS(file_id, view_root, leaf, source_filename, raw_prefix)) st.n_data_leaves++;
            else st.n_errors++;
        } else {
            // kind 1 handled above (is_empty); kind 2/3 = AoS node
            if (segs.size() > 0) createNodeGroup(view_root, segs);
        }

        // Track the integer-children of every intermediate group (AoS size).
        // For a segs[i] that is a pure integer, its parent group
        // (view_root / segs[0..i-1]) is an AoS with size >= segs[i] + 1.
        if ((kind & 1u) == 0) {   // include data leaves and AoS leaves for the tree view
            for (size_t i = 0; i < segs.size(); ++i) {
                const auto& s = segs[i];
                if (!isAllDigits(s)) continue;
                // Parent = segs[0..i-1]; skip if parent is view_root itself
                // (an integer at the TOP level is not a real IMAS AoS)
                if (i == 0) continue;
                std::string key;
                for (size_t j = 0; j < i; ++j) { if (j) key += '/'; key += segs[j]; }
                uint64_t idx = strtoull(s.c_str(), nullptr, 10);
                auto it = aosMax.find(key);
                if (it == aosMax.end()) aosMax[key] = idx;
                else if (it->second < idx) it->second = idx;
            }
        }
    }

    // Apply @aos_size = max_index + 1 on every AoS group (structural, per-instance).
    for (const auto& kv : aosMax) {
        auto gsegs = splitPath(kv.first);
        if (gsegs.empty()) continue;
        hid_t g = ensureGroups(view_root, gsegs);
        setU64Attr(g, "aos_size", kv.second + 1);
        H5Gclose(g);
        st.n_aos_groups++;
    }
}

} // namespace

// --------------------------------------------------------------------------
// Public entry points
// --------------------------------------------------------------------------

PublishStats publish(const std::string& filename, const std::string& root_group)
{
    // Two distinct (non-exclusive) handles to the same file:
    //   * write_h — an R/W handle WE OWN, used to create the view. We close it.
    //   * db      — a READ-mode PanzerDB on its OWN handle, providing getLeaves();
    //               it owns+manages that handle (auto-closed at scope exit).
    // This avoids a double-close: PanzerDB's close() closes a file handle it was
    // given, so we must not keep a shared handle that it also closes.
    hid_t write_h = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (write_h < 0) h5err("cannot open for R/W: " + filename);

    PanzerDB db(filename, PanzerDB::OpenMode::READ, /*preserve_empty*/true);
    const std::vector<PanzerDB::Leaf>& leaves = db.getLeaves();  // valid while db lives

    // Find the PanzerDB data sub-group. The PanzerDB constructor already
    // discovered it (it stores loc_id internally); we replicate the same check
    // to get the name needed for H5Dopen2 / H5Pset_virtual.
    // If index is at file root → raw_prefix = ""; otherwise first top-level
    // group containing "index" → raw_prefix = "<name>/".
    std::string raw_prefix;
    if (H5Lexists(write_h, "index", H5P_DEFAULT) <= 0) {
        // Search top-level groups (same logic as PanzerDB ctor)
        H5G_info_t ginfo;
        if (H5Gget_info(write_h, &ginfo) >= 0) {
            for (hsize_t i = 0; i < ginfo.nlinks; ++i) {
                char name[256];
                H5Lget_name_by_idx(write_h, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                                   name, sizeof(name), H5P_DEFAULT);
                hid_t gid = H5Gopen2(write_h, name, H5P_DEFAULT);
                if (gid >= 0) {
                    if (H5Lexists(gid, "index", H5P_DEFAULT) > 0) {
                        raw_prefix = std::string(name) + "/";
                        H5Gclose(gid);
                        break;
                    }
                    H5Gclose(gid);
                }
            }
        }
    }

    // Idempotency: remove any existing view group before rebuilding.
    if (H5Lexists(write_h, root_group.c_str(), H5P_DEFAULT) > 0)
        H5Ldelete(write_h, root_group.c_str(), H5P_DEFAULT);

    hid_t view_root = H5Gcreate2(write_h, root_group.c_str(),
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (view_root < 0) { H5Fclose(write_h); h5err("failed to create view root group: " + root_group); }

    PublishStats st;
    processAll(write_h, view_root, leaves, filename, raw_prefix, st);

    H5Gclose(view_root);
    H5Fclose(write_h);   // db's own handle is closed by db's destructor at scope end
    return st;
}

}  // namespace imas::view
