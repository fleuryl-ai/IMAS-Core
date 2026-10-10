// view_publisher.cpp
// Builds a user-friendly VDS view on a PanzerDB snapshot file.
// Design: see view_publisher.h.

#include "view_publisher.h"
#include "panzerdb.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <stdexcept>
#include <vector>

namespace imas {
namespace view {
namespace {

// IMAS invalid values, used as the HDF5 fill value for the cells an aggregated
// dataset cannot map (ragged slices, sparse AoS instances).
constexpr double  INVALID_F64 = -9.0e40;
constexpr int32_t INVALID_I32 = -999999;

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

std::string joinPath(const std::vector<std::string>& segs) {
    std::string out;
    for (const auto& s : segs) { if (!out.empty()) out += '/'; out += s; }
    return out;
}

bool isAllDigits(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    return true;
}

uint64_t productShape(PanzerDB::Leaf::ShapeView shape) {
    uint64_t p = 1;
    for (auto d : shape) p *= (d ? d : 1);
    return p;
}

template <typename T>
uint64_t product(const std::vector<T>& v) {
    uint64_t p = 1;
    for (auto d : v) p *= (d ? (uint64_t)d : 1ull);
    return p;
}

void bumpMax(std::map<std::string, uint64_t>& m, const std::string& k, uint64_t v) {
    auto it = m.find(k);
    if (it == m.end() || it->second < v) m[k] = v;
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

// Readers for the attributes this publisher writes (we own the vocabulary, so
// they are looked up by name rather than iterated).  All of them are optional:
// the --expand view carries fewer attributes than the collapsed one.
bool hasAttrName(hid_t obj, const char* name) {
    return H5Aexists(obj, name) > 0;
}

bool attrStr(hid_t obj, const char* name, std::string& out) {
    if (H5Aexists(obj, name) <= 0) return false;
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0) return false;
    hid_t t = H5Aget_type(a);
    std::vector<char> buf(H5Tget_size(t) + 1, 0);
    const bool ok = (H5Aread(a, t, buf.data()) >= 0);
    H5Tclose(t);
    H5Aclose(a);
    if (!ok) return false;
    out.assign(buf.data());          // stored NUL-terminated
    return true;
}

bool attrU64(hid_t obj, const char* name, uint64_t& out) {
    if (H5Aexists(obj, name) <= 0) return false;
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0) return false;
    double v = 0.0;
    const bool ok = (H5Aread(a, H5T_NATIVE_DOUBLE, &v) >= 0);
    H5Aclose(a);
    if (!ok || !(v >= 0.0)) return false;
    out = static_cast<uint64_t>(v);
    return true;
}

std::string u64Text(uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long)v);
    return buf;
}

// --------------------------------------------------------------------------
// Type helpers (global PanzerDB naming)
// --------------------------------------------------------------------------

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

// --------------------------------------------------------------------------
// Expanded mode: one VDS per index row (historical behaviour)
// --------------------------------------------------------------------------

std::vector<hsize_t> computeDims(uint64_t count, PanzerDB::Leaf::ShapeView shape) {
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
    std::vector<hsize_t> dims = computeDims(leaf.count, leaf.shape_span());
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
        uint64_t prod = productShape(leaf.shape_span());
        uint64_t tdim = leaf.ndim == 0
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

void createNodeGroup(hid_t view_root, const std::vector<std::string>& segs) {
    if (segs.empty()) return;
    hid_t g = ensureGroups(view_root, segs);
    H5Gclose(g);
}

void processExpanded(hid_t file_id, hid_t view_root,
                     const std::vector<PanzerDB::Leaf>& leaves,
                     const std::string& source_filename,
                     const std::string& raw_prefix,
                     PublishStats& st)
{
    // Collect per-group max index (for @aos_size).  Key = relative group path
    // under view_root; value = max integer child index seen.
    std::map<std::string, uint64_t> aosMax;

    for (const auto& leaf : leaves) {
        st.n_total++;
        unsigned kind = leaf.flags & 0xFu;
        if (leaf.is_empty || kind == 1) { st.n_empty_skipped++; continue; }

        std::string fullPath = std::string(leaf.path);
        auto segs = splitPath(fullPath);

        if (kind == 0) {
            if (createVDS(file_id, view_root, leaf, source_filename, raw_prefix)) st.n_data_leaves++;
            else st.n_errors++;
        } else {
            // kind 1 handled above (is_empty); kind 2/3 = AoS node
            if (!segs.empty()) createNodeGroup(view_root, segs);
        }

        // Track the integer-children of every intermediate group (AoS size).
        // For a segs[i] that is a pure integer, its parent group
        // (view_root / segs[0..i-1]) is an AoS with size >= segs[i] + 1.
        if ((kind & 1u) == 0) {   // include data leaves and AoS leaves for the tree view
            for (size_t i = 1; i < segs.size(); ++i) {
                const auto& s = segs[i];
                if (!isAllDigits(s)) continue;
                // An integer at the TOP level is not a real IMAS AoS.
                std::string key;
                for (size_t j = 0; j < i; ++j) { if (j) key += '/'; key += segs[j]; }
                bumpMax(aosMax, key, strtoull(s.c_str(), nullptr, 10));
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

// ==========================================================================
// Collapsed mode: one VDS per schema path, with the AoS levels and the time
// steps folded into leading dimensions.
// ==========================================================================

// Parent path -> names of its child GROUPS.  The last segment of a stored path
// is a leaf, never a group, so `time_slice` still counts as an AoS although it
// also holds direct leaf children (`time_slice/boundary&psi_norm`).
struct TreeInfo {
    std::map<std::string, std::set<std::string>> kids;
    mutable std::set<std::string> dynamicAos;   // collapsed paths of the kind-3 (time-growing) AoS

    bool isAos(const std::string& parent) const {
        auto it = kids.find(parent);
        if (it == kids.end() || it->second.empty()) return false;
        for (const auto& n : it->second) if (!isAllDigits(n)) return false;
        return true;
    }
};

void buildTree(const std::vector<PanzerDB::Leaf>& leaves, TreeInfo& tree) {
    for (const auto& leaf : leaves) {
        const auto segs = splitPath(std::string(leaf.path));
        for (size_t j = 0; j + 1 < segs.size(); ++j) {
            tree.kids[joinPath(std::vector<std::string>(segs.begin(), segs.begin() + j))].insert(segs[j]);
        }
    }
}

// One AoS level met while collapsing an instance path.
struct AoSLevel {
    std::string name;       // the AoS name (the group that carries @aos_size)
    std::string groupPath;  // collapsed group path of that AoS
    uint64_t    index = 0;  // this row's instance index
};

// Strips the AoS index segments from the ANCESTORS of an instance path (the
// caller keeps the last segment as the leaf name).
struct PathSplit {
    std::vector<std::string> groupSegs;
    std::vector<AoSLevel>    levels;
};

PathSplit collapseAncestors(const std::vector<std::string>& segs, const TreeInfo& tree) {
    PathSplit out;
    const size_t n = segs.size();
    for (size_t j = 0; j + 1 < n; ++j) {
        const std::string parent = joinPath(std::vector<std::string>(segs.begin(), segs.begin() + j));
        if (j > 0 && isAllDigits(segs[j]) && tree.isAos(parent)) {
            AoSLevel lv;
            lv.index     = strtoull(segs[j].c_str(), nullptr, 10);
            lv.name      = isAllDigits(segs[j - 1])
                             ? (out.groupSegs.empty() ? std::string("aos") : out.groupSegs.back())
                             : segs[j - 1];
            // The AoS name was itself kept as a group segment in the previous
            // iteration, so the current groupSegs already end with it.
            lv.groupPath = joinPath(out.groupSegs);
            out.levels.push_back(std::move(lv));
            continue;
        }
        out.groupSegs.push_back(segs[j]);
    }
    return out;
}

// Collapses every index segment of a path — used on AoS meta rows, whose last
// segment is the AoS name itself rather than a data leaf.
std::string collapseWholePath(const std::vector<std::string>& segs, const TreeInfo& tree) {
    std::vector<std::string> kept;
    for (size_t j = 0; j < segs.size(); ++j) {
        const std::string parent = joinPath(std::vector<std::string>(segs.begin(), segs.begin() + j));
        if (j > 0 && isAllDigits(segs[j]) && tree.isAos(parent)) continue;
        kept.push_back(segs[j]);
    }
    return joinPath(kept);
}

// One stored index row, expressed in the aggregated dataset's coordinates.
struct Cell {
    std::vector<uint64_t> lead;   // position in each leading dim (raw order, see AggLeaf)
    uint64_t slices = 1;          // time steps this row covers (a bulk row covers several)
    std::vector<uint64_t> shape;  // per-slice shape (empty == scalar)
    uint64_t off   = 0;
    uint64_t count = 0;
};

struct AggLeaf {
    std::vector<std::string> groupSegs;
    std::string              leafName;
    PanzerDB::DataType       dt = PanzerDB::DataType::UNKNOWN;
    std::string              schema;

    // Before the singleton squeeze: one slot per AoS level of the instance path,
    // preceded by slot 0 holding the folded time position.
    std::vector<std::string> rawLevelNames;
    bool     hasTime = false;
    uint64_t timeSize = 0;
    std::string timeName;

    // After the squeeze: the dataset's leading dimensions.
    std::vector<std::string> levelNames;
    std::vector<uint64_t>    levelSizes;

    std::vector<uint64_t> sliceShape;     // element-wise max over the rows
    bool     ragged = false;
    uint64_t nElements = 0;
    std::vector<Cell> cells;
};

// Aggregates every data row onto its schema path and resolves the leading
// dimensions.  aosSizes collects each AoS level's size (its meta row when
// present, otherwise max index + 1) and drives the @aos_size attributes.
void buildAggregates(const std::vector<PanzerDB::Leaf>& leaves,
                     const TreeInfo& tree,
                     std::map<std::string, AggLeaf>& aggs,
                     std::map<std::string, uint64_t>& aosSizes,
                     PublishStats& st)
{
    // 1. AoS sizes carried by the meta rows (kind 2 static / kind 3 dynamic).
    for (const auto& leaf : leaves) {
        const unsigned kind = leaf.flags & 0xFu;
        if (kind != 2 && kind != 3) continue;
        const std::string key = collapseWholePath(splitPath(std::string(leaf.path)), tree);
        if (key.empty()) continue;
        if (kind == 3) tree.dynamicAos.insert(key);
        if (leaf.ndim >= 1 && leaf.shape[0] > 0) bumpMax(aosSizes, key, leaf.shape[0]);
    }

    // 2. Data rows, grouped by exact instance path: a logical node stored as one
    //    row per time slice has no index in its path — the time order is the row
    //    order (see the layout note in tools/panzer_index.py).
    std::map<std::string, std::vector<const PanzerDB::Leaf*>> byPath;
    for (const auto& leaf : leaves) {
        st.n_total++;
        const unsigned kind = leaf.flags & 0xFu;
        if (leaf.is_empty || kind == 1) { st.n_empty_skipped++; continue; }
        if (kind != 0) continue;                    // AoS meta nodes are groups only
        if (leaf.count == 0) continue;
        byPath[std::string(leaf.path)].push_back(&leaf);
    }

    for (auto& kv : byPath) {
        const std::string& fullPath = kv.first;
        std::vector<const PanzerDB::Leaf*>& rows = kv.second;
        std::sort(rows.begin(), rows.end(), [](const PanzerDB::Leaf* a, const PanzerDB::Leaf* b) {
            if (a->time_index != b->time_index) return a->time_index < b->time_index;
            return a->row_id < b->row_id;
        });

        const auto segs = splitPath(fullPath);
        if (segs.empty()) continue;
        const PathSplit ps = collapseAncestors(segs, tree);
        const std::string leafName = segs.back();
        // An "index" leaf sitting at the top level would clash with the raw index
        // table name, exactly as in the expanded view.
        const std::string storedLeaf =
            (ps.groupSegs.empty() && leafName == "index") ? std::string("index_node") : leafName;

        const auto dt = static_cast<PanzerDB::DataType>(rows.front()->flags >> 4);
        const std::string key = joinPath(ps.groupSegs) + '\x1f' + storedLeaf + '\x1f' + dtypeString(dt);

        AggLeaf& agg = aggs[key];
        if (agg.cells.empty()) {
            agg.groupSegs = ps.groupSegs;
            agg.leafName  = storedLeaf;
            agg.dt        = dt;
            agg.schema    = PanzerDB::stripIndices(fullPath);
            agg.timeName  = ps.groupSegs.empty() ? std::string("time") : ps.groupSegs.back();
            for (const auto& lv : ps.levels) agg.rawLevelNames.push_back(lv.name);
        }
        for (const auto& lv : ps.levels) bumpMax(aosSizes, lv.groupPath, lv.index + 1);

        uint64_t timePos = 0;
        for (const PanzerDB::Leaf* r : rows) {
            std::vector<uint64_t> shape;
            for (size_t d = 0; d < r->ndim; ++d) shape.push_back(r->shape[d]);
            const uint64_t vol = shape.empty() ? 1 : product(shape);
            uint64_t slices = 1;
            if (vol > 0 && r->count > vol && r->count % vol == 0) slices = r->count / vol;
            else if (shape.empty() && r->count > 1)                slices = r->count;

            Cell c;
            c.lead.push_back(timePos);                       // slot 0: folded time position
            for (const auto& lv : ps.levels) c.lead.push_back(lv.index);
            c.slices = slices;
            c.shape  = std::move(shape);
            c.off    = r->offset;
            c.count  = r->count;
            agg.cells.push_back(std::move(c));
            agg.nElements += r->count;
            timePos += slices;
        }
        agg.timeSize = std::max(agg.timeSize, timePos);
    }

    // 3. Leading dimensions = folded time axis (when relevant) + AoS levels,
    //    minus the singletons, which carry no information and only inflate shapes.
    for (auto& kv : aggs) {
        AggLeaf& agg = kv.second;
        if (agg.cells.empty()) continue;

        uint64_t timeEnd = 0;
        for (const auto& c : agg.cells)
            timeEnd = std::max(timeEnd, c.lead.front() + c.slices);
        agg.hasTime  = (timeEnd > 1);
        agg.timeSize = std::max<uint64_t>(1, timeEnd);

        const size_t rawN = agg.rawLevelNames.size();
        std::vector<uint64_t> sizes(rawN, 1);
        for (const auto& c : agg.cells)
            for (size_t d = 0; d < rawN; ++d) sizes[d] = std::max(sizes[d], c.lead[d + 1] + 1);
        std::vector<size_t> keep;
        for (size_t d = 0; d < rawN; ++d) if (sizes[d] > 1) keep.push_back(d);

        if (agg.hasTime) { agg.levelNames.push_back(agg.timeName); agg.levelSizes.push_back(agg.timeSize); }
        for (size_t d : keep) {
            agg.levelNames.push_back(agg.rawLevelNames[d]);
            agg.levelSizes.push_back(sizes[d]);
        }

        // Re-line every cell's leading coordinates with the kept dimensions.
        for (auto& c : agg.cells) {
            std::vector<uint64_t> nl;
            if (agg.hasTime) nl.push_back(c.lead[0]);
            for (size_t d : keep) nl.push_back(c.lead[d + 1]);
            c.lead = std::move(nl);
        }

        // Per-slice shape = element-wise max; anything smaller leaves a hole.
        size_t rank = 0;
        for (const auto& c : agg.cells) rank = std::max(rank, c.shape.size());
        bool flat = false;
        for (const auto& c : agg.cells) {
            if (c.shape.size() == rank) continue;
            if (product(c.shape) * c.slices == c.count) continue;   // scalar-empty shape
            flat = true;                                            // ranks disagree
        }
        agg.sliceShape.assign(flat ? 1 : rank, 0);
        for (const auto& c : agg.cells) {
            if (!flat && c.shape.size() == rank) {
                for (size_t s = 0; s < rank; ++s)
                    agg.sliceShape[s] = std::max(agg.sliceShape[s], c.shape[s]);
            } else {
                agg.sliceShape[0] = std::max(agg.sliceShape[0], c.count / c.slices);
            }
        }
        for (const auto& c : agg.cells)
            if (c.shape != agg.sliceShape) agg.ragged = true;
    }
}

// A maximal group of cells shareable by one virtual mapping.
struct Run {
    size_t   firstCell = 0;
    uint64_t len       = 1;
    uint64_t tgtStep   = 1;   // step between consecutive cells along the run dimension
    uint64_t srcStep   = 1;   // step between their source blocks
};

// Buckets the cells by "all leading dims but the run dim" and walks each bucket,
// extending a run while the pattern stays affine: identical cells, constant
// target step, evenly spaced and non-overlapping source blocks.  This is what
// keeps a 13 857-instance AoS at a single mapping instead of 13 857.
std::vector<Run> buildRuns(const AggLeaf& agg) {
    const size_t leadN = agg.levelSizes.size();
    const int runDim = (leadN > 0) ? (int)leadN - 1 : -1;   // innermost AoS level, else time

    std::map<std::vector<uint64_t>, std::vector<size_t>> buckets;
    for (size_t i = 0; i < agg.cells.size(); ++i) {
        std::vector<uint64_t> key;
        for (size_t d = 0; d < leadN; ++d) if ((int)d != runDim) key.push_back(agg.cells[i].lead[d]);
        buckets[std::move(key)].push_back(i);
    }

    std::vector<Run> runs;
    for (auto& kv : buckets) {
        auto& idx = kv.second;
        if (runDim >= 0) {
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
                return agg.cells[a].lead[runDim] < agg.cells[b].lead[runDim];
            });
        }
        size_t i = 0;
        while (i < idx.size()) {
            const Cell& base = agg.cells[idx[i]];
            uint64_t tStep = 0, sStep = 0;
            size_t j = i + 1;
            while (j < idx.size()) {
                const Cell& prev = agg.cells[idx[j - 1]];
                const Cell& c    = agg.cells[idx[j]];
                if (c.count != base.count || c.shape != base.shape || c.slices != base.slices) break;
                // Merging along the time axis is only sound for single-slice rows.
                if (runDim == 0 && base.slices != 1) break;
                if (runDim >= 0 && c.lead[runDim] <= prev.lead[runDim]) break;
                if (c.off <= prev.off) break;
                const uint64_t t = (runDim >= 0) ? c.lead[runDim] - prev.lead[runDim] : 1;
                const uint64_t s = c.off - prev.off;
                if (s < c.count) break;                      // overlapping source blocks
                if (tStep == 0) { tStep = t; sStep = s; }
                else if (t != tStep || s != sStep) break;
                ++j;
            }
            Run run;
            run.firstCell = idx[i];
            run.len       = j - i;
            run.tgtStep   = (tStep == 0) ? 1 : tStep;
            run.srcStep   = (sStep == 0) ? 1 : sStep;
            runs.push_back(run);
            i = j;
        }
    }
    return runs;
}

// Fill value for unmapped cells, per stored type.  False when the type has no
// meaningful invalid value (then no fill is set).
bool setImasFill(hid_t dcpl, hid_t type, PanzerDB::DataType dt) {
    switch (dt) {
        case PanzerDB::DataType::FLOAT64:
            return H5Pset_fill_value(dcpl, type, &INVALID_F64) >= 0;
        case PanzerDB::DataType::INT32: {
            const size_t sz = H5Tget_size(type);
            if (sz < sizeof(int32_t)) return false;
            std::vector<char> buf(sz, 0);
            const int32_t v = INVALID_I32;
            memcpy(buf.data(), &v, sizeof(v));
            return H5Pset_fill_value(dcpl, type, buf.data()) >= 0;
        }
        case PanzerDB::DataType::COMPLEX128: {
            // data_raw_c128 is an array type of two IEEE 64-bit floats: both
            // components take the invalid value.
            const double buf[2] = { INVALID_F64, INVALID_F64 };
            return H5Pset_fill_value(dcpl, type, buf) >= 0;
        }
        case PanzerDB::DataType::STRING:
        case PanzerDB::DataType::LIST_OF_STRINGS:
        case PanzerDB::DataType::STRING_CHUNKED: {
            const size_t sz = H5Tget_size(type);
            std::vector<char> buf(sz, 0);   // empty string, NUL-padded to the slot width
            return H5Pset_fill_value(dcpl, type, buf.data()) >= 0;
        }
        default:
            return false;
    }
}

bool createAggVDS(hid_t file_id, hid_t view_root, const AggLeaf& agg,
                  const std::string& source_filename,
                  const std::string& raw_prefix,
                  uint64_t mappingCap,
                  PublishStats& st)
{
    const char* raw = dataRawName(agg.dt);
    if (!raw || agg.cells.empty()) return false;

    std::vector<hsize_t> dims;
    dims.reserve(agg.levelSizes.size() + agg.sliceShape.size());
    for (auto s : agg.levelSizes) dims.push_back((hsize_t)s);
    for (auto s : agg.sliceShape) dims.push_back((hsize_t)s);
    const size_t leadN  = agg.levelSizes.size();
    const size_t shapeN = agg.sliceShape.size();
    const size_t rank   = dims.size();
    const int runDim    = (leadN > 0) ? (int)leadN - 1 : -1;
    if (rank > H5S_MAX_RANK) { st.n_rank_overflow++; return false; }

    hid_t parent = ensureGroups(view_root, agg.groupSegs);
    const bool parent_is_loc = (parent == view_root);

    const std::string full_raw = raw_prefix + raw;
    hid_t src = H5Dopen2(file_id, full_raw.c_str(), H5P_DEFAULT);
    if (src < 0) { if (!parent_is_loc) H5Gclose(parent); return false; }
    hid_t src_type  = H5Dget_type(src);
    hid_t src_space = H5Dget_space(src);   // real extent: every mapping lives inside it
    H5Dclose(src);

    hid_t vspace;
    if (rank == 0) vspace = H5Screate(H5S_SCALAR);
    else vspace = H5Screate_simple((int)rank, dims.data(), nullptr);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(dcpl, H5D_VIRTUAL);
    setImasFill(dcpl, src_type, agg.dt);

    std::vector<Run> runs = buildRuns(agg);
    const bool capped = (mappingCap > 0 && runs.size() > mappingCap);
    if (capped) runs.resize(mappingCap);

    uint64_t localMappings = 0, mappedElements = 0;
    for (const Run& run : runs) {
        const Cell& c0 = agg.cells[run.firstCell];

        std::vector<hsize_t> start(rank, 0), count(rank, 1), stride(rank, 1), block(rank, 1);
        for (size_t d = 0; d < leadN; ++d) {
            // A cell spans `slices` consecutive positions on the time dim and
            // `run.len` on the run dim; a multi-slice cell is never merged along
            // time (see buildRuns), so the product is exact in every case.
            const bool isRun  = ((int)d == runDim);
            const bool isTime = (agg.hasTime && d == 0);
            start[d]  = (hsize_t)c0.lead[d];
            count[d]  = (hsize_t)((isRun ? run.len : 1) * (isTime ? c0.slices : 1));
            stride[d] = (isRun && run.len > 1) ? (hsize_t)run.tgtStep : 1;
        }
        const bool flatShape = (c0.shape.size() != shapeN);
        for (size_t s = 0; s < shapeN; ++s) {
            start[leadN + s] = 0;
            count[leadN + s] = flatShape ? 1 : (hsize_t)c0.shape[s];
        }
        if (flatShape) count[leadN] = (hsize_t)(c0.count / c0.slices);

        uint64_t tgtElements = 1;
        for (size_t d = 0; d < rank; ++d) tgtElements *= (uint64_t)count[d];
        const uint64_t srcElements = (uint64_t)run.len * c0.count;

        hid_t vsel   = H5Scopy(vspace);
        hid_t srcsel = H5Scopy(src_space);
        bool ok = true;
        if (rank > 0)
            ok = H5Sselect_hyperslab(vsel, H5S_SELECT_SET, start.data(), stride.data(),
                                     count.data(), block.data()) >= 0;
        if (ok && tgtElements != srcElements) ok = false;   // HDF5 requires equal counts
        const hsize_t sStart[1]  = { (hsize_t)c0.off };
        const hsize_t sStride[1] = { (hsize_t)run.srcStep };
        const hsize_t sCount[1]  = { (hsize_t)run.len };
        const hsize_t sBlock[1]  = { (hsize_t)c0.count };
        if (ok)
            ok = H5Sselect_hyperslab(srcsel, H5S_SELECT_SET, sStart, sStride, sCount, sBlock) >= 0;
        if (ok && H5Pset_virtual(dcpl, vsel, source_filename.c_str(), full_raw.c_str(), srcsel) >= 0) {
            ++localMappings;
            mappedElements += srcElements;
        }
        H5Sclose(srcsel);
        H5Sclose(vsel);
    }

    // A dataset with no mapping at all would read as nothing but fill values:
    // worse than not publishing it, because it looks like real data. Refuse it.
    if (localMappings == 0) {
        st.n_errors++;
        H5Pclose(dcpl);
        H5Sclose(src_space);
        H5Sclose(vspace);
        H5Tclose(src_type);
        if (!parent_is_loc) H5Gclose(parent);
        return false;
    }

    hid_t vds = H5Dcreate2(parent, agg.leafName.c_str(), src_type, vspace,
                           H5P_DEFAULT, dcpl, H5P_DEFAULT);
    const bool created = (vds >= 0);
    if (created) {
        setStrAttr(vds, "datatype", dtypeString(agg.dt));
        setU64Attr(vds, "n_elements", agg.nElements);
        setStrAttr(vds, "schema", agg.schema);
        setU64Attr(vds, "aos_ndim", leadN);
        setU64Attr(vds, "n_mappings", localMappings);
        std::string label;
        for (const auto& n : agg.levelNames) { if (!label.empty()) label += '|'; label += n; }
        for (size_t s = 0; s < shapeN; ++s) { if (!label.empty()) label += '|'; label += agg.leafName; }
        if (!label.empty()) setStrAttr(vds, "dims_label", label);
        if (agg.hasTime)  setU64Attr(vds, "time_dim", agg.timeSize);
        if (agg.ragged)   setU64Attr(vds, "ragged", 1);
        if (capped)       setU64Attr(vds, "mapping_cap", 1);
        st.n_mappings    += localMappings;
        st.max_mappings_in_one_dataset = std::max(st.max_mappings_in_one_dataset, localMappings);
        const uint64_t total = product(dims);
        st.n_fill_holes += (total > mappedElements) ? total - mappedElements : 0;
        if (capped) st.n_mapping_cap_fallback++;
        st.n_aggregated_datasets++;
        st.n_data_leaves++;
        H5Dclose(vds);
    } else {
        st.n_errors++;
    }

    H5Pclose(dcpl);
    H5Sclose(src_space);
    H5Sclose(vspace);
    H5Tclose(src_type);
    if (!parent_is_loc) H5Gclose(parent);
    return created;
}

void processCollapsed(hid_t file_id, hid_t view_root,
                      const std::vector<PanzerDB::Leaf>& leaves,
                      const std::string& source_filename,
                      const std::string& raw_prefix,
                      uint64_t mappingCap,
                      PublishStats& st)
{
    TreeInfo tree;
    buildTree(leaves, tree);

    std::map<std::string, AggLeaf> aggs;
    std::map<std::string, uint64_t> aosSizes;
    buildAggregates(leaves, tree, aggs, aosSizes, st);

    for (auto& kv : aggs)
        createAggVDS(file_id, view_root, kv.second, source_filename, raw_prefix, mappingCap, st);

    // @aos_size / @aos_dynamic on every AoS group of the collapsed tree.
    for (const auto& kv : aosSizes) {
        auto segs = splitPath(kv.first);
        if (segs.empty()) continue;
        hid_t g = ensureGroups(view_root, segs);
        setU64Attr(g, "aos_size", kv.second);
        if (tree.dynamicAos.count(kv.first)) setU64Attr(g, "aos_dynamic", 1);
        H5Gclose(g);
        st.n_aos_groups++;
    }
}

// --------------------------------------------------------------------------
// View handling shared by both modes
// --------------------------------------------------------------------------

// PanzerDB may keep its tables in a top-level sub-group; find its name so the
// VDS source paths resolve (the same check the PanzerDB constructor performs).
std::string findRawPrefix(hid_t file_id) {
    if (H5Lexists(file_id, "index", H5P_DEFAULT) > 0) return std::string();
    H5G_info_t ginfo;
    if (H5Gget_info(file_id, &ginfo) < 0) return std::string();
    for (hsize_t i = 0; i < ginfo.nlinks; ++i) {
        char name[256];
        if (H5Lget_name_by_idx(file_id, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                               name, sizeof(name), H5P_DEFAULT) < 0) continue;
        hid_t gid = H5Gopen2(file_id, name, H5P_DEFAULT);
        if (gid < 0) continue;
        const bool hasIndex = (H5Lexists(gid, "index", H5P_DEFAULT) > 0);
        H5Gclose(gid);
        if (hasIndex) return std::string(name) + "/";
    }
    return std::string();
}

PublishStats publishImpl(const std::string& filename, const PublishOptions& options)
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

    const std::string rawPrefix   = findRawPrefix(write_h);
    const std::string& root_group = options.root_group;

    // Idempotency: remove any existing view group before rebuilding.
    if (H5Lexists(write_h, root_group.c_str(), H5P_DEFAULT) > 0)
        H5Ldelete(write_h, root_group.c_str(), H5P_DEFAULT);

    hid_t view_root = H5Gcreate2(write_h, root_group.c_str(),
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (view_root < 0) { H5Fclose(write_h); h5err("failed to create view root group: " + root_group); }

    PublishStats st;
    // The expanded view legitimately collides on paths stored as one row per
    // time slice (the loser is counted in n_errors); without silencing the error
    // stack, HDF5 prints a full trace for each of them.
    H5E_auto2_t oldFunc = nullptr;
    void* oldData = nullptr;
    H5Eget_auto2(H5E_DEFAULT, &oldFunc, &oldData);
    if (!options.collapse_aos) H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    if (options.collapse_aos)
        processCollapsed(write_h, view_root, leaves, filename, rawPrefix,
                         options.max_mappings_per_dataset, st);
    else
        processExpanded(write_h, view_root, leaves, filename, rawPrefix, st);

    H5Eset_auto2(H5E_DEFAULT, oldFunc, oldData);
    H5Gclose(view_root);
    H5Fclose(write_h);   // db's own handle is closed by db's destructor at scope end
    return st;
}

// --------------------------------------------------------------------------
// Listing an already published view (read-only)
// --------------------------------------------------------------------------

// Body of the listing line for one object: kind, shape, and the annotations
// h5ls -r never prints.  Every attribute lookup is optional — the --expand view
// carries no @dims_label, @aos_ndim or @n_mappings.
bool describeObject(hid_t file, const std::string& path, std::string& body, bool& isGroup) {
    hid_t obj = H5Oopen(file, path.c_str(), H5P_DEFAULT);
    if (obj < 0) return false;
    H5O_info2_t info;
    const bool haveInfo = (H5Oget_info3(obj, &info, H5O_INFO_BASIC) >= 0);
    isGroup = haveInfo && info.type == H5O_TYPE_GROUP;

    if (isGroup) {
        body = "Group";
        uint64_t n = 0;
        if (attrU64(obj, "aos_size", n)) body += " @aos_size=" + u64Text(n);
        if (hasAttrName(obj, "aos_dynamic")) body += " @aos_dynamic";
    } else if (haveInfo && info.type == H5O_TYPE_DATASET) {
        hid_t space = H5Dget_space(obj);
        const int rank = H5Sget_simple_extent_ndims(space);
        std::vector<hsize_t> dims(rank > 0 ? (size_t)rank : 0, 1);
        if (rank > 0) H5Sget_simple_extent_dims(space, dims.data(), nullptr);
        H5Sclose(space);

        body = "Dataset {";
        for (size_t d = 0; d < dims.size(); ++d) {
            if (d) body += ", ";
            body += std::to_string((long long)dims[d]);
        }
        body += "}";

        std::string label;
        if (attrStr(obj, "dims_label", label)) body += " @dims_label=" + label;
        if (hasAttrName(obj, "ragged")) body += " @ragged";
        if (hasAttrName(obj, "mapping_cap")) body += " @mapping_cap";

        // Cells the mapping cannot reach (ragged slices, sparse AoS instances).
        // Derived from the shape and @n_elements so the view needs no extra
        // attribute — and stays correct for a view published by an older build.
        uint64_t stored = 0;
        if (attrU64(obj, "n_elements", stored)) {
            const uint64_t cells = product(dims);
            if (cells > stored) body += " holes=" + u64Text(cells - stored);
        }
    } else {
        body = "Object";
    }
    H5Oclose(obj);
    return true;
}

// Tree order: parent first, then its links in name order (what h5ls -r gives per
// directory, but here each object is one annotated line).
void walkList(hid_t file, const std::string& dir, std::vector<std::string>& out) {
    hid_t g = H5Gopen2(file, dir.c_str(), H5P_DEFAULT);
    if (g < 0) return;
    H5G_info_t gi;
    const bool listed = (H5Gget_info(g, &gi) >= 0);
    std::vector<std::string> kids;
    if (listed) {
        for (hsize_t i = 0; i < gi.nlinks; ++i) {
            char name[1024];
            if (H5Lget_name_by_idx(g, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                                   name, sizeof(name), H5P_DEFAULT) < 0) continue;
            kids.push_back(name);
        }
    }
    H5Gclose(g);

    for (const auto& k : kids) {
        const std::string full = dir + "/" + k;
        std::string body;
        bool isGroup = false;
        if (!describeObject(file, full, body, isGroup)) continue;
        out.push_back(full + '\t' + body);      // the tab is aligned in a second pass
        if (isGroup) walkList(file, full, out);
    }
}

ListViewResult listImpl(const std::string& filename, const std::string& root_group) {
    if (root_group.empty()) h5err("listView: empty root_group");
    ListViewResult res;

    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) h5err("cannot open read-only: " + filename);

    // A file whose view was never published is a normal state, not an error:
    // the caller turns found==false into "publish first".
    const std::string root = (root_group[0] == '/') ? root_group : "/" + root_group;
    if (H5Lexists(file, root.c_str(), H5P_DEFAULT) <= 0) {
        H5Fclose(file);
        return res;
    }
    res.found = true;

    std::vector<std::string> raw;
    walkList(file, root, raw);
    H5Fclose(file);

    size_t width = 0;
    for (const auto& l : raw) {
        const size_t tab = l.find('\t');
        if (tab != std::string::npos) width = std::max(width, tab);
    }
    for (auto& l : raw) {
        const size_t tab = l.find('\t');
        if (tab == std::string::npos) { res.lines.push_back(std::move(l)); continue; }
        std::string line = l.substr(0, tab);
        line.append(width + 2 - line.size(), ' ');
        line += l.substr(tab + 1);
        res.lines.push_back(std::move(line));
    }
    return res;
}

} // namespace

// --------------------------------------------------------------------------
// Public entry points
// --------------------------------------------------------------------------

PublishStats publish(const std::string& filename, const std::string& root_group) {
    PublishOptions o;
    o.root_group = root_group;
    return publishImpl(filename, o);
}

PublishStats publish(const std::string& filename, const PublishOptions& options) {
    return publishImpl(filename, options);
}

ListViewResult listView(const std::string& filename, const std::string& root_group) {
    return listImpl(filename, root_group);
}

}  // namespace view
}  // namespace imas
