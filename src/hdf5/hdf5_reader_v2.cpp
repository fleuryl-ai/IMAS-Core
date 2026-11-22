#include "hdf5_reader_v2.h"

#include "al_backend.h"
#include "al_defs.h"
#include "al_utilities.h"
#include <algorithm>
#include <set>
#include <vector>
#include <cstring>
#include <complex>
#include <sstream>
#include "hdf5_utils.h"


// Debug macro
#ifdef DEBUG_HDF5_READER_V2
#define DEBUG_PRINT(msg) \
  std::cerr << "[DEBUG " << __func__ << "] " << msg << std::endl
#else
#define DEBUG_PRINT(msg) \
  do {                   \
  } while (0)
#endif

HDF5Reader_v2::HDF5Reader_v2(std::pair<int,int> backend_version_)
    : HDF5Reader(backend_version_){} 

HDF5Reader_v2::~HDF5Reader_v2() = default;

std::string HDF5Reader_v2::getVersion()
{
    return "2.0.0";
}

void HDF5Reader_v2::closePulse(DataEntryContext *ctx, int mode, hid_t *file_id, std::unordered_map<std::string, hid_t> &opened_IDS_files, int files_path_strategy, std::string &files_directory, std::string &relative_file_path)
{
    HDF5Reader::closePulse(ctx, mode, file_id, opened_IDS_files, files_path_strategy, files_directory, relative_file_path);
}

void HDF5Reader_v2::release_session()
{
    // The strategies each hold a shared_ptr to the engine, so the engine is only really
    // released when the last of them goes -- hence dropping them first. PanzerDB's
    // destructor closes its dataset handles, which is what has to happen before /index
    // and the raw tables are unlinked from under this session.
    global_strategy.reset();
    slice_strategy.reset();
    timerange_strategy.reset();
    read_strategy = nullptr;
    session_index.reset();
    session_db.reset();
    session_gid = -1;
    anchor_may_be_stale = false;
}

void HDF5Reader_v2::open_IDS_group(OperationContext *ctx, hid_t file_id, std::unordered_map<std::string, hid_t> &opened_IDS_files, std::string &files_directory, std::string &relative_file_path)
{
    HDF5Reader::open_IDS_group(ctx, file_id, opened_IDS_files, files_directory, relative_file_path);

    // The session engine is deliberately NOT dropped here. The AL closes the IDS group
    // at the end of every operation and opens a fresh one for the next, which used to
    // throw the engine and its shared index away and rebuild both (twice per getSlice).
    // select_strategy moves the engine onto the new group id instead, and drops it there
    // only when that move is impossible.
    //
    // What this operation does owe the session is the knowledge that its anchor may now
    // be void: the group it was taken on can have been closed, HDF5 may hand that id back
    // out for another group, and another writer may have grown /index in between. Hence
    // the flag instead of a check here -- select_strategy has to answer that question
    // anyway, and answering it at the operation boundary is what keeps it out of the
    // per-node read path.
    anchor_may_be_stale = true;
}

void HDF5Reader_v2::close_group(OperationContext *ctx)
{
    HDF5Reader::close_group(ctx);
    // The group the engine is anchored on has just been H5Gclose'd: whatever
    // select_strategy believed about that anchor is void until it is re-checked.
    anchor_may_be_stale = true;
}


void HDF5Reader_v2::beginReadArraystructAction(ArraystructContext *ctx, int *size)
{
    prepare_strategy(ctx);
    if (read_strategy == nullptr) {
        // Group for this context was not opened (occurrence not in the file): empty AoS.
        *size = 0;
        return;
    }
    read_strategy->beginReadArraystructAction(ctx, size);
}

void HDF5Reader_v2::list_filled_paths(const char* dataobjectname, char*** path_list, int* size,
                                      hid_t file_id, std::unordered_map<std::string, hid_t> &opened_IDS_files,
                                      std::string &files_directory, std::string &relative_file_path)
{
    HDF5Utils hdf5_utils;
    // Temporary context so the existing group-opening logic can be reused.
    OperationContext ctx(NULL, dataobjectname, "", READ_OP);
    hid_t gid = -1;
    hdf5_utils.open_IDS_group(&ctx, file_id, opened_IDS_files, files_directory, relative_file_path, &gid);
    if (gid <= 0) {
        // Group does not exist => nothing is filled (the HLI reports an empty occurrence).
        *size = 0;
        return;
    }

    // In the PanzerDB layout the group's first level holds only physical columns
    // ({data_raw,data_scalar,data_time}_*, index, path_table, list_spans), so the
    // base class' "list the group links" assumption does not hold. Enumerate the
    // logical leaves from the /index cache instead; the group handle stays owned
    // by the caller (opened_IDS_files), so open PanzerDB with close_loc_id=false.
    std::set<std::string> logical_paths;
    {
        PanzerDB db(gid, PanzerDB::OpenMode::READ, /*preserve_empty_nodes=*/false, /*close_loc_id_on_exit=*/false);
        const auto& leaves = db.getLeaves();
        for (const auto& leaf : leaves) {
            // Node kind lives in the low 4 bits: 2 = static AoS meta-node, 3 = dynamic
            // AoS meta-node. Both are containers, not data nodes: the base class only
            // ever lists data nodes (the &AOS_SHAPE companions are skipped there).
            if ((leaf.flags & 0xFULL) == 2 || (leaf.flags & 0xFULL) == 3) continue;

            std::string path(leaf.path);  // zero-copy views: copy out before the DB dies
            // Mirror the base class' DD-path conversion first ('&' flattening, "[]" removal).
            path.erase(std::remove(path.begin(), path.end(), '['), path.end());
            path.erase(std::remove(path.begin(), path.end(), ']'), path.end());
            std::replace(path.begin(), path.end(), '&', '/');

            // Occurrence indices: panzerdb instance paths carry them as numeric segments
            // (e.g. "profiles_1d/0/ion/1/temperature"); strip them, and de-duplicate the
            // per-occurrence leaves onto one logical path (e.g. "profiles_1d/ion/temperature").
            std::vector<std::string> parts;
            std::string token;
            for (char c : path) {
                if (c == '/') {
                    parts.push_back(token);
                    token.clear();
                } else {
                    token += c;
                }
            }
            parts.push_back(token);
            std::string logical;
            bool first = true;
            for (const auto& part : parts) {
                if (part.empty()) continue;
                bool numeric = true;
                for (char c : part) {
                    if (c < '0' || c > '9') { numeric = false; break; }
                }
                if (numeric) continue;
                if (!first) logical += '/';
                logical += part;
                first = false;
            }
            if (!logical.empty()) logical_paths.insert(logical);
        }
    }

    std::vector<std::string> variables;
    variables.reserve(logical_paths.size());
    for (const auto& p : logical_paths) variables.push_back(p);  // set already sorted
    utilities::copy_stringvector_to_c_list(variables, path_list, size);
}

void HDF5Reader_v2::endAction(Context *ctx)
{
    prepare_strategy(ctx);
    auto op_ctx = static_cast<OperationContext *>(ctx);

    if (read_strategy && IDS_group_id.size() == 1 && IDS_group_id.count(op_ctx)) {
        read_strategy->endAction(ctx);
    }
}

void HDF5Reader_v2::select_strategy(OperationContext *ctx, hid_t gid) {
    // One shared engine + index per group: built on the first read that needs it,
    // adopted by every strategy of the session (3x /index load -> 1x).
    // The group id changes at every operation, so an existing engine is moved onto it
    // rather than rebuilt: rebuilding threw away the decoded /index, the memoised
    // signals and the dataset handles with their chunk caches, twice per getSlice.
    //
    // That move is asked once per operation, not once per node: this runs from
    // prepare_strategy, which every read entry point calls (read_ND_Data,
    // beginReadArraystructAction, endAction), while the group itself is opened twice per
    // operation. Re-asking it per node cost 4 303 315 re-opens of /index -- each one
    // H5Fget_name, H5Oget_info3, H5Otoken_cmp, H5Dget_space -- against 122 266 real reads
    // on core_profiles_3 of pulse 54924, ~9 s of a 22 s get() for an anchor that had not
    // moved. The boundary is where the anchor can actually have been lost (open_IDS_group,
    // close_group) and where a foreign write can show up, so that is where it is checked;
    // within one operation the file cannot change underneath a read of its own.
    const bool anchor_trusted = session_db && !anchor_may_be_stale && gid == session_gid;
    if (!anchor_trusted) {
        if (!session_db || !session_db->reanchor(gid)) {
            session_db.reset();
            session_index.reset();
            global_strategy.reset();
            slice_strategy.reset();
            timerange_strategy.reset();
            read_strategy = nullptr;
            session_db = std::make_shared<PanzerDB>(gid, PanzerDB::OpenMode::READ);
            session_index = std::make_shared<ReadIndex>(session_db);   // build_indexes() in ctor
        }
        session_gid = gid;
        anchor_may_be_stale = false;
    }

    if (ctx->getRangemode() == GLOBAL_OP) {
        if (!global_strategy) global_strategy = std::make_unique<GlobalReadStrategy>(session_db, session_index);
        read_strategy = global_strategy.get();
    } 
    else if (ctx->getRangemode() == SLICE_OP) {
        if (!slice_strategy) slice_strategy = std::make_unique<SliceReadStrategy>(session_db, session_index);
        read_strategy = slice_strategy.get();
    }
    else if (ctx->getRangemode() == TIMERANGE_OP) {
        if (!timerange_strategy) timerange_strategy = std::make_unique<TimeRangeReadStrategy>(session_db, session_index);
        read_strategy = timerange_strategy.get();
    }
    else {
        throw ALBackendException("Unknown operation context range mode", LOG);
    }
}

void HDF5Reader_v2::prepare_strategy(Context *ctx) {
    // 1. Extract the OperationContext according to the context type
    OperationContext *opctx = nullptr;
    if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
        opctx = static_cast<ArraystructContext *>(ctx)->getOperationContext();
    } else {
        opctx = static_cast<OperationContext *>(ctx);
    }

    // 2. Find the GID
    hid_t gid = -1;
    auto it = IDS_group_id.find(opctx);
    if (it != IDS_group_id.end()) {
        gid = it->second;
    }

    if (gid == -1) {
        // Mirror the base (v1) reader: an occurrence that is not in the file simply has
        // no leaves to read, so there is no active strategy for this context. The entry
        // points (read_ND_Data / beginReadArraystructAction) treat read_strategy == null
        // as "no data" (they return 0 / an empty AoS), which the AL then maps to a default
        // value -- exactly how the reference IMAS-Core reports a missing occurrence. Raising
        // here instead leaks a raw backend exception for a normal not-found read.
        read_strategy = nullptr;
        return;
    }

    // 3. Select/Initialize
    select_strategy(opctx, gid);
}

// Delegation of the read_ND_Data method to the read strategy
int HDF5Reader_v2::read_ND_Data(Context *ctx, std::string &dataset_name, std::string &timebasename,
                                 int *datatype, void **data, int *dim, int *size) {
  
  prepare_strategy(ctx);

  // Group for this context was not opened (occurrence not in the file): no data.
  if (read_strategy == nullptr) {
      return 0;
  }

  std::string dataset_name_copy = dataset_name;
  std::string timebasename_copy = timebasename;
  std::replace(dataset_name_copy.begin(), dataset_name_copy.end(), '/', '&');

  int status = read_strategy->read_ND_Data(ctx, dataset_name_copy, timebasename_copy, datatype, data, dim, size);

  return status;
}
