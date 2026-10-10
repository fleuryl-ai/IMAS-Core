# `imas` HDF5 Backend **v2** — Developer Guide

**Version:** v2.1 (guide) — 2026-10-07
**Supersedes:** `developer_guide_v2.md` (v1, the PanzerDB *engine* reference). Read the v1 guide for the engine API itself (`PanzerDB`, leaves, read/write methods); this guide covers the **backend v2 as a whole**: how the AL layer plugs into the engine, how a read/write session is organised, the current on-disk layout (12-column index), and every design decision that changed since v1.
**Target reader:** developer who knows C++ and HDF5 basics, and wants to understand, maintain or extend the code in `src/hdf5/`.

---

## 0. What is the "v2 backend"?

IMAS-Core ships two HDF5 backends side by side. The backend used for a given IDS is chosen by the *backend version string* stored in the file/URI; the factory dispatches:

```cpp
// src/hdf5/hdf5_backend_factory.cpp:13
std::pair<int,int> HDF5BackendFactory::getRequiredVersion(const std::string &backend_version_from_file)
{
    if (backend_version_from_file.compare("1.0") == 0) return {1, 0};   // legacy object-graph HDF5
    else if (backend_version_from_file.compare("2.0") == 0) return {2, 0}; // PanzerDB "compact index"
    ...
}
```

* **v1** (`hdf5_reader.cpp` / `hdf5_writer.cpp`): the historical backend. One HDF5 group/dataset per IMAS node.
* **v2** (`hdf5_reader_v2.cpp` / `hdf5_writer_v2.cpp`): the backend documented here. Every node lives in one flat table (`/index`) plus columnar raw datasets — the **PanzerDB engine** (`panzerdb.{h,cpp}`, documented in the v1 guide).

This v2.1 guide describes the *glue layer* built on top of the engine — the classes the AL (`al_lowlevel`) actually calls — plus the engine changes since the v1 guide was written. The most important changes, up front:

| Area | v1 guide (2026-09) | v2.1 (current code) |
|---|---|---|
| `/index` width | 14 columns (two never-read) | **12 columns** (`PANZER_INDEX_COLUMNS`, `panzerdb.h:144`); 14-column legacy files still read & append |
| Read sessions | one `PanzerDB` + one path index **per range mode** (×3) | **one shared engine + one shared `ReadIndex` per session/group** (`hdf5_reader_v2.cpp:75`) |
| Index lifetime | caches assumed frozen | **generation-checked** (`leaf_cache_generation()` / `ReadIndex::ensure_current()`), SWMR-ready |
| Time filter for a GLOBAL `get()` | O(slices²) linear scan | **binary search on time-ordered buckets** (−70 % measured, §7) |
| Strings | scalars ≤ 511 B only | `STRING_CHUNKED` scalars + **span table** `list_spans` for long list elements |
| `getWholeDynamicSignal` | one H5Dread per leaf | one `readLeavesUnion` (hyperslab union) |
| Timebase detection | `ends_with(name,"time")` | explicit AL timebase paths + named rules (`isTimebaseDataset`, §4.3) |
| Tools | `imas_h5ls`, `imas_h5dump` | + `panzer_index.py` (raw index dump), + `imas_publish_view` (VDS view) |

---

## 1. Architecture of the v2 backend

### 1.1 The stack

```
 al_lowlevel.c (AL C API: al_get_dataentry_data / al_put_dataentry_data, contexts)
        │   OperationContext / ArraystructContext / DataEntryContext
        ▼
 HDF5Backend  ──uses──►  HDF5BackendFactory ──creates──►  HDF5Reader / HDF5Writer (v1)
        │                                               HDF5Reader_v2 / HDF5Writer_v2 (v2)
        ▼
 ┌───────────────────────── HDF5Reader_v2 (read session) ─────────────────────────┐
 │  select_strategy():  GLOBAL_OP ──► GlobalReadStrategy                          │
 │                      SLICE_OP  ──► SliceReadStrategy                           │
 │                      TIMERANGE_OP ► TimeRangeReadStrategy                      │
 │  all three *share*  session_db (PanzerDB, READ) + session_index (ReadIndex)    │
 └───────────────────────────────────┬────────────────────────────────────────────┘
                                     ▼
 ┌───────────────────────── HDF5Writer_v2 (write session) ────────────────────────┐
 │  GLOBAL_OP  ──► PanzerDB(WRITE)      SLICE_OP ──► PanzerDB(APPEND)             │
 │  + schema @key replay from IDSDef.xml (MetadataExtractor)                      │
 └───────────────────────────────────┬────────────────────────────────────────────┘
                                     ▼
                         PanzerDB engine (panzerdb.h / .cpp)
        /index (N×12 u64)  /paths (N×256B)  data_raw_{f64,i32,c128,str}  [list_spans]
```

### 1.2 File map (`src/hdf5/`)

| File | Role |
|---|---|
| `hdf5_backend_factory.{h,cpp}` | version dispatch v1 ↔ v2 (`createReader` / `createWriter`) |
| `hdf5_reader_v2.{h,cpp}` | AL read entry points; session engine + strategy selection |
| `hdf5_writer_v2.{h,cpp}` | AL write entry points; PanzerDB WRITE/APPEND + `@key` metadata replay |
| `iread_strategy.h` | `ReadIndex` (shared path/name indexes) + `IReadStrategy` (helpers shared by the three strategies: leaf search, global read, time values, path sanitisation, node-role rules) |
| `global_read_strategy.*` | GLOBAL_OP strategy (whole node, all times) |
| `slice_read_strategy.*` | SLICE_OP strategy (one time slice, with interpolation) |
| `timerange_read_strategy.*` | TIMERANGE_OP strategy (slice range, resampling) |
| `aos_path_helpers.h` | single definition of the AoS path helpers (`localAosName`, `collectAosChain`, `sanitizeAosPath`) shared by writer and strategies |
| `panzerdb.{h,cpp}` | the storage engine (index table + columnar data), documented in the v1 guide |
| `metadata/metadata_extractor.{h,cpp}` | scrapes `@units`/`@description`/… from `IDSDef.xml` for the writer's metadata replay |
| `view_publisher.{h,cpp}`, `tools/` | inspection tools: `imas_h5ls`, `imas_h5dump`, `panzer_index.py`, `imas_publish_view` (VDS) |

### 1.3 The two AL verbs, end to end

* **Write** (`al_put_dataentry_data` → `al_start_rw_action`): `HDF5Writer_v2::setWriteStrategy` opens a `PanzerDB` (WRITE for a whole-IDS `put`, APPEND for a slice `put`), then each node is written through `write_ND_Data`, and `endAction` flushes/closes.
* **Read** (`al_get_dataentry_data`): `HDF5Reader_v2::open_IDS_group` opens the IDS group and resets any previous session; each `read_ND_Data` call picks the strategy matching the operation's *range mode* and delegates.

The AL always hands paths with `/` separators; **PanzerDB stores dataset names with `&`** (a node name that contains `/` becomes one flat level, e.g. `constraints/x_point` → `constraints&x_point`). The conversion is done once at the boundary:

```cpp
// hdf5_reader_v2.cpp:141  (read_ND_Data, before delegating)
std::replace(dataset_name_copy.begin(), dataset_name_copy.end(), '/', '&');
```

---

## 2. The on-disk format, current (12-column index)

### 2.1 Datasets of a v2 file

```
<IDS group or file root>
├── index        : N × 12 × uint64   (N = number of leaves; legacy files: N × 14)
├── paths        : N × 256 B, fixed  (one instance path per row, PATH_MAX_LEN=256)
├── data_raw_f64 : 1-D float64,  chunked, GZIP (level ChunkingConfig::compression_level, default 6)
├── data_raw_i32 : 1-D int32,    chunked
├── data_raw_c128: 1-D complex128, chunked
├── data_raw_str : 1-D fixed 512 B slots (STRING_MAX_LEN), chunked
└── list_spans   : 1-D uint64, 4 columns per entry (only present when a string
                   list element exceeds one 512 B slot; see §5)
```

Everything (scalars, tensors, time slices, AoS meta-nodes, metadata `@key`s) is one `/index` row + a slice of one raw dataset.

### 2.2 The 12 columns

Defined in `panzerdb.h:139-144`; rows are built in `panzerdb.cpp:2214-2240`:

```
column:   0     1 2 3 4 5 6      7          8        9      10          11
        ┌────┬────────────────┬──────────┬────────┬───────┬──────────┬────────────┐
        │ndim│ shape[0..5]    │time_index│ offset │ count │  flags   │ parent_id  │
        └────┴────────────────┴──────────┴────────┴───────┴──────────┴────────────┘
```

* `ndim` / `shape[6]` — shape of **one time-slice** (the time axis is *outside* the shape; see v1 guide Appendix B). Empty shape ⇒ scalar.
* `time_index` — first time step stored by this row (0 for static data; per-slice rows repeat the path once per step).
* `offset`, `count` — location in the matching `data_raw_*` dataset (element units).
* `flags` — low 4 bits = **node kind**, upper bits = **data type** (next section).
* `parent_id` — row id of the enclosing AoS meta-node, or `PANZER_NO_PARENT_ROW` (all-ones). This is how `parent_path` is known without a `/parent_paths` dataset.

The two legacy columns (leading `type`, trailing `index_value`) were always written as zero and never read back; dropping them cut 14 % of all `/index` I/O (write *and* read).

### 2.3 Decoding `flags`

```
flags = (DataType << 4) | kind
```

| kind (flags & 0xF) | meaning |
|---|---|
| 0 | normal data node |
| 1 | explicitly preserved empty node (also `Leaf::is_empty`) |
| 2 | **static AoS** meta-node (size in `shape[0]`, no payload) |
| 3 | **dynamic AoS** meta-node (time-evolving, no payload) |

| DataType (flags >> 4) | storage |
|---|---|
| 0 `FLOAT64` | `data_raw_f64` |
| 1 `INT32` | `data_raw_i32` |
| 2 `COMPLEX128` | `data_raw_c128` |
| 3 `STRING` | one 512 B slot (`count==1`) |
| 4 `LIST_OF_STRINGS` | one slot per element (compact) |
| 5 `STRING_CHUNKED` | scalar string split over `count ≥ 2` slots, reader concatenates |

### 2.4 `parent_path` reconstruction on read

`getLeaves()` (`panzerdb.cpp:1319+`) reads the whole `/index` + `/paths`, then rebuilds each leaf's `parent_path` *from the row data itself* (`parent_id` + kind + full path):

* data leaf (kind 0/1): parent = full path minus its last segment;
* AoS meta (kind 2/3): parent = full path minus *two* segments (the instance index and the AoS name).

The column positions differ between the two layouts, which is why the read maps both:

```cpp
// panzerdb.cpp:1321 — stride detection, 12-col current vs 14-col legacy
const uint64_t ncols = (dims[1] > 0) ? dims[1] : PANZER_INDEX_COLUMNS;
const bool legacy_layout = (ncols != PANZER_INDEX_COLUMNS);
const unsigned ndim_c   = legacy_layout ? 1  : 0;   // legacy shifts every column by 1
...                                             // and has a trailing 13th "index_value"
```

In APPEND mode the engine adopts the *file's* column count (`index_cols`), so legacy files keep their legacy layout instead of mixing strides.

### 2.5 The `Leaf` struct (what the read strategies consume)

```cpp
// panzerdb.h:202
struct Leaf {
    std::vector<size_t> shape;      // one time-slice shape (empty = scalar / meta)
    uint64_t time_index = 0;        // first time step of this row
    std::string_view path;          // zero-copy view into the engine's path storage
    std::string_view parent_path;   // rebuilt on read (see 2.4)
    uint64_t offset = 0, count = 0; // location in data_raw_*
    uint64_t flags = 0;             // (DataType<<4)|kind
    bool is_empty = false;
    uint64_t row_id = 0;            // NEW since v1: position in /index, key of the span table
};
```

⚠️ `path`/`parent_path` are views into the engine's leaf cache. They are valid until the next `getLeaves()` **rebuild** — which bumps `leaf_cache_generation()` so consumers can detect it (§4.2). Never cache a `Leaf*` across an operation boundary without checking the generation.

---

## 3. The write path — `HDF5Writer_v2`

The writer is intentionally thin: it translates AL contexts into PanzerDB calls and keeps the two in sync.

### 3.1 Opening a write session — `setWriteStrategy`

```cpp
// hdf5_writer_v2.cpp:91
void HDF5Writer_v2::setWriteStrategy(OperationContext* ctx, int write_mode, hid_t loc_id) {
  if (write_mode == GLOBAL_OP) {                     // whole-IDS put()
    panzer_db_ptr = std::make_unique<PanzerDB>(loc_id, PanzerDB::OpenMode::WRITE, true, false);
    // (re)build the schema @key map from IDSDef.xml for this IDS only
    ... MetadataExtractor extractor($IMAS_PREFIX/include/IDSDef.xml);
        auto flat = extractor.extract_metadata(ctx->getDataobjectName());
        for (kv : flat) {                            // "path@attr" -> value
            metadata_by_path[schema_path][attr] = value;   // O(log) lookup at write time
        }
  } else if (write_mode == SLICE_OP) {               // time-slice put()
    panzer_db_ptr = std::make_unique<PanzerDB>(loc_id, PanzerDB::OpenMode::APPEND, true, false);
  }
}
```

Two points worth understanding:

1. **GLOBAL_OP truncates** (a `put()` rewrites the IDS); **SLICE_OP appends** to the existing file, continuing the time counters where the previous session stopped (`PanzerDB::restoreTimeContext`).
2. The **metadata map is grouped by schema path** (`std::map<path, map<attr,value>>`). When the writer meets the node `electrons&0&density`, it strips the instance indices (`PanzerDB::stripIndices`), finds `electrons/density` in the map, and replays its `@units`, `@description`, … as `writeMetadata("electrons&density@units", …)` — *before* the data itself. The old flat design scanned every XML entry for every written node (O(N·M)); the grouped map is one `find()` per node.

`read_homogeneous_time` (`hdf5_writer_v2.cpp:39`) reads `ids_properties&homogeneous_time` on every APPEND open. It used to construct a whole READ `PanzerDB` (full `/index` load) for one `int32`; it is now a direct `H5Dopen2`+`H5Dread`, with a one-level descent into the child group holding `/index` when the IDS group is not the PanzerDB root (mirrors `PanzerDB::init`).

### 3.2 Opening an AoS — `beginWriteArraystructAction`

```cpp
// hdf5_writer_v2.cpp:122 (excerpt)
if (ctx->getTimed() && !ctx->getTimebasePath().empty())
    panzer_db_ptr->beginArray(localAosName(ctx), ctx->getTimebasePath()); // dynamic AoS
else
    panzer_db_ptr->beginArray(localAosName(ctx), static_cast<size_t>(*size)); // static AoS
initialized_aos.insert(ctx);
```

Before opening a nested AoS, the writer **synchronises PanzerDB's AoS stack with the AL context chain**:

```cpp
// hdf5_writer_v2.cpp:147 — "SYNCHRONIZATION: Necessary for nested AOS"
std::vector<std::string> aos_names; std::vector<int> indices;
collectAosChain(ctx->getParent(), aos_names, indices);   // root-first (name, index) per level
panzer_db_ptr->synchronizeArrayStack(aos_names, indices);
```

Why: the AL can jump between AoS elements in any order (it iterates the tree itself), while PanzerDB assumes a `beginArray/endArray` discipline. `collectAosChain` (`aos_path_helpers.h:47`) walks the open `ArraystructContext` parents and rebuilds the exact state (names + current indices) PanzerDB must be in. `localAosName` strips the parent's cumulative path and flattens `/`→`&`, so a level name like `constraints/x_point` stays **one** storage level.

### 3.3 Writing a node — `write_ND_Data`

`write_ND_Data` (`hdf5_writer_v2.cpp:182`) is the heart of the writer. Step by step:

1. **Re-sync the stack** (same `collectAosChain` + `synchronizeArrayStack` as above) whenever the context is an array-struct — this makes every write self-describing, whatever the AL called in between.
2. **Metadata branch**: if the node name contains `@` it is an IMAS metadata attribute → `panzer_db_ptr->writeMetadata(name, string)` and done. Otherwise the schema `@keys` replay described in §3.1 happens first.
3. **Resolve the time role of the written dimensions — once, for all types**:

```cpp
// hdf5_writer_v2.cpp:283
//  - inside a dynamic AoS: the time iteration is EXTERNAL (AL gives spatial dims only)
//  - explicit timebase outside a dynamic AoS: the LAST written dim is the time axis
//    ("bulk" mode: one call carries n_slices slices)
//  - no timebase: pure static tensor
const bool bulk_temporal = !timebasename_copy.empty() && !is_in_dynamic_aos;
```

This replaced three copy-pasted per-type blocks and a dead `size[1] > 20` string heuristic (see `to_improve.md` point 3).

4. **One dispatch, three cases** (for `double_data`, `integer_data`, `complex_data`):

| Case | Condition | Engine call | Meaning |
|---|---|---|---|
| bulk temporal | `bulk_temporal` | `writeDataSlices(name, base_shape, data, n_slices, timebase)` | last dim was time → append `shape.back()` slices in one call |
| dynamic AoS | `is_in_dynamic_aos` | `writeDataSlices(name, shape, data, 1, eff_timebase)` | one spatial slice; the time step comes from the AoS iteration counter |
| static | otherwise | `writeData(name, shape, data, total)` | no time axis at all |

Strings take the same three-way branch: `dim==1` is a scalar string (buffer length in `size[0]`), `dim==2` is a `[count,max_len]` flat buffer converted to `char**` with RAII vectors (`string_pointers` / `string_data_copies`). Writing string arrays with `dim > 2` throws.

5. **APPEND-mode guard**: in APPEND mode a node with no timebase (neither own nor from an enclosing dynamic AoS) is rejected — a slice `put()` without time context is meaningless.

### 3.4 Closing — `endAction`

```cpp
// hdf5_writer_v2.cpp:442 (logic)
ArraystructContext:  if (initialized_aos.count(ctx)) { panzer_db_ptr->endArray(); erase; }
OperationContext:    panzer_db_ptr->flush();   // buffers -> HDF5, makes data visible
                     panzer_db_ptr->close();   // closes handles
```

Only AoS contexts actually opened are closed (`initialized_aos` set) — empty AoS (`*size == 0`) never called `beginArray`, so they must not call `endArray`. Note the comment there: *do not re-synchronise in `endAction`*; the stack is already correct from the preceding writes, and empty AoS indices simply have no index rows (their meta-node already carries the size).

### 3.5 What PanzerDB does per write (engine recap)

For context when reading the code: `writeData*` never touch HDF5 directly. They append to RAM buffers (`data_buffer_*`, `index_buffer`, `paths_buffer`) and `append_index_row` records the 12-column row. `flush()` (or the destructor) extends each dataset with `H5Dset_extent` + one `H5Dwrite` per dataset — that is the only HDF5 write traffic. Dynamic-AoS time counters live in `aos_time_counters`; `advanceTimebase`/`restoreTimeContext` let APPEND sessions continue across gaps without holes in the data.

---

## 4. The read path — `HDF5Reader_v2` + strategies

### 4.1 One engine, one index, per session

The v1 code created one `PanzerDB` *and* one path index per range mode — up to 3× the `/index` load and cache memory per IDS read. v2.1 shares them:

```cpp
// hdf5_reader_v2.cpp:75 (select_strategy, condensed)
if (gid != session_gid) { session_db/session_index/strategies = reset; }   // new group = new session
if (!session_db) {
    session_db    = std::make_shared<PanzerDB>(gid, PanzerDB::OpenMode::READ);
    session_index = std::make_shared<ReadIndex>(session_db);   // /index parsed ONCE per group
}
// then, lazily per range mode:
GLOBAL_OP    -> global_strategy    = GlobalReadStrategy(session_db, session_index)
SLICE_OP     -> slice_strategy     = SliceReadStrategy(session_db, session_index)
TIMERANGE_OP -> timerange_strategy = TimeRangeReadStrategy(session_db, session_index)
read_strategy = <the one matching ctx->getRangemode()>;
```

`prepare_strategy(ctx)` (`hdf5_reader_v2.cpp:109`) is called at every entry point (`read_ND_Data`, `beginReadArraystructAction`, `endAction`): it maps the AL context to its IDS group id, then calls `select_strategy`. `open_IDS_group` resets the whole session (RAII closes the engine). Strategies' `endAction(OperationContext)` deliberately do **not** close the engine — it is shared, and dies with the session.

### 4.2 `ReadIndex` — the shared path index (and its lifetime contract)

`ReadIndex` (`iread_strategy.h:53`) builds, from `PanzerDB::getLeaves()`, two hash maps — and keeps the engine alive via `shared_ptr` because its keys are `string_view`s into the engine:

* `path_cache`: full instance path → all leaves at that path (one per time slice for per-slice storage);
* `name_index`: last path segment (dataset name, which never contains `/` after `&`-flattening) → same candidates. This makes the old O(N) "scan every leaf whose path ends with the name" fallback an O(1) bucket lookup with **identical** semantics.

Every bucket is **time-ordered** at build time (`std::stable_sort` by `time_index`), and `equal_time()` binary-searches it:

```cpp
// iread_strategy.h:171
static TimeRange equal_time(const std::vector<const Leaf*>& bucket, uint64_t time_index) {
    const auto lo = std::lower_bound(...); const auto hi = std::upper_bound(...);
    return {lo, hi};
}
```

This is what turned a full GLOBAL `get()` over 8,150-slice signals from O(slices²) to O(slices log slices) (−70 %, §7).

**Lifetime contract (SWMR-ready):** `PanzerDB::leaf_cache_generation()` is bumped exactly where the old leaf table dies (the `clear()` at the top of a `getLeaves()` rebuild). `ReadIndex` stamps the generation at build; `refresh_index_if_needed()` is called **at operation entry only** (never mid-operation, so `Leaf*` gathered during one read stay valid):

```cpp
// iread_strategy.h:258
void refresh_index_if_needed() {
    if (!read_index_ptr || !read_index_ptr->ensure_current()) return;
    time_values_cache.clear();                  // derived caches go too
    homogeneous_time_cache = HOMOG_TIME_UNRESOLVED;
}
```

A `flush()` on the engine does *not* move the generation (leaves stay usable across a commit); only a genuine rebuild does. `PanzerDB::invalidateLeafCache()` is the SWMR refresh hook.

### 4.3 Shared helpers in `IReadStrategy`

All three strategies inherit these; understanding them explains ~80 % of the read code.

**Path reconstruction.** `collectAosChain` (shared header) yields the open AoS `(names, indices)` chain; `getPath(ctx, include_self_index, override_timed_index)` and `buildFullPath(ctx, dataset)` turn it into `A/0/B/1/dataset`. `sanitize_path(ctx, path)` normalises an AL path (possibly partial, possibly `/`-prefixed) against the open context chain via `sanitizeAosPath` (`aos_path_helpers.h:75`). The match of a context path is **anchored at the head** of the string — AL paths are cumulative from the IDS root, so a repeated segment name higher in the hierarchy can no longer steal the match (`to_improve.md` point 5). Results are cached per `(ctx, path)`.

**Leaf lookup.** `find_leaf_for_context` (`iread_strategy.h:480`) builds the *strict* instance path (`AOS/i/.../name`) and:
1. looks it up in `path_cache` (O(1)); when the node is dynamic, narrows to the requested `time_index` with the binary-search `equal_time`;
2. falls back to `name_index` with the same time filter and a context-prefix constraint — the exact old full-scan candidate set, in O(1).

**The global read.** `read_dataset_globally` (`iread_strategy.h:905`) is what GLOBAL_OP (and the static shortcut of the other two) actually calls. Its search ladder:

```
specific path (AOS/i/.../name)                  -- exact instance
   └─ miss → generic path (AOS/.../name, dynamic levels keep no index)
                └─ miss → name_index fallback (read from a parent, e.g. "profiles_1d/signal")
```

…collects all matching leaves (time-ordered), then:

* **scalar-overwrite fix**: several static scalar rows at one path (successive overwrites) ⇒ keep the **last** one only.
* **strings**: compute the union slot range `[min(offset), max(offset+count))` of the leaves and fetch it in **one** `readStringBulk` H5Dread; scatter locally, joining `STRING_CHUNKED` scalars and spanned list elements (§5). The old code did one tiny H5Dread per slice — O(N) round-trips, the cause of "reading never returns" on 10⁵-slice files.
* **numerics**: one `readLeavesUnion` (hyperslab union, `H5S_SELECT_OR`, fast path on monotonic offsets).
* **output shape rules** (the AL contract): a rank-0 signal with >1 stored element returns 1-D `[n_values]`; a rank-0 signal with exactly 1 element returns 1-D `[1]` **only if** it reads as time-dependent (`shouldPromoteTimeScalarOnSlice`) or the AL dictionary expects 1-D (`expected_dim == 1`, captured before any branch overwrites `*dim`); a rank-k signal with multiple slices gets an appended time dimension `[shape..., n_slices]`, single slice ⇒ rank unchanged.

**Time values.** `getTimeValues(ctx, homogeneous_time, timebasename)` resolves the time base of a context:

* `homogeneous_time==1` → the global `time` signal, read once per session via `getWholeDynamicSignal` (itself one `readLeavesUnion` since v2.1).
* otherwise: try `<timed-AoS-path>/time` in `path_cache`; else generate `<timed-AoS>/i/time` for each element index; else linear-scan fallback. Result cached per `(timed_aos_path, timebase_basename)`.
* `getHomogeneousTime()` reads `ids_properties&homogeneous_time` once per session and caches it (`HOMOG_TIME_UNRESOLVED = -2` sentinel; real values −1/0/1).

**Node-role rules (v2.1, replaces heuristics).** `collectTimebaseSchemaPaths` + `isTimebaseDataset` decide "is this dataset *the* timebase?" by matching the AL-supplied `timebasename` and the nearest timed AoS's `getTimebasePath()` (schema-normalised via `stripIndices`), with an exact-final-segment-`time` legacy fallback. The old `ends_with(name, "time")` swallowed `time_step`, `lifetime`, … `shouldPromoteTimeScalarOnSlice` encodes the scalar-promotion AL convention (root/static-AoS time scalars get a size-1 slice dim; dynamic-AoS scalars stay scalars).

### 4.4 `GlobalReadStrategy` — GLOBAL_OP

The whole node, all stored times:

* `beginReadArraystructAction`: `getAOSShape(path)` fills the AoS size (for dynamic AoS: number of stored slices).
* `read_ND_Data`: `return read_dataset_globally(...)` — that's all the file does (`global_read_strategy.cpp:50`).

### 4.5 `SliceReadStrategy` — SLICE_OP

One time slice at `ctx->getTime()`:

```cpp
// slice_read_strategy.cpp:78 (condensed)
if (timebasename.empty() && !isTimedContext(ctx))
    return read_dataset_globally(...);            // static node: no time involved

time_basis = getTimeValues(ctx, homogeneous_time, timebasename);
res = panzer_db_ptr->readInterpolatedData(full_path, time, time_basis, interp, ...);
if (shouldPromoteTimeScalarOnSlice(ctx, timebasename, *datatype)) { size[*dim] = 1; (*dim)++; }
```

`beginReadArraystructAction` resolves `slice_idx = getTimeIndex(timebase_path, time, interp)` and reports the AoS size with that dynamic index substituted (`getPath(ctx, false, slice_idx)`) — a timed AoS reports size 1 (the one selected slice). `readInterpolatedData` internally handles **gaps**: it resolves the requested time to the nearest available slice (`nearestAvailableSliceIndex`, closest/previous/linear modes) instead of assuming every time index exists.

### 4.6 `TimeRangeReadStrategy` — TIMERANGE_OP

The range `[tmin, tmax]` (optionally resampled onto `dtime`). The interesting v2.1 behaviour (from the gap fix):

* The timebase dataset itself is special-cased via `isTimebaseDataset` / `is_homogeneous_time_basis_dataset` (`timerange_read_strategy.cpp:92`): the *time* signal with resampling is produced by `resample_timebasis`, never by slicing data.
* For a **standalone dynamic signal** (no dynamic-AoS context, numeric types), the strategy reads **one resolved slice per requested time** (`readInterpolatedData` in a loop, `timerange_read_strategy.cpp:197-283`). Reading one concatenated global buffer and slicing it by element offset was wrong when slices were missing: a gap shifted later values and allowed reads past the buffer end.
* Strings and dynamic-AoS contexts keep the "read globally then slice" path below that block.
* Output follows the AL convention: spatial dims first, appended time dim last.

### 4.7 Choosing: which strategy does what to my call?

| AL range mode | Typical call | Strategy | Engine primitives |
|---|---|---|---|
| GLOBAL_OP | `get()` of a whole IDS | Global | `readLeavesUnion` / `readStringBulk` / `getAOSShape` |
| SLICE_OP | `get()` with `time_selection` | Slice | `getTimeIndex`, `readInterpolatedData` |
| TIMERANGE_OP | `get()` with `time_range_selection` | TimeRange | `getTimeRangeIndices`, `resample_timebasis`, per-time `readInterpolatedData` |
| AOS read | iteration over AoS elements | same strategy, via `beginReadArraystructAction` | size resolution + per-element `find_leaf_for_context` |

---

## 5. Strings: 512 B slots, `STRING_CHUNKED`, and the span table

`data_raw_str` is **fixed-width**: 512 B slots (`STRING_MAX_LEN=512`, `panzerdb.cpp:60`). This keeps the file SWMR-safe (no variable-length type), but IMAS allows arbitrary-length strings, so v2.1 added two mechanisms:

* **Scalar longer than one slot** → one row, `count = k` slots, `DataType::STRING_CHUNKED`; the reader concatenates the slots (see `read_dataset_globally`'s string scatter and `readMetadata`'s join).
* **List element longer than one slot** → the list leaf stays one row, and each element's placement is recorded in the **`list_spans` dataset**: 4 u64 per entry `(row_id, element_idx, slot_offset, slot_span)`. Read-side access:

```cpp
// panzerdb.h:955
if (panzer_db_ptr->isListSpanned(leaf->row_id)) {
    for (const auto& se : panzer_db_ptr->getListSpans(leaf->row_id))
        for (uint64_t k = 0; k < se.slot_span; ++k)
            joined += slots[base + se.slot_offset + k];   // element = concat of its slots
}
```

The span table is loaded lazily and cached per row id; a file without long strings has no `list_spans` dataset at all (compact lists: one slot per element, `slot_span == 1` implied).

The scalar-vs-list contract is shape-driven and centralised: `PanzerDB::isStringScalarLeaf` (`panzerdb.h:517`) — empty shape ⇒ scalar string, rank-1 shape ⇒ list, whatever the element count.

---

## 6. Metadata (`@key` attributes)

Metadata entries are ordinary static string leaves whose path carries an `@` suffix: `flux_loop&field@units = "T"`.

* **Write**: two sources, both funnelling into `PanzerDB::writeMetadata` (deduplicated per session via `written_metadata_schema_paths`):
  1. explicit AL writes (node name contains `@`, §3.3);
  2. **schema replay** from `IDSDef.xml` (`MetadataExtractor`, §3.1), so a v2 file carries the dictionary's `units`, `description`, … for free.
* **Read**: `PanzerDB::readMetadata(instance_path)` strips indices to the schema path and resolves that schema's `@key` leaves through `metadata_by_schema`, a hash map built once inside `getLeaves()` (was an O(N) prefix scan *per read call*). Strategies call it on every node read (harmless and cheap now); it also marks the schema processed.
* Values are stored in the map as **indices into `cached_leaves`**, not `Leaf*` — the map is built while `cached_leaves` still grows, and stored pointers would dangle on reallocation (this exact bug caused a SEGFAULT in `test_direct_api_validation` during development; the index-based design is the guard).

---

## 7. Performance notes (measured)

From the read-parameter study (`to_improve.md` point 13, file `core_profiles_3.h5`: 171k rows, 28 signals × 8,150 per-slice leaves, GLOBAL get):

| change | effect |
|---|---|
| time-ordered buckets + binary-search time filter | GLOBAL get **3707 ms → 1105 ms** (−70 %) |
| `readLeavesUnion` for `getWholeDynamicSignal` | O(#chunks) tiny reads → 1 union read |
| one shared engine/index per session (was 3) | 3× `/index` loads → 1 |
| `metadata_by_schema` at build time | O(N) scan per read → 1 hash lookup |
| `readStringBulk` union range | per-leaf string reads → 1 H5Dread |
| 12-column `/index` | −14 % of index I/O |
| chunk size / HDF5 chunk cache tuning | **no measurable effect** for this pattern (each chunk touched once) — defaults unchanged |
| compression off | −21 % read time, but ×2.1 file size → **compression kept** (deflate level 1 was the cheapest useful setting in that study; production default is level 6) |

Tuning hooks if you need them: `PanzerDB::setDefaultChunkingConfig()` (process-wide, must be set *before* construction — chunking is decided in `init()`), `setReadCache()`, and the `IMAS_HDF5_FAPL` env override in `HDF5Utils::openIDSFile` (unset ⇒ `H5P_DEFAULT`).

---

## 8. Errors & memory ownership

* **Ownership across the AL boundary**: every `read_ND_Data`-style output buffer is `malloc`-style and caller-freed. Internally, allocation is RAII (`std::unique_ptr<T[]>` / `make_unique`): the buffer is `release()`d to the caller **only on success**; any throw (including inside `readLeavesUnion`/`readTensor`) frees it. The remaining manual `malloc/free` juggling lives in `TimeRangeReadStrategy` around `PanzerDB::readInterpolatedData` (its C-style API still returns raw pointers) — known open point (`to_improve.md` §7).
* **Exceptions**: backend-level problems throw `ALBackendException(..., LOG)`; engine errors (`std::exception`) are caught in `readLeafData` and rewrapped as `ALBackendException("PanzerDB read error: ...")`. A read that finds nothing returns `0`/empty, it does **not** throw.
* **Empty data**: `total_elements == 0` ⇒ `*data = nullptr`, `*dim = 0`, return success. Empty leaves (kind 1) exist in the index but carry no payload.

---

## 9. Inspection tools (`src/hdf5/tools/`)

| Tool | What it shows |
|---|---|
| `imas_h5ls <file> [-r] [-a] [-m]` | the **logical** IMAS tree reconstructed from `/index` (no `.h5` suffix) |
| `imas_h5dump <file> [path]` | values + metadata; supports slice syntax `profiles_1d[0:5]/grid/rho` |
| `panzer_index.py <f.h5> [--limit N --match R --kinds ... --raw --no-parent]` | the **physical** index table, decoded; handles both 12- and 14-column layouts; derived `slices` column and `timebase(s)` header. Plain Python + h5py, no build step |
| `imas_publish_view <f.h5> [--root N] [--expand] [--max-mappings N] [--stats]` | builds an idempotent **VDS** view under `imas_view/` so `h5ls`/`h5dump`/`h5py` can browse the file with zero IMAS dependency. Purely additive; never touches `index/paths/data_raw_*` |
| `imas_publish_view <f.h5> --list [--root N]` | **read-only, publishes nothing**: prints the view that is already in the file, one annotated line per object (dims, `@dims_label`, `@ragged`, `@aos_size`, `holes=…`). Exits 2 when the file has no view |

**Collapse mode (default) — one dataset per *logical* node.** Every Array-of-Structures level and every time step becomes a **leading dimension** instead of a separate HDF5 object, so `h5ls -r` lists the IMAS tree rather than one path per instance (verbatim output):

```
/imas_view/time_slice/boundary&psi_norm                                 Dataset {195}
/imas_view/time_slice/constraints&flux_loop/measured                    Dataset {195, 17}
/imas_view/time_slice/boundary&outline&r                                Dataset {195, 519}
/imas_view/grids_ggd/grid/space/objects_per_dimension/object/nodes      Dataset {3, 13857, 3}
```

The same objects with their stored annotations (read back with `h5dump -A` or h5py — see below):

```
/imas_view/time_slice                                     Group  @aos_size=195 @aos_dynamic=1
/imas_view/time_slice/boundary&psi_norm                   {195}        @dims_label=time_slice @time_dim=195 @n_mappings=97
/imas_view/time_slice/boundary&outline&r                  {195, 519}   @dims_label=time_slice|boundary&outline&r @ragged=1 @n_mappings=191
```

dims = `[n_time?, n_aos_0, …, n_aos_k-1, *slice_shape]`; AoS levels outermost→innermost; a leading time dim appears when one logical path is stored as several index rows (per-slice rows *or* one bulk row with `count == n_slices`). Singleton AoS levels are squeezed (`grids_ggd/0/grid/0/…` → `grids_ggd/grid/…`). Unmapped cells keep the IMAS invalid value as HDF5 fill: `-9E+40` for reals (both components of a complex), `-999999` for integers, empty string for text. Attributes: `@datatype`, `@n_elements`, `@schema`, `@aos_ndim` (how many leading dims are AoS/time), `@dims_label` (one name per dim), `@n_mappings`, plus `@time_dim` / `@ragged` / `@mapping_cap` when relevant; AoS groups carry `@aos_size` and `@aos_dynamic=1`. **Object names never contain `[`** — the HDF5 CLI parses brackets as a selection, which makes such an object unreachable by `h5dump -d` (the v1 limitation); `&` is harmless.

**How to see the annotations** — `h5ls -r` prints **no attribute at all** (verified on HDF5 1.14.4; it is not a version quirk), so the annotations above are invisible in a plain recursive listing. The one-liner is the tool's own read-only listing:

```bash
imas_publish_view f.h5 --list      # publishes nothing: reads the view back, one annotated line per object
```

```
/imas_view/grids_ggd                                                    Group @aos_size=1 @aos_dynamic
/imas_view/time_slice/boundary&outline&r                               Dataset {195, 519} @dims_label=time_slice|boundary&outline&r @ragged holes=31105
/imas_view/grids_ggd/grid/space/objects_per_dimension/object/nodes     Dataset {3, 13857, 3} @dims_label=objects_per_dimension|object|nodes holes=83142
```

`holes=N` is derived at print time (`product(dims) − @n_elements`), so it needs no extra attribute and appears only when non-zero; it counts cells left to the fill value by ragged slices or sparse AoS instances. The file is opened `H5F_ACC_RDONLY` — `--list` cannot change it. On `equilibrium_1.h5`: `h5ls -r` = 124 lines (9 of them the PanzerDB tables + the `/` root, 115 inside the view including the `imas_view` group itself), `--list` = 114 (the view contents only). When the file holds no view, `--list` prints a hint and exits 2.

For the raw attribute values (including `@n_mappings`, `@datatype`, `@n_elements`, which `--list` keeps out of the line), use `h5ls -v -r` for *names* (6 lines per object) or `h5dump` for *values* — `-d` selects a dataset, `-g` selects a group (`-d` on a group errors with *"is a hard link"*). Attributes are stored as `H5T_IEEE_F64LE` scalars or `H5T_C_S1` strings:

```bash
h5ls   -v -r f.h5                                   # attribute names (6 lines per object)
h5dump -H -A -d "imas_view/time_slice/boundary&outline&r" f.h5   # dataset: names + values
h5dump -H -A -g "imas_view/time_slice" f.h5                       # group: @aos_size, @aos_dynamic
```

Note `-A` (all attributes), not `-a NAME` (selects one attribute by name).

`--expand` restores the historical view (one VDS per index row). It is kept for compatibility only: rows sharing one exact path collide, so `equilibrium_1.h5` publishes 43 596 VDS and **loses 12 998 rows** (the tool reports them and exits 1). Collapse mode publishes the same file in 100 datasets with 0 errors. `--max-mappings N` is a safety valve (a dataset exceeding it keeps its first mappings and is flagged `@mapping_cap=1`); the measurements below put the worst dataset at 195 mappings, so the default (unlimited) is safe.

Measured on the four reference files (collapse mode / `--stats`): `equilibrium_1` 100 datasets, 8 658 mappings, 145 353 fill cells, +975 KiB, 2.2 s; `core_profiles_3` 28 / 28 / 0, +872 KiB, 4.2 s; `reflectometer_profile_v2` 33 / 33 / 0, +42 KiB; `magnetics_THIN` 273 / 273 / 0, +225 KiB. `h5ls -r` line count drops 74 652 → **124** (`equilibrium_1`) and 73 389 → **45** (`core_profiles_3`); files already storing one row per path (`reflectometer_profile_v2`, `magnetics_THIN`) gain readability rather than length. VDS caveat: the source filename is baked in; renaming or moving the file breaks the view.

---

## 10. Known limitations / open points

Tracked in `to_improve.md`; the ones still open as of v2.1:

* **#3 (partial)**: node *roles* (timebase, shape role) are still inferred from AL context + shape; a fully file-level persistence (`@timebasepath` attribute or an index role bit) would make role resolution independent of the calling context.
* **#7 (partial)**: `readInterpolatedData`'s raw-pointer API still forces manual `malloc/free` error paths in `TimeRangeReadStrategy`.
* **#8**: `HDF5Reader_v2::endAction` guard `IDS_group_id.size() == 1` — with several IDS groups open, `endAction` does nothing; the key test should be the context's own group.
* **#9**: no thread safety — mutable per-session caches plus process-wide `static` flags (`HDF5Writer_v2::compression_enabled`, chunk cache sizes) race across parallel sessions. Treat one backend instance = one thread.

---

## 11. How to extend the backend (checklist)

* **New range mode / read behaviour** → subclass `IReadStrategy`, reuse `read_dataset_globally`/`find_leaf_for_context`/`getTimeValues`; wire it in `HDF5Reader_v2::select_strategy`. Never call `refresh_index_if_needed()` mid-operation.
* **New stored type** → add a `PanzerDB::DataType`, a `data_raw_*` dataset + buffer + flush path in `init/flush`, and the mapping in `read_dataset_globally`'s type switch.
* **New path normalisation rule** → change it **once** in `aos_path_helpers.h` (and its unit test `test_path_sanitization`), not in a strategy.
* **Index layout change** → bump `PANZER_INDEX_COLUMNS`, update the column map in `getLeaves()` (keep the legacy stride branch), and `panzer_index.py`; APPEND adopts the file's extent so old files keep their layout.
* **Verification** → `ctest -R "test_al_|test_engine_|test_swmr_|test_index_golden|test_direct_api_validation|test_path_|test_read_node_roles"`; the suite is the contract.

---

## Change log

* **v2.1 (2026-10-07)** — first guide for the backend **v2 as a whole** (AL ↔ engine layer). Documents: the shared session engine + `ReadIndex` with the generation-checked lifetime contract; the 12-column `/index` (and 14-column legacy compatibility); `STRING_CHUNKED` + `list_spans` span table; `@key` metadata replay and `metadata_by_schema`; the three read strategies with the gap-safe TIMERANGE path and the named node-role rules; the writer's single time-role resolution and AoS-stack synchronisation; tools (`panzer_index`, `imas_publish_view`); measured performance history. See `developer_guide_v2.md` (v1) for the PanzerDB engine API reference itself, and `to_improve.md` for the change-by-change rationale.

---

*End of guide.*
