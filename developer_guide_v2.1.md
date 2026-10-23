# `imas` HDF5 Backend **v2** — Developer Guide

**Version:** v2.2 (guide) — 2026-10-21
**Supersedes:** `developer_guide_v2.md` (v1, the PanzerDB *engine* reference). Read the v1 guide for the engine API itself (`PanzerDB`, leaves, read/write methods); this guide covers the **backend v2 as a whole**: how the AL layer plugs into the engine, how a read/write session is organised, the current on-disk layout (7-column packed index + interned path table), and every design decision that changed since v1.
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

This v2.2 guide describes the *glue layer* built on top of the engine — the classes the AL (`al_lowlevel`) actually calls — plus the engine changes since the v1 guide was written. The most important changes, up front:

| Area | v1 guide (2026-09) | v2.2 (current code) |
|---|---|---|
| `/index` width | 14 columns (two never-read) | **7 packed columns + an interned `/path_table`** (`PANZER_INDEX_COLUMNS_V3`, `panzerdb.h:154`); 12-column and 14-column files are still read and appended (§2) |
| Path text | one 256 B `/paths` row per index row | the path text is stored **once per distinct path** in `/path_table`; `/paths` is legacy-only and `/parent_paths` is gone, rebuilt on read (§2.1, §2.4) |
| Read sessions | one `PanzerDB` + one path index **per range mode** (×3) | **one engine + one `ReadIndex` for the whole read session**, moved onto each operation's group id by `PanzerDB::reanchor()` (§4.1) |
| Write sessions | one engine per operation | **one `PanzerDB(APPEND)` per slice session**, flushed per operation and closed at `closePulse()` (§3.1) |
| Index lifetime | caches assumed frozen | generation-checked (`leaf_cache_generation()` / `ReadIndex::ensure_current()`) **and parked between engines** (`IndexHandoff`), so a pulse decodes its table once (§3.6, §4.2) |
| Time filter for a GLOBAL `get()` | O(slices²) linear scan | binary search on time-ordered buckets (`ReadIndex::equal_time`) plus `PanzerDB::rowsForTime()` in the engine (§4.2, §4.3) |
| Strings | scalars ≤ 511 B only | `STRING_CHUNKED` scalars + **span table** `list_spans`, created only when something needs it (§5) |
| `getWholeDynamicSignal` | one H5Dread per leaf | one `readLeavesUnion`, **memoised** for as long as the decoded index is unchanged (§4.3, §7) |
| Timebase detection | `ends_with(name,"time")` | explicit AL timebase paths + named rules (`isTimebaseDataset`), including time bases that sit on an AoS element (§4.3, §4.6) |
| Tools | `imas_h5ls`, `imas_h5dump` | + `panzer_index.py` (decodes v3/v2/legacy), + `imas_publish_view` (VDS view, with the read-only `--list`) |

**What changed since the previous revision of this guide (2026-10-07)** — the parts a maintainer has to know before touching the code:

* the **v3 index format**: 7 packed u64 columns and interned paths (§2);
* **sessions, not operations, own the engine**: the reader re-anchors one `PanzerDB` across AL operations (§4.1), the writer keeps one APPEND engine for a whole slice session (§3.1), and the decoded table is **handed between engines** through `IndexHandoff` (§3.6);
* the **time base is read once per session** (`getWholeDynamicSignal` memo + a "this signal is not in the file" negative cache), which is what removed the per-slice cost of an optional signal (§7);
* `/list_spans` is deleted with the other tables on a WRITE and **created on demand** on an APPEND (§5) — a stale span table used to describe new data with old element boundaries, silently;
* the filter pipeline order (`shuffle` **before** `deflate`) is fixed and pinned by a test (§3.6, §7);
* a v1 pulse can be **upgraded to 2.0** (erase the old layout, relabel the file) and the v2→v1 conversion is behind an explicit switch (§1.4);
* the published **VDS view shows every AoS level as a leading dimension**, sized by its declared `@aos_size` (§9).

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
 │  all three *share*  session_db (PanzerDB, READ) + session_index (ReadIndex);   │
 │  the AL opens a new group id per operation, so the engine is *re-anchored*     │
 │  onto it (PanzerDB::reanchor) instead of rebuilt — see §4.1                    │
 └───────────────────────────────────┬────────────────────────────────────────────┘
                                     ▼
 ┌───────────────────────── HDF5Writer_v2 (write session) ────────────────────────┐
 │  GLOBAL_OP  ──► PanzerDB(WRITE)      SLICE_OP ──► PanzerDB(APPEND),            │
 │                                                     kept for the whole pulse   │
 │  + schema @key replay from IDSDef.xml (MetadataExtractor)                      │
 └───────────────────────────────────┬────────────────────────────────────────────┘
                                     ▼
                         PanzerDB engine (panzerdb.h / .cpp)
   /index (N×7 u64)  /path_table (unique paths)  data_raw_{f64,i32,c128,str}  [list_spans]
   + a parsed-index handoff between engines on one file (§3.6)
```

### 1.2 File map (`src/hdf5/`)

| File | Role |
|---|---|
| `hdf5_backend.{h,cpp}` | the AL-facing backend: opens the pulse, picks the version, owns the v1↔v2 upgrade/conversion switches and the write-version env override |
| `hdf5_backend_factory.{h,cpp}` | version dispatch v1 ↔ v2 (`createReader` / `createWriter`) |
| `hdf5_reader_v2.{h,cpp}` | AL read entry points; **session engine** (`session_db` + `session_index`), strategy selection, `reanchor`, `release_session` |
| `hdf5_writer_v2.{h,cpp}` | AL write entry points; PanzerDB WRITE/APPEND + `@key` metadata replay; owns the APPEND engine until the pulse ends |
| `iread_strategy.h` | `ReadIndex` (shared path/name indexes) + `IReadStrategy` (helpers shared by the three strategies: leaf search, global read, time values, path sanitisation, node-role rules) |
| `global_read_strategy.*` | GLOBAL_OP strategy (whole node, all times) |
| `slice_read_strategy.*` | SLICE_OP strategy (one time slice, with interpolation) |
| `timerange_read_strategy.*` | TIMERANGE_OP strategy (slice range, resampling) |
| `aos_path_helpers.h` | single definition of the AoS path helpers (`localAosName`, `collectAosChain`, `sanitizeAosPath`) shared by writer and strategies |
| `panzerdb.{h,cpp}` | the storage engine (index table + columnar data), documented in the v1 guide |
| `metadata/metadata_extractor.{h,cpp}` | scrapes `@units`/`@description`/… from `IDSDef.xml` for the writer's metadata replay |
| `direct_access_api.cpp`, `path_parser.{h,cpp}`, `direct_reader.{h,cpp}`, `imas_h5read/` | the **direct-access read path**: a parsed path in, a `TensorView` out, reading PanzerDB leaves without AL contexts (`imas_h5read` is the MATLAB/Fortran front end built on it). `DirectReader` itself is still a narrow proof of concept |
| `view_publisher.{h,cpp}`, `tools/` | inspection tools: `imas_h5ls`, `imas_h5dump`, `panzer_index.py`, `imas_publish_view` (VDS view) |

### 1.3 The two AL verbs, end to end

* **Write** (`al_put_dataentry_data` → `al_start_rw_action`): `HDF5Writer_v2::setWriteStrategy` opens a `PanzerDB` (WRITE for a whole-IDS `put`, APPEND for a slice `put` — and one APPEND engine is kept for the whole slice session), each node goes through `write_ND_Data`, `endAction` **flushes** (the commit point), and the engine itself is released at `closePulse()` / `deleteData()`.
* **Read** (`al_get_dataentry_data`): `HDF5Reader_v2::open_IDS_group` opens the IDS group **without dropping the session engine**; `prepare_strategy` → `select_strategy` re-anchors that engine on the new group id, and each `read_ND_Data` call picks the strategy matching the operation's *range mode* and delegates. The session ends at `closePulse()` or `release_session()`.

The AL always hands paths with `/` separators; **PanzerDB stores dataset names with `&`** (a node name that contains `/` becomes one flat level, e.g. `constraints/x_point` → `constraints&x_point`). The conversion is done once at the boundary:

```cpp
// hdf5_reader_v2.cpp:155  (in read_ND_Data, opened at :148, before delegating)
std::replace(dataset_name_copy.begin(), dataset_name_copy.end(), '/', '&');
```

### 1.4 Which backend reads a file, and what happens across generations

The version stored in the pulse decides the reader/writer class; nothing in the file format is versioned by the backend label (see the no-bump rule):

```cpp
// hdf5_backend.cpp:27-28
const int HDF5Backend::HDF5_BACKEND_VERSION_MAJOR = 2;
const int HDF5Backend::HDF5_BACKEND_VERSION_MINOR = 0;
```

Three environment switches, all read per session (`hdf5_backend.cpp:52-63`), control what happens when the label and the requested target disagree:

| Switch | Effect |
|---|---|
| `BACKEND_DEFAULT_WRITE_VERSION` | the version this process writes when the file has no label (otherwise 2.0) |
| `BACKEND_ALLOW_AUTO_UPGRADE=true` | lets a **v1 pulse be rewritten as 2.0**: without it the write throws (`hdf5_backend.cpp:135`). With it, the write **erases the v1 object graph first** and re-labels the file — a half-upgraded pulse is worse than a refused one, because the next open would read the old "1.0" label and hand the file to the v1 reader, which never sees what the upgrade wrote |
| `BACKEND_HDF5_FORCE_CONVERSION_TO_V1=true` | the other direction, for tests only (`hdf5_backend.cpp:154`); without it a v2→v1 rewrite is refused |

Guarded by tests: `test_al_upgrade_v1_to_v2` (erase + relabel) and `test_al_convert_v2_to_v1` (the lock stays when the switch is off). Caveat when reading a converted pulse: the **v1 AL read path is not maintained in this tree**, so a conversion test must assert the *storage* (dataset names, extents, stored values), not an AL-level re-read. And note that `test_al_convert_v2_to_v1` checks the absence of the v2 tables with `H5Lexists` **at the file root**, where those names have never lived (§2.1) — that check is vacuously true; it is left as is because the erase deletes the whole IDS file, so it was never what protected the conversion.

---

## 2. The on-disk format, current (v3: 7 packed columns + `/path_table`)

Three index generations coexist in the wild and **all three stay readable**. What a *new* file gets is **v3**; a file written before that keeps its own layout, and an APPEND never mixes two of them.

### 2.1 Datasets of a v2 file

```
<IDS group or file root>
├── index        : N × 7 × uint64   (N = number of rows)  + attribute "index_version" (uint32)
├── path_table   : M × 256 B, fixed (M = number of DISTINCT path texts, PATH_MAX_LEN=256)
├── data_raw_f64 : 1-D float64,  chunked, shuffle + GZIP
├── data_raw_i32 : 1-D int32,    chunked, shuffle + GZIP
├── data_raw_c128: 1-D complex128, chunked, shuffle + GZIP
├── data_raw_str : 1-D fixed 512 B slots (STRING_MAX_LEN), chunked
└── list_spans   : 1-D uint64, 4 u64 per entry — created only when something needs it (§5)

(older files also carry: paths — N × 256 B, row-aligned with /index — and, older still,
 parent_paths — retired since M1, ignored on read, still deleted by a WRITE so that a
 file from an earlier generation cannot leave a stale table behind: panzerdb.cpp:277-291)
```

Note where the objects live: **inside the group named after the IDS** (`core_profiles/index`, `core_profiles/path_table`, …), not at the file root. A `H5Lexists(f, "index")` written at the root is vacuous, and so is any "absent" assertion made there.

Storage parameters, all **frozen when the file is created** (`init()`, and therefore not changeable on an existing file):
* the usage-hint profile — `time_series` for WRITE/APPEND, `interactive` for READ (`panzerdb.cpp:261-263`); `time_series` means index chunk 4096 rows, `data_raw_f64` 65 536 elements, i32 131 072, c128 32 768, str 128 slots, path table 4096 entries, **deflate level 1** (`configureChunking`, `panzerdb.cpp:614-674`);
* filter order: **`shuffle` added first, `deflate` last** (`panzerdb.cpp:315-316` and `:713-715`). HDF5 applies filters in add-order; reversing them leaves the shuffle permuting already-compressed bytes, and every numeric buffer costs ~+20 % of file size (see §7 and `test_engine_filter_pipeline_order`);
* `H5D_FILL_TIME_NEVER` + `H5D_ALLOC_TIME_INCR`;
* the by-name constructor aligns objects on 4 MB (`H5Pset_alignment(fapl, 4096, 4 MB)`, `panzerdb.cpp:179`) for Lustre striping. That inflates the **address space**, not the allocated blocks: on this Lustre a padded file reads as sparse holes (`stat` ≫ `du`). Compare `H5Dget_storage_size` sums, `stat -c%s` and `du -B1` before concluding anything about size.

Everything (scalars, tensors, time slices, AoS meta-nodes, metadata `@key`s) is one `/index` row + a slice of one raw dataset.

### 2.2 The layouts, and how the engine picks one

The version is an attribute of `/index`; a file without it is classified from the dataset's real column extent:

```cpp
// panzerdb.h:144-165
inline constexpr uint64_t PANZER_INDEX_COLUMNS = 12;        // v2 layout (u64 per column + row-aligned /paths)
inline constexpr uint32_t PANZER_INDEX_VERSION_V2 = 2;
inline constexpr uint32_t PANZER_INDEX_VERSION_V3 = 3;      // new files
inline constexpr uint64_t PANZER_INDEX_COLUMNS_V3 = 7;      // packed u64 columns + /path_table
inline constexpr uint32_t PANZER_NO_PARENT_ROW_V3 = 0xFFFFFFFFu;
inline constexpr uint32_t PANZER_NO_PATH_ROW_V3   = 0xFFFFFFFFu;
inline constexpr uint64_t PANZER_V3_NDIM_SHIFT = 8;
```

| Layout | Columns | Path text | Written by |
|---|---|---|---|
| **v3** (new files) | 7 packed u64 | `/path_table`, one entry per **distinct** path | `PANZER_INDEX_VERSION_V3`, set in the WRITE branch of `init()` (`panzerdb.cpp:297-299`) |
| **v2** | 12 u64 | `/paths`, 256 B **per row** | files from before the v3 change |
| **legacy** | 14 u64 | `/paths` + `/parent_paths` | the earliest v2 files; the two retired columns are `type` (already in `flags >> 4`) and `index_value` (already in the path text) |

**v3 packing** (writer: `append_index_row`, `panzerdb.cpp:3057-3085`; reader: `getLeaves`, `panzerdb.cpp:1865+`):

```
col0  offset (u64)
col1  count  (u64)
col2  path_id (u32, low) | parent_id (u32, high; 0xFFFFFFFF = no parent)
col3  time_index (u32, low) | (flags | ndim<<8) (u32, high)
col4  shape[0] | shape[1]<<32     col5  shape[2] | shape[3]<<32     col6  shape[4] | shape[5]<<32
```

* `Leaf::flags` keeps its public meaning (`kind | DataType<<4`): `ndim` lives in bits 8..11 of the stored word and is masked off before it reaches a `Leaf`.
* Because the small fields are u32, `append_index_row` **throws** rather than truncating when `time_index`, a parent row id, or a shape dimension exceeds `0xFFFFFFFF`, and `intern_path` when the path-id space is exhausted (`panzerdb.cpp:1208-1215`). A pulse with more than ~4·10⁹ rows or distinct paths is not what this format is for.
* `count` remains element units of the matching `data_raw_*`; `shape` is still the shape of **one time-slice** (the time axis stays outside the shape); `time_index` is the first step stored by the row. A row may cover several steps (`count = slice_volume × n_steps`) — see §3.5.
* The path *text* is emitted only for a first-occurring path (`intern_path` + `flushPathTable`); the same signal written 8 000 times shares one 256 B entry. That is the point of v3: the wide text column used to be repeated once per row.

In APPEND the engine adopts the **file's** extent (`index_cols = idims[1]`) and its `index_version`, so a v2 file keeps being appended to as a v2 file (`panzerdb.cpp:414-427`).

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

### 2.4 Reading the index back: `getLeaves()`

`getLeaves()` (`panzerdb.cpp:1788`) is the single entry point every read path uses. Per call it:

1. takes the column count from the extent and selects the v3 or the v2/legacy column map (`panzerdb.cpp:1830-1838`);
2. **decodes only what is new**: if an earlier engine parked its decoded prefix for this file (the index handoff, §3.6), `index_rows_loaded` is where reading resumes, and only the rows appended since are read (`panzerdb.cpp:1845-1852`);
3. unpacks each row into a `Leaf`, resolving the path text from `/path_table` (v3) or from the row-aligned `/paths` block (v2/legacy);
4. **skips a data row whose `offset + count` runs past the materialised extent of its `data_raw_*` dataset** (`panzerdb.cpp:1961`-`1970` for v3, `:2143`-`2158` for v2/legacy). That is the reader half of the crash-safe commit ordering (§3.6): a row written before its payload turned the payload into a corrupt read, now it is just an invisible row;
5. rebuilds the derived tables the strategies rely on: `leaf_lookup` (path → rows), `parent_lookup`, `metadata_by_schema`, `cached_dynamic_aos_roots`, `max_time_at_dynamic_root`, and `time_lookup_unsafe_paths` (§4.2);
6. rebuilds `parent_path` **from the row data itself** — there is no `/parent_paths` dataset any more:
   * data leaf (kind 0/1): parent = full path minus its last segment;
   * AoS meta (kind 2/3): parent = full path minus *two* segments (the instance index and the AoS name).

### 2.5 The `Leaf` struct (what the read strategies consume)

```cpp
// panzerdb.h:227
struct Leaf {
    uint64_t shape[6];        // one time-slice shape (ndim==0 => scalar / meta-node)
    uint8_t  ndim = 0;        // how many shape entries are meaningful
    uint64_t time_index = 0;  // first time step of this row
    std::string_view path;          // zero-copy view into the engine's path storage
    std::string_view parent_path;   // rebuilt on read (see 2.4)
    uint64_t offset = 0, count = 0; // location in data_raw_*
    uint64_t flags = 0;             // (DataType<<4)|kind
    bool is_empty = false;
    uint64_t row_id = 0;            // position in /index; key of the span table (§5)
    ShapeView shape_span() const;   // borrowed view over shape[0..ndim-1]
};
```

`shape` is **inline storage, not a `std::vector`**: the cache holds 10⁵–10⁷ leaves and a vector meant one malloc per row. Iterate with `shape_span()` or `shape[0..ndim-1]`.

⚠️ `path`/`parent_path` are views into the engine's leaf cache. They are valid until the next `getLeaves()` **rebuild** — which bumps `leaf_cache_generation()` so consumers can detect it (§4.2). Never cache a `Leaf*` across an operation boundary without checking the generation.


---

## 3. The write path — `HDF5Writer_v2`

The writer is intentionally thin: it translates AL contexts into PanzerDB calls and keeps the two in sync.

### 3.1 Opening a write session — `setWriteStrategy`

```cpp
// hdf5_writer_v2.cpp:91 (condensed)
if (write_mode == GLOBAL_OP) {                     // whole-IDS put()
    panzer_db_ptr.reset();                         // drop the slice-session engine FIRST
    panzer_db_ptr = std::make_unique<PanzerDB>(loc_id, PanzerDB::OpenMode::WRITE, true, false);
    // (re)build the schema @key map from IDSDef.xml for this IDS only
    ... MetadataExtractor extractor($IMAS_PREFIX/include/IDSDef.xml);
        auto flat = extractor.extract_metadata(ctx->getDataobjectName());
        for (kv : flat) {                            // "path@attr" -> value
            metadata_by_path[schema_path][attr] = value;   // O(log) lookup at write time
        }
} else if (write_mode == SLICE_OP) {               // time-slice put()
    if (!panzer_db_ptr || !panzer_db_ptr->reanchor(loc_id))
        panzer_db_ptr = std::make_unique<PanzerDB>(loc_id, PanzerDB::OpenMode::APPEND, true, false);
}
```

Four points worth understanding:

1. **GLOBAL_OP truncates** (a `put()` rewrites the IDS); **SLICE_OP appends**, continuing the time counters where the previous session stopped (`PanzerDB::restoreTimeContext`).
2. **One APPEND engine per slice session, not per operation.** The AL closes the IDS group at the end of every `putSlice()` and hands a fresh one, but the engine's handles — and with them the decoded table, the time counters, and HDF5's **write chunk cache** for the tail of every table — survive that. Rebuilding per time step cost an init plus a close (`restoreTimeContext` walking the whole leaf table, the dirty tail chunks written back through `H5Dclose`), which grows with what has already been written: quadratic over a session. `reanchor` is what keeps the engine valid across the group hand-over; it is refused (and the engine rebuilt) when the table moved underneath it — see §3.6.
3. A **WRITE engine must be released before it is replaced**: it deletes `/index` and the raw datasets, and that must not happen while another engine still holds handles on them (hence the `reset()` at the top of the GLOBAL_OP branch).
4. The **metadata map is grouped by schema path** (`std::map<path, map<attr,value>>`). When the writer meets `electrons&0&density`, it strips the instance indices (`PanzerDB::stripIndices`), finds `electrons/density`, and replays its `@units`, `@description`, … as `writeMetadata("electrons&density@units", …)` — *before* the data itself. The old flat design scanned every XML entry for every written node (O(N·M)).

`read_homogeneous_time` (`hdf5_writer_v2.cpp:39`) reads `ids_properties&homogeneous_time` on every APPEND open. It used to construct a whole READ `PanzerDB` (full `/index` load) for one `int32`; it is now a direct `H5Dopen2`+`H5Dread`, with a one-level descent into the child group holding `/index` when the IDS group is not the PanzerDB root (mirrors `PanzerDB::init`).


### 3.2 Opening an AoS — `beginWriteArraystructAction`

```cpp
// hdf5_writer_v2.cpp:135 (excerpt)
if (*size == 0) return;                       // empty AoS: nothing opened, nothing to close
...
// hdf5_writer_v2.cpp:164 — "SYNCHRONIZATION: Necessary for nested AOS"
std::vector<std::string> aos_names; std::vector<int> indices;
if (ctx->getParent()) collectAosChain(ctx->getParent(), aos_names, indices);
if (!aos_names.empty()) panzer_db_ptr->synchronizeArrayStack(aos_names, indices);

const std::string aos_name = localAosName(ctx);                 // :171
if (ctx->getTimed() && !ctx->getTimebasePath().empty())
    panzer_db_ptr->beginArray(aos_name, ctx->getTimebasePath()); // dynamic AoS
else
    panzer_db_ptr->beginArray(aos_name, static_cast<size_t>(*size)); // static AoS
initialized_aos.insert(ctx);                                     // :181
```

Why the synchronisation: the AL can jump between AoS elements in any order (it iterates the tree itself), while PanzerDB assumes a `beginArray/endArray` discipline. `collectAosChain` (`aos_path_helpers.h`) walks the open `ArraystructContext` parents and rebuilds the exact state (names + current indices) PanzerDB must be in — so opening AoS *B* inside element 3 of AoS *A* names its paths with *A*'s current index. `localAosName` strips the parent's cumulative path and flattens `/`→`&`, so a level name like `constraints/x_point` stays **one** storage level.

### 3.3 Writing a node — `write_ND_Data`

`write_ND_Data` (`hdf5_writer_v2.cpp:195`) is the heart of the writer. Step by step:

1. **Re-sync the stack** (`collectAosChain(ctx, …)` + `synchronizeArrayStack`, `:219`/`:222`) whenever the context sits under an array-struct — this makes every write self-describing, whatever the AL called in between.
2. **Metadata branch**: a node name containing `@` (`is_metadata`, `:199`) is an IMAS metadata attribute → `panzer_db_ptr->writeMetadata(name, string)` (`:325`) and done. Otherwise the schema `@keys` replay of §3.1 runs first (`metadata_by_path.find(schema_path)`, `:257`).
3. **Resolve the time role of the written dimensions — once, for all types**:

```cpp
// hdf5_writer_v2.cpp:302
//  - inside a dynamic AoS: the time iteration is EXTERNAL (AL gives spatial dims only)
//  - explicit timebase outside a dynamic AoS: the LAST written dim is the time axis
//    ("bulk" mode: one call carries n_slices = shape.back() slices)
//  - no timebase: pure static tensor
const bool bulk_temporal = !timebasename_copy.empty() && !is_in_dynamic_aos;
```

This replaced three copy-pasted per-type blocks and a dead `size[1] > 20` string heuristic (see `to_improve.md` point 3).

4. **APPEND-mode guard** (`:316`): in APPEND mode a node with no timebase (neither own nor from an enclosing dynamic AoS) throws — a slice `put()` without time context cannot be placed on a time axis.
5. **One dispatch, three cases** (for `double_data`, `integer_data`, `complex_data`):

| Case | Condition | Engine call | Meaning |
|---|---|---|---|
| bulk temporal | `bulk_temporal` | `writeDataSlices(name, base_shape, data, n_slices, timebase)` | last dim was time → append `shape.back()` slices in one call |
| dynamic AoS | `is_in_dynamic_aos` | `writeDataSlices(name, shape, data, 1, eff_timebase)` | one spatial slice; the time step comes from the AoS iteration counter |
| static | otherwise | `writeData(name, shape, data, total)` | no time axis at all |

Strings (`:382`) take the same three-way branch: `dim==1` is a scalar string (buffer length in `size[0]`), `dim==2` is a `[count,max_len]` flat buffer converted to `char**` with RAII vectors (`string_pointers` / `string_data_copies`) — and note that with a timebase `dim==2` is **one** time slice of `size[0]` strings, the time axis coming from the timebase and not from the buffer. `dim > 2` throws.

### 3.4 Closing — `endAction`, `closePulse`, `deleteData`

```cpp
// hdf5_writer_v2.cpp:455 (logic)
ArraystructContext:  if (initialized_aos.count(ctx)) { panzer_db_ptr->endArray(); erase; }
OperationContext:    panzer_db_ptr->flush();                            // commit, always
                     if (getOpenMode() != APPEND) panzer_db_ptr->close();  // WRITE only
```

Only AoS contexts actually opened are closed (`initialized_aos` set) — empty AoS (`*size == 0`) never called `beginArray`, so they must not call `endArray`. The comment there still says *do not re-synchronise in `endAction`*: the stack is already correct from the preceding writes, and empty AoS indices simply have no index rows (their meta-node already carries the size).

The split between `flush()` and `close()` is the session rule of §3.1: **the commit is the operation boundary, the engine is not.** Closing the APPEND engine per `putSlice()` meant an `H5Dclose` over the whole table — the dirty tail chunk of `/index`, `/path_table` and every raw table written back out — and then a reopen. `closePulse` (`:490`) is where the slice session's engine is finally released, and `deleteData` (`:499`) drops it first because the tables its handles point at are about to be unlinked.

### 3.5 What PanzerDB does per write (engine recap)

For context when reading the code: `writeData*` never touch HDF5 directly. They append to RAM buffers (`data_buffer_f64/_i32/_c128/_str`, `index_buffer`, `path_table_buffer`, `list_spans_buffer`) and `append_index_row` (`panzerdb.cpp:3035`) records the row — 7 packed columns in v3 (§2.2). `flush()` (`:949`) extends each dataset with `H5Dset_extent` + one `H5Dwrite` per dataset; that is the only HDF5 write traffic.

Inside `writeDataSlicesImpl` (`:2637`) the time of each row is `base_time + i`, where

```cpp
// panzerdb.cpp:2678
base_time = std::max((uint64_t)dynamic_aos_current_iteration, signal_next_time);
// :2689  catch-up, only for a signal that already has data in the file:
if (hasFileData(full_path) && !dynamic_aos_timebase.empty()) {
    ... base_time = std::max(base_time, time_next - n_slices);   // :2694
}
```

so an APPEND session that resumes after a gap places its slices at the timebase position instead of leaving holes, while a **fresh** signal is not dragged to the timebase's current iteration (`hasFileData` is the fresh/continuing discriminator; the same pair of rules applies to strings at `:2826`/`:2851`). Strings also get **run grouping** (`:2872`-`:2931`): a consecutive run of non-empty scalar strings is merged into a single row (`string_run_open`), because one 512 B slot costs one index row otherwise; `STRING_CHUNKED` list scalars keep one row each, and spanned lists one row per slice.

Dynamic-AoS time counters live in `aos_time_counters`; `restoreTimeContext` (`:871`) rebuilds them — plus the `file_time_base` snapshot read through `hasFileData` — when an engine is built from scratch.

### 3.6 One engine per session: `reanchor`, the index handoff, and commit ordering

Three mechanisms keep a long `putSlice()` session from paying for what it has already written. All of them are guarded by identity checks, because each one reuses state across a boundary where the file may have moved.

**`PanzerDB::reanchor(hid_t)` (`panzerdb.cpp:1466`)** — the AL closes the IDS group at the end of every operation and hands a fresh one; the engine and its dataset handles outlive it. `reanchor` moves the engine to the new group when it is safe: only READ and APPEND engines (a WRITE engine still holds the tables it created), only engines *not* owning the file (`owned_file_id < 0`), same file name, and `/index` re-opened **through the caller's handle** — never through ours, because our handles belong to the file struct we opened them on and would answer with what we already believe rather than what is on disk. Then:

- the `/index` object must be the same (`H5Oget_token` comparison): a recreated dataset is a different table;
- for APPEND the gate is `rows_now == next_row_id`, **not** "the table did not grow". *We* grew it, by exactly the rows this engine assigned ids for and `flush()` wrote. Any other count means somebody else moved the table and this engine's decoded state is stale — rebuild;
- on success `file_time_base = aos_time_counters` (at an operation boundary this engine's counters *are* the "what the file held" snapshot that `hasFileData` needs), and the AoS stack (`array_stack`, `current_path`, `path_prefix`, `dynamic_level`) is cleared, so an AoS the AL abandoned mid-way cannot name the next operation's paths with the previous operation's prefix.

**The index handoff (`:1341`-`:1460`)** — decoding the tables dominates engine opening (171k rows on `core_profiles`: ~100 ms, a third of it inflating `/index` and the path column). Since a SLICE_OP session used to build one engine per time step, that cost was quadratic in the rows already written. So `close()` parks the decoded state (`handBackIndexHandoff`, `:1420`) in a **single `thread_local` slot** (`indexHandoff()`, `:1367` — per thread so two write sessions cannot park into each other's slot; one slot rather than a map so a session walking many IDS files replaces it instead of accumulating one decoded table per file), and the next engine on the same file takes it back (`adoptIndexHandoff`, `:1377`, called from `init()` at `:499`) and reads only the rows appended since. What travels: leaves, path blocks, lookups, metadata, dynamic-AoS roots, the whole-signal memo and the negative "no slice here" cache — and for READ only, the time counters (an APPEND engine must rebuild them from the rows in the file, so `file_time_base` never becomes "what this process wrote"). Before adopting, the parked state is re-validated against the file: same file name, same `/index` token, extent that did not shrink, and the same `(offset, count)` in its last row. `/index` is append-only by construction, so those checks are exactly what a rewrite fails. `adoptIndexHandoff()` returns whether the parked state covered every row the table has now, which is when `init()` may skip `restoreTimeContext()`.

**Commit ordering inside `flush()` (`:949`)** — payload tables first, then `/path_table` (`flushPathTable`, `:1090`) and `/list_spans` (`:1095`), and `/index` **last**, followed by `H5Fflush(file_id, H5F_SCOPE_GLOBAL)` (`:1138`) as the cross-process SWMR barrier. A reader may only ever see a row whose payload is already on disk; if the process dies in the middle, the worst case is payload bytes no row points at (harmless, reclaimed by the next rewrite) instead of a row pointing at bytes that were never written. As a second line of defence, `getLeaves` drops rows whose `(offset, count)` runs past the dataset extent (`:1961`-`1970` for v3, `:2143`-`2158` for v2/legacy) and marks those paths as unsafe for time lookups. `test_engine_commit_order_guardrail` keeps the ordering from being re-shuffled.

**Handle ownership.** `close()` (`:1315`) closes the group or file it anchored on, and then `owned_file_id` — the file this engine opened in its own constructor. That is not cosmetic: a file left registered is cleaned up by HDF5 at library termination, which crashes doing so (`H5_term_library` → `H5F_term_package` → `H5I_clear_type` → `H5VL_file_close` → `H5I_find_id`). When a rewrite is about to unlink tables, `HDF5Backend::dropIDSStorageForRewrite` calls `hdf5Reader->release_session()` and both `close_datasets()` first (`hdf5_backend.cpp:289`-`302`).

**Filter order.** When a dataset gets filters, they are pushed **shuffle before deflate** (`:315`-`316`, and again in `configureChunking` at `:713`-`715`). The reversed order was the whole cause of the v2-vs-v1 size gap: deflate before shuffle re-orders the compressed stream and destroys the grouping shuffle was there to create (≈19-20 % on the measured fixtures). `test_engine_filter_pipeline_order` asserts the order.

---

## 4. The read path — `HDF5Reader_v2` + strategies

### 4.1 One engine, one index, per read session

The v1 code created one `PanzerDB` *and* one path index per range mode — up to 3× the `/index` load and cache memory per IDS read. v2 shares them, and now keeps them across the whole session:

```cpp
// hdf5_reader_v2.cpp:86 (select_strategy, condensed)
if (session_db && session_db->reanchor(gid)) {
    session_gid = gid;                       // same engine, decoded index, handles, chunk caches
} else {
    session_gid = gid;
    session_db.reset(); session_index.reset();
    global_strategy.reset(); slice_strategy.reset(); timerange_strategy.reset();
    session_db    = std::make_shared<PanzerDB>(gid, PanzerDB::OpenMode::READ);
    session_index = std::make_shared<ReadIndex>(session_db);   // /index parsed ONCE per session
}
// then, lazily per range mode:
GLOBAL_OP    -> global_strategy    = GlobalReadStrategy(session_db, session_index)
SLICE_OP     -> slice_strategy     = SliceReadStrategy(session_db, session_index)
TIMERANGE_OP -> timerange_strategy = TimeRangeReadStrategy(session_db, session_index)
read_strategy = <the one matching ctx->getRangemode()>;
```

`prepare_strategy(ctx)` (`:123`) is called at every entry point (`read_ND_Data` `:148`, `beginReadArraystructAction`, `endAction`): it maps the AL context to its IDS group id, then calls `select_strategy`. Two rules to keep in mind:

* **`open_IDS_group` (`:53`) deliberately does *not* drop the session.** The AL closes the IDS group at the end of every operation and hands a fresh one for the next; throwing the engine away there meant rebuilding the decoded `/index`, the memoised signals and the dataset handles — twice per `getSlice()`. `select_strategy` moves the engine onto the new group id instead and rebuilds only when that move is refused (see §3.6; for READ the gate is `rows_now == index_rows_loaded`, which is also how a SWMR refresh is detected).
* **`release_session()` (`:38`) is the explicit teardown.** The strategies each hold a `shared_ptr` to the engine, so it drops them *first*, then `session_index`/`session_db`. It is called by `HDF5Backend::dropIDSStorageForRewrite` (`hdf5_backend.cpp:308`) before the tables are unlinked, because PanzerDB's destructor closes dataset handles on objects that are about to disappear. Strategies' `endAction(OperationContext)` still do not close the engine — it is shared, and dies with the session.

### 4.2 `ReadIndex` — the shared path index (and its lifetime contract)

`ReadIndex` (`iread_strategy.h:53`) builds, from `PanzerDB::getLeaves()`, two hash maps — and keeps the engine alive via `shared_ptr` because its keys are `string_view`s into the engine:

* `path_cache`: full instance path → all leaves at that path (one per time slice for per-slice storage);
* `name_index`: last path segment (dataset name, which never contains `/` after `&`-flattening) → same candidates. This makes the old O(N) "scan every leaf whose path ends with the name" fallback an O(1) bucket lookup with **identical** semantics.

Every bucket is **time-ordered** at build time (`build_indexes`, `:94`: `std::stable_sort` by `time_index`), and `equal_time()` (`:171`) binary-searches it:

```cpp
// iread_strategy.h:171
static TimeRange equal_time(const std::vector<const Leaf*>& bucket, uint64_t time_index) {
    const auto lo = std::lower_bound(...); const auto hi = std::upper_bound(...);
    return {lo, hi};
}
```

This is what turned a full GLOBAL `get()` over 8,150-slice signals from O(slices²) to O(slices log slices) (−70 %, §7). `find_path` (`:159`) and `find_name` (`:182`) are the only accessors.

**Lifetime contract (SWMR-ready):** `PanzerDB::leaf_cache_generation()` is bumped exactly where the old leaf table dies (the `clear()` at the top of a `getLeaves()` rebuild). `ReadIndex` stamps the generation at build; `is_current()` (`:137`) compares it, and `ensure_current()` (`:150`) rebuilds only when it moved. `refresh_index_if_needed()` (`:258`) is called **at operation entry only** (never mid-operation, so `Leaf*` gathered during one read stay valid):

```cpp
// iread_strategy.h:258
void refresh_index_if_needed() {
    if (!read_index_ptr || !read_index_ptr->ensure_current()) return;
    time_values_cache.clear();                  // derived caches go too
    homogeneous_time_cache = HOMOG_TIME_UNRESOLVED;
}
```

A `flush()` on the engine does *not* move the generation (leaves stay usable across a commit); only a genuine rebuild does. `PanzerDB` keeps caches of its own beside these maps — the path→rows lookup `leaf_lookup`, the whole-signal memo `whole_signals`, the negative "no slice here" cache (`panzerdb.h:434`/`:449`) — and they follow the same rule (`gen == leaf_cache_gen && rows == index_rows_loaded`). All of them travel with the index handoff of §3.6, which is how a 200-slice `getSlice()` session stops decoding the same 171k rows 200 times.

### 4.3 Shared helpers in `IReadStrategy`

All three strategies inherit these; understanding them explains ~80 % of the read code.

**Path reconstruction.** `collectAosChain` (shared header) yields the open AoS `(names, indices)` chain; `getPath(ctx, include_self_index, override_timed_index)` and `buildFullPath(ctx, dataset)` turn it into `A/0/B/1/dataset`. `sanitize_path(ctx, path)` normalises an AL path (possibly partial, possibly `/`-prefixed) against the open context chain via `sanitizeAosPath` (`aos_path_helpers.h:75`). The match of a context path is **anchored at the head** of the string — AL paths are cumulative from the IDS root, so a repeated segment name higher in the hierarchy can no longer steal the match (`to_improve.md` point 5). Results are cached per `(ctx, path)`.

**Leaf lookup.** `find_leaf_for_context` (`iread_strategy.h:487`) builds the *strict* instance path (`AOS/i/.../name`) and:
1. looks it up in `path_cache` (O(1)); when the node is dynamic, narrows to the requested `time_index` with the binary-search `equal_time`;
2. falls back to `name_index` with the same time filter and a context-prefix constraint — the exact old full-scan candidate set, in O(1).

The fallback has a guard that is easy to break and expensive when broken (`:560`): when the context has **no** prefix (a top-level query), any leaf living under an AoS is rejected. Without it, a top-level `n_e` read matched `channel(0).n_e` — the first leaf sharing the last path segment — and the AL then wrote that member back as a root-level node on the round-trip: the 2× file-size inflation bug.

**The engine's own time narrowing.** The strategies do not scan leaf lists per time step either: `PanzerDB::rowsForTime(path, time_index)` (`panzerdb.h:488`, `panzerdb.cpp:3321`) binary-searches the row list of a path — rows of one signal have increasing, non-overlapping time ranges — and returns a borrowed `RowSlice`. Paths whose rows do not satisfy that order (meta rows among data rows, out-of-order appends) are recorded in `time_lookup_unsafe_paths` while `leaf_lookup` is built and fall back to the full sweep, which keeps the narrowing sound rather than merely fast.

**The global read.** `read_dataset_globally` (`iread_strategy.h:921`) is what GLOBAL_OP (and the static shortcut of the other two) actually calls. Its search ladder:

```
specific path (AOS/i/.../name)                  -- exact instance
   └─ miss → generic path (AOS/.../name, dynamic levels keep no index)
                └─ miss → name_index fallback (read from a parent, e.g. "profiles_1d/signal")
```

…collects all matching leaves (time-ordered), then:

* **scalar-overwrite fix**: several static scalar rows at one path (successive overwrites) ⇒ keep the **last** one only.
* **strings**: compute the union slot range `[min(offset), max(offset+count))` of the leaves and fetch it in **one** `readStringBulk` H5Dread (`:1160`); scatter locally, joining `STRING_CHUNKED` scalars and spanned list elements (§5). The old code did one tiny H5Dread per slice — O(N) round-trips, the cause of "reading never returns" on 10⁵-slice files.
* **numerics**: one `readLeavesUnion` (hyperslab union, `H5S_SELECT_OR`, fast path on monotonic offsets). The buffer is RAII-held and only released to the caller on success (`:1272`-`:1285`) — a throw inside the union read used to leak it.
* **output shape rules** (the AL contract): a rank-0 signal with >1 stored element returns 1-D `[n_values]`; a rank-0 signal with exactly 1 element returns 1-D `[1]` **only if** it reads as time-dependent (`shouldPromoteTimeScalarOnSlice`) or the AL dictionary expects 1-D (`expected_dim == 1`, captured at `:938` **before** any branch overwrites `*dim`); a rank-k signal with multiple slices gets an appended time dimension `[shape..., n_slices]`, single slice ⇒ rank unchanged.

**Time values.** `getTimeValues(ctx, homogeneous_time, timebasename)` (`:325`) resolves the time base of a context:

* `homogeneous_time==1` → the global `time` signal, read once per session through `PanzerDB::getWholeDynamicSignal` (`panzerdb.cpp:4383`): one `readLeavesUnion`, then **memoised** in `whole_signals`. Without the memo, a time base stored one row per step touched every chunk of the raw dataset once per slice read — a session reading N slices paid the full pass N times.
* `homogeneous_time==0`, under a timed AoS: try `<timed-AoS-path>/time` in `path_cache`; else generate `<timed-AoS>/i/time` for each element index from `getDynamicAOSSize`; else linear-scan fallback. Cached per `(timed_aos_path, timebase_basename)`.
* `homogeneous_time==0`, **element-local time base** (a dynamic signal inside a *static* AoS, so no timed ancestor): the base is read relative to the current element — `buildFullPath(ctx, timebase)` — falling back to the root timebase when it is empty. The cache key is `"ELEMENT:" + that path` (`:375`), so each AoS element gets its own base and repeated reads of one element stay cheap.
* `getHomogeneousTime()` (`:307`) reads `ids_properties&homogeneous_time` once per session and caches it (`HOMOG_TIME_UNRESOLVED = -2` sentinel; real values −1/0/1).

**Gaps.** `PanzerDB::nearestAvailableSliceIndex` (`panzerdb.cpp:3344`) resolves an index to the slice actually stored, and records in `no_slice_at_any_index` the paths found to hold **no** slice for a time base of a given length (stale-check + lookup `:3382`-`:3390`, recording `:3445`). An optional signal this IDS does not carry otherwise cost a full scan of the time base per absent signal per slice read; the entry stores the length that was covered, so a later read whose time base reaches further re-probes instead of inheriting the `-1`. The same helper resolves the path variants first (exact path, generic path with the dynamic index dropped, index substituted), which is why the answer can be given from `leaf_lookup` without touching the payload.

**Node-role rules (replace the heuristics).** `collectTimebaseSchemaPaths` (`:760`) + `isTimebaseDataset` (`:800`) decide "is this dataset *the* timebase?" by matching the AL-supplied `timebasename` and the nearest timed AoS's `getTimebasePath()` (schema-normalised via `stripIndices`), with an exact-final-segment-`time` legacy fallback. The old `ends_with(name, "time")` swallowed `time_step`, `lifetime`, … `shouldPromoteTimeScalarOnSlice` (`:825`) encodes the scalar-promotion AL convention (root/static-AoS time scalars get a size-1 slice dim; dynamic-AoS scalars stay scalars).

### 4.4 `GlobalReadStrategy` — GLOBAL_OP

The whole node, all stored times:

* `beginReadArraystructAction` (`global_read_strategy.cpp:26`): `getAOSShape(path)` fills the AoS size (for dynamic AoS: number of stored slices).
* `read_ND_Data` (`:50`): `return read_dataset_globally(...)` — that's all the file does.

### 4.5 `SliceReadStrategy` — SLICE_OP

One time slice at `ctx->getTime()`:

```cpp
// slice_read_strategy.cpp:72 (condensed)
refresh_index_if_needed();
if (timebasename.empty() && !isTimedContext(ctx))
    return read_dataset_globally(...);            // static node: no time involved

time_basis = getTimeValues(ctx, homogeneous_time, timebasename);
res = panzer_db_ptr->readInterpolatedData(full_path, time, time_basis, interp, ...);
if (shouldPromoteTimeScalarOnSlice(ctx, timebasename, *datatype)) { size[*dim] = 1; (*dim)++; }
```

`beginReadArraystructAction` resolves `slice_idx = getTimeIndex(timebase_path, time, interp)` and reports the AoS size with that dynamic index substituted (`getPath(ctx, false, slice_idx)`) — a timed AoS reports size 1 (the one selected slice). `readInterpolatedData` internally handles **gaps**: it resolves the requested time to the nearest available slice (`nearestAvailableSliceIndex`, closest/previous/linear modes) instead of assuming every time index exists. Note that every read of a node also pulls its metadata through `readMetadata(full_path)`, which marks the path processed so the `@keys` of that node are not re-read later in the operation.

### 4.6 `TimeRangeReadStrategy` — TIMERANGE_OP

The range `[tmin, tmax]` (optionally resampled onto `dtime`). The behaviour that matters (from the gap fix and the element-local time bases):

* The timebase dataset itself is special-cased (`timerange_read_strategy.cpp:92`-`:100`): `is_homogeneous_time_basis_dataset` (homogeneous and named `time`) versus `is_time_dataset && !is_homogeneous…`, both resolved through `isTimebaseDataset` (`:94`). With resampling, the *time* signal is produced by `resample_timebasis`, never by slicing data.
* The node's time base is re-resolved per read when it is **element-local** (`per_element_timebase`, `:181`-`:184`: `homogeneous_time == 0` with a timebase but no timed context) or when the session-level vector came back empty.
* An element-local time base is stored as a single array, so the signal path below would re-emit that whole array once per requested time — for a time-node read it is **cut to the window** instead (`:212`-`:224`), which is what the AL expects.
* For a **standalone dynamic signal** (no dynamic-AoS context, numeric types), the strategy reads **one resolved slice per requested time** (`readInterpolatedData` in a loop, `:232`-`:315`). Reading one concatenated global buffer and slicing it by element offset was wrong when slices were missing: a gap shifted later values and allowed reads past the buffer end. The first requested time is probed once for shape/dims and its payload is **kept** as the first slot of the output rather than re-read (`:262`-`:293`).
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

`data_raw_str` is **fixed-width**: 512 B slots (`STRING_MAX_LEN=512`, `panzerdb.cpp:60`; the type size is set at `:350`). This keeps the file SWMR-safe (no variable-length type), but IMAS allows arbitrary-length strings, so two mechanisms cover what does not fit:

* **Scalar longer than one slot** → one row, `count = k` slots, `DataType::STRING_CHUNKED` (`panzerdb.h:691`); the reader concatenates the slots (see `read_dataset_globally`'s string scatter and `readMetadata`'s join). A value longer than `STRING_MAX_LEN-1` is rejected at write time (`panzerdb.cpp:59`).
* **List element longer than one slot** → the list leaf stays one row, and each element's placement is recorded in the **`/list_spans` dataset**: 4 u64 per entry `(row_id, element_idx, slot_offset, slot_span)` (`SpanEntry`, `panzerdb.h:261`). Read-side access:

```cpp
// panzerdb.h:1173 / :1184  (isListSpanned / getListSpans — both lazy-load the table once)
if (panzer_db_ptr->isListSpanned(leaf->row_id)) {
    for (const auto& se : panzer_db_ptr->getListSpans(leaf->row_id))
        for (uint64_t k = 0; k < se.slot_span; ++k)
            joined += slots[base + se.slot_offset + k];   // element = concat of its slots
}
```

**The span table is created on demand, not at file creation** (`ensureListSpansDataset`, `panzerdb.cpp:1149`): a file with no long strings has no such dataset at all (compact lists: one slot per element, `slot_span == 1` implied), and an APPEND into a file written before the table existed still finds somewhere to put new spans. It is chunked (4096 u64) because an unlimited extent cannot be contiguous. On the WRITE side it is in the "delete what you are about to recreate" list (`panzerdb.cpp:291`) for a reason worth remembering: a WRITE engine handed a `loc_id` does not truncate the file, so a name left out of that list keeps its rows from the file's previous life — and since row ids restart at 0, stale spans would describe the *new* lists' element boundaries. That is silent corruption, which is also why `flushListSpans` (`:1180`) treats a missing table as an error rather than dropping the buffer. Reads load the whole table once and bucket it by `row_id` (`loadListSpans`, `:1287`; cache `cached_list_spans`, `panzerdb.h:298`).

How many rows a string signal costs (`writeDataSlicesImpl` string branch, `:2872`-`:2931`): compact scalar strings merge a **run** of consecutive non-empty slices into one row (`string_run_open`, one row per run — the run's slots are contiguous, so a per-slice row would be 1 index row per 512 B); `STRING_CHUNKED` scalars keep one row each; spanned lists one row per slice.

The scalar-vs-list contract is shape-driven and centralised: `PanzerDB::isStringScalarLeaf` (`panzerdb.h:711`) — empty shape ⇒ scalar string, rank-1 shape ⇒ list, whatever the element count.

---

## 6. Metadata (`@key` attributes)

Metadata entries are ordinary static string leaves whose path carries an `@` suffix: `flux_loop&field@units = "T"`.

* **Write**: two sources, both funnelling into `PanzerDB::writeMetadata` (deduplicated per session via `written_metadata_schema_paths`, `panzerdb.h:579`):
  1. explicit AL writes (node name contains `@`, §3.3);
  2. **schema replay** from `IDSDef.xml` (`MetadataExtractor`, §3.1), so a v2 file carries the dictionary's `units`, `description`, … for free.
* **Read**: `PanzerDB::readMetadata(instance_path)` (`panzerdb.cpp:4495`) strips indices to the schema path and resolves that schema's `@key` leaves through `metadata_by_schema` (`panzerdb.h:507`), a hash map built once inside `getLeaves()` (`:2017` / `:2202`) — it used to be an O(N) prefix scan *per read call*. Strategies call it on every node read (harmless and cheap now); it also marks the schema processed, which is what stops the same `@units` from being re-read inside one operation.
* Values are stored in the map as **indices into `cached_leaves`**, not `Leaf*` — the map is built while `cached_leaves` still grows, and stored pointers would dangle on reallocation (this exact bug caused a SEGFAULT in `test_direct_api_validation` during development; the index-based design is the guard). Because the map is keyed by `string_view`s into the leaf storage, it travels with the index handoff (`:1394`/`:1438`) instead of being rebuilt.

---

## 7. Performance notes (measured)

**Session lifetime, not call count, is where the cost was.** Both paths used to rebuild their engine per AL operation.

| change | measured effect |
|---|---|
| read session keeps its engine (`reanchor`, §4.1) | `getSlice` at 1000 slices × 3000 points: **22 166 → 2 517 µs/call** (×8,8), and **764 652 → 2 517 µs (×304)** counted from the pre-memo baseline — at 1000 slices the v2 read path now sits *below* the v1 one (2 517 vs 4 382 µs). Note the engine rebuild itself was only ~2,45 ms of that 22 ms call: what the session actually keeps is the **chunk cache**, which lives on the dataset handle, so a fresh engine re-decompressed on every call what the previous call had already read |
| write session keeps its APPEND engine (§3.1, §3.6) | `put_slice` 1000×1000: **6165 → 1943 µs/call** (from 1,7× slower than the v1 backend to 1,8× faster). A rebuilt engine paid ~1,5 ms of init — `restoreTimeContext` walking ~6 000 leaves *per engine*, i.e. quadratic over the session — plus ~1,17 ms of `close()`, almost all inside `H5Dclose` writing the dirty tail chunks back |
| index handoff (`IndexHandoff`, §3.6) | removes the ~100 ms decode of 171k rows that each rebuilt engine used to pay; a refresh after a foreign append costs **4,53 ms** for 500 new rows on top of 171k leaves, against ~90 ms for a full re-decode — what is left in those 4,53 ms is `rebuildDynamicRootTimeIndex()`, still a full pass over the leaves (§10) |
| whole-signal memo `whole_signals` | a time base stored one row per step was assembled with **one `readTensor` per leaf** (1000 single-element reads per call, each hitting its own 512 Ko chunk); now one union read per session — on its own **764 652 → 57 714 µs/call** at 1000×3000 |
| negative cache `no_slice_at_any_index` | removes one full time-base scan per absent optional signal per slice read: **230 472 → 242** availability probes per call at 1000 slices |
| `rowsForTime` binary search | a single `get()` was O(n_slices) per node; a full time-range `get()` O(n_slices²) |

**Index format (v3, §2.2)** — bench 300 signals × 1000 steps (301 001 rows, 302 distinct paths): raw index I/O **96 → 56 B/row (−36 %)**, and the per-row 256 B `/paths` re-read is gone (`/path_table` = 10,9 Ko for the whole file vs 663 Ko of `/paths`). Decoded leaf cache: the two 256 B/row blocks (paths + rebuilt parents) collapse into the path table — **~87 Mo → ~77 Ko** on `core_profiles_3.h5` — and one malloc per row disappears. String signals cost one index row per written **run** instead of per step (§5).

**File size.**
* Filter order (shuffle before deflate, §3.6): **≈ −19 %** on the measured fixtures — by far the biggest size lever found; chunk size was worth <0,5 %, and level 1→6 ≈ −1,5 % more.
* Chunk profile is chosen in `init()`: **`time_series` for WRITE/APPEND** (the one that lands on disk), `interactive` for READ (no compression). Overridable before creation with `setChunkingHint()` / `setCompressionLevel()` / `setDefaultChunkingConfig()`. Small files pay ≥1 chunk per touched dataset (`magnetics_FAT.h5`: 170 Ko logical → 4 138 Ko on disk); big files are dominated by compression.
* `H5Pset_alignment(4096, 4 MB)` on the by-name constructor inflates the **address space**: 105,1 Mo apparent for 14,5 Mo actually allocated on this Lustre. Measure three numbers before concluding anything: Σ`H5Dget_storage_size`, `stat -c%s`, `du -B1`.

**Commit cost.** Of a ~8-11 ms write step, `H5Fflush(GLOBAL)` is **99 %** of it — user CPU recompressing the dirty tail chunk of every touched table, not I/O (200 commits: 7,4 ms of `pwrite64` against 1663,8 ms of flush). Caching extents to skip `H5Dget_space` was measured at ~20 µs and abandoned (`to_improve.md` §23: it also risks truncation, since `H5Dset_extent` with a smaller extent truncates). The real levers are chunk size and compression level, all frozen at creation: 2000 commits × 20 signals went **8,65 → 0,62 ms/commit** with index chunk 512 / f64 chunk 8192, at 15,4 Mo allocated either way.

**Read path history** (study on `core_profiles_3.h5`, 171k rows, 28 signals × 8150 per-slice leaves, GLOBAL `get()`): time-ordered buckets + binary-search time filter **3707 → 1105 ms (−70 %)**; `readLeavesUnion` for `getWholeDynamicSignal` (O(#chunks) tiny reads → 1 union read); one shared engine/index per session (was 3); `metadata_by_schema` (O(N) scan per read → 1 hash lookup); `readStringBulk` union range (per-leaf string reads → 1 H5Dread). Chunk-cache *tuning* (size/nbytes/w0) measured **no effect** for that pattern — each chunk is touched once.

**Measuring anything here:** set `IMAS_AL_DISABLE_VALIDATE=1` (the AL validation pass otherwise dominates), and treat `bench_read` numbers with care — it passes an empty `timebasename` for AoS levels it does not mark dynamic, so its TIMERANGE/GLOBAL figures can describe a path the AL would never take (§11).

---

## 8. Errors & memory ownership

* **Ownership across the AL boundary**: every `read_ND_Data`-style output buffer is `malloc`-style and caller-freed. Internally, allocation is RAII (`std::unique_ptr<T[]>` / `make_unique`): the buffer is `release()`d to the caller **only on success**; any throw (including inside `readLeavesUnion`/`readTensor`) frees it. The remaining manual `malloc/free` juggling lives in `TimeRangeReadStrategy` around `PanzerDB::readInterpolatedData` (its C-style API still returns raw pointers — `:217`-`:307`) — known open point (`to_improve.md` §7).
* **A failed HDF5 read raises, it does not return a buffer.** `readTensor` (`panzerdb.cpp:2268+`) throws `ALBackendException` on `H5Dget_space`, `H5Sselect_hyperslab` or `H5Dread` failure. Before this, a failing read left the caller's uninitialised allocation being returned as if it were data (REVIEW_PERF_hdf5_v2 item E1).
* **The dtype rule is width, not type.** `readTensor` chooses `mem_type` from `leaf.flags >> 4` — never from its template argument `T` — so a *narrower* `T` would be overrun; the guard is `sizeof(T) < stored_size` → throw (`:2300`). The opposite direction is deliberate and load-bearing: `readInterpolatedData` routes `alconst::integer_data` through `readDataByIndex(double**)`, which over-allocates and whose consumer reads the buffer back as int32. Do **not** add an exact "T must equal the leaf dtype" assert — it breaks every integer slice and time-range read. `test_read_tensor_contract` pins both directions.
* **Exceptions**: backend-level problems throw `ALBackendException(..., LOG)`; engine errors (`std::exception`) are caught in `readLeafData` and rewrapped as `ALBackendException("PanzerDB read error: ...")` (`iread_strategy.h:902`). A read that finds nothing returns `0`/empty, it does **not** throw.
* **Empty data**: `total_elements == 0` ⇒ `*data = nullptr`, `*dim = 0`, return success. Empty leaves (kind 1) exist in the index but carry no payload.

---

## 9. Inspection tools (`src/hdf5/tools/`)

| Tool | What it shows |
|---|---|
| `imas_h5ls <file> [-r] [--show-aos] [--show-metadata]` | the **logical** IMAS tree reconstructed from `/index` (no `.h5` suffix) |
| `imas_h5dump <file> [path1] [path2] …` | values + metadata; paths accept the slice syntax `profiles_1d[0:5]/grid/rho` |
| `panzer_index.py <f.h5> [--limit N --match R --kinds data,empty,aos-static,aos-dyn --raw --no-parent]` | the **physical** index table, decoded. Reads the `index_version` attribute and mirrors `PanzerDB::getLeaves()`: v3 (7 packed columns + `/path_table`) is expanded into the same 12-column view the code uses; v2 (12) and legacy (14) are read directly. Header line names the layout and the number of interned paths (`7 colonnes v3 packées + /path_table interné (79109 chemins uniques)`), and the extents list carries `path_table=…` when present. Plain Python + h5py, no build step |
| `imas_publish_view <f.h5> [--root N] [--expand] [--no-aos-groups] [--max-mappings N] [--stats]` | builds an idempotent **VDS** view under `imas_view/` so `h5ls`/`h5dump`/`h5py` can browse the file with zero IMAS dependency. Purely additive; never touches `index`/`path_table`/`data_raw_*`. Every publish opens the file RDWR and deletes+recreates `<root>`, so the grouped and the `--no-aos-groups` layouts never coexist |
| `imas_publish_view <f.h5> --list [--root N]` | **read-only, publishes nothing** (`H5F_ACC_RDONLY`): prints the view that is already in the file, one annotated line per object (dims, `@dims_label`, `@ragged`, `@aos_size`, `holes=…`). Exits 2 when the file has no view |

**Collapse mode (default) — one dataset per *logical* node.** Every Array-of-Structures level and every time step becomes a **leading dimension** instead of a separate HDF5 object, so `h5ls -r` lists the IMAS tree rather than one path per instance (verbatim output):

```
/imas_view/time_slice/boundary&psi_norm                                 Dataset {400}
/imas_view/time_slice/constraints&flux_loop/measured                    Dataset {400, 17}
/imas_view/time_slice/boundary&outline&r                                Dataset {400, 587}
/imas_view/grids_ggd/grid/space/objects_per_dimension/object/nodes      Dataset {1, 1, 1, 3, 15402, 3}
```

The same objects through the tool's own read-only listing (`--list`, which is where the annotations are visible — see below):

```
/imas_view/time_slice                                     Group @aos_size=400 @aos_dynamic
/imas_view/time_slice/boundary&psi_norm                   Dataset {400} @dims_label=time_slice
/imas_view/time_slice/boundary&outline&r                  Dataset {400, | 587} @dims_label=time_slice|boundary&outline&r @ragged holes=77123
/imas_view/grids_ggd/grid/space/objects_per_dimension/object/nodes
                                                          Dataset {1, 1, 1, 3, 15402, | 3} @dims_label=grids_ggd|grid|…|object|nodes holes=92412
```

The `|` in a shape sits at `@aos_ndim`: everything left of it is an AoS/time level, everything right is the leaf's own slice shape — which is what makes `nodes` readable as "3 levels of size 1, one of size 3, 15 402 AoS instances, then 2".

dims = `[n_time?, n_aos_0, …, n_aos_k-1, *slice_shape]`; AoS levels outermost→innermost; a leading time dim appears when one logical path is stored as several index rows (per-slice rows *or* one bulk row with `count == n_slices`). **Every AoS level keeps a dimension, size 1 included** — `h5ls` prints no attribute, so the shape is what makes a size-1 level visible. A level's length is its **declared** `@aos_size`, not the largest instance index that happens to carry data (`buildAggregates` pads from `aosSizes`), so two nodes under the same AoS print the same length for that level and unwritten instances are holes. Unmapped cells keep the IMAS invalid value as HDF5 fill: `-9E+40` for reals (both components of a complex), `-999999` for integers, empty string for text. Attributes: `@datatype`, `@n_elements`, `@schema`, `@aos_ndim` (how many leading dims are AoS/time), `@dims_label` (one name per dim), `@n_mappings`, plus `@time_dim` / `@ragged` / `@mapping_cap` when relevant; AoS groups carry `@aos_size` and `@aos_dynamic=1`. **Object names never contain `[`** — the HDF5 CLI parses brackets as a selection, which makes such an object unreachable by `h5dump -d` (the v1 limitation); `&` is harmless.

`--no-aos-groups` publishes no AoS group at all: every dataset sits flat under the view root with its path joined by `:` (`imas_view/grids_ggd:grid:space:objects_per_dimension:object:geometry`), so a recursive `h5ls` lists datasets only and each AoS size still shows in the shape. `:` is not an HDF5 separator and never appears in an IMAS name, so `h5dump -d` and h5py still reach the object; `@aos_size`/`@aos_dynamic` are then absent (0 groups).

**How to see the annotations** — `h5ls -r` prints **no attribute at all** (verified on HDF5 1.14.4; it is not a version quirk), so the annotations above are invisible in a plain recursive listing. The one-liner is the tool's own read-only listing:

```bash
imas_publish_view f.h5 --list      # publishes nothing: reads the view back, one annotated line per object
```

```
/imas_view/code&output_flag                                            Dataset {400} @dims_label=time
/imas_view/time_slice                                                  Group @aos_size=400 @aos_dynamic
/imas_view/time_slice/boundary&outline&r                               Dataset {400, | 587} @dims_label=time_slice|boundary&outline&r @ragged holes=77123
/imas_view/grids_ggd/grid/space/objects_per_dimension/object/nodes     Dataset {1, 1, 1, 3, 15402, | 3} @dims_label=grids_ggd|grid|space|objects_per_dimension|object|nodes holes=92412
```

`holes=N` is derived at print time (`product(dims) − @n_elements`), so it needs no extra attribute and appears only when non-zero; it counts cells left to the fill value by ragged slices, sparse AoS instances, or the declared-size padding. The file is opened `H5F_ACC_RDONLY` — `--list` cannot change it. On `equilibrium_1.h5` it prints 171 lines (154 datasets + 17 AoS groups). When the file holds no view, `--list` prints a hint and exits 2.

For the raw attribute values (including `@n_mappings`, `@datatype`, `@n_elements`, which `--list` keeps out of the line), use `h5ls -v -r` for *names* (6 lines per object) or `h5dump` for *values* — `-d` selects a dataset, `-g` selects a group (`-d` on a group errors with *"is a hard link"*). Attributes are stored as `H5T_IEEE_F64LE` scalars or `H5T_C_S1` strings:

```bash
h5ls   -v -r f.h5                                   # attribute names (6 lines per object)
h5dump -H -A -d "imas_view/time_slice/boundary&outline&r" f.h5   # dataset: names + values
h5dump -H -A -g "imas_view/time_slice" f.h5                       # group: @aos_size, @aos_dynamic
```

Note `-A` (all attributes), not `-a NAME` (selects one attribute by name).

`--expand` restores the historical view (one VDS per index row). It is kept for compatibility only: rows sharing one exact path collide, so on `equilibrium_1.h5` it publishes **74 701** datasets and **skips 41 097 rows**, and the file grows from 162,206,730 B to 244,684,074 B. Collapse mode publishes the same file as 154 datasets with 0 errors. `--max-mappings N` is a safety valve (a dataset exceeding it keeps its first mappings and is flagged `@mapping_cap=1`); the worst dataset in the reference files needs 400 mappings, so the default (unlimited) is safe.

**Measured on the reference files, 2026-10-21 (collapse mode, `--stats`)** — all with 0 errors, 0 rank overflows, 0 mapping-cap fallbacks:

| file | datasets | AoS groups | mappings | max in one dataset | cells left as fill | size delta | wall |
|---|---|---|---|---|---|---|---|
| `equilibrium_1.h5` | 154 | 17 | 30 585 | 400 | 369 340 | +50 408 B | 2,8 s |
| `core_profiles_3.h5` | 28 | 7 | 28 | 1 | 0 | +584 B | 13,2 s |
| `reflectometer_profile_v2.h5` | 33 | 3 | 33 | 1 | 0 | +45 056 B | 0,04 s |
| `magnetics_THIN.h5` | 273 | 19 | 321 | 2 | 0 | +1 328 B | 0,05 s |

The view is metadata only (virtual mappings, no payload), so it costs tens of kilobytes, never data. What changes is what a plain `h5ls -r` shows: on `reflectometer_profile_v2.h5` it goes from 9 lines (the root, the IDS group and the six PanzerDB tables) to 46 lines naming the IMAS nodes. `h5ls -r` of `equilibrium_1.h5` prints 181 lines. VDS caveat: the source filename is recorded as a string and resolved at **read** time, so renaming or moving a published file leaves a view that lists fine but silently reads back as the fill value (exit 0, no HDF5 error) — re-publish after any rename.

---

## 10. Known limitations / open points

Tracked in `to_improve.md`; the ones still open as of this writing:

* **#3 (partial)**: node *roles* (timebase, shape role) are still inferred from the AL context + shape; a fully file-level persistence (`@timebasepath` attribute or an index role bit) would make role resolution independent of the calling context.
* **#7 (partial)**: `readInterpolatedData`'s raw-pointer API still forces manual `malloc/free` error paths in `TimeRangeReadStrategy`.
* **#8**: `HDF5Reader_v2::endAction` guard `IDS_group_id.size() == 1` (`hdf5_reader_v2.cpp:81`) — with several IDS groups open, `endAction` does nothing; the test should be the context's own group.
* **#9**: no thread safety — mutable per-session caches plus process-wide `static` flags (`HDF5Writer_v2::compression_enabled`, `hdf5_writer_v2.h:45`) race across parallel sessions. The index handoff is `thread_local`, so parallel sessions do not corrupt each other, but one slot per thread means a session alternating between two IDS files re-decodes from scratch each time — and the parked table is **held until another file replaces it**, not released at the end of the pulse. Treat one backend instance = one thread.
* **A slice write still pays for a read session** (measured in `to_improve.md` §25, not fixed): 1 000 `putSlice()` calls build **999 READ engines** — `core_profiles_IDSBase::putSlice` → `readIdsTimeMode` (al-cpp) → `al_read_data` → `HDF5Reader_v2::select_strategy` — which is **457 µs of the 1 943 µs/call** at 1000 × 1000. The read session of §4.1 is there, but its gate refuses each time because the table *has* grown — by this very write. The way out is an incremental `/index` reload (the mechanism §3.6 already uses between engines on one file), never a relaxed gate.
* **`rebuildDynamicRootTimeIndex()` is still a full pass over the leaves** (`panzerdb.cpp:2225`), so the incremental index refresh of §7 plateaus at ~4,5 ms regardless of how few rows were added. Its keys are `std::string_view`s into the strings held by `cached_dynamic_aos_roots` (`panzerdb.h:457`/`:496`), so the map has to be rebuilt from scratch whenever that vector moves — which is why nobody has made this one incremental.
* **`getVersion(ctx)`'s `switch` has no `break`** between `open_pulse`/`force_open_pulse` and `create_pulse`/`force_create_pulse` (`hdf5_backend.cpp:117`-`168`): the pulse branches fall through into `return getTargetVersion()`, so the writer-installing tail never runs in pulse modes — harmless while the constructor installs the target writer, but for an `access_mode` outside those four cases the function returns `versionToReturn` having never written it.
* **Dead code on the read path**: `IReadStrategy::readLeafData` (`iread_strategy.h:849`) has no caller left — it chooses `T` from the AL-requested datatype, which is exactly the dtype assumption §8 says not to make. `readLeavesUnion` selects the *dataset* from the caller's `DataType`, so a wrong caller type would read another table's offsets; not demonstrated reachable, but it is the one place where the width rule of §8 is not enforced.

---

## 11. How to extend the backend (checklist)

* **New range mode / read behaviour** → subclass `IReadStrategy`, reuse `read_dataset_globally`/`find_leaf_for_context`/`getTimeValues`; wire it in `HDF5Reader_v2::select_strategy`. Never call `refresh_index_if_needed()` mid-operation.
* **New stored type** → add a `PanzerDB::DataType`, a `data_raw_*` dataset + buffer + flush path in `init/flush`, and the mapping in `read_dataset_globally`'s type switch **and** in `readTensor`'s width table (§8).
* **New path normalisation rule** → change it **once** in `aos_path_helpers.h` (and its unit test `test_path_sanitization`), not in a strategy.
* **Index layout change** → bump `PANZER_INDEX_VERSION_V3` / `PANZER_INDEX_COLUMNS_V3` (`panzerdb.h:144-165`), change the packing in `append_index_row`, the decode branch in `getLeaves()`, the decoding in `panzer_index.py`, and keep the older branches: APPEND adopts the file's extent and `index_version`, so existing files must keep reading and appending unchanged. Then re-capture the oracle: `test_index_golden` runs in `verify` mode by default and in `capture` mode regenerates `tests/hdf5_backend/fixtures/goldens/index_surface.golden` — the diff of that file is the review surface of a format change (`test_engine_index_v3` covers v3 + upward compatibility).
* **Anything that changes a session's engine lifetime** → both invalidation gates must stay non-vacuous: `test_al_slice_read_after_append_between_reads` (read → append by another backend → read) and `test_al_slice_write_after_foreign_append` (write → foreign append → write). Both gates pin their own target version (`setenv("BACKEND_DEFAULT_WRITE_VERSION", "2.0", 1)` at the top of `main()`, as `test_al_convert_v2_to_v1.cpp:61-65` does): the target comes from the environment, and an ambient `1.0` would silently move the whole test onto the v1 path, where there is no session engine to invalidate. The handoff itself is guarded by `test_engine_index_handoff`, handle ownership by `test_engine_handle_lifecycle` and `test_engine_flush_dataspace_handles`, commit ordering by `test_engine_commit_order_guardrail`, and cache lifetime by `test_index_lifetime_guard`.
* **Verification** → build the target, then from the build directory:
  `ctest -R "test_al_|test_engine_|test_swmr_|test_index_|test_path_|test_read_|test_strip_indices|test_time_resolution|test_direct_api_validation|test_publish_view|test_h5read_overloads"`
  The suite is the contract, and `src/hdf5/CMakeLists.txt` is the authoritative list (each test carries a comment naming the behaviour it pins). `bench_read` is built but deliberately **not** registered as a test: it passes an empty `timebasename` for AoS levels it does not mark dynamic, so its GLOBAL/TIMERANGE numbers describe a call the AL would not make — treat its output as a hint, never as a measurement of AL behaviour.

---

## Change log

* **v2.2 (2026-10-21)** — guide brought up to date with the code (the backend format stays **2.0**; nothing here is a format bump). Rewritten or added: the **v3 index layout** (7 packed u64 columns, interned `/path_table`, `index_version` attribute, upward compatibility with the 12- and 14-column files, u32 overflow guards) in §2; **session-scoped engines** on both sides — the reader re-anchored across AL operations and the writer keeping one `PanzerDB(APPEND)` for a whole slice session — with their invalidation gates and the `IndexHandoff` that parks the decoded table between engines, in §3.1/§3.6/§4.1; the **crash-safe commit ordering** (payloads → path table → span table → `/index` → `H5Fflush`) and the reader's dangling-row guard, §3.6; the **whole-signal memo** and the **"no slice at any index" negative cache** plus `rowsForTime`, §4.2/§4.3; **element-local time bases** and the TIMERANGE window cut, §4.6; `/list_spans` lifecycle (created on demand, deleted by WRITE) and **string run grouping**, §5; the `readTensor` width-vs-dtype contract and the rule that a failed HDF5 read raises, §8; the **upgrade/conversion switches** (`BACKEND_ALLOW_AUTO_UPGRADE`, `BACKEND_HDF5_FORCE_CONVERSION_TO_V1`, `BACKEND_DEFAULT_WRITE_VERSION`) and what a rewrite has to erase and relabel, §1.4. §7 now carries the measured session/commit/index numbers rather than only the 2026 read study, §9 the current view options (`--no-aos-groups`, declared-size padding, `--list`) with measurements re-taken on the reference files, and §10 the open points that still stand (the read session a slice write keeps paying for, `rebuildDynamicRootTimeIndex()`, the `getVersion` fall-through, the handoff slot's retention). Test names, the `ctest -R` pattern and the `bench_read` caveat updated in §11. See `to_improve.md` points 21-28 for the change-by-change rationale.
* **v2.1 (2026-10-07)** — first guide for the backend **v2 as a whole** (AL ↔ engine layer). Documents: the shared session engine + `ReadIndex` with the generation-checked lifetime contract; the 12-column `/index` (and 14-column legacy compatibility); `STRING_CHUNKED` + `list_spans` span table; `@key` metadata replay and `metadata_by_schema`; the three read strategies with the gap-safe TIMERANGE path and the named node-role rules; the writer's single time-role resolution and AoS-stack synchronisation; tools (`panzer_index`, `imas_publish_view`); measured performance history. See `developer_guide_v2.md` (v1) for the PanzerDB engine API reference itself, and `to_improve.md` for the change-by-change rationale.

---

*End of guide.*
