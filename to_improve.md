# Points to improve — HDF5 backend v2 (`src/hdf5`)

Results of the v2 backend analysis (factory / reader_v2 / writer_v2 / IReadStrategy / PanzerDB).

## 1. Three PanzerDB instances per read session — RESOLVED (shared engine + shared index)

`HDF5Reader_v2` keeps one lazily-created PanzerDB per range mode
(`global_strategy`, `slice_strategy`, `timerange_strategy` — `hdf5_reader_v2.h:41-45`).
Each one re-loads the whole `/index` and rebuilds its own `path_cache` + `name_index`
(`iread_strategy.h:306-332` in the `IReadStrategy` ctor). On large files this means
up to 3× the index load and 3× the cache memory.

**Improvement:** share a single PanzerDB (+ its path index) across the strategies;
the strategies only differ in how they select time slices, not in the data they hold.

**Resolution (implemented):**
- New `ReadIndex` class in `iread_strategy.h` owns the engine-wide indexes
  (`path_cache`, `name_index`, `schema_aos_paths`) and keeps the PanzerDB alive
  through a `shared_ptr` (the string_view keys / Leaf pointers stay valid for its
  lifetime).
- `IReadStrategy(hid_t)` ctor → `IReadStrategy(std::shared_ptr<PanzerDB>, std::shared_ptr<ReadIndex>)`:
  it no longer opens the engine nor builds the index; the three strategies adopt
  the session's pair (`global_read_strategy.*`, `slice_read_strategy.*`, `timerange_read_strategy.*`).
- `HDF5Reader_v2` gained `session_db` / `session_index` / `session_gid`: `select_strategy()`
  builds the engine + index once per opened group (reset when the gid changes),
  every strategy then shares them.
- Strategies' `endAction(OperationContext)` no longer call `panzer_db_ptr->close()`
  (it would close the engine shared by the other modes); the engine is flushed/closed
  by RAII on session reset (`open_IDS_group`) or at reader destruction.

Verified: full `al` rebuild + `ctest -R "test_al_|test_engine_|test_swmr_|test_index_golden|test_direct_api_validation|test_h5read_overloads|test_path_parser_syntax"` → 73/73 pass.

## 2. `HDF5Writer_v2::read_homogeneous_time` is heavy — RESOLVED

`hdf5_writer_v2.cpp:38-54` constructed a fresh `PanzerDB(..., READ)` on the group —
loading the whole index table — to read one `int32` scalar, on every SLICE_OP
(APPEND) open. Same cost repeated via `IReadStrategy::getHomogeneousTime()`
(`iread_strategy.h:142`) and `PanzerDB::readScalar` which scanned all leaves
linearly (`panzerdb.h:885-900`).

**Resolution (implemented):**
- `readScalar` read path: O(1) `leaf_lookup` + `getHomogeneousTime()` cached
  per session (done with point 2's read side).
- `HDF5Writer_v2::read_homogeneous_time` (write side): a fresh READ PanzerDB
  (full `/index` load) per APPEND open → now a direct `H5Dopen2`+`H5Dread` of
  the `ids_properties&homogeneous_time` scalar, mirroring PanzerDB::init's
  one-level descent (child group holding `/index`) when needed.

## 3. Fragile heuristics

- `write_ND_Data()` decides "[count,max_len]" vs "[spatial,time]" with
  `size[1] > 20` (`hdf5_writer_v2.cpp:490`) — magic threshold.
- Slice promotion of a time-dependent scalar: "behaviour is ambiguous" +
  heuristic (`slice_read_strategy.cpp:140-150`).
- Timebase detection by suffix `ends_with(dataset_name, "time")`
  (`timerange_read_strategy.cpp:88`) — matches any node named e.g. `…&time_step`.
- Scalar-vs-list detection from `first_leaf->shape` in
  `read_dataset_globally` (`iread_strategy.h:1118-1121`).

**Improvement:** carry the AL-declared shape/role to the index explicitly
(extra column or flags bit) instead of inferring it.

**Resolution (write side, implemented):**
- The writer's "last written dim = time" rule (`timebasename` present outside a
  dynamic AoS) is now computed ONCE for all types (`bulk_temporal`/`base_shape`/
  `n_slices_dyn` in `write_ND_Data`) instead of being re-derived inside three
  copy-pasted per-type blocks; the dispatch collapses to one shared resolution.
- The dead string heuristic `if (size[1] > 20)` is gone: with an explicit
  timebase, the AL always hands one slice of `size[0]` strings (dim=2 is the
  [count,max_len] buffer layout) and both old branches did the same call.

**Still open (read side):** the slice promotion of time-dependent scalars and the
`ends_with("time")` timebase-name detection encode AL conventions/ambiguities;
making them explicit needs the extra index metadata above.

## 4. Path reconstruction duplicated ~6 times

The block "strip the parent path from `arr->getPath()` + `replace('/','&')`" is
copy-pasted in: `find_leaf_for_context` (`iread_strategy.h:369-398`),
`getPath` (`:491-550`), `buildFullPath` (`:560-599`),
`sanitize_path` (`:620-…`), `beginWriteArraystructAction`
(`hdf5_writer_v2.cpp:109-149`), `write_ND_Data` (`:213-251`).

**Improvement:** one shared `localName(arrCtx)` / `buildInstancePath(ctx)` helper.

**Resolution (implemented):** one shared header `aos_path_helpers.h` defines
`localAosName(ctx)` (strip parent prefix + `/`→`&`) and
`collectAosChain(ctx, names, indices [, timed flags])`; the six copy-pasted
blocks (2 in `hdf5_writer_v2.cpp`, 4 in `iread_strategy.h`:
`find_leaf_for_context`, `getPath`, `buildFullPath`, `read_dataset_globally`)
now call these helpers. `getPath`'s dynamic-level index override is driven by
the helper's `timed` flags; `read_dataset_globally`'s `target_time_index` picks
the outermost timed level exactly as the old deepest-first scan did (each timed
level used to overwrite it).

## 5. `sanitize_path` uses `rfind(ctxPath)` — RESOLVED

`iread_strategy.h:656`: the *last* occurrence of the context path inside the
remaining string wins; with a repeated segment name higher up the hierarchy
(e.g. `A/B/…/B/C`), the wrong match is picked. The boundary checks only test
the two adjacent characters.

**Improvement:** anchor the match (search from the tail with exact segment
sequence, or work segment-per-segment on a parsed path).

**Resolution (implemented):** the AL hands *cumulative* paths to every AoS level
(that is what `localAosName` relies on when it strips the parent prefix), so a
path that runs through an open context necessarily **starts** with that context's
path. `sanitize_path` now anchors the match at the head of the string
(`remaining.compare(0, len, ctxPath)` + the next char must be `/` or end of
string), and the matched block plus everything below it become ONE `&`-flattened
name. With chain `A` → `A/B` and input `A/B/A/B/time`, the old code produced
`A/B/A&B/time` (rfind stole the second `A/B`, then the parent `A` matched the
first one and left `B` as a stray level); it now produces `A&B/A&B&time`.
The pure string logic moved to `aos_path_helpers.h` (`sanitizeAosPath(ctx_paths,
path)`); `sanitize_path` keeps only the `(ctx, path)` cache and the chain walk.
New unit test `test_path_sanitization` covers the empty/`time` special cases,
relative paths, absolute-through-context, shallower-ancestor anchoring, no
partial-word match, and the repeated-segment regression.

Verified: full suite **75/75** pass.

## 6. Lifetime coupling between caches and PanzerDB — RESOLVED (generation-checked)

`path_cache`, `name_index`, `leaf_lookup` store `const PanzerDB::Leaf*` and
`std::string_view` pointing into PanzerDB's internal storage
(`cached_leaves`, `cached_paths_blocks`). It works today because the
`leaves_cache_valid` cache is never invalidated after construction, but any
future refresh of `getLeaves()` would leave dangling views in the strategies.

**Improvement:** make the invalidation contract explicit (assert
`leaves_cache_valid` at strategy use, or have `getLeaves()` rebuild
the strategy caches at the same time).

**Resolution (implemented, SWMR-ready):** the contract is now *checked*, and an
index that moves is followed instead of being assumed frozen.
- `PanzerDB::leaf_cache_generation()`: a counter bumped **exactly where the old
  content dies** (the `cached_leaves.clear()` at the top of a `getLeaves()`
  rebuild). Note that `flush()` only sets `leaves_cache_valid=false`: it does NOT
  free anything, so leaves held by readers stay usable across a commit — the
  generation deliberately does not move there.
- `PanzerDB::invalidateLeafCache()` (public): the SWMR refresh hook — refresh the
  file image, invalidate here, and every consumer notices through the generation.
- `ReadIndex` stamps `built_generation` at build; `is_current()` /
  `ensure_current()` compare it and rebuild the maps only when the engine really
  moved (one `uint64` compare when nothing changed). `path_cache`/`name_index`
  became private behind `find_path()` / `find_name()`, so no strategy can hold a
  raw map iterator.
- `IReadStrategy::refresh_index_if_needed()` is called at OPERATION ENTRY
  (`read_ND_Data` + `beginReadArraystructAction` of the three strategies,
  `getTimeValues`, `find_leaf_for_context`, `read_dataset_globally`). Never in the
  middle: the `Leaf*` gathered during an operation (`time_leaves_map`,
  `sorted_leaves`) then stay usable until that operation ends. When a rebuild did
  happen, the values derived from the old leaves are dropped too
  (`time_values_cache`, `homogeneous_time_cache`).
- Dead code of the same pass removed: `schema_aos_paths` +
  `build_aos_schema_index()` were never read anywhere (they costed an O(N)
  `stringstream` walk per session open). Drive-by perf: `restoreTimeContext()`
  copied the whole leaf table (`auto leaves = getLeaves()`) — now `const auto&`.

Verified by `test_index_lifetime_guard`: an index built over an APPEND engine
survives a `flush()` (same generation, old `Leaf*` still usable), then a leaf-cache
rebuild bumps the generation, `is_current()` goes false and `ensure_current()`
re-synchronises it (the new slice becomes visible). Full suite: **76/76** pass.

## 7. Raw memory management at the AL boundary — PARTIALLY RESOLVED (iread_strategy helpers)

`void** data` + `malloc` everywhere:
- `readLeafData` (`iread_strategy.h:764-781`): if `readTensor` throws after the
  `malloc`, `*data` leaks.
- `read_dataset_globally` numeric path and the whole
  `TimeRangeReadStrategy::read_ND_Data` (`timerange_read_strategy.cpp:281-383`)
  manually juggle `malloc`/`free` on every error branch.

**Resolution (implemented, `iread_strategy.h`):**
- `readLeafData`: numerical + string outputs now use `std::unique_ptr<T[]>`
  (`make_unique`); ownership handed to the caller with `release()` on success,
  auto-freed on any throw (the `catch` rewrap no longer leaks).
- `read_dataset_globally`: numeric allocation is a `unique_ptr` per type
  (`buf_d`/`buf_i`/`buf_c`); a throw inside `readLeavesUnion` frees the buffer,
  release() only after `res == 0`; the string 2D buffer is also a
  `unique_ptr<char[]>`.
- `read_homogeneous_time`… unchanged elsewhere: the buffers still `malloc()`d by
  `PanzerDB::readInterpolatedData` (handed to
  `TimeRangeReadStrategy::read_ND_Data`) keep their manual `free()` juggling —
  fixing them means moving ownership into the engine API.

Note: the ASAN build (`build-asan/`) cannot run in this environment (shadow
memory reservation fails on AFS); verification here is the 73/73 normal suite.

## 8. Suspicious guard in `HDF5Reader_v2::endAction`

`hdf5_reader_v2.cpp:65`: `if (read_strategy && IDS_group_id.size() == 1 && …)` —
the session is only closed when exactly one IDS group is open. With
multi-IDS operations (nested dataobjects) `endAction` silently does nothing.

**Improvement:** key the test on the context's own group, not on the global map size.

## 9. No thread safety

Mutable caches (`path_cache`, `time_values_cache`, `sanitized_path_cache`,
`cached_leaves`, …) plus `static` flags (`HDF5Writer_v2::compression_enabled`,
`hdf5_writer_v2.h:45-47`). Parallel reads of several pulses would race.

**Improvement:** document the single-thread contract, or move per-session state
out of the class/statics.

## 10. Index row schema still carries dead columns — RESOLVED

`index_buffer` is 14 columns (`panzerdb.h:270`) with values kept "always zero"
(see the M1 note at `append_index_row`, `panzerdb.h:1063`); `parent_paths_dset`
is still declared (`panzerdb.h:220`) though parent_path is now resolved on read.

**Resolution (implemented):**
- `/index` now has `PANZER_INDEX_COLUMNS = 12` columns (`ndim | shape[6] | time |
  offset | count | flags | parent_id`): the leading "type" and trailing
  "index_value" columns are no longer written (-2 u64 = -14 % of every /index
  byte written AND read).
- Read compatibility: `getLeaves()` takes the stride from the dataset extent
  (`dims[1]`) and maps both the 12-column and the legacy 14-column layouts, so
  existing files still read; `APPEND` adopts the file's own column count
  (`index_cols`), so legacy files keep the legacy layout.
- `test_engine_commit_order_guardrail` (which hand-crafts an index row) made
  layout-aware to match.

## 12. `getWholeDynamicSignal` reads one H5Dread per leaf — RESOLVED

`PanzerDB::getWholeDynamicSignal` (`panzerdb.cpp:3405`) looped a per-leaf
`readTensor`: for a timebase appended many times (one row per session) that is
O(#chunks) small HDF5 reads per `getTimeValues` call.

**Resolution:** one `readLeavesUnion` over the time-sorted leaves (hyperslab
union, monotonic-offset fast path), with the sequential loop kept as fallback.

`index_buffer` is 14 columns (`panzerdb.h:270`) with values kept "always zero"
(see the M1 note at `append_index_row`, `panzerdb.h:1063`); `parent_paths_dset`
is still declared (`panzerdb.h:220`) though parent_path is now resolved on read.

## 11. `readMetadata` rescans the whole index — RESOLVED

`PanzerDB::readMetadata` (`panzerdb.cpp:3495`) scanned every leaf of the file with
a `rfind(prefix,0)` prefix test; it is called by the strategies on **every**
`read_ND_Data` (slice: `slice_read_strategy.cpp:119`, timerange:
`timerange_read_strategy.cpp:430`, global: `iread_strategy.h:910`). Even with the
`written_metadata_schema_paths` guard, each new schema path costs one full
O(N) index walk — tens of full walks per session on 10^5-leaf files.

**Resolution (implemented):**
- `getLeaves()` now also builds `metadata_by_schema`
  (schema path → indices into `cached_leaves`): one hash pass at index-build time,
  using the text before the '@' suffix (metadata rows only, non-empty).
- `readMetadata` = one hash lookup on the stripped schema path, then tiny H5Dreads
  of the few @key values (STRING_CHUNKED join preserved).
- Values are *indices*, not `Leaf*`: the index is built while `cached_leaves`
  still grows (`emplace_back` reallocations would dangle stored pointers — that
  is exactly what a first `Leaf*` attempt exposed, via the SEGFAULT of
  `test_direct_api_validation`).

## 13. Read-parameter study on `core_profiles_3.h5` — the O(slices²) time filter dominates

Instrument (`tests/hdf5_backend/bench_read.cpp`, built but NOT a ctest test):
`--engine` (index load + full leaf re-read per-leaf / union-grouped) and
`--global` (real GLOBAL_OP walk through `HDF5Backend`, tree rebuilt from the
index: 244,513 `readData` calls, 83.4 MB of raw doubles, content checksummed).

File: 171,165 index rows (legacy 14 columns), 9.89M doubles in `data_raw_f64`
(79 MB raw / 36.7 MB on disk, DEFLATE 1 + SHUFFLE), chunk 65536 elements;
28 data paths, each stored as **8,150 per-slice leaves**.

Measured (median of 3, AFS, warm / cold via `posix_fadvise(DONTNEED)`):

| configuration | GLOBAL read | full leaf re-read (engine) |
|---|---|---|
| as-is, chunk 65536 + DEFLATE 1 | 3707 ms | 374 ms (+118 ms index/paths) |
| chunk 16384 / 262144 / 1048576 (DEFLATE 1) | 1118 / 1202 / 1112 ms | 407 / 420 / 404 ms |
| **no compression**, chunk 65536 | 934 ms (cold 1010) | 239 ms (file 81 MB, +110 %) |
| **no compression**, chunk 262144 | 944 ms | 243 ms |
| read chunk cache 1 / 8 / 64 / 128 / 256 / 512 MB, slots 211…10099, w0 0…1 | no effect | 361–402 ms (noise) |

Conclusions:
- **Chunk size and the HDF5 chunk cache are NOT the lever** for this access
  pattern (each chunk is touched once by a full read; the cost is per-call
  overhead + filter CPU). No default was changed for them.
- **Compression** costs ~21 % of the GLOBAL read time for 2.1x the file size
  (deflate level 1 is already the cheapest useful setting) → keep it.
- The dominant cost was algorithmic: for a signal stored as N per-slice leaves,
  every element read **linearly scanned the N candidates** of its path
  (`path_cache` / `name_index` buckets) → O(N²) per full `get()`
  (8,150² × 28 signal scans).

**Resolution (implemented):** `ReadIndex` now stores its buckets **time-ordered**
(`std::stable_sort` by `time_index` at build time) and exposes `equal_time()`,
used by `read_dataset_globally` (specific path, generic path, name fallback) and
`find_leaf_for_context`: the time filter became a binary search.
Same content (identical checksum / element counts), **3707 ms → 1105 ms
(−70 %)** on the GLOBAL walk; full suite 76/76.
Measurement hooks added: `PanzerDB::setDefaultChunkingConfig()`,
`PanzerDB::setReadCache()`, and the `IMAS_HDF5_FAPL` environment override in
`HDF5Utils::openIDSFile` (unset ⇒ `H5P_DEFAULT`, unchanged production behaviour).
