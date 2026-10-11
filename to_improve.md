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

**Resolution (read side, conservative pass):**
- Added `IReadStrategy` helpers (`nearestTimedContext`, `cleanFlatPath`,
  `lastFlatSegment`, `collectTimebaseSchemaPaths`, `isTimebaseDataset`,
  `shouldPromoteTimeScalarOnSlice`) so the old ad-hoc heuristics have named rules.
- `TimeRangeReadStrategy::read_ND_Data` no longer uses `ends_with(dataset_name, "time")`.
  It first matches the requested dataset against the explicit timebase path from
  the AL `timebasename` argument or the nearest timed AoS context
  (`getTimebasePath()`), and only falls back to an exact final segment `time`
  for legacy paths that do not expose a timebase path. Fields such as
  `time_step` or `lifetime` are therefore no longer swallowed as timebases.
- `SliceReadStrategy::beginReadArraystructAction` now uses the timed context's
  declared timebase (`getTimebasePath()`) instead of hardcoding `/time`.
  `test_read_node_roles` adds a dynamic AoS whose timebase is named `clock`.
- Slice promotion of time-dependent scalars is now expressed by
  `shouldPromoteTimeScalarOnSlice()` rather than an inline ambiguous block.
- Scalar-vs-list shape handling is centralized through `PanzerDB::isStringScalarLeaf`;
  the leaf shape remains authoritative (scalar: empty shape; list: rank 1), but
  the read code no longer re-inlines that rule.

**Still open (read side):** fully file-level role persistence (`@timebasepath`,
`@shape_role`, or an index role column/bit) is still useful for files read without
enough AL context and for making the scalar/list contract independent of legacy
shape inference.

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

## 14. SLICE_OP / TIMERANGE_OP never measured — the same O(slices²) sweep, second copy — RESOLVED

Point 13 fixed the time filter on the **GLOBAL** path only. The slice and
time-range strategies resolve their leaf through `PanzerDB` instead of
`ReadIndex`, and that side still swept every row of a path linearly.

`bench_read` gained AL-level modes (`--slice`, `--timerange`, `--nslices`,
`--range`, `--dt`, `--interp`) that walk the tree through `HDF5Backend` and
checksum the content, so a mode can be compared against itself before/after.
Files (warm, AFS, median of 3): `core_profiles_3.h5` = 122,263 index rows,
8,150 time slices, one row per slice per signal; `equilibrium_1.h5` = 56,594 rows.

| mode | cp before | cp after | eq before | eq after |
|---|---|---|---|---|
| GLOBAL | 1111 ms | 1120 ms | 1388 ms | 1399 ms |
| 20 × SLICE_OP | 2636 ms (131.8 ms/get) | 2581 ms (129.1 ms/get) | 8743 ms (437 ms/get) | 8795 ms (440 ms/get) |
| TIMERANGE, full span | **12 949 ms** | **1763 ms (7.3×)** | 1009 ms | 1000 ms |
| TIMERANGE, `--dt 0.1` | 576 ms | 550 ms | — | 974 ms |

Content identical everywhere (cp GLOBAL/TIMERANGE hash `f91016520c33243d`,
244,513 nodes / 19,780,050 doubles in both modes; cp SLICE `cc02212c7c86c329`).
`perf record` on the cp TIMERANGE walk: `PanzerDB::isTimeInLeaf` alone was
**64.4 %** of the time — `leaf_lookup[path]` holds one row per time slice, so
resolving one node cost O(n_slices) and a full-span range get cost
O(n_slices²). After the fix that symbol is out of the profile.

**Resolution (implemented):** `PanzerDB::rowsForTime()` narrows a path's row
list by binary search on `time_index` (11 lookup sites in `readDataByIndex` /
`readStringDataByIndex` / `readIntDataByIndex` / `readComplexDataByIndex` and in
`nearestAvailableSliceIndex`'s `path_covers`). Rows of one path are appended with
increasing, non-overlapping time spans, so at most one row can match; the
ordering is **verified row by row while `leaf_lookup` is built** — any path whose
rows are meta rows, out of order, or overlapping is recorded in
`time_lookup_unsafe_paths` and keeps the original sweep. Doing the check inside
the build loop matters: a separate validating pass measured **+6 %** on the
SLICE_OP walk (the index is rebuilt per operation, see `REVIEW_PERF_hdf5_v2.md` §A).

Also replaced, in `DataInterpolation::getSlicesTimesIndices` /
`getTimeRangeIndices`, the forward/backward scans of the timebase by
`lower_bound`/`upper_bound` (trailing `-9.e40` invalid run trimmed), and hoisted
the per-point `y_slices` map out of `interpolate_with_resampling`. That part is
**within measurement noise (±5 %)** on these two files — the AL-level timebase was
not the hot lookup; it is kept because it is the same quadratic shape on files
with a longer timebase, and because it costs nothing.

Safety net added first: `tests/hdf5_backend/test_time_resolution.cpp` pins the
exact contract of both lookups (returned index, `SLICE_INF`/`SLICE_SUP`, and the
`range` left untouched by some branches) against an oracle copy of the original
scans — 854 comparisons, 4 interpolation modes, fixtures covering single-point
and 8,150-point timebases, trailing invalid runs, `tmin > tmax`, out-of-range
windows and resampling. Suite: 84/84.

**Still open, measured:** cp SLICE_OP ≈ 130 ms/get is dominated by zlib `inflate`
plus the per-operation index reload (`PanzerDB::getLeaves` + `ReadIndex::build_indexes`
+ `~PanzerDB` ≈ 7 % of that walk, and the 94–151 ms index load of point 1 is paid
once per operation because `endAction` closes the file). eq SLICE_OP ≈ 440 ms/get
is ≈ 30 % `inflate`: chunk/filter granularity, not read-path algorithm. cp
TIMERANGE (1763 ms) is still 1.6× a GLOBAL read of the same content.

## 15. Two read-path defects from the bottom-up re-read: swallowed HDF5 error, duplicated slice — RESOLVED (2026-10-11)

`REVIEW_PERF_hdf5_v2.md` items E1 and E3, both fixed.

**E1 — `readTensor` numeric ignored the `H5Dread` status.** The body of
`if (status < 0)` was literally a comment, so a failing read came back as a
success carrying whatever the caller's buffer held. `H5Dget_space` and
`H5Sselect_hyperslab` were unchecked too, unlike the `std::string`
specialisation which already threw. Now all three raise
`ALBackendException` (handles and the COMPLEX128 array type closed first).

Not reachable through the index on a healthy file — the `getLeaves()` guardrail
skips data rows running past the materialised extent — which is why it never
showed up. It is reachable with a half-written index, so it is pinned directly:
`tests/hdf5_backend/test_read_tensor_contract.cpp` hands a hand-built `Leaf` to
`readTensor` (offset past the extent, and offset inside / count past it).
Against the pre-fix code that test reports 3 failures (silent success); with
the fix, 85/85 ctest.

**E3 — TIMERANGE read slice 0 twice.** The shape probe
(`timerange_read_strategy.cpp`) read `requested_times[0]`, reported its shape,
then `free()`d the payload while the fill loop started again at `k = 0` and read
the same slice a second time. The probe payload is now kept and copied to slot 0
and the loop starts at `k = 1`; because the free moved later, the two
early-return paths that sit between (`elem_bytes == 0`, `malloc` failure) free it
themselves so nothing leaks.

**Measured: no gain, as expected.** E3 removes exactly one slice read per
`get()` — 1 of 8150 on a full-span cp range, 1 of 3 on a 2-point range — and the
walk is dominated by the once-per-operation index reload (94–151 ms, point 1).
Medians of 3, cp, checksums identical before/after:

| cas cp TIMERANGE | avant | après |
|---|---|---|
| plage complète (8150 pas) | 1751.6 ms | 1717.4 ms |
| 10 points | 113.7 ms | 110.5 ms |
| 2 points | 107.4 ms | 108.9 ms |

The spread inside a single run (1711–1744 ms on the full span) is larger than the
differences: everything here is noise. E3 is a wasted-work and leak fix, not a
performance one. The TIMERANGE lever that does measure is **E4** — reading the
whole signal and truncating it in RAM — still open. eq regression check
(TIMERANGE 979 ms, SLICE 436 ms/get) unchanged within noise.

## 16. `readTensor` picked its memory type from the leaf, never from `T` (item E2) — RESOLVED (2026-10-11)

`mem_type` comes from `leaf.flags >> 4`, so `H5Dread` writes the stored element
width no matter what the template argument is: `readTensor<int32_t>` on a FLOAT64
leaf writes 8 bytes per element into a 4-bytes-per-element buffer. The exposure is
`iread_strategy.h:842 readLeafData`, which chooses `T` from the datatype the **AL
asks for** rather than from the leaf — a dictionary/file disagreement reaches the
overflow. `readLeafData` has no callers (verified by grep over `src/` and
`include/`), so the defect is latent, not live.

Guard added at the single choke point, and it is a **width** check, not an
equality check, because the opposite direction is a live idiom I found while
looking: `readInterpolatedData` (`panzerdb.cpp:3837`) routes `integer_data`
through `readDataByIndex(double**)`, which over-allocates 8 bytes per element and
whose consumer reads the buffer back as int32 — the values are correct, only the
allocation is 2× too big. An exact-match assert would therefore have broken every
integer slice read. The test pins both directions: narrow buffer raises, wider
buffer (double over an INT32 leaf) must still return the int32 values at the front.

Two consequences recorded as new items rather than fixed: E10 (the integer-via-double
malloc, ×2 per integer tranche) and E11 (`readLeavesUnion` selects the *dataset* from
the caller's `DataType`, so the aggregation branch of `readDataByIndex` hard-coding
FLOAT64 would read INT32 offsets out of `data_raw_f64` if that combination were ever
reachable — not demonstrated reachable, both real paths take the type from the file).

No behaviour change on the bench: cp/eq hashes identical to the point-14 baselines
(`f91016520c33243d`, `cc02212c7c86c329`, `eb9093018cb67253`, `d92537f55c5f2e4b`),
cp GLOBAL 1110 ms / SLICE 127.9 ms/get / TIMERANGE 1731 ms, eq 1384 / 438 / 985 ms.
Suite 85/85 (test renamed to `test_read_tensor_contract`).

