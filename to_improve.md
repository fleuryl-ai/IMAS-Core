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
performance one. (The sentence that used to close this point — “the TIMERANGE lever
that does measure is E4” — was wrong and is retracted in point 17.) eq regression
check (TIMERANGE 979 ms, SLICE 436 ms/get) unchanged within noise.

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

## 17. E4 (TIMERANGE “read the whole signal, then truncate”) is never reached — mesuré (2026-10-11)

Before touching it, the branch was instrumented with a one-shot counter (the
`dynamic_index == -1` branch with a non-empty `timebasename`,
`timerange_read_strategy.cpp:292`) and run over everything available:

| où | hits |
|---|---|
| cp — `--timerange`, `--timerange --range` (2 points), `--timerange --dt 0.1`, `--slice`, `--global` | 0 |
| eq — mêmes 5 modes | 0 |
| `reflectometer_profile_v2.h5` — 3 plages | 0 |
| `ctest` (85 tests) | 0 |

Two claims are therefore retracted, mine in point 15 and in `REVIEW_PERF_hdf5_v2.md`:
E4 is **not** “the TIMERANGE lever”, and it does not explain cp TIMERANGE full span
being 1.6× cp GLOBAL. The measurement against that: both modes return identical content
(244 513 nodes, 19 780 050 doubles, hash `f91016520c33243d`), so a branch that never runs
cannot account for the gap. The gap itself is real; its cause is **not established** and
measuring it is its own task.

Side observation from the same run, **retracted as a backend bug**: on
`reflectometer_profile_v2.h5` a 1 s window appeared to return the same 2 571 141 doubles
as the full span. That came from the bench harness — `bench_read.cpp:244` passes
`timebasename = ""` for AoS levels it does not mark dynamic, which takes the static
fallback `timerange_read_strategy.cpp:155`. Probed through `HDF5Backend` with the
timebasename the AL takes from the dictionary (`n_e/time`), the same file filters:
`dim=2 size=[100,5126]` on the full span, `size=[100,164]` on `[34.942, 34.95]`. See
point 19 for the real inhomogeneous-time bug found while chasing this one.

Decision: leave E4 alone. Nothing measured justifies refactoring the 380-line
`read_dataset_globally` for a branch no available file, IDS or test reaches.

## 18. Cause of the remaining GLOBAL→TIMERANGE gap, measured — and one fix applied (2026-10-11)

**Instrumentation:** temporary call counters on the four payload readers
(`readSliceDirect`, `readLeavesUnion`, `readTensor`, `readStringBulk`), `--reps 1`, cp:

| mode | calls HDF5 | éléments lus | éléments voulus |
|---|---|---|---|
| GLOBAL | 228 203 (`readLeavesUnion`) | 19 804 501 | 19 804 501 |
| TIMERANGE pleine plage | 228 200 (`readSliceDirect`) | 19 788 200 | — |
| TIMERANGE 2 points | 56 | 4 856 | — |
| SLICE (20 gets) | 560 + 80 | 48 560 + 489 020 | — |

So the two full walks make **the same number of HDF5 reads of the same volume**
(≈228 k reads of ≈87 elements, no read amplification: disk == wanted). The 1.6× is
therefore not I/O. `perf` comparing the two walks names it: in the TIMERANGE profile
`stripIndices` 1.84 % + `getline` 1.04 % + the stream-construction cluster
(`std::locale::locale` 1.74 %, `basic_ios::_M_cache_locale` 1.07 %,
`__ostream_insert` 1.84 %) ≈ 7 %, allocation churn (`_int_free` 3.0 + `malloc` 2.33 +
`cfree` 1.36) 6.7 % vs 3.8 % in GLOBAL, string hashing 4.7 % vs 3.3 %, `memmove` 6.98 %
vs 2.6 %. Decompression is *lower* in TIMERANGE (11.9 % vs 18 %) because the per-node
bookkeeping takes a larger share. `stripIndices` is called **twice per node**
(`iread_strategy.h:758`, `:800`).

**Fix applied:** `PanzerDB::stripIndices` rewritten without `stringstream`/`getline`
(one pass, one string). Equivalence proven, not assumed: `test_strip_indices.cpp` pins
a 22-case table (empty segments collapsing, all-index paths → empty, near-numeric
segments kept, non-ASCII), and a standalone differential run of the old vs new body over
300 021 inputs (21 hand cases + 300 000 fuzzed paths over an alphabet containing `/`)
reports **0 mismatches**.

**Measured** (medians of 3, cp, hash `f91016520c33243d` unchanged):

| mode | avant | après |
|---|---|---|
| TIMERANGE pleine plage | 1731 ms | **1541 ms (−11 %)** |
| SLICE (par get) | 127.9 ms | **120.2 ms (−6 %)** |
| GLOBAL | 1110 ms | 1076 ms (~−3 %, dans le bruit : étalement 1074-1131) |

Suite 86/86. Reste à attaquer dans le même poste : les constructeurs de chemin en
`stringstream` du chemin GLOBAL (`iread_strategy.h:966`, `:1003`) et le hachage de
chaîne par nœud.

**Correction de E5** (revue) : dans les profils `perf` de la marche TIMERANGE cp, les seuls
`std::sort` visibles sont celui de `ReadIndex::build_indexes()` (0,14 % — il relève du poste
A1) et celui de `read_dataset_globally` (0,01 %). Les tris listés par E5 (les quatre
`readXByIndex`, `getTimeIndex`, `getWholeDynamicSignal`, `sorted_leaves`) n'y apparaissent
pas : leur bucket compte un seul élément sur ce chemin, et la branche d'agrégation
`time_index == -1` n'est pas la marche par nœud. Ils ne sont pas la cause du 1,6×.

## 19. TIMERANGE sans filtrage en temps non homogène (AoS statique, base de temps par élément) — BUG CORRIGÉ (2026-10-11)

**Reproduit** : IDS en `homogeneous_time = 0`, AoS **statique** dont la base de temps est
posée sur l'élément d'AoS (la forme de `reflectometer_profile/channel[]/n_e`) — le
TIMERANGE renvoyait **rc = 0 sur tout** (le signal *et* le nœud `time`), y compris sur la
plage complète. Test : `tests/hdf5_backend/test_al_timerange_inhom_static_aos.cpp`,
12 échecs avant le correctif, ALL PASSED après (vérifié en stashant les deux sources).

**Cause** : `TimeRangeReadStrategy::read_ND_Data` résolvait `time_basis_vector` par
`getTimeValues(ctx, homogeneous_time)` — **sans le `timebasename` du nœud**, alors que le
chemin SLICE le passe (`slice_read_strategy.cpp:84`). Or `IReadStrategy::getTimeValues` ne
trouve une base de temps locale à un élément d'AoS statique qu'à l'aide de ce
`timebasename` ; sans lui la résolution rend un vecteur vide, `getTimeRangeIndices` rend
-1, et la lecture rend 0.

**Correction** (`timerange_read_strategy.cpp`, `iread_strategy.h`) :
1. passer `timebasename` à `getTimeValues`, et re-résoudre **par nœud** quand la base est
   par élément (`homogeneous_time == 0 && !timebasename.empty() && !isTimedContext(ctx)`) —
   sinon le deuxième élément hériterait de la base du premier ;
2. cache `time_values_cache["ELEMENT:" + chemin]` pour la branche locale (la clé porte
   l'index d'élément, via `buildFullPath`), sinon relecture du timebase à chaque nœud ;
3. le nœud `time` de l'élément passait par le chemin « signal » et sortait en `dim = 2`
   avec le tableau complet répété une fois par temps demandé ; il sort maintenant la base
   **coupée à la fenêtre** (`dim = 1`).

**Coût mesuré** : la première version (résolution par nœud dans tous les cas) faisait passer
le TIMERANGE cp de 1541 à 1933 ms (+25 % : ~1,6 µs de `sanitize_path` × 244 513 nœuds).
La restriction au cas par-élément ramène la médiane à 1578 ms (étalement 1542-1581 sur
5 reps), hash cp inchangé `f91016520c33243d`. Suite : 87/87.

**Rectification sur `reflectometer_profile_v2.h5`** : la mesure initiale (« la fenêtre est
ignorée, 2 571 141 doubles quel que soit l'intervalle ») venait du **harness de bench**, pas
du backend — `bench_read.cpp:244` passe `timebasename = ""` aux nœuds d'un AoS qu'il ne
déclare pas dynamique, ce qui déclenche le repli global (`timerange_read_strategy.cpp:155`).
Ce fichier est en `homogeneous_time = 1` avec un `time` racine de 5128 points
(34,942 → 41,356). Sonde `HDF5Backend` avec le `timebasename` que l'AL tirerait du
dictionnaire (`n_e/time`) : plage complète → `dim=2 size=[100,5126]` ; fenêtre
`[34.942, 34.95]` → `size=[100,164]` et non 5128. Le filtrage se faisait donc déjà.
`[35, 36]` → rc = 0 est **correct** : la base saute de 34,952 à 37,742 et ne contient aucun
point dans [35, 36] (vérifié sur le payload).

**Reste ouvert** : `timebasename` vide sur un nœud *dynamique* donne toujours une lecture
globale (fenêtre ignorée). Inatteignable par l'AL — tout nœud `dynamic` a un `timebasepath`
au dictionnaire (`channel/n_e/data` : `timebasepath="n_e/time"`, vérifié dans IDSDef.xml) —
donc pas de correctif sans reproducteur.

## 20. `restoreTimeContext()` : ~11 ms par ouverture perdus en allocations par feuille — TRAITÉ (2026-10-11)

**Coût réel (mesuré, sonde chrono temporaire dans `restoreTimeContext`, pilotée par
`E6_PROBE=1`)** : sur cp (171 165 feuilles, 29 compteurs, 1 root AoS dynamique) la boucle
coûte **11,3 ms par ouverture** (78,8 ms sur 7 ouvertures), à côté de `getLeaves()`
**90 ms** — c'est le second poste du coût d'ouverture, pas les 1,65 % annoncés par la revue.
Sur une marche SLICE à 20 get (1 ouverture), 11,3 ms ≈ **4,7 % du total** ; c'est bien ce
que donne l'A/B ci-dessous.

**La prémisse de la revue est fausse** : la table n'est *pas* inutile en READ.
`getDynamicAOSSize()` lit `aos_time_counters` et est chargé sur le chemin de lecture
(`iread_strategy.h:429`, `direct_reader.cpp:49`). Ce qui est supprimable, c'est le coût par
feuille, pas l'appel.

**Ce qui coûtait** : par feuille, `aos_path + "/"` (allocation) × chaque root AoS dynamique,
plus une `std::string` clé par feuille (allocation + copie), pour une table qui finit à
29 entrées sur cp.

**Correction** (`panzerdb.cpp`, boucle 2 de `restoreTimeContext`) :
1. clé `std::string` **réutilisée** (`key.assign(leaf.path...)`) — une seule capacité
   amortie pour toute la boucle ;
2. plus de concaténation par root : le parent AoS est trouvé par comparaison de préfixe sur
   `leaf.path` (`leaf.path[aos_size] == '/'` + `compare`), le plus imbriqué gagnant ;
3. `file_time_base` n'est plus copié qu'en mode **APPEND** — `hasFileData()` (son seul
   consommateur, lignes 2260/2292/2402/2422) n'est appelé que depuis
   `writeDataSlicesImpl`/`writeDataSlices`, donc la copie est morte en READ ; et en WRITE
   elle serait vide de toute façon, `init()` supprimant `/index` (lignes 196-204) avant
   l'appel à `restoreTimeContext()` (ligne 408). En eq cela évite la copie d'une carte de
   43 598 entrées à chaque ouverture.

**Preuve d'équivalence** : différentiel old/new (`/tmp/time_ctx_diff`, deux implémentations
sur les mêmes feuilles) sur 9 groupes d'IDS couvrant les trois formats d'index — 14 colonnes
legacy, 12 colonnes v2, 7 colonnes v3 (core_profiles_3, equilibrium_1, magnetics_FAT,
reflectometer v1/v2, /tmp/bench cp+eq, test_equilibrium_v2, test_reflectometer_profile_v2) :
**131 598 entrées de compteurs comparées, 0 écart**. L'ancienne table concordait aussi avec
`getDynamicAOSSize` vivant pour chaque root.

**Test goupille** : `tests/hdf5_backend/test_engine_time_context.cpp` — fige la table
reconstruite à l'ouverture (`A/0/B` = 5, `A/1/B` = 2, compteurs par chemin 3 / 5, feuille en
mode bulk = **1** pas de temps quelle que soit sa taille, `getAOSShape().front()` = 5) et
vérifie qu'APPEND repart de cette table (2 tranches de plus → taille 7, tranche 6 relisible).
Il passe **aussi sur HEAD** : les valeurs figées sont le comportement préexistant, pas
celui inventé par la réécriture.

**A/B — coût d'ouverture** (le chrono `index` de `bench_read`, qui encadre la construction
du moteur : relecture de `/index` + table de compteurs ; A = HEAD, B = boucle réécrite seule,
C = B + snapshot réservé à APPEND ; paires A/C strictement alternées, 4 reps chacune) :

| fichier | A (ms/ouverture) | C (ms/ouverture) | écart |
|---|---|---|---|
| cp (171 165 feuilles, 29 compteurs) | 113,6 / 108,1 / 95,9 / 95,2 | 108,8 / 102,5 / 91,9 / 91,1 | **−4 à −5 ms**, chaque rep de C sous son rep de A |
| eq (58 162 feuilles, 43 598 compteurs) | 60,0 / 49,1 / 41,4 / 40,5 / 40,2 | 53,9 / 46,2 / 37,0 / 34,8 / 35,0 | **−5 à −7 ms**, même ordre |

Concordant avec la sonde : la boucle passe de 11,3 à **6,2 ms** par ouverture sur cp.

**A/B — temps mural** (médiane de 3 séries de `--reps 3`) et décomposition des deux
sous-corrections :

| fichier | opération | A | B (boucle) | C (+ APPEND-only) |
|---|---|---|---|---|
| cp | SLICE (20 get) | 119,59 ms | 114,06 ms (−4,7 %) | 114,07 ms |
| cp | TIMERANGE | 1572,1 ms | 1558,3 ms (−0,9 %) | 1560,3 ms |
| cp | GLOBAL | 1037–1099 ms | 1056–1058 ms | 1036–1145 ms — dans le bruit |
| eq | SLICE (20 get) | 427,36 ms | 427,05 ms | **414,72 ms** (−3,0 %) |
| eq | TIMERANGE | 918,0 ms | 907,4 ms | **886,1 ms** (−3,5 %) |
| eq | GLOBAL | 1333–1395 ms | 1332–1378 ms | 1351–1456 ms — dans le bruit |

Lecture : sur cp la table ne compte que 29 entrées, donc tout le gain vient de la boucle ;
sur eq la boucle est noyée par les 43 598 entrées et le gain vient de la copie de carte
supprimée. GLOBAL (une seule ouverture par rep, lecture dominateur) reste sous le **plancher
de bruit de ±4 %** de cette machine : un premier bloc A/C sur eq GLOBAL annonçait +4 % de
ralentissement ; l'alternance stricte A-C-A-C le réfute (C ≈ A sur la partie lecture, et C
gagne nettement sur le chrono d'ouverture).

Empreintes de contenu inchangées sur les trois opérations et les deux fichiers
(`cc02212c7c86c329` / `f91016520c33243d` / `d92537f55c5f2e4b` / `eb9093018cb67253`).
Suite : **88/88**. Sonde chrono retirée du source.

**Ce qui reste** : la table de compteurs est encore reconstruite entièrement à chaque
ouverture (eq : 43 598 entrées de `std::unordered_map`). Le vrai gain d'ouverture reste A1
(rechargement incrémental de `/index`) : en régime chaud le chrono d'ouverture est de
95→92 ms sur cp et 40→34 ms sur eq, c'est donc ~92 ms / ~34 ms par ouverture qui subsistent.

---

## 21. Rechargement incrémental de `/index` entre moteurs d'un même fichier (A1) — TRAITÉ (2026-10-11)

**Le coût mesuré d'abord.** `setWriteStrategy(SLICE_OP)` (`hdf5_events_handler.cpp:35`) crée
un PanzerDB APPEND neuf par pas de temps, donc `init()` → `restoreTimeContext()` →
`getLeaves()` relit **toute** la table à chaque pas. Chrono `index` de `bench_read` en régime
chaud, sources HEAD : **86,5–112,8 ms** par ouverture sur cp (171 165 feuilles), **42,0–52,3 ms**
sur eq. C'est ce qui rend l'écriture quadratique : T pas coûtent O(T × N).

**Le mécanisme.** L'état décodé survit au moteur qui l'a produit : `PanzerDB::IndexHandoff`
garé dans un `thread_local` (un seul bloc — le dernier fichier touché, pas une carte par
fichier), rendu par `close()` (`handBackIndexHandoff`), repris par `init()`
(`adoptIndexHandoff`). Neuf conteneurs sont échangés par `std::swap` ; `leaf.path` et
`parent_path` restent valides parce que `cached_paths_blocks` est un `std::list` de blocs
(nœuds stables). `getLeaves()` ne relit que la queue `[index_rows_loaded, n_rows)` en
`H5Sselect_hyperslab` et pousse un bloc de plus ; `leaf_cache_gen` continue de monter, donc
le contrat des consommateurs qui détiennent des `Leaf*` est respecté. `rebuildDynamicRootTimeIndex()`
reste une reconstruction complète : ses clés sont des `string_view` pointant dans
`cached_dynamic_aos_roots`, qui grossit — fusion incrémentale les ferait danser.

**Quatre garde-fous** — la table n'est réutilisable que si elle est toujours la tête de la
table disque : même nom de fichier HDF5 (`H5Fget_name`), même objet `/index` (`H5Otoken_cmp` —
un dataset recréé a un nouveau jeton, ce que `WRITE` fait), extent qui n'a pas rétréci, et
même couple `(offset, count)` sur la dernière ligne lue. L'ouverture WRITE (truncate) purge
l'état garé de ce fichier. Un moteur d'un autre fichier ne reprend jamais rien.

**Piège rencontré** : l'identité (nom + jeton) doit être relevée dans `init()`, pas dans
`close()`. Dans le chemin AL, le groupe sur lequel le moteur est accroché est déjà fermé
quand `PanzerDB::close()` s'exécute (`H5Iget_type(file_id) == -1`) : relevée à la fermeture,
l'identité est perdue et l'écrivain ne gare jamais rien — la première version du correctif
ne changeait donc rien pour lui.

**Ouverture, même set des deux côtés** (médiane de 4 reps) :

| fixture | chrono `index` avant | après |
|---|---|---|
| cp, mode moteur | 89,4 ms | **11,8 ms** |
| eq, mode moteur | 52,3 ms | **11,8 ms** |
| cp, mode AL | 91,2–116,1 ms | **6,1–6,9 ms** |
| eq, mode AL | 42,0–58,9 ms | **7,1–7,7 ms** |

**Écrivain AL** — 8 `put` SLICE_OP sur une copie du pulse `core_profiles_3` (8 150 pas) :

| | HEAD | avec handoff |
|---|---|---|
| open+begin (moyenne 8 pas) | 98,36 ms | **29,66 ms** (158 ms au premier pas froid, 10,1–13,3 ms ensuite) |
| TOTAL par pas | 108,94 ms | **40,22 ms** |
| régime établi (pas 1–7) | 104,5 ms | **20,8 ms** (−80 %) |

La courbe pas-à-pas n'est plus montante : 123,6 → 115,0 → 118,8 → 90,2 → 78,2 → 74,2 → 91,1 →
96,0 ms avant, 158,2 → 13,3 → 12,7 → 12,3 → 10,2 → 10,2 → 10,2 → 10,1 ms après.

**Temps mural** (médiane, paires strictement alternées HEAD / corrigé) : cp SLICE 20 get
2 300,6 → **692,6 ms** (115,03 → 34,63 ms/get), cp TIMERANGE 1 646,5 → **1 472,2 ms**
(−10,6 %), cp GLOBAL 1 142,9 → **985,7 ms** (−13,8 %), eq SLICE 8 458,8 → **7 734,8 ms**
(−8,6 %), eq TIMERANGE 956,5 → **927,8 ms** (−3,0 %), eq GLOBAL 1 372,0 → 1 429,9 ms
= +4,2 %, sous le plancher de bruit de ±4 % (GLOBAL n'ouvre qu'un moteur par rep).

**Différentiel de décodage** — la table obtenue par la chaîne incrémentale est identique à un
chargement à froid d'une copie du même fichier, et aux deux décodages HEAD : v3 en chaîne de
5 moteurs APPEND (2 500 feuilles, préfixes garés 0/500/1000/1500/2000) → `388d4d2498cbdb91`
partout ; cp 14 colonnes (171 165 feuilles) → `8a812af4014fb41b` ; eq 12 colonnes
(58 162 feuilles) → `254bc8cd49ba2fdc`. Empreintes de contenu des opérations inchangées
(`cc02212c7c86c329` / `f91016520c33243d` / `d92537f55c5f2e4b` / `eb9093018cb67253` côté AL,
`c46c2d515f8dba31` / `a381e258df2800b0` côté moteur).

**Mémoire** : l'état garé n'ajoute pas une seconde copie de la table — RSS max 208 744 kB
pour une seule ouverture contre 209 516 kB pour une chaîne de quatre sur cp (+0,4 %). Il est en
revanche conservé tant qu'un autre fichier n'est pas ouvert : la mémoire rendue par `close()`
ne l'est plus.

**Le lecteur du même moteur** (`invalidateLeafCache()` puis `getLeaves()`, le chemin du rafraîchissement
SWMR) tombe sur la même voie incrémentale : 500 lignes ajoutées puis rafraîchies sur la copie du
pulse cp → **4,53 ms** pour 171 165 → 171 665 feuilles, contre ~90 ms de re-décodage complet en
chaud sur HEAD. Ce qui reste dans ces 4,53 ms n'est pas le disque mais `rebuildDynamicRootTimeIndex()`
(parcours complet des 171 665 feuilles) et la réallocation de `cached_leaves`.

**Test goupille** : `tests/hdf5_backend/test_engine_index_handoff.cpp` — chaîne de cinq moteurs
qui relit les tranches garées par les précédents (chemins internés avant le handoff), reopen
sans rien ajouter, truncate WRITE qui ne doit pas faire ressurgir les lignes supprimées, et
fichier voisin qui ne doit rien hériter. Il échoue sur les sources HEAD (3 échecs, génération
qui repart à 1). Suite : **89/89**.

**Ce qui reste** : l'autre idée du poste A1 — garder l'APPEND engine ouvert entre opérations —
n'est pas faite (le handoff obtient le même effet sans toucher au cycle de vie AL) ; les
~6 ms par ouverture encore debout sont `restoreTimeContext()` + l'ouverture HDF5 du fichier ;
`endAction` de l'écrivain coûte encore ~9,4 ms par pas. Et un défaut indépendant, **repris et
corrigé au point 22** : ouvertures APPEND répétées ⇒ segfault à la sortie du processus dans
`H5_term_library` (`H5F_term_package → H5I_clear_type → H5VL_file_close → H5I_find_id`),
reproduit ici encore après 10 ouvertures sur cp — les mesures ci-dessus restent valides (le
crash est après le dernier chrono).

---

## 22. Un moteur ouvert par nom de fichier ne ferme jamais son fichier — CORRIGÉ (2026-10-12)

**Le symptôme.** Ouvertures APPEND répétées d'un même fichier ⇒ **segfault à la sortie du
processus**, dans le nettoyage HDF5 : `H5_term_library → H5F_term_package → H5I_clear_type →
H5VL_file_close → H5I_find_id`. Reproduit sans l'AL ni `HDF5Writer_v2` : une boucle
`{ PanzerDB db(path, APPEND); }` suffit. Les données sur disque étaient justes (`flush()`
commite), ce qui explique que rien d'autre ne l'ait attrapé.

**La mesure avant, pas l'intuition.** Compteur d'objets vivants (`H5Fget_obj_count(H5F_OBJ_ALL, …)`),
12 cycles APPEND sur une copie de cp (171 165 feuilles) :

| après le cycle n° | 0 | 1 | 2 | 3 | … | 9 |
|---|---|---|---|---|---|---|
| `FILE` | 1 | 2 | 3 | 4 | … | 10 |
| `DATASET` / `GROUP` | 0 | 0 | 0 | 0 | … | 0 |

Un `hid_t` de fichier survit par moteur — et par lui seul. Le **READ fuit pareil** (6 cycles ⇒
6 ids fichier) mais HDF5 referme sans bruit des fichiers lecture seule : `exit=0`. En APPEND le
fichier est ouvert `RDWR` et modifié, et HDF5 plante en le refermant. C'est aussi pourquoi
`bench_read` (qui lit) n'a jamais crashé, alors que la boucle d'écriture, si.

**Revérifié le 2026-10-13** sur un fichier minimal à tables dans un groupe enfant (la fixture cp
de la table ci-dessus est partie avec `/tmp`, vide entre deux journées de travail) : 6 cycles
APPEND sans le correctif ⇒ `FILE=1,2,3,4,5,6` après chaque cycle, `DATASET`/`GROUP` à 0, et
**`exit=139` sur les trois runs** (le gestionnaire `atexit` de la sonde ne s'imprime même pas :
HDF5 meurt dans son propre nettoyage, qui passe donc *avant* lui). Avec le correctif : `FILE=0`
partout, à chaque cycle comme à la terminaison, **`exit=0` sur les trois runs**. En READ sans le
correctif, même fuite (6 ids) puis `FILE=0` à la terminaison et `exit=0` : c'est HDF5 qui les a
fermés lui-même, sans bruit — la nuance qui explique l'absence de crash côté lecture.

**La cause.** `should_close_loc_id` vaut `false` par défaut (`panzerdb.h:330`) et le
constructeur par nom de fichier ne le met jamais à `true` alors que c'est lui qui fait le
`H5Fopen`/`H5Fcreate`. Quand `/index` n'est pas à la racine — le cas de **tous** les fichiers
IMAS réels, où les tables vivent dans le groupe `<ids>` — `init()` descend (panzerdb.cpp:245-250) :
il remplace `file_id` par le groupe enfant et passe `should_close_loc_id` à `true`. `close()`
fait alors `H5Gclose(file_id)` sur le groupe et **ne ferme jamais le fichier**. Si `/index` est
à la racine, la branche `else if (H5Iget_type(file_id) == H5I_FILE)` rattrape le coup — d'où un
défaut invisible sur les petits fichiers plats des tests.

**Le correctif.** Le moteur qui a ouvert le fichier le possède : membre `owned_file_id`
(`panzerdb.h:270-274`), pris dans le constructeur par nom (panzerdb.cpp:204-208), fermé dans
`close()` quand `init()` est allé travailler dans un groupe (panzerdb.cpp:1299-1310). Rien
d'autre ne change de main : le constructeur par `loc_id` garde `owned_file_id = -1` et la
sémantique de `close_loc_id_on_exit` est inchangée.

**Après**, mêmes compteurs, mêmes cycles : `FILE=0 DATASET=0 GROUP=0 …` après chaque cycle **et**
au moment de la terminaison HDF5, `exit=0`. En READ aussi (6 cycles ⇒ 0).

**Test goupille** : `tests/hdf5_backend/test_engine_handle_lifecycle.cpp` — fichier posé comme un
fichier pulse (`/index` dans un groupe enfant), cinq moteurs APPEND sur le chemin du fichier, et
la variation du nombre d'ids vivants doit être nulle ; le même contrôle sur un fichier plat, qui
déjà marchait, garde la branche de rattrapage sous verrou. Sur les sources HEAD : **3 échecs et
segfault à la sortie** ; après : `ALL PASSED`, `exit=0`. Suite : **90/90**.

**Ce que ça coûte.** Fermer le fichier est du travail qui n'était simplement jamais fait. Sur la
boucle d'ouverture cp, paires strictement alternées (3 runs de 12 ouvertures chacun, moyenne des
ouvertures établies) : **6,15 / 6,23 / 6,26 ms → 6,41 / 7,45 / 7,74 ms**, soit +0,2 à +1,5 ms par
ouverture (~+20 % du socle de ~6,2 ms). Le chrono `index` de `bench_read` passe de 11,5 à 16,1 ms
sur cp et 16,4 à 18,9 ms sur eq (mesure unique, à prendre comme ordre de grandeur). Le temps mural
des marches n'en porte pas la trace dans la limite de bruit : cp SLICE 554,9 → 520,9 ms,
cp TIMERANGE 1 490,0 → 1 492,8 ms, cp GLOBAL 989,2 → 974,4 ms, eq SLICE 7 838,8 → 7 640,8 ms,
eq TIMERANGE 920,3 → 920,4 ms, eq GLOBAL 1 494,1 → 1 448,1 ms. La boucle d'écriture AL est
inchangée (19,5–22,7 ms par pas établi, contre 20,8 ms au point 21) et sort proprement.
**Empreintes de contenu identiques** avant/après sur les six marches (`cc02212c7c86c329`,
`f91016520c33243d`, `d92537f55c5f2e4b`, `eb9093018cb67253`, `c46c2d515f8dba31`,
`a381e258df2800b0`).

**Ce qui reste** dans le même coin : rien de mesuré. Le soupçon était que `close()` appelle
`H5Gclose(file_id)` sur un id que l'AL a déjà fermé (`H5Iget_type(file_id) == -1`, observé en
traçant A1) et aille fermer un handle **vivant** dont le numéro aurait été recyclé. Sonde (moteur
sur un groupe qu'il ne possède pas, `close_loc_id_on_exit = false`, groupe fermé par le
propriétaire pendant que le moteur vit, nouveau groupe ouvert avant la destruction du moteur) :
`close()` n'y ferme **rien du tout** — `should_close_loc_id` est `false` et
`H5Iget_type(file_id) == -1`, donc aucune des deux branches ne part. Compteurs : le groupe du
propriétaire se ferme (`H5Gclose → 0`), moteur détruit il reste 1 groupe vivant (le nouveau), qui
reste utilisable (`H5Gopen2 → 2`) ; les ids ne sont d'ailleurs pas des petits entiers séquentiels
ici (`gA=144115188075855872`, `gB=144115188075855873`). Le moteur ne ferme que ce qu'il possède :
le groupe enfant ouvert par `init()`, le `loc_id` si on le lui a confié, ou le fichier qu'il a
ouvert (`owned_file_id`).

## 23. A3 (supprimer les `H5Dget_space` redondants de `flush()`) — MESURÉ, ABANDONNÉ ; le vrai coût est la recompression du chunk de queue (2026-10-13)

**La question.** `flush()` pose le même motif six fois (une fois par table, plus
`flushPathTable` et `flushListSpans`) : `H5Dget_space` → lire l'extent → `H5Sclose` →
`H5Dset_extent` → **second** `H5Dget_space` → hyperslab → `H5Dwrite`. A3 proposait de remplacer
le premier appel par un extent mis en cache (`disk_size_*` existe déjà et porte les offsets des
lignes d'index). Avant d'implémenter : mesurer la part de ces appels. Porte fixée avec l'utilisateur :
si c'est sous le bruit, on s'arrête là et on rapporte.

**Comment j'ai attribué.** Sonde `.qwen/tmp/flush_bench.cpp` (hors dépôt) : un seul moteur APPEND,
`T` fois `{ writeData("sig<k>", {2}, v, 2); flush(); }`, `flush()` chronométré à chaque tour.
En l'absence de symboles exploitables côté HDF5, instrumentation *temporaire* dans `flush()`
(RAII accumulant `first_get_space` / `set_extent` / `H5Fflush` / total, imprimée à la sortie du
processus), annulée par `git checkout` avant toute implémentation. Toutes les mesures avec
`IMAS_AL_DISABLE_VALIDATE=1` (l'AL valide l'IDS à chaque `put`/`put_slice`, ce qui met du CPU dans
la fenêtre mesurée — `common/doc_common/use_ids.rst:241`).

**Décomposition, 400 commits sur une copie de cp** (`/index` 171 165 × 14) :

| poste | cumulé 400 commits | par commit | part |
|---|---|---|---|
| premier `H5Dget_space` (6 sites) | 4,7 ms | 0,012 ms | 0,14 % |
| `H5Dset_extent` | 3,6 ms | 0,009 ms | 0,11 % |
| `H5Fflush(GLOBAL)` | **3 289,8 ms** | **8,22 ms** | **99,3 %** |
| `flush()` total | 3 311,4 ms | 8,28 ms | 100 % |

Sur eq (103 Mo, `/index` 58 162 × 12) : `H5Fflush` 4 518,0 / total 4 545,7 = **99,4 %**, flush
médian 11,12 ms ; `first_get_space` 7,3 ms, `set_extent` 4,0 ms sur les 400 commits. Sur
magnetics_THIN (178 Ko, 542 lignes d'index) : flush médian **3,51 ms**, `H5Fflush` 1 411,7 /
1 419,8 = 99,4 % ; sur magnetics_FAT (4,2 Mo, **mêmes** 542 lignes) : 3,51 ms — le coût ne suit
donc pas le volume de données.

**Verdict A3 : 0,25 % du commit.** Rien d'implémenté, rien de committé. Le cache d'extent aurait
gagné ~20 µs par commit et aurait introduit un invariant risqué (`H5Dset_extent` plus petit
**tronque** : il n'aurait plus fallu se tromper sur qui détient la vérité de l'extent).

**Ce que la mesure a livré à la place.** `H5Fflush` n'est **pas** de l'E/S : sur 200 commits de cp,
`strace` compte 1 781 `pwrite64` pour **7,41 ms** et 7,7 ms de syscalls au total, quand `H5Fflush`
coûte 1 663,8 ms ; `/usr/bin/time -v` donne **user 1,93 s / sys 0,12 s**. C'est du CPU utilisateur
dans HDF5. Or `H5Fflush` est aussi le moment où HDF5 écrit les **chunks sales** des datasets
chunkés — donc où il les **recompresse**. Les tables sont chunkées avec `shuffle` + `deflate` :
dans les fixtures réelles, chunk index 4096×14 (458 Ko), paths 4096 (1 Mo), f64 65536 (512 Ko),
i32 131072 (512 Ko), deflate **niveau 1** ; un commit salit le chunk de queue de chaque table
touchée, soit ~2 Mo à recompresser à chaque pas. Sonde de confirmation (`.qwen/tmp/dcpl.cpp` pour
lire les dcpl ; mode `fresh` de `flush_bench` pour créer un fichier neuf avec
`PanzerDB::setDefaultChunkingConfig`, 400 commits, fichier frais donc sans contenu déjà en place) :

| compression / chunks (index · path_table · f64) | flush médian | taille du fichier |
|---|---|---|
| L6 / 8192 · 8192 · 131072 (défaut) | **11,32 ms** | 2 107 180 |
| L1 / 8192 · 8192 · 131072 | 6,24 ms | 8 398 772 |
| aucune / 8192 · 8192 · 131072 | 1,02 ms | 14 680 064 |
| **L6 / 512 · 512 · 8192** | **1,05 ms** | **2 100 379** |
| L1 / 512 · 512 · 8192 | 0,59 ms | 2 100 863 |
| L6 / 1024 · 1024 · 16384 | 1,71 ms | 2 100 722 |

Tout est constant sauf la politique de chunk/compression : **passer les chunks de 8192 à 512
divise le coût du commit par ~10 à taille de fichier identique** sur cette charge (11,32 → 1,05 ms).
Le plancher restant (~0,6–1,0 ms) est le prix du mécanisme HDF5 lui-même : `H5Fflush(GLOBAL)` est
la barrière de visibilité SWMR, il reste nécessaire.

**Ce que cette table ne dit pas.** Le fichier frais ne contient que 400 lignes : le coût de
compression caché dans la taille est donc sous-estimé, et c'est justement là que se joue l'arbitrage
(chunk plus petit ⇒ moins de recompression par commit, moins bon ratio sur les fichiers à millions
de lignes). Avant toute décision de format, il faut la même matrice sur un volume réel (∼10⁵–10⁶
lignes) : temps total d'écriture **et** taille finale. Les chunks étant figés à la création, un
tel choix ne touche que les fichiers neufs.

**Bonus trouvé en instrumentant — un dataspace laissé ouvert par commit (corrigé depuis).** Dans le
bloc `/index` de `flush()`, le premier `filespace` (panzerdb.cpp:1100, HEAD `f9d58ea`) est écrasé par
le second `H5Dget_space` (panzerdb.cpp:1106) sans passer par `H5Sclose` : les cinq autres blocs le
ferment, celui qui écrit le marqueur de commit — donc le seul qui tourne à chaque pas — non.

Comptable, contrairement à ce que j'avais conclu sur le moment : `H5Fget_obj_count` ne suit pas les
spaces (pas de `H5F_OBJ_DATASPACE`) et `H5Inmembers(H5I_DATASPACE, …)` est en effet refusé
(`H5I.c:206 cannot call public function on library type`), mais **`H5Iiterate(H5I_DATASPACE, …)`
marche** et rend le compte exact (20 000 espaces laissés ouverts par erreur = 20 000 comptés). Rien
n'est signalé par la bibliothèque, ni au commit ni à la sortie du processus.

Coût mesuré sur un seul moteur, 5 000 commits d'un signal de 5 valeurs (`.qwen/tmp/idrss.cpp`,
RSS via `/proc/self/statm`, pente entre 1 000 et 5 000 commits) : **0,88 ko/commit avant, 0,07 ko/
commit après**. Le nombre de pas d'écriture n'étant pas borné, la croissance ne l'était pas non
plus. Correctif : un `H5Sclose(filespace)` avant `H5Dset_extent`, comme dans les cinq autres blocs.
Goupille : `test_engine_flush_dataspace_handles` — 200 commits, delta du compteur attendu à 0
(200 avant le correctif, 0 après), plus le chemin de lecture, déjà propre. Indépendant de A3.

### Prolongement : la même question à volume réel, et l'alignement 4 Mo (2026-10-13)

La table ci-dessus venait d'un fichier de 400 lignes et salissait la table de chemins à chaque
commit (un signal neuf par pas), ce que l'AL ne fait pas. Nouveau harnais
(`.qwen/tmp/volume_bench.cpp`) : **S signaux créés une fois, puis T commits qui appendent V valeurs
sur les signaux déjà écrits** — donc seules `index` et `data_raw_*` sont salies à chaque pas.
`T=2000`, `S=20`, `V=60` ⇒ 40 000 lignes d'index, 2,4 M de doubles (19,2 Mo bruts), valeurs
identiques dans toutes les configs (tendance lisse + bruit quantifié, même tirage), donc tailles
comparables ; l'empreinte de lecture (`check=72608414.4`) est la même partout.

Le réglage d'alignement (`H5Pset_alignment(fapl, 4096, 4 Mo)`, posé par `b26c1c8` 2026-03-04 «
Alignment on file system blocks (e.g., Lustre stripe size = 4MB) ») a été rendu temporairement
réglable par variable d'environnement pour la mesure, puis annulé (`git checkout`).

| config | flush médian | écriture totale | lecture complète | taille fichier | blocs alloués |
|---|---|---|---|---|---|
| L1, chunks défaut, **align 4 Mo** (état du code) | 8,65 ms | 17,9 s | 84 ms | 105 109 364 | 14 471 168 |
| L1, chunks défaut, align 64 Ko | 8,64 ms | 17,8 s | 88 ms | **16 963 444** | — |
| L6, chunks défaut, align 64 Ko | 11,29 ms | 23,6 s | 87 ms | 16 895 152 | — |
| aucune compression, chunks défaut, align 64 Ko | 0,43 ms | 0,93 s | 66 ms | 26 411 008 | — |
| L1, index 2048 · f64 32768, align 64 Ko | 2,30 ms | 4,79 s | 83 ms | 18 744 933 | — |
| L1, index 512 · f64 8192, align 64 Ko | 0,63 ms | 1,38 s | 89 ms | 21 347 779 | — |
| L1, index 512 · f64 8192, **align 4 Mo** | **0,62 ms** | 1,45 s | 89 ms | **1 233 173 955** | 15 433 728 |
| L1, index 2048 · f64 32768, **align 4 Mo** | 2,30 ms | 4,87 s | 84 ms | 398 462 982 | 14 663 680 |
| L6, index 512 · f64 8192, **align 4 Mo** | 0,84 ms | 2,02 s | 92 ms | 1 228 979 502 | 14 942 208 |

Quatre choses à en retenir :

1. **L'alignement 4 Mo gonfle l'espace d'adresses, pas les blocs disque** — c'est la correction
 d'une affirmation écrite plus tôt dans cette section, qui parlait de « ×6,2 en taille ». Sur ce
 Lustre, le trou laissé entre deux objets alignés n'est jamais écrit : c'est un **trou** (`du` le
 voit). La dernière colonne l'établit : à alignement 4 Mo, `index 512 · f64 8192` occupe
 **1,23 Go apparents pour 15,4 Mo alloués**, et les chunks défaut **105 Mo apparents pour 14,5 Mo
 alloués** — donc ~14,5 Mo dans toutes les configs, quel que soit l'alignement. Coût réel du
 padding : ce que voit `stat`, `ls -l`, `H5Fget_filesize`, et tout outil non *sparse-aware* (cp,
 tar, rsync, transfert GT1 → 1,23 Go déplacés au lieu de 15 Mo). Attention : sur les fixtures déjà
 en place, le padding lui **est matérialisé** (`magnetics_FAT.h5` : 4 237 197 apparents pour
 4 239 360 alloués, pour 170 355 d'payload) — le trou n'est donc pas garanti selon la façon dont
 le fichier a été écrit. Le mécanisme exact n'est pas établi : l'inflation ne suit pas le nombre de
 chunks (la table « fichier frais » ci-dessus, mêmes chunks de 8192 mais 400 commits, reste à
 2,1 Mo apparents alors que la charge à 2000 commits atteint 1,23 Go), elle semble suivre le nombre
 d'événements d'allocation dans le gestionnaire d'espace du fichier. À caractériser avant d'en
 tirer une règle.
2. **Sans toucher l'alignement, des chunks plus petits divisent le commit par 14** : 8,65 → 0,62 ms,
 écriture totale 17,9 s → 1,45 s, et la lecture reste identique (84 → 89 ms). Le prix n'est pas du
 disque mais de l'espace d'adresses (105 Mo → 1,23 Go apparents). Avec alignement 64 Ko et mêmes
 chunks, la taille apparente retombe à 21,3 Mo pour le même gain de commit (0,63 ms).
3. **L'alignement en lui-même ne coûte rien en temps** : 8,65 vs 8,64 ms de commit et 84 vs 88 ms de
 lecture entre 4 Mo et 64 Ko à chunks défaut. Son seul effet mesuré ici est la taille apparente. Le
 bénéfice supposé (bande Lustre de 4 Mo) **n'est pas mesurable sur cet hôte** : seule une campagne
 sur le stockage cible peut statuer.
4. **La lecture ne se dégrade pas** avec des chunks de 8192 : 88 → 89 ms pour 2,4 M valeurs lues en
 40 000 `readTensor` de 60 éléments (tout tient dans le cache de chunks de 64 Mo). Ce n'est pas
 une mesure de lecture en fenêtre temporelle sur de gros fichiers : à revérifier avant d'en faire
 un argument, mais le coût ici est dominé par le nombre d'appels par feuille, pas par la taille de
 chunk.

**Ce que ces chiffres ne couvrent pas.** Données synthétiques (peu compressibles : sur ces valeurs,
L6 et L1 donnent la même taille à 0,4 % près, alors que `core_profiles_3.h5` compresse ses payloads
d'un facteur ~2,2 — la compression reste donc gagnante en taille sur des données réelles) ; un seul
processus écriveur. Le dépôt est bien sur **Lustre** (`/afs/eufus.eu` → `10.41.0.101@o2ib,…:/efgwfs/efgw`,
type `lustre`, options `flock,lazystatfs,encrypt`) — contrairement au caveat noté ici même au départ.
Conséquence pratique pour mesurer : avec `lazystatfs`, `du` renvoie **512 octets** juste après une
grosse écriture ; il faut relancer (ou `stat -c%s`) pour obtenir les blocs réels. Toute décision sur
chunks/alignement ne touche que les fichiers neufs (figés à la création) et demande une validation
sur le stockage cible.

## 24. `get_slice` en V2 : la base de temps est relue sur toute la plage de `data_raw_f64` à chaque appel — mesuré (2026-10-18)

**Le symptôme** (`rapport_slices.html`) : le coût *par appel* de `getSlice` en V2 croît avec le
nombre de tranches du fichier — 9,1 ms à 100 tranches, 204 ms à 1000 tranches × 1000 points — alors
que `putSlice` reste plat (~3-7 ms). V1 est plat à ~3,5-4,2 ms sur les mêmes fichiers.

**Dispositif.** Harnais de `../IMAS-Cpp` : `examples/benchmark.cpp` + `test_slices.sh`, binaire
`build/examples/example-cpp-benchmark`, `IMAS_AL_DISABLE_VALIDATE=1`. Le harnais compilait
`build/_deps/al-core-src` (FetchContent, `AL_CORE_VERSION=main`) ; ce checkout est à `bc002f5` et son
`src/hdf5/panzerdb.cpp` est **identique octet à octet** à l'arbre local → les chiffres du rapport
mesurent bien le code courant. Harnais depuis repointé sur l'arbre local via
`-DFETCHCONTENT_SOURCE_DIR_AL-CORE=<IMAS-Core>`.

**Fixtures** (dernière configuration du balayage `sp` arrivée à terme, 1000 tranches × 3000 points
× 8 quantités) : `test_slices_sp_v2_slice/core_profiles.h5` md5 `730eb7a3c3d0e8f2cfe59cea5e1774b2`
(17 082 277 o) ; `test_slices_sp_v1_slice/core_profiles.h5` md5 `310275fa968eef7249e8ba7659ceaf17`
(19 622 396 o).

**Base mesurée, même fichier, par appel de `getSlice` :**

| backend | mesure | µs/appel |
|---|---|---|
| V2 | 100 appels, fichier juste écrit | 764 652 |
| V2 | 100 appels, passe suivante (cache chaud) | 756 453 |
| V1 | 100 appels | 3 964 puis 3 759 |

Soit **×195 V2/V1**, et le fait que les deux passes V2 concordent écarte l'effet de page cache.
Rapprochement avec le balayage du jour : à 1000×1000, un appel de tranche coûte 204 122 µs alors
qu'un `get` de l'IDS entier coûte 477 547 µs en une seule fois — une tranche ≈ 43 % d'une lecture
complète.

**`perf record -g` (20 appels)** : `adler32_z` 29,78 % self + `inflate_fast` 26,15 % = **55,9 % en
décompression zlib** ; `__memmove_avx512_unaligned_erms` 3,79 % ; `std::_Hash_bytes` 1,44 % ; le reste
en adresses non résolues de libhdf5 (limite connue de l'outil sur ce host).
**`strace` (3 appels)** : 4914 `pread64` pour 91 714 600 o, soit **29 MB relus par appel** pour un
fichier de 17 MB, en lectures unitaires de ~21 KB (des chunks compressés).

**Compteurs temporaires dans `panzerdb.cpp` (2 appels)** :

```
sliceDirect=24 elems=54 006          → 12 appels/appel, 27 003 éléments/appel : l'hyperslab de la
                                        tranche elle-même est correcte et petite
[PZU] first=time rows=1000 range_elems=26 975 998 sum_counts=1000 span=26 975 998
tensorCalls=6 002 elems=6 002        → 3 001 readTensor d'un seul élément par appel
leavesRebuilds=4                     → 2 par appel
rowsForTime=1 485 032 (742 516/appel) ; availableProbes=563 148 (281 574/appel)
```

**Cause.** `iread_strategy.h:340/352/379/384` résout la base de temps à chaque lecture par
`PanzerDB::getWholeDynamicSignal("time")`, qui assemble toutes les lignes du chemin puis appelle
`readLeavesUnion` — celle-ci fait **un seul `H5Dread` sur la plage englobante** `[min(offset),
max(offset+count))`. Or `putSlice` écrit le `time` en **une ligne d'un élément par pas**, entrelacée
avec les 8 signaux : 1000 lignes dont la plage fait 26 975 998 doubles. On lit donc **215,8 MB pour
récupérer 8 KB**, à chaque appel — d'où la linéarité en nombre de tranches (cette plage vaut ~la
taille totale du payload) et les 56 % de zlib.

**Ce n'est pas corrigeable par le choix de la sélection.** Chunk `data_raw_f64` = 65 536 doubles ;
les lignes de `time` sont espacées de ~24 001 éléments (un pas = 8×3000 points + 1 temps), donc
26 975 998 / 65 536 = **412 chunks, et chaque chunk contient au moins une valeur de temps** : toute
lecture complète de la base de temps décompresse les 412 chunks, en hyperslab contigu ou en
stridé. Le levier est donc de **ne la lire qu'une fois**, pas de la relire mieux. (Relire le point 12
de ce fichier : le passage à `readLeavesUnion` avait supprimé un `H5Dread` par feuille ; sur cette
forme — beaucoup de lignes d'un élément éparpillées — il remplace N petites lectures par une lecture
de toute la plage.)

**Coûts secondaires** : 2 reconstructions du cache de leaves par appel, 742 516 `rowsForTime` et
281 574 sondes de disponibilité dans `nearestAvailableSliceIndex` (dont les boucles de recherche
linéaire sont commentées « few slices expected »).

### Lecture erronée à corriger ici même : la cause dominante n'était pas la lecture du `time` racine

Première lecture, cohérente avec `perf` (56 % de zlib) et avec la trace `[PZU]` : la plage de
26 975 998 éléments lue pour reconstituer `time` semblait être le gouffre. **Faux** : la mémoisation
de cette lecture n'a rendu que 165 ms sur 764. En traçant `readTensor`, les 3 001 lectures d'un seul
élément par appel (que les compteurs annonçaient déjà) apparaissent sur un **autre** chemin :
`PanzerDB::getTimeIndex` (appelé par `slice_read_strategy.cpp:27` à chaque `getSlice`) assemblait la
base de temps du AoS dynamique — `profiles_1d/time`, et non le `time` racine — **avec un `readTensor`
par leaf** : 1000 leaves d'un élément, espacées de 27 003 éléments dans `data_raw_f64`, donc 1000
décompressions de chunk de 512 KB par appel, cache de chunks en thrashing permanent. C'est ça qui
rend le coût par appel proportionnel au nombre de tranches.

### Correctif appliqué (2026-10-18) et mesure avant/après

Deux changements, aucun touchant au format (`HDF5_BACKEND_VERSION` reste 2.0) :

1. **`getWholeDynamicSignal` mémoïse le signal lu** (`whole_signals`, valable tant que
   `leaf_cache_gen` et `index_rows_loaded` n'ont pas bougé) et **garde ce cache avec l'index décodé**
   dans `IndexHandoff`, donc d'un moteur à l'autre de la même session de lecture — le même mécanisme
   que l'item 21 pour `/index`.
2. **`getTimeIndex` passe par cette lecture groupée** au lieu d'assembler la base de temps leaf par
   leaf (une décompression de chunk par pas → une passe sur le dataset, une fois par session).

`getWholeDynamicSignal` devient `const` (elle ne remplit qu'un cache) ; ses membres sont `mutable`.

Coût **par appel** de `getSlice` en V2, même harnais, mêmes fichiers, Q=8 :

| tranches × points | avant | après |
|---|---|---|
| 100 × 100 | 9 098 µs | 8 560 µs |
| 100 × 1000 | 16 507 µs | 7 661 µs |
| 100 × 3000 | 27 947 µs | 8 352 µs |
| 100 × 5000 | 61 629 µs | 9 979 µs |
| 1000 × 100 | 61 374 µs | 52 441 µs |
| 1000 × 1000 | 204 122 µs | 53 012 µs |
| 1000 × 3000 | 764 652 µs | **57 714 µs** (×13,2) |

La dépendance à la taille de tranche a **disparu** : à 1000 tranches l'appel coûte 52-58 ms quel que
soit P (avant : 61 → 764 ms), et à 100 tranches 8,5-10 ms quel que soit P (avant : 9 → 62 ms). Il
reste un facteur ~6 entre 100 et 1000 tranches (8,5 → 52 ms), qui n'est plus de la décompression.

V1 sur le même fichier 1000×3000 : 3 950 µs, inchangé (chemin non touché).
Correctitude : `ctest` **91/91** (dont le sous-ensemble `slice|append|dynamic|time_slice` 35/35),
après rebuild complet de `build` avec le `CPLUS_INCLUDE_PATH`/`LD_LIBRARY_PATH` et l'`IMAS_PREFIX`
(documentés ailleurs) ; `putSlice` et `get` global inchangés sur les configs mesurées.

**Ce qui reste ouvert.** La croissance en nombre de tranches n'a pas disparu : à 1000×100 (payload
de 6,4 MB seulement) l'appel coûte encore 52 ms contre 8,5 ms à 100×100, alors que V1 est plat à
~3,8 ms. Le résiduel n'est plus de la décompression mais du parcours d'index par appel — les
281 574 sondes de `nearestAvailableSliceIndex` et les 2 reconstructions du cache de leaves. C'est le
prochain objet, à traiter séparément.



