# Reducing file-corruption risk — commit ordering for the PanzerDB index

**Branch:** `swmr-fixed-string-col` @ `8db33b8`
**Date:** 2026-09-07
**Question:** Should we update the `index` **after** writing the data
datasets? What reduces the risk of corruption?

## TL;DR

**Yes.** The `index` is the *commit marker* — the thing that tells a reader
which rows exist. It must be **the last thing written**, and the barrier
(`H5Fflush`) must be **after it**. The current order is the opposite
(index first). Fix: **reorder `flush()` to `paths → data_raw_* → index →
H5Fflush`**, and add a reader guardrail against dangling rows.

---

## 1. What the code does today (verified on this branch)

`PanzerDB::flush()` — `src/hdf5/panzerdb.cpp`:

| # | Line | Statement |
|---|---|---|
| 1 | `:722` / `:731` | `H5Dset_extent(index_dset, …)` → `H5Dwrite(index_dset, …)` — **index first** |
| 2 | `:744` / `:750` | `H5Dset_extent(paths_dset, …)` → `H5Dwrite(paths_dset, …)` — then paths |
| 3 | `:768`+ … | `H5Dset_extent` + `H5Dwrite` for `data_raw_f64 / i32 / str / c128` |
| 4 | `:880` | `H5Fflush(file_id, H5F_SCOPE_GLOBAL)` — commit barrier, last |

All the bytes above are first staged in HDF5's **in-memory file image**; the
`H5Fflush(GLOBAL)` at `:880` pushes the whole image to the OS/disk. So the
*cross-process* visibility is governed by that single barrier — but the
*on-disk ordering* of individual chunks is **index → paths → data**, which is
the dangerous direction for crashes.

Note the content (offsets, counts, path_ids) in `index_buffer` is already
final at `flush()` time (assigned when the node is registered in
`writeData`) — the three pushes are pure disk writes. So reordering them is
safe: it changes only **which bytes hit disk first**, not their values.

## 2. Two distinct failure modes — don't conflate them

| Mode | Trigger | What protects it |
|---|---|---|
| **C1. Concurrent SWMR read** | a second process re-reads while the writer is mid-flush | the **`H5Fflush(GLOBAL)` barrier** — it flushes the whole image at one point |
| **C2. Crash / power loss** | process killed mid-flush, no clean close | the **on-disk order** of the chunks — the thing that needs fixing |

The barrier already handles C1 well. **Reordering is what genuinely reduces
C2 (corruption).** Both are worth doing; they solve different problems.

## 3. The core principle: the index is the commit pointer

Treat the `index` exactly like a database **WAL / sequence counter**. The
rule for a commit marker is: **write the payload first, advance the marker
last.** Because the two asymmetric failure outcomes are:

| Failure | Consequence |
|---|---|
| Payload (paths/data_raw) on disk, **index not advanced** | *Harmless.* The new node is simply not discoverable. The extra slots are unindexed slack. On next `open()`, `updateDiskSizes()` recomputes `disk_size_*` from the on-disk **extent** (`H5Dget_space`), so offset accounting self-heals. The next `writeData` just keeps writing after that extent. No reader ever reaches the orphan slots. |
| **Index advanced**, payload not on disk | *Corruption.* A reader sees a row, follows its `offset`+`count` into `data_raw_*`, and reads beyond the written data → out-of-range / garbage, or an HDF5 error. **This is the bad case.** |

So: **dangling index rows are expensive; orphan payloads are cheap.** Write in
the order that makes the cheap outcome the one a crash produces.

## 4. Concrete change

Reorder `flush()` (content unchanged, only the push order moves):

```cpp
void PanzerDB::flush() {
    if (index_buffer.empty()) return;

    // 1) payload: paths
    if (!paths_buffer.empty())       { /* existing paths write block  (:735-752) */ }

    // 2) payload: data_raw_* (f64, i32, str, c128)
       { /* existing four data write blocks (:759-859) */ }

    // 3) COMMIT MARKER: index — last, so a crash before this point
    //    leaves only unindexed payload (recoverable), never a dangling row
    hsize_t n_new_rows = index_buffer.size() / 14;
    /* existing index extend+write (:713-733), moved here */

    // 4) barrier
    if (file_id >= 0) H5Fflush(file_id, H5F_SCOPE_GLOBAL);

    // 5) clear buffers
    index_buffer.clear(); data_buffer_f64.clear(); /* … */
}
```

Only the *order* of the three blocks changes; the code in each block is
already correct and already present. No new writes, no new API.

## 5. Reader guardrail (defense in depth)

Even with the right order, add a cheap check so a *truly* partial file can
never serve garbage. In the reader's leaf iteration, before using a row:

```cpp
// row r is valid only if its payload exists
hsize_t data_extent = /* H5Dget_space(extent) of the row's data_raw dataset */;
if (row.offset + row.count > data_extent)  continue;   // dangling row → skip
if (path_id[row.path_id] is empty/absent)   continue;   // orphan row → skip
```

This mirrors the "optimistic skip" used by LevelDB / RocksDB / LMDB readers.
It turns a would-be corrupt read into an ignored row (the node appears the
next flush), which is exactly the semantics we want.

## 6. Optional (stronger) — adopt HDF5's official SWMR contract

With `data_raw_str` now fixed-width (the last blocker removed), we can:

- open the writer with `H5F_ACC_SWMR_WRITE` and readers with `H5F_ACC_SWMR_READ`;
- commit via the official API: `H5SWMR_data_boundary` / `H5Oget_info` with
  `H5O_INFO_LATEST_TRACKED`.

This gives HDF5-level guarantees about how many dataset "rows" are safely
visible to a reader, replacing our barrier + guardrail. The reorder in §4 is
still good hygiene (correct crash ordering) and stays valid.

## 7. What NOT to do

- **Don't keep index-first.** That is the corruption direction.
- **Don't add a second `H5Fflush` mid-flush** — one barrier at the end is the
  commit point; extra flushes only cost latency and don't add atomicity.
- **Don't assume `H5Dclose`/`H5Fclose` is enough** in the normal (non-close)
  append path — the whole point of the barrier is visibility *before* close.

## 8. Test plan

1. **Order regression:** existing `test_swmr_string_fixed_width` (writer +
   concurrent reopen-readers) must stay green after the reorder.
2. **Crash-durability fuzz (new, optional):** writer appends N slices; at a
   random point send `SIGKILL`; reopen READ; assert (a) no dangling index row,
   (b) every visible row's `offset+count <= data_* extent`, (c) `getLeaves()`
   has no empty `path`. Run across a few seed points.
3. **Guardrail unit test:** hand-craft a file where `index` has one row whose
   `offset` exceeds the `data_raw` extent; assert the reader skips it and the
   rest round-trips.

---

## Summary

- Current order (`index → paths → data_raw → flush`) risks a **dangling index
  row** on crash → real corruption.
- Reorder to **`paths → data_raw_* → index → H5Fflush`** — the index (commit
  marker) is last; a crash then only leaves **cheap, recoverable orphan
  payload**.
- Add a **reader check** `offset+count <= data extent` so any partial file is
  safely skipped, never served.
- Optionally adopt the **official `H5F_ACC_SWMR_*` API** for a stronger
  (HDF5-guaranteed) commit visibility contract.

*Generated 2026-09-07 from `src/hdf5/panzerdb.cpp` on branch
`swmr-fixed-string-col @ 8db33b8`.*
