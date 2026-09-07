# Are `index` and `paths` appended atomically?

**Question:** When we append one row to the `/index` dataset we also append one
row to the `/paths` dataset. Are these two additions atomic? Can the file become
corrupt — index row present, `paths` row missing? Is there some kind of transaction?

**Short answer:** **No, this is not a transaction in the database sense.** HDF5
provides neither a write-ahead log, nor a two-phase commit, nor a
multi-dataset atomic-commit API. What we have is only an *imposed write order*
plus a *commit barrier* (`H5Fflush`). Two guarantees must not be confused:
ordering, and durability/visibility.

---

## 1. In `flush()`, the write order is: index → paths → data_raw_*

`src/hdf5/panzerdb.cpp` (`PanzerDB::flush()`, lines 719-864):

```
panzerdb.cpp:723   H5Dwrite(index_dset, ...)         // index extended + written (H5Dset_extent :714)
panzerdb.cpp:742   H5Dwrite(paths_dset, ...)         // paths extended + written (H5Dset_extent :736)
panzerdb.cpp:766.. H5Dwrite(data_dset_f64/i32/str/c128, ...)
```

These `H5Dwrite` calls land in **the in-memory file image** (HDF5's internal
buffer), and then in the **OS page cache** — not directly on disk.

## 2. The weak point: the current branch has **no `H5Fflush`**

`panzerdb.cpp:855-856`:

```cpp
// OPTIMIZATION: do not force a disk flush on every call.
// Let the OS and HDF5 coalesce writes in their caches.
```

There is **no `H5Fflush(GLOBAL)`** in `panzerdb.cpp` on the current branch
(`index_table_v2`). So cross-process visibility only happens at
`H5Fclose` (inside `close()`, line 877). This is exactly problem (b) identified
in the SWMR report, and already fixed on the **`swmr-fixed-string-col` branch**
but not here.

---

## Three scenarios: "index is there, but `paths` is not?"

### A. Crash mid-`flush()` (SIGKILL / power loss) → possible in theory

If the process is killed between `H5Dwrite(index_dset, …)` (line 723) and
`H5Dwrite(paths_dset, …)` (line 742), the in-memory image had been partially
pushed into the page cache. The on-disk file may remain inconsistent: an
index row with no matching path string. **But** because nothing is
`H5Fflush`'d, both writes are usually still in memory buffers when the
process dies → the on-disk file has *neither* the index row *nor* the path
(the state of the last real flush). The risk appears mainly if a partial
flush was already pushed.

> **Conclusion A:** an index/paths inconsistency is **possible**, it is just
> not *the common case*. Nothing guarantees against it.

### B. An SWMR reader re-reading during an in-progress `flush()` → the real danger

Because there is no barrier, a "fresh" reader (`H5Fopen`) observes the page
cache state at its own instant `t`. The writer does `H5Dwrite(index)` at
`t1` and `H5Dwrite(paths)` at `t2 > t1`. Between `t1` and `t2`, a concurrent
reader **can** observe the index extended but `paths` not yet → an orphan
row. **This is the scenario we removed by placing
`H5Fflush(file_id, H5F_SCOPE_GLOBAL)` at the END of `flush()`**, after all
writes: that IS the "commit barrier".

### C. HDF5 simply does not offer a transaction → it is a *library gap*, not a bug to fix

What we have is a *commit barrier* (`H5Fflush` at the end of `flush()`), not
an atomic 2PC commit.

---

## The safety net on the reader side

In `getLeaves()` (`panzerdb.cpp:1180-1219`) the reader reads the whole
index, the whole `paths`, and does the join in memory. If it observes an
index row without a corresponding path row:

- the `path_id` points at an empty slot (or at garbage from the chunk);
- the resulting `path` matches no field requested by the client (an IMAS
  field will never match an orphan string);
- **the row becomes invisible** for that request — the corruption is
  therefore *masked* as much as it is *possible*, but *not guaranteed*.

---

## What "plays the role of a transaction"

| Mechanism | Position | Effect |
|---|---|---|
| Order index → paths → data_raw | *inside* `flush()` | Guaranteed in the memory image. **No durability atomicity.** |
| `H5Fflush(GLOBAL)` | **at the end** of `flush()` (`swmr-fixed-string-col`) | Cross-process commit point. The only near-atomic mechanism we have. |
| `H5Fclose` (`close()`) | End of session | Forces an `H5Fflush` if one was not done. |

## What remains theoretically exposed, even after the fix

Even with the barrier, this is **not** a real 2PC. A very aggressive reader
that does `H5Dread(index)` just before `H5Dread(paths)` in the same window
can still observe a gap. What the barrier guarantees is only that the
**writer** stopped at a known point and pushed everything before it. A
well-written reader honours this protocol by:

1. `H5Fopen` in mode `H5F_ACC_RDONLY`;
2. reading the last index row, then `paths`;
3. if the last `row_id`'s path is empty/broken, ignore that row (logical
   rollback).

That is the equivalent of **read-your-writes + optimistic skip** used by most
key-value stores (LevelDB, RocksDB, LMDB) — but without the WAL they have.

---

## Recommendation

To bring the current branch (`index_table_v2`) up to the same level as
`swmr-fixed-string-col`:

1. **Add** `H5Fflush(file_id, H5F_SCOPE_GLOBAL)` at the end of `flush()`
   (~12 lines, already validated by our tests).
2. **Option 2:** adopt `H5F_ACC_SWMR_WRITE/READ` + `H5SWMR_data_boundary`
   for an official HDF5 contract (that is the *correct* mode — but the
   conversion of `data_raw_str` to fixed width is the prerequisite, already
   done on the SWMR branch).
3. Document in the guide that appending an index row is **not atomic** with
   its path; the guarantee is the commit barrier's.

The report `report/swmr_string_fixed_width.md` (sections 1-3) already covers
this in more depth; a dedicated "Is this a transaction?" section can be added
there if useful for the slides.

---

*Answer generated 2026-09-07 from `src/hdf5/panzerdb.cpp` (branch
`index_table_v2`) and cross-checked against `swmr-fixed-string-col`.*
