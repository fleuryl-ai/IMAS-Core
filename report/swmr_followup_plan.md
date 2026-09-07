# Plan for the remaining SWMR work

**Context:** the branch `swmr-index-last` (pushed to `origin`, upstream set) is in place and is a linear extension of:

```
index_table_v2  @ 6fb6a9e
        └── swmr-fixed-string-col  @ 8db33b8    (SWMR fixed-width + H5Fflush barrier)
                    └── swmr-index-last  @ d154bf8   (index-last commit ordering + reader guardrail)
```

This is the *forward plan* for the SWMR story. It is intentionally staged so that every step is independently
verifiable and reversible. Each step ends in a state that has been proven good enough to ship, and only then
do we move to the next one.

---

## 1. Where we are, and what still needs to happen

| # | Item | Status |
|---|------|--------|
| 1 | `data_raw_str` fixed-width 512 B + `H5Fflush(GLOBAL)` commit barrier | ✅ shipped (`8db33b8`) |
| 2 | Crash-safe **commit ordering** (`/index` written last) + reader guardrail | ✅ shipped (`d154bf8`) |
| 3 | **Deterministic** regression (`test_engine_commit_order_guardrail`) | ✅ 5/5 stable |
| 4 | **In-process-threads** SWMR smoke (`test_swmr_string_fixed_width`) | ⚠️ **flaky** on the AFS host — see §2 |
| 5 | **Live shared-FS (AFS) writer/reader validation** | ❌ not yet run |
| 6 | **Official HDF5 SWMR API** (`H5F_ACC_SWMR_WRITE/READ` + `H5SWMR_data_boundary`) | ❌ not yet adopted |
| 7 | Merge `swmr-index-last` into `index_table_v2` (or into `swmr-fixed-string-col` directly) | ❌ pending 4/5/6 |

Steps **4 → 5 → 6** are the remaining SWMR work. Step **7** happens after 6 is green (or after 5, if 6 is deferred).
The order 4/5/6 is important: 4 tells us the *in-process* test is unreliable; 5 gives us an honest
measurement on a real shared FS; and 6 is the *correct* way to make SWMR work under HDF5's own guarantees,
which 4 and 5 do not. Step 4 should be **fixed or replaced** before step 6 is considered, because
step 4 is *the* regression test we run for this exact feature.

---

## 2. §4 — fix or retire the flaky in-process SWMR smoke test

### 2.a. Why it is flaky (root cause)

The test opens the *same file* from a `std::thread` (reader) while the **main thread** is the writer.
In HDF5, "multiple readers, one writer" is guaranteed by the *process* model, not the *thread* model.
Two openers in the same process share HDF5's per-process file cache (MDC), so their concurrent
`H5Fopen` / `H5Dread` / `H5Fflush` calls race on shared internal state (the "HDF5: infinite loop
closing library" chain is the MDC refusing to close a file it thinks is still in use, from the other
thread's perspective). On some runs it happens to be serialized and the test passes; on others it
segfaults in HDF5's atexit path. This is exactly the failure mode the test was intended to detect —
but it is detecting **HDF5's single-process model**, not a bug in PanzerDB.

**Fix (preferred): replace threads with processes.** A small `fork()` (or, on Windows, `CreateProcess`)
spawns the reader as a *separate process*. The writer is the parent. Communication is "just do it" (the
reader writes exit code 0/1 to stdout, or the reader process exits with code 0 iff every snapshot is
clean). This changes the test from "two threads racing" to "two processes that genuinely see each other
through the shared page cache" — which is what SWMR actually promises.

**Alternative: retire the test** and rely on the 2-process §5 validation. Simpler, but we lose the
in-repo regression for the barrier + fixed-width combo.

Concrete change (2-process version) — sketch:

```cpp
// writer (parent): append N slices, flush after each, exit 0.
// reader (child):  READER_ITERS loops { open fresh (H5F_ACC_RDONLY), read, verify, close };
//                  on any torn read: exit 1.
// parent asserts:  child exited 0 AND the file reads back N slices end-to-end.
```

Rough cost: ~40 lines of C++ (replace the `std::thread reader([&]{ ... });` block with
`pid_t child = fork(); if (child == 0) { <reader block>; _exit(rc); } else { waitpid(child,&st,0); }`).
Windows variant: use `CreateProcess` with a small command-line arg that flips the reader/writer role.

### 2.b. Acceptance criteria

- `ctest --repeat until-pass:5 test_swmr_string_fixed_width` — 5×0 failures across runs.
- The test still asserts **0 torn reads** (each reader snapshot's `slice_<t>` matches).
- No `HDF5: infinite loop closing library` / no segfault in a 100-run loop.

---

## 3. §5 — live shared-FS (AFS) writer/reader validation

### 3.a. What it is

A *different* test (or a small script) that runs writer and reader as **two OS processes on this AFS
host** (which they already are in 4-b — the §2 fix). It must run against a **real file on the AFS
mount** (`/afs/eufus.eu/...`), NOT against `/tmp` on the local disk (which has a different coherence
model). The AFS client cache is the specific concern: POSIX `fsync` / `H5Fflush` don't always invalidate
it, so cross-node visibility can lag. This test catches that.

### 3.b. How to run

Assume the §2 fix is in. The 2-process smoke IS the AFS validation as long as the file path is on the
AFS mount:

```sh
# On the AFS host, from anywhere:
export PANZER_TEST_FILE_PATH=/afs/eufus.eu/g2itmdev/user/g2lfleur/IMAS-Core/build/src/hdf5/test_swmr_string.panzer
cd build/src/hdf5 && ./test_swmr_string_fixed_width
```

Or a shell harness that (a) creates the parent/child, (b) runs 500 iterations with 20 ms writer pacing
and 5 ms reader pacing, (c) asserts `exit 0` each time, (d) prints the "N snapshots × 0 torn reads"
line. A 10-minute soak on AFS is a reasonable acceptance bar.

### 3.c. Acceptance criteria (gate)

- **100% of the runs** finish with reader exit 0 and writer exit 0.
- **0 torn reads** across the entire soak (not per-snapshot, but cumulative).
- The file on disk (post-run, closed) shows `data_raw_str (23, ...) |S512` (h5py) and all 6 slices in
  order.

If §5 fails (torn read on AFS), the fix is likely *not* "add more flushes" but **adopt the official SWMR
API** — that is §6. §6 gives HDF5 its own `H5SWMR_data_boundary` protocol, which is exactly designed for
this FS-coherence issue.

---

## 4. §6 — Phase 2: adopt HDF5's official SWMR API

### 4.a. What changes in the code

`src/hdf5/panzerdb.cpp` (and the two files that call `PanzerDB`'s constructor — `hdf5_writer_v2.cpp`
or wherever the writer is created, and `iread_strategy.h` / wherever readers open the file):

| Site | Change |
|------|--------|
| `PanzerDB` constructor (WRITE mode, fresh file) | Open with `acc_flags \|= H5F_ACC_SWMR_WRITE`. Currently `H5Fcreate(..., H5F_ACC_TRUNC, ...)` — OR in the flag. |
| `PanzerDB` constructor (READ mode) | Open with `acc_flags \|= H5F_ACC_SWMR_READ`. |
| `PanzerDB::flush()` — end of the function | **Replace** `H5Fflush(file_id, H5F_SCOPE_GLOBAL);` with the official commit: `H5O_info_t oinfo; H5Oget_info(file_id, &oinfo, H5O_INFO_LATEST_TRACKED); H5SWMR_data_boundary(index_dset, &oinfo, &maxrow);`. The max row returned is the visible boundary for a reader. |
| `PanzerDB::close()` | No change (H5Fclose is fine in SWMR mode). |
| Readers (in `getLeaves` or the IReadStrategy layer) | Optional: use `H5Oget_info(..., H5O_INFO_LATEST_TRACKED)` before the extent-based guardrail so we know exactly how many rows are visible. This *replaces* the extent check (which was a proxy for "how many rows exist") with the *actual* "how many rows are committed". |
| **Not changed**: `createOptimizedDataset` chunking, alignment (`H5Pset_alignment`), the two-level ordering (index last) and the extent guardrail — both stay, they are complementary to the official API. |

### 4.b. What stays, what is replaced

- **Stays:** fixed-width `data_raw_str` (SWMR requires fixed-width on the write side — we have it), the
  index-last write order, the reader extent guardrail (as a second line of defence).
- **Replaced:** our `H5Fflush(GLOBAL)` "manual commit barrier" → HDF5's `H5SWMR_data_boundary`-
  tracked commit (stronger: it's HDF5's own guarantee of how many rows a reader can safely see, and it
  integrates with HDF5's coherence tracking for the file).

### 4.c. Gating for this step

- §5 (AFS live) must be green **first** — if AFS validation fails on the current (H5Fflush-barrier)
  code, we know §6 is *needed*.
- After adopting §6, **repeat** §5 (AFS live) — and it should now be green.
- All §4 in-repo regression tests (1/2/3 from §1) must still pass — they are mode-agnostic.
- Full `ctest` (68 + any 2-process SWMR test from §2) must pass.

### 4.d. Files to touch (Phase 2 only)

| File | Role |
|------|------|
| `src/hdf5/panzerdb.cpp` | `open()` (flags), `flush()` (commit call), close (unchanged), `getLeaves()` (optional: latest-row instead of extent check). |
| `src/hdf5/panzerdb.h` | Public surface unchanged. If we expose `lastSWMRvisibleRow()` (optional helper for readers), one new member. |
| `src/hdf5/hdf5_writer_v2.cpp` or wherever | If the *writer* is created with `PanzerDB`, no extra change — the flag is inside the PanzerDB ctor. |
| `src/hdf5/iread_strategy.h` or wherever | If the *reader* is created with `PanzerDB`, no extra change — same reason. |
| `tests/hdf5_backend/test_swmr_string_fixed_width.cpp` | (After §2 fix) unchanged in intent; the 2-process variant is what validates §6. |

---

## 5. §7 — merge & close-out

Once §6 is green (or deferred):

1. Re-run full `ctest` (68 + new 2-process SWMR test).
2. Squash or keep the two `PzDB:` commits (8db33b8 + d154bf8 + Phase-2 commits) as a single clean
   sequence against `index_table_v2` (`cherry-pick` or `rebase`).
3. Open a PR against `index_table_v2`. Suggested title:
   **`PzDB: SWMR — fixed-width string col, commit ordering + official SWMR API`**.
4. Once merged, delete the local `swmr-index-last` / `swmr-fixed-string-col` branches; keep
   `index_table_v2` as the integration branch.

---

## 6. Order of next sessions (summary)

| Next session | Deliverable | Acceptance |
|---|---|---|
| S1 | Fix/replace `test_swmr_string_fixed_width` with the 2-process version (per §2). | 5×0 failures across runs; no segfault in a 100-run loop. |
| S2 | Live AFS validation (per §3) — just run the test 500× on the AFS path. | 100% exit 0, 0 torn reads across the entire soak. |
| S3 | Adopt `H5F_ACC_SWMR_WRITE/READ` + `H5SWMR_data_boundary` (per §4). | S2 re-runs green; `ctest` full green; h5py on-disk unchanged. |
| S4 | PR against `index_table_v2` (per §5). | PR merged; local branches cleaned up. |

Each session is independent and can be stopped at any point; the tree is always in a valid state
(the previous commits stand on their own).

---

*Written 2026-09-07 on branch `swmr-index-last @ d154bf8`, `pushed to origin`. This is the plan the
user asked to be written "somewhere" — it lives in `report/swmr_followup_plan.md` alongside the other
SWMR reports.*
