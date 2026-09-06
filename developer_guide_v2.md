# `imas` HDF5 Backend — PanzerDB Developer Guide

**Version:** v1 (guide)
**Status:** Reference for IMAS-Core HDF5 backend (reader/writer), C++
**Target reader:** Scientist or engineer with a physics background and general C++/HDF5/C familiarity. No prior knowledge of this code base is assumed.

---

## 0. Before you start: the three "engines" of the HDF5 backend

If you are new to this code base, the single most important thing to understand is **which of the three layers you are talking to**. Everything else falls out of that one distinction.

| Layer | What it is | When you'd go there |
|-------|-----------|--------------------|
| **High-level C/C++ API** (`al_*` in `al_lowlevel.h`) | Thin typed C-style wrappers that open a pulse file, start a read/write *action* (global / slice / timerange / arraystructure), and move a data object back and forth. Internally it instantiates a `PanzerDB` and drives it. | Writing *your own* client or driver, or extending existing plugins / backends. This is the "official" public interface of the library. |
| **PanzerDB engine** (`panzerdb.{h,cpp}`) | The columnar + index-table storage engine the HDF5 backend uses. You can open a `.h5` *in your own program* without going through `al_*`. This is the "real code" of the format. | When you need to read/write the raw file from a *script*, *tool*, or *driver* and don't want to pull in the whole `al_*` layer. |
| **Direct Access API** (`direct_access_api.cpp`, `TensorView`) | A thin C++ wrapper over `PanzerDB` that gives you a numpy-like immutable `TensorView` (zero-copy slices, in-memory views) at any path. | When you want a fast, expressive, "read this path and give me a view" interface — for Python, for notebooks, for interactive analysis. |

All three of these sit **on top of the same `PanzerDB` engine**, and all three share the same on-disk format. This guide treats the engine as the source of truth. Once you understand the engine, the other two layers are thin sugar.

---

## 1. Roadmap

This chapter is the whole guide at a glance. Read it first, then pick a section.

### 1.1 Logical map

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          HDF5 PULSE FILE  (*.h5)                         │
│  ┌────────────────┐   ┌────────────────────┐   ┌─────────────────────┐  │
│  │   index(N×14)  │   │   paths(N×256B)     │   │ data_raw_{f64,i32  │  │
│  │   uint64       │   │   fixed-size text   │   │ _c128,str}         │  │
│  └────────────────┘   └────────────────────┘   └─────────────────────┘  │
│  ┌  PanzerDB  engine  (columnar + index table; AO/DO + gap-friendly)   │
│  ┌────────────────────┴──────────────────────────────────────────────┐  │
│  │ Direct Access API → TensorView                                     │  │
│  │ C/C++ al_*  API  (global / slice / timerange / AOS)               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Section-by-section

| § | What you will learn | Key classes / APIs in scope |
|---|---------------------|-----------------------------|
| 2 | How the **engine** is built: columnar raw-data datasets + a *single* index row per "leaf". What a leaf is, what a static vs dynamic AoS is, how a gap lives on disk. | `Leaf`, `PanzerDB::OpenMode`, `panzer_index` |
| 3 | **Core concepts** — timebase & time-slice, AoS/DO, index, path forms (schema vs instance), data type system, open modes, interpolation modes. | `PathComponents`, `DataType`, `InterpMode`, `OpenMode` |
| 4 | **Instantiating / opening PanzerDB** in your own program (both on a path and on an existing HDF5 group/loc). | `PanzerDB(path,mode,...)`, `PanzerDB(loc_id,mode,...)` |
| 5 | **Writing** data: `beginArray` (static & dynamic), `incrementArrayIndex`/`setCurrentArrayIndex`, `endArray`, `writeData`, `writeDataSlices`, `writeMetadata`, `flush`/`close`. Full worked write program. | `PanzerDB::beginArray`, `writeData`, `writeDataSlices`, `writeMetadata` |
| 6 | **Reading** — the meat. Six intents, each with signature / params / return-convention / ≥1 real example: (a) *inspect* the index; (b) *size / shape* of a static or dynamic AoS; (c) *type* of a leaf; (d) *read one value / one tensor / one slice*; (e) *batched reads*; (f) *gaps & interpolation*. | `getLeaves`, `getAOSShape`, `getDynamicAOSSize`, `isDynamicAOS`, `getLeafType`, `readDataByIndex` (+ 3 variants), `readScalar`, `readTensor`, `readSliceDirect`, `readLeavesUnion`, `isTimeInLeaf`, `nearestAvailableSliceIndex`, `getTimeIndex`, `readInterpolatedData`, `getWholeDynamicSignal`, `readMetadata` |
| 7 | **Worked end-to-end examples** of the 5 signature scenarios, with *verified output* on a concrete 23-leaf file. | See each example |
| 8 | **Choosing an interface** — al_* vs Direct-Access vs raw PanzerDB, and when to reach for which. | All three |
| 9 | **Performance & I/O notes** — chunking, cache, batch reads, what the engine does for you already. | `ChunkingConfig`, `configureChunking` |
| 10 | **Error handling & ownership** conventions (return codes, `malloc`-or-`std::vector` ownership, empty leaves). | `ALException`, `ALLowlevelException`, `PanzerDB::Leaf` |
| 11 | **Quick-reference tables**: every method, its signature, and a one-line purpose; full `DataType` / `Kind` / `OpenMode` / `InterpMode` decoding. | — |
| A–E | **Appendices**: A) raw `/index` 14-column layout detail (M1); B) two time models (signal-slice vs path-instance) — side by side; C) path model in full (schema vs instance, `stripIndices`); D) full data-type + flag decoding; E) the full write program used to generate every example in §7. | — |

### 1.3 "If you only want X, go to Y"

| You want… | Where |
|-----------|-------|
| Size of a **static** AoS (§7.1) | §6.2 |
| Size of a **dynamic** AoS (§7.2) | §6.2 |
| A 1-D field in a static AoS — **no** dynamic AoS (§7.3) | §6.4 |
| A slice of a dynamic quantity **at time t** (§7.4) | §6.4 |
| A **whole** dynamic signal (all its times) (§7.5) | §6.6 |
| **Gaps** in a quantity (§7.6) | §6.5 |
| **Interpolate** at a non-existent time (§7.7) | §6.6 |
| **Locate** a time in a timebase (§7.8) | §6.5 |
| **Batch** read across all times (§7.9) | §6.4 |
| **Metadata** attach & fetch (§7.10) | §6.7 |
| The **write** program that built them all | §5, Appendix E |

---

## 2. The engine — how a PanzerDB file is built

### 2.1 The design problem

IMAS plasma data has a *fixed, nested* schema that looks, at first glance, very much like the legacy HDF5 object-graph layout: an AoS `profiles_1d` that has an AoS `ion` (one per species) that itself has a leaf `temperature` (a 1-D array in space). And the whole thing is often *dynamic* in time: every time step the entire sub-tree is re-instantiated.

Writing that shape as a *nested* HDF5 object-graph is what the legacy backend did. The new, "compact-index" backend ("PanzerDB") makes a very deliberate architectural choice:

> **Do not store the tree as a tree. Store it as a table.**
> Every logical node ("leaf") becomes **one row** of a single, flat `index` dataset, and all its payload — whatever numeric type — is appended to a *single, columnar* `data_raw_<type>` dataset of the right type.

That single decision buys, for free:

* **Append-friendly I/O**: writers just `append + index-row` and never touch existing HDF5 blocks.
* **Columnar I/O**: all `double`s in one chunked dataset, all `int32`s in another, all `complex128` in a third; readers pull out *only* the slice they need, with a plain HDF5 `hyperslab`.
* **Gap-friendly**: a missing time step is just *the leaf is not there in the index*; nothing has to be pre-allocated.
* **Zero tree-traversal on disk**: lookups are one `index` row + one `data_raw_*` hyperslab.

### 2.2 On-disk layout (what a PanzerDB file always looks like)

```
<instance group or file root>
├── index            : N × 14 × uint64        (N = number of "leaves")
├── paths            : N × 256B (fixed C1)     (N = same N)
├── data_raw_f64     : 1-D float64, chunked
├── data_raw_i32     : 1-D int32,  chunked
├── data_raw_c128    : 1-D (8B × 2), chunked   (complex128)
└── data_raw_str     : 1-D variable-length UTF-8, chunked
```

Verified on the working example file (23 leaves) of §7:

```
dataset  data_raw_c128   shape=(0,)   dtype=('<f8',(2,))
dataset  data_raw_f64    shape=(57,)  dtype=float64
dataset  data_raw_i32    shape=(1,)   dtype=int32
dataset  data_raw_str    shape=(1,)   dtype=object
dataset  index           shape=(23, 14)  dtype=uint64
dataset  paths           shape=(23,) dtype=|S256
```

**Nothing else is in the file.** Every other "object" (an AoS, a dynamic signal, a metadata entry) is one row of `index` + the corresponding slice of `data_raw_*`. That's the whole file: a small fixed-size index table on top of four chunked columnar datasets.

### 2.3 What a "leaf" is

A *leaf* is the unit of addressing. Every node that PanzerDB knows about is a leaf:

* a numeric **scalar** or **tensor** at some path (the most common case);
* the **Ao/DO meta record** (static or dynamic AoS) — it occupies its own row, has a `size` field, and has no payload;
* a **metadata entry** (one string at every path of the form `<path>@<key>`).

A leaf's in-memory representation is `Leaf`, and it is the *only* struct the public API uses:

```cpp
struct Leaf {
    std::vector<size_t> shape;     // Dimensions of one time-slice (empty for a scalar or AoS meta).
    uint64_t time_index = 0;       // First time-step in this leaf's payload.
    std::string_view path;         // Instance path, e.g. "profiles_1d/0/ion/1/temperature".
    std::string_view parent_path;  // Immediate parent node.
    uint64_t offset = 0;           // Starting element in its data_raw_* dataset.
    uint64_t count = 0;            // Number of stored elements (slice_volume × n_steps).
    uint64_t flags = 0;            // Low 4 bits: kind; upper bits: data-type.
    bool is_empty = false;         // Convenience mirror of (flags & 0xF) == 1.
};
```

The **row-14 × uint64 encoding** of a leaf in `index` is an implementation detail you can inspect in Appendix A.

### 2.4 The two kinds of AoS — the single most important concept

| | **Static AoS** | **Dynamic AoS** |
|---|---|---|
| Size | Known at creation, stored in the leaf's `shape[0]`. | Grows over time; the "size" is the number of time steps written, discoverable only at read time. |
| `flags` | `kind == 2` | `kind == 3` |
| Example | `ion` (list of species in a profile), `flux_loop` (list of channels), `channel` | `profiles_1d`, `diagnostics` (a diagnostic that may or may not have been filled at a given time) |
| How it's stored | One meta-leaf per instance, e.g. `profiles_1d/0/ion`; its `shape[0] = n_species`. | One meta-leaf for the dynamic AoS itself (e.g. `profiles_1d`); its "children" are a *sequence of leaves*, one per time step (or a set of leaves that all share a `time_index`). |

The engine enforces — and the user guide relies on — a *single* dynamic AoS per nesting chain. Multiple independent dynamic AoSes at different branches are of course fine (e.g. `profiles_1d` *and* `diagnostics` can both be dynamic, as long as neither is *nested inside the other*).

### 2.5 The two time models (how time actually lives on disk)

Once you've accepted that everything's *one row per leaf*, time can be expressed in **two** different, co-distinguishable ways, both present in a single file. This is the single most subtle concept of the format, so we give it its own mini-section.

**Model A — "slice-index model" (a time series as one leaf).**
A single, *root-level* quantity like `vessel_power` is written with `writeDataSlices(name, base_shape, data, n_slices, "")`. PanzerDB stores it as **one leaf per write batch** — all `n_slices` live in that leaf's payload, at consecutive offsets. Reading slice `t` is just a hyperslab at `offset + t * slice_volume`.

If the writer skips one time-step (i.e. writes `n=2`, then `n=1` two "steps" later), you get **two leaves with different `time_index` values** — and the gap is just *the leaf you never wrote*. That is what a gap *is*, on disk.

**Model B — "path-instance model" (time as a path component).**
If the same quantity lives *inside a dynamic AoS*, e.g. `profiles_1d/ion/temperature`, each time step is represented by a *separate instance* of the dynamic AoS — the path literally carries the slice index. So each time step is a **distinct full path**, not a repeated name: `profiles_1d/0/ion/0/temperature`, `profiles_1d/1/ion/0/temperature`, `profiles_1d/2/ion/0/temperature` are three different index rows. The "time" is thus a *path component*, not a payload dimension. (Such a row *also* carries a `time_index` column; in this layout the instance index and the `time_index` agree, but you read it by *path*, using the `time_index` only to pick the slice within that instance.)

Both models are read through the **same** public API (`readDataByIndex`, `readSliceDirect`, `readScalar`, …) — the engine transparently handles which model a path points at. §6.5 explains the reading differences; Appendix B has a side-by-side comparison of the on-disk rows for both.

### 2.6 The M1 "no `/parent_paths`" design

The legacy (M0) encoding stored *both* the full instance path and the parent path per leaf, which is O(depth × path_length) redundant text. The M1 encoding stores the full instance path (fixed 256B in `paths`) *only*, plus a `parent_id` (uint64) that is a *row number* into the same index table. `parent_path` is reconstructed on read from that row number (via the known `kind` and the `flags`). This is the *byte-identical* contract we expose to readers — the `Leaf::parent_path` field is always correct as if it had been stored — but it means a leaf no longer carries duplicated path bytes.

This matters for **SWMR** (see `project/compact-index-swmr.md`) because the reader never *writes* to the index table; the writer only appends new rows, and the reader only re-scans — so the fixed-width, append-only row layout is the binding constraint for concurrent read-only access.

---

## 3. Core concepts, terminology, and enumerations

### 3.1 Static vs. dynamic AOS, with concrete examples

```cpp
// STATIC AoS: size is fixed for the life of the file.
db.beginArray("flux_loop", 3);          // 3 channels; 0,1,2
  db.writeData("data", {5}, buf, 5);    // flux_loop/0/data — a 5-element vector
  db.incrementArrayIndex();
  db.writeData("data", {5}, buf, 5);    // flux_loop/1/data
  ...
db.endArray();

// DYNAMIC AoS: has a timebase; size = number of time-slices written.
db.beginArray("profiles_1d", "time");   // the timebase is the path "time"
  // time-step t = 0
  db.beginArray("ion", 2);
    db.writeData("temperature", {5}, buf, 5);   // profiles_1d/0/ion/0/temperature
    db.incrementArrayIndex();
    db.writeData("temperature", {5}, buf, 5);   // profiles_1d/0/ion/1/temperature
  db.endArray();
  db.incrementArrayIndex();                       // advance to t = 1
  ...
db.endArray();
```

Note the *two separate* `incrementArrayIndex()` calls — one on `ion` (static), one on `profiles_1d` (dynamic). The engine advances the *current* element on the stack top of the array of open AOS; the "whose" is implicit in the order of `beginArray`/`endArray`.

### 3.2 Path forms — schema vs. instance

A leaf always carries an **instance path** — the fully-qualified, index-baked name of a concrete leaf. When reading you can hand `PanzerDB` one of:

* the exact instance path (`profiles_1d/1/ion/0/temperature`);
* a *schema* path where you've stripped a specific instance and want "all of them" (`profiles_1d/ion/temperature`) — see `readLeavesUnion`;
* a *time-bounded* expression (e.g. `profiles_1d[time=1.5]/ion/0/temperature`) used only by the **high-level** AL API.

The conversion between them is `PanzerDB::stripIndices(path)`, which removes *every purely-numeric path component*. It's `static` — you can call it on a literal string from anywhere.

### 3.3 Data types

PanzerDB supports four *storage* types and a *meta* type (for AoS) plus one *kind* for empty-preserved nodes:

```cpp
enum class DataType : uint64_t {
    FLOAT64      = 0,   // double
    INT32        = 1,   // int32_t
    COMPLEX128   = 2,   // std::complex<double>
    STRING       = 3,   // single / multi-element strings
    LIST_OF_STRINGS = 4, // convenience: multi-element STRING
    UNKNOWN      = 99
};
```

The *kind* (low 4 bits of `Leaf::flags`) is **separate** from the `DataType`:

| `flags & 0xF` | Meaning |
|---------------|---------|
| 0 | Regular data leaf (payload stored in `data_raw_*`). |
| 1 | *Preserved empty* leaf — a node that should exist in the index but has no data yet. |
| 2 | **Static AoS** meta leaf. |
| 3 | **Dynamic AoS** meta leaf. |

`Leaf::is_empty` is a convenience boolean for "kind==1."

### 3.4 Open modes

```cpp
enum class OpenMode { WRITE, READ, APPEND };
```

* **`WRITE`** — create a new file (truncating if it exists).
* **`READ`** — open an existing file read-only. This is the mode you'll be in 95 % of your read code, because *opening* in `READ` gives you **all** the read methods and *none* of the write methods are usable.
* **`APPEND`** — read *and* append. Use this when you're extending an existing pulse file with more time-slices.

### 3.5 Interpolation modes

These are the modes you'll pass to `readInterpolatedData`, `getTimeIndex`, `nearestAvailableSliceIndex`, and friends. They come from the shared `al_const.h`:

```cpp
alconst::closest_interp      // "closest in time"
alconst::previous_interp     // "last available <= requested"
alconst::linear_interp       // "linear interp between bracketing"
```

(There is also an *undefined*/"pass-through" mode for write calls; see `al_const.h`.)

---

## 4. Instantiating / opening PanzerDB in your own program

### 4.1 The two constructors

```cpp
// (1) A PanzerDB root at a filesystem path.
PanzerDB(const std::string& filename, OpenMode mode, bool preserve_empty = false);

// (2) A PanzerDB root *inside an existing HDF5 group* (the "loc_id" form).
PanzerDB(hid_t loc_id, OpenMode mode,
         bool preserve_empty = false,
         bool close_loc_id_on_exit = false);
```

* (1) is the path the AL backend uses for *top-level* instances. (2) is the path it uses when the instance lives *inside* a larger file (an IDS group). Either form yields an identical on-disk layout — only the *parent group* differs.
* **`preserve_empty = true`** tells PanzerDB to emit a *preserved-empty leaf* (`kind == 1`) for every AoS node you `beginArray` but *never* `writeData` into. The default is `false` (empty AoS meta records are dropped). Use `true` if your reader relies on the *shape* of the empty AoS being present.
* **`close_loc_id_on_exit = true`** (form 2 only) makes the constructor take *ownership* of `loc_id`: the destructor will `H5Gclose(loc_id)`. Leave it `false` if you opened the group and plan to close it yourself.

Both constructors are **non-copyable** — the class holds HDF5 identifiers, and a copy would double-close them. Move-semantics are not advertised either; pass `PanzerDB&` (or `PanzerDB*`) around.

### 4.2 A minimal, complete reader — "hello index"

```cpp
#include <iostream>
#include "panzerdb.h"          // from <PROJ>/src/hdf5
#include "al_const.h"          // for alconst::double_data etc. (only where needed)

int main(const char* path) {
    // READ: open an existing file (this is the mode you'll be in 95 % of the time).
    PanzerDB db(path, PanzerDB::OpenMode::READ /*, false*/);

    // "Inspection" intent (see §6.1): the whole index, as std::vector<Leaf>.
    const std::vector<PanzerDB::Leaf>& leaves = db.getLeaves();
    std::cout << "file has " << leaves.size() << " index rows\n";
    for (const auto& lf : leaves) {
        std::cout << "  [" << lf.path
                  << "]  kind=" << (lf.flags & 0xFULL)
                  << "  time=" << lf.time_index
                  << "  shape=" << lf.shape.size() << " dim(s)"
                  << "  count=" << lf.count << "\n";
    }
    return 0;
}
```

That's it. `getLeaves()` is the *single* read entry point that every other read method feeds off, and it is *cached* for the life of the object — you pay the cost of loading and indexing the table **once**, and every `readDataByIndex` / `readSliceDirect` / `getAOSShape` call afterwards is a plain hash-map lookup + a small `data_raw_*` hyperslab.

### 4.3 Open modes, in one line each

* **`WRITE`** — create a fresh file, truncating any existing one. All write + read methods are usable. (In practice: the AL backend uses `WRITE`/`APPEND`.)
* **`READ`** — read-only. All read methods usable; every write method will throw or silently no-op.
* **`APPEND`** — read + *append-only* writes. Use when you're adding more time-slices to an existing file. This is the mode a solver runs in after a "first write."

`panzer.getOpenMode()` returns which of these you are in.

### 4.4 Configuration (chunking)

The engine ships with a chunking/compression profile and four usage-hints. You can override *before* opening the file in `WRITE` mode:

```cpp
PanzerDB db("out.h5", PanzerDB::OpenMode::WRITE, true);
db.setChunkingHint("array_of_structures");  // or "time_series", "bulk_write", "interactive"
db.setCompressionLevel(6);                   // gzip 0–9
// db.disableCompression();                  // alternative
```

In `READ`/`APPEND` the chunking is *restored from the file*; `configureChunking` is a no-op on those modes. `getChunkingStats()` / `printChunkingStats()` are read-only diagnostics (cumulative chunk I/O counters for the life of the object).

---

## 5. The Write API

All write methods are **buffered in RAM** and physically flushed either by an explicit `flush()` or on `close()`/destruction. This is why a write session doesn't show partial state to another reader until `flush()` — a property PanzerDB deliberately exploits in the *SWMR* story (writer appends to a RAM buffer; reader polls `index`; the reader never needs to see a half-written row).

### 5.1 Array-of-Structures (AoS) lifecycle

| Method | What it does |
|--------|-------------|
| `beginArray(const std::string& name, size_t size)` | Opens a **static** AoS of `size` elements. Emits one `kind==2` meta-leaf. |
| `beginArray(const std::string& name, const std::string& timebase)` | Opens a **dynamic** AoS whose timebase is `timebase` (a *path*, usually `"time"`). Emits one `kind==3` meta-leaf. |
| `beginArray(ArrayLevel& level)` | "Manual" form: you construct an `ArrayLevel` and push it. For advanced clients who want to set `container_row_id`, `declared_size`, `current_index` by hand. |
| `incrementArrayIndex()` | Advance the *current element* on the AoS stack top from `i` to `i+1`. This is how you move to the next element of a static AoS *and* how you advance to the next time-slice of a dynamic AoS. It's a *no-op* (silent) when no AoS is open. |
| `setCurrentArrayIndex(size_t new_index)` | Jump (instead of step) to a specific element on the top. Rarely needed. |
| `endArray()` | Close the AoS whose `beginArray` is *still open*, pop it off the stack. Must be called exactly once per `beginArray`. |
| `getArrayStackSize()` / `isArrayStackEmpty()` | Introspection on the stack. |
| `synchronizeArrayStack(aos_names, indices)` | Rebuild the internal AoS stack to match an authoritative list of names + indices (used by the AL backend to reconcile its own context state). Not normally needed from raw PanzerDB code. |
| `isInsideDynamicAOS(std::string* timebase)` | Introspection: are we currently inside a dynamic AoS? |

#### The `beginArray` idiom you'll always use

```
beginArray(name, size);              // static
beginArray(name, timebase_name);     // dynamic
  …write leaves…
  incrementArrayIndex();             // advance to i+1 / t+1
  …write leaves…
  incrementArrayIndex();
  …
endArray();
```

The *order* of `beginArray` and `endArray` determines the nesting. The engine keeps them on a real stack (`array_stack` — see `panzerdb.h`), and a common bug is to call `endArray` when the stack top isn't the AoS you intended to close. `incrementArrayIndex` always operates on the *top* of that stack — which is why the order above matters.

### 5.2 Data leaves

| Method (template) | Signature (double form) | Purpose |
|----|----|----|
| `writeData<T>` | `writeData(const std::string& name, const std::vector<size_t>& shape, const T* data, size_t count, const std::string& timebase = "")` | Write a **static** tensor (or scalar). `shape` = per-slice shape (e.g. `{5}` for a 5-vector, `{}` for a scalar). `timebase` *must* be empty for a static write — a non-empty `timebase` throws. |
| `writeDataSlices<T>` | `writeDataSlices(const std::string& name, const std::vector<size_t>& base_shape, const T* data, size_t n_slices, const std::string& timebase)` | Write **`n_slices` consecutive time-slices** of a dynamic quantity, contiguously. `base_shape` = shape of a *single* slice. Available for `double`, `int32_t`, `std::complex<double>`, and string lists. |

`<T>` is one of `double`, `int32_t`, `std::complex<double>`, or `const char*` (strings). The engine appends the payload to the matching `data_raw_*` buffer and emits *one* `index` row (static) or `n_slices` rows (dynamic — see the "two time models" discussion in §2.5).

**Ownership is clear**: the `data` pointer is *read only*; the engine copies. You can `free` or `delete[]` your array the instant the call returns.

### 5.3 Metadata

```cpp
db.writeMetadata("profiles_1d/ion/temperature@units", "eV");   // schema path + "@key"
```

Metadata is stored as a **string leaf** at a *schema-path* + `"@" + key`. `readMetadata(instance_path)` (see §6.9) *strips the instance indices* from the leaf you're asking about and pulls in every `@…` sibling.

### 5.4 Terminal / diagnostics

| Method | Notes |
|--------|-------|
| `flush()` | Move *all* buffered index + data rows to disk. **Call it before another process might read the file** — PanzerDB has no built-in inter-process visibility signal. |
| `close()` | `flush()` **and** close all HDF5 identifiers. Idempotent — the destructor calls it, so explicit `close()` is optional (but nice for clarity in long-lived clients). |
| `dumpLeafIndex()` | Debug: prints every row of the in-memory index. |

### 5.5 The full write program we'll use in §7

This is the exact program that produces the file every read example in this guide reads back. Save it, link it (see Appendix F for the exact build command), and it's runnable.

```cpp
#include <iostream>
#include <vector>
#include "panzerdb.h"

int main(const char* path) {
    const std::vector<size_t> five = {5};

    PanzerDB db(path, PanzerDB::OpenMode::WRITE, /*preserve_empty=*/true);

    // ── root scalars ─────────────────────────────────────────────────
    double mn = 21.0;  db.writeData("machine_number", {}, &mn, 1);
    int32_t vi = 9;    db.writeData("version",        {}, &vi, 1);

    // ── static AoS:  flux_loop[3] > data[5]  ─────────────────────────
    db.beginArray("flux_loop", 3);
    for (int i = 0; i < 3; ++i) {
        double d[5]{0};
        for (int k = 0; k < 5; ++k) d[k] = 100.0 * i + k;
        db.writeData("data", {5}, d, 5);
        if (i < 2) db.incrementArrayIndex();
    }
    db.endArray();

    // ── dynamic AoS:  profiles_1d[t=0,1,2] > ion[2] > temperature[5]  ─
    auto T = [](int t, int j) {
        std::vector<double> v(5);
        for (int k = 0; k < 5; ++k) v[k] = 1000.0*t + 10.0*j + k;
        return v;
    };
    db.beginArray("profiles_1d", /*timebase=*/"time");       // enter (t=0)
    for (int t = 0; t < 3; ++t) {
        if (t > 0) db.incrementArrayIndex();                 // dynamic advance → t
        db.beginArray("ion", 2);
        { auto v0 = T(t, 0); db.writeDataSlices("temperature", five, v0.data(), 1, ""); }
        db.incrementArrayIndex();                             // ion j=0 → j=1
        { auto v1 = T(t, 1); db.writeDataSlices("temperature", five, v1.data(), 1, ""); }
        db.incrementArrayIndex();                             // ion sentinel
        db.endArray();                                        // pop ion
    }
    db.incrementArrayIndex();                                 // profiles_1d sentinel
    db.endArray();                                            // pop profiles_1d

    // ── dynamic AoS with a GAP:  tgap[t=0,1,3] > x (t=2 skipped)  ───
    db.beginArray("tgap", "time");
    double x0 = 100.0; db.writeDataSlices("x", {}, &x0, 1, "");   // t=0
    db.incrementArrayIndex();                                     // t=1
    double x1 = 200.0; db.writeDataSlices("x", {}, &x1, 1, "");   // t=1
    db.incrementArrayIndex();                                     // t=2  ← skipped
    db.incrementArrayIndex();                                     // t=3
    double x3 = 400.0; db.writeDataSlices("x", {}, &x3, 1, "");   // t=3
    db.incrementArrayIndex();
    db.endArray();

    // ── root dynamic signal + its timebase (for interpolation demo)  ─
    double P[4] = {100.0, 200.0, 150.0, 300.0};
    db.writeDataSlices("power", {1}, P, 4, "");
    double TB[4] = {0.0, 1.0, 2.0, 3.0};
    db.writeDataSlices("time",  {1}, TB, 4, "");

    // ── metadata ─────────────────────────────────────────────────────
    db.writeMetadata("profiles_1d/ion/temperature@units", "eV");

    db.close();
    return 0;
}
```

After running this, the file has **23 leaves** and the on-disk datasets look like (verified):

```
dataset  data_raw_c128   shape=(0,)   dtype=('<f8',(2,))
dataset  data_raw_f64    shape=(57,)  dtype=float64
dataset  data_raw_i32    shape=(1,)   dtype=int32
dataset  data_raw_str    shape=(1,)   dtype=object
dataset  index           shape=(23, 14)  dtype=uint64
dataset  paths           shape=(23,) dtype=|S256
```

---

## 6. The Read API — the actual meat

This is the section you'll live in. All read methods share one contract:

* **Open in `READ` mode.** Every method below assumes `PanzerDB db("file.h5", PanzerDB::OpenMode::READ)`.
* **Return convention.** Methods that *allocate* for you return an `int` status: **`0` on success, `-1` on "not found / gap / error."** On success they hand you a pointer you **own** (`free` it when done) for the C-style APIs. Templated methods (`readTensor<T>`, `readSliceDirect<T>`, `readScalar<T>`) hand you `std::vector`-sized buffers you fill yourself.
* **`time_index` convention.** In any `*ByIndex` call, **`time_index == -1`** means "static, or give me the aggregation over all available time steps"; any **`time_index >= 0`** means "give me the data at that specific time-step."
* Every method that needs an index does its lookup in the *cached* table from `getLeaves()`, then does a single (usually tiny) `data_raw_*` hyperslab. So **open once, call as many times as you like.**

### 6.1 Inspect the index — `getLeaves()`

```cpp
const std::vector<PanzerDB::Leaf>& getLeaves() const;
```

* **Returns:** a *const reference* to the cached `std::vector<Leaf>` (see §2.3 for `Leaf`). **Do not copy it** for iteration — use a range-for over the reference.
* **When to call:** it's cheap to call repeatedly (returns the cached vector), but call it *at least once* before you need any `Leaf` pointer, because every other read method either takes a `Leaf&` or looks paths up in the same internal map.

**Verified output on the §7 file (23 rows):**

| # | `path` | kind | time | count | shape | type |
|---|--------|------|------|-------|-------|------|
| 0 | `machine_number` | 0 (data) | 0 | 1 | `{}` | f64 |
| 1 | `version` | 0 (data) | 0 | 1 | `{}` | i32 |
| 2 | `flux_loop` | **2 (static AoS)** | – | – | `{3}` | – |
| 3 | `flux_loop/0/data` | 0 | 0 | 5 | `{5}` | f64 |
| 4 | `flux_loop/1/data` | 0 | 0 | 5 | `{5}` | f64 |
| 5 | `flux_loop/2/data` | 0 | 0 | 5 | `{5}` | f64 |
| 6 | `profiles_1d` | **3 (dynamic AoS)** | – | – | – | – |
| 7 | `profiles_1d/0/ion` | **2 (static AoS)** | – | – | `{2}` | – |
| 8 | `profiles_1d/0/ion/0/temperature` | 0 | **0** | 5 | `{5}` | f64 |
| 9 | `profiles_1d/0/ion/1/temperature` | 0 | **0** | 5 | `{5}` | f64 |
| 10 | `profiles_1d/1/ion` | **2 (static AoS)** | – | – | `{2}` | – |
| 11 | `profiles_1d/1/ion/0/temperature` | 0 | **1** | 5 | `{5}` | f64 |
| 12 | `profiles_1d/1/ion/1/temperature` | 0 | **1** | 5 | `{5}` | f64 |
| 13 | `profiles_1d/2/ion` | **2 (static AoS)** | – | – | `{2}` | – |
| 14 | `profiles_1d/2/ion/0/temperature` | 0 | **2** | 5 | `{5}` | f64 |
| 15 | `profiles_1d/2/ion/1/temperature` | 0 | **2** | 5 | `{5}` | f64 |
| 16 | `tgap` | **3 (dynamic AoS)** | – | – | – | – |
| 17 | `tgap/x` | 0 | **0** | 1 | `{}` | f64 |
| 18 | `tgap/x` | 0 | **1** | 1 | `{}` | f64 |
| 19 | `tgap/x` | 0 | **3** | 1 | `{}` | f64  ← *gap at t=2* |
| 20 | `power` | 0 | 0 | 4 | `{1}` | f64 |
| 21 | `time` | 0 | 0 | 4 | `{1}` | f64 |
| 22 | `profiles_1d/ion/temperature@units` | 0 (meta) | 0 | 1 | `{}` | str |

Look at three things in that table — they are the whole model in miniature:

1. **`flux_loop`** (row 2) is a `kind==2` leaf with `shape={3}` — the *size of a static AoS is stored in its row's shape*. Its three `data` children (rows 3–5) are *separate leaves*, one per instance.
2. **`profiles_1d`** (row 6) is `kind==3` with *no shape* — a dynamic AoS has no fixed size. Its instances (`profiles_1d/0/…`, `profiles_1d/1/…`, `profiles_1d/2/…`) are *distinct leaves* whose `time` column is `0`, `1`, `2`. **That is the path-instance model of §2.5.**
3. **`tgap/x`** (rows 17–19) is the *same path* appearing three times with `time` = `0, 1, 3` — t=2 was never written. **A gap is simply the missing row.** That is the slice-index model of §2.5.

### 6.2 Size & shape of an AoS

#### `getAOSShape` — the workhorse

```cpp
std::vector<size_t> getAOSShape(const std::string& level_name) const;
```

* **`level_name`** — the path of the AoS *meta-leaf* (e.g. `"flux_loop"`, `"profiles_1d/0/ion"`).
* **Returns:** a `std::vector<size_t>` with a **single element: the size**. For a static AoS it's the declared element count; for a dynamic AoS it's the **number of time-slices actually written**. An empty vector (`size() == 0`) means the path was not found — *always check `!v.empty()` before indexing*.

**Verified:**

```cpp
PanzerDB db("demo.h5", PanzerDB::OpenMode::READ);

auto s_static  = db.getAOSShape("flux_loop");         // {3}  -> 3 channels  (static AoS)
auto s_nested  = db.getAOSShape("profiles_1d/0/ion");// {2}  -> 2 species   (static AoS, nested inside a dynamic one)
auto s_dynamic = db.getAOSShape("profiles_1d");       // {3}  -> 3 time-slaps (dynamic AoS)
auto s_bad     = db.getAOSShape("no_such_aos");       // {}   -> NOT FOUND (empty vector!)

std::cout << "flux_loop = "     << (s_static.size()  ? s_static[0]  : -1) << "\n";
std::cout << "profiles_1d = "   << (s_dynamic.size() ? s_dynamic[0] : -1) << "\n";
```

```
flux_loop   = 3
profiles_1d = 3
```

> **Gotcha (costs people an afternoon):** `getAOSShape` returns an *empty vector*, not `0` and not `-1`, for an unknown path. `v[0]` on an empty vector is **undefined behaviour** (it segfaulted in our test harness until we added the `.size()` guard). Always `v.empty() ? fallback : v[0]`.

#### `getDynamicAOSSize` — the "how many slices" fast path

```cpp
size_t getDynamicAOSSize(const std::string& aos_path) const;
```

Returns the number of time-slices written for a *dynamic* AoS, or `0` if the path is unknown. This is the *counter* form — it's the number the *writer* tracked, so it's correct even mid-stream. For the §7 file: `getDynamicAOSSize("profiles_1d") == 3`.

#### `isDynamicAOS` — the one-line classifier

```cpp
bool isDynamicAOS(const std::string& aos_path) const;
```

Returns `true` iff `aos_path` is a `kind==3` meta-leaf. Use it to *branch* between "static AoS" handling (fixed children indices) and "dynamic AoS" handling (time-slices). Verified on §7:

```cpp
db.isDynamicAOS("profiles_1d");   // true
db.isDynamicAOS("flux_loop");     // false
```

**Rule of thumb:** whenever a path might be either, *ask*. Don't assume. The size query tells you how many, *but only after* you know which *kind* you're counting.

### 6.3 Type of a leaf — `getLeafType`

```cpp
imas::direct_access::DataType getLeafType(const std::string& path);   // THROWS if not found
```

* **`path`** — the *full instance path* of the data leaf.
* **Returns:** the public `DataType` (`DOUBLE`, `INT32`, `COMPLEX_DOUBLE`, `STRING`, `LIST_OF_STRINGS`, …). **Throws** `ALException` if the path is not in the index — unlike `getAOSShape` (empty vector), a type query is *binary*: found or exception.
* **Use for:** picking the right `read*` variant / template parameter *before* you allocate. It decodes the `flags >> 4` bits for you.

**Verified:**

```cpp
db.getLeafType("flux_loop/0/data");                // DataType::DOUBLE
db.getLeafType("profiles_1d/0/ion/0/temperature"); // DataType::DOUBLE
// db.getLeafType("profiles_1d/0/ion/0/nope");     // THROWS ALException
```

> Internally `flags >> 4` is an *internal* storage enum (`FLOAT64 / INT32 / COMPLEX128 / STRING / LIST_OF_STRINGS`); `getLeafType` maps it onto the public `imas::direct_access::DataType`. Use the **public** one in your code.

### 6.4 Reading one value, one tensor, or one slice

This is the 80 % case. There are **five** overlapping ways; pick by "does the caller know the `Leaf`?" and "does the caller want a specific time step?"

#### (a) `readDataByIndex` — path + time, returns a buffer **you own**

```cpp
int  readDataByIndex(const char* full_data_path, int64_t time_index,
                     uint64_t* ndim_out, uint64_t shape_out[6], double** data_out);
int  readComplexDataByIndex(const char*, int64_t, uint64_t*, uint64_t[6], std::complex<double>**);
int  readIntDataByIndex  (const char*, int64_t, uint64_t*, uint64_t[6], int32_t**);
int  readStringDataByIndex(const char*, int64_t, uint64_t*, uint64_t[6], char**);
```

* **`full_data_path`** — an *instance path* (e.g. `"flux_loop/1/data"`, `"profiles_1d/1/ion/1/temperature"`, `"tgap/x"`). This is the form you'd get from iterating `getAOSShape` + `getLeaves()`.
* **`time_index`** — `-1` → static / aggregate; `>=0` → that slice. For a *dynamic* signal written with `writeDataSlices`, the engine finds the leaf by path **and** verifies the slice exists (`isTimeInLeaf`); a gap returns **`-1`** (see §6.6).
* **`ndim_out`, `shape_out`** — out-params describing the returned buffer: `ndim` dims, `shape_out[0..ndim-1]` per-dim sizes (up to 6 dims).
* **`data_out`** — on success, a `malloc`'d buffer of *exactly* `shape_out[0]*…*shape_out[ndim-1]` elements; **you must `free(*data_out)`** (not `delete[]`) on the *double/int/complex* variants; on the string variant, `free` each element and the array (see `readStringDataByIndex` docs — it is a `char*[]` of NUL-terminated strings).

**Verified (all on the §7 file):**

```cpp
PanzerDB db("demo.h5", PanzerDB::OpenMode::READ);

// (i) A 1-D static leaf (flux_loop/1/data — a 5-vector, no time dimension).
{
  uint64_t ndim; uint64_t shape[6]; double* p;
  int r = db.readDataByIndex("flux_loop/1/data", /*time=*/-1, &ndim, shape, &p);
  // r==0, ndim==1, shape[0]==5
  std::cout << "flux_loop/1/data = ";
  for (auto i = 0u; i < shape[0]; ++i) std::cout << p[i] << " ";
  free(p);
}
// flux_loop/1/data = 100 101 102 103 104

// (ii) A time-slice of a dynamic quantity (profiles_1d t=1, ion j=1).
{
  uint64_t ndim; uint64_t shape[6]; double* p;
  int r = db.readDataByIndex("profiles_1d/1/ion/1/temperature", /*time=*/1, &ndim, shape, &p);
  // r==0, ndim==1, shape[0]==5
  for (auto i = 0u; i < shape[0]; ++i) std::cout << p[i] << " ";
  free(p);
}
// profiles_1d/1/ion/1/temperature @t=1  =  1010 1011 1012 1013 1014
```

#### (b) `readScalar` — one value at a path, typed

```cpp
template<typename T> T readScalar(const std::string& path, int* status) const;   // T ∈ {double,int32_t,...}
```

`status` is set to `0` on success, `-1` otherwise. `readScalar<double>("machine_number", &st) == 21.0`.

#### (c) `readTensor` — read an *entire leaf* (all its time-slices) into a buffer you sized

```cpp
template<typename T> void readTensor(const Leaf& leaf, T* out_buffer) const;
```

You *already know the `Leaf`* (from `getLeaves()`), so you already know `leaf.count` and `leaf.shape`. Allocate `leaf.count` elements (for `T`-sized) and pass the pointer. This is the *lowest-level* "hand me the raw bytes" call — it does a single `data_raw_*` hyperslab and copies into your buffer. No allocation, no path lookup.

**Verified:**

```cpp
const auto& L = db.getLeaves();
const PanzerDB::Leaf* leaf = nullptr;
for (const auto& lf : L) if (std::string(lf.path) == "flux_loop/0/data") { leaf = &lf; break; }

std::vector<double> v(leaf->count);           // count==5
db.readTensor<double>(*leaf, v.data());       // no return, no allocation
// v == [0, 1, 2, 3, 4]
```

#### (d) `readSliceDirect` — one time-slice of a dynamic leaf, into a buffer you sized

```cpp
template<typename T> int readSliceDirect(const Leaf& leaf, int64_t time_index, T* out_buffer) const;
```

Same idea as `readTensor`, but you're asking for *one slice* of a *multi-slice* leaf at a known `time_index`. `slice_volume * n_steps` → you allocate `slice_volume` elements. Returns `int 0/-1`.

**When to prefer (d) over (a):** when you already have the `Leaf` (e.g. you're *looping over a known set of leaves*) and you want to avoid the *path → leaf* hash lookup. If you're calling "give me slice 3 of `power`" repeatedly, `(a)` is clearer, `(d)` is marginally faster.

#### (e) `readLeavesUnion` — many leaves into one buffer, in order

```cpp
int readLeavesUnion(const std::vector<const Leaf*>& leaves, void* buffer, DataType dtype) const;
```

* **`leaves`** — the set to read, **already sorted by desired output order** (typically time). Each must be the same type.
* **`buffer`** — a `dtype`-sized array of `sum(leaf.count for leaf in leaves)` elements.
* **`dtype`** — the `PanzerDB::DataType` of the leaves (all must match).
* **Returns** `0/-1`.

The engine is clever about *monotonic* offset runs: **contiguous offsets collapse into one `H5S_SELECT_OR` hyperslab**, otherwise it falls back to per-leaf reads. So this is the "I want all time-slices of one quantity, in order" API — one call, one buffer, and the batched-read optimisation happens for free.

**Verified** (all 3 time-slices of `profiles_1d/{t}/ion/0/temperature`, ion 0, t=0..2):

```cpp
std::vector<const PanzerDB::Leaf*> ls;
const auto& L = db.getLeaves();
for (const auto& lf : L) {
    if  (std::string(lf.path) == "profiles_1d/0/ion/0/temperature" ||
         std::string(lf.path) == "profiles_1d/1/ion/0/temperature" ||
         std::string(lf.path) == "profiles_1d/2/ion/0/temperature")
        ls.push_back(&lf);
}
std::sort(ls.begin(), ls.end(), [](const auto* a, const auto* b){
    return a->time_index < b->time_index;                       // 0, 1, 2
});

std::vector<double> out(15);                                   // 3 leaves × 5
int r = db.readLeavesUnion(ls, out.data(), PanzerDB::DataType::FLOAT64);
// r == 0
// out == [0,1,2,3,4,   1000,1001,1002,1003,1004,   2000,2001,2002,2003,2004]
```

### 6.5 Handling gaps — the "time-slice doesn't exist" case

A *gap* is one of two things, on disk:

* **Slice-index model:** the leaf was *never written* at that `time_index` (see `tgap/x` in the index table — only rows at `time = 0, 1, 3`).
* **Path-instance model:** the *instance path* `…/2/…` was never created (a dynamic AoS instance the writer skipped).

You express "I want time 2" in both models the same way; the engine handles which one applies. The three methods you'll reach for:

#### `isTimeInLeaf` — the "does this exact (leaf, time) exist?" probe

```cpp
bool isTimeInLeaf(const Leaf& leaf, int64_t time_index) const;
```

Purely arithmetic on the leaf: `time_index ∈ [leaf.time_index, leaf.time_index + (leaf.count / slice_volume))`. **O(1), no I/O, no exception.** Use it to *branch* before you even call a `read*` method.

#### `nearestAvailableSliceIndex` — the "closest slice that *does* exist" resolver

```cpp
int64_t nearestAvailableSliceIndex(const char* full_path,
                                   int64_t requested_idx,
                                   int64_t prefer_direction,   // -1, +1, or 0
                                   const std::vector<double>& time_basis,
                                   double requested_time = -1.0) const;
```

* **`full_path`** — a path that *exists* (or the schema of one) and identifies the quantity.
* **`requested_idx`** — the time index you actually want.
* **`prefer_direction`**:
  * **`-1`** → "previously": the largest available index `<= requested`.
  * **`+1`** → "next": the smallest available index `>= requested`.
  * **`0`** → "nearest in time": whichever is closer in *value* (uses `time_basis` + `requested_time`); ties → lower index.
* **`time_basis`** — the *actual time values* `time[0..]` (not their labels!) — the engine needs them to compute "nearest" for `prefer_direction == 0`.
* **Returns** an *available* index, or `-1` if *nothing* is available.

**Verified** on the `tgap` example (written at `t = 0, 1, 3`; `t = 2` is the gap):

```
readDataByIndex("tgap/x", 0)  →  100
readDataByIndex("tgap/x", 1)  →  200
readDataByIndex("tgap/x", 2)  →  -1   (GAP — this path/time was never written)
readDataByIndex("tgap/x", 3)  →  400

nearestAvailableSliceIndex("tgap/x", 2, -1, time, 2.0) =  1
nearestAvailableSliceIndex("tgap/x", 2, +1, time, 2.0) =  3
nearestAvailableSliceIndex("tgap/x", 2,  0, time, 2.0) =  1   (tie → lower index)
```

That's the whole "find the gap" idiom in three lines. In production code you'd do:

```cpp
int r   = db.readDataByIndex("tgap/x", 2, &ndim, shape, &p);
if (r == -1) {
     int64_t avail = db.nearestAvailableSliceIndex("tgap/x", 2, /*nearest*/ 0, time_basis, requested_time);
     r = db.readDataByIndex("tgap/x", avail, &ndim, shape, &p);   // now it succeeds
}
```

#### `getTimeIndex` — "which time-slice corresponds to *this* time value?"

```cpp
int64_t getTimeIndex(const std::string& timebase_path, double requested_time, int interp_mode) const;
```

* **`timebase_path`** — the *path of the timebase signal* (e.g. `"time"`). It reads that signal, sorts by its values, and finds the index closest / previous / linear to `requested_time`, according to `interp_mode`.
* **Returns** an index `>= 0`, or `-1` if the timebase doesn't exist or is empty.

**Verified** on the `time` signal (`time = [0, 1, 2, 3]`):

```
getTimeIndex("time", 1.5, alconst::closest_interp)   =  2
getTimeIndex("time", 1.5, alconst::previous_interp)  =  1
getTimeIndex("time", 1.5, alconst::linear_interp)    =  1   (linear returns the lower bracketing index)
```

### 6.6 Interpolation and "whole dynamic signal"

#### `readInterpolatedData` — "give me the value at *a time that was never written*"

```cpp
int readInterpolatedData(
        const char* full_data_path, double time,
        const std::vector<double>& time_basis,
        int interp_mode,          // alconst::closest_interp / previous_interp / linear_interp
        int datatype,             // alconst::double_data / complex_data / char_data
        uint64_t* ndim_out, uint64_t shape_out[6],
        void** data_out,
        bool expect_time_dim = true);
```

This is the *only* read method that **synthesises** data: on `linear_interp` it reads the two bracketing slices, then **interpolates** between them. The *interpolation factor* is computed with the *actual times* of the slice you got, so it stays correct across gaps (the two slices it picks are the nearest *available* ones, not the nearest *label* ones — see the `nearestAvailableSliceIndex` inside).

**Verified** on `power = [100, 200, 150, 300]` at `time = [0, 1, 2, 3]`, requested at `time = 1.5`:

```
readInterpolatedData("power", 1.5, time, alconst::closest_interp, ..., &data)
    →  data[0] = 150   (closest-in-time picks the nearest *stored* sample)
readInterpolatedData("power", 1.5, time, alconst::linear_interp, ..., &data)
    →  data[0] = 175   (0.5 * 200 + 0.5 * 150)
```

Note `data_out` is a `void*` you `free` (or reinterpret and use). `shape_out` is `{1}` for a scalar signal; for a `5-vector` leaf it's `{5}`.

#### `getWholeDynamicSignal` — "hand me all time-slices of this signal, in time-order"

```cpp
std::vector<double> getWholeDynamicSignal(const std::string& dataset_name);
```

* **`dataset_name`** — the *full path* of the signal (outside an AoS).
* **Returns** a `std::vector<double>` with one entry per available time-slice, in time order. **Empty** if the path doesn't exist. This is the "I don't care about the individual slices, just give me the whole time-series" API — one call.

**Verified** on `power = [100, 200, 150, 300]`:

```cpp
std::vector<double> s = db.getWholeDynamicSignal("power");
// s == [100, 200, 150, 300]
```

### 6.7 Metadata — attach & fetch

```cpp
// Write (WRITE/APPEND mode):
void writeMetadata(const std::string& schema_path, const std::string& value);

// Read (READ mode):
std::map<std::string, std::string> readMetadata(const std::string& instance_path);

// Both: static utility to convert an instance path to a schema path.
static std::string stripIndices(const std::string& instance_path);
```

`writeMetadata` accepts a **schema path + `@key`** (e.g. `"profiles_1d/ion/temperature@units"`). `readMetadata` accepts an **instance path** (e.g. `"profiles_1d/0/ion/1/temperature"`), strips the instances internally, and returns **every** `@key` attached to that schema path.

**Verified** on the §7 file:

```
readMetadata("profiles_1d/0/ion/0/temperature")  →  { "units": "eV" }
stripIndices("profiles_1d/0/ion/1/temperature")  →  "profiles_1d/ion/temperature"
```

### 6.8 Which `read*` do I pick? — a decision tree

```
        do you know the exact Leaf?
         ┌──── yes ─────┐          ┌──── no ────┐
         ▼              ▼          ▼            ▼
   do you want a    do you want  do you want   do you want
   specific slice?  the whole   a specific    a specific time
   at time t?       leaf?       slice at t?   with *interpolation*?
         │              │              │              │
         ▼              ▼              ▼              ▼
  readSliceDirect  readTensor   readDataByIndex   readInterpolatedData
  (or readDataBy-  (or all of   (or readLeaves-   (closest / previous /
   Index if you    the data)    Union if you      linear interp)
   have a path)                     want many)
```

And for "how big is this AoS?" — always `getAOSShape` — *after* you've checked `isDynamicAOS` to know which number is being counted.

---

## 7. Worked end-to-end examples

Everything in this chapter runs against the **same 23-leaf file** you created in §5.5 (`/tmp/pzdemo/demo_guide.h5`). The contents, so the numbers below are meaningful:

| Path | Kind | What it holds |
|---|---|---|
| `machine_number` | scalar | `21.0` (double) |
| `version` | scalar | `9` (int32) |
| `flux_loop` | **static AoS, size 3** | — |
| `flux_loop/{i}/data` | 1-D field `[5]` | `100·i + [0,1,2,3,4]` |
| `profiles_1d` | **dynamic AoS, 3 time-slices (t=0,1,2)** | — |
| `profiles_1d/{t}/ion` | static AoS, size 2 | — |
| `profiles_1d/{t}/ion/{j}/temperature` | 1-D field `[5]` | `1000·t + 10·j + [0,1,2,3,4]` |
| `tgap` | **dynamic AoS** | — |
| `tgap/x` | scalar series written at **t=0,1,3** (gap at t=2) | `100, 200, — , 400` |
| `power` | dynamic scalar series, 4 slices | `100, 200, 150, 300` |
| `time` | 1-D time basis `[4]` | `0, 1, 2, 3` |
| `profiles_1d/ion/temperature@units` | metadata | `"eV"` |

These are the ten scenarios a scientist actually reaches for. Each snippet is complete (open → do the thing → print), uses only the public API, and its output is the **exact** text the verifier produced.

### 7.1 Size of a static AoS — "how many flux loops?"

A static AoS's member count is stored in its meta-leaf's `shape[0]`. `getAOSShape` returns it (wrapped in a `std::vector<size_t>`, empty when the path doesn't exist — so always guard `empty()` first, or you will read `v[0]` on a null and segfault).

```cpp
#include "panzerdb.h"
#include <iostream>
#include <vector>

static std::string size_of(PanzerDB& db, const std::string& aos) {
    std::vector<size_t> s = db.getAOSShape(aos);
    return s.empty() ? std::string("n/a") : std::to_string(s[0]);
}

int main() {
    PanzerDB db("/tmp/pzdemo/demo_guide.h5", PanzerDB::OpenMode::READ, false);
    std::cout << "flux_loop         size = " << size_of(db, "flux_loop")         << "   (static, declared size)\n";
    std::cout << "profiles_1d/0/ion size = " << size_of(db, "profiles_1d/0/ion") << "   (static, declared size)\n";
    std::cout << "profiles_1d/7/ion size = " << size_of(db, "profiles_1d/7/ion") << "   (n/a: that index does not exist)\n";
    return 0;
}
```

```text
flux_loop         size = 3   (static, declared size)
profiles_1d/0/ion size = 2   (static, declared size)
```

**Takeaway:** the number `3` is not "how many rows for flux_loop in the table" (there are three `flux_loop/i/data` leaves) — it is the *declared* member count, read straight from the meta-leaf, without scanning the index. That is exactly what `al_read_aos_size` / a Direct-Access `list_nodes` would report.

### 7.2 Size of a dynamic AoS — "how many time steps of profiles_1d?"

A dynamic AoS has **no** declared size; its size is "how many time-slices were written". `isDynamicAOS` tells you which counting rule applies; `getAOSShape` and `getDynamicAOSSize` then return the number of slices.

```cpp
#include "panzerdb.h"
#include <iostream>
#include <vector>

static std::string size_of(PanzerDB& db, const std::string& aos) {
    std::vector<size_t> s = db.getAOSShape(aos);
    return s.empty() ? std::string("n/a") : std::to_string(s[0]);
}

int main() {
    PanzerDB db("/tmp/pzdemo/demo_guide.h5", PanzerDB::OpenMode::READ, false);

    std::cout << "profiles_1d isDynamicAOS       = " << db.isDynamicAOS("profiles_1d") << "\n";
    std::cout << "profiles_1d getAOSShape (size) = " << size_of(db, "profiles_1d")
              << "   (prints \"n/a\" if the path is not an AoS)\n";
    std::cout << "profiles_1d getDynamicAOSSize  = " << db.getDynamicAOSSize("profiles_1d") << "\n";
    std::cout << "flux_loop   isDynamicAOS       = " << db.isDynamicAOS("flux_loop") << "\n";
    return 0;
}
```

```text
profiles_1d isDynamicAOS       = 1
profiles_1d getAOSShape (size) = 3   (prints "n/a" if the path is not an AoS)
profiles_1d getDynamicAOSSize  = 3
flux_loop   isDynamicAOS       = 0
```

**Takeaway:** `getAOSShape("profiles_1d")` returns the vector `[3]` — the same `3` as the static case in §7.1, but here it is *derived* (max descendant `time_index` + 1), not stored. For a dynamic AoS you almost always want `getDynamicAOSSize`; use `isDynamicAOS` to route the two cases.

### 7.3 A 1-D field *inside* a static AoS — the `flux_loop[i]/data` case (no dynamic AoS)

This is the case where **nothing is time-dependent**: the only "extension" is the 1-D spatial/profile index, and the AoS index `i` is *static*. For each member, the path is `flux_loop/<i>/data`; the data itself is a plain `[5]` field. There is no time to worry about, so a slice read with `time_index = -1` ("static") is exactly right.

```cpp
#include "panzerdb.h"
#include <iostream>
#include <vector>

static void p1d(const char* lab, const double* d, size_t n){
    std::cout << "    " << lab << " = [";
    for (size_t i=0;i<n;++i) std::cout << d[i] << (i+1<n?", ":"");
    std::cout << "]" << std::endl;
}

int main() {
    PanzerDB db("/tmp/pzdemo/demo_guide.h5", PanzerDB::OpenMode::READ, false);

    size_t n = db.getAOSShape("flux_loop").empty() ? 0
              : (size_t)db.getAOSShape("flux_loop")[0];   // how many loops
    for (size_t i = 0; i < n; ++i) {
        std::string path = "flux_loop/" + std::to_string(i) + "/data";
        uint64_t ndim = 0; uint64_t shape[6] = {0}; double* data = nullptr;
        int r = db.readDataByIndex(path.c_str(), -1, &ndim, shape, &data);   // -1 ⇒ static, no time
        if (r == 0 && data) { p1d(path.c_str(), data, shape[0]); free(data); }
    }
    return 0;
}
```

```text
    flux_loop/0/data = [0, 1, 2, 3, 4]
    flux_loop/1/data = [100, 101, 102, 103, 104]
    flux_loop/2/data = [200, 201, 202, 203, 204]
```

**Takeaway:** this is the "no time axis" pattern. `time_index = -1` selects the *only* version of the leaf (its `shape` and `count` live on that single index row). The engine does **path substitution for you**: `flux_loop/1/data` is matched to the exact row — you never write `flux_loop/data[1]` or handle the AoS index manually.

### 7.4 A slice of a dynamic quantity at a given time

The dynamic quantity here is nested: `profiles_1d/<t>/ion/<j>/temperature`, where the *time* lives in the dynamic-AoS index and `j` (ion) is static. Pick one full instance path, then ask for it **at a specific time**.

```cpp
#include "panzerdb.h"
#include <iostream>

static void p1d(const char* lab, const double* d, size_t n){
    std::cout << "    " << lab << " = [";
    for (size_t i=0;i<n;++i) std::cout << d[i] << (i+1<n?", ":"");
    std::cout << "]" << std::endl;
}

int main() {
    PanzerDB db("/tmp/pzdemo/demo_guide.h5", PanzerDB::OpenMode::READ, false);

    const std::string path = "profiles_1d/1/ion/1/temperature";   // t=1, ion/1
    uint64_t ndim = 0; uint64_t shape[6] = {0}; double* data = nullptr;

    int r = db.readDataByIndex(path.c_str(), 1, &ndim, shape, &data);   // time_index = 1
    std::cout << "    @ t=1  res=" << r;
    if (r == 0 && data) { p1d("temperature", data, shape[0]); free(data); }

    data = nullptr;
    r = db.readDataByIndex(path.c_str(), -1, &ndim, shape, &data);      // -1 ⇒ all slices
    std::cout << "\n    @ all  res=" << r;
    if (r == 0 && data) { p1d("temperature", data, shape[0]); free(data); }
    return 0;
}
```

```text
    @ t=1  res=0
    temperature = [1010, 1011, 1012, 1013, 1014]

    @ all  res=0
    temperature = [1010, 1011, 1012, 1013, 1014]
```

**Takeaway:** because we already **indexed the dynamic AoS into the path** (`profiles_1d/1/...`), there is exactly one slice behind it at `time_index=1` — so `@ all` (`-1`) and `@ t=1` coincide here. (For a *flat* dynamic signal like `power` / `tgap/x`, `-1` concatenates **every** slice — see 7.5.) Note the path itself carries `t=1`; the `time_index` you pass is the *slice within that instance*. Both resolve because the engine stores one index row per `(instance, time)`.

### 7.5 The whole time-series of a flat dynamic signal

For a dynamic scalar/signal that is **not** inside an AoS, the one-call "give me everything in time order" is `getWholeDynamicSignal`.

```cpp
#include "panzerdb.h"
#include <iostream>

int main() {
    PanzerDB db("/tmp/pzdemo/demo_guide.h5", PanzerDB::OpenMode::READ, false);
    std::vector<double> s = db.getWholeDynamicSignal("power");
    std::cout << "    power.size() = " << s.size() << "  values = [";
    for (size_t i=0;i<s.size();++i) std::cout << s[i] << (i+1<s.size()?", ":"");
    std::cout << "]" << std::endl;

    std::vector<double> none = db.getWholeDynamicSignal("does/not/exist");
    std::cout << "    unknown path size = " << none.size() << " (empty ⇒ not found)" << std::endl;
    return 0;
}
```

```text
    power.size() = 4  values = [100, 200, 150, 300]
    unknown path size = 0 (empty ⇒ not found)
```

**Takeaway:** `power` was written as four consecutive slices via `writeDataSlices` (each `n_slices=1`), so the *whole* signal is their concatenation in time order. No loop, no gap-handling needed — this is the cheapest way to export a time-series.

### 7.6 Finding gaps in a time-dependent quantity

`tgap/x` was written at `t=0,1,3` — the slice at `t=2` was **never written** (a gap). Three APIs expose this cleanly.

**(a) Detect the gap** — a per-time read returns `-1` where no slice exists:

```cpp
for (int64_t t = 0; t < 4; ++t) {
    uint64_t ndim = 0; uint64_t shape[6] = {0}; double* data = nullptr;
    int r = db.readDataByIndex("tgap/x", t, &ndim, shape, &data);
    // r == -1  ⇔  no data at this time (a gap)
    std::cout << "    t=" << t << "  res=" << r;
    if (r == 0 && data) { std::cout << "  value=" << data[0]; free(data); }
    std::cout << std::endl;
}
```

```text
    t=0  res=0  value=100
    t=1  res=0  value=200
    t=2  res=-1
    t=3  res=0  value=400
```

**(b) Fall back to the nearest available slice** — `nearestAvailableSliceIndex` tells you *which* slice to read instead:

```cpp
std::vector<double> tb = {0.0, 1.0, 2.0, 3.0};   // the "time" time-basis
const std::string path = "tgap/x";

std::cout << "    prefer previous ( -1) : " << db.nearestAvailableSliceIndex(path.c_str(), 2, -1, tb, 2.0) << "\n";
std::cout << "    prefer next      (+1) : " << db.nearestAvailableSliceIndex(path.c_str(), 2, +1, tb, 2.0) << "\n";
std::cout << "    nearest-in-time ( 0) : " << db.nearestAvailableSliceIndex(path.c_str(), 2,  0, tb, 2.0) << "\n";
```

```text
    prefer previous ( -1) : 1
    prefer next      (+1) : 3
    nearest-in-time ( 0) : 1
```

**Takeaway:** `t=2` is absent; "prefer the largest available ≤ 2" gives `1`, "prefer the smallest available ≥ 2" gives `3`, "nearest in time (tie→smaller)" gives `1`. This is exactly the semantics the AL layer uses to implement "read at the closest available time" without crashing on a gap.

### 7.7 Interpolated read at an arbitrary time

For a quantity that *has* a time-basis (`time = [0,1,2,3]`), you can ask for its value **between** samples. `readInterpolatedData` does the locate-then-read in one call.

```cpp
#include "panzerdb.h"
#include "al_const.h"
#include <iostream>

int main() {
    PanzerDB db("/tmp/pzdemo/demo_guide.h5", PanzerDB::OpenMode::READ, false);
    const std::vector<double> time = {0.0, 1.0, 2.0, 3.0};

    for (int mode : {alconst::closest_interp, alconst::linear_interp}) {
        uint64_t ndim = 0; uint64_t shape[6] = {0}; void* data = nullptr;
        int r = db.readInterpolatedData("power", 1.5, time, mode, alconst::double_data,
                                        &ndim, shape, &data);
        std::cout << "    mode=" << mode << "  res=" << r;
        if (r == 0 && data) { std::cout << "  power(1.5)=" << ((double*)data)[0]; free(data); }
        std::cout << std::endl;
    }
    return 0;
}
```

```text
    mode=1  res=0  power(1.5)=150
    mode=3  res=0  power(1.5)=175
```

**Takeaway:** `power = [100, 200, 150, 300]` at `t = [0,1,2,3]`. At `t=1.5`: **closest** sample is `t=2 → 150`; **linear** is `0.5·200 + 0.5·150 = 175`. `mode` is one of `alconst::{closest_interp, previous_interp, linear_interp}`; `datatype` is one of `alconst::{char_data, integer_data, double_data, complex_data}`. `data_out` (a `void*`) must be `free`d.

### 7.8 Resolving a time to an index (the "locate" primitive)

Under the hood every time-based read calls `getTimeIndex(time, requested, mode)` to find **which stored index** to use. You call it directly when you need the index itself (e.g. to then call a per-leaf `readSliceDirect`).

```cpp
#include "panzerdb.h"
#include "al_const.h"
#include <iostream>

int main() {
    PanzerDB db("/tmp/pzdemo/demo_guide.h5", PanzerDB::OpenMode::READ, false);
    // "time" is the standalone 1-D basis [0,1,2,3] written in §5.5
    std::cout << "    closest  @1.5 -> index " << db.getTimeIndex("time", 1.5, alconst::closest_interp)  << "\n";
    std::cout << "    previous @1.5 -> index " << db.getTimeIndex("time", 1.5, alconst::previous_interp) << "\n";
    std::cout << "    linear   @1.5 -> index " << db.getTimeIndex("time", 1.5, alconst::linear_interp)  << "\n";
    return 0;
}
```

```text
    closest  @1.5 -> index 2
    previous @1.5 -> index 1
    linear   @1.5 -> index 1
```

**Takeaway:** `getTimeIndex` returns a **stored index**, not a value. `closest`→`2`, `previous`→`1`. (`linear` returns the lower bracket index `1`; the actual blend happens in `readInterpolatedData`, not here — see 7.7.) This is the piece you reach for when you have the *time* in hand but the *slice-by-slice* APIs (`readSliceDirect`, `readTensor`) need an *index*.

### 7.9 Batched read of a whole time-stack in one call

Reading three leaves one by one issues three HDF5 I/O round-trips. `readLeavesUnion` coalesces them into a **single** hyperslab-union read, writing them contiguously in the order you provide.

```cpp
#include "panzerdb.h"
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    PanzerDB db("/tmp/pzdemo/demo_guide.h5", PanzerDB::OpenMode::READ, false);

    // Collect the ion/0/temperature leaf at every time-slice (t=0,1,2)
    std::vector<const PanzerDB::Leaf*> leaves;
    for (const auto& lf : db.getLeaves()) {
        std::string p(lf.path);
        if (p.rfind("profiles_1d/", 0) == 0 && p.find("/ion/0/temperature") != std::string::npos)
            leaves.push_back(&lf);
    }
    std::sort(leaves.begin(), leaves.end(),
        [](const PanzerDB::Leaf* a, const PanzerDB::Leaf* b){
            return std::string(a->path) < std::string(b->path);   // deterministic / time order
        });

    std::cout << "    matched leaves = " << leaves.size() << std::endl;
    std::vector<double> out(leaves.size() * 5);                   // 3 leaves × 5
    int r = db.readLeavesUnion(leaves, out.data(), PanzerDB::DataType::FLOAT64);
    std::cout << "    res=" << r << "  union = [";
    for (size_t i=0;i<out.size();++i) std::cout << out[i] << (i+1<out.size()?", ":"");
    std::cout << "]" << std::endl;
    return 0;
}
```

```text
    matched leaves = 3
    res=0  union = [0, 1, 2, 3, 4, 1000, 1001, 1002, 1003, 1004, 2000, 2001, 2002, 2003, 2004]
```

**Takeaway:** the 15-element output is the three `temperature` slices stacked **in the order you sorted** (t=0 → 1 → 2). The engine picks a single `H5S_SELECT_OR` hyperslab when the offsets are monotonic, and falls back to sequential reads otherwise. Use it for "give me this field across all its time-slices" — it's the fast path.

### 7.10 Reading back metadata (units, etc.)

Metadata is keyed by **schema path** (`... @units`, no numeric indices) and read back by **instance path** (the engine strips the indices for you).

```cpp
#include "panzerdb.h"
#include <iostream>

int main() {
    PanzerDB db("/tmp/pzdemo/demo_guide.h5", PanzerDB::OpenMode::READ, false);

    auto md = db.readMetadata("profiles_1d/0/ion/0/temperature");   // instance path
    std::cout << "    metadata(" << md.size() << " key(s)):" << std::endl;
    for (const auto& kv : md) std::cout << "        " << kv.first << " = " << kv.second << std::endl;

    std::cout << "    stripIndices(\"profiles_1d/0/ion/1/temperature\") = "
              << PanzerDB::stripIndices("profiles_1d/0/ion/1/temperature") << std::endl;
    return 0;
}
```

```text
    metadata(1 key(s)):
        units = eV
    stripIndices("profiles_1d/0/ion/1/temperature") = profiles_1d/ion/temperature
```

**Takeaway:** one write (`writeMetadata("profiles_1d/ion/temperature@units", "eV")`) is visible from **every** instance of that schema node — `profiles_1d/0/ion/0/temperature`, `/1/`, all of them — because the index stores the schema path, not the instance path. `stripIndices` is the pure-static utility that maps instance → schema.

---

## 8. Which interface should I use?

Three public layers sit on top of the **same** PanzerDB engine and the **same** on-disk format (§0, §2). You never have to choose between formats — you choose between *ergonomics*.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 You (the application) — pick ONE of these                │
├─────────────────────────────────────────────────────────────────────────┤
│  (A) IMAS C API (al_*)          — Context-driven, schema-aware,      │
│                                   provenance + AoS iteration.     │
│  (B) Direct-Access C++ API      — TensorView, zero-copy slices,  │
│                                   list_nodes, read_tensor().         │
│  (C) PanzerDB C++ (this guide)  — explicit paths/types/shape, the │
│                                   closest layer to the engine.       │
├─────────────────────────────────────────────────────────────────────────┤
│  PanzerDB::  readDataByIndex / readInterpolatedData / readLeavesUnion  │
│  (the storage engine: /index table + columnar data_raw_* datasets)      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.1 The three layers at a glance

| Layer | You give it… | You get back… | Reach for it when… |
|---|---|---|---|
| **(A) IMAS C API** | a `Context` (`IDS`, `path`, `time_request`, `interpolation_mode`, `access_mode`, …) | `data`, ` datatype`, `dim`, `shape[]` + `status` code | you are writing an IMAS tool and want AoS iteration, pulses, provenance, and the "standard" client experience across backends. |
| **(B) Direct-Access** | an IDs name + a path (or a path *template* + AoS indices) | an immutable `TensorView` (numpy-like, zero-copy `slice()`) | you are building a Python/notebook/interactive front-end and want "read this, slice that, give me a view". |
| **(C) PanzerDB (C++)** | an explicit instance path + type + shape + time | allocated buffer / `Leaf` / index info | you need full control, batched `readLeavesUnion`, exact byte-level ownership, or you *are* writing a backend. |

There is no right or wrong — they differ only in how much the engine does for you. (C) is the substrate; (A) and (B) are conveniences over it.

### 8.2 The high-level C API is Context-driven — here is exactly what reaches PanzerDB

The IMAS C API (`al.h`, provided by the `AL` library you link as `-lal`) is generated from the IMAS schema and operates on an opaque `Context`. A client calls a schema-flavoured function (e.g. "read a data signal by index"), fills the Context, and the AL *core* dispatches to the active backend.

In this codebase, that backend contract is `al_backend.h::readData(...)`:

```cpp
// include/al_backend.h — the virtual call the AL core makes into the backend:
virtual int readData(Context *ctx,
                     std::string fieldname,        // the schema field, e.g. "data" / "temperature"
                     std::string timebasename,     // the timebase, e.g. "time"
                     void** data,                  // [out] allocated payload
                     int* datatype,                // [inout] CHAR/INTEGER/DOUBLE/COMPLEX_DATA
                     int* dim,                     // [out] 0=scalar … up to MAXDIM
                     int* size) = 0;               // [out] per-dimension sizes
```

The PanzerDB backend implements this by translating the Context into the explicit calls you saw in §6–§7: `readDataByIndex` for a single slice / static node, `readInterpolatedData` when `ctx->interpolation_mode != UNDEFINED`, `getAOSShape` / `getDynamicAOSSize` for AoS sizing, `nextSlice`-style gap resolution via `nearestAvailableSliceIndex`. **Every** `al_read_*` you call in a tool ultimately lands on one of those. That is why a value you read through the C API and a value you read through PanzerDB are always identical — same row, same leaf, same buffer.

> Practical consequence: if your tool is already using `al_*`, you do **not** need to drop down to PanzerDB for ordinary reads. Drop down only for (i) the batched `readLeavesUnion` export path, (ii) a custom backend, or (iii) a fast out-of-process reader that links the engine directly (§9.4).

### 8.3 The Direct-Access C++ API (`read_tensor`) — for views, not ownership

The Direct-Access layer (`include/direct_access_api.h`) wraps PanzerDB and returns a **zero-copy** `TensorView`:

```cpp
TensorView read_tensor(const std::string& ids_name, const std::string& path);
TensorView read_tensor(const std::string& ids_name,
                       const std::string& path_template,   // e.g. "profiles_1d/ion/temperature"
                       const std::vector<int>& aos_indices);// e.g. {1, 1}  →  .../1/ion/1/...
auto [nodes, aos_map] = list_nodes(ids_name, /*recursive=*/true, /*show_aos=*/true, /*show_metadata=*/true);
```

`TensorView` is indexed by `SliceSelection` (a top-level struct: `Type::{Index, Range, All}`, with helpers `SliceSelection::at(i)` / `::between(start, end)` / `::all()`), and `slice(...)` returns **sub-views with no copy** — exactly the thing a notebook wants:

```cpp
TensorView t = read_tensor(pulse, "profiles_1d/ion/temperature");
// Take the first element along every leading axis → a zero-copy sub-view:
TensorView t0 = t.slice({ SliceSelection::at(0) });        // one slice
// A [start..end) range on the first axis, keep the rest:
TensorView tr = t.slice({ SliceSelection::between(1, 3) });
// t0 / tr are views over t's storage; no bytes were copied.
```

Because `TensorView` holds a `shared_ptr<char[]>`, the buffer stays alive as long as any view does — no `free()`, no dangling pointer, and `.metadata()` (e.g. the `@units` attached to the schema) is carried along. This is the opposite contract from PanzerDB's allocated `readDataByIndex` buffer, which **you** own and must `free`.

### 8.4 Decision heuristic (what I reach for)

```
do you build an IMAS *tool* (GUI, CLI, plugin)?
   └─ YES → (A) IMAS C API. You get AoS iteration + pulses + provenance for free.

do you expose IMAS to Python / a dashboard / a notebook?
   └─ YES → (B) Direct-Access TensorView. Zero-copy view, easy to wrap in numpy.

do you write a *backend*, an export pipeline, or a tight read loop?
   └─ YES → (C) PanzerDB directly. You need the batched union read, exact
            offsets, or you cannot afford the AL overhead.
```

The rest of this chapter shows the (C) path, because that is where the *semantics* live; the (A) and (B) layers are thin, well-tested shims on top of it.

---

## 9. Performance & I/O notes

### 9.1 Two time models — what actually happens per read

§2.5 says the two time models coexist in one file. The **cost** of each is different:

| Model | Storage cost per write | Read cost per slice | Why |
|---|---|---|---|
| **(A) Slice-index model** | one row per `(path, time)` — the raw dataset grows | **O(1)** `getLeaves()` cache hit → one hyperslab | the leaf you want is *exactly one row*; the offset is already in `offset`. No scan. |
| **(B) Path-instance model** | one row per *instance* (each time-slice is a distinct path) | **O(1)** once the path is known — but you must *know* the path | every time-slice is a separate entry in `/index`; the cache is keyed by (path) and you need the concrete `/2/ion/...` string first (`stripIndices` or `list_nodes` to enumerate). |

Both are O(1) in the *data* (HDF5 hyperslab = one `H5Dread` at `(offset,count)`). The cost difference is in *which* index row you need — and the two models differ only in whether you compute that path by substituting an index, or by enumerating it.

### 9.2 What is cached, and what is re-read

`getLeaves()` is **the single read path**. Once it has run, the leaf cache (`cached_leaves` + `leaf_lookup` hash by path) is *reused* by: `getAOSShape`, `getDynamicAOSSize`, `isDynamicAOS`, `getLeafType`, `readScalar`, `readTensor`, `readSliceDirect`, `readDataByIndex`, `readInterpolatedData`, `nearestAvailableSliceIndex`, `readLeavesUnion`. No other cache is invalidated except by `append_index_row`/`incrementArrayIndex` (write side) — so a read-only session pays the index parse **exactly once**.

If you open a huge file in READ mode and call **100** different data reads, that is *not* 100 index parses. It is 1 parse + 100 hyperslab reads.

### 9.3 Batched read — the `readLeavesUnion` fast path

`readLeavesUnion` (7.9) chooses a **single** hyperslab-union (`H5S_SELECT_OR`) when leaf offsets are **monotonic**; otherwise it falls back to N sequential reads. The engine makes that choice for you — but you still get the best case by **sorting your leaves** (ascending on `offset`, which usually coincides with ascending on `time_index`) before the call. The §7.9 example sorts by path, which is a proxy for that ordering in a parent-first-written file.

If your offsets are scattered (e.g. you are reading the time-slices in a different order from how they were written), or they are spread far apart in the raw dataset, `readLeavesUnion` will pay the sequential cost. In that case, a `getWholeDynamicSignal` + local selection is often cheaper — one contiguous read, then you index in RAM.

### 9.4 When direct PanzerDB beats the AL layer

You pay a *small, constant* overhead in the AL layer: Context marshalling (path, type, shape, time), pulse bookkeeping, provenance recording, and exception-to-status conversion. For **single** reads that is negligible. For **thousands** of small reads in a loop (an export tool, a batch reader, `dump`-style utilities), the direct PanzerDB path avoids the marshalling and lets you pipeline (pre-allocate the buffer, reuse `leaf` pointers, `free` only at the end). That is the regime where (C) in §8.1 is unambiguously better than (A)/(B).

### 9.5 Tuning knobs (write-side)

| Knob | Meaning | Where |
|---|---|---|
| `setChunkingHint("time_series" \| "bulk_write" \| "array_of_structures" \| "interactive")` | selects a preset for chunk sizes (row count, bytes-per-chunk, per-type) | before the first `beginArray`/`writeData` in WRITE mode — the chunk sizes are *pinned* at dataset creation |
| `setCompressionLevel(0–9)` / `disableCompression()` | GZIP level on the **new** datasets | same — new datasets only |
| `ChunkingStats` / `printChunkingStats()` | cumulative bytes/chunks + compression ratio | any time — a `const` read |

Do not expect to change these on an **APPEND** open: `readChunkingConfig()` *restores* the settings from the on-disk file, so every appender to one file sees the same preset (the whole point — the file's layout is self-describing).

If you see an HDF5 "too many chunk reads" warning, or you are writing very small slices repeatedly, check:

1. You are in WRITE or APPEND (READ mode cannot write) — obvious but worth writing.
2. Your `n_slices` in `writeDataSlices` is at least 1 — 0-slice writes are a no-op, and *not* a way to "reserve a time index" (use `advanceTimebase` instead, §10.2).
3. You are not calling `flush()` on every slice; `flush()` forces an `H5Dclose`+reopen on every dataset (visible I/O cost, even if you don't `close()`).

### 9.6 A rule of thumb for read latency

```
   one readDataByIndex  ≈  one leaf lookup in a cached hash map + one H5Dread
                          (for a 5-double leaf: ~µs, HDF5 metadata dominates)

   one readInterpolatedData  ≈  getTimeIndex() + one or two readDataByIndex()
                          (the extra leaf lookup is the cost; the blend is math)

   one readLeavesUnion  ≈  one leaf-lookup loop + one H5Dread (monotonic)
                          or  N leaf-lookups + N H5Dreads (scattered)
```

If your workload is dominated by "read one slice, then compute, then read the next slice", you are in the first row. If *both* are dominated by HDF5 metadata, the lever is to move more work into the *engine* (`readLeavesUnion`, or a custom "multi-leaf" routine) rather than in a client-side loop that re-opens `getLeaves()` every iteration (it is cached, so you should not; but if you *do* re-`open()` with a fresh `PanzerDB` per slice, you pay it).

---

## 10. Error handling & ownership

There is no `try/catch` in the C-API world (the AL layer catches and converts to a `status` code), so the *engine* has two distinct contracts: **`int` returns** for the data paths, and **exceptions** for the metadata/type paths. Getting this wrong is where most of the segfaults come from.

### 10.1 Return-code contract: `0`/`-1`, and who owns the buffer

| Method | Return | Buffer on success | You must |
|---|---|---|---|
| `readDataByIndex` / `readComplexDataByIndex` / `readIntDataByIndex` / `readStringDataByIndex` | `0` ok, `-1` not-found/failed | `malloc`'d `void*` / `char*` / `int32_t*` / `std::complex<double>*` | `free(*data_out)` (or `delete[]` for the string-array variant — check the variant). The buffer is the *caller's* to own from that point. |
| `readInterpolatedData` | `0` ok, `-1` failed | same as above | `free(*data_out)`. |
| `readLeavesUnion` | `0` ok, `-1` failed | **caller's** `void*` (it's filled, not allocated) | Nothing to `free` — you pass a `std::vector<T>` in. |
| `readScalar<T>` | sets `*status` to `0`/`-1` | N/A (returns by value) | Nothing. |
| `readTensor<T>` | void, but **throws** on type/path mismatch | fills **your** buffer | Buffer must already be correctly sized (`leaf.count` elements). |
| `readSliceDirect<T>` | `0`/`-1` | fills **your** buffer | Same — you own the buffer. |
| `getWholeDynamicSignal(name)` | `std::vector<double>` (empty ⇒ not found) | self-managed (`std::vector`) | Nothing. |
| `nearestAvailableSliceIndex` | `int64_t` (`-1` = nothing available) | N/A | Nothing. |
| `getTimeIndex` | `int64_t` (≥0 index, or `-1` if no such time) | N/A | Nothing. |

**The #1 mistake** (and the one that produced the §0 segfault) is treating `getAOSShape` / `getLeafType` like the others: `getAOSShape` **returns an empty `std::vector<size_t>`** when the path is unknown, so `v[0]` is undefined behaviour. `getLeafType` **throws `ALException`** for the same condition. Neither returns `-1`. The rule: *for the "does it exist?" checks, use the exception/empty-value contract, not the int contract.*

### 10.2 Gap semantics — what "not found at t=X" actually means

`readDataByIndex(path, t, …)` returning `-1` is **not** an I/O error. It means one of:

1. The path does not exist at all (typos, wrong instance), **or**
2. The path exists but no slice is stored at that `time_index` (a **gap**).

The engine does not currently distinguish the two at the *return code*. To tell them apart:

```cpp
int r = db.readDataByIndex("tgap/x", 2, &ndim, shape, &data);   // r == -1
// Is the path even there?
const PanzerDB::Leaf* lf = nullptr;
for (const auto& l : db.getLeaves())
    if (std::string(l.path) == "tgap/x") { lf = &l; break; }
// lf != nullptr, lf.time_index is 0  ⇒  the path is there, the *slice* is the gap.
```

The clean recovery path is `nearestAvailableSliceIndex` (7.6): it returns **the index of the available slice** and `-1` only when no slice for the path exists at all.

### 10.3 The `time_index = -1` sentinel — one constant, two meanings

`-1` in the `time_index` position means:

- **"static"** when the path has *no* dynamic parent (e.g. `flux_loop/1/data`): read the single slice, which is the only one stored for that leaf; or
- **"all slices, concatenated"** when the path *does* have a dynamic parent (e.g. `power`): return the entire time-series stacked.

Both semantics are implemented by the same code path (`writeDataSlices` at write time, and the slice lookup at read time). That is what makes `getWholeDynamicSignal` cheap on the read side: it is exactly the `-1` case, filtered by type, for the common "flat double signal" case.

### 10.4 The exception contract (type and metadata paths)

These throw `ALException` (or a derived `ALBackendException`) on not-found — you **must** catch them:

```cpp
imas::direct_access::DataType dt;
try {
    dt = db.getLeafType("some/unknown/path");           // throws ALBackendException
} catch (const ALBackendException& e) {
    std::cerr << "no such leaf: " << e.what() << "\n";
}

// metadata writes are fire-and-forget (no return value), but *reads* may throw
// if the schema path cannot be identified at all:
try {
    auto md = db.readMetadata("no/such/schema");
} catch (const ALException& e) { std::cerr << "metadata: " << e.what() << "\n"; }
```

Everything else in the §6/§7 data paths returns an int and never throws for the "not found at this time" case — so you can write a tight C-style loop (as in 7.6) without `try/catch`, and branch on the return.

### 10.5 `PanzerDB` object lifetime

- `PanzerDB` is **non-copyable** (it holds HDF5 file/dataset identifiers and a leaf cache). Pass by reference, or move the owning `std::unique_ptr<PanzerDB>` around.
- `close()` is idempotent and *also* runs from the destructor — safe to call both.
- `flush()` is for **Writers** (to make buffered writes visible to other readers *without* closing). A Reader doesn't call `flush()` — there's no buffer to flush.
- If you are building a backend or embedding PanzerDB inside another `HDF5` process (see the `hid_t loc_id` constructor), decide *once* who owns the `loc_id` (`close_loc_id_on_exit=true` for "we created it", `false` for "the caller will close it").

---

## 11. Quick-reference tables

### 11.1 Every public method, one line each

| Method | Signature (abridged) | One-liner |
|---|---|---|
| `getLeaves()` | `const std::vector<Leaf>&` | The index table (cached). Start here. |
| `getAOSShape(aos)` | `std::vector<size_t>` | AoS member/slice count — **empty if path unknown**. |
| `getDynamicAOSSize(aos)` | `size_t` | Slices written on a dynamic AoS (`0` if unknown). |
| `isDynamicAOS(aos)` | `bool` | Dynamic vs static AoS — route the two counting rules. |
| `getLeafType(path)` | `imas::direct_access::DataType` (throws) | Storage type at a path (not a return-code API). |
| `readDataByIndex(path,ti,nd,sh,d)` | `int` | Read by path + time; the workhorse. Allocates. |
| `readComplexDataByIndex` / `readIntDataByIndex` / `readStringDataByIndex` | `int` | Same, typed. |
| `readScalar<T>(path,&st)` | `T` | Read a scalar by path; status out-param. |
| `readTensor<T>(leaf,buf)` | `void` (throws) | Fill a caller-sized buffer from a known `Leaf`. |
| `readSliceDirect<T>(leaf,ti,buf)` | `int` | Fills caller's buffer for one slice of a signal. |
| `readLeavesUnion(leaves,buf,dtype)` | `int` | Batched union read for many leaves. |
| `isTimeInLeaf(leaf,ti)` | `bool` | Does this leaf hold this time-slice? |
| `nearestAvailableSliceIndex(path,i,dir,tb,tt)` | `int64_t` | Gap-aware fallback index (`-1` if none). |
| `getTimeIndex(tb,t,mode)` | `int64_t` | Time → stored index. |
| `readInterpolatedData(...)` | `int` | Locate + read (+ interpolate). Allocates. |
| `getWholeDynamicSignal(name)` | `std::vector<double>` | All slices of a flat signal, in time order. |
| `beginArray(name,size)` / `beginArray(name,tb)` | `void` | Enter a static / dynamic AoS. |
| `beginArray(ArrayLevel&)` | `void` | Enter with full control (rarely needed). |
| `incrementArrayIndex()` / `setCurrentArrayIndex(i)` | `void` | Move within the top AoS. |
| `endArray()` | `void` | Pop the top AoS. |
| `isInsideDynamicAOS(&tb)` | `bool` | Are we in a dynamic AoS right now? |
| `advanceTimebase(name,n)` | `void` | Skip `n` slices on a dynamic AoS (write-side gap). |
| `writeData<T>(name,shape,data,count,tb)` | `void` | Write one static node. |
| `writeDataSlices(name,shape,data,n,tb)` | `void` | Write `n` slices of a dynamic signal. |
| `writeMetadata(schema@key,value)` | `void` | Attach metadata to a schema path. |
| `readMetadata(instance)` | `std::map<string,string>` | Read all keys for a schema node. |
| `stripIndices(instance)` | `static std::string` | instance → schema path. |
| `flush()` / `close()` / `dumpLeafIndex()` | `void` | Terminal / debug lifecycle. |
| `setChunkingHint(h)`, `setCompressionLevel(n)`, `disableCompression()` | `void` | WRITE-side tuning (before first write). |
| `getChunkingStats()` / `printChunkingStats()` | `ChunkingStats` / `void` | Cumulative I/O counters + ratios. |
| `synchronizeArrayStack(names, idx)` | `void` | Rebuild the array stack to a known AL state. |
| `getArrayStackSize()` / `isArrayStackEmpty()` | `size_t` / `bool` | Introspection of the write stack. |
| `getOpenMode()` | `OpenMode` | READ / WRITE / APPEND. |

### 11.2 Decoding a `Leaf`

| Field | Meaning |
|---|---|
| `shape` | Dimensions of **one slice** (empty ⇒ scalar / meta). |
| `time_index` | First time-slice stored in this row (0 for static). |
| `path` | Full instance path (e.g. `profiles_1d/0/ion/0/density`). |
| `parent_path` | Path of the immediate parent (rebuilt from `parent_id` and the *parent's* row at load). |
| `offset`, `count` | Byte-free element offset and element count in the raw dataset. |
| `flags` | Low 4 bits = kind; upper bits = `PanzerDB::DataType`. See §11.3. |
| `is_empty` | Convenience flag for `flags & 0xF == 1`. |

### 11.3 Decoding `Leaf.flags`

| `flags & 0xF` (kind) | Meaning |
|---|---|
| `0` | normal data node (or metadata). |
| `1` | explicitly-preserved empty node. |
| `2` | static AoS meta-node (size in `shape[0]`). |
| `3` | dynamic AoS meta-node (time-evolving). |

`flags >> 4` is the **storage** `PanzerDB::DataType`:

| value | type |
|---|---|
| `0` | `FLOAT64` (double) |
| `1` | `INT32` |
| `2` | `COMPLEX128` |
| `3` | `STRING` |
| `4` | `LIST_OF_STRINGS` |
| `99` | `UNKNOWN` |

The **public** `imas::direct_access::DataType` (what `getLeafType` returns) is a *different* enum — see §11.6.

### 11.4 `OpenMode`

| Mode | Semantics |
|---|---|
| `WRITE` | Create / truncate a new file. |
| `READ` | Open for read-only. Cannot write. |
| `APPEND` | Open for read + append-only writes; existing rows keep their ids; `next_row_id` is seeded from the on-disk index count. |

### 11.5 `alconst` — the interpolation & data-type constants

| Constant | Value | Meaning |
|---|---|---|
| `alconst::undefined_interp` | `0` | No interpolation. |
| `alconst::closest_interp` | `1` | Nearest stored sample. (§7.7, 7.8) |
| `alconst::previous_interp` | `2` | Largest stored index ≤ requested time. |
| `alconst::linear_interp` | `3` | Linear blend bracketed by the two surrounding samples. |
| `alconst::char_data` | `50` | Strings. |
| `alconst::integer_data` | `51` | `int32`. |
| `alconst::double_data` | `52` | `double`. |
| `alconst::complex_data` | `53` | `std::complex<double>`. |

> These are the *public* constants of the AL layer. `PanzerDB::DataType` (internal storage) and `imas::direct_access::DataType` (external read) each have their **own** enum values — see §11.3 and §11.6.

### 11.6 The three data-type enums, side by side

| C++ type | `PanzerDB::DataType` (storage) | `imas::direct_access::DataType` (getLeafType) | `alconst::*` (AL API) |
|---|---|---|---|
| `double` | `FLOAT64 = 0` | `DOUBLE = 1` | `double_data = 52` |
| `int32_t` | `INT32 = 1` | `INT32 = 2` | `integer_data = 51` |
| `std::complex<double>` | `COMPLEX128 = 2` | `COMPLEX_DOUBLE = 6` | `complex_data = 53` |
| `char[N]` | `STRING = 3` | `STRING = 4` | `char_data = 50` |
| `char*[N]` (list) | `LIST_OF_STRINGS = 4` | `LIST_OF_STRINGS = 7` | (n/a) |

All three are **independent** enumerations — do not `static_cast` between them. Use the layer's own enum in that layer's calls.

---

# Appendices

## Appendix A — The M1 on-disk layout (the 14-column index)

This is the part of the design that is *not* visible through the API, but it is what makes every read O(1) and what a writer can rely on when reasoning about gaps. The two authoritative sources are the column convention recorded in `new_example_check_index_design.md` §3, and the `index_buffer` / `Leaf` documentation in `src/hdf5/panzerdb.h`.

### A.1 Top-level datasets

```
/index        (N, 14)  uint64   one row per logical node (the index)
/paths        (N,    )  fixed 256-byte C1 strings   full instance path per row
/data_raw_f64 (M_f, )  float64   every double payload, type-grouped
/data_raw_i32 (M_i, )  int32     every int32 payload
/data_raw_c128(M_c, )  complex128 every complex128 payload
/data_raw_str (M_s, )  variable-length UTF-8          every string payload
(no /parent_paths)
```

There is **no `/parent_paths` dataset** in the M1 layout. (That was the legacy design.) Parent identity is stored *numerically* in the index and the parent path is *reconstructed on read* — Appendix A.3. The reason: a fixed, narrow, append-only row is **SWMR-safe** (a single writer, multiple readers) — a variable-width parent-text column written concurrently is what breaks the reader. Keeping the row `14 × uint64`, with the full text in a separate fixed-width `paths` dataset, is what lets a reader that lags a writer still see a consistent row.

### A.2 The 14 columns (uint64, left-to-right)

| col | name | meaning |
|---:|---|---|
| `[0]` | `type` | storage type tag (0 = float64, 1 = int32, …). In the 23-leaf file: mostly `0`; the `version` row is `1`. |
| `[1]` | `ndim` | number of dimensions of **one slice**. |
| `[2..7]` | `shape[0..5]` | the slice shape. `shape[0]` is the declared size of a static AoS meta-node. |
| `[8]` | `time_index` | first time-slice stored in this row (0 for static). |
| `[9]` | `offset` | element offset inside `data_raw_<type>`. |
| `[10]` | `count` | element count for this row. |
| `[11]` | `flags` | low 4 bits = **kind** (0 data, 1 empty, 2 static AoS, 3 dynamic AoS); upper bits = storage type. |
| `[12]` | `parent_id` | the **row number** of the enclosing AoS meta-node, or `0xFFFFFFFFFFFFFFFF` (NO_PARENT) for root rows. |
| `[13]` | `index_value` | the instance index within that parent AoS at write time (0 when absent). |

Worked row (from `new_example_check_index_design.md`, a 15-row variant of the §7 file):

```
row  3: [0, 1, 5,  0,0,0,0,0,  0,  0, 5,  0, 2, 0]
        │  │  │       │        │   │   │   │  │  │
        │  │  │       │        │   │   │   │  │  └─ index_value = j = 0
        │  │  │       │        │   │   │   │  └──── parent_id   = row 2 (the ion meta-node)
        │  │  │       │        │   │   │   └─────── flags       = 0 (kind data, type float64)
        │  │  │       │        │   │   └──────────── count       = 5
        │  │  │       │        │   └──────────────── offset      = 0 in data_raw_f64
        │  │  │       │        └─────────────────── time_index    = 0
        │  │  │       └──────────────────────────── shape[0]      = 5  (a [5] field)
        │  │  └───────────────────────────────────── ndim         = 1
        │  └───────────────────────────────────────── (type slot)  = 0 (float64)
        └───────────────────────────────────────────── row id       = 3
```

The full 23-leaf file (re-verified with `h5py` on the current build) gives:

```
index        (23, 14)  uint64
paths        (23,    )  |S256          (256-byte fixed-width C1 strings)
data_raw_f64 (57,   )  float64         1 (machine_number) + 15 (flux data) + 30 (temperature)
                                      + 3 (tgap/x @t=0,1,3) + 4 (power) + 4 (time)
data_raw_i32 ( 1,   )  int32           version
data_raw_c128( 0,   )  complex128      (none in the §7 file)
data_raw_str ( 1,   )  (vlen UTF-8)    profiles_1d/ion/temperature@units = "eV"
```

### A.3 Reconstructing `parent_path` on read

`parent_path` is not stored; it is derived. The rule (from `new_example_check_index_design.md` §4, implemented in `getLeaves()`):

| `kind` (flags & 0xF) | parent_path is |
|---:|---|
| 0 or 1 (data / empty) | `full_path` minus the **last** segment (the leaf name). |
| 2 or 3 (AoS meta) | `full_path` minus the **last two** segments (instance + name). |
| `parent_id == NO_PARENT` | `""` (a root). |

That is exactly what you saw in the §6.1 dump: `profiles_1d/0/ion/0/temperature` → parent `profiles_1d/0/ion/0`; `profiles_1d/0/ion` → parent `profiles_1d/0`; `profiles_1d` → parent `""`.

### A.4 Why this layout, in one paragraph

The engine's single invariant is: **a reader never waits on a writer, and a writer never waits on a reader.** (SWMR — an explicit, future requirement for the IMAS backend.) The three consequences are: (1) the index row is a *fixed* `14 × uint64`, so it can be published as one atomic 112-byte word and never torn; (2) the variable-width path text is kept in a *separate* fixed-width `paths` dataset so a partially-written row is still self-consistent; (3) `parent_path` is *derived* rather than stored so the writer does not have to double-write text on every row. Everything else in §2 — the columnar `data_raw_*`, the `Leaf` struct, the two time models — is downstream of that one invariant.

---

## Appendix B — The two time models, side by side

The §7 file has **both** models present, and every read in §7–§9 works because the engine does not care which one produced a row. Here is the difference, made concrete.

| | **Model A: slice-index** (e.g. `power`, `tgap/x`) | **Model B: path-instance** (e.g. `profiles_1d/*/ion/*/temperature`) |
|---|---|---|
| How the time lives | in the `time_index` column **of one row per slice**, under the same path | in the **path itself** (`.../1/...`); each slice has its own path |
| Number of index rows per leaf | 1 row per `(path, slice)` — `power` → 4 rows | 1 row per `(instance path)` — `.../temperature` appears 6× (3 time × 2 ion) |
| How you *address* a slice | `path` + `time_index` (the index does the work) | `full instance path` (you must know/enumerate the concrete index) |
| Gap handling | `readDataByIndex(path, t)` → `-1` where the slice was skipped; `nearestAvailableSliceIndex` recovers | gaps appear as whole missing instance paths; enumerate via `getLeaves()` |
| Best fit | flat time-series, root scalars, "signal" fields | nested AoS trees where each time-slice is a distinct "experiment step" |
| Batched read | `readLeavesUnion` over the N slice-rows of one path | `readLeavesUnion` over the N instance-paths |

Both are the *same* on-disk row — the difference is which column carries the time. `power`'s 4 slices are 4 rows sharing one path with `time_index` = 0,1,2,3; `profiles_1d`'s 6 temperature rows are 6 *different* paths. Reading either is identical: resolve the row (by path+time or by full path), then `H5Dread` at `(offset, count)`.

### B.1 A single leaf, both shapes

```text
Model A — flat signal, one path, time_index drives it:
  tgap/x   time_index=0  offset=0   count=1     ← path repeated
  tgap/x   time_index=1  offset=1   count=1
  tgap/x   time_index=3  offset=2   count=1     ← the gap (t=2 never written)
                                    count=1

Model B — nested AoS, path drives it:
  profiles_1d/0/ion/0/temperature   offset= 1   count=5
  profiles_1d/0/ion/1/temperature   offset= 6   count=5
  profiles_1d/1/ion/0/temperature   offset=11   count=5     ← path carries the time
  ...
```

### B.2 Choosing at write time

- You have a 1-D signal evolving in time, *outside* an AoS → **Model A** (`writeDataSlices` with `n_slices>1`, or one call per slice).
- You have a nested structure where each time-step is a whole "scene" (an AoS of fields) → **Model B** (`beginArray(name, timebase)` → static `beginArray` for the inner nodes → `writeData`/`writeDataSlices`).

If you are unsure, Model B is the general case and also handles a single time-step — use it when the *structure* is tree-shaped. Model A is the special case where the structure is *flat* and only the value changes.

---

## Appendix C — Path model: schema vs instance, and `stripIndices`

There are **two** kinds of path, and the engine keeps them straight by convention:

| | **Instance path** | **Schema path** |
|---|---|---|
| Carries | numeric indices (concrete member) | none — the "type" of the node |
| Example | `profiles_1d/0/ion/1/temperature` | `profiles_1d/ion/temperature` |
| Used by | `readDataByIndex`, `readMetadata`, `getLeafType`, `readTensor`, per-`Leaf` ops | `writeMetadata`, `readLeavesUnion` (to match a whole stack) |
| Storage | one row per instance | one row per schema node (the meta-node holds the schema; each instance row points at it via `parent_id`) |

`stripIndices` is the **pure-static** conversion from instance → schema:

```cpp
static std::string PanzerDB::stripIndices(const std::string& path) {
    // "profiles_1d/0/ion/1/temperature" → "profiles_1d/ion/temperature"
    // (every purely-numeric path segment is removed)
}
```

That is the engine's *internal* conversion. In your code you do not need to call it for ordinary reads (the `read*` family accepts instance paths and handles substitution internally). Reach for it when you are:

1. **Iterating a schema** — you know the schema path (`"profiles_1d/ion/temperature"`) and want to enumerate its *instances* (`profiles_1d/0/ion/0/temperature`, `/1/`, …). Use `getLeaves()` and a `stripIndices`-style filter (`p` such that `stripIndices(p) == schema`), as in 7.9.
2. **Writing generic code** that must *not* depend on a concrete instance but on a *family* of leaves.

### C.1 Path substitution inside a dynamic AoS

`readDataByIndex` and friends accept a *single* full path — the path you give already includes every numeric instance. The engine resolves it by looking up the leaf directly (it is a single hash lookup on the path string). There is no "template" substitution in the API — `profiles_1d/ion/temperature` on its own is *not* a valid instance path; you must always pass one of the concrete `profiles_1d/<i>/ion/<j>/temperature` strings (or use the `read_tensor(ids, path_template, aos_indices)` Direct-Access overload, which does the substitution for you).

### C.2 What a "schema path" row *does* have in the index

The meta-node *for a schema* (e.g. `profiles_1d` as a dynamic AoS, or `profiles_1d/0/ion` as a static AoS) is **one row**. Its `shape[0]`, when the AoS is static, is the *declared* size; when dynamic, `shape` is empty and the size is derived from its descendants' `time_index` values. The `index_value` of that meta-node is meaningless (there is no parent to be an index *within*). The *instance* rows — the `profiles_1d/0/...` and `profiles_1d/1/...` data leaves — each have a `parent_id` pointing back to the meta-node's row. That is how `stripInstances`/`getLeaves` can walk up and down without a second path table.

---

## Appendix D — Type & flag decoding, with the verified 23-row table

### D.1 Decoding `Leaf.flags`

Two independent encodings sit in the same 64-bit `flags` word:

- **Low 4 bits** — the *kind*: 0 = data, 1 = preserved-empty, 2 = static AoS meta, 3 = dynamic AoS meta.
- **Upper bits** — the storage `PanzerDB::DataType`: `FLOAT64=0`, `INT32=1`, `COMPLEX128=2`, `STRING=3`, `LIST_OF_STRINGS=4`, `UNKNOWN=99`.

### D.2 The full 23-row decode (verified from the current build)

| path | kind | time | count | shape | dtype (flags>>4) |
|---|:---:|:---:|:---:|---|:---:|
| `machine_number` | 0 | 0 | 1 | `{}` | 0 (f64) |
| `version` | 0 | 0 | 1 | `{}` | **1 (int32)** |
| `flux_loop` | **2** (static AoS) | 0 | 0 | **`{3}`** | 0 |
| `flux_loop/{0,1,2}/data` | 0 | 0 | 5 | `{5}` | 0 |
| `profiles_1d` | **3** (dynamic AoS) | 0 | 0 | `{0}` | 0 |
| `profiles_1d/{0,1,2}/ion` | 2 | 0 | 0 | `{2}` | 0 |
| `profiles_1d/{t}/ion/{0,1}/temperature` | 0 | 0,1,2 | 5 | `{5}` | 0 |
| `tgap` | 3 | 0 | 0 | `{0}` | 0 |
| `tgap/x` ×3 (t=0,1,3) | 0 | 0/1/**3** | 1 | `{}` | 0 |
| `power` | 0 | 0 | 4 | `{1}` | 0 |
| `time` | 0 | 0 | 4 | `{1}` | 0 |
| `profiles_1d/ion/temperature@units` | 0 | 0 | 1 | `{}` | **3 (STRING)** |

The *column-sum* implied by this table is exactly what `h5py` reports on disk: `data_raw_f64 = 1 + 15 + 30 + 3 + 4 + 4 = 57`, `data_raw_i32 = 1`, `data_raw_str = 1`, `data_raw_c128 = 0`.

### D.3 The three data-type enums, fully

| C++ | `PanzerDB::DataType` | `imas::direct_access::DataType` | `alconst::*` |
|---|---|---|---|
| `double` | `FLOAT64 = 0` | `DOUBLE = 1` | `double_data = 52` |
| `int32_t` | `INT32 = 1` | `INT32 = 2` | `integer_data = 51` |
| `std::complex<double>` | `COMPLEX128 = 2` | `COMPLEX_DOUBLE = 6` | `complex_data = 53` |
| `char[N]` | `STRING = 3` | `STRING = 4` | `char_data = 50` |
| `char*[N]` | `LIST_OF_STRINGS = 4` | `LIST_OF_STRINGS = 7` | (n/a) |

(`imas::direct_access::DataType` is a C++ `enum class` with no explicit values, so it is `FLOAT=0, DOUBLE=1, INT32=2, INT64=3, STRING=4, COMPLEX_FLOAT=5, COMPLEX_DOUBLE=6, LIST_OF_STRINGS=7, UNKNOWN=8` — a full enumeration, independent of the other two.)

### D.4 `alconst` ranges

| Constant family | Value range |
|---|---|
| `undefined_interp / closest_interp / previous_interp / linear_interp` | `0 / 1 / 2 / 3` |
| `char_data / integer_data / double_data / complex_data` | `50 / 51 / 52 / 53` (i.e. `DATA_TYPE_0 + {0,1,2,3}`, `DATA_TYPE_0 = 50`) |
| `read_op / write_op / slice_op / global_op / timerange_op / open_pulse …` | `30 / 31 / 21 / 20 / 22` and up from `ACCESS_PULSE_0 = 40` (see `include/al_defs.h.in`) |

The two *interp* and *data-type* families are disjoint (0–3 vs 50–53), so you can check `x <= 3` to distinguish them from data types — the AL core does the same in `data_interpolation.cpp`.

---

## Appendix E — The full canonical write program

This is the *exact* source the §7 file was produced from (the write half of `.qwen/tmp/demo_guide.cpp`). §5.5 already narrates each line; this is the listing as it would go into a test or an example. Everything from `PanzerDB db(path, WRITE, true);` through `db.close();` below is the canonical artifact.

```cpp
#include "panzerdb.h"

int main() {
    const std::string path = "/tmp/pzdemo/demo_guide.h5";

    PanzerDB db(path, PanzerDB::OpenMode::WRITE, /*preserve_empty=*/true);

    // --- (1) root scalars / static 1-D ---
    double mn  = 21.0;  db.writeData("machine_number", {}, &mn, 1);
    int32_t vi = 9;     db.writeData("version",        {}, &vi, 1);

    // --- (2) STATIC AoS: flux_loop[3] > data[5] ---
    db.beginArray("flux_loop", 3);
    for (int i = 0; i < 3; ++i) {
        double d[5];
        for (int k = 0; k < 5; ++k) d[k] = 100.0 * i + k;
        db.writeData("data", {5}, d, 5);
        if (i < 2) db.incrementArrayIndex();   // 0,1,2
    }
    db.endArray();

    // --- (3) DYNAMIC AoS: profiles_1d[t=0,1,2] > ion[2] > temperature[5] ---
    auto temp = [](int t, int j) {
        std::vector<double> d(5);
        for (int k = 0; k < 5; ++k) d[k] = 1000 * t + 10 * j + k;
        return d;
    };
    db.beginArray("profiles_1d", "time");            // enter the dynamic AoS at t=0
    for (int t = 0; t < 3; ++t) {
        if (t > 0) db.incrementArrayIndex();         // → t
        db.beginArray("ion", 2);
        { auto d0 = temp(t, 0); db.writeDataSlices("temperature", {5}, d0.data(), 1, ""); }
        db.incrementArrayIndex();                    // → ion/1
        { auto d1 = temp(t, 1); db.writeDataSlices("temperature", {5}, d1.data(), 1, ""); }
        db.incrementArrayIndex();                    // sentinel for ion
        db.endArray();
    }
    db.incrementArrayIndex();                          // sentinel for profiles_1d
    db.endArray();

    // --- (4) DYNAMIC AoS with a GAP: tgap[x] written at t=0,1,3 (t=2 skipped) ---
    db.beginArray("tgap", "time");
    double x0 = 100.0;  db.writeDataSlices("x", {}, &x0, 1, "");   // t=0
    db.incrementArrayIndex();                                    // → t=1
    double x1 = 200.0;  db.writeDataSlices("x", {}, &x1, 1, "");   // t=1
    db.incrementArrayIndex();                                    // → t=2  (SKIP: no slice)
    db.incrementArrayIndex();                                    // → t=3
    double x3 = 400.0;  db.writeDataSlices("x", {}, &x3, 1, "");   // t=3
    db.incrementArrayIndex();
    db.endArray();

    // --- (5) flat dynamic signals (Model A) ---
    double p[4] = {100.0, 200.0, 150.0, 300.0};
    db.writeDataSlices("power", {1}, p, 4, "");
    double tb[4] = {0.0, 1.0, 2.0, 3.0};
    db.writeDataSlices("time",  {1}, tb, 4, "");

    // --- (6) metadata ---
    db.writeMetadata("profiles_1d/ion/temperature@units", "eV");

    db.close();
    return 0;
}
```

Running this (Appendix F) and then re-reading (`h5py`) yields exactly the layout in §A.1: (23,14) index rows, 256-B/row `paths`, 57/1/0/1 for `data_raw_{f64,i32,c128,str}`.

### E.1 What the writer actually does per call

| Call | Work done |
|---|---|
| `beginArray(name, size)` | Push a static AoS level onto the write stack; register one meta-row (kind=2) in `/index`. |
| `beginArray(name, timebase)` | Push a dynamic AoS level; register one meta-row (kind=3); the *current time* is the value in `aos_time_counters[name]`. |
| `incrementArrayIndex()` | `current_index++` on the stack top; for a dynamic AoS, `aos_time_counters[name]++` also bumps (so the next write is at the next slice — the gap mechanism). |
| `writeData<T>(name, shape, data, count, "")` | Append `count` elements to `data_raw_<T>`, append **one** `/index` row (time = 0), bump the *enclosing* dynamic counter (if any). |
| `writeDataSlices(name, shape, data, n, timebase)` | Append `n × shape` elements to `data_raw_<T>`, append **one** `/index` row (time = current, count = `n*shape`), bump the timer by `n`. |
| `endArray()` | Pop the AoS level; restore `path_prefix`. |
| `writeMetadata(schema@key, value)` | Append a string to `data_raw_str` + one index row flagged as a metadata leaf. |
| `flush()` / `close()` | Force every append to HDF5; `close()` also closes all datasets and the file. |

### E.2 A note on the "sentinel" `incrementArrayIndex`

You will see two patterns:

- **`beginArray` + `writeData` + `endArray` for the LAST member**: `endArray` does not `increment` — it leaves `current_index` at the last written value. If you want to write N members, you `beginArray`, then loop N times doing `writeData` + `incrementArrayIndex` (the last one *is* the increment that lands on the sentinel; `endArray` then pops).
- **`increment` before the write** in a loop (the §5/§7 patterns): because the *first* write at `i=0` happens at `current_index=0`, we only `increment` *after* members 0..N-2 and before members 1..N-1.

Both patterns are equivalent for N≥1; the choice is only a matter of readability.

---

## Appendix F — Building a one-off reader against `libal.so`

The canonical build recipe used to produce every example in §7. Substitute your own `PROJ` and `HDF5/IMPI` paths as needed (this one matches the AFS/iimpi install on the build machine).

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJ=/afs/eufus.eu/g2itmdev/user/g2lfleur/IMAS-Core
HDF5=/gw/swimas/software/HDF5/1.14.4.3-iimpi-2023b
CXX=/gw/swimas/software/GCCcore/13.2.0/bin/g++
OUTDIR=/tmp/pzdemo
OUT=${OUTDIR}/demo_guide
SRC=${PROJ}/.qwen/tmp/demo_guide.cpp
mkdir -p "${OUTDIR}"

export LD_LIBRARY_PATH="/gw/swimas/software/GCCcore/13.2.0/lib64:${HDF5}/lib:/gw/swimas/software/intel-compilers/2023.2.1/compiler/2023.2.1/linux/compiler/lib/intel64:/gw/swimas/software/impi/2021.10.0-intel-compilers-2023.2.1/mpi/2021.10.0/lib:/gw/swimas/software/impi/2021.10.0-intel-compilers-2023.2.1/mpi/2021.10.0/lib/release:${PROJ}/build:${LD_LIBRARY_PATH:-}"

echo "=== compiling ==="
"${CXX}" -std=c++17 -O1 -march=native \
  -I "${PROJ}/src/hdf5" \
  -I "${PROJ}/include" \
  -I "${PROJ}/build/include" \
  -I "${HDF5}/include" \
  -DHDF5 \
  "${SRC}" -o "${OUT}" \
  -L "${PROJ}/build" -lal \
  -L "${HDF5}/lib" -lhdf5 -lz -ldl

echo "=== running ==="
"${OUT}"
```

### F.1 Reading only (no write)

For a *read-only* tool that does not need PanzerDB's write stack, you do **not** need to re-implement any of §5 — just the two `#include`s and the constructor in READ mode. The same build recipe works; the link line is unchanged because `-lal` pulls in both the read *and* write halves of the engine (they are in the same `libal.so`), and the linker only includes what you actually call.

If you are linking against a C-only tool (or want to use the AL high-level API rather than raw `PanzerDB`), the same `libal.so` exports the `al_read_*`/`al_write_*` C interface — the Context marshalling and the backend dispatch are both in the same shared object. Pick the layer per §8; the build recipe is identical.

### F.2 Running on the AFS host

`LD_LIBRARY_PATH` is only needed for *execution*; the compile line is self-contained. If you see `symbol lookup error` or `libal.so: cannot open shared object file`, the issue is always an `LD_LIBRARY_PATH` problem, not a *link* problem — the binary records which `libal.so` it links (`ldd` will tell you), and it must be findable on the search path at run time.

### F.3 A minimal CMake alternative

If the tool is going to live in the repo, the idiomatic way to get the same build is to add a `CMakeLists.txt` next to it with:

```cmake
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(al REQUIRED)        # the AL package from this repository
add_executable(my_reader my_reader.cpp)
target_include_directories(my_reader PRIVATE
    ${HDF5_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/hdf5
    ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(my_reader PRIVATE al HDF5::HDF5)
```

(Exact package name and target names depend on your AL/CMake setup — this repo exposes `libal.so` at `$BUILD/build/libal.so` and the headers at `include/` and `src/hdf5/`; the CMake wiring depends on how your project's top-level `CMakeLists.txt` exposes them. The standalone `g++` recipe in Appendix F works without a CMake build, which is useful for throwaway readers.)

---

## Change log

* v1 (2026-09-05) — initial public version of this guide. Covers the engine (§2), the concepts (§3), every PanzerDB method with a verified example (§5–§7), the three-layer stack (§8), performance & I/O (§9), error handling (§10), and quick-reference tables (§11). Appendices A–F: M1 on-disk layout, the two time models, the path model, type decoding, and a reproducible build recipe. All examples are built and run against the current `libal.so` and verified against the same 23-leaf file.

---

*End of guide.*

---


