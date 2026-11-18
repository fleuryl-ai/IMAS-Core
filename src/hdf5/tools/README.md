# IMAS HDF5 Exploration Tools

This directory contains command-line utilities designed to explore and inspect IMAS data stored in HDF5 files via the `direct_access` layer. These tools mimic the behavior and output format of the standard HDF5 utilities (`h5ls`, `h5dump`) but are tailored for the specific logical structure of IMAS data.

## Overview

Unlike standard HDF5 tools that show the physical layout of the file, `imas_h5ls` and `imas_h5dump` use the `direct_access` API to read the `PanzerDB` index. This allows them to reconstruct and display the **logical hierarchy** of IMAS Data Structure (IDS), even though the underlying physical storage is optimized as a flat structure.

### Key Tools

1. **`imas_h5ls`**: Lists the content of an IDS file.
2. **`imas_h5dump`**: Displays the detailed structure and data values of an IDS or specific signals.

---

## imas_h5ls User Guide

`imas_h5ls` is used to browse the structure of an IDS file.

### Usage
```bash
imas_h5ls <filename_base> [options]
```
*Note: Do not include the `.h5` extension in the filename.*

### Options
- `-r`: Recursive mode. List all nodes and sub-nodes in the hierarchy.
- `-a`: Show AoS (Array of Structures) nodes explicitly.
- `-m`: Show metadata nodes (e.g., attributes starting with `@`).

### Examples
**List top-level nodes:**
```bash
imas_h5ls my_ids_file
```

**Recursively list all datasets in a file:**
```bash
imas_h5ls my_ids_file -r
```

---

## imas_h5dump User Guide

`imas_h5dump` provides a detailed view of the datasets, including their dimensions, types, metadata, and actual data values.

### Usage
```bash
imas_h5dump <filename_base> [path_to_signal]
```

### Features
- **Data Display**: Formats multi-dimensional arrays recursively with curly braces `{}`.
- **Metadata**: Displays associated metadata such as `units`, `description`, and `source` for each signal.
- **Path Support**: You can dump the entire file or target a specific path using the IMAS notation (e.g., `equilibrium/time`).
- **Slicing Support**: Supports IMAS slice syntax in the path (e.g., `core_profiles/profiles_1d[0:5]/grid/rho_tor_norm`).

### Examples
**Dump the entire content of a file:**
```bash
imas_h5dump my_ids_file
```

**Dump a specific signal with its metadata:**
```bash
imas_h5dump my_ids_file equilibrium/time
```

**Dump a specific time slice of a profile:**
```bash
imas_h5dump my_ids_file "core_profiles/profiles_1d[10]/grid/rho_tor_norm"
```

---

### panzer_index User Guide

`panzer_index` prints the **raw PanzerDB v2 index table** — the physical
`index` (2D uint64) and `paths` datasets that `imas_h5ls`/`imas_h5dump` are
built on. Each row is decoded (kind, data type, ndim, shape, time, offset,
count, parent), and both the current 12-column layout and the legacy
14-column one are supported, exactly like `PanzerDB::getLeaves()`.

Unlike the other two tools, it is a plain Python script (h5py only): no C++
API, no build step.

Two derived views help with time series:
- **`slices` column** = number of time slices of the signal. The time
  dimension is stored *outside* the leaf shape (by design, see
  `writeDataSlicesImpl` in `panzerdb.cpp`): in the "bulk" layout one row's
  `count = n_slices × shape`, in the legacy "per-slice" layout the row is
  repeated once per slice. The column reports `rows(path) × count/shape`,
  and `-` for static data.
- **`timebase(s)` header line** = the F64 dataset(s) named `time` in the
  group (convention used by the AL/tests: root `time`, or `<aos-dyn>/time`).
  Note the signal→timebase association is **not persisted** in the table:
  the `t` column is the write's *start* time index, and the reader resolves
  the timebase from the dictionary path.

#### Usage
```bash
env/bin/python3 src/hdf5/tools/panzer_index.py <fichier.h5> [options]
```

#### Options
- `--limit N`: print only the first N selected rows
- `--match MOTIF`: keep rows whose path contains MOTIF
- `--kinds data,empty,aos-static,aos-dyn`: keep only these node kinds
- `--raw`: undecoded raw columns (tab-separated), no header — handy to cross-check against `h5dump`
- `--no-parent`: print the numeric `parent_id` instead of the resolved parent path

#### Examples
```bash
# Full table of a small file
env/bin/python3 src/hdf5/tools/panzer_index.py magnetics_THIN.h5

# First 20 rows of a large file
env/bin/python3 src/hdf5/tools/panzer_index.py core_profiles_3.h5 --limit 20

# Only AoS meta-nodes
env/bin/python3 src/hdf5/tools/panzer_index.py core_profiles_3.h5 --kinds aos-static,aos-dyn --limit 20
```

---

## Technical Note: Logical vs. Physical Structure

`PanzerDB` stores data in a flattened HDF5 format to optimize high-performance I/O for slices. If you use the standard `h5ls` on an IMAS file, you will see a large list of numbered datasets. 

The `imas_h5` tools are essential because they use the `/index` dataset within the file to map these physical datasets back into the human-readable IMAS hierarchy (e.g., `magnetics/flux_loop(1)/flux/data`).
