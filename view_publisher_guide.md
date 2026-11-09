# `imas_publish_view` — user guide

The tool publishes a **human-readable view** of a PanzerDB (HDF5 backend v2) file:
one virtual dataset per logical IMAS node, laid out the way the IMAS tree reads,
next to the raw tables. It does not rewrite the data — it adds a view you can
browse with plain HDF5 tools (`h5ls`, `h5dump`, h5py).

Binary: `build/src/hdf5/tools/imas_publish_view`
Sources: `src/hdf5/tools/imas_publish_view.cpp` (CLI), `src/hdf5/view_publisher.{h,cpp}` (library)

## 1. Running it

The binary links against `libal.so` and the iimpi HDF5/MPI stack, so the library
path must be set:

```bash
export LD_LIBRARY_PATH=/gw/swimas/software/GCCcore/13.2.0/lib64:\
/gw/swimas/software/HDF5/1.14.4.3-iimpi-2023b/lib:\
/gw/swimas/software/intel-compilers/2023.2.1/compiler/2023.2.1/linux/compiler/lib/intel64:\
/gw/swimas/software/impi/2021.10.0-intel-compilers-2023.2.1/mpi/2021.10.0/lib:\
/gw/swimas/software/impi/2021.10.0-intel-compilers-2023.2.1/mpi/2021.10.0/lib/release:\
/gw/swimas/software/Boost/1.83.0-GCC-13.2.0/lib:$PWD/build

build/src/hdf5/tools/imas_publish_view <file.h5> [options]
```

## 2. Read-only or not? — the important part

**Exactly one mode is read-only: `--list`.** Everything else publishes, which
means opening the file read-write and **replacing** the view group.

| Command | Touches the file? | What happens |
|---|---|---|
| `file.h5 --list` | **No** (`H5F_ACC_RDONLY`) | prints the existing view, nothing is written |
| `file.h5` | **Yes** (read-write) | deletes the view group, writes a new one |
| `file.h5 --expand` | **Yes** | same, in the per-index-row layout |
| `file.h5 --no-aos-groups` | **Yes** | same, datasets flat under the view root |
| `file.h5 --max-mappings N` | **Yes** | same, with a cap on mappings per dataset |
| `file.h5 --stats` | depends on the other flags | output only — it does not make a publish read-only |

Publishing is **idempotent**: running the same command twice leaves one view, not
two. But a publish **replaces** the previous view, so switching between grouped
and `--no-aos-groups` removes the other layout — the two do not coexist, because
both live under the same group name (`imas_view` by default).

Measured on a copy of `magnetics_THIN.h5`: `--list` left the file byte-identical
(438 581 B, hash of `index` + `data_raw_*` unchanged); `--no-aos-groups` changed
the size to 426 789 B and removed every AoS group — while the hash of the raw
tables stayed the same. So a publish rewrites **the view**, never **the data**.

## 3. Options

| Option | Applies to | Effect |
|---|---|---|
| `--list` | — | inspect an existing view, read-only |
| `--root <name>` | both | view group name, default `imas_view` |
| `--expand` | publish | historical layout: one VDS per index row, every AoS instance in its own subgroup |
| `--no-aos-groups` | publish | no AoS group at all; one dataset per node, path folded into the name with `:` |
| `--max-mappings N` | publish | above N mappings a dataset keeps only its first slices and is flagged `@mapping_cap=1`; `0` (default) = unlimited |
| `--stats` | publish | print the counters block |

List mode prints the listing and exits; `--expand`, `--no-aos-groups` and
`--max-mappings` are ignored there.

## 4. Examples

### Inspect without changing anything

```bash
build/src/hdf5/tools/imas_publish_view equilibrium_1.h5 --list
```
```
/imas_view/grids_ggd/grid/space/objects_per_dimension/geometry_content&description  Dataset {1, 1, 1, 3} @dims_label=grids_ggd|grid|space|objects_per_dimension
```

If the file has no view yet:

```
ERROR: no '/imas_view' view in /tmp/vg/noview.h5 — publish it first: imas_publish_view /tmp/vg/noview.h5
```
exit code **2**.

### Publish the default (grouped) view

```bash
build/src/hdf5/tools/imas_publish_view copy_of_equilibrium.h5 --stats
```
```
Published collapsed VDS view under '/imas_view' in /tmp/vg/eq.h5
  datasets published:   100
  AoS groups sized:     17
  empty nodes skipped:  0
  errors (skipped):     0
  virtual mappings:       9014
  max in one dataset:     198
  cells left as fill:     238346
  rank overflow skipped:  0
  mapping-cap fallbacks:  0
```

`h5ls` then shows the IMAS tree:

```
/imas_view/time_slice/boundary&elongation                      Dataset {198}
/imas_view/grids_ggd/grid/space/objects_per_dimension/geometry_content&description  Dataset {1, 1, 1, 3}
```

Every AoS level is a leading dimension, size 1 included, and each level's length
is its **declared** AoS size — so two nodes under the same AoS always print the
same length for that level.

### Flat view: no AoS groups at all

```bash
build/src/hdf5/tools/imas_publish_view copy.h5 --no-aos-groups --stats
```
```
  datasets published:   100
  AoS groups sized:     0
  virtual mappings:       9014
  cells left as fill:     238346
```
```
/imas_view/grids_ggd:grid:space:objects_per_dimension:object:geometry  Dataset {1, 1, 1, 3, 13815, 2}
```

A recursive `h5ls` then lists datasets only (0 groups), and each AoS size is
still visible in the shape. `:` is not an HDF5 path separator and never appears
in an IMAS name, so the object stays reachable — verified through h5py and
`h5dump -d 'imas_view/grids_ggd:grid:space:geometry_type&index'`.

### Safety valve

```bash
build/src/hdf5/tools/imas_publish_view copy.h5 --max-mappings 20
```
With `N=20` on that file, 97 of the 100 datasets were trimmed to their first
slices and flagged `@mapping_cap=1`. Prefer the default (unlimited): the largest
dataset in these fixtures needs 198 mappings and publishes fine.

## 5. Reading the published view

`h5ls` prints **no attribute at all**, and `h5ls -v -r` prints attribute *names*
without values:

```
/imas_view/time_slice/boundary&elongation Dataset {198/198}
    Attribute: aos_ndim scalar
        Type:      native double
    Attribute: datatype scalar
        Type:      8-byte null-terminated ASCII string
```

To see the annotation values, use `h5dump -H -A` or h5py:

```python
import h5py
h = h5py.File("copy.h5", "r")
d = h["imas_view/grids_ggd/grid/space/objects_per_dimension/object/geometry"]
print(d.shape)          # (1, 1, 1, 3, 13815, 2)
print(d.attrs)          # @aos_ndim 5, @datatype 'float64', @dims_label '...',
                        # @n_elements 14416, @n_mappings 1, @schema '...'
```

The product of the dims (82 890) is larger than `@n_elements` (14 416): the
difference is **holes** — declared AoS instances with no stored value. They read
back as the IMAS invalid value (`-9E+40` for reals, both components of a
complex; `-999999` for integers; empty string for text):

```python
d[0, 0, 0, 0, 7208, :]   # array([-9.e+40, -9.e+40])  — instance never written
```

`--list` reports them per dataset as `holes=<n>` and separates the leaf's own
shape from the structure dims with a bar:

```
.../object:geometry  Dataset {1, 1, 1, 3, 13815, | 2} @dims_label=grids_ggd|grid|space|objects_per_dimension|object|geometry holes=68474
```

### Two things to watch

1. **The view refers to the file by name.** After publishing, renaming or moving
   the file leaves a view that still *lists* correctly but **reads the fill
   value silently** (exit 0, no HDF5 error): the virtual mappings point at the
   old name. Re-publish after any rename.
2. **A publish changes the file size.** Re-publishing in another layout does not
   return the file to its previous size (HDF5 keeps free space). The data does
   not change, only the view.

## 6. How it works (for developers)

Line numbers are from `src/hdf5/view_publisher.cpp`.

### Pipeline

```
publishImpl (:918)  open H5F_ACC_RDWR, delete + recreate <root>
  └ findRawPrefix (:901)   locate the group holding /index ("" or a sub-group)
  └ getLeaves()            the index rows: path, parent_id, kind, shape, offset, count
  └ processCollapsed (:862)
       ├ buildTree          node/AoS structure from the leaf paths
       ├ buildAggregates (:466)   one AggLeaf per LOGICAL path
       ├ buildRuns (:628)         source ranges coalesced into mappings
       └ createAggVDS (:711)      one virtual dataset + annotations
```

The view is derived **from the index tables**, not by walking HDF5 groups: the
raw layout is a flat buffer plus `index`/`path_table`, so there is no HDF5 tree
to walk. `kind` on each row separates data leaves from the AoS meta rows that
declare an array's size.

### Dims: logical path, declared sizes

`buildAggregates` folds every AoS level (and the time axis, when a signal is
stored as one row per slice) into leading dims. Two rules matter:

- **Every level is a dim, including size 1.** `h5ls` prints no attribute, so the
  shape is the only place a size can show.
- **A level's length is the declared AoS size**, not the largest instance index
  seen in that dataset. `AggLeaf::rawLevelPaths` keys each level exactly like the
  `@aos_size` map, so the same AoS yields the same length in every sibling
  dataset and the shape agrees with `@aos_size` by construction. Instances with
  no data are simply left unmapped.

`AggLeaf` therefore holds `rawLevelNames`/`rawLevelPaths` (before any folding)
and the final `levelNames`/`levelSizes`.

### Mappings, and why there are so few

Each dataset is a virtual dataset whose storage is a list of source selections.
`buildRuns` walks the leaf's cells in dim order and **coalesces contiguous source
ranges**: adjacent instances whose data is adjacent in `data_raw_*` collapse into
one strided mapping. In the test fixture, `D/0` and `D/3` (declared size 5, only
those two instances written) become a **single** mapping onto target positions 0
and 3 — `@n_mappings = 1`, not 2.

Two HDF5 constraints shape this code:

- This HDF5 build (1.14.4-3 iimpi) has the **DCPL-only** VDS API:
  `H5Pset_layout(H5D_VIRTUAL)` + `H5Pset_virtual` + `H5Dcreate2`. The `H5VS*`
  streaming API is absent from its headers.
- A mapping is rejected unless the **source and destination selections hold the
  same number of elements**. A dataset that ends up with no mapping reads as
  nothing but fill, so `createAggVDS` refuses it rather than publishing it.

`--max-mappings N` trims a dataset's mappings to its first slices and flags it
`@mapping_cap=1`. Dims whose product would exceed HDF5's maximum rank are skipped
and counted as `rank overflow skipped`.

### Holes

Unmapped cells take the dataset's fill value, set to the IMAS invalid value:
`-9E+40` for reals (both components of a complex), `-999999` for integers, empty
string for text. This is what makes a sparse AoS or a ragged slice readable
without inventing a mask object.

### Annotations

Written on each dataset: `@datatype`, `@n_elements`, `@schema`, `@aos_ndim`
(how many leading dims come from the structure — the leaf shape starts after
them), `@dims_label`, and `@n_mappings`, plus `@ragged` / `@mapping_cap` when
they apply. On the AoS groups: `@aos_size` and `@aos_dynamic` — absent in
`--no-aos-groups` mode, since no group is created.

Note the storage types: the numeric annotations are **`native double`**, the
others fixed-length ASCII. So in h5py, `attrs['aos_ndim']` comes back as
`5.0`, not `5`.

`describeObject` (:971) formats one listing line from those attributes; `listView`
(:1104) opens the file `H5F_ACC_RDONLY` and visits `<root>`. The bar in
`Dataset {1, 1, 1, 3, 13815, | 2}` is inserted at `@aos_ndim`.

### Why the view refers to the file by name

`H5Pset_virtual` records the source as a **filename + dataset path** string. It
is resolved when the virtual dataset is read, not when it is written. That is why
renaming a published file leaves a view that lists fine and reads fill silently:
the recorded name no longer exists, and HDF5 substitutes the fill value instead
of failing. Re-publish after renaming.

### Guardrails

The publisher is additive and confined to `<root>`: it never creates, deletes or
modifies `index`, `paths`, `path_table` or `data_raw_*`, never touches the read
path, and takes a snapshot of the index at publish time (a view is not
maintained as data keeps being appended). It does not bump the backend format
version. The regression coverage lives in `tests/hdf5_backend/test_publish_view.cpp`
(fixture writes `A/B/val`, `C/vec5`, a declared-but-sparse `D[5]`, per-slice and
bulk time series, and a ragged `equilibrium/time_slice` pair; then checks the
grouped view, the flat view, the listing, and switching back).

## 7. Exit codes

| Situation | Code |
|---|---|
| success (publish, or `--list` on a published file) | 0 |
| no file argument, unknown option, empty `--root`, unparsable `--max-mappings` | 2 (usage on stderr) |
| `--list` on a file with no view | 2 (`ERROR: no '<root>' view in <file> — publish it first: ...`) |

