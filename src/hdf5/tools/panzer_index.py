#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panzer_index — affiche la table d'index brute d'un fichier IMAS-HDF5 produit
par le backend PanzerDB v2 (src/hdf5).

Lit les datasets ``index`` (2D uint64) et ``paths`` (C_S1 fixe 256 B/ligne)
de chaque groupe produit par PanzerDB, et affiche une table décodée :
kind, type, ndim, shape, time, offset, count, parent — le même décodage que
``PanzerDB::getLeaves()`` (panzerdb.cpp), sans passer par le C++ :

  - kind  = flags & 0xF  ; 0=data 1=empty 2=AoS statique 3=AoS dynamique
  - type  = flags >> 4   ; 0=F64 1=I32 2=C128 3=STR 4=STR_LIST 5=STR_CHUNKED
  - layout v3 (7 colonnes + attribut index_version=3) : lignes packées 56 o et
    chemins internés dans /path_table ; décodé ici vers la vue 12 colonnes
  - layout v4 (7 colonnes + index_version=4) : même packing, bit 12 du mot
    drapeaux = bloc brut (0 = data_raw_*, 1 = data_time_f64). La colonne
    affichée « blk » nomme le dataset où « off » pointe ; elle ne passe JAMAIS
    par type (= flags >> 4), qui reste masqué à 0xFF comme côté lecteur C++
  - layout v5 (7 colonnes + index_version=5) : le bloc brut occupe les bits 12
    et 13 (0 = data_raw_*, 1 = data_time_f64, 2 = data_scalar_f64/_i32/_c128 —
    les scalaires ayant une valeur par pas de temps, ex. global_quantities&ip)
  - layout v2 12 colonnes : ndim | shape[6] | time | off | count | flags | parent
  - layout 14 colonnes (legacy) : + col. "type" en tête et "index_value" en queue (jamais lues)

Usage:
  env/bin/python3 src/hdf5/tools/panzer_index.py <fichier.h5>
      [--limit N] [--match MOTIF] [--kinds data,empty,aos-static,aos-dyn]
      [--raw] [--no-parent]

Lecture seule : le fichier n'est jamais ouvert en écriture.
"""

import argparse
import os
import sys


def _import_h5py():
    try:
        import h5py
        return h5py
    except ModuleNotFoundError:
        # <root>/src/hdf5/tools/panzer_index.py -> retenter avec le venv du projet.
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        py = os.path.join(root, "env", "bin", "python3")
        if os.path.exists(py) and os.path.realpath(py) != os.path.realpath(sys.executable):
            os.execv(py, [py] + sys.argv)
        raise


h5py = _import_h5py()

NO_PARENT = 0xFFFFFFFFFFFFFFFF
KIND_NAMES = {0: "data", 1: "empty", 2: "aos-static", 3: "aos-dyn"}
KIND_ALIASES = {name: k for k, name in KIND_NAMES.items()}
TYPE_NAMES = {0: "F64", 1: "I32", 2: "C128", 3: "STR", 4: "STR_LIST", 5: "STR_CHUNKED"}
RAW_DATASETS = ("data_raw_f64", "data_raw_i32", "data_raw_c128", "data_raw_str",
                "data_time_f64",
                "data_scalar_f64", "data_scalar_i32", "data_scalar_c128")


def parent_path_of(path, kind):
    """Parent path d'un noeud, comme ``PanzerDB::getLeaves()`` (parent_path_of) :
    une feuille data retire 1 segment (son nom), un meta-noeud AoS en retire 2
    (instance + nom). Ne dépend pas du layout : le chemin est la source de vérité."""
    n_strip = 2 if kind in (2, 3) else 1
    end = len(path)
    for _ in range(n_strip):
        slash = path.rfind("/", 0, end)
        end = slash if slash != -1 else 0
    return path[:end]


def colmap(ncols):
    """Indices des colonnes décodées ; miroir de ``PanzerDB::getLeaves()``.

    ``block`` est la colonne du bloc brut (v4 : 1 bit ; v5 : 2 bits) ; None
    ailleurs : une ligne v2/legacy n'a pas de bloc séparé, ses offsets sont dans
    data_raw_*."""
    if ncols == 14:  # legacy : "type" en tête + "index_value" en queue (jamais relues)
        return dict(ndim=1, shape=2, time=8, off=9, cnt=10, flags=11, parent=12, block=None)
    if ncols == 12:  # v2 : ndim | shape[6] | time | offset | count | flags | parent_id
        return dict(ndim=0, shape=1, time=7, off=8, cnt=9, flags=10, parent=11, block=None)
    if ncols == 13:  # vue décodée du packing v3/v4 : les 12 colonnes v2 + bloc
        return dict(ndim=0, shape=1, time=7, off=8, cnt=9, flags=10, parent=11, block=12)
    raise ValueError(
        "dataset 'index' à {} colonnes : layout inconnu (attendu 7 (v3/v4), 12 ou 14)".format(ncols))


M32 = 0xFFFFFFFF


def index_version(group):
    """Version du format telle qu'estampée (0 = sans attribut, déduit des colonnes)."""
    v = group["index"].attrs.get("index_version", 0)
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def is_v3(group):
    """Layout packé 7 colonnes (v3 comme v4 : même packing, le bloc est un bit)."""
    return index_version(group) >= 3 or group["index"].shape[1] == 7


def decode_v3(group):
    """Décode les 7 colonnes u64 packées du layout v3/v4 (+ /path_table interné)
    dans la représentation 12 colonnes v2 augmentée d'une 13e colonne pour le
    bloc brut, miroir de PanzerDB::getLeaves().

    col0 offset | col1 count
    col2 path_id(u32 bas) | parent_id(u32 haut, 0xFFFFFFFF=racine)
    col3 time(u32 bas) | (flags|ndim<<8|block<<12)(u32 haut)
    col4..6 shape[0..5] (2×u32 par colonne)
    Retourne (rows_décodées, paths, n_entrées_path_table).
    """
    import numpy
    raw = group["index"][:]
    n = raw.shape[0]
    table = []
    if "path_table" in group:
        table = [bytes(s).decode("utf-8", "replace").rstrip("\x00")
                 for s in group["path_table"][:]]
    dec = numpy.zeros((n, 13), dtype=numpy.uint64)
    paths = []
    for i in range(n):
        r = raw[i]
        pid = int(r[2]) & M32
        parent32 = int(r[2]) >> 32
        time_v = int(r[3]) & M32
        sflags = int(r[3]) >> 32
        flags = sflags & 0xFF
        ndim = (sflags >> 8) & 0xF
        block = (sflags >> 12) & 3   # v4 : 0/1 ; v5 : 0=data_raw_*, 1=time, 2=scalar
        dec[i][0] = ndim
        for d in range(6):
            dec[i][1 + d] = (int(r[4 + d // 2]) >> (32 * (d % 2))) & M32
        dec[i][7] = time_v
        dec[i][8] = int(r[0])
        dec[i][9] = int(r[1])
        dec[i][10] = flags
        dec[i][11] = NO_PARENT if parent32 == 0xFFFFFFFF else parent32
        dec[i][12] = block
        paths.append(table[pid] if pid < len(table) else "?id{}".format(pid))
    return dec, paths, len(table)


def panzer_groups(f):
    """Groupes contenant un dataset 'index' (comme l'auto-détection de PanzerDB)."""
    groups = []
    if "index" in f:
        groups.append(("/", f))
    for key in f:
        obj = f[key]
        if isinstance(obj, h5py.Group) and "index" in obj:
            groups.append((key, obj))
    return groups


def read_paths(group, n_rows):
    if "paths" not in group:
        return [""] * n_rows
    p = group["paths"]
    n = min(n_rows, p.shape[0])
    paths = [bytes(s).decode("utf-8", "replace").rstrip("\x00") for s in p[:n]]
    paths += [""] * (n_rows - len(paths))
    return paths


def dump_group(name, group, args):
    idx = group["index"]
    if idx.ndim != 2:
        print("[{}] dataset 'index' non 2D : ignoré".format(name), file=sys.stderr)
        return
    n_rows, ncols = idx.shape
    v3 = is_v3(group)
    if v3:
        rows, paths, n_path_entries = decode_v3(group)
        ncols_decoded = 13
    else:
        rows = idx[:]  # ndarray (n_rows, ncols), uint64
        paths = read_paths(group, n_rows)
        ncols_decoded = ncols
        n_path_entries = None
    cm = colmap(ncols_decoded)  # lève ValueError si layout non prévu

    # Nombre de lignes par path : le layout « per-slice » (legacy) écrit UNE LIGNE
    # PAR SLICE (la colonne t reste 0, le temps c'est l'ordre des lignes),
    # le layout « bulk » (nouveau) met toutes les slices dans count d'une ligne.
    # slices_total = nb_lignes(path) × (count // prod(shape)).
    path_counts = {}
    for i in range(n_rows):
        if int(rows[i][cm["flags"]] & 0xF) == 0:  # data rows only
            path_counts[paths[i]] = path_counts.get(paths[i], 0) + 1

    kind_counts = {}
    selected = []
    for i in range(n_rows):
        row = rows[i]
        kind = int(row[cm["flags"]] & 0xF)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if args.kinds_set is not None and kind not in args.kinds_set:
            continue
        if args.match and args.match not in paths[i]:
            continue
        selected.append((i, paths[i], kind, row))
    if args.limit is not None:
        selected = selected[:args.limit]

    if args.raw:
        for i, path, _kind, row in selected:
            print("\t".join([str(i), path] + [str(int(v)) for v in row]))
        return

    if v3:
        ver = index_version(group)
        layout = "{} colonnes v{} packées + /path_table interné ({} chemins uniques)".format(
            idx.shape[1], ver if ver else "3/4", n_path_entries)
    elif ncols == 14:
        layout = "14 colonnes  (legacy : type@col0 et index_value@col13 non lues)"
    else:
        layout = "12 colonnes v2 (/paths aligné sur les lignes)"
    lines = [
        "### {}".format(name),
        "layout : " + layout,
    ]
    kinds_str = "  ".join(
        "{}={}".format(KIND_NAMES.get(k, "k{}".format(k)), c)
        for k, c in sorted(kind_counts.items()))
    lines.append("lignes : {}   kinds : {}".format(n_rows, kinds_str if kinds_str else "-"))
    # Timebase = dataset nommé exactement « time » (convention des tests / AL) :
    # racine du groupe (time series standalone) ou <aos-dyn>/time (AoS dynamique).
    # NB : l'association signal→timebase n'est PAS persistée dans la table
    # (le « t » d'une ligne est l'index temporel de DÉBUT de la write) ;
    # le lecteur la résout par le chemin du dictionnaire.
    tb = []
    for i in range(n_rows):
        row = rows[i]
        path = paths[i]
        if (int(row[cm["flags"]] & 0xF) == 0
                and int(row[cm["flags"]] >> 4) == 0  # F64
                and (path == "time" or path.rsplit("/", 1)[-1] == "time")):
            in_time_block = cm["block"] is not None and int(row[cm["block"]]) == 1
            tb.append("{} (row {}, {} pts{})".format(
                path, i, int(row[cm["cnt"]]),
                ", data_time_f64" if in_time_block else ""))
    if tb:
        lines.append("timebase(s) : " + "  ".join(tb))
    extents = ["{}={}".format(d, group[d].shape[0]) for d in RAW_DATASETS if d in group]
    if "path_table" in group:
        extents.append("path_table={} chemins uniques".format(group["path_table"].shape[0]))
    if "list_spans" in group:
        extents.append("list_spans={} entrées".format(group["list_spans"].shape[0] // 4))
    if extents:
        lines.append("extents : " + "  ".join(extents))
    if args.match or args.kinds or args.limit is not None:
        lines.append("affiché : {} ligne(s) sur {}".format(len(selected), n_rows))
    print("\n".join(lines))
    if not selected:
        print("(aucune ligne ne correspond aux filtres)")
        return

    table = []
    for i, path, kind, row in selected:
        flags = int(row[cm["flags"]])
        dtype = flags >> 4
        type_s = "-" if kind in (1, 2, 3) else TYPE_NAMES.get(dtype, "t{}".format(dtype))
        # « blk » : où « off » pointe. v4 range les bases de temps dans
        # data_time_f64 et v5 les scalaires par pas de temps dans data_scalar_* ;
        # v3/v2/legacy n'ont qu'un bloc F64, tout est « raw ».
        blk_col = cm["block"]
        blk_id = -1 if blk_col is None else int(row[blk_col])
        blk_s = "-" if kind in (1, 2, 3) else {0: "raw", 1: "time", 2: "scalar"}.get(blk_id,
                                                                                     "?{}".format(blk_id))
        ndim = int(row[cm["ndim"]])
        if ndim:
            shape = "[" + ",".join(
                str(int(row[cm["shape"] + j])) for j in range(min(ndim, 6))) + "]"
        else:
            shape = "-"
        parent_id = int(row[cm["parent"]])
        if args.no_parent:
            parent = "-" if parent_id == NO_PARENT else str(parent_id)
        else:
            parent = "-" if parent_id == NO_PARENT else parent_path_of(path, kind) or "-"
        # slices = nb_lignes(même path) × (count / prod(shape)) :
        #   - layout « bulk » : 1 ligne, count = slice_size × n_slices (writeDataSlicesImpl) ;
        #   - layout « per-slice » (legacy) : une ligne par slice, count = slice_size.
        # La dim. temps est toujours hors du shape. Numériques seulement (les slots de
        # chaînes ne sont pas des slices temporelles).
        slices = "-"
        if kind == 0 and dtype in (0, 1, 2):
            slice_size = 1  # ndim=0 -> slice_size=1 ; un scalaire statique a toujours count=1
            for j in range(ndim):
                slice_size *= int(row[cm["shape"] + j])
            cnt = int(row[cm["cnt"]])
            if slice_size and cnt % slice_size == 0:
                total = path_counts.get(path, 1) * (cnt // slice_size)
                if total > 1:
                    slices = str(total)
        table.append((
            str(i), path, KIND_NAMES.get(kind, "k{}".format(kind)), type_s,
            blk_s, str(ndim), shape, str(int(row[cm["time"]])), slices,
            str(int(row[cm["off"]])), str(int(row[cm["cnt"]])), parent))

    headers = ("row", "path", "kind", "type", "blk", "ndim", "shape", "t", "slices",
               "off", "count", "parent")
    widths = [max(len(h), max(len(r[j]) for r in table)) for j, h in enumerate(headers)]

    def fmt(cells):
        return "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(cells)).rstrip()

    print(fmt(headers))
    print(fmt(tuple("-" * w for w in widths)))
    for r in table:
        print(fmt(r))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="panzer_index",
        description="Affiche la table d'index brute (datasets 'index' + 'paths') "
                    "d'un fichier IMAS-HDF5 produit par le backend PanzerDB v2 (src/hdf5).")
    ap.add_argument("file", help="fichier .h5 produit par le backend v2")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="n premières lignes affichées (défaut : toutes)")
    ap.add_argument("--match", metavar="MOTIF", help="garde les lignes dont le path contient MOTIF")
    ap.add_argument("--kinds", metavar="A,B,...",
                    help="filtre sur : data,empty,aos-static,aos-dyn")
    ap.add_argument("--raw", action="store_true",
                    help="lignes brutes non décodées (tabulées), sans en-tête")
    ap.add_argument("--no-parent", action="store_true",
                    help="parent_id numérique au lieu du path parent résolu")
    args = ap.parse_args(argv)

    if args.kinds:
        kinds_set = set()
        for tok in args.kinds.split(","):
            tok = tok.strip()
            if tok not in KIND_ALIASES:
                ap.error("kind inconnu '{}' (valides : {})".format(
                    tok, ", ".join(KIND_NAMES.values())))
            kinds_set.add(KIND_ALIASES[tok])
        args.kinds_set = kinds_set
    else:
        args.kinds_set = None
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit doit être > 0")
    if args.limit is not None:
        args.limit = int(args.limit)

    try:
        f = h5py.File(args.file, "r")
    except Exception as e:
        print("ERREUR : impossible d'ouvrir '{}' : {}".format(args.file, e), file=sys.stderr)
        return 2

    groups = panzer_groups(f)
    if not groups:
        print("ERREUR : aucun groupe PanzerDB (dataset 'index') trouvé dans '{}'".format(args.file),
              file=sys.stderr)
        f.close()
        return 2

    status = 0
    for gi, (name, group) in enumerate(groups):
        if gi:
            print()
        try:
            dump_group(name, group, args)
        except ValueError as e:
            print("ERREUR :", e, file=sys.stderr)
            status = 2
    f.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
