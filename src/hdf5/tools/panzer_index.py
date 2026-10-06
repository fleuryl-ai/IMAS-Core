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
  - layout 12 colonnes (actuel) : ndim | shape[6] | time | off | count | flags | parent
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
RAW_DATASETS = ("data_raw_f64", "data_raw_i32", "data_raw_c128", "data_raw_str")


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
    """Indices des colonnes décodées ; miroir de ``PanzerDB::getLeaves()``."""
    if ncols == 14:  # legacy : "type" en tête + "index_value" en queue (jamais relues)
        return dict(ndim=1, shape=2, time=8, off=9, cnt=10, flags=11, parent=12)
    if ncols == 12:  # actuel : ndim | shape[6] | time | offset | count | flags | parent_id
        return dict(ndim=0, shape=1, time=7, off=8, cnt=9, flags=10, parent=11)
    raise ValueError(
        "dataset 'index' à {} colonnes : layout inconnu (attendu 12 ou 14)".format(ncols))


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
    cm = colmap(ncols)  # lève ValueError si layout non prévu

    rows = idx[:]  # ndarray (n_rows, ncols), uint64
    paths = read_paths(group, n_rows)

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

    lines = [
        "### {}".format(name),
        "layout : {} colonnes{}".format(ncols, "  (legacy : type@col0 et index_value@col13 non lues)" if ncols == 14 else ""),
    ]
    kinds_str = "  ".join(
        "{}={}".format(KIND_NAMES.get(k, "k{}".format(k)), c)
        for k, c in sorted(kind_counts.items()))
    lines.append("lignes : {}   kinds : {}".format(n_rows, kinds_str if kinds_str else "-"))
    extents = ["{}={}".format(d, group[d].shape[0]) for d in RAW_DATASETS if d in group]
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
        table.append((
            str(i), path, KIND_NAMES.get(kind, "k{}".format(kind)), type_s,
            str(ndim), shape, str(int(row[cm["time"]])),
            str(int(row[cm["off"]])), str(int(row[cm["cnt"]])), parent))

    headers = ("row", "path", "kind", "type", "ndim", "shape", "t", "off", "count", "parent")
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
