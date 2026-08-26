# Plan — Gaps temporels pour IDS à base de temps HOMOGÈNE (`homogeneous_time = 1`)

**Date** : 26/08/2026
**Périmètre** : `src/hdf5/` (backend HDF5 v2 / PanzerDB) + `tests/hdf5_backend/`
**Précédent** : `docs/PLANS_correction_lecture_gaps.md` (cas inhomogène `homogeneous_time = 0`, déjà corrigé)

---

## 1. Problème (diagnostic confirmé)

Les tests de gaps existants (`test_gap_{closest_prev,linear,timerange}_read.cpp`,
`test_dynamic_aos_gap.cpp`) utilisent tous `homogeneous_time = 0` avec un **AoS
dynamique** + timebase `time` imbriquée. Ce cas passe (60/60).

Le cas `homogeneous_time = 1` n'est **pas** couvert. En le testant
(signal dynamique autonome à la racine, sur la timebase racine `time`, écrit
étape par étape en APPEND), les échecs confirment le non-support :

- **lecture par slice** : une slice non écrite est **absorbée** (la timeline se
  comprime) au lieu de former un trou résolu.
  - `closest sigA@0.2 → 103` (on espère 101) — la valeur de la slice i=3 glisse
    sur le temps de i=2 ;
  - `closest sigB@0.1 → 202` (on espère 200) ;
  - `linear` identiquement décalé.
- **lecture timerange** (signal à la racine) : même compression **et** lecture au-delà
  de la fin (valeurs « garbage » renvoyées).

## 2. Cause racine — écriture (CASE 2 de `writeDataSlices`)

`PanzerDB::writeDataSlicesImpl` (src/hdf5/panzerdb.cpp:1865-1876) et l'équivalent
strings (panzerdb.cpp:1986-1992) : lorsque le signal n'est **pas** dans un AoS
dynamique (cas homogène à la racine), l'index temps est calculé comme

```cpp
base_time = aos_time_counters[full_path];   // compteur PROPRE du signal
```

c.-à-d. le signal aligne sa N-ème écriture sur sa **N-ème** position, indépendamment
des timebases / des autres signaux. Si on saute `sigA` à i=2 (mais on écrit bien
`time` + `sigC`), `sigA` se comprime : sa 3ᵉ écriture atterrit sur `time_index` 2 au
lieu de 3. → Pas de trou résolvable, timeline compactée.

En comparaison, le CASE 1 (AoS dynamique, panzerdb.cpp:1843-1866) ancre
correctement sur l'itération du AoS dynamique / le compteur de la timebase :
`base_time = max(dynamic_aos_current_iteration, signal_next_time)`, avec
alignement sur le compteur de la timebase. C'est pourquoi le cas inhomogène passe
et le cas homogène échoue.

**Le correctif doit rendre le CASE 2 cohérent avec la timebase** : un signal
écrivé en APPEND doit occuper la slice correspondant à la position courante de la
timebase de référence (compteur de la timebase racine / du AoS dynamique), pas
simplement sa N-ème écriture.

## 3. Tests de reproduction (écrits, à conserver vert)

- `tests/hdf5_backend/test_gap_homog_slice_read.cpp`
  `tests/hdf5_backend/test_gap_homog_timerange_read.cpp`

Pattern : `homogeneous_time = 1`, signal dynamique autonome à la racine sur la
timebase racine `time`, écrit en 5 APPEND (i=0..4) :
- `sigA = 100 + i`, **trou à i=2**
- `sigB = 200 + i`, **trous à i=1 et i=3**
- `sigC = 300 + i`, plet

Espérances de lecture (slices présentes = valeurs aux indices non sautés) :

| Lecture                       | Espérance |
|-------------------------------|-----------|
| closest `sigA@0.2` (trou)     | 101 (slice i=1) |
| closest `sigB@0.1` (trou)     | 200 (slice i=0) |
| closest `sigB@0.3` (trou)     | 202 (slice i=2) |
| previous `sigA@0.2` (trou)    | 101 |
| previous `sigB@0.3` (trou)    | 202 |
| linear `sigA@0.2` (trou)      | 102 (moyenne 101/103) |
| linear `sigB@0.1` (trou)      | 201 (moyenne 200/202) |
| linear `sigB@0.3` (trou)      | 203 (moyenne 202/204) |
| linear `sigC@0.2` (plet)      | 302 |
| timerange closest `sigA`      | [100, 101, 101, 103, 104] (trou → last≤) |
| timerange linear  `sigA`      | [100, 101, 102, 103, 104] |
| `sig_never` (jamais écrit)    | non disponible (retour 0) |

> Les espérances exactes (moyennes vs last-≤ pour le timerange) seront fixées à la
> lecture de la sémantique ciblée du timerange ; l'objectif est d'abord que la
> slice trouée renvoie une valeur **d'une slice existante** et non un décalage.

## 4. Piste de correctif (à affiner en implémentation)

1. **Écriture homogène** : dans `writeDataSlicesImpl`, pour un signal dynamique
   autonome (pas dans un AoS dynamique), déterminer la timebase de référence
   (timebase racine `time` via la property `homogeneous_time`, ou la timebase
   explicite passée dans `writeDataSlices`), et calculer `base_time` à partir du
   compteur de cette timebase (position courante de l'append), pas du compteur
   propre du signal. Garder `max(...)` avec la borne connue du signal pour les
   writes chunkés (`chunk = plusieurs slices d'un coup`).
2. **Lecture timerange** : vérifier que la résolution par slice
   (`nearestAvailableSliceIndex` + `readInterpolatedData`) est bien invoquée pour
   un signal autonome (aujourd'hui le temps vient de la timebase globale) ; corriger
   le chemin timerange si il lit via `read_dataset_globally` puis compresse.
3. **Lecture slice** : déjà routée via `readInterpolatedData`/`nearestAvailableSliceIndex`
   — une fois l'écriture corrigée, le trou devient résolvable et ces lectures
   devraient passer sans modification du reader.

## 5. Validation

- `ctest` complet (≥62 tests) sur 3 runs consécutifs → 100 % vert.
- Mettre à jour `docs/REPORT_correction_lecture_gaps.md` (nouvelle section
  homogeneous_time=1 : diagnostic, correctif, résultats).
- Aucun debug mort ni `printf` residual (grep `printf|DEBUG_PRINT` actif).

## 6. Points de vigilance

- Ne pas casser le cas inhomogène (AoS dynamique) déjà vert : le patch doit être
  neutre sur `dynamic_aos_path` non vide.
- `aos_time_counters` est initialisé dans `restoreTimeContext` (panzerdb.cpp:637-694)
  en mode READ/APPEND : s'assurer que le compteur de la timebase y est présent.
- Les writes chunkés (`n_slices > 1`) doivent continuer à fonctionner (magnetics,
  profiles) : conserver la composante `max(signal_next_time, ...)` de base_time.
- Le signal « jamais écrit » doit rester **non disponible** (retour 0), pas une
  erreur (exceptr).
