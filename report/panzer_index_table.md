# PanzerDB (backend HDF5 v2) — La table d'index et l'architecture de stockage

**Document technique de référence — backend v2 (IMAS-Core)**
Date : 27/08/2026
État : production (suite de tests 62/62, 3 runs consécutifs)

Ce document décrit de manière exhaustive la table d'index du backend v2
(`PanzerDB`) : spécification exacte des colonnes, capacités maximales,
exemples concrets de stockage, algorithmes de lecture/écriture avec
complexités, gestion des caches, et arguments d'ingénierie pour une
présentation à des clients de type ITER.

---

## 1. Vue d'ensemble et layout du fichier

PanzerDB ne stocke **pas** une hiérarchie HDF5 classique (groupes/datasets
par nœud IMAS). Le fichier est composé de **7 datasets à apposition**
(append-only, extensibles à `H5S_UNLIMITED`) :

| Dataset | Type HDF5 | Dims | Rôle |
|---|---|---|---|
| `index` | `H5T_STD_U64LE` | **N × 14** | Table d'index : métadonnées de chaque feuille/méta-nœud |
| `paths` | `S1(256)` null-pad, UTF-8 | N | Chemin complet de la feuille (colonne « path ») |
| `parent_paths` | `S1(256)` null-pad, UTF-8 | N | Chemin du parent (colonne « parent_path ») |
| `data_raw_f64` | `IEEE_F64LE` | M₁ | Flux de tous les doubles (8 o/él.) |
| `data_raw_i32` | `STD_I32LE` | M₂ | Flux de tous les entiers 32 bits (4 o/él.) |
| `data_raw_c128` | array 2×f64 | M₃ | Flux de tous les complexes 128 bits (16 o/él.) |
| `data_raw_str` | `C_S1` variable, UTF-8 | M₄ | Flux de toutes les chaînes |

**Philosophie**

- *Colonne par type* : les données de même type sont concaténées dans un
  unique dataset 1D → écritures 100 % séquentielles, amicales
  Lustre/parallèle.
- *Émulation AoS par chemins* : la structure IMAS (array of structures,
  imbriquée, statique ou dynamique) est reconstituée par des chemins
  littéraux (`profiles_1d/2/neutral/0/temperature`), portés par la table
  `index`. Aucun groupe HDF5 n'est créé : zéro metadata bloat, zéro
  restructuration, ajout pur append.
- *Temps implicite* : la dimension temporelle d'une feuille existe seulement
  dans la table d'index (`time_index` + `count`), pas dans la forme du
  dataset brut. C'est ce qui permet de traiter des signaux qui ont des
  **gaps** (lignes absentes) de façon native.

Propriétés de création (`createOptimizedDataset`, `init`) : chunking
configuré par *usage hint*, compression GZIP (deflate niveau 1) + shuffle,
`FILL_TIME_NEVER`, `ALLOC_TIME_INCR`, FAPL avec alignment 4 Mo (stripe
Lustre), méta-blocs 2 Mo, cache de chunks 64 Mo et 10 007 slots.

---

## 2. La table d'index : spécification exacte

La table (dataset `index`) est un tableau **N lignes × 14 colonnes**, chaque
colonne étant un `uint64` (little-endian), soit **112 octets par ligne**.
N n'est borné que par `H5S_UNLIMITED` (2⁶⁴-1 lignes). Les lignes sont
jamais modifiées ni supprimées : seule une extension du dataset est
réalisée au flush (`H5Dset_extent` + `H5Dwrite`).

### 2.1 Colonnes (numérotation exacte du code)

| Col | Nom | Type | Sens |
|---|---|---|---|
| 0 | `type` | u64 | **Réservé** (toujours 0 en écriture). Le type effectif est dans les bits hauts de `flags`. |
| 1 | `ndim` | u64 | Nombre de dimensions spatiales (0…6). |
| 2…7 | `shape[0..5]` | u64 | Extentions de la **feuille** (une seule slice temporelle). Dimensions inutilisées = 0. |
| 8 | `time_index` | u64 | **Indice de la première slice temporelle** de cette ligne. Static = 0. |
| 9 | `offset` | u64 | **Offset (en éléments)** dans le dataset typé (`data_raw_*`) du premier élément écrit par cette ligne. |
| 10 | `count` | u64 | Nombre d'éléments écrits par cette ligne (`slice_volume × n_slices`). |
| 11 | `flags` | u64 | Bits 0-3 : rôle de la ligne — `0` = donnée, `1` = donnée vide, `2` = méta-nœud AoS **statique**, `3` = méta-nœud AoS **dynamique**. Bits hauts (>>4) : type — `0` f64, `1` i32, `2` c128, `3` string, `4` list de strings. |
| 12…13 | `—` | u64 | **Réservées** (toujours 0). Marge d'extension sans changer le format. |

Le nombre de slices temporelles portées par une ligne se déduit :
`n_slices = count / (shape[0]·shape[1]·… )` (voir `isTimeInLeaf`).

Les chemins sont stockés dans deux datasets séparés (une valeur par ligne,
256 octets null-pad, UTF-8, `strncpy` borné) :

| Dataset | Contenu pour la ligne `paths[i]` |
|---|---|
| `paths[i]` | Chemin complet, ex. `profiles_1d/2/neutral/0/temperature` |
| `parent_paths[i]` | Chemin du **parent immédiat sans le nom de la feuille**, ex. `profiles_1d/2/neutral/0` ; `""` si racine |

`parent_paths` permet des requêtes « enfants de X » en **un** accès hash
(`parent_lookup`) sans parser de chaînes.

> Nom de niveau AoS contenant un `/` IMAS (segments composés du modèle) :
> remplacé par `&` à la sérialisation (côté écriture `HDF5Writer_v2`, côté
> lecture `buildFullPath`), et reconverti au décodage. Ex. : un segment
> `constraints/x_point` est stocké littéralement comme
> `constraints&x_point` dans `paths` / `parent_paths`.

### 2.2 Rappels d'ordre : une ligne ≠ un dataset IMAS

- Un **signal dynamique** écrit « par session » peut produire **plusieurs
  lignes** de même `path` (une par bloc de slices — notamment en APPEND —,
  chacune avec son `time_index` de départ et son `offset` ; pour les
  **chaînes**, c'est toujours une ligne par slice, cf. §7.3).
- Une **méta-ligne AoS** (flags 2/3) porte le **déclaré** dans `shape[0]`
  (taille du tableau de structures) ; elle ne porte pas de données
  (`offset=count=0`).
- En mode WRITE, chaque `beginArray` insère sa méta-ligne ; en APPEND,
  l'insertion est dédupliquée via `leaf_lookup` (pas de doublon).

### 2.3 Buffers RAM d'écriture et politique de flush

| Buffer | Croissance | Contenu |
|---|---|---|
| `index_buffer` | ×2 (`BUFFER_GROWTH_FACTOR`), capacité initiale `index_chunk_rows × 14` u64 | Lignes 14×u64 |
| `data_buffer_f64` / `_i32` / `_c128` | ×2 | Éléments bruts |
| `data_buffer_str` | ×2 | `std::string` |
| `paths_buffer` / `parent_paths_buffer` | ×2 | 256 B par ligne |

`append_index_row` : 2 `strncpy` + 14 `memcpy` u64 → **O(1)**, aucune
allocation en régime permanent. `flush()` (fin de session, ou `close()`
) : 7 extensions de dataset + 7 écritures hyperslab → **une seule**
transaction HDF5 par buffer, donc une I/O par flush quelle que soit la
taille des données. Le seul flush déclenché est l'explicite
(`HDF5Writer_v2::endAction` / destructeur) — l'ancien seuil mémoire
(`AUTO_FLUSH_THRESHOLD`) n'était jamais appelé et a été supprimé le
27/08/2026 (cf. `panzerdb_removing_dead_codes.md`).

---

## 3. Capacités maximales (colonnes `uint64`)

Les colonnes `time_index`, `offset`, `count`, `shape[i]` sont des
`uint64` : plafond **2⁶⁴ − 1** ≈ **1,8447 × 10¹⁹**.

| Grandeur | Plafond |
|---|---|
| Éléments d'un même fichier de double | 2⁶⁴−1 él. × 8 o = **≈ 1,48 × 10²⁰ B ≈ 147,6 EB** |
| Éléments de 32 bits | ≈ **73,8 EB** |
| Éléments complexes 128 | ≈ **295,2 EB** |
| Pas de temps d'un signal (`time_index`) | 1,84 × 10¹⁹ pas → à 1 µs/pas : **≈ 584 000 ans** ; à 1 ms/pas : ≈ 584 ans |
| Dimensions d'une feuille | **6** (`shape[6]`, `ndim ≤ 6`), chaque dimension ≤ 2⁶⁴−1 |
| Lignes d'index | 2⁶⁴−1 ≈ 1,84 × 10¹⁹ (≈ 2,07 zetta-octets de table brute) |

Ces chiffres sont des **plafonds théoriques du format** ; la contrainte
pratique reste la capacité disque et les limites internes HDF5. En
pratique, un fichier IMAS ITER de plusieurs To est à des années-lumière de
ces plafonds : le format est structurellement « à preuve de croissance ».

> Le chemin (`paths`) n'est pas borné par le u64 mais par `PATH_MAX_LEN` =
> 256 octets (255 chars utiles) — largement au-delà du chemin IMAS le plus
> long du modèle (≈ 60 chars).

---

## 4. Exemples concrets complets

Contexte : écriture globale (`GLOBAL_OP`, mode WRITE).

- L'exemple **4.1** est un dataobject à base de temps homogène
  (`ids_properties&homogeneous_time = 1`), avec
  `time = [0.0, 0.1, 0.2, 0.3]` (4 valeurs).
- Les exemples **4.2** et **4.3** sont des dataobjects **globaux**
  (aucun `homogeneous_time`, aucune timebase).

Les tables ci-dessous montrent **toutes** les lignes de `/index`
respectives, dans l'ordre d'append réel (ordre d'appels), avec `paths` et
`parent_paths`.

Valeurs utilisées :

- `profiles_1d/neutral/temperature` : `temperature(t, x) = 10·t + x`,
  `x ∈ {0,1,2}` → t0 `[0,1,2]`, t1 `[10,11,12]`, t2 `[20,21,22]`,
  t3 `[30,31,32]`.
- `flux_loop/flux/data` : `data(boucle, x) = 10·boucle + x+1`, `x ∈ 0..3`.
- `rogowski_coil/position/r` : `r(bobine i, position j) = 10·i + j + 1`.

### 4.1 `profiles_1d/neutral/temperature` — AoS **dynamique** (temps), structure imbriquée, champ 1D

`profiles_1d` est une AoS dynamique (timebase `time`, à la racine du
dataobject en mode `homogeneous_time=1`), `neutral` une structure statique
(1 élément), `temperature` un champ 1D (3 points).
4 pas de temps, écriture **slice par slice** (session WRITE puis mêmes
lignes en APPEND).

**Contenu de `data_raw_f64`** (offsets en éléments) :

```
offset: 0        1        2        3   | 4   5   6  | 7   8    9  | 10  11  12 | 13  14  15
        0.0     0.1      0.2      0.3 | 0   1   2 | 10  11  12 | 20  21  22 | 30  31  32
        ← time (4 él.) →        ← t0 →   ← t1 →     ← t2 →      ← t3 →
```

**Table `/index` (lignes dans l'ordre d'append) :**

| # | `paths` | `parent_paths` | ndim | shape | time_index | offset | count | flags |
|---|---|---|---|---|---|---|---|---|
| 0 | `ids_properties&homogeneous_time`³ | `""`² | 0 | `[]` | 0 | 0 | 1 | 0 (**i32**, dans `data_raw_i32`) |
| 1 | `time` | `""` | 0 | `[]` | 0 | 0 | 4 | 0 (f64) |
| 2 | `profiles_1d` | `""` | 1 | `*[0]*`¹ | 0 | 0 | 0 | **3 (AoS dynamique)** |
| 3 | `profiles_1d/0/neutral` | `profiles_1d/0` | 1 | `[1]` | 0 | 0 | 0 | **2 (AoS statique)** |
| 4 | `profiles_1d/0/neutral/0/temperature` | `profiles_1d/0/neutral/0` | 1 | `[3]` | **0** | 4 | 3 | 0 |
| 5 | `profiles_1d/1/neutral` | `profiles_1d/1` | 1 | `[1]` | 0 | 0 | 0 | 2 |
| 6 | `profiles_1d/1/neutral/0/temperature` | `profiles_1d/1/neutral/0` | 1 | `[3]` | **1** | 7 | 3 | 0 |
| 7 | `profiles_1d/2/neutral` | `profiles_1d/2` | 1 | `[1]` | 0 | 0 | 0 | 2 |
| 8 | `profiles_1d/2/neutral/0/temperature` | `profiles_1d/2/neutral/0` | 1 | `[3]` | **2** | 10 | 3 | 0 |
| 9 | `profiles_1d/3/neutral` | `profiles_1d/3` | 1 | `[1]` | 0 | 0 | 0 | 2 |
| 10 | `profiles_1d/3/neutral/0/temperature` | `profiles_1d/3/neutral/0` | 1 | `[3]` | **3** | 13 | 3 | 0 |

¹ Pour une AoS **dynamique**, la taille n'est pas connue à la création :
`shape[0]` de la méta-ligne vaut 0. La taille effective est déduite à la
lecture : `max time_index` de **toutes** les feuilles descendantes (tout
niveau d'imbrication) + 1 (§6) — ce qui rend l'exemple ci-dessus
valable tel quel, sans champ direct nécessaire.

² Les nœuds écrits à la racine du dataobject (aucun AoS ouvert lors de
l'appel) ont `parent_paths` vide (voir `append_index_row` :
`parent_path = path_prefix`, qui vaut `""` à la racine).

³ Convention d'aplatissement : les `/` des noms multi-segments IMAS sont
remplacés par `&` à la sérialisation (côté écriture `hdf5_writer_v2.cpp:50`
`read_homogeneous_time` utilise le même nom ; côté écriture
`hdf5_writer_v2.cpp` `std::replace(...,'/','&')`), puis reconvertis au
décodage. C'est pourquoi le nœud IMAS `ids_properties/homogeneous_time`
apparaît comme un **segment unique** `ids_properties&homogeneous_time`
dans `paths`.

Lecture de `profiles_1d/2/neutral/0/temperature` : recherche hash
`leaf_lookup["profiles_1d/2/neutral/0/temperature"]` → ligne 8 →
hyperslab `data_raw_f64[10..12]` → `[20,21,22]`. **Une seule** opération
de lecture HDF5, aucune copie intermédiaire.

> Si la ligne 5 (t=1) avait été sautée au moment de l'écriture (gap), la
> timeline du signal resterait `[0,2,3]` et un accès à t=1 renverrait la
> slice la plus proche disponible (mode `closest`) — c'est exactement le
> comportement corrigé et testé par `test_gap_closest_prev_read`
> et `test_gap_homog_slice_read`.

### 4.2 `flux_loop/flux/data` — AoS **statiques**, feuille 1D

`flux_loop` (2 éléments), `flux` (1 élément), `data` champ 1D (4 points).
Écriture purement globale, `time_index` = 0 pour toutes les lignes de données.

**Contenu de `data_raw_f64` :**

```
offset: 0  1  2  3  | 4  5  6   7
        1  2  3  4  | 11 12 13 14     (boucle 0 / boucle 1)
```

**Table `/index` :**

| # | `paths` | `parent_paths` | ndim | shape | time_index | offset | count | flags |
|---|---|---|---|---|---|---|---|---|
| 0 | `flux_loop` | `""` | 1 | `[2]` | 0 | 0 | 0 | **2** |
| 1 | `flux_loop/0/flux` | `flux_loop/0` | 1 | `[1]` | 0 | 0 | 0 | **2** |
| 2 | `flux_loop/0/flux/0/data` | `flux_loop/0/flux/0` | 1 | `[4]` | 0 | 0 | 4 | 0 |
| 3 | `flux_loop/1/flux` | `flux_loop/1` | 1 | `[1]` | 0 | 0 | 0 | **2** |
| 4 | `flux_loop/1/flux/0/data` | `flux_loop/1/flux/0` | 1 | `[4]` | 0 | 4 | 4 | 0 |

Lecture globale de `flux_loop/flux/data` (tous les éléments) : l'API
construit le chemin par instance (`flux_loop/0/flux/0/data`, puis
`flux_loop/1/flux/0/data`), chaque access = 1 hash O(1) + 1 hyperslab.
Taille de `flux_loop` : `getAOSShape` parcourt les enfants (`parent_lookup`)
→ max index = 1 → **2**.

### 4.3 `rogowski_coil/position/r` — AoS statiques imbriquées, donnée **0D**

`rogowski_coil` (2), `position` (3), `r` scalaire (`ndim = 0`, `count = 1`).

**Contenu de `data_raw_f64` :**

```
offset: 0   1   2   | 3   4   5
        1   2   3   | 11  12  13     (bobine 0 : pos 0,1,2 / bobine 1 : pos 0,1,2)
```

**Table `/index` :**

| # | `paths` | `parent_paths` | ndim | shape | time_index | offset | count | flags |
|---|---|---|---|---|---|---|---|---|
| 0 | `rogowski_coil` | `""` | 1 | `[2]` | 0 | 0 | 0 | **2** |
| 1 | `rogowski_coil/0/position` | `rogowski_coil/0` | 1 | `[3]` | 0 | 0 | 0 | **2** |
| 2 | `rogowski_coil/0/position/0/r` | `rogowski_coil/0/position/0` | **0** | `[]` | 0 | 0 | **1** | 0 |
| 3 | `rogowski_coil/0/position/1/r` | `rogowski_coil/0/position/1` | 0 | `[]` | 0 | 1 | 1 | 0 |
| 4 | `rogowski_coil/0/position/2/r` | `rogowski_coil/0/position/2` | 0 | `[]` | 0 | 2 | 1 | 0 |
| 5 | `rogowski_coil/1/position` | `rogowski_coil/1` | 1 | `[3]` | 0 | 0 | 0 | **2** |
| 6 | `rogowski_coil/1/position/0/r` | `rogowski_coil/1/position/0` | 0 | `[]` | 0 | 3 | 1 | 0 |
| 7 | `rogowski_coil/1/position/1/r` | `rogowski_coil/1/position/1` | 0 | `[]` | 0 | 4 | 1 | 0 |
| 8 | `rogowski_coil/1/position/2/r` | `rogowski_coil/1/position/2` | 0 | `[]` | 0 | 5 | 1 | 0 |

Lecture scalaire : `readScalar` → hash + hyperslab d'**8 octets** sur
`data_raw_f64`. La taille de `position` est retrouvée par
`getAOSShape("rogowski_coil/0/position")` = max index enfants + 1 = 3.

---

## 5. Algorithmes de recherche de chemin et complexité

Toutes les recherches s'appuient sur le cache de index construit une fois
(`getLeaves`) et les dictionnaires associatifs. `L` désigne la longueur
du chemin (≤ 255).

### 5.1 Construction du cache (uniquement au premier appel) — `getLeaves`

1. Lecture **intégrale** de `index` (N×14 u64) + `paths` + `parent_paths`
   → O(N·256) en I/O, une seule fois.
2. Pour chaque ligne : `leaf_lookup[path].push_back(i)` et
   `parent_lookup[parent_path].push_back(i)` → O(N) insertions hash.
3. Détection des racines AoS dynamiques (flags=3) →
   `cached_dynamic_aos_roots`.
4. `Leaf.path` / `Leaf.parent_path` sont des `std::string_view` **zéro-copy**
   dans les blocs tampons → aucune allocation par feuille.

Reconstruction : **incrémentale** — si N a augmenté depuis le cache, seules
les lignes `[N_vieilles, N)` sont lues et insérées (cas APPEND sur un
fichier déjà ouvert, ou append par un autre process puis relecture).

### 5.2 Lecture slice (dynamique) — `pz_readData_by_index` /
`readInterpolatedData`

Pour un chemin cible `P` et un indice temporel `k` :

1. **Stratégie 1 — chemin direct** : `leaf_lookup.find(P)` → O(1) hash
   (coût du hash O(L)) → liste (souvent 1) des feuilles `F`.
2. Pour chaque feuille : `isTimeInLeaf(F, k)` — comparaison bornes dérivées
   (`k ∈ [F.time_index, F.time_index + F.count/slice_volume)`) → O(1).
3. Sinon **stratégie 2 — chemin générique** (AoS dynamique, index omis) :
   `leaf_lookup.find("profiles_1d/neutral/0/temperature")` → O(1).
4. Sinon **stratégie 3 — chemin substitué** :
   `leaf_lookup.find("profiles_1d/" + k + "/neutral/0/temperature")` → O(1).
5. Lecture : `readSliceDirect` = 1 `H5Dread` hyperslab sur le dataset typé,
   offset `F.offset + (k − F.time_index)·slice_volume` → **1 I/O**.

**Gaps** : `nearestAvailableSliceIndex(P, k, dir, time_basis, t)`
(panzerdb.cpp:2385) — explore les indices adjacents avec les 3 stratégies
ci-dessus jusqu'à trouver une slice existante ; complexité **O(g)** avec
g le nombre de slices manquantes autour de k (g typiquement 1-3, et chaque
test O(1)). Résolution : `closest` → plus proche en temps (égalité
→ indice le plus bas) ; `previous` → dernière slice ≤ k, sinon première ≥ k ;
`linear` → paire (inf, sup) réelles pour l'interpolation.
`readInterpolatedData` (panzerdb.cpp:3205) orchestre : résolution
inf/sup + 1 à 2 `readSliceDirect` + interpolation `DataInterpolation`.

**Lecture globale** (`time_index == -1`) : `leaf_lookup.find(P)` → k feuilles
→ tri O(k log k) → `readLeavesUnion` : si les offsets sont strictement
croissants (cas d'écriture chronologique, le plus fréquent),
**une** sélection hyperslab fusionnée (`H5Sselect_or`, blocs contigus
fusionnés) → 1 I/O ; sinon repli séquentiel (k I/O) pour préserver l'ordre.

**Recherche par temps** (`getTimeIndex`) : `leaf_lookup.find(chemin_timebase)`
→ concaténation des feuilles O(T log T) (T = nb pas) → interpolation
`DataInterpolation::getSlicesTimesIndices` → O(T) (T petit par signal).

### 5.3 Écriture

| Opération | Coût |
|---|---|
| `beginArray` (méta) | O(1) hash (dédup en APPEND via `leaf_lookup`) + append O(1) |
| `writeData` (statique) | append buffer O(1) + `append_index_row` O(1) |
| `writeDataSlices` (1 session, n slices) | 1 append O(n·slice) + 1 ligne index |
| Calcul du `base_time` (CASE 1 AoS dyn / CASE 2 standalone) | O(1) (compteurs `aos_time_counters`) |
| `synchronizeArrayStack` (changement d'instance AoS) | O(d) (d = profondeur, ≤ 5 en IMAS) |
| `flush()` | 7 I/O HDF5 quelle que soit la taille (1 par buffer) |

L'écriture est donc **O(1) par feuille** (en mémoire), et l'amortissement
sur le disque est maximal : n'importe quel volume de données se flush en
un nombre constant de transactions.

---

## 6. Taille des tableaux de structures (AoS)

**Statiques** (`getAOSShape`, panzerdb.cpp:2219) :

1. `leaf_lookup.find(niveau)` → méta-ligne (flags=2) → O(1).
2. `parent_lookup.find(niveau)` → feuilles dont le parent est `niveau` → O(c).
3. Si une de ces feuilles est elle-même une AoS instanciée (`niveau/<idx>/…`),
   le max index des instances est utilisé → **taille = max index + 1**.
4. Sinon (cas le plus courant, y compris AoS encore vide) : taille
   **déclarée** dans `shape[0]` de la méta-ligne (cf. §4.2 : `shape=[2]` → 2).

**Dynamiques** (flags=3) :

1. Une fois par session, au chargement du cache (`getLeaves`) :
   pour chaque racine dynamique, on calcule le **max `time_index` de toutes
   les feuilles de donnée descendantes, quelle que soit leur profondeur**
   (map `max_time_at_dynamic_root`, `rebuildDynamicRootTimeIndex`), O(feux ×
   racines) sur une passe unique.
2. Lookup de la taille : `parent_lookup` est remplacé par un accès map O(1)
   → **Taille = max `time_index` (toute profondeur) + 1**. Le numéro de
   slice *est* l'instance.
3. Côté écriture (sans relecture du fichier) :
   `aos_time_counters[niveau]` → O(1) ; exposé par
   `getDynamicAOSSize` (panzerdb.cpp:2309).
4. En mode APPEND, ces compteurs sont **restaurés** par
   `restoreTimeContext` (panzerdb.cpp:637) : O(N) une fois à l'ouverture,
   à partir de la table seule (aucune donnée lue).
5. Le parcours ancien (enfants **directs** uniquement) est volontairement
   écarté car il renvoyait 0 pour une AoS dynamique qui ne contient que des
   sous-structures statiques (ex. IMAS réel `time_slice/ggd/theta/values`
   ou `profiles_1d/e_field_n_phi/{plus,minus,parallel}`) — corrigée, test
   `test_dynamic_aos_nested_only`.

**Cas d'AoS imbriquées** : une AoS statique dans une AoS dynamique
(`time_slice/k/ggd/theta/values`) porte un chemin **par instance** k : la
taille de `time_slice` est obtenue par le `max time_index` de **toutes**
ses feuilles descendantes (profondeur quelconque), et la taille de `ggd` /
`theta` par la route statique (§6 ci-dessus) — la dérive « feuille directe
seule » qui limitait ce calcul est corrigée au §9.2.

---

## 7. Ajout d'une donnée : ajout **global** vs ajout **par slice**

Chaîne d'appels complète (AL → writer v2 → PanzerDB), avec les méthodes
réelles du code (`hdf5_writer_v2.cpp`, `panzerdb.cpp`).

### 7.1 Ajout global (`GLOBAL_OP` → `PanzerDB::OpenMode::WRITE`)

Commun aux 3 exemples (l'ordre d'appels AL est identique) :

| Étape | Méthode | Effet côté PanzerDB |
|---|---|---|
| Ouverture | `HDF5Backend::openPulse` → `setWriteStrategy(GLOBAL_OP)` | `PanzerDB(gid, WRITE)` → `init` : creation des 7 datasets + `configureChunking("time_series")` |
| Début op | `beginAction(opCtx)` | — |
| `ids_properties/homogeneous_time` | `writeData` → **writer_v2 CASE 3** (statique pur) → `writeData<int32_t>` → `writeDataImpl` | ligne index (ndim=0) + élément dans `data_buffer_i32` |
| `time` (si présent) | `writeData` → **writer_v2 CASE 1** (timebase explicite, hors AoS dyn) → `writeDataSlices(double)` → `writeDataSlicesImpl` **CASE 2** (standalone) | 1 ligne (time_index=0, count=N) ; compteur `aos_time_counters["time"]=N` |
| Pour chaque AoS (statique) | `beginArraystructAction(ctx, size)` → `synchronizeArrayStack` → `beginArray(name, size)` → `beginArray(ArrayLevel&)` | méta-ligne (flags=2, shape=[size]) ; `array_stack.push_back` ; `path_prefix` mis à jour |
| Pour chaque instance | `nextIndex` → `incrementArrayIndex` | `current_index++` |
| Écriture des données | `write_ND_Data` → `synchronizeArrayStack` → `isInsideDynamicAOS` | ex. 4.2/4.3 : **writer_v2 CASE 3** (statique) → `writeData<double>` → `writeDataImpl` (time_index=0). ex. 4.1 (dans AoS dyn) : **writer_v2 CASE 2** (n_slices=1) → `writeDataSlices<double>` → `writeDataSlicesImpl` **CASE 1** « data in dynamic AOS » (base_time dérivé de l'itération AoS, panzerdb.cpp:1838) |
| Métadonnées IMAS (units, …) | `writeMetaData` → `writeData<const char*>` → `writeDataSlices(char**)` | ligne par `schema@attribut` ; dédup via `written_metadata_schema_paths` |
| Fin AoS | `endAction(ctx)` → `endArray` | popup stack, restauration `path_prefix` |
| Fin op | `endAction(opCtx)` → `flush()` + `close()` | 7 extensions+écritures HDF5 (sections 2.3) |

**Exemple 4.1 (AoS dynamique, écrit en une session de 4 slices)** :
à chaque itération t, `writeDataSlicesImpl` (cas 1 — AoS dynamique) calcule
`base_time = max(itération AoS, fin du signal)` puis
`append_index_row("profiles_1d/t/neutral/0/temperature", …, time_index=t,
offset=courant, count=3)`. Résultat : les 4 lignes 4/6/8/10 de la table
(§4.1) sans aucun dataset séparé.

### 7.2 Ajout par slice (`SLICE_OP` → `PanzerDB::OpenMode::APPEND`)

| Étape | Méthode | Effet |
|---|---|---|
| Ouverture | `openPulse(APPEND)` → `setWriteStrategy(SLICE_OP)` | `PanzerDB(gid, APPEND)` → `init` : ouverture des datasets existants, `readChunkingConfig`, `updateDiskSizes`, **`restoreTimeContext`** (reconstruit `aos_time_counters` et les racines dyn depuis la table) |
| `time` (+1 valeur) | `writeData` → CASE 1 → `writeDataSlicesImpl` **CASE 2** | `base_time = aos_time_counters["time"]` (=4) → ligne `time_index=4, offset=16, count=1` |
| `profiles_1d` | `synchronizeArrayStack(["profiles_1d"],[4])` → `beginArray(name,timebase)` (charge dynamique) | méta-ligne **déjà existante** → `leaf_lookup` : aucune nouvelle ligne (dédup `panzerdb.cpp:1331`) ; `current_index=4` ; `invalidateDynamicAOSCache` |
| `neutral` | `synchronizeArrayStack` → `beginArray("neutral",1)` | **nouvelle** méta-ligne `profiles_1d/4/neutral` (flags=2, shape=[1]) — chemin inédit pour cette instance |
| `temperature` | `write_ND_Data` → CAS 2 (`isInsideDynamicAOS` vrai, `n_slices=1`) → `writeDataSlices` → `writeDataSlicesImpl` **CAS 1** | `base_time = max(itération AoS = 4, fin du signal = 4)` ; l'ancrage timebase du CAS 1 recherche l'entrée ``profiles_1d/<nom_timebase>`` dans `aos_time_counters` — pour du `homogeneous_time` (time à la racine, clé `"time"`) il ne s'applique pas, c'est la synchronisation d'indexation qui garantit l'alignement → ligne `time_index=4, offset=17, count=3` |
| Fin | `endArray`×2 → `flush()` + `close()` | table +3 lignes (`time`, méta `neutral`, `temperature`) ; `data_raw_f64` +4 éléments (1 pour `time` à l'offset 16, 3 pour la slice aux offsets 17-19) |

Le fichier final est **indistinguable** d'une écriture en une session :
la table porte les slices 0..4, chacune adressée par son `time_index`.

### 7.3 Cas spécifiques

- **§4.2 / §4.3 (AoS purement statiques)** : seul l'ajout global existe ;
  en APPEND, `write_ND_Data` exige un timebase (`hdf5_writer_v2.cpp:334`) —
  un Ajout slice sur `flux_loop/…` n'est donc **pas applicable**, par
  conception IMAS (ces dataobjects sont globaux dans le modèle).
- **Écriture par bloc** (n > 1 slices en une fois) : même méthode,
  `count = n·slice`, `time_index` continu → les lectures par slice restent
  O(1) grâce à `isTimeInLeaf` + `readSliceDirect`.
- **Listes de chaînes** : `writeDataSlices(char**)` insère **une ligne par
  slice** (chaque sous-signal string étant variable en taille) ; les
  numériques utilisent une ligne par bloc — les deux modes coexistent et
  sont lisibles de la même façon.

---

## 8. Caches mis en place

### 8.1 Caches de lecture (côté panzerdb.h:252-282)

| Cache | Contenu | Rôle | Invalidation / coût |
|---|---|---|---|
| `cached_leaves` | `vector<Leaf>` (path/parent en `string_view`) | Représentation RAM **complète** de la table ; base de toutes les recherches | Recharger intégral si invalidé ; `leaves_cache_valid=false` après flush |
| `leaf_lookup` | `map<string_view, vector<size_t>>` | « chemin → feuilles » : le cœur des O(1) | Replié avec `cached_leaves` |
| `parent_lookup` | `map<string_view, vector<size_t>>` | « parent → enfants » : taille AoS, enfants directs, sans parsing | idem |
| `cached_paths_blocks` / `cached_parent_paths_blocks` | Blocs 256 B/ligne | Support mémoire des `string_view` (zéro-copy) | idem |
| `cached_dynamic_aos_roots` | `vector<string>` | Racines AoS dynamiques (flags=3) : stratégies générique/substituée sans parcours N | idem |
| `max_time_at_dynamic_root` | `map<string_view,uint64>` | max `time_index` descendant par racine dynamique ; sert `getAOSShape` | `rebuildDynamicRootTimeIndex`, idem |
| `scratch_str` | Buffer de travail | Éviter malloc/free des `std::string` dans `pz_readStringData_by_index` | — |
| `disk_size_*` | Taille courante des `data_raw_*` en RAM | Offset d'écriture sans `H5Dget_space` | Mis à jour au flush (WRITE), `updateDiskSizes` (APPEND) |

### 8.2 Caches d'écriture

| Cache | Rôle |
|---|---|
| `aos_time_counters` | `chemin → prochaine slice libre` ; O(1) pour base_time, détection de gaps, restauration APPEND (`restoreTimeContext`) |
| `path_prefix` | Préfixe de chemin courant, maintenu directement par `beginArray(ArrayLevel)`/`endArray` — évite le re-joignage à chaque écriture |
| `cached_dynamic_aos_path` + `dynamic_aos_path_valid` | Chemin de l'AoS dynamique courant, invalidé par `invalidateDynamicAOSCache()` dans les 3 `beginArray*` et `endArray` (une invalidation manquante dans la signature temporelle est à l'origine de la régression « gaps homogènes », corrigée en 07/2026) |
| `written_metadata_schema_paths` | Dédup des métadonnées IMAS (`schema@attribut`) : une seule écriture par attribut, y compris en APPEND |
| Buffers RAM (section 2.3) | Amortissement I/O (le plus gros « cache » : tout n'est écrit qu'au flush) |

### 8.3 Caches HDF5 (FAPL/DAPL)

- **Chunk cache** : 64 Mo / 10 007 slots, taux de rétention 0,75
  (`H5Pset_chunk_cache`) — les chunks fréquents (temps, derniers slices)
  restent en RAM.
- **Méta-blocs** 2 Mo : les métadonnées HDF5 sont agrégées → moins d'IOPS.
- **Alignment** 4 Mo au-delà de 4 Ko d'objet : alignement sur les stripes
  Lustre des systèmes de fichiers parallèles.
- **Compression** deflate-1 + shuffle sur les flux numériques : vitesse de
  compression proche du brut, gain 30-60 % sur données physiques réelles.
- **Chunks** adaptés à l'usage (`configureChunking` : `time_series`,
  `bulk_write`, `interactive`, `array_of_structures`).

### 8.4 Cohérence multi-process

Chaque process écrit dans sa propre pile de buffers + sa propre table
d'index (aucun verrouillage interne) : HDF5 sérialise les accès au disque.
Les lecteurs `APPEND`/`READ` rechargent **incrémentalement** les nouvelles
lignes au prochain `getLeaves` (détecté par N différent) — aucun besoin de
re-lire tout le fichier.

---

## 9. Points clés pour une présentation client (ITER)

### 9.1 Ce que le format apporte

1. **Performance d'écriture « streaming »** : append-only, sans
   fragmentation, sans restructuration, écritures I/O-saturées sur Lustre
   (alignment 4 Mo). L'écriture d'un tir complet (To de données) se
   fait en une passe séquentielle.
2. **Lecture temporelle minimale** : une slice = 1 `H5Dread` hyperslab,
   sans intermédiaires, quel que soit le nombre de slices existantes pour
   ce signal. La table d'index complète tient en mémoire (112 o/ligne →
   ~2,4 Mo pour 20 000 feuilles), donc la latence d'adresse est
   indépendante de la taille du fichier.
3. **Support natif des gaps** : les slices manquantes d'un signal ne
   compromettent pas les autres. Résolution `closest`/`previous`/
   `linear` documentée et testée (incl. `homogeneous_time=1`,
   `docs/REPORT_correction_lecture_gaps.md`). Un signal peut être échantillonné
   plus ou moins denses que la timebase de référence sans perte de
   cohérence temporelle.
4. **Continuité des sessions** : plusieurs écritures SLICE
   (diagnostics différents, phases) se combinent dans le même fichier,
   même avec des nœuds IMAS différents (chaque dataobject est un groupe
   HDF5 autonome contenant ses 7 datasets). La table d'index n'est jamais
   réécrite, seulement prolongée.
5. **Indépendance de modèle** : le backend ne connaît que les chemins IMAS
   (aucun type IMAS en dur) → aucune modification du backend n'est
   nécessaire pour un upgrade de version de modèle IMAS (v4.1 → v5,
   …) : seuls les chemins/attributs changent.
6. **Métadonnées IMAS embarquées** : attributs (`units`, `formulae`, …)
   écrits une fois par dataobject, lisibles par le lecteur sans
   configuration externe (conforme au standard IMAS d'auto-descriptivité).
7. **Plafonds** (§3) : ~150 EB de données doubles par fichier, 6
   dimensions, 1,8 × 10¹⁹ pas de temps — le format ne sera jamais la
   contrainte, même pour des tir ITER multi-décennaux archivés en un
   dataobject unique.

### 9.2 Points de vigilance à mentionner

| Limite | Détail |
|---|---|
| Pas de mises à jour in-place d'un élément | Le modèle est append-only ; une « modification » = nouvelle session. C'est le cas pour tout backend IMAS v2. |
| Chemin max 255 chars | Limite matérielle du format ; aucun cas IMAS connu au-delà. |
| 6 dimensions max par feuille | Le modèle IMAS n'exige pas plus de 3. |
| Taille AoS statique inférée | Déduite des enfants (ou du déclarant) ; si un AoS est créé **et** vide (`size>0` mais 0 écriture), il n'apparaît pas dans la table ni côté enfants. |
| Un seul AoS dynamique par niveau | Contrainte documentée et vérifiée (`beginArray` lève une exception si deux AoS dynamique sont imbriqués) — cohérent avec le modèle IMAS (un seul `timed` par chaîne). |
| ~~`getAOSShape` sur une AoS dynamique « sans champs directs »~~ | **Corrigé** : la taille d'une AoS dynamique est maintenant déduite du `max time_index` de **toutes** ses feuilles descendantes (profondeur quelconque), calculée une unique fois par session dans `rebuildDynamicRootTimeIndex` (panzerdb.cpp) et lue en O(1) dans `getAOSShape`. Précédemment on ne regardait que les enfants *directs* : la taille renvoyait 0 et la lecture globale était tronquée à zéro pour les 24 AoS dynamiques du modèle IMAS qui ne contiennent que des sous-structures — ex. `time_slice/ggd/theta/values`, `profiles_1d/e_field_n_phi/{plus,minus,parallel}`, `temporary/dynamic_float1d`… Test de régression dédié : `test_dynamic_aos_nested_only` (échoue sans le correctif, passe avec). |

### 9.3 Glossaire rapide

| Terme | Définition |
|---|---|
| **Feuille / Leaf** | Nœud terminal de la table d'index : soit une donnée (flags 0/1), soit une méta-ligne AoS (flags 2/3). |
| **AoS statique** | Tableau de structures à taille fixe déclarée au moment de `beginArray` (flux_loop, position, …). |
| **AoS dynamique** | Tableau de structures qui grandit dans le temps (profiles_1d, …) ; l'indice spatial **est** l'indice temporel. |
| **Timebase** | Champ 1D (ex `time`) dont les valeurs (t) indexent les slices ; stocké comme tout autre signal. |
| **Slice** | Échantillon temporel d'un signal = `count/slice_volume` éléments consécutifs dans `data_raw_*`. |
| **Gap** | Slice absente d'un signal ; résolue à la lecture par `closest`/`previous`/`linear`. |
| **Flags** | Bits 0-3 = rôle, bits 4+ = type de la feuille. |
| **Buffer** | RAM d'écriture, flushée en 1 I/O au `close()`. |
| **`aos_time_counters`** | Registre (RAM) `chemin → prochaine slice` ; source de vérité pour l'ancrage de gaps. |

---

## 10. Annexe — Références code

| Élément | Emplacement |
|---|---|
| Structure `Leaf`, 14 colonnes | `src/hdf5/panzerdb.h:166` (doc) ; `append_index_row` `panzerdb.cpp:2070` |
| Création des 7 datasets | `panzerdb.cpp:158-233` ; `createOptimizedDataset` `:464` |
| Buffers & seuils de flush | `panzerdb.h:219-226` ; `flush()` `panzerdb.cpp:714` |
| `getLeaves` (cache + lookups) | `panzerdb.cpp:1349` |
| `getAOSShape` (tailles) | `panzerdb.cpp:2250` (dynamique : map `max_time_at_dynamic_root`) ; `rebuildDynamicRootTimeIndex` (panzerdb.h / .cpp, à la construction du cache) ; `getDynamicAOSSize` `:2309` |
| Lecture slice direct | `readSliceDirect` `panzerdb.cpp:992` ; `pz_readData_by_index` `:2481` |
| Interpolation + gaps | `readInterpolatedData` `:3205` ; `nearestAvailableSliceIndex` `:2385` |
| Lecture batch | `readLeavesUnion` `:1146` |
| Écriture slices | `writeDataSlicesImpl` `:1814` (CASE 1 `:1838`, CASE 2 `:1871`) ; strings `:1961` |
| Écriture statique | `writeDataImpl` `:1624` |
| Déclaration AoS | `beginArray(name,size)` `:1252` ; `beginArray(name,timebase)` `:1266` ; core `:1311` ; `endArray` `:2190` |
| Restauration APPEND | `restoreTimeContext` `:637` |
| Writer AL→PanzerDB | `hdf5_writer_v2.cpp` (`beginWriteArraystructAction` `:79`, `write_ND_Data` `:192`, `endAction` `:577`) |
| Lecteur AL→PanzerDB | `slice_read_strategy.cpp` (`read_ND_Data` `:83`) ; `iread_strategy.h:542` (`buildFullPath`) |
| Correctif gaps (rapport) | `docs/REPORT_correction_lecture_gaps.md` §7 (homogène) ; `docs/PLANS_gap_homog_timebase.md` |
| Tests de référence | `tests/hdf5_backend/test_profiles_1d_dynamic_signal_1d.cpp`, `test_gap_closest_prev_read.cpp`, `test_gap_homog_slice_read.cpp`, `test_gap_homog_timerange_read.cpp`, **`test_dynamic_aos_nested_only.cpp`** (AoS dynamique contenant uniquement des AoS statiques imbriquées) |
