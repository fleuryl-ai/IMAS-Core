# Rapport d'implémentation — format d'index v3 (taille + lecture)

Date : 2026-10-08
Périmètre : points 4, 7, 13, 14, 16 de `REVIEW_PERF_hdf5_v2.md` (Phase 3 « format »
+ shape inline), avec compatibilité ascendante. Suite logique de `index_plan.md` /
`index_plan_impl.md` (M1 parent_id était déjà en place).

## 1. Résumé

Nouveau format d'index **v3** pour le backend PanzerDB (`src/hdf5`) :

| | v2 / legacy | v3 |
|---|---|---|
| `/index` | 12 ou 14 colonnes u64 (96–112 B/ligne) | **7 colonnes u64 packées (56 B/ligne)** |
| Chemins | `/paths` aligné sur les lignes (256 B × N) | **`/path_table` interné (chemins uniques uniquement)** |
| Version du format | déduite des colonnes | **attribut `index_version=3` sur `/index`** |
| Chaînes temporelles | 1 ligne d'index PAR pas | **1 ligne par RUN de pas consécutifs** |
| `Leaf::shape` | `std::vector` (1 malloc/ligne) | **tableau inline `uint64_t[6]` + `ndim`** (vue `shape_span()`) |

Les fichiers v2 (12 colonnes) et legacy (14 colonnes) restent **lisibles et
extensibles en APPEND** ; les nouveaux fichiers sont écrits en v3.

## 2. Layout v3

`/index` : tableau `uint64[N, 7]` (tous les codes I/O HDF5 existants inchangés) :

| col | contenu |
|---|---|
| 0 | `offset` (u64) |
| 1 | `count` (u64) |
| 2 | `path_id` (u32 bas) \| `parent_id` (u32 haut, `0xFFFFFFFF` = racine) |
| 3 | `time_index` (u32 bas) \| (`flags` \| `ndim<<8`) (u32 haut) |
| 4–6 | `shape[0..5]` (2 × u32 par colonne) |

- `Leaf::flags` public = `stored & 0xFF` → **valeur de flags identique au contrat v2**
  (le ndim est encodé dans les bits 8–11 du mot stocké, masqué à la lecture).
- `parent_path` : reconstruit en `string_view` **préfixe du chemin** dans le bloc
  de la `/path_table` (règle M1 : −1 segment pour les data, −2 pour les méta
  AoS) — plus aucun buffer parent (le bloc 256 B/ligne de `cached_parent_paths_blocks`
  disparaît du chemin v3).
- Gardes-fous u32 : exceptions si `time_index`, une dimension de shape, un
  `path_id` ou un `parent_id` dépasse 2^32.
- `/path_table` : même dtype que l'ancien `/paths` (C_S1 fixe 256 B, chunked,
  deflate) ; append-only ; ids stables en APPEND (table chargée à l'ouverture).

## 3. Jalons implémentés

### V1 — version de format (point 16)
- Écriture d'un attribut `index_version` (u32) à la création de `/index`.
- `init()` (APPEND/READ) : lecture de l'attribut, sinon déduction par le nombre de
  colonnes (7 → v3 ; 12/14 → v2/legacy). Dispatch de tout le reste (datasets,
  encodage, décodage) sur cette version.

### V2/V3 — interning + packing (points 13+14)
- Writer : `intern_path()` (map texte → id, texte émis une seule fois),
  `flushPathTable()` (commit du payload AVANT la ligne d'index — même ordre
  crash-safe que les données), `loadPathTable()` (graine d'ids en APPEND).
- Reader : branche v3 dans `getLeaves()` (décodage des 7 colonnes + vues dans la
  table de chemins) ; branche v2/legacy inchangée. Contrat public `Leaf`
  byte-identique (oracle golden).

### V4 — une ligne d'index par run pour les chaînes (point 4)
- `writeDataSlices(const char* const*)` regroupe les pas consécutifs à taille de
  pas uniforme en UNE ligne (`count = slots/step × longueur du run`), comme le
  faisait déjà `writeDataSlicesImpl` pour le numérique.
- Exceptions restées par pas : scalaires `STRING_CHUNKED` (les slots sont des
  morceaux d'UNE valeur) et listes spanned (table `/list_spans` indexée par ligne).
- Les lecteurs géraient déjà les feuilles chaînes multi-pas
  (`readStringDataByIndex`, `read_dataset_globally`, `isTimeInLeaf`).

### Correctif induit nécessaire (bug latent préexistant)
En comptant les LIGNES comme dimension temps, `getAOSShape` (via
`max_time_at_dynamic_root`) et `list_nodes` sous-comptaient le temps dès qu'une
feuille portait plusieurs pas — déjà faux pour les écritures bulk numériques.
Corrigé : la couverture est `max(time_index + n_steps)` (fin exclusive), avec
`n_steps = count / ∏shape` (1 pour `STRING_CHUNKED`).

### V5 — shape inline (point 7)
- `Leaf` : `uint64_t shape[6]` + `uint8_t ndim`, vue `shape_span()`
  (surface `size/empty/[]/range-for` identique à l'ancien vecteur).
- ~40 sites adaptés dans `panzerdb.{h,cpp}`, `iread_strategy.h`,
  `direct_access_api.cpp`, `view_publisher.cpp`, `test_index_golden.cpp`,
  `bench_read.cpp`. Zéro allocation par feuille ; `Leaf` ne grossit que de 24 B
  (et le bloc parent de 256 B/ligne disparaît en v3).

### V6 — outils, tests, mesures
- `tools/panzer_index.py` : décodage v3 (attribut, 7 colonnes, `/path_table`),
  toujours v2/legacy ; en-tête indiquant le layout et le nombre de chemins uniques.
- Nouveau test `test_engine_index_v3` : estampille v3, interning (entrées ≪ lignes),
  round-trip numérique + chaînes fusionnées, continuité APPEND (ids et interning),
  lecture d'un fichier **legacy 12 colonnes synthétique** (sans attribut).
- `test_engine_commit_order_guardrail` : était un **no-op en Release** (`assert`
  supprimé par NDEBUG) ; réécrit avec `HARD_CHECK` + injection de ligne pendante
  aware v3 / v2 / legacy.

## 4. Mesures

Bench 300 signaux × 1000 pas (SLICE_OP, 301 001 lignes d'index, 302 chemins uniques) :

| dataset | v3 (bench 301k lignes) | legacy core_profiles_3 (171k lignes) |
|---|---|---|
| `/index` | 2,66 Mo (56 B/ligne brut) | 1,08 Mo (112 B/ligne brut) |
| chemins | `/path_table` **10,9 Ko** | `/paths` 663 Ko |
| RAM décodée | 16,9 Mo (buffer idx) + ~77 Ko (table) | 19,2 Mo + blocs chemins/parents 2×256 B/ligne |

Gains structurels :
- I/O index brut : **−36 %** (96→56 B/ligne) et suppression complète de la
  relecture de `/paths` (256 B/ligne, ~44 Mo non compressés sur le fichier benché
  de la revue).
- RAM du cache de feuilles : les deux blocs de 256 B par ligne (paths + parents
  reconstruits) sont remplacés par la table des chemins uniques → ~87 Mo → ~77 Ko
  sur core_profiles_3 ; plus une malloc par ligne en moins (V5).
- Lignes d'index pour les signaux de chaînes : divisées par le nombre de pas par
  appel d'écriture.

## 5. Fichiers modifiés

- `src/hdf5/panzerdb.h` : constantes v3, membres (version, `/path_table`, map
  d'interning), `Leaf` inline + `shape_span()`, déclarations d'aides.
- `src/hdf5/panzerdb.cpp` : `init()` (version/datasets), `append_index_row`
  (encodage v3), `flush`/`flushPathTable`, `getLeaves()` (branche v3),
  `intern_path`/`loadPathTable`, runs chaînes, `rebuildDynamicRootTimeIndex`
  (couverture), `getAOSShape`.
- `src/hdf5/direct_access_api.cpp` : couverture temporelle dans `list_nodes`.
- `src/hdf5/iread_strategy.h`, `view_publisher.cpp` : adaptation `shape_span()`.
- `src/hdf5/tools/panzer_index.py` : support v3.
- `tests/hdf5_backend/test_engine_index_v3.cpp` (nouveau),
  `test_engine_commit_order_guardrail.cpp` (réécrit),
  `test_index_golden.cpp` / `bench_read.cpp` (surface shape),
  `fixtures/goldens/index_surface.golden` (recapturé : fusion des lignes chaînes +
  dimensions temps corrigées pour les bulk), `src/hdf5/CMakeLists.txt`.

## 6. Validation

- `ctest` : **81/81** tests verts (dont golden verify et le nouveau test v3).
- Golden recapturé avec diff minimal : fusions V4 (3×cnt=1 → 1×cnt=3) et
  corrections de dims (ex. `dyn/sig` (1)→(3)) — pas de régression de surface.
- Vérification outillage : `panzer_index.py` sur un fichier v3 (décodage complet)
  et sur `core_profiles_3.h5` (legacy 14 colonnes, inchangé).

## 7. Reste à faire (hors périmètre)

- Phase 1 revue : hyperslab stride + lecture directe dans le buffer appelant
  (points 8/11), caches des valeurs de timebase (9) et des tailles d'AoS (12),
  recherche binaire des gaps (10), alignement FAPL adaptatif (5, taille).
- Phase 2 revue : rechargement **incrémental** de `/index` + `/path_table`
  (point 1) et moteur APPEND partagé entre opérations d'écriture.
- Optionnel : bits de rôle dans `flags` (point 15, sans gain de taille) et table
  de récap par signal (point 17).
