# Plan d'implémentation — index compact (index_plan.md → code réel)

> Décision retenue : **s'aligner sur le code réel** (schéma 3 datasets) et **séparer les
> indices d'AoS du texte de chemin**, en résolvant le chemin à la lecture. Aucune compatibilité
> avec anciens fichiers requise (confirmé). Contrat à préserver : la surface publique `Leaf`.
>
> **Ancre code réelle** (à l'inverse de l'hypothèse 336 o / `path[256]` du doc) : 3 datasets
> parallèles → **624 o par nœud** :
>   - `/index`         : 14×`uint64` = 112 o (`panzerdb.cpp:178-189`)
>   - `/paths`         : 256 o/ligne (`C_S1` NULLPAD, `panzerdb.cpp:212-216`)
>   - `/parent_paths`  : 256 o/ligne (`panzerdb.cpp:218`)
> et le parent/child est résolu **par texte de chemin** (pas de `parent_id` numérique).

---

## 0. Résumé

L'idée du doc « enfants = 80 o, chemin résolu par `parent_id` » est **réalisable dans le design
actuel**. Le gain vient de ne plus stocker de texte par enfant (ni chemin, ni parent) : les enfants
deviennent **100 % numériques** et partagent les noms via une **table de symboles** `/names`.

Cible : **624 o → ~112 o / enfant** (~82 %), 30 M slices index ≈ 1,9 Go vs ≈ 10,1 Go → reste en cache
RAM/L3, et la bande passante écriture (solver) divise par ~5.

### Schema cible par nœud (largeur `/index` inchangée = 14 colonnes, zéro nouveau dtype sur /index)
| colonne | usage actuel | usage cible |
|---------|--------------|-------------|
| `row[0]` | `type` (toujours 0, `append_index_row` 1833) → **libre** | **`name_id`** |
| `row[1]` | `ndim` | `ndim` |
| `row[2..7]` | `shape[0..5]` | `shape[0..5]` |
| `row[8]` | `time_idx` | `time_idx` (inchangé — indices dynamiques déjà numériques) |
| `row[9]` | `offset` | `offset` |
| `row[10]` | `count` | `count` |
| `row[11]` | `flags` (kind 4 bits bas + `DataType`>>4) | `flags` (inchangé) |
| `row[12]` | `0` (libre) | **`parent_id`** (id de rangée ; sentinel = racine) |
| `row[13]` | `0` (libre) | **`index_value`** (indice d'instance `[i]/[g]/[d]` ; 0 si absent) |

+ dataset **`/names`** (vlen `C_S1`, noms dé-dupliqués) ; **suppression de `/paths` + `/parent_paths`**.

### Contrat de non-régression (garde-fou unique et suffisant)
Les 9 champs de chaque `Leaf` fournis par `getLeaves()` doivent rester **byte/value identiques** :
`(path, parent_path, time_index, offset, count, flags, shape)` (+ `is_empty`). Si c'est respecté,
le lot entier (~30 sites, cf. §« Surface à auditer » : `leaf_lookup`, `parent_lookup`, `stripIndices`,
`list_nodes`, `iread_strategy.h`, `hdf5_writer_v2`, `readTensor`) est pré
servé. **`parent_lookup` est donc clé par le texte exact de `parent_path`** → à piquer au golden.

---

## 1. Ancre code (vérifiée)

- **`ArrayLevel`** — `panzerdb.h:118-128` (`name, is_dynamic, declared_size, current_index,
  actual_count, had_write, aos_full_path`). On ajoute `container_row_id` et `name_id`.
- **`Leaf`** — `panzerdb.h:183-194`. `path`/`parent_path` = `std::string_view` (contrat public).
- **`flags`** (`= (DataType)<<4`) bits bas : `0` donnée / `1` vide / `2` AoS statique / `3` AoS dynamique.
- **`append_index_row`** (point d'écriture unique) — `panzerdb.cpp:1833-1884` ; seuls `row[0..11]`
  pleins, **`row[12]`,`row[13]` libres** ; `strncpy` 2× 256 o/entrée.
- **5 sites d'émission** : `beginArray`(1145) ; `writeDataImpl<T>`(1456) ; `writeData<char>`(1565) ;
  `writeDataSlicesImpl<T>`(1679, 1 ligne/call, `time_idx=base_time`
  , `count=slice_size*n_slices`) ; `writeDataSlices<char*const*>`(1816, 1 ligne/slice).
- **Lecture** — `getLeaves()`(1155-1266) : `H5Dread` du `idx`, allocate les buffers, construit les
  `Leaf`, `leaf_lookup[leaf.path]`, `parent_lookup[leaf.parent_path]`, `cached_dynamic_aos_roots`
  (`flags==3`).
- **`stripIndices`** — `panzerdb.cpp:3106` (chemin d'instance → schema), consommé par
  `list_nodes`, `hdf5_writer_v2.cpp:266`, `getNodes`.
- **`getAOSShape`** — `panzerdb.cpp:1955-2040` : dynamic → `max_time_at_dynamic_root` ;
  static → `parent_lookup.find(level_name)` puis substr index de `leaf.path`.
- **`iread_strategy.h:259-260`** : `leaf.parent_path.rfind(timed_aos_path,0)==0` et
  `leaf.path.find_last_of('/')` → le dernier `/` doit rester au même endroit.
- **Datasets** : `init()` 178-218 ; delete-à-WRITE 159-165 ; open APPEND/READ 239-242/261-264 ;
  `reopen` 375-376 ; `flush()` 689-862 (paths 717-753) ; `close()` 871-874 ; `PATH_MAX_LEN=256`(55).
- **`/strings` déjà vlen** — `panzerdb.cpp:197-198`(create), `816-817`(flush), `1365-1366`(read) :
  `H5Tcopy(H5T_C_S1)` + `H5Tset_size(…, H5T_VARIABLE)` + write/read via `std::vector<char*>`.
  → pattern vlen **déjà en place**, réutilisé pour `/names` (zéro nouveau risque HDF5).
- **`/names`** : vlen recommandé (table symboles petite → 16 o/cellule négligeable) ; alt. = `C_S1`
  fixe 64–128 (OK aussi vu la taille). Décider en M2.

---

## 2. Surface à auditer (5 fichiers)

- **Émission (`panzerdb.{h,cpp}`)** : `append_index_row`, les 5 sites, `ArrayLevel`, `init`,
  delete-WRITE, open/reopen, `flush`, `close`, `getLeaves`.
- **Lecture moteur (`panzerdb.cpp`)** : `getLeaves`, `getAOSShape`, `stripIndices`, les 3 stratégies
  `readDataByIndex`(2360-2391 f64 / 2530-2546 i32 / 2748-2764 str), `rebuildDynamicRootTimeIndex`,
  clés `aos_time_counters` (636, 664, 673, 1282, 3169, 3242, 3280).
- **API publique** : `direct_access_api.cpp` — `list_nodes`(56-175, incl.
  `path_occurrence_count` → `time_dim`), `getAOSShape`(75), `getNodes`(157, 164, 169), `read_tensor`(451, 468).
- **Stratégie header** : `iread_strategy.h:259-260`.
- **Writer v2** : `hdf5_writer_v2.cpp:266` (`stripIndices`).
- **Tools** : `tools/imas_h5ls.cpp:60-79`, `tools/imas_h5dump.cpp:45-124` (consomment `NodeInfo.path`).

---

## 3. Jalons (chacun = build vert + suite de tests + dump `h5ls`/`h5dump`)

### M0 — Oracle de non-régression (AVANT tout)
`test_index_golden.{cpp,h}` + fixture `goldens/` (nouveau), enregistré dans `src/hdf5/CMakeLists.txt`
(macro `al_add_hdf5_test`, 60-75). Capture **sur le code actuel** :
  - la surface `Leaf` complète (sortie de `getLeaves()` : `path, parent_path, time_index, offset,
    count, flags, shape, is_empty`), triée, sur un corpus représentatif (tests `engine_*` + `al_*`
    profiles_1d / dynamic_aos / gap / root_scalar / char*const*).
  - **et** la sortie de `list_nodes`/`getNodes` (`schemas`, `logical_dims`, `aos_sizes`,
    NodeType) — contrat des CLI `imas_h5ls`/`imas_h5dump` pas couvert par la surface `Leaf` seule.
  - `stripIndices(leaf.path)` sur l'ensemble (consommé par `hdf5_writer_v2`, `getNodes`).
Livraison : `goldens/*.golden` captés. **Tout M1..M3 doit produire strictement ces lignes.**

### M1 — `parent_id` / `index_value` + reconstruction `parent_path` (Design 1)
`append_index_row` + 5 sites (écrire `row[0]=name_id`, `row[12]=parent_id`, `row[13]=index_value`) ;
`ArrayLevel` +`container_row_id` ; `getLeaves` reconstruit `leaf.parent_path` **via `parent_id`**
(plus de `/parent_paths`) ; **on garde `/paths`**. `h5ls` : `/parent_paths` absent, `/paths` présent.
56 tests + golden vert.

### M2 — Table de symboles `/names` (vlen) + enfants numériques (Design 2)
`init`/delete-WRITE/`open`/`reopen`/`flush`/`close` : création `/names` (vlen), suppression
`/paths`+`/parent_paths` de la lecture, flush `/names` (pattern `816-819`), id stable en APPEND
(recharger max existants à l'ouverture, `1131-1142`). Enfants : `name_id`+`parent_id`+`index_value`
uniquement. `h5ls` : 2 datasets méta + `/names`. 56 tests + golden vert. **C'est ici qu'on obtient
~82 %.**

### M3 — Cas limites
AoS imbriqués/dynamiques (règle d'inclusion de l'index, cf. asymétrie
`beginArray`(1126-1129, brute) vs `writeDataSlicesImpl`(1594-1598, non-dyn)) ; nœuds vides (`flags==1`) ;
dynamic gaps/chunking (invariants `aos_time_counters`/`base_time`) ; `char*const*` (root_scalar,
string_complex, verydeep, ultradeep) ; audit `iread_strategy.h:259-260` (règle `find_last_of('/')`).

### M4 — Perf & validation
Benchmark 30 M slices : taille index réelle (cible ~76 %+, viser ~82 %), bande passante écriture I/O
(80/112 o/événement vs 624 o), `h5ls`/`h5dump` final, `list_nodes`/`getNodes`/`read_tensor` verts.

---

## 4. Ce qui change / ce qui ne change PAS

**Change** : `append_index_row`(signature + corps), 5 sites, `ArrayLevel`, `getLeaves`, `init`,
`delete-WRITE`, `open/reopen`, `flush`, `close`, nouveau `test_index_golden`, fixtures `goldens/`.
**Ne change PAS (contrat)** : `Leaf`(9 champs), `flags`, `time_idx`, `offset`, `count`, la largeur
`14` de `/index`, les datasets de données brutes, `aos_time_counters`/`base_time`, `readTensor`,
`getAOSShape`, `stripIndices`, les 56 tests.

---

## 5. Risques & garde-fous
1. **Composition path / inclusion de l'index** (dynamique vs statique ; asymétrie §M1/§M3) →
   **golden comme oracle unique, pas de règle à main**.
2. **Continuité d'IDs en APPEND** (`name_id`/`parent_id`) → recharger à l'ouverture.
3. **`parent_lookup` clavé par le texte exact de `parent_path`** (non-remplaçable par `parent_id`
   tant que l'oracle est in place) → `getLeaves` reconstruit le texte avant de peupler.
4. **`iread_strategy.h`** dépend de `parent_path.rfind` + `find_last_of('/')` → piquer au golden.
5. **Tools / writer v2** consomment `stripIndices(leaf.path)` → piquer au golden.

---

## 6. Definition-of-done
- [ ] M0 : `test_index_golden` capté sur le code actuel (oracle figé).
- [ ] M1 : `parent_id`/`index_value` + reconstruction `parent_path` ; `/parent_paths` supprimé.
- [ ] M2 : `/paths` supprimé, `/names` (vlen) présent, enfants 100 % numériques, 56 tests + golden vert.
- [ ] M3 : AoS imbriqués/dynamiques, nœuds vides, gaps, `char*const*`, root_scalar verts.
- [ ] M4 : benchmark 30 M, réduction index mesurée ≥ ~76 % (viser ~82 %), CLI `imas_h5ls`/`imas_h5dump`
      verts sur un corpus représentatif.
- [ ] DoD : `Leaf` 9 champs byte-identiques ; `h5ls`/`h5dump` = datasets attendus (`/index`, `/names`,
      4 datasets de données brutes).
