# PanzerDB — Suppression du code mort

Ce document est la **feuille de route d'exécution** du nettoyage du code mort
de la classe `PanzerDB` (`src/hdf5/panzerdb.{h,cpp}`). Il précise les
décisions prises, l'inventaire exact des lignes supprimées, ce qui reste,
les risques et le protocole de validation.

Date : 2026-08-27
Base : commit `9fd1b4f` (branche `develop_upstream`)
Rapports connexes : `panzer_index_table.md` (structure de l'index),
`panzerdb_caches.md` (inventaire des caches).

---

## 1. Contexte et motivation

L'audit des caches (`panzerdb_caches.md`) a mis en évidence qu'une part
importante de la surface de code de `PanzerDB` est inerte :

- fonctions **déclarées mais jamais appelées** ;
- membres **déclarés mais jamais écrits/lus** ;
- options de configuration **jamais appliquées** ;
- un **brique fonctionnelle rendue morte** par une instruction qui la précède
  (la branche incrémentale de `getLeaves`).

Ce code pollue la lecture, fausse la documentation (des rapports citent des
mécanismes qui n'ont jamais d'effet) et introduit des bugs latents (un flag
d'invalidation non réinitialisé sur du code qui ne s'exécute jamais).

**Objectif** : supprimer toute cette surface sans changer le comportement
observable, en s'appuyant sur la suite de tests complète (63 ctest) comme
filet de sécurité.

**Principe** : on ne répare pas du code mort pour le garder vivant — on le
supprime, sauf si une option de design le justifie (décisions L0 ci-dessous).

---

## 2. Décisions de design (lot L0)

Trois éléments du lot L0 n'étaient pas des suppressions triviales : ils
impliquaient un choix. Les options sont documentées avec l'option retenue.

### L0.1 — Branche incrémentale de `getLeaves`

**Constat** : dans `getLeaves()` (panzerdb.cpp:1349), la ligne 1355
`cached_leaves.clear()` est exécutée **avant** le test
`if (!cached_leaves.empty())` de la ligne 1372. La condition est donc
**toujours fausse** : le bloc de rechargement incrémental (1371-1389) est
inatteignable. En pratique, toute invalidation du cache déclenche un
rechargement complet de l'index depuis le disque.

| Option | Description | Verdict |
|---|---|---|
| **A. Supprimer la branche morte** *(retenue)* | Force le comportement existant (full reload) en l'explicitant. Supprime ~19 lignes de code inatteignable et les 3 chemins de lecture conditionnels (`start_row == 0` vs non). | ✅ Retenu : le comportement reste **exactement le même**, le code devient plus lisible. Le vrai cache chaud est `leaves_cache_valid`, qui n'est pas touché. |
| B. Réactiver la branche | Enlever le `clear()` de 1355 pour permettre le re-patch incrémental. | Refusé : c'est une **fonctionnalité nouvelle**, pas un nettoyage. Elle serait à réimplémenter avec soin (les pointeurs des blocs changeaient entre rechargements, il faut vérifier la cohérence de `leaf_lookup`/`parent_lookup` sur un append). À envisager dans un ticket perf dédié, avec benchmark. |

### L0.2 — Chemin index temporel

**Constat** : le triplet `buildTimeIndex` / `findLeafByTime` /
(`time_range_index` + `leaf_metadata_cache` + `time_index_valid`) est un
mécanisme de recherche O(log n) par range de temps. `findLeafByTime` n'a
**aucun appelant**. `leaf_metadata_cache` sert aussi à `readSliceDirect`
(en production, 5 call sites) pour éviter de recaler `slice_volume`.
`time_index_valid`, une fois posé, **n'est jamais réinitialisé**, ce qui
constitue un bug latent si ce chemin était réactivé sur une session APPEND.

| Option | Description | Verdict |
|---|---|---|
| A. Supprimer `findLeafByTime` seul | Garde `buildTimeIndex` + caches pour `readSliceDirect`. | Partiel. |
| **B. Supprimer tout le chemin** *(retenue)* | `buildTimeIndex`, `findLeafByTime`, `time_range_index`, `leaf_metadata_cache` (struct `LeafMetadata`), `time_index_valid`, le struct `TimeRangeP` (exclusif à ce chemin). `readSliceDirect` recalcule `slice_volume` à la volée à partir de `leaf.shape` (déjà en mémoire) : coût ~3 multiplications, **zéro I/O supplémentaire**. | ✅ Retenu : le gain est net (-4 membres, -1 struct, -2 méthodes, -1 bug latent) et la pénalité de `readSliceDirect` est négligeable. Bonus : `restoreTimeContext` (cpp:653-662) calcule déjà `slice_volume` exactement ainsi, ce qui valide la justesse de l'approche. |
| C. Garder + patcher l'invalidation | Ajouter `time_index_valid=false` aux 3 points d'effacement. | Refusé : on maintient du code dont la seule utilité est un optimisation marginale, au prix d'un invariant subtil à maintenir. |

### L0.3 — Mécanique auto-flush

**Constat** : `AUTO_FLUSH_THRESHOLD` (100 Mo), `CRITICAL_FLUSH_THRESHOLD`
(500 Mo), `getTotalBufferSize()` et `autoFlushIfNeeded()` sont déclarés
mais **jamais appelés**. Le seul appel possible est commenté à
panzerdb.cpp:1720. Le flush effectif reste le `flush()` explicite appelé
par `HDF5Writer_v2::endAction` (hdf5_writer_v2.cpp:597) et le destructor.

| Option | Description | Verdict |
|---|---|---|
| **A. Supprimer la mécanique** *(retenue)* | Supprime 2 `constexpr`, 2 déclarations, 1 commentaire. Le garde-fou réel (flush explicite par l'API haute) reste en place. | ✅ Retenu : supprimer une mécanique inerte est plus honnête que de la garder « pour plus tard ». Si on a besoin d'un auto-flush un jour, on le réimplémente proprement avec les seuils documentés. |
| B. Garder et appeler | Réactiver `autoFlushIfNeeded()` dans `append_index_row`. | Refusé : changer le comportement de la session d'écriture (un flush intermédiaire peut perturber les tests qui observent le buffer en mémoire) n'est pas du nettoyage. À envisager dans un ticket dédié, documenté. |

---

## 3. Inventaire des suppressions — lot Commit 1 (L0)

Toutes les lignes référencées sont dans `src/hdf5/panzerdb.h` (h) et
`src/hdf5/panzerdb.cpp` (cpp), sauf mention.

### 3.1 L0.1-A — Branche incrémentale de `getLeaves`

| Élément | Emplacements |
|---|---|
| `uint64_t start_row = 0;` | cpp:1371 |
| Bloc incrémental entier | cpp:1372-1389 |
| Conditionnement `if (start_row == 0)` sur `cached_paths_blocks` / `cached_parent_paths_blocks` / `cached_dynamic_aos_roots` | cpp:1399-1403 (simplifié en `clear()` inconditionnel) |
| Chemin index conditionnel | cpp:1408-1419 (garder le seul `H5S_ALL` de 1409) |
| Chemin `paths_dset` conditionnel | cpp:1435-1444 (garder le seul `H5S_ALL` de 1436) |
| Chemin `parent_paths_dset` conditionnel | cpp:1454-1464 (garder le seul `H5S_ALL` de 1456) |

Après suppression : `read_count = n_rows` (1397) est la seule variable de
lecture ; les 3 `if (start_row == 0)` disparaissent.

### 3.2 L0.2-B — Chemin index temporel

**Membres supprimés (h)** :

| Membre | Ligne h |
|---|---|
| `mutable std::map<TimeRangeP, size_t> time_range_index;` (déb) | 261 |
| `mutable bool time_index_valid = false;` | 278 |
| `struct LeafMetadata { size_t slice_volume; size_t n_time_steps; TimeRangeP time_range; };` | 281-285 |
| `mutable std::vector<LeafMetadata> leaf_metadata_cache;` | 286 |
| `void buildTimeIndex() const;` | 288 |
| `const Leaf* findLeafByTime(const std::string&, int64_t) const;` | 289 |
| `struct TimeRangeP { ... }` (entier) | 114-130 |

**Définitions supprimées (cpp)** :

| Fonction | Lignes cpp |
|---|---|
| `PanzerDB::buildTimeIndex()` (avec section commentée « 2. CONSTRUCTION ... ») | 910-949 |
| `PanzerDB::findLeafByTime()` (avec section commentée « 3. RECHERCHE ... ») | 951-987 |

**Réécriture `PanzerDB::readSliceDirect()`** (cpp:993-1049) :

- Supprimer `buildTimeIndex();` (998) et `size_t leaf_idx = &leaf - &getLeaves()[0];` (999).
- Supprimer `const auto& meta = leaf_metadata_cache[leaf_idx];` (1000).
- Calculer inline, juste avant l'utilisation :
  ```cpp
  uint64_t slice_volume = 1;
  for (size_t s : leaf.shape) if (s > 0) slice_volume *= s;
  if (slice_volume == 0) slice_volume = 1;
  ```
  C'est le même calcul que `restoreTimeContext` (cpp:653-659) — déjà prouvé valide.
- Remplacer `meta.slice_volume` par `slice_volume` (1029-1030).
- Le reste de la fonction (dset_id / mem_type / hyperslab / H5Dread) est inchangé.

**Commentaires mis à jour** :

| Emplacement | Avant | Après |
|---|---|---|
| cpp:47 | « Caching: ... `buildTimeIndex` creates an O(log n) lookup structure ... » | « Caching : la méthode `getLeaves` met en cache la table d'index complète en mémoire. » |
| h:53-55 | « Optimized Reads : ... direct-to-buffer hyperslab reads (`readSliceDirect`) ... » | Invaré (la fonction reste). On garde simplement la mention de `readSliceDirect`. |

### 3.3 L0.3-A — Mécanique auto-flush

| Élément | Emplacement |
|---|---|
| `static constexpr size_t AUTO_FLUSH_THRESHOLD = 100'000'000;` | h:225 |
| `static constexpr size_t CRITICAL_FLUSH_THRESHOLD = 500'000'000;` | h:226 |
| Commentaire « Thresholds for auto-flush ... » | h:224 |
| `size_t getTotalBufferSize() const;` | h:240 |
| `void autoFlushIfNeeded();` | h:241 |
| `//autoFlushIfNeeded();` (commentaire) | cpp:1720 |

---

## 4. Inventaire des suppressions — lot Commit 2 (L1)

Suppressions « sûres » : aucune n'a d'appelant dans le dépôt (vérifié par
`rg` sur `src/`, `include/`, `tests/`, `py/`), et aucune n'est exposée
dans les headers publics (`include/`).

| # | Élément | Emplacement(s) | Statut avant |
|---|---|---|---|
| 1 | `mutable std::vector<int32_t> scratch_i32;` | h:271 | Declaré, commentaire « Not used in current code but ready ». Jamais accès. |
| 2 | `mutable std::vector<std::complex<double>> scratch_c128;` | h:272 | Idem. |
| 3 | `std::map<std::string, std::string> metadata_map;` (PanzerDB) | h:276 | Jamais accès. **Attention** : `HDF5Writer_v2::metadata_map` (hdf5_writer_v2.h:23) est **d'un autre type** et **est utilisé** (cpp:71, cpp:285) — il reste. |
| 4 | `std::string findDynamicAOSParent(const std::string&, const std::vector<Leaf>&);` | h:808-809 | Jamais appelée. |
| 5 | Définition `PanzerDB::findDynamicAOSParent` | cpp:696-710 | Avec le commentaire « Helper to find dynamic AOS parent of a leaf ». |
| 6 | `template<typename T> int readMultipleSlices(const std::vector<const Leaf*>& leaves, T* output) const;` (décl + doc) | h:609-617 | Jamais appelée. |
| 7 | Définition `PanzerDB::readMultipleSlices` | cpp:1060-1140 (section « 5. LECTURE GROUPÉE POUR SIGNAUX CONTIGUS ») | Aucune instantiation appelée. |
| 8 | 3 instantiations explicites | cpp:1142-1144 | `readMultipleSlices<double / int32_t / std::complex<double>>` |
| 9 | `uint64_t getTimeBaseLength(const std::string&) const;` | h:435 | Jamais appelée. |
| 10 | Définition `PanzerDB::getTimeBaseLength` | cpp:2154-2164 | — |
| 11 | `uint64_t getLastTimeIndex(const std::string&) const;` | h:436 | Jamais appelée. |
| 12 | Définition `PanzerDB::getLastTimeIndex` | cpp:2166-2177 | — |
| 13 | `size_t metadata_cache_size = 16 * 1024 * 1024;` | h:95 | Jamais appliqué (aucun `H5Pset_meta_block_size` / meta-cache DAPL configuré). On **ne réimplémente pas** la configuration pour lui donner vie, on la retire. |
| 14 | `mutable std::string cached_path_prefix;` | h:229 | Jamais lu. |
| 15 | `mutable bool path_prefix_dirty = true;` | h:230 | Toujours `true` au moment où `rebuildPathPrefix()` se lance ; la variable de « cache » n'a jamais d'effet. |
| 16 | `void rebuildPathPrefix();` (décl) | h:232 | Jamais appelé. |
| 17 | Définition `PanzerDB::rebuildPathPrefix` | cpp:629-635 | — |
| 18 | Écriture de `path_prefix_dirty = true;` | cpp:1262 (dans `beginArray(name,size)`), cpp:2246 (dans `endArray`) | Supprimées. Le champ `path_prefix` est déjà géré directement par `beginArray(ArrayLevel)` (pas touché). |

**Mise à jour des rapports** (après le code compilant) :

- `report/panzerdb_caches.md` : mettre le §2 (`buildTimeIndex`), §3.5
  (`cached_path_prefix`), le tableau §5, et le §7 à jour pour refléter
  que ces éléments n'existent plus.
- `report/panzer_index_table.md` : lignes 120 (AUTO_FLUSH), 348, 480
  (`findLeafByTime` / `time_range_index`), 482 (`scratch_i32/c128`).

---

## 5. Ce qui RESTE (garde-fous)

Liste explicite des éléments qui sont **vivants** et **non touchés**,
pour éviter toute confusion avec des éléments supprimés de même nom :

| Élément | Raison |
|---|---|
| `scratch_str` (h:273) | Utilisé en production, `pz_readStringData_by_index` (cpp:2884-2921). |
| `leaf_lookup` / `parent_lookup` / `max_time_at_dynamic_root` | Cœur du cache de lecture, utilisé par `getLeaves`, `getAOSShape`, `pz_read*Data_by_index`. |
| `cached_dynamic_aos_roots`, `cached_paths_blocks`, `cached_parent_paths_blocks` | Livrés par `getLeaves`, utilisés par la lecture et `restoreTimeContext`. |
| `aos_time_counters` | Utilisé par `restoreTimeContext` (cpp:643-693) et le compteur de temps par AoS. |
| `written_metadata_schema_paths` | Utilisé par `writeMetaData` (cpp:3446) pour dédupliquer les `@key`. |
| `data_buffer_f64 / i32 / c128 / str`, `index_buffer` | Buffer d'écriture, utilisé par `flush()` et `append_index_row`. |
| `chunk_config.index_chunk_rows / data_chunk_* / path_chunk_entries / enable_compression / compression_level` | Utilisés par `createOptimizedDataset` et `configureChunking`. |
| `chunk_config.chunk_cache_size / chunk_cache_nslots` | Utilisés par `configureReadCache` (cpp:351-352) et la construction (cpp:155-156). |
| `HDF5Writer_v2::metadata_map` (hdf5_writer_v2.h:23) | Utilisé (hdf5_writer_v2.cpp:71, 285). |
| `flush()`, `flushDataBuffer< T >`, `flushPathBuffer`, `flushComplexBuffer`, `flushStringBuffer` | Chemin réel de l'écriture, appelé par `endArray`/`endAction`/destructor. |

---

## 6. Risques et atténuation

| Risque | Probabilité | Impact | Atténuation |
|---|---|---|---|
| Perdre une optimisation de re-patch incrémental (L0.1) | — | Faible : le cache chaud est `leaves_cache_valid`, qui reste ; le full-reload est déjà le comportement réel. | Documenté dans `panzerdb_caches.md` ; à re-évaluer avec un benchmark si une session APPEND devient lente. |
| Pénalité `readSliceDirect` (L0.2) | — | Nul : `slice_volume` est recalculé depuis `leaf.shape` (en mémoire), ~3 multiplications, zéro I/O. | Comparaison directe avec le calcul identique déjà présent dans `restoreTimeContext` (cpp:653-662). |
| Supprimer une API qui serait appelée par du code externe hors dépôt | Faible | Moyen | Vérifié `rg` sur `src/`, `include/`, `tests/`, `py/`, et `panzerdb.h` n'est inclus nulle part hors `src/hdf5/` et `tests/`. |
| Cacher un bug latent (`time_index_valid` non réinitialisé) | — | Résolu : le flag est supprimé avec son chemin. | — |
| Rendre `rebuildPathPrefix` vide au lieu de le supprimer | — | Faible : on supprime la fonction, le champ et les 2 écritures du flag. `path_prefix` (le champ réel, pas le `_cached`) reste en place pour `beginArray`. | Vérifié par `rg path_prefix` après la suppression. |
| Builder une erreur de type à cause de `TimeRangeP` (h:114-130) qui est exclusif au chemin supprimé | — | Faible : vérifié qu'aucune autre utilisation n'existe (seulement `panzerdb.{h,cpp}`). | Build + ctest complet. |

---

## 7. Ordre d'exécution

```
1. Étape 0    : Écrire ce document (report/panzerdb_removing_dead_codes.md)
2. LOT L0     : panzerdb.{h,cpp} — L0.1-A, L0.2-B, L0.3-A
               → git add + git commit (1er commit)
                → cmake --build build/ --srcdir . -j8
                → ctest --test-dir build/src/hdf5 -R "^test_panzer" (suite complète 63 tests)
                   (3 passes, diff stable)
3. LOT L1     : panzerdb.{h,cpp} (items 1-18 du §4)
                → MAJ report/panzerdb_caches.md + panzer_index_table.md
               → git add + git commit (2e commit)
                → cmake --build build/ --srcdir . -j8
                → ctest 3 passes, diff stable
2. Grep final : rg sur tout le dépôt (hors build/, report/) pour vérifier
                qu'aucun symbole supprimé ne subsiste hors rapport.
```

---

## 8. Validation

### Build

```
cmake --build build/ --srcdir . -j8
```

Zéro erreur, zéro avertissement sur les symboles supprimés.

### Tests

63 ctest dans `build/src/hdf5`. La suite de référence comprend
`test_panzer.cpp`, `test_panzer_dynamic_aos.cpp`,
`test_dynamic_aos_nested_only.cpp` (le test de régression du bug
`max_time_at_dynamic_root`), `test_panzer_read_by_index_*.cpp`, etc.

Passes : 3 fois, diff stable, `PASS` sur les 63.

### Grep final de confirmation

```
git grep -nE 'findLeafByTime|readMultipleSlices|autoFlushIfNeeded|getTotalBufferSize|AUTO_FLUSH_THRESHOLD|CRITICAL_FLUSH_THRESHOLD|time_index_valid|leaf_metadata_cache|LeafMetadata|TimeRangeP|getTimeBaseLength|getLastTimeIndex|metadata_cache_size|cached_path_prefix|path_prefix_dirty|rebuildPathPrefix|scratch_i32|scratch_c128|findDynamicAOSParent' -- src include tests py
```

Attendu : **zéro résultat** dans `src/`, `include/`, `tests/`, `py/`.
Les mentions restantes sont autorisées uniquement dans `report/*.md`
(documentation historique).

---

## 9. Messages de commit (proposés)

**Commit 1** :
```
PzDB: drop dead time-index path, dead incremental-branch, and autoFlush

Remove code that is either unreached or never-called:

- getLeaves(): the incremental-reload branch (start_row > 0) is
  unreachable because cached_leaves is cleared at the top of the
  invalidation path. Collapse the three read sites
  (index, paths, parent_paths) to their unconditional H5S_ALL path.
- buildTimeIndex() / findLeafByTime() / time_range_index /
  leaf_metadata_cache / time_index_valid / LeafMetadata / TimeRangeP
  form an O(log n) time-range lookup that no call site uses.
  readSliceDirect() recomputes slice_volume on the fly from leaf.shape
  (in memory, no additional I/O), mirroring restoreTimeContext().
  Side benefit: removes the time_index_valid staleness bug.
- AUTO_FLUSH_THRESHOLD / CRITICAL_FLUSH_THRESHOLD /
  getTotalBufferSize() / autoFlushIfNeeded() are declared but never
  called. The explicit flush() (and HDF5Writer_v2::endAction) remains
  the only flush path.

PanzerDB-only change; no public header (include/) or test affected.
```

**Commit 2** :
```
PzDB: remove unused members, unimplemented helpers, stale config

Dead code (verified zero callers in src/, include/, tests/, py/):
- scratch_i32, scratch_c128 (never used; scratch_str kept, used by
  pz_readStringData_by_index)
- metadata_map (PanzerDB) — distinct from HDF5Writer_v2::metadata_map,
  which IS used
- findDynamicAOSParent (never called; restoreTimeContext inlines the
  equivalent loop)
- readMultipleSlices<T> + 3 explicit instantiations (never called;
  readLeavesUnion is what's on the hot path)
- getTimeBaseLength, getLastTimeIndex (never called)
- chunk_config.metadata_cache_size (declared, never applied to HDF5)
- cached_path_prefix + path_prefix_dirty + rebuildPathPrefix()
  (rebuilt lazily but never invoked)
- Update report/panzerdb_caches.md and report/panzer_index_table.md
  to reflect the final state.

PanzerDB-only change; no public header or test affected.
```

---

## 10. Suivi post-mortem

Après les 2 commits :

1. `git log --oneline -3` pour vérifier la base.
2. `git diff --stat BASE..HEAD` pour confirmer le périmètre attendu
   (seuls `src/hdf5/panzerdb.{h,cpp}` et 3 fichiers de `report/`).
3. `ctest --test-dir build/src/hdf5 --output-on-failure` final.
4. `grep -nE 'findLeafByTime|readMultipleSlices|autoFlushIfNeeded|...' src/` → zéro.

Ce document reste dans `report/` comme traçabilité de la décision et de
l'exécution. S'il revient des tests en échec ou des points d'attention,
ils seront ajoutés ici.
