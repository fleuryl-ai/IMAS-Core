# PanzerDB (backend HDF5 v2) — Caches en lecture et en écriture

**Document technique de référence — backend v2 (IMAS-Core)**
Date : 27/08/2026 (rév. 2, après nettoyage du code mort)
Complément de `panzer_index_table.md` (section 8) — inventaire complet, fonctionnement, invalidation, et classement par impact performance (avis d'ingénierie, sans benchmark).

Tous les mécanismes décrits vivent dans `src/hdf5/panzerdb.{h,cpp}` à moins d'indication contraire.
Le nettoyage du code mort correspondant est documenté dans `panzerdb_removing_dead_codes.md`.

---

## 1. Inventaire

| # | Cache | Famille | Rôle principal |
|---|---|---|---|
| 1 | `cached_leaves` (+ `cached_paths_blocks`, `cached_parent_paths_blocks`) | lecture | Représentation RAM complète de la table d'index (`getLeaves`, `panzerdb.cpp:1158`) |
| 2 | `leaf_lookup` / `parent_lookup` | lecture | Recherche chemin → feuilles en O(1) |
| 3 | `cached_dynamic_aos_roots` + `max_time_at_dynamic_root` | lecture | Navigation AoS dynamique sans parcours O(N) |
| 4 | Buffers d'écriture RAM (`index_buffer`, `data_buffer_*`, `paths_buffer`, `parent_paths_buffer`) | écriture | Amortissement I/O : tout n'est écrit qu'au flush |
| 5 | `disk_size_f64/i32/c128/str` | écriture | Offset de l'append sans interrogation HDF5 |
| 6 | `aos_time_counters` | écriture | Numéro de slice suivant par signal (gaps, chunked write, alignement sur timebase) |
| 7 | `cached_dynamic_aos_path` + `dynamic_aos_path_valid` | écriture | Chemin de l'AoS dynamique courant, évite la remontée du stack |
| 8 | Chunk cache HDF5 (DAPL, 64 Mo / 10 007 slots) | HDF5 | Cache de chunks des datasets, commun à lecture **et** écriture |
| 9 | FAPL (alignment 4 Mo, méta-blocs 2 Mo) | HDF5 | Aggrégation de la metadata au niveau fichier |
| 10 | `written_metadata_schema_paths` | écriture/lecture | Dédoublonnage des entrées metadata `@key` |
| 11 | `scratch_str` | lecture | Tampon réutilisable pour `pz_readStringData_by_index` |

> **Note de maintenance (27/08/2026)** : le code mort antérieur —
> `time_range_index` + `leaf_metadata_cache` + `buildTimeIndex`/`findLeafByTime`
> + `TimeRangeP`, `cached_path_prefix` + `path_prefix_dirty` +
> `rebuildPathPrefix`, `autoFlushIfNeeded` + `getTotalBufferSize` + seuils
> `AUTO_FLUSH/CRITICAL_FLUSH_THRESHOLD`, `scratch_i32/c128`, `metadata_map`
> (PanzerDB), `findDynamicAOSParent`, `readMultipleSlices`,
> `getTimeBaseLength`, `getLastTimeIndex`, `chunk_config.metadata_cache_size` —
> a été **supprimé** (voir `panzerdb_removing_dead_codes.md`). Il n'existe plus
> de liste de code mort dans `panzerdb.{h,cpp}`.

---

## 2. Lecture

### 2.1 `cached_leaves` — la pierre angulaire (`panzerdb.cpp:1158`)

**Contenu** : `std::vector<Leaf>` — une structure par ligne de la table d'index
(`path`, `parent_path`, `shape`, `time_index`, `offset`, `count`, `flags`).
Les chemins sont des `std::string_view` **zéro-copy** pointant vers les blocs
`cached_paths_blocks` / `cached_parent_paths_blocks` (`panzerdb.h:235-242`),
chaque bloc étant `n_lignes × PATH_MAX_LEN` (256 o), allocés une fois par
segment lu (`panzerdb.cpp:1196-1199`).

**Fonctionnement** :

1. Premier appel `getLeaves()` (toute opération de lecture passe par là :
   `getAOSShape`, `pz_read*Data_by_index`, `readInterpolatedData`,
   `getTimeIndex`, `readMetadata`, `get_leaf_type`, `is_dynamic_aos`…) :
   - `H5Dget_space` sur `index` → N lignes ;
   - `H5Dread` complet de `index` (N×14 u64, `:1194`), `paths`, `parent_paths` ;
   - construction **en place** des `Leaf` (`emplace_back`, sans copie),
     remplissage des `leaf_lookup`/`parent_lookup` (`:1257`), détection des
     racines AoS dynamiques (`flags==3`), puis `rebuildDynamicRootTimeIndex()`
     (`:1265`) et `leaves_cache_valid = true`.
2. Appels suivants : retour immédiat du `const&` — aucun I/O.
3. **Reconstruction : toujours rechargement total.** C'est un choix explicite :
   à chaque invalidation, le cache entier est effacé
   (`:1164-1167` : `cached_leaves`, `leaf_lookup`, `parent_lookup`,
   `max_time_at_dynamic_root`) et la table d'index est relue intégralement.
   (Une branche « incrémentale » existait mais était inatteignable ; elle a
   été supprimée le 27/08/2026 — cf. `panzerdb_removing_dead_codes.md` §3.1.)
4. Invalidation : `leaves_cache_valid = false` après `flush()`
   (`panzerdb.cpp:861`) — cohérence stricte en mode APPEND/écrit-puis-lit.
   Deux autres points posent le flag à faux : l'ouverture WRITE (`:243`)
   et l'ouverture APPEND (`:265`), avant la construction initiale.

**Complexité** : coût d'entrée O(N) (un seul `H5Dread` par dataset, pas N
lectures HDF5) ; coût d'état O(1). Mémoire : N × ~224 o (Leaf ~72 o + vector
shape ~48 o + 2×256 o de chemins par ligne, amortis sur blocs).

**Coût à retenir** : chaque invalidation = relecture **intégrale** des N
lignes (2.1-3) — comportement volontaire et maintenant explicite.
En APPEND écrit-puis-lit très alterné, ce re-read intégral domine (cf. §7).

### 2.2 `leaf_lookup` / `parent_lookup` (peuplés à `panzerdb.cpp:1257`)

**Contenu** : `unordered_map<string_view, vector<size_t>>` — « chemin →
indices des feuilles ». Remplit le rôle d'index B-tree de la legacy
hdf5. `parent_lookup` est la réciproque (parent → enfants) : il alimente
`getAOSShape` (`panzerdb.cpp:1979`) sans parcourir les N feuilles.

**Fonctionnement** : peuplés **uniquement** à la construction du cache
(2.1) ; tout lookup courant (`getAOSShape`, `getTimeIndex`,
`nearestAvailableSliceIndex`, `pz_read*`, `is_dynamic_aos`, dédup de
`beginArray` en APPEND…) passe par `leaf_lookup.find(view)` en O(len)
moyen, au lieu d'une balayée O(N) en O(len) par comparaison de chaîne.

**Invalidateur** : `getLeaves()` (`:1164-1166`) — clear systématique avant
chaque re-pop. Pas de fuite : la reconstruction est toujours intégrale
(2.1-3), donc les deux maps sont intégralement reconstruites à chaque fois.

### 2.3 `cached_dynamic_aos_roots` + `max_time_at_dynamic_root`

**Contenu** :
- `cached_dynamic_aos_roots` : chemins des méta-nœuds AoS dynamiques
  (`flags==3`), `panzerdb.h:243`. Remplace l'ancien « balayage de toutes les
  feuilles pour trouver le parent dynamique » (cf.
  `restoreTimeContext` `panzerdb.cpp:628-693`) par une itération sur quelques
  racines ; utilisé par `nearestAvailableSliceIndex`, `getWholeDynamicSignal`
  (`:3103`), et la stratégie de résolution de chemin (direct / générique /
  substitué) des `pz_read*Data_by_index`.
- `max_time_at_dynamic_root` : `unordered_map<string_view, uint64>`,
  `panzerdb.h:250` — max `time_index` sur **toutes les feuilles données
  descendantes (toute profondeur)** d'une racine dynamique, construit par
  `rebuildDynamicRootTimeIndex()` (`panzerdb.cpp:1271`), appelé une fois par
  build de cache (`:1265`). Sert `getAOSShape` (`:1979`) : sans lui, le size
  d'un AoS dynamique dont les données sont toutes derrière des AoS statiques
  imbriqués (`time_slice/ggd/theta/values`) était 0 (voir
  `panzer_index_table.md` §9.2, corrigé le 27/08/2026).

**Invalidateur** : `getLeaves()` (clears au `:1167` et au retour n_rows==0) —
les deux structures meurent/revivent à chaque reconstruction du cache.

### 2.4 `readTensor` / `readSliceDirect` / `readLeavesUnion` (lecture par hyperslab)

Ce ne sont pas des caches au sens mémoïsation, mais ils **évitent** des
copies : `H5Dread` hyperslab directement dans le buffer de sortie
(`readSliceDirect` `panzerdb.cpp:890-954`), ou union de hyperslabs
(`readLeavesUnion` `:956`, fallback séquentiel si offsets non monotones).
`readSliceDirect` recalcule à la volée le volume du slice à partir de
`leaf.shape` (déjà en mémoire, zéro I/O) — calcul identique à celui de
`restoreTimeContext` (`:660-668`). C'est la contrepartie d'usage du chunk
cache HDF5 ci-dessous : le travail est fait par gros morceaux, pas par
élément.

---

## 3. Écriture

### 3.1 Buffers RAM — le « cache » principal du chemin d'écriture

| Buffer | Contenu | Tailles initiales |
|---|---|---|
| `index_buffer` | lignes 14×u64 de `index` | `reserve(index_chunk_rows × 14)` (`panzerdb.cpp:224`) |
| `data_buffer_f64/i32/c128` | valeurs brutes à append | `reserve(data_chunk_*)` (`:225-228`) |
| `data_buffer_str` | chaînes | `reserve(data_chunk_str)` (`:227`) |
| `paths_buffer` / `parent_paths_buffer` | 256 o/ligne, null-pad | croissance ×2 (BUFFER_GROWTH_FACTOR) |

**Fonctionnement** :

- `append_index_row` (`panzerdb.cpp:1856`) : `strncpy` borné dans le buffer
  de chemins (pas d'allocation de `std::string` par ligne d'index),
  insertion de la ligne de 14 u64 ; aucune écriture HDF5.
- `writeDataImpl` (`:1412`) / `writeDataSlicesImpl` (`:1600`) : calcul de
  l'`offset` via `disk_size_* + buffer.size()` (3.2), insertion dans le
  buffer typé, mise à jour des compteurs (3.3).
- `flush()` (`panzerdb.cpp:689`) : pour chaque dataset concerné,
  `H5Dget_space` → `H5Dset_extent` (extension du dataset) → `H5Dwrite` du
  bloc entier. **Un seul appel `H5Dwrite` par dataset par flush**, quelle que
  soit la granularité des `writeData*` intermédiaires. Puis invalidation du
  cache de feuilles (`:861`) + `close()` à la destruction.

**Ordre de grandeur** : une session d'écriture typique (N feuilles,
M valeurs) passe de M appels `H5Dwrite` à « nombre distinct datasets ×
nombre de flush » — en pratique 1–2 appels/dataset/fichier. C'est la
différence entre du I/O par ligne (insupportable sur Lustre) et du I/O par
bloc (natif Lustre/stripe).

**Borne mémoire** : le seul flush déclenché est l'explicite
(`flush()`), appelé par `HDF5Writer_v2::endAction`
(`hdf5_writer_v2.cpp:597`) et par le destructeur. Les buffers vivent donc
 jusqu'au prochain `endArray`/`endAction` — pour une session d'écriture
 massive (plusieurs Go) entre deux `endAction`, la RAM consommée est celle du
 fichier. (L'ancien mécanisme d'auto-flush, `autoFlushIfNeeded`, n'était jamais
 appelé ; il a été supprimé le 27/08/2026 plutôt que maintenu — si un garde-fou
 RAM devient nécessaire, réimplémenter proprement avec seuil documenté.)

### 3.2 `disk_size_*` (`panzerdb.cpp:279`, `:841-844`)

**Contenu** : taille courante (éléments) de chaque `data_raw_*` sur disque,
tenue à jour au flush.

**Fonctionnement** : les `writeDataImpl*` calculent l'`offset` de la ligne
d'index comme `disk_size_T + buffer_T.size()` — O(1), sans `H5Dget_space`
(appel HDF5, coûteux en locking). Inversement, sans ce cache, chaque ligne
d'index exigerait un `H5Dget_space` → N appels HDF5 par flush.

**Invalidateur** : rechargé par `updateDiskSizes()` à l'ouverture en APPEND
(`:279`) ; mis à jour par chaque flush.

### 3.3 `aos_time_counters` (`panzerdb.h:183`)

**Contenu** : `unordered_map<string, uint64>` — « chemin (signal ou AoS) →
prochaine slice à écrire ».

**Fonctionnement** : c'est le **cœur** du support du temps implicite. Trois
usages distincts :

1. **Numérotation séquentielle** (`writeDataSlicesImpl`
   `panzerdb.cpp:1632-1660`) : `base_time = max(counter[full_path],
   counter[dynamic_aos_path])` — le `max` avec le conteneur dynamique
   permet des écritures **chunked** (un signal écrit en plusieurs `H5Dwrite`,
   chacune partant de `base_time = counter`) et un alignement sur ce qui est
   progressivement écrit par l'AoS parent.
2. **Alignement sur la timebase** (`:1635-1650`) : pour laisser des
   **gaps** (lignes absentes), `base_time` est remonté à
   `counter[timebase] - n_slices` — sans ce cache, l'append compresserait
   la timeline du signal (régression « gaps homogènes » de 07/2026).
3. **Restore à l'ouverture** (`restoreTimeContext` `:628-693`, une fois par
   APPEND) : recomputation du `next_time` de chaque feuille à partir du cache
   de feuilles (`count/slice_volume`) pour pouvoir reprendre la numérotation
   là où le fichier s'est arrêté.

**Fonctionnement en lecture** : le même `map` est relu par
`getDynamicAOSSize` (`:2062`) et `getCurrentTimeForAOS` /
`advanceTimeForAOS` (API externe, utilisée par le plugin).

**Invalidateur** : jamais explicitement vidé (un APPEND sur un nouveau
fichier fait juste un objet neuf).

### 3.4 `cached_dynamic_aos_path` + `dynamic_aos_path_valid` (`panzerdb.cpp:3045`)

**Fonctionnement** : cache du chemin de l'AoS dynamique le plus proche dans
`array_stack` (`panzerdb.h:218-220`). Sans cache, chaque `writeDataSlices`
remonte le stack — O(profondeur) par écriture, et une **invalidation
manquante** en `beginArray(name, timebase)` serait une régression silencieuse
(deux écritures sous des `beginArray` différents avec le même `name` ne
s'aligneraient plus sur le même conteneur dynamique — la régression « gaps
homogènes » citée au §8.2 de `panzer_index_table.md`, corrigée en 07/2026).

**Invalidateur** : `invalidateDynamicAOSCache()` (`:3076`), appelé dans les
deux `beginArray(name, …)` (`:1072, 1117`) et `endArray` (`:1976`).

### 3.5 `written_metadata_schema_paths` (`panzerdb.cpp:3175`)

**Fonctionnement** : `std::unordered_set<string>` — un chemin *schema*
(d'instance, index retiré par `stripIndices`) est marqué « écrit » après le
premier `writeMetaData` et **re-vu** par `readMetadata` (évite de relire le
dataset `@key` pour une feuille qui a déjà fourni sa metadata). Sans ce
set, chaque écriture d'instances multiples d'un même type re-déclarerait
les `@key` (doublonnage de lignes dans `index`) et chaque lecture
d'instances multiples relirait les `@key` (I/O inutiles). Coût d'entrée
minime, gain sur les AoS larges (des centaines d'instances × des dizaines de
`@key`).

**Invalidateur** : jamais (session-only).

---

## 4. Caches HDF5 (FAPL / DAPL)

| Mécanisme | Valeur | Où |
|---|---|---|
| **Chunk cache** (Raw Data Cache) | 64 Mo, 10 007 slots, rétention 0,75 | `H5Pset_chunk_cache` sur le DAPL des datasets (`panzerdb.cpp:154-156`, `configureReadCache` `:338-384`) |
| **Méta-blocs** FAPL | 2 Mo aggrégés | `H5Pset_meta_block_size(fapl, 2 MiB)` (`:73`) |
| **Alignment** FAPL | objets > 4 Ko alignés sur 4 Mo (stripe Lustre) | `H5Pset_alignment(fapl, 4096, 4 MiB)` (`:70-71`) |

**Fonctionnement du chunk cache** : HDF5 matérialise les chunks (512 Ko à
1 Mo selon le usage hint) dans ces 64 Mo, LRU sur 10 007 slots. C'est ce qui
rend rentable la lecture « pailletée » (hyperslab discontiguous, cf.
`readLeavesUnion`) : un chunk touché une fois reste en RAM, pas de re-I/O si
on relit la même fenêtre de temps. En écriture, il amortit aussi la
compression (GZIP niveau 1) : les chunks compressés sont écrits en
bloc, décompressés à la lecture.

**Impact** : sur un workflow « lire la même slice 10 fois de suite »
(interpol, vérifications), c'est le mécanisme qui rend le deuxième passage
~30× plus rapide (ordre de grandeur). Sur un workflow « balayé » (lecture
de N slices distinctes), il est quasi neutre (1:1 miss).

---

## 5. Classement par impact performance (avis, sans mesure)

Critères utilisés : **fréquence d'appel dans le chemin critique** ×
**travail éliminé par appel** × **coût d'entrée** (première
reconstruction). Ordre décroissant.

**1. Buffers d'écriture RAM (3.1) — impact : CRITIQUE**
Chaque `writeData*` appelle HDF5 0 fois au lieu de 1 : c'est la raison pour
laquelle PanzerDB tient sur Lustre. Sans eux, le facteur est de l'ordre de
`N_feuilles / (nb_datasets × nb_flushes)` ≈ 10³–10⁴ sur des fichiers IMAS.

**2. `cached_leaves` + `leaf_lookup` + `parent_lookup` (2.1, 2.2) — impact : HAUT**
Fondation de **tout** le chemin de lecture : `getAOSShape`,
`pz_read*Data_by_index`, `readInterpolatedData`, `readMetadata`,
`getTimeIndex`, `nearestAvailableSliceIndex` passent tous par `getLeaves()`.
Sans eux, chaque appel = `H5Dget_space` + `H5Dread` de `index` + `paths` +
balayée O(N). Le gain est d'autant plus visible que N est grand (typ. 10⁴–10⁵
pour IMAS, ~1 Mo de cache ≈ 1 ms d'équipement au lieu de dizaines de `H5D*`
par lookup). `leaf_lookup` en particulier transforme des comparaisons de
chaînes O(len) × N en un `find` O(len).

**3. HDF5 chunk cache (4) — impact : HAUT (lectures itératives), NÉGLIGEABLE (balayé)**
64 Mo de chunks en RAM : c'est ce qui rend rentable l'interpolation itérative
(le même chunk est relu) et `readLeavesUnion` (les chunks voisins sont
chauds). En balayé pur (une fois par slice), l'effet est nul. Position 3 et
non 2 car il est commun et pas sous le contrôle direct du code PanzerDB —
mais c'est le plus gros levier « gratuit » sur les cas d'usage INTERACTIF
(démos ITER, replay).

**4. `aos_time_counters` (3.3) — impact : MOYEN, mais CORRECTNESS-DEPENDANT**
O(1) par `writeDataSlices`, évite une lecture HDF5 pour savoir « où j'en
suis ». Surtout, il est **nécessaire** pour les gaps / chunked writes /
alignement timebase — sans lui ce n'est pas une question de perf, c'est une
question de bon sens.

**5. `disk_size_*` (3.2) — impact : MOYEN**
Évite un `H5Dget_space` par appel `writeData*` → gain direct sur le
chemin d'écriture chaud. Coût d'entrée : 1 `H5Dget_space` par dataset à
l'ouverture. Gain : N appels HDF5 épargnés.

**6. `written_metadata_schema_paths` (3.5) — impact : MOYEN (AoS large), FAIBLE (AoS petit)**
Évite du doublon en écriture et du re-read en lecture. Gain proportionnel au
nombre d'instances × de `@key`. Négligeable pour des scalaires.

**7. `cached_dynamic_aos_roots` + `max_time_at_dynamic_root` (2.3) — impact : FAIBLE (O(log n)), CORRECTESS pour `getAOSShape`**
L'itération sur « quelques » racines plutôt que N feuilles est un gain
constant (facteur ≈ N / #racines). Le cache `max_time_at_dynamic_root`
rend `getAOSShape` O(1) au lieu d'un parcours de children.

**8. `cached_dynamic_aos_path` (3.4) — impact : FAIBLE, mais CORRECTESS**
Évite O(profondeur) par `writeDataSlices`. Gain microsecondes. Valeur
principale : la traçabilité de l'invalidation.

**9. `scratch_str` (§1-11) — impact : FAIBLE**
Tampon réutilisé par `pz_readStringData_by_index`
(`panzerdb.cpp:2613`) : évite N allocations de `std::string` sur une lecture
de liste de chaînes.

---

## 6. Tableau récapitulatif de la cohérence (invalidation)

| Cache | Point de build | Point d'invalidation | Risque |
|---|---|---|---|
| `cached_leaves` | `getLeaves()` | `leaves_cache_valid=false` post-`flush()` (`:861`), ouverture WRITE/APPEND (`:243, 265`) | **Faible** : cohérence stricte ; coût de re-read intégral en APPEND (2.1) |
| `leaf_lookup` / `parent_lookup` | `getLeaves()` | clear systématique avant re-pop (`:1164-1166`) | **Faible** (reconstruction toujours intégrale) |
| `cached_dynamic_aos_roots` | `getLeaves()` | clear à chaque reconstruction | **Faible** |
| `max_time_at_dynamic_root` | `rebuildDynamicRootTimeIndex()` | clear dans `getLeaves` + n_rows==0 | **Faible** : construction idempotente |
| Buffers (3.1) | `append_index_row`, `writeData*` | `flush()` | **Aucun** (données non-durables par conception) |
| `disk_size_*` | `flush()` | `updateDiskSizes()` en APPEND | **Faible** |
| `aos_time_counters` | `writeDataSlices*`, `restoreTimeContext` | jamais | **Faible** (session-only) |
| `cached_dynamic_aos_path` | `getDynamicAOSPath()` | `invalidateDynamicAOSCache()` dans les 3 `beginArray`/`endArray` (`:1072, 1117, 1976`) | **Moyen** historiquement (régression 07/2026) — à garder sous test |
| `written_metadata_schema_paths` | `writeMetaData` / `readMetadata` | jamais | **Faible** |

---

## 7. Points d'attention / actions suggérées

> Statut au 27/08/2026, après nettoyage : les anciens points 1 (code mort),
> 2 (`time_index_valid` stale), 3, 5, 6 sont **dégagés par la suppression**
> (cf. `panzerdb_removing_dead_codes.md` §7). Il reste :

1. **Coût de la reconstruction post-flush** (2.1-3) : dans un mode APPEND
   écrit-puis-lit alterné, chaque `flush()` suivi d'une `getLeaves()` coûte
   un re-read intégral. Si un tel pattern devient fréquent, envisager un
   append-in-place (réutiliser le `vector<Leaf>`, ne re-read que la delta
   `index[N_old … N)` et ré-ancrer les `string_view` si besoin) — à
   implémenter proprement avec un test dedicated (cf.
   `panzerdb_removing_dead_codes.md` §2, L0.1 option B).

2. **Borne mémoire des buffers d'écriture** (3.1) : le seul flush est
   l'explicite d'`endAction` ; une session massive entre deux `endAction`
   consomme la RAM du bloc en cours. Documenter cette borne ; réimplémenter
   un auto-flush seulement si un besoin réel apparaît.

3. **`scratch_str` global** (2.4, §1-11) : tampon de classe, non réentrant ;
   acceptable car PanzerDB est mono-session, mais à garder en tête si des
   threads partagent une instance.

---

*Fin du rapport.*
