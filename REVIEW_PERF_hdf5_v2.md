# Revue performance — backend v2 (`src/hdf5`)

Relecture de `panzerdb.{h,cpp}`, `hdf5_writer_v2`, `hdf5_reader_v2`, `iread_strategy.h`
et des 3 stratégies de lecture, en tenant compte de `to_improve.md` (points 1–13 déjà
traités). Voici ce qui reste, par ordre d'impact.

## A. Écriture — le gros coût restant : rechargement complet de l'index par SLICE_OP (quadratique)

1. **Un PanzerDB APPEND neuf par pas de temps.** `hdf5_events_handler.cpp:35` appelle
   `setWriteStrategy(SLICE_OP)` par événement → `HDF5Writer_v2::setWriteStrategy`
   (`hdf5_writer_v2.cpp:118`) crée un nouveau PanzerDB → `init()` →
   `restoreTimeContext()` → `getLeaves()` qui relit **tout** `/index` et **tout**
   `/paths` (panzerdb.cpp:1344-1361). Sur le fichier de benchmark (171k lignes ≈ 44 Mo
   de tables), écrire T pas coûte O(T × N). Idées :
   - **Garder l'APPEND engine ouvert** entre opérations de la même session (ne le
     re-créer que si le gid change), comme le fait déjà le reader (`session_db`).
   - **Rechargement incrémental de `getLeaves()`** : `/index` et `/paths` sont
     append-only ; mémoriser `n_rows_loaded` et ne lire que les lignes nouvelles
     (`H5Sselect_hyperslab` à partir de l'ancien extent), en pushant un nouveau bloc
     dans `cached_paths_blocks` (le `std::list` est déjà fait pour ça). Bénéficie aussi
     au **lecteur SWMR** (`invalidateLeafCache` → refresh O(delta) au lieu de O(N)).
2. **`restoreTimeContext` (panzerdb.cpp:691-755)** : alloue `aos_path + "/"` par
   (leaf × root) dans la boucle interne, et copie chaque `leaf.path` en `std::string`
   pour la map. Pré-calculer les préfixes racine une fois, comparer en `string_view`,
   et incrémenter seulement sur les nouvelles lignes si A1 est fait.
3. **`flush()` (panzerdb.cpp:759)** refait un `H5Dget_space` par dataset pour connaître
   l'extent courant alors que `disk_size_*` est déjà caché — réutiliser les compteurs
   (gagner 6 appels méta HDF5/flush).
4. **Écrire les listes/chaînes en bulk en UNE ligne d'index** :
   `writeDataSlices(const char**)` (panzerdb.cpp:2097) émet une ligne par pas là où le
   path numérique `writeDataSlicesImpl` en émet une seule pour `n_slices` — le reader
   sait déjà gérer une feuille multi-step. Réduit le nombre de lignes d'index (et le
   travail de A1) pour les signaux temporels de chaînes.
5. **`H5Pset_alignment(fapl, 4096, 4MB)` (panzerdb.cpp:75)** : chaque objet ≥4KB est
   aligné sur 4MB → fichiers de test/petits pulses gonflés et trous de padding. Adapter
   l'alignement à la taille attendue (ou au moins au mode WRITE vs APPEND).

## B. Lecture — `getLeaves()` et lectures de données

6. **parent_path = préfixe du path, pas un nouveau buffer.** La règle M1
   (panzerdb.cpp:1379-1397) garantit que le parent est le path complet amputé de 1–2
   segments : `leaf.parent_path` peut être un `string_view` **dans le même bloc**
   `paths_data` (fin calculée par `memrchr`). Supprime l'allocation de
   `cached_parent_paths_blocks` (N×256 B, ~44 Mo sur le fichier benché), le `memcpy`
   par ligne et le `std::string` temporaire de `parent_path_of`.
7. **`Leaf::shape` = `std::vector<size_t>`** → une malloc par ligne (171k par rebuild).
   Un tableau inline `size_t shape[6]` + `ndim` élimine ça.
8. **`readLeavesUnion` (panzerdb.cpp:1102) lit `[lo,hi)` entier** : quand les feuilles
   d'un signal sont entrelacées avec d'autres signaux écrits au même pas (cas typique
   temps-réel), la range couvre la data des *autres* signaux — amplification ≈ nb de
   signaux co-écrits (28× sur le bench). Les offsets étant monotones par pas, la
   sélection est **quasi-uniforme** : détecter le stride
   (`offset[i+1]-offset[i]`) et faire un `H5Sselect_hyperslab` **stride**
   (start, stride, count), ou fusionner les blocks contigus. De plus, utiliser une
   **memspace sélectionnée** pour lire directement dans le buffer de l'appelant —
   supprime `tmp` (pic mémoire ×2) et le scatter memcpy. Même remarque pour
   `readStringBulk` dans `read_dataset_globally` (iread_strategy.h:1133-1138).
9. **`getTimeIndex` (panzerdb.cpp:2401) relit la timebase depuis le disque à chaque
   appel.** `SliceReadStrategy::beginReadArraystructAction`
   (slice_read_strategy.cpp:27) l'appelle à chaque ouverture d'AoS → relecture du
   timebase plusieurs fois par `get()`. Ajouter un cache des valeurs de timebase dans
   PanzerDB (clé = path, invalidé par `leaf_cache_generation`), comme le fait déjà
   `time_values_cache` côté stratégie.
10. **`nearestAvailableSliceIndex` (panzerdb.cpp:2528-2539)** : recherche **linéaire**
    index par index dans les gaps, chaque test faisant jusqu'à 3 hash lookups +
    construction de `substituted_path`. Parcourir les lignes du bucket (déjà écrites en
    ordre de temps) avec une recherche binaire sur `time_index` → O(log) au lieu de
    O(gap).
11. **`readStringDataByIndex` (panzerdb.cpp:2933)** lit **toutes** les slots de la
    feuille (`scratch_str.resize(count)`) pour extraire un seul pas : hyperslab exact
    `[offset + local_step*element_size, element_size)` quand la feuille a plusieurs pas.
12. **`getAOSShape` (static, panzerdb.cpp:2350-2390)** rescanne les enfants avec
    `stoll`+allocs à chaque appel (appelé par AoS ouvert du slice strategy). Cacher le
    résultat par path pour la durée de la génération de cache.

## C. Table d'index — évolutions qui valent le coup

13. **Interning des chemins (plus gros gain structurel)** : colonne `path_id` (u32)
    dans `/index` + table `/path_table` des chemins uniques. Les lignes répétées
    (signal × 8150 pas, même path) passent de 256 B/ligne à 4 B ; le bench (171k
    lignes) : ~44 MB → ~1 MB lus en RAM. `leaf_lookup`/`parent_lookup` construits sur
    des ids (hash u32 au lieu de strlen+hash de chaîne) ; parent_path devient
    `(path_id, n_strip)` ou un id. C'est l'évolution n°1 du schéma.
14. **Packing des colonnes** : ndim/shape/time/flags/parent tiennent en u32
    (offset/count en u64) → dataset compound : 96 B → ~56 B/ligne (-40 % d'I/O index).
    Combiné avec 13 : ~24 B/ligne.
15. **Bit de rôle dans `flags`** (timebase / scalaire / liste / signal temporel) :
    achève le point 3 "still open" de to_improve.md — supprime
    `stripIndices`/`isTimebaseDataset`/heuristiques de shape à chaque lecture, et rend
    le contrat scalaire-vs-liste indépendant de l'inférence legacy.
16. **Attribut de version de format** explicite (au lieu d'inférer 12/14 colonnes via
    `dims[1]`) pour débloquer 13/14 sans devinettes.
17. (Optionnel) **Table de récap par signal** (path_id → range de lignes, min/max
    time, dtype) pour que `read_dataset_globally`/`getWholeDynamicSignal` trouvent
    leurs feuilles par plage de lignes sans buckets de strings.

## D. Vérification / instrumentation

18. Étendre `tests/hdf5_backend/bench_read.cpp` avec un **bench d'écriture SLICE_OP**
    (T puts sur un fichier à N croissant) pour confirmer la courbe quadratique de A1
    avant/après — c'est la seule path non mesurée par to_improve.md.

**Top 5 si un seul effort** : (1) engine APPEND partagé + rechargement incrémental de
l'index, (6) parent_path en string_view préfixe, (8) hyperslab stride + lecture directe
dans le buffer sortant, (13) path interning, (9) cache des timebases.
