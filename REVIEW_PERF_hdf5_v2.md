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
   — *fait en partie (2026-10-11, cf. E6)* : plus d'allocation par feuille ni par root,
   comparaison en préfixe, snapshot `file_time_base` réservé à APPEND. Le reliquat
   (« incrémenter seulement sur les nouvelles lignes ») dépend de A1.
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
    construction de `substituted_path`. **PARTIELLEMENT TRAITÉ (2026-10-10)** : le
    balayage des *lignes d'un path* (le coût réel de chaque test) est devenu une
    recherche binaire sur `time_index` via `PanzerDB::rowsForTime()` — cp TIMERANGE
    12 949 ms → 1 763 ms, `isTimeInLeaf` sorti du profil (to_improve.md point 14).
    Reste la boucle **index par index** sur `time_basis` : interroger directement
    les lignes triées du bucket en binaire supprimerait aussi cette boucle.
11. **`readStringDataByIndex` (panzerdb.cpp:2933)** lit **toutes** les slots de la
    feuille (`scratch_str.resize(count)`) pour extraire un seul pas : hyperslab exact
    `[offset + local_step*element_size, element_size)` quand la feuille a plusieurs pas.
12. **`getAOSShape` (static, panzerdb.cpp:2350-2390)** rescanne les enfants avec
    `stoll`+allocs à chaque appel (appelé par AoS ouvert du slice strategy). Cacher le
    résultat par path pour la durée de la génération de cache.

## C. Table d'index — évolutions qui valent le coup

13. **Interning des chemins (plus gros gain structurel)** — **FAIT (layout v3)** :
    colonne `path_id` (u32) dans `/index` + table `/path_table` des chemins uniques
    (`panzerdb.h:152-154` `PANZER_INDEX_VERSION_V3` / `PANZER_INDEX_COLUMNS_V3`,
    création `panzerdb.cpp:206-208`, lecture `panzerdb.cpp` branche v3 de `getLeaves`).
    Les nouveaux fichiers écrivent en v3 ; le lecteur garde la branche v2/legacy
    (12/14 colonnes + `/paths`) pour les fichiers existants — c'est là que restent les
    2 allocations `std::string`/feuille et les 2 blocs de 256 B.
14. **Packing des colonnes** — **FAIT (layout v3)** : `/index` v3 = 7 u64 packés.
    Un fichier v3 se recharge donc déjà au quart du volume ; rien de fait pour les
    fichiers v2.
15. **Bit de rôle dans `flags`** (timebase / scalaire / liste / signal temporel) :
    achève le point 3 "still open" de to_improve.md — supprime
    `stripIndices`/`isTimebaseDataset`/heuristiques de shape à chaque lecture, et rend
    le contrat scalaire-vs-liste indépendant de l'inférence legacy.
16. **Attribut de version de format explicite** — **FAIT** : attribut `index_version`
    lu à l'ouverture (branche READ de `init()`) avec repli sur `dims[1]` pour les
    fichiers sans attribut.
17. (Optionnel) **Table de récap par signal** (path_id → range de lignes, min/max
    time, dtype) pour que `read_dataset_globally`/`getWholeDynamicSignal` trouvent
    leurs feuilles par plage de lignes sans buckets de strings.

## D. Vérification / instrumentation

18. Étendre `tests/hdf5_backend/bench_read.cpp` avec un **bench d'écriture SLICE_OP**
    (T puts sur un fichier à N croissant) pour confirmer la courbe quadratique de A1
    avant/après — c'est la seule path non mesurée par to_improve.md.

**Top 5 si un seul effort** (mis à jour le 2026-10-11 après mesures) : (1) engine APPEND
partagé + rechargement incrémental de l'index — c'est ce qui domine le SLICE cp
(≈95-150 ms d'index par opération), (8) hyperslab stride + lecture directe dans le buffer
sortant (`readLeavesUnion` : 28× d'amplification sur cp), (9) cache des timebases,
E5 (**invalidé** — voir ci-dessous), bookkeeping par nœud du chemin TIMERANGE (cause du
1,6× mesurée : `stripIndices` + `stringstream`, hashing, churn d'allocation —
1,6 → 1,4 après la première correction).

## E. Nouveaux constats (2026-10-10) — lignes vérifiées dans le code courant

### Correction (à traiter avant les performances)
E1. **`readTensor` numeric ignore le statut de `H5Dread`** (`panzerdb.cpp:1865-1868`) :
le bloc `if (status < 0) { // Handle HDF5 read error }` est **vide** → un échec de
lecture remonte comme un succès avec un buffer non initialisé. La spécialisation
`std::string` jette, elle. — **TRAITÉ (2026-10-11)** : `H5Dread`, `H5Dget_space` et
`H5Sselect_hyperslab` lèvent maintenant `ALBackendException` (fermeture poignées +
`mem_type` COMPLEX128 avant le throw). Test `test_read_tensor_contract.cpp` : 3 échecs
contre le code précédent, OK après (to_improve.md point 15).
E2. **`readTensor<T>` ne valide pas `T` contre le dtype du leaf** : `mem_type` vient de
`leaf.flags >> 4`, jamais de `T`. Un `readTensor<int32_t>` sur un leaf FLOAT64 écrit
8 octets/élément dans un buffer de `count` `int32_t` → débordement. Le chemin chaud
alloue d'après `actual_type_enum` (`iread_strategy.h:1264-1269`) donc n'est pas exposé ;
`iread_strategy.h:842 readLeafData` l'est — il choisit `T` d'après le `datatype` **demandé
par l'AL**, pas d'après le leaf, donc un désaccord dictionnaire/fichier atteint le débordement
— et `readLeafData` n'a **aucun appelant** (code mort). — **TRAITÉ (2026-10-11)** :
garde de largeur dans `readTensor` (`sizeof(T) < taille_stockée` → `ALBackendException`).
Garde de **largeur** et non d'égalité, parce que le sens inverse est un idiom vivant :
`readInterpolatedData` (`panzerdb.cpp:3837`) route `integer_data` vers
`readDataByIndex(double**)`, qui sur-alloue 8 o/élément et dont le consommateur relit le
buffer en int32 — donc un `assert` d'égalité exact aurait cassé la lecture des signaux
entiers en mode SLICE. Test `test_read_tensor_contract.cpp` (une `Leaf` fabriquée à la main
par type) : échec reproduit avant le correctif, et les deux sens figés après.

### Performances
E3. **TIMERANGE avec resampling lit la tranche 0 deux fois** : la sonde de forme
(`timerange_read_strategy.cpp:235-245`, sur `requested_times[0]`) est `free()`, puis la
boucle `for (k = 0 …)` (`:259-276`) relit exactement la même tranche. — **TRAITÉ
(2026-10-11)** : payload de la sonde recopié dans le slot 0, boucle à partir de `k = 1`.
**Aucun gain mesurable** (1 lecture économisée sur 8150 en plage complète ; bruit de
±30 ms) — to_improve.md point 15. Gain de travail gaspillé et de fuite, pas de perf.
E4. **TIMERANGE sans resampling relit le signal entier puis le tronque en RAM**
(`timerange_read_strategy.cpp:292` `read_dataset_globally`, puis `:360-367` malloc +
memcpy + `free(full_data)`) : lire `[tmin, tmax]` coûte la lecture complète + 2 copies.
— **NON ATTEIGNABLE, mesuré (2026-10-11)** : la branche (`dynamic_index == -1` avec
`timebasename` non vide) a été instrumentée d'un compteur ; **0 hit** sur cp (5 modes de
bench), **0 hit** sur eq (5 modes), **0 hit** sur reflectometer_profile_v2 (3 plages),
**0 hit** sur les 85 tests. Affirmation précédente de cette revue (« c'est le vrai levier
TIMERANGE », et le 1,6× GLOBAL→TIMERANGE qu'elle devait expliquer) est **fausse** : les
deux modes lisent un contenu identique (244 513 nœuds, 19 780 050 doubles, même hash),
donc l'écart vient d'ailleurs et demande sa propre mesure. La branche reste un pire cas
théorique ; aucune mesure ne justifie d'y toucher.
E5. **Re-tri de buckets déjà triés par temps** : `iread_strategy.h:1070`
(`sorted_leaves`), `panzerdb.cpp:2841` (`getTimeIndex`), `:3069` `:3258` `:3532` `:3707`
(les quatre `readXByIndex`), `:3944` (`getWholeDynamicSignal`).
— **INVALIDÉ comme explication du 1,6× (mesuré 2026-10-11)** : dans les profils `perf` de la
marche TIMERANGE cp, les seuls `std::sort` visibles sont celui de
`ReadIndex::build_indexes()` (0,14-2,85 % selon le mode — il appartient au poste A1, pas ici)
et celui de `read_dataset_globally` (0,01 %). Les tris cités par E5 (les quatre
`readXByIndex`, `getTimeIndex`, `getWholeDynamicSignal`, `sorted_leaves`) n'apparaissent pas :
sur la marche par nœud leur bucket contient un seul élément, et la branche d'agrégation
`time_index == -1` n'est pas ce chemin. À ne traiter que pour de gros buckets d'agrégation.
E6. **`restoreTimeContext()` s'exécute aussi en mode READ** — appelé à la fin d'`init()`
après le if/else de mode (`panzerdb.cpp:408`) ; alloue `aos_path + "/"` par
(feuille × root) plus une copie `std::string` par feuille. Mesuré **1,65 %** de la
marche SLICE cp (le moteur étant recréé à chaque opération, cf. A1).
— **TRAITÉ (2026-10-11)**, et la mesure de la revue était **trop basse** : la boucle coûte
**11,3 ms par ouverture** sur cp (sonde chrono, 171 165 feuilles), soit ~4,7 % d'une marche
SLICE à 20 get — pas 1,65 %. La **prémisse est en revanche fausse** : la table n'est pas
inutile en READ, `getDynamicAOSSize()` y est chargé (`iread_strategy.h:429`,
`direct_reader.cpp:49`). Corrigé : clé `std::string` réutilisée, parent AoS trouvé par
préfixe sur `leaf.path` (plus de concaténation par root), snapshot `file_time_base` réservé
à APPEND. Équivalence prouvable sur les trois formats d'index : **131 598 entrées de
compteurs comparées, 0 écart** ; test goupille `test_engine_time_context` (passe aussi sur
HEAD). Gain le plus reproductible, sur le chrono d'ouverture (`index`) en paires A/C
alternées : **−4 à −5 ms/ouverture sur cp** (95 → 92 ms en régime chaud), **−5 à −7 ms sur
eq** (40 → 34 ms) — la boucle passant de 11,3 à 6,2 ms sur cp. Temps mural : cp SLICE
−4,7 % (gain de la boucle), eq SLICE −3,0 % et eq TIMERANGE −3,5 % (gain du snapshot
supprimé, la carte de eq comptant 43 598 entrées) ; **GLOBAL sous le plancher de bruit
±4 %** dans les deux sens — ne pas chercher de signal GLOBAL sur ce poste. Empreintes
inchangées ; suite 88/88. Détails : `to_improve.md` point 20.
E7. **Buffers de chaînes zero-initialisés pour rien** : `panzerdb.cpp:1917` et `:1962`
(`std::vector<char> buf(n * STRING_MAX_LEN, 0)` puis `H5Dread` qui écrase tout).
E8. **Métadonnées lues puis jetées** : `slice_read_strategy.cpp:111`,
`iread_strategy.h:978`, `timerange_read_strategy.cpp:437` appellent `readMetadata` et
descartent le résultat ; `readMetadata` marque le schema traité, donc la valeur n'est
plus récupérable ensuite → une relecture disque inutile par schema et par session.
E9. **API directe** : `direct_access_api.cpp:321-327` relit la feuille sonde dans la
boucle qui suit ; `get_time_vector` fait un `readScalar` (= 1 `H5Dread`) par pas alors
que `getWholeDynamicSignal("time")` est appelé juste en dessous.
E10. **Signaux entiers lus via l'API double** (constaté en cherchant E2) :
`readInterpolatedData` envoie `integer_data` dans `readDataByIndex(double**)`
(`panzerdb.cpp:3837`), qui fait `malloc(count * sizeof(double))` là où 4 o/élément
suffiraient — ×2 sur chaque tranche entière lue en SLICE/TIMERANGE. Les valeurs sont
justes (le consommateur relit en int32), donc c'est uniquement du gaspillage ; le
corriger veut dire router `integer_data` vers `readIntDataByIndex`, ce qui change la
taille du bloc rendu à l'AL — à faire avec le test AL entier, pas à l'aveugle.
E11. **`readLeavesUnion` choisit le DATASET d'après le `DataType` passé par l'appelant,
pas d'après le leaf** (`panzerdb.cpp:1275`) : les quatre agrégations lui passent leur
propre type (`:3102` FLOAT64, `:3565` COMPLEX128, …). Si `readDataByIndex(time_index=-1)`
était appelé sur un path stocké en INT32 avec ≥2 feuilles, la plage serait lue dans
`data_raw_f64` aux offsets de `data_raw_i32` → données d'un autre signal. **Non démontré
atteignable** : les GLOBAL d'entiers passent par `read_dataset_globally` avec
`actual_type_enum` pris du fichier (`iread_strategy.h:1278`), et `direct_reader.cpp:75`
est codé en dur sur `z_ion` (double). À verrouiller par un test si l'exposition apparaît.

### Fausse alerte — ne pas « corriger »
`cached_paths_blocks` / `cached_parent_paths_blocks` **ne fuient pas** : la file est
vidée à chaque rebuild (`panzerdb.cpp:1514-1515`), avant les deux branches de layout ;
elle ne contient qu'un bloc. Le coût réel reste le rechargement **total** de `/index` +
`/paths` à chaque rebuild (point A1), pas l'accumulation de blocs.
