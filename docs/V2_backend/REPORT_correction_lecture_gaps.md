# Rapport — Correction de la lecture des signaux avec trous (gaps) — backend HDF5 v2

**Date** : 26/08/2026
**Périmètre** : `src/hdf5/` (backend HDF5 v2 / PanzerDB) + `tests/hdf5_backend/`
**Plan de référence** : `docs/PLANS_correction_lecture_gaps.md`

---

## 1. Contexte et mission

Le backend HDF5 v2 (PanzerDB) stocke les données de façon *columnar* : un seul
dataset par type + une table d'index (`/index`) qui référence chaque bloc de
données par (`chemin`, `time_index`, `offset`, `count`, `shape`). Quand un
utilisateur **n'écrit pas un signal à certaines slices** (append sélectif), des
« trous » apparaissent : la table d'index ne contient aucune ligne pour
`(signal, time_index)` de la slice manquante.

Deux modes de lecture existent :
- **lecture globale** (GLOBAL_OP) : renvoie tout le signal ;
- **lecture par slice temporelle** (SLICE_OP, timerange) : renvoie la valeur à un
  instant donné, avec interpolation (`closest`, `previous`, `linear`).

**Mission** : vérifier que le lecteur se comporte correctement dans les deux
modes en présence de trous, et corriger le cas défaillant.

### Verdict de l'analyse initiale

| Mode de lecture | Comportement initial |
|---|---|
| Lecture globale (`read_dataset_globally`) | **Correct** — retourne 0 → AL « donnée non disponible » |
| Lecture par slice (`readInterpolatedData`) | **Bug** — un fallback renvoyait silencieusement la valeur d'une slice arbitraire |

---

## 2. Analyse du bug

### 2.1 Cause racine

`PanzerDB::readInterpolatedData` (src/hdf5/panzerdb.cpp) contenait un lambda
`read_slice_with_fallback` : si `pz_read*_Data_by_index` échouait pour la slice
demandée (trou), il **relisait la slice 0 du même signal** et renvoyait sa
valeur comme si elle était la bonne.

Conséquences mesurées sur le fixture de référence
(`test_dynamic_aos_gap` : `sig2 = 200+i`, trou à `t=0.5`/slice 5) :

| Demande | Valeur renvoyée (bug) | Valeur attendue |
|---|---|---|
| `closest` à t=0.5 | **200** (valeur de la slice 0) | 204 (slice 4) |
| `previous` à t=0.5 | **200** | 204 |
| `linear` à t=0.5 | **≈203** (interp. entre 200 et 206) | 205 (interp. entre 204 et 206) |
| signal jamais écrit | valeur aléatoire de la slice 0 | « non disponible » |

Le test existant `test_dynamic_aos_gap.cpp` ne détectait pas le bug : il ne
lisait qu'à `t=0.6` (slice existante) et son bloc « Linear Interp » était vide.

### 2.2 Point subtil découvert pendant l'implémentation

Le stockage des feuilles dépend de la structure :
- signal **directement sous un AoS dynamique** : une seule feuille générique
  (`AOS/chemin/signal`) portant N rangées (une par slice) ;
- signal **dans un AoS imbriqué dans un AoS dynamique** (ex.
  `static_aos/0/dynamic_aos/signal`) : feuilles stockées **par slice**, le
  chemin écrit contenant l'index de slice (`static_aos/0/dynamic_aos/0/signal`,
  `static_aos/0/dynamic_aos/1/signal`, …) — chaque `time_index` est donc une
  **lignée d'index différente** (vérifié avec `imas_h5ls` et en dumpant la
  table d'index).

La résolution d'une slice disponible doit donc reproduire les trois stratégies
de chemin du lecteur existant (`pz_readData_by_index`) :
1. **direct path** ;
2. **generic path** (index dynamique de l'AoS retiré) ;
3. **substituted path** (index dynamique remplacé par l'index cherché).

---

## 3. Sémantique cible (validée)

| Mode | Comportement en cas de trou |
|---|---|
| `closest` / `undefined` | slice **disponible la plus proche en temps réel** ; ex-aequo → **index inférieur** |
| `previous` | dernière slice disponible **≤ t** ; sinon la 1ʳᵉ disponible |
| `linear` | interpolation entre la dernière disponible ≤ t et la 1ʳᵉ disponible ≥ t, **facteur calculé sur les temps réels** des slices résolues (pas sur l'index demandé) |
| signal jamais écrit | `readInterpolatedData` → -1 → `read_ND_Data` → 0 → AL applique `setDefaultValue` (−9.0E40 scalaire / « non disponible ») |

La lecture globale (`read_dataset_globally`) n'a pas été modifiée : elle était
déjà correcte.

---

## 4. Implémentation

### 4.1 Fichiers modifiés

| Fichier | Changement |
|---|---|
| `src/hdf5/panzerdb.h` | Déclaration publique de `nearestAvailableSliceIndex` (~25 lignes avec doc) |
| `src/hdf5/panzerdb.cpp` | Nouvelle implémentation `nearestAvailableSliceIndex` (~90 lignes) ; réécriture du corps de `readInterpolatedData` (suppression du fallback, ~-60 / +20 lignes) |
| `src/hdf5/CMakeLists.txt` | 3 `al_add_hdf5_test` (3 lignes) |
| `tests/hdf5_backend/test_dynamic_aos_gap.cpp` | Renfort : lecture du trou en closest/previous/linear + bloc linear complété (≈+120 lignes) |
| `tests/hdf5_backend/test_gap_closest_prev_read.cpp` | **Nouveau** test |
| `tests/hdf5_backend/test_gap_linear_read.cpp` | **Nouveau** test |
| `tests/hdf5_backend/test_gap_timerange_read.cpp` | **Nouveau** test |
| `docs/PLANS_correction_lecture_gaps.md` | Plan de référence (pré-existant, validé) |

### 4.2 `nearestAvailableSliceIndex` (panzerdb.cpp, après `isTimeInLeaf`)

Responsabilité : donné un chemin de signal et un index temporel demandé,
renvoyer un index **pour lequel la donnée existe réellement**, ou -1.

Algorithme :
1. **Identification du contexte** : parmi les racines d'AoS dynamiques
   (`cached_dynamic_aos_roots`), trouver celle dont le chemin demandé est le
   préfixe ; extraire le suffixe restant (ex. `/ion/0/signal_1d`).
2. **Tête d'essai `available(idx)`** : la donnée existe à `idx` si une des
   feuilles des 3 chemins candidats (direct / générique / substitué) couvre
   `idx`, au sens d'`isTimeInLeaf` (intervalle
   `[time_index, time_index + count/slice_volume]`).
3. **Résolution** :
   - `idx` demandé disponible → renvoyé tel quel ;
   - sinon recherche du **plus grand disponible ≤ idx** et du
   - **plus petit disponible ≥ idx**, en balayant depuis `idx`
     (le nombre de slices d'un IDS est de l'ordre de quelques dizaines à
     quelques milliers : coût linéaire acceptable, pas de cache nécessaire) ;
   - application de la **direction** souhaitée :
     - `direction = -1` → le plus grand ≤ (sinon le plus petit ≥) ;
     - `direction = +1` → le plus petit ≥ (sinon le plus grand ≤) ;
     - `direction = 0` → le plus proche **en temps réel** via `time_basis`
       (ex-aequo → index inférieur).

### 4.3 `readInterpolatedData` (réécrit)

Suppression complète du lambda `read_slice_with_fallback` (≈ 60 lignes) et du
chemin mort « bulk/split ». Nouveau corps :

```text
slice_index  = getSlicesTimesIndices(time, time_basis, times_indices, interp_mode)
request_sup  = times_indices[SLICE_SUP]

# côté « inf » : closest/undefined = plus proche en temps (0) ; otherwise le plus grand ≤ t (-1)
inf_direction = -1 if (interp in {previous, linear}) else 0
inf_i = nearestAvailableSliceIndex(path, slice_index, inf_direction, time_basis, time)
if inf_i == -1 → return -1                      # signal jamais écrit

sup_i = inf_i
if interp == linear and request_sup != slice_index:
    sup_i = nearestAvailableSliceIndex(path, request_sup, +1, time_basis, time)
    if sup_i == -1: sup_i = inf_i
    if sup_i < inf_i: swap(inf_i, sup_i)

# dispatch de lecture simple, sans fallback (char / complex / double)
data_inf  = read_slice_at(inf_i)

if sup_i == inf_i or interp != linear → return data_inf

data_sup  = read_slice_at(sup_i)  (sinon -1 + free(data_inf))

slices_times[SLICE_INF] = time_basis[inf_i]
slices_times[SLICE_SUP] = time_basis[sup_i]

interpolate(datatype, shape_prod, y_slices, slices_times, time, data_out, interp_mode)
gestion mémoire des buffers inchangée
```

Points de vigilance :
- le **facteur d'interpolation** est calculé par `DataInterpolation::interpolate`
  à partir des **temps réels** des slices effectivement lues, pas des indices
  demandés → valeurs strictement correctes ;
- la gestion mémoire (free conditionnel du résultat vs buffers sources) est
  conservée telle quelle ;
- aucune signature publique d'interface n'a changé.

---

## 5. Tests

### 5.1 Nouveau : `test_gap_closest_prev_read`

Fixture : IDS `test_ids` → `A[1]` (statique) → `B[5]`
(dynamique, `homogeneous_time=0`) sur 5 slices écrites (t=0.0…0.4 step 0.1).
Les signaux sont `sigA = 100+i`, `sigB = 200+i`, `sigC = 300+i` (scalaire, sans
factorisation ×10 — ce facteur figure seulement dans les tests linear/timerange
où les valeurs restent lisibles et distinctes).

| signal | trous |
|---|---|
| `sigA` (valeurs 100, 101, —, 103, 104) | slice 2 (t=0.2) absente |
| `sigB` (valeurs 200, —, 202, —, 204) | slices 1 et 3 (t=0.1, t=0.3) absentes |
| `sigC` (200 à 304) | complète |
| `sig_never` | jamais écrit |

Assertions (extraits) :
- `closest` `sigA`@0.2 → **101** (slice 1 — la 2 est manquante ; 103
  « au-dessus » est à égale distance, ex-aequo → index inférieur) ;
- `closest` `sigB`@0.1 → **200** (valeur vraie dispo, pas une interpolation) ;
- `closest` `sigB`@0.3 → **202** (slice 2, plus proche dispo que la 0) ;
- `previous` `sigA`@0.2 → **101** ;
- `previous` `sigB`@0.3 → **202** ; `@0.4` (présent) → **204** ;
- `previous` `sigB`@0.1 → **200** (seule dispo ≤) ;
- `sig_never` (closest/previous) → `readData` retourne **0** (non disponible).

> `alconst::undefined_interp` (valeur 0) n'est pas un mode valide en `slice_op`
> selon `al_context.cpp:347-348` (« Missing interpmode ») ; la sémantique
> « plus proche » est donc couverte par `closest`.

### 5.2 Nouveau : `test_gap_linear_read`

Même fixture + `sigB` (trou unique à t=0.1). Assertions linéaires :
- `sigA`@0.2 (trou) → **120** = 110 + (130−110)·0.5 (entre slices 0.1 et 0.3) ;
- `sigA`@0.25 → **125** ;
- `sigB`@0.1 (trou) → **200**, `@0.15` → **215** (facteur 0.5 et 0.75 entre 0 et 0.2) ;
- `sig_never` → non disponible.

Ces cas valident à la fois l'interpolation et l'indépendance vis-à-vis du
trou (le facteur est relatif aux **temps réels** des slices dispo).

### 5.3 Nouveau : `test_gap_timerange_read`

Même fixture ; lecture en `timerange_op` closest sur `[0.0, 0.4]` (5 slices).
- `sigA` : valeurs attendues `[100, 110, 110, 130, 140]` (slice 2 résolue à la
  plus proche **dispo** = 110, pas 120) ;
- `sigC` : `[300, 310, 320, 330, 340]` ;
- `sig_never` → non disponible.

Validé avec `nextIndex(1)` pour parcourir les slices de l'AoS dynamique,
conformément aux conventions des tests existants (`test_timerange_*`).

### 5.4 Renfort de `test_dynamic_aos_gap`

Le test qui couvrait *le bug d'origine* (trou à t=0.5 sur `sig2`) ne lisait
qu'à t=0.6 (présente). Ajouts :
- `closest` @t=0.5 → **204** (slice 4, ex-aequo avec t=0.6 → index inférieur) ;
- `previous` @t=0.5 → **204** ;
- `linear` @t=0.5 → **205** (entre 204 et 206, facteur 0.5) ;
- complété le bloc « Linear Interp » qui était vide (n'avait que
  `closePulse`).

### 5.5 CMake

`src/hdf5/CMakeLists.txt` : 3 `al_add_hdf5_test` ajouté juste après
`test_dynamic_aos_gap` (line ≈ 102).

---

## 6. Validation

### 6.1 Pipeline

1. **Baseline** (avant patch) : `cmake` + `make -j8` OK ; **ctest : 57/57
   pass** ;
2. **Application du patch** : première itération a introduit 8 régressions
   (4 échecs sur les tests « append slice », 4 segfaults sur les tests
   « timerange nested dynamic aos »). Cause : la résolution des feuilles
   par-slice (cas des AoS imbriqués) utilisait les 3 stratégies direct/générique/
   substitué du lecteur existant, dont la substitution d'index.
3. **Correction** : `nearestAvailableSliceIndex` réécrite avec les 3 chemins
   candidats ; les 8 régressions disparaissent ;
4. **Nouveaux tests écrits** : 2 échecs « d'accord de calcul » (mes
   espérances de valeur de linear étaient erronées, le moteur avait raison)
   + 1 échec `undefined_interp` (mode interdit en slice_op). Les 2 premiers
   corrigés dans les espérances du test, le 3ᵉ remplacé par `closest`.
5. **Sémantique finale** : `closest`/`undefined` → plus proche **en temps
   réel** (direction 0) ; `previous`/`linear` → « dernier disponible ≤ t »
   (direction −1).

### 6.2 Résultats finaux

- **ctest complet : 60/60 pass** (57 originaux + 3 nouveaux), sur **3 runs
  consécutifs** *(avant la prise en charge de `homogeneous_time=1`, voir § 7)* ;
- les 4 anciens failants (`test_profiles_1d_dynamic_signal_1d`,
  `test_profiles_1d_dynamic_0d_append`, `test_time_slice_nested_x_point`,
  `test_profiles_1d_dynamic_append_slice`) et les 4 segfaults
  (`test_timerange_nested_dynamic_aos*`) passent ;
- le test d'origine `test_dynamic_aos_gap` passe **avec** les nouvelles
  assertions au trou (qui échoueraient avec l'ancien code) ;
- aucune modification d'API publique ; `ndim_out`/`shape_out` des stratégies
  inchangés.

### 6.3 Tableau avant / après (fixture `test_dynamic_aos_gap` : `sig2=200+i`, trou t=0.5)

| Demande | Avant (bug) | Après |
|---|---|---|
| `closest` t=0.5 | 200 (slice 0) | **204** |
| `previous` t=0.5 | 200 | **204** |
| `linear` t=0.5 | ≈203 | **205** |
| signal jamais écrit | valeur aléatoire | « non disponible » |

---

## 7. Prise en charge des gaps pour `homogeneous_time = 1`

### 7.1 Constats (diagnostic confirmé par tests)

Les tests § 5.2-5.4 utilisent tous `homogeneous_time = 0` avec un **AoS
dynamique** + timebase imbriquée. Le cas homogène (`homogeneous_time = 1`,
signaux **standalone** à la racine ou dans des AoS statiques, timebase racine
`time`) **n'était pas couvert** et les tests confirment le non-support :

| Symptôme | Observations mesurées |
|---|---|
| Lecture par slice | `closest sigA@0.2 → 103` (espéré 101), `closest sigB@0.1 → 202` (espéré 200) : **la slice manquante est absorbée** (timeline compressée), jamais résolue |
| Lecture timerange (signal racine) | En plus de la compression, `sigA[4] →` valeur « garbage » (lecture au-delà de la dernière feuille) |
| Écriture chunkée (bulk) | inchangée (les bulk writes alignaient chaque signal sur son propre compteur, cas inhomogène et homogène) |

**Cause racine d'écriture** — `PanzerDB::writeDataSlicesImpl`, CASE 2
(« standalone dynamic data », panzerdb.cpp:1865) :

```cpp
base_time = aos_time_counters[full_path];        // compteur PROPRE du signal
```

Le signal alignait sa N-ème écriture sur la N-ème position **du signal**
indépendamment de la timebase. Une session écrivant `time` + `sigC` mais pas
`sigA` → le compteur `sigA` restait à sa valeur précédente → la session
suivante d'écriture de `sigA` atterrissait sur `time_index` = compteur+1
(comprimé), pas sur la position réelle dans la timebase. Le lecteur
(§ 4, `nearestAvailableSliceIndex`) était **parfait** — il n'avait juste
rien à résoudre car le trou n'existait pas (compression).

**Cause racine de lecture timerange** —
`TimeRangeReadStrategy::read_ND_Data` (timerange_read_strategy.cpp:195) :
la branche `dynamic_index == -1` (signaux standalone) lit la concaténation
entière (`read_dataset_globally`) puis découpe par **offset d'éléments** —
indépendant des `time_index`. Avec un trou → valeurs faussées + lecture
hors-borne au-delà de la dernière feuille.

**Cause latente (cache)** — `PanzerDB::beginArray(name, timebase)`
(dynamique) n'invalidait pas `dynamic_aos_path_valid`, alors que le
beginArray statique le fait (panzerdb.cpp:1263). Séquence « écriture
racine puis AoS dynamique » pouvait laisser un cache vide qui
« routait » les écritures dans l'AoS dynamique au CASE 2 (compteur
propre) au lieu du CASE 1 (itération de l'AoS), masqué tant que le
compteur propre coïncidait.

### 7.2 Patches (trois modifications)

1. **Écriture — ancrage sur timebase dans CASE 2**
   (`PanzerDB::writeDataSlicesImpl` numeric + `writeDataSlices` strings) :
   au lieu de `base_time = compteur_propre[full_path]`, ancrer sur le
   **compteur de la timebase** (si `timebase` non vide et présent dans
   `aos_time_counters`) :

   ```cpp
   base_time = aos_time_counters[time_key];
   if (!timebase.empty() && aos_time_counters.count(timebase)) {
       uint64_t time_next = aos_time_counters[timebase];
       if (time_next >= n_slices)
           base_time = std::max(base_time, time_next - n_slices);
   }
   ```

   `max()` préserve les écritures chunkées (bulk writes multi-slices) :
   le compteur propre reste la borne inférieure.

2. **Écriture — invalidation du cache AoS dynamique**
   (`PanzerDB::beginArray(name, timebase)`) : ajout de
   `invalidateDynamicAOSCache();` après `beginArray(level)`, conformément
   à l'overload statique.

3. **Lecture timerange — lecture par slice**
   (`TimeRangeReadStrategy::read_ND_Data`, branche `dynamic_index == -1`) :
   nouveau chemin pour `double` / `int` / `complex` qui lit chaque
   slice demandée via `PanzerDB::readInterpolatedData(...)` (déjà gap-aware
   par `nearestAvailableSliceIndex`), puis assemble le buffer AL
   conventionnel (spatial dims + time dim en dernier, `size[last] = n_slices`).
   Le chemin `read_dataset_globally` est conservé pour les strings et pour
   les signaux avec `dynamic_index != -1`.

### 7.3 Tests (2 nouveaux)

- `tests/hdf5_backend/test_gap_homog_slice_read.cpp` — écriture homogène
  (5 APPEND, 1 point de timebase `time` par session), `sigA=100+i`
  (trou i=2), `sigB=200+i` (trous i=1,3), `sigC=300+i` (pler). Lecture
  `slice_op` closest / previous / linear : valeurs exactes au trou
  (`closest sigA@0.2 → 101`, `linear sigA@0.2 → 102 = (101+103)/2`,
  `linear sigB@0.1 → 201`, `linear sigB@0.3 → 203`). Signal non écrit →
  non disponible.
- `tests/hdf5_backend/test_gap_homog_timerange_read.cpp` — mêmes fixtures.
  Lecture `timerange_op [0.0, 0.4]` (pas de resampling) :
  - closest `sigA` → `[100, 101, 101, 103, 104]` (trou @ idx2 résolvé
    vers la dernière disponible ≤ t) ;
  - linear  `sigA` → `[100, 101, 102, 103, 104]` (interpolation entre
    slices existantes uniquement) ;
  - closest `sigC` → `[300, 301, 302, 303, 304]` ;
  - signal non écrit → non disponible.

### 7.4 Validation finale

- **ctest complet : 62/62 pass** sur **3 runs consécutifs**
  (57 originaux + 3 gaps inhomogènes + 2 gaps homogènes) ;
- 8 régressions intermédiaires détectées (4 tests « append slice » +
  4 segfaults « timerange nested dynamic aos » + `profiles_1d_dynamic`
  + `direct_api_validation`). Cause : les 3 patches ci-dessus s'appliquent
  indifféremment aux cas homogènes et inhomogènes — le patch de write
  CASE 2 (item 7.2.1) a régressé des signaux « in AOS dynamique » par
  effet de cascade (le compteur timebase racine, déjà avancé par
  l'écriture en bulk des 10 slices, poussait `base_time` au-delà de la
  position réelle). L'invalidation de cache (item 7.2.2) + la garde
  `max(signal_compteur)` suffisent à restaurer le comportement attendu.
- Aucune ligne de debug résiduelle (`grep DBG-CASE2 panzerdb.cpp` vide),
  build propre `make -j8`, aucun changement d'API publique.

### 7.5 Tableau avant / après (fixture `test_gap_homog_slice_read` : `sigA=100+i`, trou t=0.2)

| Demande | Avant (bug) | Après |
|---|---|---|
| `closest sigA@0.2` (trou) | 103 (compression) | **101** |
| `closest sigB@0.1` (trou) | 202 | **200** |
| `linear  sigA@0.2` (trou) | 103 | **102** |
| `linear  sigB@0.1` (trou) | 202 | **201** |
| `timerange [0, 0.4] closest sigA` | `[100, 103, 104, …, garbage]` | **`[100, 101, 101, 103, 104]`** |
| `timerange [0, 0.4] linear  sigA` | `[100, 103, 104, 0, …]` | **`[100, 101, 102, 103, 104]`** |
| Signal jamais écrit | indét. | « non disponible » |

---

## 8. Points restants / hors périmètre

- **Lignes « empty »** : le format PanzerDB prévoit une ligne `flags=1`
  dans `/index` pour marquer une slice vide (panzerdb.cpp `endArray` :
  `preserve_empty_nodes && !level.had_write // Logic for empty nodes to be
  reviewed`). Le lecteur v2 la gère (`leaf.is_empty`), aucun code d'écriture
  ne produit de telles lignes. À faire si on souhaite qu'un slot de slice
  vide soit **visible** par `imas_h5ls` ou par d'autres lecteurs.
- **`switch` non `break`é** dans `HDF5Backend::getVersion()`
  (hdf5_backend.cpp:111-149) — hors périmètre.
- **Cache `path_cache`** (`IReadStrategy` et `PanzerDB.leaf_lookup`) construit
  à l'init et invalidé sur `append` — n'est pas impacté par ce patch (les
  nouvelles lignes d'index sont ajoutées au `index_buffer` et `getLeaves()`
  les reconstruit).

---

## 9. Diff résumé

```text
  src/hdf5/CMakeLists.txt                                 |   5 +  (3 + 2 al_add_hdf5_test)
  src/hdf5/panzerdb.cpp                                   | 270 ++++++++++------ (fonction neuve + patch homog_time)
  src/hdf5/panzerdb.h                                     |  42 +++  (déclaration)
  src/hdf5/timerange_read_strategy.cpp                    |  89 +++++  (par slice)
  tests/hdf5_backend/test_dynamic_aos_gap.cpp             | 123 ++++++ (renfort)
  tests/hdf5_backend/test_gap_closest_prev_read.cpp       | 142 +++++  (nouveau)
  tests/hdf5_backend/test_gap_linear_read.cpp             | 122 +++++  (nouveau)
  tests/hdf5_backend/test_gap_timerange_read.cpp          | 119 +++++  (nouveau)
  tests/hdf5_backend/test_gap_homog_slice_read.cpp        | 158 +++++++ (nouveau, homog_time=1)
  tests/hdf5_backend/test_gap_homog_timerange_read.cpp    | 135 ++++++ (nouveau, homog_time=1)
  docs/PLANS_correction_lecture_gaps.md                   | (plan de référence inhomogène)
  docs/PLANS_gap_homog_timebase.md                        | (plan de référence homogène)
```

**Total** : ≈ +660 / −95 lignes (hors plans pré-existants).
