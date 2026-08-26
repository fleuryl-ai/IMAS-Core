# Plan — Correction de la lecture des signaux avec trous (backend HDF5 v2)

## 1. Bug

`PanzerDB::readInterpolatedData` (src/hdf5/panzerdb.cpp:3079) contient un lambda
`read_slice_with_fallback` (panzerdb.cpp:3099-3156) qui, si la slice demandée est
absente, relit silencieusement la **slice 0** du même chemin et renvoie sa valeur.

Conséquences :
- `closest` / `previous` → une valeur arbitraire (valeur de la slice 0) est
  renvoyée au lieu de la slice disponible la plus proche ;
- `linear` → interpolation entre une valeur fausse (slice 0) et la valeur réelle
  du voisin → résultats erronés (ex. trou à t=0.5 : 203 renvoyé au lieu de 205) ;
- signal jamais écrit dans le temps → lecture indue de la slice 0.

Le test existant `test_dynamic_aos_gap.cpp` ne détecte pas le bug car il ne lit
que `t=0.6` (slice 6, qui existe). La lecture globale
(`IReadStrategy::read_dataset_globally`, iread_strategy.h:792-1122) se comporte
correctement (retour 0 → AL "no data" → défaut IMAS) : elle n'est pas modifiée.

## 2. Sémantique ciblée

| Mode        | Comportement en cas de trou                                            |
|-------------|------------------------------------------------------------------------|
| `closest`   | slice disponible la plus proche en `|t|` ; **ex-aequo → index inférieur** |
| `previous`  | dernière slice disponible ≤ t, sinon la 1ʳᵉ disponible                |
| `linear`    | interpolation entre dernière disponible ≤ t et 1ʳᵉ disponible ≥ t, facteur calculé sur les temps réels des slices résolues |
| signal jamais écrit | `readInterpolatedData` → −1 → `read_ND_Data` → 0 → `Lowlevel::setDefaultValue` (−9.0E40 scalaire, NULL tableau) |

`DataInterpolation::interpolate` (src/data_interpolation.cpp:194-327) est déjà
correcte si les temps des slices résolues (`slices_times[SLICE_INF]/[SLICE_SUP]`)
sont les temps réels des slices effectivement lues : aucune modification nécessaire.

## 3. Patch

### 3.1 `src/hdf5/panzerdb.h`
Déclenrer (public, const) :
```cpp
int64_t nearestAvailableSliceIndex(const char* full_path, int64_t requested_idx,
                                   int64_t prefer_direction,
                                   const std::vector<double>& time_basis) const;
```
`prefer_direction` : −1 = préférer le côté inférieur, +1 = le côté supérieur,
0 = le plus proche (ex-aequo → index inférieur).

### 3.2 `src/hdf5/panzerdb.cpp`
1. **Nouvelle fonction `nearestAvailableSliceIndex`** (après `getTimeIndex`,
   ~line 2390) :
   - collecte les feuilles du chemin via `leaf_lookup` ; si vide, tentative du
     "generic path" (chemin AoS + suffixe sans index, logique
     panzerdb.cpp:2505-2535) pour couvrir les signaux dans AoS dynamique ;
   - pour chaque feuille : plage d'indices couverts
     `[time_index, time_index + count/slice_volume − 1]` avec
     `slice_volume = ∏(shape>0)` (formule identique à `update_time_index_cache`,
     panzerdb.cpp:928-942) ;
   - si `requested_idx` ∈ plage → le renvoie ;
   - sinon candidats ≤ et ≥ :
     - `prefer_direction = −1` → le plus grand ≤ si non nul, sinon le plus petit ≥ ;
     - `prefer_direction = +1` → symétrique ;
     - `prefer_direction = 0` → min de `|time_basis[c] − time_basis[requested_idx]|`,
       ex-aequo → index inférieur ;
   - −1 si aucune feuille.

2. **Réécriture de `readInterpolatedData`** (panzerdb.cpp:3079-3201) :
   - suppression du lambda `read_slice_with_fallback` (y compris le chemin mort
     "bulk/split" lines 3134-3151) ;
   - résolution des indices :
     ```
     slice_request  = getSlicesTimesIndices(time, time_basis, times_indices, interp_mode)
     request_sup    = times_indices[SLICE_SUP]
     inf_req        = (interp == LINEAR) ? slice_request : (interp == PREVIOUS) ? slice_request : request_sup
     inf_i = nearestAvailableSliceIndex(path, inf_req,   -1, time_basis)
     sup_i = nearestAvailableSliceIndex(path, request_sup, +1, time_basis)
     if (inf_i == -1) return -1
     if (sup_i  == -1) sup_i = inf_i
     if (sup_i  < inf_i) std::swap(inf_i, sup_i)
     ```
   - lecture unique par dispatch existant
     (`pz_readData_by_index` / `pz_readStringData_by_index` /
     `pz_readComplexData_by_index`) sans fallback ;
   - si `inf_i == sup_i` ou `interp != LINEAR` → renvoyer `data_inf` ;
   - sinon lecture de `sup_i`, `slices_times[SLICE_INF] = time_basis[inf_i]`,
     `slices_times[SLICE_SUP] = time_basis[sup_i]`, puis
     `DataInterpolation::interpolate` ; gestion mémoire `free` conservée.
   - pas de changement de signature publique, `ndim_out`/`shape_out` inchangés
     pour les stratégies (SliceReadStrategy, TimeRangeReadStrategy).

## 4. Tableau avant/après

Fixture : `A/0/B` dynamique, 7 slices `t = 0.0…0.6 step 0.1`, `sig_gap[i] = 200+i`,
trou à la slice 5 (t=0.5) ; `sig_both_edges` = 201 @0.1 et 206 @0.6 ; `sig_never`
jamais écrit.

| Demande                    | Avant (bug)   | Après        |
|----------------------------|---------------|--------------|
| closest  t=0.5 (`sig_gap`) | 200 (slice 0) | **204**      |
| closest  t=0.55 (`sig_gap`)| 206           | 206          |
| previous t=0.5 (`sig_gap`) | 200           | **204**      |
| previous t=0.05 (`sig_gap`)| fallback      | **200**      |
| linear   t=0.5 (`sig_gap`) | 0.5·200+0.5·206 = 203 | **205** (=204+0.5·(206−204)) |
| linear   t=0.55 (`sig_gap`)| 203           | **205.5**    |
| linear   t=0.3 (`sig_both_edges`) | valeur fausse | **203** (=201+0.4·5) |
| `sig_never`, n'importe quel t / mode | slice 0 indue | **0** → −9.0E40 |

## 5. Tests

Style : C++ autonome, `DataEntryContext` + `HDF5Backend` directement
(identique à `test_dynamic_aos_gap.cpp`), URI sous `./test_db_...`.

### Fixture d'écriture (commune au 3 nouveaux tests)
- `ids_properties&homogeneous_time = 0`, AoS statique `A` (size 1), AoS
  dynamique `B` (timebase `time`), 7 slices écrites en SLICE_OP
  `t = 0.0, 0.1, ..., 0.6` ;
- `time` : tous les slices (0.0..0.6) ;
- `sig_gap` : tous sauf 0.5 (valeurs 200.0+i) ;
- `sig_both_edges` : uniquement t=0.1 (201.0) et t=0.6 (206.0) ;
- `sig_never` : jamais écrit ;
- `sig_1d` : shape [4], valeurs `[i*10+t, ...]`, trou à t=0.3 ;
- `sig_2d` : shape [2,3], valeurs déterministes, trous à t=0.3 et t=0.5.

### `test_gap_closest_prev_read.cpp`
- closest `sig_gap` : @0.5 → **204** ; @0.55 → 206 ; @0.4 → 204 ;
- closest `sig_both_edges` : @0.45 → 206 ; @0.25 → **201** ;
- previous `sig_gap` : @0.5 → **204** ; @0.05 → 200 ; @0.55 → **204** ;
- previous `sig_both_edges` : @0.3 → **201** ; @0.6 → 206 ;
- undefined (= closest) `sig_gap` : @0.5 → **204** ;
- `sig_never` : closest, previous, linear → `backend.readData` return **0**.

### `test_gap_linear_read.cpp`
- `sig_gap` : @0.5 → **205** ; @0.55 → **205.5** ; @0.45 → **204.5** ;
  @0.05 → 200.5 ; @0.6 → 206 ;
- `sig_both_edges` : @0.3 → **203** ; @0.5 → **205** ;
- `sig_1d` (shape [4]) : trou à 0.3 → linear @0.4 = 0.5·(valeur à 0.2) +
  0.5·(valeur à 0.5), pour chaque des 4 éléments ; `dim=1`, `size[0]=4` ;
- `sig_2d` (shape [2,3]) : trous à 0.3/0.5 → linear @0.4 = 0.5·(0.2) +
  0.5·(0.5) ; `dim=2`, `size=[2,3]` ;
- `sig_never` → retour **0**.

### `test_gap_timerange_read.cpp`
- timerange `tmin=0.0, tmax=0.6, dtime=0.05, interp=linear` sur AoS dynamique :
  - `sig_gap` → 13 points ; points attendus : idx 0→200, 10→**205**, 11→**205.5**,
    12→206 (les autres interpolés linéairement entre slices réelles
    disponibles 0.0-0.4 et 0.6, facteur sur temps réels) ;
  - `sig_both_edges` → série interpolée entre 0.1 (201) et 0.6 (206) ;
  - `sig_never` → retour **0** ;
- resampling `dtime.size()==1` (dtime=0.1) sur `sig_both_edges`, t∈[0.0, 0.6]
  interp closest → 201, 201, (0.2), 201?, ..., 206 — assertions sur les
  positions exactes attendues.

### Renforcement `test_dynamic_aos_gap.cpp`
- compléter le bloc "2. Linear Interp" (actuellement vide, line 174-182) :
  t=0.6 → 206 (inchangé) ; ajouter t=0.5 → **205** ;
- ajouter un bloc "3. Closest Interp at t=0.5" : `sig2` → **204** ;
- (optionnel) ajouter lecture de signal inexistant → retour 0.

## 6. CMake

`src/hdf5/CMakeLists.txt`, après la ligne 102 (`test_dynamic_aos_gap`) :
```cmake
al_add_hdf5_test(test_gap_closest_prev_read)
al_add_hdf5_test(test_gap_linear_read)
al_add_hdf5_test(test_gap_timerange_read)
```

## 7. Exécution & vérification

1. Écrire ce plan (`docs/PLANS_correction_lecture_gaps.md`).
2. Baseline : `cmake -S . -B build && cmake --build build -j8`, puis
   `ctest --test-dir build -R "gap|slice|timerange|panzer"` **avant** le patch.
3. Appliquer le patch (panzerdb.h/.cpp), rebuild.
4. Écrire les 3 tests, les enregistrer, rebuild, les lancer.
5. `ctest --test-dir build` (tous les tests de `tests/hdf5_backend`) ; analyse
   cas par cas des échecs éventuels. Les tests "pas d'exception"
   (ex. `test_bug_b_field_na_slice`) tolèrent un retour 0 ; les assertions sur
   valeurs après trou sont mises à jour aux valeurs du §4.

## 8. Hors périmètre (consigné, non inclus)

- Écriture de lignes "empty" (`flags=1`) par le writer v2
  (panzerdb.cpp:2177, bloc "à revoir") ;
- `switch` sans `break` de `HDF5Backend::getVersion` (hdf5_backend.cpp:111-149) ;
- aucune modification de `DataInterpolation` ni du backend v1.
