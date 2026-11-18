# Correction de l'index d'une slice après un gap — note technique

Contexte : lecture/écriture HDF5 d'ImasCore (moteur PanzerDB, chemin
`writeDataSlices` → `writeDataSlicesImpl` et `append_index_row`).

## Idée centrale

L'index d'une slice **n'est pas la valeur temps physique** (les secondes du
dataset `time`). C'est un **compteur de pas logique** (`uint64_t`), conservé dans
le dictionnaire `aos_time_counters`, qui porte **un compteur par chemin** :
un pour chaque signal (`.../sigA`), un pour la **base de temps maître**
(`.../B/time`), un pour l'Ao dynamique contenant les slices.

```cpp
std::unordered_map<std::string, uint64_t> aos_time_counters;          // panzerdb.h:229
```

## Question 1 — Ce compteur est-il « basé sur la base de temps » ?
**Avance-t-elle malgré un gap ?**

- L'index d'un signal est **ancré à la base de temps maître**, mais la valeur
  utilisée n'est pas le temps : c'est la **position** (compteur de pas) de la
  base de temps, `aos_time_counters[time_path]`, couplée à `current_index` de
  l'Ao (avancé par `incrementArrayIndex()`).
- **Oui, la base de temps maître avance à chaque pas, y compris les pas où un
  signal n'est pas écrit** — c'est ce qui ouvre le « trou » de séquence.
- En revanche, le compteur du **signal** concerné **ne saute pas** les pas qu'il
  a omis : il ne progresse que quand on lui écrit une slice.
- Conséquence : au pas manquant, la base de temps porte un index (ex. 2) que le
  signal n'a pas encore atteint (ex. 1). **Cet écart est le gap.**

## Question 2 — À quel moment l'index de la slice d'après le gap est-il positionné ?

- Il est calculé **à l'écriture de la slice suivante**, dans `writeDataSlices`,
  et **non pas à l'avance** :
  ```cpp
  base_time = std::max( (uint64_t)dynamic_aos_current_iteration,   // position courante de l'Ao
                        signal_next_time );                        // compteur du signal (sa dern. slice +1)
  // FIX: « Alignment with timebase in case of gap » (panzerdb.cpp ~1746 / ~1884)
  if ( time_next >= n_slices )                       // time_next = aos_time_counters[time_path]
      base_time = std::max( base_time, time_next - n_slices );
  ```
  C'est le terme `time_next − n_slices` qui **ré-ancore** l'index du signal sur
  la base de temps quand le signal est en retard (gap en cours).
- Cet index devient la `time_idx` de la ligne `/index` :
  `append_index_row(..., /*time_idx=*/ base_time, ...)` ;
  puis le compteur du signal passe à `base_time + n_slices`.

## Question 3 — Le compteur interne survit-il à une nouvelle session ?

L'écriture de la slice d'après le gap a souvent lieu dans **une nouvelle session**
(nouvel objet `PanzerDB`), alors que `aos_time_counters` est purement **en mémoire**.
S'il n'était pas persisté tel quel, ne serait-il pas perdu entre les deux ?

**Non** — il est **reconstruit à chaque réouverture en `APPEND`/`READ`** par
`restoreTimeContext()` (panzerdb.cpp:649), appelé en fin d'`init()` (panzerdb.cpp:297).
La vraie source de vérité persistée est l'index `/index` (`time_index` + `count`),
**pas** le compteur :

- pour chaque feuille de données : `next = leaf.time_index + leaf.count/slice_volume`
  (= fin de la dernière slice écrite) → ressèchement de `aos_time_counters[feuille]`
  **et** de l'AoD parent (le plus profond gagne) ;
- pour la base de temps maître (feuille de données ordinaire) : idem → c'est ce qui
  fournit le `time_next` du FIX de la correction (panzerdb.cpp:1777-1782).

Le doc du header le dit explicitement (panzerdb.h:946) :

> Re-establishes the write-time context (aos_time_counters) from the on-disk leaves
> after opening an existing file … appending continues where the previous session
> stopped, **including gap alignment on the enclosing time base**.

Conséquence : la correction post-gap fonctionne **à la frontière de session**, car
`time_next` (nb de pas de base déjà écrits) est re-dérivable de l'index disque.

**Mise en garde.**

- `restoreTimeContext()` ne s'exécute que si `mode == APPEND || mode == READ`
  (panzerdb.cpp:650). Une ouverture en `WRITE` tronque et repart de 0 : c'est alors un
  fichier neuf, pas une suite — la continuité (et le gap) serait perdue.
- Le FIX suppose le cas standard — la base de temps du pas en cours a déjà été écrite
  (« Assume timebase has already been written for this slice », panzerdb.cpp:1747) :
  d'où l'ordre d'écriture « temps d'abord, signal ensuite ».

## Ce que cela implique concrètement

- **Aucune ligne « vide » n'est écrite au pas manquant.** Le gap est un trou dans
  la séquence des `time_index`.
- Ce trou est **inféré à la lecture** et résolu selon le mode choisi
  (`closest` / `previous` / `linear`), qui ramènent toujours à des slices
  réelles (ou à l'interpolation entre deux slices réelles) — cf. la note
  « Gestion des gaps temporels » (slide_gestion_gaps.md).

## Traçabilité (jeux de tests)

`tests/hdf5_backend/test_al_gap_*.cpp` — le signal est posé sur les index
`0, 1, 3, 4` (trou en 2) pendant que la base de temps couvre `0..4` ; à la
lecture : `closest` à `t` (index 2) renvoie l'index 1, `previous` renvoie l'index
1, `linear` interpole entre l'index 1 et l'index 3.
