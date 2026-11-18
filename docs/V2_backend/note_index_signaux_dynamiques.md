# Comment l'index PanzerDB distingue les signaux dynamiques

> Note technique — modèle d'index `Leaf` (engine `PanzerDB`), vérifié dans le code.
> Question de départ : comment distinguer dans l'index un signal dynamique *via un AOS
> dynamique* (`flux_loop[i].data`) d'un signal dont la feuille est elle-même
> dynamique (sans AOS dynamique) ?

---

## 1. Ce que `shape` code (et surtout ce qu'il ne code PAS)

Dans la structure `Leaf` (`src/hdf5/panzerdb.h:192`) :

```cpp
std::vector<size_t> shape;      // Dimensions of a single time slice (empty for scalars/meta-nodes).
uint64_t time_index = 0;        // First time step stored in this leaf (0 for static data).
uint64_t offset = 0;            // Element offset of this leaf inside its raw data dataset.
uint64_t count = 0;             // Number of stored elements (slice_volume * n_time_steps).
uint64_t flags = 0;             // low 4 bits: node kind; upper bits: DataType.
```

`shape` = **les dimensions spatiales d'une seule slice**. Le **temps n'y est jamais** représenté.
Il est porté par :

- le **nombre de lignes d'index** pour le même chemin,
- le champ `time_index`,
- `count` + `flags` (rôle du nœud).

`shape` exprime donc le **rang spatial** de la feuille :

| Folie (feuille) | `ndim` | `shape` |
|---|---|---|
| scalaire | 0 | vide (pas de `shape[0]`) |
| tableau 1D de N points | 1 | `[N]` → `shape[0] = N` |
| tableau 2D (a×b) | 2 | `[a, b]` |

> **Conclusion :** `shape[0]` ≠ 0 / == 0 dit « tableau 1D ou scalaire », **pas** « constant ou
> dynamique ». Ce n'est pas le bon axe pour distinguer les deux catégories.

### Ta hypothèse initiale, corrigée

> « shape[0] sera différent de 0 pour le signal constant alors qu'il vaut 0 pour
> `flux_loop[i].data` car celui-ci n'a pas d'extension spatiale ? »

C'est à l'envers (et auto-contradictoire avec « `data` est 1D ») :

- `flux_loop[i].data` = tableau des points du contour → **a** une extension spatiale → `shape[0] = N ≠ 0`.
- `flux_loop[i].position_r` = position du loop → **scalaire** → `ndim = 0`, `shape` vide.

Si les deux étaient « 1D » comme tu l'indiques, alors **les deux** auraient `shape[0] = N ≠ 0`,
donc `shape[0]` ne pourrait PAS les distinguer. Le rang spatial dit `data` (tableau) vs
`position_r` (scalaire) — mais rien sur le temps.

---

## 2. Ce qui distingue VRAIMENT les deux catégories dans l'index

Les 4 bits bas de `flags` encodent le **rôle du nœud** ; les bits supérieurs, le type de
donnée. Le discriminant est donc **`flags` (rôle) + l'ancêtre (`parent_id`)**, pas `shape` :

- `flags & 0xF == 0` → **feuille de données** (double / int / complex / string)
- `flags & 0xF == 2` → **méta-nœud AOS statique** (sa taille est dans `shape[0]`, `count = 0`) — `src/hdf5/panzerdb.cpp:1184`
- `flags & 0xF == 3` → **méta-nœud AOS dynamique** (idem, taille dans `shape[0]`)

Note : pour un **méta-nœud AOS**, le nombre d'éléments est dans `shape[0]` (et `count = 0`,
pas de données brutes). Pour une **feuille de données**, `shape` = dimensions spatiales d'une
slice et `count` = nombre d'éléments stockés.

---

## 3. Les 3 catégories et leur « signature d'index »

Une ligne d'index a le layout (`panzerdb.cpp:1971`, commentaire de rangée ~`panzerdb.h:249`) :

```
[0]=type  [1]=ndim  [2..7]=shape[]  [8]=time_index
[9]=offset  [10]=count  [11]=flags  [12]=parent_id  [13]=index_value
```

### Catégorie A — dynamique parce que le chemin contient un AOS dynamique

Ex. `flux_loop[i].data`, `flux_loop[i].position_r` :

- `flux_loop` = **1 ligne méta-nœud `flags = 3`**, `shape[0] = n_loops`, `count = 0`.
- Chaque signal = **1 ligne par instance `i`**, avec :
  - `parent_id` → ligne `flux_loop` (la méta-nœud dynamique),
  - `index_value = i` (l'indice d'instance dans l'AOS),
  - `time_index` = l'itération `i` (ancre sur le compteur de l'AOS dynamique),
  - `shape` = dimensions spatiales,
  - `count` = taille d'une slice (`N` pour `data`, `1` pour `position_r`).

→ Ici le **temps est porté par l'AOS dynamique parent** (`parent_id` + `index_value`), pas par
la feuille elle-même.

### Catégorie B — la feuille est elle-même indexée en temps (pas d'AOS dynamique)

La feuille (kind **0**) porte **sa propre dimension temps** : elle n'est pas ancrée sur un
AOS dynamique, mais sur un **AOS statique (kind 2)** ou la racine, et l'extension temporelle
se retrouve soit dans `count` (forme **bulk**), soit dans une séquence de `time_index`
(forme **itérative**).

> **Cas canonique (le tien)** — `A` = AOS statique ; `A[i].data` = un **signal de flux**
> (valeur scalaire par pas), **N échantillons** en temps, sans extension spatiale.
>
> ⚠️ **Deux « ndim » distincts — ne pas confondre :**
>
> - **En IMAS (côté `readData`) : `ndim = 1`, `shape = [N]`** — le signal est bien **1D**
>   (longueur N), car le lecteur **ajoute l'axe temps** : `*ndim_out = leaf.shape.size() + 1`,
>   dernier dim = `count / ∏shape` (`panzerdb.cpp:2384-2387`, idem int/complex/string
>   `2801-2805`). Ici : `0 + 1 = 1`, dim `[N]`.
> - **Dans l'index stocké (une seule ligne, path `A/i/data`) :**
>   - colonne `row[1]` = `Leaf.shape.size()` = **0** → `Leaf.shape` **vide** (rang **spatial** ;
>     ce 0 n'est **pas** la dimension IMAS) ; case `row[2]` (slot `shape[0]`) = 0 ;
>   - **`count = N`** (= volume-slice `1` × `N` pas) — ici **tout le `count` est du temps** ;
>   - `time_index = 0`, `offset` = début des `N` doubles consécutifs, `flags` = double (kind 0) ;
>   - `parent_id` → méta-ligne `A` (kind 2, AOS statique), `index_value = i` ;
>   - (cheminement : `writeData("…","data","time",…) dim=1,size=[N]` →
>     `hdf5_writer_v2.cpp:335-345` strip le **dernier dim = temps** → `base_shape={}` →
>     `writeDataSlices(name, {}, data, N, "time")` → **une** `append_index_row`,
>     `count = 1 × N = N`, `panzerdb.cpp:1804-1807`).
>
> **Pont entre les deux :** `IMAS_dim = Leaf.shape.size() + (count > ∏shape ? 1 : 0)`.
> Ici `0 + 1 = 1`. → le **« ndim » d'IMAS** compte le temps ; le champ **`Leaf.shape` de
> l'index** ne compte que le rang **spatial**.

**Règle du lecteur (`panzerdb.cpp:2381-2389`) — comment `dim` IMAS est re-dérivée :**

```
slice_vol = ∏(leaf.shape)                      # volume spatial d'une slice
if  count > slice_vol :                          # il y a un axe temps → +1
    ndim_out = leaf.shape.size() + 1
    derniere_dim = count / slice_vol             # (nb de pas)
else :                                           # statique
    ndim_out = leaf.shape.size()
```

**Nombre de lignes — dépend du write driver, pas de la catégorie :**

- **forme bulk** (assign tableau `A[i].data = [v0..vN]`, une seule `writeData`) →
  **1 ligne, `count = N`** ;
- **forme itérative** (un échantillon temporel par `writeData`, N fois) →
  **N lignes de `count = 1`**, `time_index = 0..N-1` ;
- **chaînes** : toujours « 1 ligne par slice » (boucle), `panzerdb.cpp:1915-1935`.

Pour un signal scalaire dynamique écrit normalement, la forme est **bulk ⇒ 1 ligne, `count = N`**.

### Tableau récapitulatif

| | `data` sous `flux_loop[i]` (A) | `position_r` sous `flux_loop[i]` (A) | `A[i].data` scalaire sous AOS statique (B, ton cas) |
|---|---|---|---|
| `flags & 0xF` | 0 | 0 | 0 |
| index : rang spatial (`Leaf.shape.size()`) / `shape` | 1 / `[N]` | 0 / `[]` | 0 / `[]` |
| **dim IMAS** (temps inclus, côté `readData`) | 1 (spatial) | 0 (scalaire) | **1** = `0 + 1` temps |
| `parent_id` | → AOS dynamique (flags=3) | → AOS dynamique (flags=3) | → AOS statique (flags=2) |
| `count` (bulk) | **N (spatial)** | 1 | **N (temps)** |
| nb de lignes | 1 par instance `i` | 1 par instance `i` | **1** (bulk) |
| le temps est porté par | `parent_id` + `index_value` | `parent_id` + `index_value` | `count` (bulk) / `time_index` (itératif) |

> Lecture : dans la colonne **B (ton cas)**, la feuille est **scalaire** (rang spatial `0`, mais
> **dim IMAS = `0 + 1` = 1** via `panzerdb.cpp:2384`) et `count = N` **tout entier est du temps** ;
> dans la colonne **A `data`**, le `count = N` est **spatial** (taille du tableau de points,
> rang spatial `1`, dim IMAS `1`), et le temps est porté par l'AOS dynamique parent.
> C'est la même « 1 ligne, `count ≠ 1` », mais **B** = N pas de temps, **A** = N points spatiaux.
> ⚠️ Le `0` de B (colonne « rang spatial de l'index ») n'est **pas** la dimension IMAS.

---

## 4. Le piège : `"time"` vs `""` dans `writeData(...)` sous un AOS dynamique

Sous un AOS dynamique, le writer détecte `isInsideDynamicAOS`
(`src/hdf5/hdf5_writer_v2.cpp:311`) et écrit **`n_slices = 1` par itération, quelle que soit
la base de temps passée** :

- `data` avec `"time"` → **CASE 1 dynamique** (`hdf5_writer_v2.cpp:330-364`)
  → `writeDataSlices(shape, 1, tb)` ;
- `position_r` avec `""` → **CASE 2** (`hdf5_writer_v2.cpp:347-373`)
  → `writeDataSlices(shape, 1, aos_timebase)` ;
- (pas d'AOS dynamique + `""`) → **CASE 3 statique**
  → `writeData` = **une seule ligne, `time_index = 0`** (`panzerdb.cpp:1565`).

Le `"time"` / `""` ne change **pas** la ligne d'index quand la feuille est dans un AOS dynamique :
il est utilisé seulement pour l'alignement des compteurs internes `aos_time_counters` (gestion
des gaps). C'est l'**ancêtre (AOS dynamique)** qui rend `data` et `position_r` « dynamiques par
héritage » ; ils apparaissent **identiques** dans l'index, sauf `shape` / `count`.

---

## 5. Réponse à la question initiale

> Comment distingue-t-on les 2 signaux dans l'index ?

- **Constant vs dynamique**, et **dynamique-par-héritage (A) vs dynamique-feuille (B)** :
  par **`flags` (rôle du nœud)** + **`parent_id`**, pas par `shape`.
  - sous un **AOS dynamique (flags = 3)** → catégorie A (temps porté par `parent_id`/`index_value`) ;
  - sous un **AOS statique (flags = 2)** ou racine, **temps porté par la feuille elle-même**
    (`count` en forme **bulk**, ou séquence `time_index` en forme **itératif**) → catégorie B ;
  - **1 seule ligne `time_index = 0`, `count` = taille spatiale seule (aucune extension temps)**
    → feuille statique/constante.
- **`data` vs `position_r`** (tous deux sous le même AOS dynamique) : par **`ndim` / `shape`** et
  `count` (tableau 1D avec extension spatiale vs scalaire), **pas** par un champ « temps ».

## Références (file:line)

- `src/hdf5/panzerdb.h:192` — `struct Leaf` (`shape`, `time_index`, `count`, `flags`).
- `src/hdf5/panzerdb.h:249` — layout des 14 colonnes de la ligne d'index.
- `src/hdf5/panzerdb.cpp:1971` — `append_index_row` (construction de la rangée).
- `src/hdf5/panzerdb.cpp:2023` — `isInsideDynamicAOS` (détecte l'AOS dynamique ancêtre).
- `src/hdf5/panzerdb.cpp:1565` — chemine statique `writeDataImpl` (`time_index = 0`, 1 ligne).
- `src/hdf5/panzerdb.cpp:1695-1807` — `writeDataSlicesImpl` (CASE 1 ancre sur l'AOS dyn, CASE 2 feuille autonome) ; écriture commune = **une** `append_index_row` (`count = slice_size × n_slices`).
- `src/hdf5/panzerdb.cpp:1915-1935` — `writeDataSlices` (string) : **1 ligne par slice** (boucle sur les échantillons).
- `src/hdf5/panzerdb.cpp:1184` — rangée méta-nœud AOS (`shape[0]` = taille, `flags` 2/3, `count = 0`).
- `src/hdf5/hdf5_writer_v2.cpp:300-373` — routage AL (`timebasename` vs `is_in_dynamic_aos`) vers `writeData`/`writeDataSlices`.

## Optionnel : voir les lignes réelles

Un petit dump est possible avec `PanzerDB::dumpLeafIndex()` (`src/hdf5/panzerdb.cpp:2023`)
sur une structure `flux_loop` minimale (2 loops, `data` 1D + `position_r` scalaire) pour voir
côte à côte les feuilles et le méta-nœud AOS.
