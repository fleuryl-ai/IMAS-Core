# Vérification du design d'index M1 — `STATIC > DYNAMIC > STATIC > LEAF`

> Test de non-régression du design « compact-index M1 » (suppression de
> `/parent_paths`, reconstruction de `parent_path` par strip sur `full_path`)
> sur la structure imbriquée **à deux AoS statiques encadrant un AoS dynamique** :
>
> ## `A_static[i].profiles_1d[t].ion[j].temperature`
>
> - `A_static` — AoS **statique**, 2 instances (`i = 0,1`)
> - `profiles_1d` — AoS **dynamique** (time), 2 pas de temps (`t = 0,1`)
> - `ion` — AoS **statique**, 2 instances (`j = 0,1`)
> - `temperature` — data leaf 1D, 5 valeurs spatiales
>
> Pourquoi ce cas : le golden et les tests existants ne couvrent que
> `STATIC > DYNAMIC > LEAF` (golden `na`/`nb`) ou `DYNAMIC > STATIC > LEAF`
> (`profiles_1d/0/ion`). L'encadrement **des deux côtés par un AoS statique**
> de l'AoS dynamique est un cas **nouveau** — c'est l'objet de cette vérification.

---

## 1. Ordre d'écriture (writer C++ PanzerDB, API haut-niveau)

```cpp
db.beginArray("A_static", 2);                    // i, taille 2
for (i in 0..1) {
    db.beginArray("profiles_1d", "time");        // dynamique, re-begun à chaque i
    for (t in 0..1) {
        db.beginArray("ion", 2);                  // statique, re-begun à chaque (i,t)
        for (j in 0..1) {
            db.writeDataSlices("temperature", {5}, d_ij, 1, "");
            if (j < 1) db.incrementArrayIndex();  // j -> next
        }
        db.endArray();                            // pop ion
        if (t < 1) db.incrementArrayIndex();      // t -> next
    }
    db.endArray();                                // pop profiles_1d
    if (i < 1) db.incrementArrayIndex();          // i -> next
}
db.endArray();                                    // pop A_static
```

Fichier produit : `SDS_static_dyn_static.h5`.

Physique du fichier (`h5ls -r`) :

```
/                        Group
data_raw_c128  Dataset {0/Inf}
data_raw_f64   Dataset {40/Inf}     # 8 temperature × 5 doubles = 40
data_raw_i32   Dataset {0/Inf}
data_raw_str   Dataset {0/Inf}
index          Dataset {15/Inf, 14} # 15 lignes × 14 u64
paths          Dataset {15/Inf}     # texte fixe 256 B/ligne (pas de vlen → SWMR-safe)
```

**Pas de dataset `/parent_paths`** — c'est l'effet M1 : le texte du parent
n'est plus stocké, seul `paths` + `index` existent.

---

## 2. Reconstitution de `parent_path` (sortie `PanzerDB::getLeaves()`)

15 lignes : `1 A_static + 2 profiles_1d + 4 ion + 8 temperature`.

| row | index path (`/paths`) | kind | parent_path reconstruit |
|--:|--|--:|--|
| 0 | `A_static` | 2 stat | `` (NO_PARENT) |
| 1 | `A_static/0/profiles_1d` | 3 dyn | `A_static` |
| 2 | `A_static/0/profiles_1d/0/ion` | 2 stat | `A_static/0/profiles_1d` |
| 3 | `A_static/0/profiles_1d/0/ion/0/temperature` | 0 data | `A_static/0/profiles_1d/0/ion/0` |
| 4 | `A_static/0/profiles_1d/0/ion/1/temperature` | 0 data | `A_static/0/profiles_1d/0/ion/1` |
| 5 | `A_static/0/profiles_1d/1/ion` | 2 stat | `A_static/0/profiles_1d` |
| 6 | `A_static/0/profiles_1d/1/ion/0/temperature` | 0 data | `A_static/0/profiles_1d/1/ion/0` |
| 7 | `A_static/0/profiles_1d/1/ion/1/temperature` | 0 data | `A_static/0/profiles_1d/1/ion/1` |
| 8 | `A_static/1/profiles_1d` | 3 dyn | `A_static` |
| 9 | `A_static/1/profiles_1d/0/ion` | 2 stat | `A_static/1/profiles_1d` |
| 10 | `A_static/1/profiles_1d/0/ion/0/temperature` | 0 data | `A_static/1/profiles_1d/0/ion/0` |
| 11 | `A_static/1/profiles_1d/0/ion/1/temperature` | 0 data | `A_static/1/profiles_1d/0/ion/1` |
| 12 | `A_static/1/profiles_1d/1/ion` | 2 stat | `A_static/1/profiles_1d` |
| 13 | `A_static/1/profiles_1d/1/ion/0/temperature` | 0 data | `A_static/1/profiles_1d/1/ion/0` |
| 14 | `A_static/1/profiles_1d/1/ion/1/temperature` | 0 data | `A_static/1/profiles_1d/1/ion/1` |

### Application des règles de strip (panzerdb.cpp:1244–1257)

| ligne | kind | opération | résultat |
|--:|--:|--|--|
| 0 | (NO_PARENT) | `parent_id == 0xFFFF…F` → `""` | `` |
| 1 | 3 dyn | `A_static/0/profiles_1d` − 2 → `A_static` | `A_static` |
| 2 | 2 stat | `A_static/0/profiles_1d/0/ion` − 2 → `A_static/0/profiles_1d` | `A_static/0/profiles_1d` |
| 3 | 0 data | `…/ion/0/temperature` − 1 → `…/ion/0` | `A_static/0/profiles_1d/0/ion/0` |

> Points de vigilance confirmés corrects :
> - `A_static/0/profiles_1d` → parent = `A_static` (le `0` est l'instance de
>   `A_static`, **pas** de `profiles_1d`).
> - `A_static/0/profiles_1d/1/ion` → parent = `A_static/0/profiles_1d` (le `1`
>   est l'instance `t` de `profiles_1d`, **pas** de `ion`).
> - `…/ion/0/temperature` → parent = `…/ion/0` (le dernier `0` est l'instance
>   `j` de `ion`, **pas** `temperature`).

---

## 3. Colonnes M1 brutes (`/index`, u64)

| row | full_path | kind | parent_id (row[12]) | index_value (row[13]) | flags (row[11]) |
|--:|--|--:|--:|--:|--:|
| 0 | `A_static` | 2 | `0xFFFFFFFFFFFFFFFF` | 0 | 0x02 |
| 1 | `A_static/0/profiles_1d` | 3 | 0 | 0 | 0x03 |
| 2 | `A_static/0/profiles_1d/0/ion` | 2 | 1 | 0 | 0x02 |
| 3 | `A_static/0/profiles_1d/0/ion/0/temperature` | 0 | 2 | 0 | 0x00 |
| 4 | `A_static/0/profiles_1d/0/ion/1/temperature` | 0 | 2 | 1 | 0x00 |
| 5 | `A_static/0/profiles_1d/1/ion` | 2 | 1 | 1 | 0x02 |
| 6 | `A_static/0/profiles_1d/1/ion/0/temperature` | 0 | 5 | 0 | 0x00 |
| 7 | `A_static/0/profiles_1d/1/ion/1/temperature` | 0 | 5 | 1 | 0x00 |
| 8 | `A_static/1/profiles_1d` | 3 | 0 | 1 | 0x03 |
| 9 | `A_static/1/profiles_1d/0/ion` | 2 | 8 | 0 | 0x02 |
| 10 | `A_static/1/profiles_1d/0/ion/0/temperature` | 0 | 9 | 0 | 0x00 |
| 11 | `A_static/1/profiles_1d/0/ion/1/temperature` | 0 | 9 | 1 | 0x00 |
| 12 | `A_static/1/profiles_1d/1/ion` | 2 | 8 | 1 | 0x02 |
| 13 | `A_static/1/profiles_1d/1/ion/0/temperature` | 0 | 12 | 0 | 0x00 |
| 14 | `A_static/1/profiles_1d/1/ion/1/temperature` | 0 | 12 | 1 | 0x00 |

Lignes complètes (14 u64) :

```
row  0: [0, 1, 2,  0,0,0,0,0,  0,  0, 0,  2, 18446744073709551615, 0]   # A_static (size 2), NO_PARENT
row  1: [0, 1, 0,  0,0,0,0,0,  0,  0, 0,  3, 0, 0]                      # profiles_1d @i=0 (t=0), parent=A_static
row  2: [0, 1, 2,  0,0,0,0,0,  0,  0, 0,  2, 1, 0]                      # ion @i0,t0 (size 2), parent=profiles_1d(i0)
row  3: [0, 1, 5,  0,0,0,0,0,  0,  0, 5,  0, 2, 0]                      # temp i0,t0,j0 (off 0, cnt 5), parent=ion(i0,t0)
row  4: [0, 1, 5,  0,0,0,0,0,  0,  5, 5,  0, 2, 1]                      # temp i0,t0,j1 (off 5)
row  5: [0, 1, 2,  0,0,0,0,0,  0,  0, 0,  2, 1, 1]                      # ion @i0,t1, parent=profiles_1d(i0)
row  6: [0, 1, 5,  0,0,0,0,0,  1, 10, 5,  0, 5, 0]                      # temp i0,t1,j0 (t=1, off 10)
row  7: [0, 1, 5,  0,0,0,0,0,  1, 15, 5,  0, 5, 1]                      # temp i0,t1,j1 (t=1, off 15)
row  8: [0, 1, 0,  0,0,0,0,0,  0,  0, 0,  3, 0, 1]                      # profiles_1d @i=1 (t=0), parent=A_static
row  9: [0, 1, 2,  0,0,0,0,0,  0,  0, 0,  2, 8, 0]                      # ion @i1,t0, parent=profiles_1d(i1)
row 10: [0, 1, 5,  0,0,0,0,0,  0, 20, 5,  0, 9, 0]                      # temp i1,t0,j0 (off 20)
row 11: [0, 1, 5,  0,0,0,0,0,  0, 25, 5,  0, 9, 1]                      # temp i1,t0,j1 (off 25)
row 12: [0, 1, 2,  0,0,0,0,0,  0,  0, 0,  2, 8, 1]                      # ion @i1,t1, parent=profiles_1d(i1)
row 13: [0, 1, 5,  0,0,0,0,0,  1, 30, 5,  0, 12, 0]                     # temp i1,t1,j0 (t=1, off 30)
row 14: [0, 1, 5,  0,0,0,0,0,  1, 35, 5,  0, 12, 1]                     # temp i1,t1,j1 (t=1, off 35)
```

(Convention des colonnes : `[0]` type, `[1]` ndim, `[2..7]` shape[0..5],
`[8]` time_index, `[9]` offset, `[10]` count, `[11]` flags, `[12]` parent_id,
`[13]` index_value. `data_raw_f64` = 40 doubles, offsets 0/5/10/…/35.)

---

## 4. Pourquoi le design tient à toutes les profondeurs

La règle de reconstruction est **locale à chaque ligne** :

```
data leaf (kind 0/1) → parent_path = full_path        − 1 segment (le nom)
AoS meta  (kind 2/3) → parent_path = full_path        − 2 segments (instance + nom)
parent_id = NO_PARENT  → parent_path = ""
```

Elle ne dépend **ni** du nombre de niveaux **ni** du type (statique/dynamique)
de chaque ancêtre. L'invariante exploité est garantie par le writer :

- **`writeData` (data leaf)** — `full_path = path_prefix [+ "/" + current_index(parent)] + "/" + name`
  (panzerdb.cpp:1447–1466) → **1 segment** au-delà du parent.
- **`beginArray` (AoS meta)** — `full_path = path_prefix [+ "/" + current_index(parent)] + "/" + name`
  (panzerdb.cpp:1121–1133) → **2 segments** au-delà du parent.

L'instance du parent immédiate (statique ou dynamique) est **toujours** bue
dans le chemin du fils. Le strip local est donc un inverse déterministe, quel
que soit l'encadrement.

### Sémantique des 3 colonnes M1

| colonne | valeur observée | signification |
|--:|--|--|
| `parent_id` (row[12]) | pointeur vers la **ligne** du conteneur (`0` = `A_static`, `1`/`8` = `profiles_1d per i`, `2`/`5`/`9`/`12` = `ion per (i,t)`) | ancrage numérique, remplace le texte dupliqué |
| `index_value` (row[13]) | `array_stack.back().current_index` à l'écriture | = instance du parent immédiat : `i` pour `profiles_1d`, `t` pour `ion`, `j` pour `temperature` |
| `flags` (row[11]) | `(DataType << 4) \| kind` | `kind` détermine le nb de segments à strip-er |

### SWMR-safe

Tout est en **large fixe** : `u64` dans `index`, `C_S1` 256 B dans `paths`.
Zéro vlen → compatible avec le mode SWMR (write) que l'utilisateur prévoit
d'implémenter. La profondeur (3 AoS) et l'encadrement par statiques ne
changent rien.

---

## 5. Conclusion

**Le design M1 fonctionne correctement sur `STATIC > DYNAMIC > STATIC > LEAF`**
(cas de profondeur jusqu'ici non couvert par le golden ni les tests).
Les 15 lignes du `/index` réel reconstituent toutes leurs `parent_path`
exactement, sans colonne de texte de parent, et la structure reste SWMR-safe.

Les cas de profondeur connus à ce jour et tous validés :

| structure | couverture |
|--|--|
| `STATIC > DYNAMIC > LEAF` | golden `na`/`nb` ✓ |
| `DYNAMIC > STATIC > LEAF` | `profiles_1d/0/ion` ✓ |
| `STATIC > DYNAMIC > STATIC > LEAF` | **ce fichier** ✓ (nouveau) |

_Fichiers demo : `/tmp/pzdemo/SDS_static_dyn_static.h5` (h5ls/h5dump
accessibles). Code du writer de test : `.qwen/tmp/gen_static_dyn_static.cpp`._
