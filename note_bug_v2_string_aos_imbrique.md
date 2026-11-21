# Bug HDF5 v2 — lecture STRING sous AoS imbriqué (CORRIGÉ 2026-11-21)

**Statut : CORRIGÉ (2026-11-21).** Distinct du bug de dimension numérique
corrigé aussi (cf. commit `4c225e5`, `iread_strategy.h` branche N-D `want_time_dim`) :
ceci **n'est pas un bug de `dim`**.

## Symptôme

`pytest imas/test/test_latest_dd_autofill.py --hdf5` (IMAS-Python) échouait sur
`STR_0D`/`STR_1D` :
- `Unequal nodes …: str('RANDOM…'), None)` — un nœud lu `None` alors que la réf
  a une valeur, et un autre nœud recevant la valeur d'un voisin,
- 1 cas `expected CHAR_DATA in 1D but got CHAR_DATA in 2D` — un nœud lu comme une
  liste d'éléments appartenant à des sous-nœuds.

Les nœuds touchés (selon la graine aléatoire du test) :
- `ids_properties/{plugins,provenance,...}` — AoS imbriqués :
  `plugins/node[i]/put_operation[j]/[library[k]/]{name,version,…,repository,parameters}` ;
- données : `profiles_1d[i]/ion[j]/density_fit/source` et `…/ion[j]/state[k]/density_fit/source`.

Dépendance aux données : la topologie aléatoire (quelles instances sont remplies,
lequel porte des `library`/`state` vides vs pleins) décide de quel nœud le
bug se manifeste. `random.seed(1)` sur `core_profiles` le reproduit à 2 nœuds.

## Cause racine (confirmée par trace `DEBUG_HDF5_READER_V2`)

**Ce n'est ni un bug d'écriture ni un bug de `dim`.** L'écriture PanzerDB est
correcte (les 228/229 feuilles non vides de `ids_properties/plugins` sont sur le
bon chemin d'instance, avec la bonne valeur ; le 296ᵉ est un string **vide**,
jamais écrit — comportement attendu pour une valeur vide).

Le défaut est le **fallback `name_index`** de
`IReadStrategy::read_dataset_globally` (`src/hdf5/iread_strategy.h`). Quand la
lecture n'a pas sa feuille exacte (valeur vide → pas de ligne d'index), le
fallback « dernier segment == nom » + `context_prefix` acceptait **toute** feuille
dont le chemin commence par ce préfixe, donc aussi les membres de sous-arbres
plus profonds :

- lire `…/node/2/readback/0/repository` servait
  `…/node/2/readback/0/library/0/repository` ;
- lire `…/ion/1/density_fit&source` servait les 3 listes `…/ion/1/state/0,1,2/
  density_fit&source` **fusionnées** (d'où la liste de 8 éléments au lieu de 3).

Un nœud **vide** renvoyait ainsi la valeur d'un voisin/du sous-arbre au lieu de
rester vide → corruption du round-trip sur AoS imbriqués.

Trace (graine 1, `core_profiles`) : 7232 lectures, dont **6** par le fallback,
toutes 6 cas de ce sur-accouchement ; aucune lecture « du parent » légitime
(ex. `profiles_1d/signal` depuis la racine) n'emprunte ce chemin avec un
`context_prefix` non vide.

## Correctif

Dans la branche `context_prefix non vide` du fallback, on n'accepte un
candidat que si, **au-delà du préfixe de contexte**, ses parties nommées
(les indices d'instance AoS retirés) sont **exactement** le nom du dataset
demandé. Les membres d'un sous-arbre (`library/0/repository`,
`state/0/density_fit&source`, …) sont donc écartés ; un nœud absent/ vide
renvoie bien « vide » et le round-trip repasse.

Le cas racine (`context_prefix` vide, règle `parent_path` vide) est inchangé.

## Vérification (IMAS-Python, après le correctif)

- `pytest imas/test/test_latest_dd_autofill.py --hdf5` → **328 passed, 0 failed,
  164 skipped, RC=0** (avant : 5 IDS en échec, dont `core_profiles`/`amns_data`).
- `core_profiles` graine 1→11 toutes **PASS** (avant : échec non-déterministe).
- Sweep régression `test_dbentry / test_latest_dd_manual / test_latest_dd_resample /
  test_lazy_loading / test_ids_coordinates / test_ids_validate / test_ids_struct_array /
  test_ids_toplevel` → **169 passed, 0 failed** sur `--hdf5` **et** `--memory`.

Repro déterministe : `core_profiles`, `fill_with_random_data` seed 1, put puis
get, comparer `ids_properties/plugins/…` et `profiles_1d/ion/density_fit/source`.
