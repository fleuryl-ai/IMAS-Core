# Gestion des gaps temporels — paragraphe pour la slide ITER

**Proposition de titre de slide :** *ImasCore — résolution déterministe des gaps temporels*

## Paragraphe (principal)

Quand le temps demandé `t` ne correspond à aucune slice effectivement acquise pour un
signal (on parle alors de **gap**), **ImasCore ne retourne jamais une valeur arbitraire
ni celle d’une slice erronée** : la valeur affichée est toujours dérivée de données
réelles, selon la politique choisie par l’application —

- **closest** : la slice existante la plus proche en temps;
- **previous** : la dernière slice acquise avant `t` (on « maintient » la dernière valeur connue);
- **linear** : l’interpolation linéaire entre la dernière slice `≤ t` et la première `≥ t`,
  calculée sur les temps réels de ces deux slices.

Ce choix est **exposé à l’API** (paramètre `interpmode`, sélectionné opération par
opération) : pour une lecture ponctuelle, le mode est choisi explicitement
(`closest` / `previous` / `linear`); pour un ré‑échantillonnage sur une grille temps,
le mode `linear` est appliqué par défaut. Le comportement est donc **déterministe,
traçable et sans fabrication de données**.

## Version courte (corps de slide, en puces)

- Point demandé hors des slices acquises → **gap** : aucune valeur inventée, aucune
  slice arbitraire.
- Trois politiques sélectionnables à l’appel (`interpmode`) :
  - `closest` — slice existante la plus proche en temps;
  - `previous` — dernière slice `≤ t` (holding de la dernière valeur);
  - `linear` — interpolation entre les deux slices existantes qui encadrent `t`.
- Garanties : résultat toujours fondé sur des données réelles, comportement
  déterministe et identique entre versions (reproduit par jeux de tests dédiés).

## Note de bas de slide (source interne)

Comportement implémenté dans le moteur de lecture HDF5 d’ImasCore
(`PanzerDB::readInterpolatedData`), sélectionné via le paramètre `interpmode` de l’API AL
(`closest_interp` / `previous_interp` / `linear_interp`), couverture dédiée :
`test_al_gap_closest_prev` et `test_al_gap_linear`.
