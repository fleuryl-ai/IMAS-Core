# Plan d'implémentation : Outils d'exploration `imas_h5ls` et `imas_h5dump`

## Objectif

Créer deux nouveaux outils en ligne de commande, `imas_h5ls` et `imas_h5dump`, qui imitent le comportement et le format de sortie des utilitaires HDF5 standards (`h5ls`, `h5dump`).

Ces outils ne liront pas la structure physique du fichier HDF5, mais s'appuieront sur l'API `direct_access` pour lire l'index de `PanzerDB` et reconstruire la **structure logique** hiérarchique des données IMAS.

---

## Contexte : Reconstruire une Hiérarchie à partir d'un Index Plat

Le défi principal de cette tâche est que `PanzerDB` stocke toutes les données dans un format plat, optimisé pour l'écriture et la lecture de tranches. Toutes les informations sur la structure sont contenues dans un dataset d'index (`/index`).

Notre mission est de développer une logique qui peut lire cette liste plate de chemins (ex: `profiles_1d/0/ion/1/z_ion`) et en déduire une arborescence de "groupes" et de "datasets" logiques qui ait du sens pour l'utilisateur, comme le ferait `h5ls`.

La fonction `PanzerDB::getLeaves()` sera la source de vérité unique pour cette reconstruction.

---

## Plan d'Implémentation : `imas_h5ls`

### Phase 1: Création du Squelette de l'Application

L'objectif est de mettre en place l'exécutable, la gestion des arguments de la ligne de commande et la structure de base de l'outil.

1.  **Création du Fichier Source**
    *   Créer un nouveau fichier : `src/tools/imas_h5ls.cpp`.
    *   Ce fichier contiendra la fonction `main` de notre outil.

2.  **Mise à Jour du Système de Build (CMake)**
    *   Modifier `src/tools/CMakeLists.txt` pour ajouter la cible de notre nouvel exécutable et le lier à notre bibliothèque `al`.
        ```cmake
        add_executable(imas_h5ls imas_h5ls.cpp)
        target_link_libraries(imas_h5ls PRIVATE al)
        ```

3.  **Implémentation de la Logique `main`**
    *   Dans `imas_h5ls.cpp`, la fonction `main` devra :
        *   Analyser les arguments de la ligne de commande pour extraire le nom du fichier HDF5 et les options (ex: `-r`).
        *   Appeler la nouvelle fonction de l'API `direct_access` que nous créerons dans la phase 2.
        *   Itérer sur la structure de données retournée pour formater et afficher la sortie sur la console.

### Phase 2: Augmenter l'API `direct_access` pour l'Exploration

C'est le cœur de la logique. Nous ajoutons une nouvelle fonctionnalité à notre API qui sera responsable de la reconstruction de la hiérarchie.

1.  **Définir la Structure de Données des Nœuds**
    *   Dans `include/direct_access_api.h`, définir une nouvelle structure qui peut représenter soit un "groupe", soit un "dataset".
        ```cpp
        // Represents a node in the logical data hierarchy.
        enum class NodeType { GROUP, DATASET };
        struct NodeInfo {
            std::string path;
            NodeType type;
            std::vector<size_t> dims;
            bool is_dynamic = false;
        };
        ```

2.  **Créer la Nouvelle Fonction d'API**
    *   Déclarer une nouvelle fonction publique dans `include/direct_access_api.h`:
        ```cpp
        std::vector<NodeInfo> list_nodes(const std::string& ids_name, bool recursive);
        ```

3.  **Implémenter `list_nodes`**
    *   Dans `src/hdf5/direct_access_api.cpp`, cette fonction effectuera les opérations suivantes :
        a.  Instancier `PanzerDB` en mode lecture.
        b.  Appeler `db.getLeaves()` pour obtenir la liste de toutes les entrées de l'index.
        c.  Utiliser un `std::set<std::string>` pour stocker les chemins uniques des "groupes" logiques.
        d.  Utiliser un `std::map<std::string, const PanzerDB::Leaf*>` pour stocker les "datasets" logiques, en associant un chemin de schéma à une feuille représentative.
        e.  **Itérer sur chaque `leaf` de l'index** :
            i.  Prendre le chemin complet (`leaf.path`).
            ii. **Extraire les groupes parents** : Découper le chemin par `/` et ajouter tous les préfixes (`profiles_1d`, `profiles_1d/0`, `profiles_1d/0/ion`) au `set` des groupes.
            iii. **Extraire le dataset** : Obtenir le "chemin de schéma" en enlevant les indices numériques (ex: `profiles_1d/ion/z_ion` avec `PanzerDB::stripIndices`). Si ce chemin n'est pas déjà dans la `map`, l'ajouter avec un pointeur vers la `leaf` actuelle.
        f.  **Construire le Vecteur de Résultat** :
            i.  Itérer sur le `set` des groupes pour créer des `NodeInfo` de type `GROUP`.
            ii. Itérer sur la `map` des datasets. Pour chaque dataset, utiliser la `leaf` représentative pour extraire ses dimensions (`leaf.shape`) et déterminer si un de ses parents est un AoS dynamique (avec notre fonction `is_dynamic_aos`). Créer un `NodeInfo` de type `DATASET`.
        g.  Trier le vecteur de `NodeInfo` par ordre alphabétique de chemin et le retourner.

### Phase 3: Finaliser l'Affichage de `imas_h5ls`

Nous connectons l'application en ligne de commande à notre nouvelle API et nous nous occupons du formatage.

1.  **Appeler l'API**
    *   Dans `imas_h5ls.cpp`, la fonction `main` appellera `imas::direct_access::list_nodes(filename, recursive_flag)`.

2.  **Formater la Sortie**
    *   Itérer sur le vecteur de `NodeInfo` retourné.
    *   Pour chaque `NodeInfo`, formater la sortie pour qu'elle corresponde au style de `h5ls`.
    *   Afficher "Group" ou "Dataset" en fonction du `node.type`.
    *   Pour les `DATASET`, formater les dimensions, en ajoutant `/Inf` si `node.is_dynamic` est vrai.

---

## Phase Future : Implémentation de `imas_h5dump`

Une fois `imas_h5ls` fonctionnel, l'implémentation de `imas_h5dump` sera beaucoup plus simple.

1.  **Créer `imas_h5dump.cpp`** et l'ajouter au build CMake.
2.  Dans son `main`, réutiliser la fonction `imas::direct_access::list_nodes` pour obtenir la liste de tous les datasets.
3.  **Itérer sur tous les `NodeInfo` de type `DATASET`**.
4.  Pour chaque dataset, appeler `imas::direct_access::read_tensor` pour lire son contenu complet en mémoire.
5.  Implémenter une nouvelle fonction d'assistance, "TensorPrinter", qui prend un `TensorView` en entrée et formate son contenu (données, type, dimensions) sur la console dans un style similaire à `h5dump`.
