# Plan d'implémentation : API d'accès direct aux données

## Objectif

Créer une nouvelle API C++ bien définie pour lire des sous-ensembles de données (tenseurs, slices) directement depuis le backend de stockage (HDF5/PanzerDB), sans passer par l'API de haut niveau `get_slice`. Cette API sera exposée via un fichier d'en-tête clair, et servira de fondation pour de futurs outils en ligne de commande et pour un accès programmatique performant.

---

## Phase 1: Conception et Définition de l'API

L'objectif de cette phase est de définir l'interface publique de la nouvelle API et de préparer sa structure dans le projet.

1.  **Création du Fichier d'En-tête de l'API (`direct_access_api.h`)**
    *   Créer un nouveau fichier `include/direct_access_api.h`.
    *   Ce fichier contiendra toutes les déclarations publiques de l'API.

2.  **Définition du Conteneur de Données (`TensorView`)**
    *   Pour retourner des données de types et de dimensions variables, nous définirons une structure ou une classe `TensorView`. Cela évite une API templatisée complexe et fournit un objet auto-descriptif.
    *   Définition dans `direct_access_api.h`:
        ```cpp
        #include <vector>
        #include <string>
        #include <memory>

        enum class DataType {
            FLOAT,
            DOUBLE,
            INT32,
            INT64,
            STRING,
            // ... autres types si nécessaire
        };

        class TensorView {
        public:
            // Constructeurs et destructeur
            // ...

            // Accesseurs
            const std::vector<size_t>& dims() const;
            DataType type() const;
            void* data() const;

            template<typename T>
            T* as() {
                // Vérification de type ici
                return static_cast<T*>(data());
            }

        private:
            std::vector<size_t> dimensions_;
            DataType data_type_;
            std::unique_ptr<char[]> buffer_; // Pointeur vers le buffer de données
        };
        ```

3.  **Définition de la Signature de l'API (`read_tensor`)**
    *   La fonction principale aura deux surcharges pour correspondre à la demande.
    *   Définition dans `direct_access_api.h`:
        ```cpp
        namespace imas {
        namespace direct_access {

        TensorView read_tensor(
            const std::string& ids_name,
            const std::string& path
        );

        TensorView read_tensor(
            const std::string& ids_name,
            const std::string& path_template,
            const std::vector<int>& aos_indices
        );

        } // namespace direct_access
        } // namespace imas
        ```

4.  **Intégration au Système de Build (CMake)**
    *   Créer `src/direct_access_api.cpp` pour l'implémentation.
    *   Modifier le `src/CMakeLists.txt` pour compiler ce nouveau fichier et l'ajouter à la bibliothèque `al-core`.

---

## Phase 2: Analyseur de Chemin (Path Parser)

Le cœur de la flexibilité de l'API réside dans sa capacité à interpréter les chemins.

1.  **Créer une Classe `PathParser`**
    *   Créer une classe dédiée dans un nouveau fichier (`src/path_parser.h` et `.cpp`).
    *   Le constructeur prendra le chemin brut (ex: `"A[3]/B[:]/data"`).

2.  **Implémenter la Logique d'Analyse**
    *   La méthode `parse()` décomposera le chemin en segments.
    *   Chaque segment contiendra le nom du nœud (`A`, `B`, `data`) et l'information de sélection (`[3]`, `[:]`, `null`).
    *   Une structure interne sera utilisée pour stocker ce résultat structuré.
        ```cpp
        enum class SelectionType { ALL, INDEX, SLICE };
        struct PathSegment {
            std::string node_name;
            SelectionType selection;
            size_t index;
            // D'autres champs pour les slices plus tard
        };
        ```

3.  **Tests Unitaires**
    *   Créer `tests/test_path_parser.cpp`.
    *   Tester de multiples formats :
        *   `"a/b/c"` (pas d'indices)
        *   `"a[0]/b/c"` (un indice)
        *   `"a[3]/b[:]/c"` (indice et slice complète)
        *   Chemins invalides pour tester la gestion d'erreurs.

---

## Phase 3: Implémentation de la Lecture (AoS Statique)

Cette phase implémente la lecture pour le cas le plus simple : les tableaux de structures (AoS) statiques.

1.  **Logique d'Accès aux Données**
    *   Dans `direct_access_api.cpp`, implémenter la fonction `read_tensor`.
    *   Utiliser le `PathParser` pour décomposer le chemin.
    *   Interagir avec la bibliothèque HDF5 bas niveau pour ouvrir le fichier correspondant à l'IDS.

2.  **Récupération des Métadonnées**
    *   Pour le nœud final (la "feuille" du chemin), construire le chemin vers les métadonnées de dimension (ex: `flux_loop&flux&data@dim` dans l'index PanzerDB).
    *   Lire cet attribut pour obtenir les dimensions du tenseur complet.

3.  **Sélection des Données (HDF5 Hyperslab)**
    *   Construire une sélection "hyperslab" HDF5 en utilisant les indices extraits par le `PathParser`.
    *   Par exemple, pour `"flux_loop[3]/flux/data"`, si `data` est un tableau de 100x10, la sélection HDF5 ciblera le bloc `(3, 0, 0)` à `(3, 99, 9)` (en supposant que `flux_loop` est la première dimension).

4.  **Remplissage de `TensorView`**
    *   Allouer un buffer de la taille requise pour la slice.
    *   Exécuter la lecture HDF5 (`H5Dread`) dans ce buffer.
    *   Construire et retourner l'objet `TensorView` avec le buffer, le type de données et les dimensions de la *slice* lue.

5.  **Tests d'Intégration**
    *   Créer un fichier de test HDF5 avec une structure connue.
    *   Écrire un test `tests/test_direct_api_static.cpp` qui appelle `read_tensor` et vérifie que les données, les dimensions et le type retournés sont corrects.

---

## Phase 4: Prise en Charge des AoS Dynamiques (Slices Temporelles)

Cette phase étend l'API pour gérer les structures de données qui évoluent dans le temps.

1.  **Identifier l'AoS Dynamique**
    *   Le système devra déterminer quel segment du chemin correspond à un AoS dynamique (généralement basé sur la présence d'un nœud `time`).
    *   La logique de lecture devra s'adapter lorsqu'une sélection temporelle est détectée.

2.  **Implémenter la Lecture Temporelle**
    *   La sélection ne se fait plus par un simple indice de tableau, mais par une plage de temps.
    *   Une logique supplémentaire sera nécessaire pour mapper la sélection temporelle (ex: `time[10:20]`) aux indices correspondants dans les données.

3.  **Tests Unitaires**
    *   Étendre les tests avec un fichier HDF5 contenant un AoS dynamique.
    *   Valider que la lecture d'une slice temporelle retourne les bons "pas de temps".

---

## Phase 5 (Futur): Slicing Avancé et Améliorations

Planification pour les évolutions futures afin de garantir que l'architecture est pérenne.

1.  **Étendre le `PathParser`**
    *   Modifier le parser pour qu'il reconnaisse les syntaxes avancées : `[2:]`, `[:10]`, `[1:10:2]` (début, fin, pas).

2.  **Mettre à Jour la Logique de Sélection HDF5**
    *   Adapter la création de l'hyperslab pour qu'elle corresponde à ces sélections avancées.

3.  **Optimisation des Performances**
    *   Analyser les performances et optimiser l'allocation mémoire et les lectures HDF5.
    *   Envisager un système de cache de métadonnées si l'ouverture et la lecture des attributs deviennent un goulot d'étranglement.
