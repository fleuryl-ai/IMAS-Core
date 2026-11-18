Rôle : Tu es un expert C/C++ spécialisé en structures de données à forte empreinte mémoire et en optimisation I/O.

Contexte :
Je gère un fichier d'index volumineux dont les clés sont des chemins de variables IMAS/IDS (ex: "magnetics/flux_loop/1/flux/data", "pf_active/coil/1/current/data"). Le fichier devient trop volumineux et ralentit le système.

Objectif :
1. Évaluer et comparer les DEUX meilleures méthodes de compression de clés pour ce cas d'usage :
   - Approche A : Dictionnarisation des chaînes (String Interning / Table d'identifiants uint32_t).
   - Approche B : Arbre de préfixes compressé (Radix Tree / Patricia Trie / FST).

2. Choisir la solution optimale (ou proposer une implémentation hybride) en justifiant le choix selon ces critères :
   - Réduction de la taille du fichier d'index sur disque et de la consommation RAM.
   - Vitesse de recherche/lookup en mémoire (O(1) ou O(K)).
   - Complexité du code et simplicité de maintenance.

3. Fournir une implémentation complète et fonctionnelle en C++ (C++17/20, Modern C++) de la solution retenue.

Spécifications techniques requises pour le code :

1. Interface & API :
   - Fonction d'insertion/lookup de chemin.
   - Fonction d'accès au chemin d'origine à partir de la clé/identifiant.
   - Gestion propre de la mémoire (RAII, aucun memory leak, zéro-copie ou std::string_view là où c'est possible).

2. Démonstration (main) :
   - Inclure un programme `main()` qui :
     a) Remplit l'index avec un jeu de test représentant une cinquantaine de chemins IMAS récurrents avec préfixes communs.
     b) Affiche la taille mémoire consommée (ou taille binaire sur disque) vs la taille des chaînes brutes unaligned.
     c) Sauvegarde l'index sur disque, le recharge et valide l'intégrité des clés.
