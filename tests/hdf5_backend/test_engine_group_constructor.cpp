// test_panzer_group_constructor.cpp
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <hdf5.h> // Nécessaire pour les opérations sur les fichiers/groupes HDF5

int main() {
    std::cout << "=== TEST PanzerDB: Constructeur avec un identifiant de groupe ===\n\n";
    const std::string filename = "test_group_constructor.h5";
    const std::string group_name = "magnetics";
    hid_t file_id = -1;
    hid_t group_id = -1;

    try {
        // ===================================================================
        // 1. Créer un fichier HDF5, un groupe, et y écrire un scalaire via PanzerDB
        // ===================================================================
        {
            // Créer le fichier HDF5
            file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            assert(file_id >= 0);
            std::cout << "Fichier HDF5 '" << filename << "' créé.\n";

            // Créer un groupe dans le fichier
            group_id = H5Gcreate2(file_id, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            assert(group_id >= 0);
            std::cout << "Groupe '" << group_name << "' créé.\n";

            // Instancier PanzerDB avec l'ID du groupe pour l'écriture
            // Le dernier 'false' indique à PanzerDB de ne pas fermer l'ID du groupe à sa destruction
            PanzerDB db(group_id, PanzerDB::OpenMode::WRITE, true, false);

            std::cout << "Écriture d'un scalaire 'temperature' dans le groupe '" << group_name << "'...\n";
            double temp_value = 123.45;
            db.writeData("temperature", {}, &temp_value, 1);

            db.close(); // Vide les données sur le disque
            std::cout << "PanzerDB (écriture) fermé.\n\n";
            
            // On ne ferme pas le groupe ici, car on va le réutiliser pour la lecture.
        }

        // ===================================================================
        // 2. Réouverture du groupe et lecture du scalaire
        // ===================================================================
        {
            // Instancier PanzerDB avec le même ID de groupe pour la lecture
            // Le dernier 'true' indique à PanzerDB de fermer l'ID du groupe à sa destruction
            PanzerDB db(group_id, PanzerDB::OpenMode::READ, true, true);
            std::cout << "PanzerDB (lecture) ouvert sur le groupe existant.\n";

            std::cout << "Lecture du scalaire 'temperature'...\n";
            int status = -1;
            double temp = db.readScalar<double>("temperature", &status);
            assert(status == 0);
            assert(std::abs(temp - 123.45) < 1e-9);
            std::cout << "Scalaire 'temperature' lu avec succès: " << temp << "\n";

            db.close(); // PanzerDB ferme aussi group_id car close_loc_id_on_exit=true
            group_id = -1; // L'ID n'est plus valide
            std::cout << "PanzerDB (lecture) et groupe HDF5 fermés.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Une exception a été capturée: " << e.what() << std::endl;
        if (group_id >= 0) H5Gclose(group_id);
        if (file_id >= 0) H5Fclose(file_id);
        return 1;
    }

    // ===================================================================
    // 3. Nettoyage final
    // ===================================================================
    if (file_id >= 0) {
        H5Fclose(file_id);
        std::cout << "Fichier HDF5 fermé.\n";
    }

    std::cout << "\nTOUS LES TESTS RÉUSSIS !\n";
    return 0;
}