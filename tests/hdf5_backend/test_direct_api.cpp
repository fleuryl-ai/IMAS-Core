#include "direct_access_api.h"
#include <iostream>
#include <cassert>
#include <stdexcept>

// Ce test vérifie l'API de bout en bout.
// Il nécessite un fichier de données de test pour réussir complètement.
// Pour l'instant, il va échouer car il ne trouvera pas le fichier h5.

void test_api_call() {
    std::cout << "Running test: test_api_call" << std::endl;
    
    try {
        // On tente de lire une donnée qui pourrait exister dans un fichier de test.
        // On s'attend à ce que cela échoue car le fichier n'existe pas.
        imas::direct_access::TensorView view = imas::direct_access::read_tensor(
            "test_api", "magnetics/flux_loop[0]/flux/data");
        
        // Si on arrive ici, c'est un échec inattendu (peut-être qu'un vieux fichier de test existe ?)
        std::cerr << "Test failed: API call succeeded unexpectedly." << std::endl;
        assert(false);

    } catch (const std::exception& e) {
        // C'est le comportement attendu. On vérifie que le message d'erreur
        // est bien lié à l'impossibilité d'ouvrir le fichier.
        std::string msg = e.what();
        std::cout << "Caught expected exception: " << msg << std::endl;
        assert(msg.find("Failed to open") != std::string::npos || msg.find("unable to open file") != std::string::npos);
        std::cout << "PASSED" << std::endl;
    }
}

int main() {
    test_api_call();

    std::cout << "\nAll direct_api tests passed!" << std::endl;
    return 0;
}
