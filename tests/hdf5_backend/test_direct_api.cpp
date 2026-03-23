#include "direct_access_api.h"
#include <iostream>
#include <cassert>
#include <stdexcept>

// Ce test est conçu pour être un test d'intégration.
// Il nécessite un fichier HDF5 de test ("test_api.h5") avec une structure de données connue.
// Pour l'instant, il ne fait que vérifier que l'appel à l'API existe et
// qu'il lance une exception "not implemented", ce qui est le comportement attendu.

void test_static_aos_read_fails_gracefully() {
    std::cout << "Running test: test_static_aos_read_fails_gracefully" << std::endl;
    
    try {
        // On tente de lire une donnée qui existerait dans un fichier de test.
        // Chemin: magnetics.flux_loop[2].flux.data
        imas::direct_access::TensorView view = imas::direct_access::read_tensor(
            "test_api", "magnetics/flux_loop[2]/flux/data");
        
        // Si on arrive ici, le test a échoué car l'exception n'a pas été lancée.
        assert(false && "API call did not throw the expected 'not implemented' exception.");

    } catch (const std::runtime_error& e) {
        // C'est le comportement attendu pour l'instant.
        std::string msg = e.what();
        std::cout << "Caught expected exception: " << msg << std::endl;
        assert(msg.find("not implemented") != std::string::npos);
    }
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    test_static_aos_read_fails_gracefully();

    std::cout << "\nAll direct_api tests passed (for now)!" << std::endl;
    return 0;
}
