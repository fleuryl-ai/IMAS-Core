#include "direct_access_api.h"
#include <iostream>
#include <cassert>
#include <stdexcept>

void test_temporal_slice_read_fails_gracefully() {
    std::cout << "Running test: test_temporal_slice_read_fails_gracefully" << std::endl;
    
    try {
        imas::direct_access::TensorView view = imas::direct_access::read_tensor(
            "test_api", "pf_active/channel[3]/time_of_flight[:]");
        
        assert(false && "API call should have failed because test_api.h5 does not exist.");

    } catch (const std::exception& e) {
        // C'est le comportement attendu jusqu'à ce que le fichier de test existe.
        std::string msg = e.what();
        std::cout << "Caught expected exception: " << msg << std::endl;
        assert(msg.find("Failed to open") != std::string::npos || msg.find("unable to open file") != std::string::npos);
        std::cout << "PASSED" << std::endl;
    }
}

int main() {
    test_temporal_slice_read_fails_gracefully();

    std::cout << "\nAll direct_api tests passed!" << std::endl;
    return 0;
}
