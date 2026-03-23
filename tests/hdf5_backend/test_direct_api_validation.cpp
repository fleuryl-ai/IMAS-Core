#include "direct_access_api.h"
#include "panzerdb.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <numeric>

// ÉTAPE 1: Génération de données plus complexes avec un AoS imbriqué
void generate_test_file(const std::string& filename) {
    if (std::filesystem::exists(filename)) std::filesystem::remove(filename);
    PanzerDB db(filename, PanzerDB::OpenMode::WRITE);

    const int time_steps = 5;
    const int ion_size = 3;
    const int state_size = 2; // Nouveau niveau d'imbrication

    // Écrire le vecteur de temps
    std::vector<double> time_data(time_steps);
    std::iota(time_data.begin(), time_data.end(), 1.0); // time = [1.0, 2.0, 3.0, 4.0, 5.0]
    db.writeDataSlices("time", {1}, time_data.data(), time_steps, "time");

    db.beginArray("profiles_1d", "time");
    for (int t = 0; t < time_steps; ++t) {
        db.setCurrentArrayIndex(t);
        db.beginArray("ion", ion_size);
        for (int i = 0; i < ion_size; ++i) {
            db.setCurrentArrayIndex(i);
            db.beginArray("state", state_size);
            for (int s = 0; s < state_size; ++s) {
                db.setCurrentArrayIndex(s);
                // Valeur unique pour chaque point de donnée
                double z_ion_val = 100.0 + (t * 10.0) + (i * 1.0) + (s * 0.1);
                db.writeData("z_ion", {}, &z_ion_val, 1);
            }
            db.endArray(); // state
        }
        db.endArray(); // ion
    }
    db.endArray(); // profiles_1d
    db.close();
}

// ÉTAPE 2: Validation du slice par indice sur la structure profonde
void validate_index_slice_read() {
    std::cout << "\n--- Validating Index Slice Read ---\n";
    const std::string ids_name = "test_direct_api_validation";
    const std::string path = "profiles_1d[1:3]/ion/state/z_ion";
    
    auto view = imas::direct_access::read_tensor(ids_name, path);

    assert(view.dims().size() == 3);
    assert(view.dims()[0] == 2); // t=1, t=2
    assert(view.dims()[1] == 3); // 3 ions
    assert(view.dims()[2] == 2); // 2 states

    const double* data = view.as<double>();
    for (int t_slice = 0; t_slice < 2; ++t_slice) {
        int t = 1 + t_slice;
        for (int i = 0; i < 3; ++i) {
            for (int s = 0; s < 2; ++s) {
                double expected = 100.0 + (t * 10.0) + (i * 1.0) + (s * 0.1);
                double actual = data[(t_slice * 3 * 2) + (i * 2) + s];
                assert(std::abs(expected - actual) < 1e-9);
            }
        }
    }
    std::cout << "[OK] Index slice content validated.\n";
}

// ÉTAPE 3: Validation pour le slice temporel
void validate_time_slice_read() {
    std::cout << "\n--- Validating Time Slice Read ---\n";
    const std::string ids_name = "test_direct_api_validation";
    const std::string path = "profiles_1d[time=1.5:3.5]/ion/state/z_ion";
    
    auto view = imas::direct_access::read_tensor(ids_name, path);

    assert(view.dims().size() == 3);
    assert(view.dims()[0] == 2); // t=2.0 (index 1), t=3.0 (index 2)
    assert(view.dims()[1] == 3); // 3 ions
    assert(view.dims()[2] == 2); // 2 states

    const double* data = view.as<double>();
    for (int t_slice = 0; t_slice < 2; ++t_slice) {
        int t = 2 + t_slice;
        for (int i = 0; i < 3; ++i) {
            for (int s = 0; s < 2; ++s) {
                double expected = 100.0 + (t * 10.0) + (i * 1.0) + (s * 0.1);
                double actual = data[(t_slice * 3 * 2) + (i * 2) + s];
                assert(std::abs(expected - actual) < 1e-9);
            }
        }
    }
    std::cout << "[OK] Time slice content validated.\n";
}

// ÉTAPE 4: Nouvelle validation pour l'interpolation au plus proche
void validate_time_interp_read() {
    std::cout << "\n--- Validating Time Interpolation (Closest) ---\n";
    const std::string ids_name = "test_direct_api_validation";
    // Le temps 2.7 est plus proche de 3.0 (index 2) que de 2.0 (index 1).
    const std::string path = "profiles_1d[time=2.7]/ion/state/z_ion";
    
    auto view = imas::direct_access::read_tensor(ids_name, path);

    // Dimensions attendues: [1 time_slice, 3 ions, 2 states]
    assert(view.dims().size() == 3);
    assert(view.dims()[0] == 1); // un seul temps, le plus proche
    assert(view.dims()[1] == 3); // 3 ions
    assert(view.dims()[2] == 2); // 2 states

    const double* data = view.as<double>();
    int t = 2; // L'indice de temps attendu est 2 (correspondant à t=3.0)
    for (int i = 0; i < 3; ++i) {
        for (int s = 0; s < 2; ++s) {
            double expected = 100.0 + (t * 10.0) + (i * 1.0) + (s * 0.1);
            double actual = data[(i * 2) + s];
            assert(std::abs(expected - actual) < 1e-9);
        }
    }
    std::cout << "[OK] Time interpolation content validated.\n";
}


int main() {
    generate_test_file("test_direct_api_validation.h5");
    
    validate_index_slice_read();
    validate_time_slice_read();
    validate_time_interp_read(); // Appel du nouveau test
    
    std::cout << "\nAll direct_api_validation tests passed!\n";
    return 0;
}
