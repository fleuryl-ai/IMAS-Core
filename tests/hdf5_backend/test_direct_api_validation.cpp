#include "direct_access_api.h"
#include "panzerdb.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <numeric>

void generate_test_file(const std::string& filename) {
    if (std::filesystem::exists(filename)) std::filesystem::remove(filename);
    PanzerDB db(filename, PanzerDB::OpenMode::WRITE);

    // --- Écrire la propriété homogeneous_time ---
    int32_t homogeneous_time_val = 0; 
    db.writeData("ids_properties/homogeneous_time", {}, &homogeneous_time_val, 1);

    // --- Données pour les tests ---
    const int time_steps = 5;
    const int ion_size = 3;
    const int state_size = 2;

    std::vector<double> global_time_data(time_steps);
    std::iota(global_time_data.begin(), global_time_data.end(), 10.0);
    db.writeDataSlices("time", {1}, global_time_data.data(), time_steps, "time");

    db.beginArray("profiles_1d", "time");
    for (int t = 0; t < time_steps; ++t) {
        db.setCurrentArrayIndex(t);
        double time_val = static_cast<double>(t + 1);
        db.writeData("time", {}, &time_val, 1);

        double a_val = 10*t;
        db.writeDataSlices("sig_dyn", {}, &a_val, 1, "time");

        db.beginArray("ion", ion_size);
        for (int i = 0; i < ion_size; ++i) {
            db.setCurrentArrayIndex(i);
            db.beginArray("state", state_size);
            for (int s = 0; s < state_size; ++s) {
                db.setCurrentArrayIndex(s);
                
                // Écrire la donnée double
                double z_ion_val = 100.0 + (t * 10.0) + (i * 1.0) + (s * 0.1);
                db.writeData("z_ion", {}, &z_ion_val, 1);
                
                // CORRECTION : Écrire la donnée entière qui manquait
                int32_t a_z_val = t * 100 + i * 10 + s; // Valeur unique pour le test
                db.writeData("a_z", {}, &a_z_val, 1);
            }
            db.endArray(); // state
        }
        db.endArray(); // ion
    }
    db.endArray(); // profiles_1d

    // --- Données pour le test de liste de chaînes de caractères (CORRIGÉ) ---
    const std::vector<const char*> diags = {"bolometer", "interferometer", "thomson_scattering", "ece"};
    db.beginArray("diagnostics", diags.size());
    for (size_t i = 0; i < diags.size(); ++i) {
        db.setCurrentArrayIndex(i);
        // CORRECTION: Utiliser writeDataSlices, qui est la bonne méthode pour les chaînes.
        db.writeDataSlices("name", {1}, &diags[i], 1, "");
    }
    db.endArray(); // diagnostics
    db.writeMetaData("profiles_1d/ion/state/z_ion@units", "eV");
    db.close();
}

void validate_index_slice_read() {
    std::cout << "\n--- Validating Index Slice Read ---\n";
    const std::string ids_name = "test_direct_api_validation";
    const std::string path = "profiles_1d[1:3]/ion/state/z_ion";
    
    auto view = imas::direct_access::read_tensor(ids_name + ".h5", path);

    assert(view.dims().size() == 3);
    assert(view.dims()[0] == 2);
    assert(view.dims()[1] == 3);
    assert(view.dims()[2] == 2);

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

void validate_time_slice_read() {
    std::cout << "\n--- Validating Time Slice Read ---\n";
    const std::string ids_name = "test_direct_api_validation";
    const std::string path = "profiles_1d[time=1.5:3.5]/ion/state/z_ion";
    
    auto view = imas::direct_access::read_tensor(ids_name + ".h5", path);

    assert(view.dims().size() == 3);
    assert(view.dims()[0] == 2);
    assert(view.dims()[1] == 3);
    assert(view.dims()[2] == 2);

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

void validate_time_interp_read() {
    std::cout << "\n--- Validating Time Interpolation (Closest) ---\n";
    const std::string ids_name = "test_direct_api_validation";
    const std::string path = "profiles_1d[time=2.7]/ion/state/z_ion";
    
    auto view = imas::direct_access::read_tensor(ids_name + ".h5", path);

    assert(view.dims().size() == 3);
    assert(view.dims()[0] == 1);
    assert(view.dims()[1] == 3);
    assert(view.dims()[2] == 2);

    const double* data = view.as<double>();
    int t = 2; // t=2.7 est plus proche de 3.0 (index 2)
    for (int i = 0; i < 3; ++i) {
        for (int s = 0; s < 2; ++s) {
            double expected = 100.0 + (t * 10.0) + (i * 1.0) + (s * 0.1);
            double actual = data[(i * 2) + s];
            assert(std::abs(expected - actual) < 1e-9);
        }
    }
    std::cout << "[OK] Time interpolation content validated.\n";
}

void validate_linear_interp_read() {
    std::cout << "\n--- Validating Time Interpolation (Linear) ---\n";
    const std::string ids_name = "test_direct_api_validation";
    // Le temps est [1.0, 2.0, 3.0, 4.0, 5.0].
    // On demande le point de temps 2.5, qui est à mi-chemin entre 2.0 (index 1) et 3.0 (index 2).
    const std::string path = "profiles_1d[time=2.5,interp=linear]/ion/state/z_ion";
    
    auto view = imas::direct_access::read_tensor(ids_name + ".h5", path);

    // Les dimensions doivent correspondre à une seule tranche de temps interpolée
    assert(view.dims().size() == 2);
    assert(view.dims()[0] == 3); // 3 ions
    assert(view.dims()[1] == 2); // 2 states

    const double* data = view.as<double>();

    // Vérifier les valeurs interpolées
    for (int i = 0; i < 3; ++i) {
        for (int s = 0; s < 2; ++s) {
            // Valeur au temps t=2.0 (index 1)
            double val_t1 = 100.0 + (1 * 10.0) + (i * 1.0) + (s * 0.1);
            // Valeur au temps t=3.0 (index 2)
            double val_t2 = 100.0 + (2 * 10.0) + (i * 1.0) + (s * 0.1);
            // La valeur interpolée à t=2.5 doit être la moyenne
            double expected = (val_t1 + val_t2) / 2.0;
            
            double actual = data[i * 2 + s];
            assert(std::abs(expected - actual) < 1e-9);
        }
    }
    std::cout << "[OK] Linear interpolation content validated.\n";
}

void validate_in_memory_slice() {
    std::cout << "\n--- Validating In-Memory Slicing ---\n";
    const std::string ids_name = "test_direct_api_validation";
    // 1. Lire un bloc de données multidimensionnel en mémoire
    const std::string path = "profiles_1d[1:3]/ion/state/z_ion"; // Dims: {2, 3, 2}
    auto view_orig = imas::direct_access::read_tensor(ids_name + ".h5", path);

    // 2. Test 1: Extraire la première tranche de temps (indice 0 de la vue)
    // Cela correspond au temps d'origine t=1
    auto view_t1 = view_orig.slice({ imas::direct_access::SliceSelection::at(0) });
    
    assert(view_t1.dims().size() == 2);
    assert(view_t1.dims()[0] == 3); // 3 ions
    assert(view_t1.dims()[1] == 2); // 2 states

    const double* data_t1 = view_t1.as<double>();
    int t = 1; // Temps d'origine
    for (int i = 0; i < 3; ++i) {
        for (int s = 0; s < 2; ++s) {
            double expected = 100.0 + (t * 10.0) + (i * 1.0) + (s * 0.1);
            double actual = data_t1[i * 2 + s];
            assert(std::abs(expected - actual) < 1e-9);
        }
    }
    std::cout << "[OK] In-memory slice on first dimension validated.\n";

    // 3. Test 2: Extraire le deuxième ion (indice 1) sur tous les temps
    // C'est un test important car les données ne sont pas contiguës dans le buffer d'origine
    auto view_i1 = view_orig.slice({ imas::direct_access::SliceSelection::all(), imas::direct_access::SliceSelection::at(1) });

    assert(view_i1.dims().size() == 2);
    assert(view_i1.dims()[0] == 2); // 2 tranches de temps
    assert(view_i1.dims()[1] == 2); // 2 states

    const double* data_i1 = view_i1.as<double>();
    int i = 1; // Ion d'origine
    for (int t_slice = 0; t_slice < 2; ++t_slice) {
        t = 1 + t_slice;
        for (int s = 0; s < 2; ++s) {
            double expected = 100.0 + (t * 10.0) + (i * 1.0) + (s * 0.1);
            double actual = data_i1[t_slice * 2 + s];
            assert(std::abs(expected - actual) < 1e-9);
        }
    }
    std::cout << "[OK] In-memory slice on middle dimension (non-contiguous) validated.\n";
}

void validate_list_of_strings_read() {
    std::cout << "\n--- Validating List of Strings Read ---\n";
    const std::string ids_name = "test_direct_api_validation";
    const std::string path = "diagnostics/name";

    auto view = imas::direct_access::read_tensor(ids_name + ".h5", path);

    const std::vector<std::string> expected_strings = {"bolometer", "interferometer", "thomson_scattering", "ece"};
    size_t max_len = 0;
    for(const auto& s : expected_strings) {
        if (s.length() > max_len) max_len = s.length();
    }
    size_t string_dim = max_len + 1;

    assert(view.type() == imas::direct_access::DataType::LIST_OF_STRINGS);
    assert(view.dims().size() == 2);
    assert(view.dims()[0] == 4); // 4 chaînes
    assert(view.dims()[1] == string_dim);

    const char* data = view.as<char>();

    for (size_t i = 0; i < expected_strings.size(); ++i) {
        std::string actual(data + i * string_dim);
        assert(actual == expected_strings[i]);
    }

    std::cout << "[OK] List of strings content validated.\n";
}

void validate_int_read() {
    std::cout << "\n--- Validating INT32 Read ---\n";
    const std::string ids_name = "test_direct_api_validation";
    // Lire la donnée entière pour le temps t=1, ion i=0, state s=0
    const std::string path = "profiles_1d[1]/ion[0]/state[0]/a_z";
    
    auto view = imas::direct_access::read_tensor(ids_name + ".h5", path);

    // Vérifier le type et les dimensions
    assert(view.type() == imas::direct_access::DataType::INT32);
    assert(view.dims().size() == 1);
    assert(view.dims()[0] == 1);

    // Vérifier la valeur
    const int32_t* data = view.as<int32_t>();
    int32_t expected = 6;
    assert(data[0] == expected);

    std::cout << "[OK] INT32 content validated.\n";
}

void validate_metadata_read() {
    std::cout << "\n--- Validating Metadata Read ---\n";
    const std::string ids_name = "test_direct_api_validation";
    // On lit n'importe quelle instance de z_ion, les métadonnées sont les mêmes pour toutes.
    const std::string path = "profiles_1d[0]/ion[0]/state[0]/z_ion";
    
    auto view = imas::direct_access::read_tensor(ids_name + ".h5", path);

    // Récupérer et valider les métadonnées
    const auto& metadata = view.metadata();
    assert(!metadata.empty());
    assert(metadata.count("units") == 1);
    assert(metadata.at("units") == "eV");

    std::cout << "[OK] Metadata content validated.\n";
}

void validate_dynamic_parent_read() {
    std::cout << "\n--- Validating Read from Dynamic Parent (sig_dyn) ---\n";
    const std::string ids_name = "test_direct_api_validation";
    
    // Test 1: Lecture d'un slice temporel (t=2, index temporel 2)
    const std::string path_slice = "profiles_1d[2]/sig_dyn";
    auto view_slice = imas::direct_access::read_tensor(ids_name + ".h5", path_slice);
    
    assert(view_slice.type() == imas::direct_access::DataType::DOUBLE);
    assert(view_slice.dims().size() == 1); // C'est un scalaire par slice, donc 1 valeur
    assert(view_slice.dims()[0] == 1);
    assert(std::abs(view_slice.as<double>()[0] - 20.0) < 1e-9);

    // Test 2: Lecture de tous les temps (-1)
    const std::string path_all = "profiles_1d/sig_dyn";
    auto view_all = imas::direct_access::read_tensor(ids_name + ".h5", path_all);
    
    assert(view_all.type() == imas::direct_access::DataType::DOUBLE);
    assert(view_all.dims().size() == 1);
    assert(view_all.dims()[0] == 5); // 5 pas de temps
    
    const double* data_all = view_all.as<double>();
    for (int t = 0; t < 5; ++t) {
        assert(std::abs(data_all[t] - (10.0 * t)) < 1e-9);
    }
    
    std::cout << "[OK] Read from Dynamic Parent validated.\n";
}

int main() {
    generate_test_file("test_direct_api_validation.h5");
    
    validate_index_slice_read();
    validate_time_slice_read();
    validate_time_interp_read();
    validate_linear_interp_read();
    validate_list_of_strings_read();
    validate_int_read();
    validate_in_memory_slice();
    validate_metadata_read();
    validate_dynamic_parent_read(); // Appel du nouveau test

    std::cout << "\nAll direct_api_validation tests passed!\n";
    return 0;
}
