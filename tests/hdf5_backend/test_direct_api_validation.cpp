#include "direct_access_api.h"
#include "panzerdb.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>

void generate_test_file(const std::string& filename) {
    if (std::filesystem::exists(filename)) std::filesystem::remove(filename);
    PanzerDB db(filename, PanzerDB::OpenMode::WRITE);

    const int time_steps = 5;
    const int ion_size = 3;

    db.beginArray("profiles_1d", "time");
    for (int t = 0; t < time_steps; ++t) {
        db.setCurrentArrayIndex(t);
        db.beginArray("ion", ion_size);
        for (int i = 0; i < ion_size; ++i) {
            db.setCurrentArrayIndex(i);
            double z_ion_val = 10.0 + t + i;
            db.writeData("z_ion", {}, &z_ion_val, 1);
        }
        db.endArray(); // ion
    }
    db.endArray(); // profiles_1d
    db.close();
}

void validate_slice_read() {
    const std::string ids_name = "test_direct_api_validation";
    // On lit les tranches de temps t=1 et t=2 pour tous les ions.
    const std::string path = "profiles_1d[1:3]/ion/z_ion";
    
    auto view = imas::direct_access::read_tensor(ids_name, path);

    // Dimensions attendues: [2 time_slices, 3 ions]
    assert(view.dims().size() == 2);
    assert(view.dims()[0] == 2); // t=1, t=2
    assert(view.dims()[1] == 3); // 3 ions

    const double* data = view.as<double>();
    for (int t_slice = 0; t_slice < 2; ++t_slice) {
        int t = 1 + t_slice;
        for (int i = 0; i < 3; ++i) {
            double expected = 10.0 + t + i;
            double actual = data[t_slice * 3 + i];
            assert(std::abs(expected - actual) < 1e-9);
        }
    }
    std::cout << "[OK] Slice content validated.\n";
}

int main() {
    generate_test_file("test_direct_api_validation.h5");
    validate_slice_read();
    std::cout << "\nAll direct_api_validation tests passed!\n";
    return 0;
}
