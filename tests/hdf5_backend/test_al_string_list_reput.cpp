// @file  test_al_string_list_reput.cpp
// @brief A second ids_put() on the same IDS must not describe the new strings with the
//        element boundaries of the first one.
//
// This is the path that reaches the hole: an AL GLOBAL put hands HDF5Writer_v2 an already
// open IDS file, so its PanzerDB(WRITE) reuses that file instead of truncating it, and drops
// the tables it is about to recreate. /list_spans belongs to them -- and it is not an empty
// optimization, it holds the boundaries between the elements of a list of strings whose
// elements are longer than one 512 B slot. Left in place, its rows are keyed by row id, the
// row ids restart at 0, and the new list is read back cut and shifted: values that look
// plausible and are not.
//
// The two puts differ in WHICH element overflows, so a surviving table cannot pass by having
// the right number of rows.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

const std::string URI = "imas:hdf5?path=./test_db_string_list_reput";

static int failures = 0;

// A string longer than one 512 B slot, hence stored across several.
static std::string long_string(size_t len, char base) {
    std::string s(len, 'x');
    for (size_t i = 0; i < len; ++i) s[i] = (char)(base + (i % 26));
    return s;
}

// One GLOBAL put of a list of strings: coordinate_system(0)/coordinate(0)/value_labels.
// The paths follow test_al_string_list_1d_slice, which the dictionary accepts.
static void put_labels(HDF5Backend& backend, DataEntryContext& ctx,
                       const std::vector<std::string>& labels, int mode) {
    backend.openPulse(&ctx, mode);
    OperationContext opCtx(&ctx, "test_ids", "", WRITE_OP);
    backend.beginAction(&opCtx);

    int homogeneous_time = 1;
    backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time,
                      alconst::integer_data, 0, nullptr);

    int cs_size = 1;
    ArraystructContext csCtx(&opCtx, "coordinate_system", "");
    backend.beginArraystructAction(&csCtx, &cs_size);
    int coord_size = 1;
    ArraystructContext coordCtx(&csCtx, "coordinate", "");
    backend.beginArraystructAction(&coordCtx, &coord_size);

    // A list of strings is a 2D char array: one row per element, all rows the same width.
    int max_len = 1;
    for (const std::string& s : labels) max_len = std::max(max_len, (int)s.size() + 1);
    std::vector<char> buffer(labels.size() * max_len, 0);
    for (size_t i = 0; i < labels.size(); ++i)
        std::memcpy(buffer.data() + i * max_len, labels[i].data(), labels[i].size());
    int dim = 2;
    int size[] = {(int)labels.size(), max_len};
    backend.writeData(&coordCtx, "value_labels", "", buffer.data(), alconst::char_data, dim, size);

    backend.endAction(&coordCtx);
    backend.endAction(&csCtx);
    backend.endAction(&opCtx);
    backend.closePulse(&ctx, mode);
}

// Reads the list back and compares it against what was written.
static void get_labels(HDF5Backend& backend, DataEntryContext& ctx,
                       const std::vector<std::string>& want) {
    backend.openPulse(&ctx, OPEN_PULSE);
    OperationContext opCtx(&ctx, "test_ids", "", READ_OP);
    backend.beginAction(&opCtx);

    int cs_size = 0;
    ArraystructContext csCtx(&opCtx, "coordinate_system", "");
    backend.beginArraystructAction(&csCtx, &cs_size);
    int coord_size = 0;
    ArraystructContext coordCtx(&csCtx, "coordinate", "");
    backend.beginArraystructAction(&coordCtx, &coord_size);

    void* data = nullptr;
    int datatype = alconst::char_data;
    int dim = 0;
    int size[H5S_MAX_RANK];
    backend.readData(&coordCtx, "value_labels", "", &data, &datatype, &dim, size);

    if (!data || dim != 2 || size[0] != (int)want.size()) {
        std::cout << "  FAIL: value_labels came back as " << dim << "D";
        for (int i = 0; i < dim; ++i) std::cout << " [" << size[i] << "]";
        std::cout << ", expected 2D [" << want.size() << " x ...]\n";
        ++failures;
    } else {
        const char* chars = static_cast<const char*>(data);
        for (size_t i = 0; i < want.size(); ++i) {
            const std::string got(chars + i * size[1]);
            if (got != want[i]) {
                std::cout << "  FAIL: element " << i << " is " << got.size() << " chars ("
                          << got.substr(0, 24) << "...), expected " << want[i].size()
                          << " chars (" << want[i].substr(0, 24) << "...)\n";
                ++failures;
            }
        }
    }
    free(data);

    backend.endAction(&coordCtx);
    backend.endAction(&csCtx);
    backend.endAction(&opCtx);
    backend.closePulse(&ctx, OPEN_PULSE);
}

int main() {
    try {
        std::cout << "\n=== Test: two ids_put() of a list of long strings on one IDS ===\n";
        if (fs::exists(URI.substr(URI.find('=') + 1)))
            fs::remove_all(URI.substr(URI.find('=') + 1));

        // The second list is the same length and the same number of elements as the first,
        // with the long one moved from element 1 to element 0.
        const std::vector<std::string> first  = {"short", long_string(700, 'a')};
        const std::vector<std::string> second = {long_string(900, 'b'), "short"};

        {
            HDF5Backend backend;
            DataEntryContext ctx(URI);
            put_labels(backend, ctx, first, FORCE_CREATE_PULSE);
        }
        {
            HDF5Backend backend;
            DataEntryContext ctx(URI);
            get_labels(backend, ctx, first);
        }
        {
            HDF5Backend backend;
            DataEntryContext ctx(URI);
            put_labels(backend, ctx, second, OPEN_PULSE);   // the second ids_put()
        }
        {
            HDF5Backend backend;
            DataEntryContext ctx(URI);
            get_labels(backend, ctx, second);
        }

        fs::remove_all(URI.substr(URI.find('=') + 1));
        if (failures) {
            std::cout << failures << " failure(s).\n";
            return 1;
        }
        std::cout << "[OK] the second put owns the list it wrote.\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
