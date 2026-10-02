// @file  test_engine_write_read_list_spanned.cpp
// @brief Span-table regression test: list elements >STRING_MAX_LEN-1 span
//        multiple 512B slots (Option B), IMAS-compliant, SWMR-safe.
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <filesystem>

namespace fs = std::filesystem;

static std::string make_string(int len, char base = 'a') {
    std::string s(len, base);
    for (int i = 0; i < len; ++i) s[i] = (char)(base + (i % 26));
    return s;
}

template <int N>
static void write_and_read(const std::array<std::string, N>& items,
                           const std::string& filename)
{
    if (fs::exists(filename)) fs::remove(filename);

    const char* flat[N];
    for (int i = 0; i < N; ++i) flat[i] = items[i].c_str();

    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        db.beginArray("A", 1);
        db.beginArray("B", "time");
        db.writeDataSlices("sig", {N}, flat, 1, "time");
        db.endArray();
        db.endArray();
        db.close();
    }

    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        uint64_t ndim = 0, shape[6] = {};
        char* data = nullptr;
        int rc = db.readStringDataByIndex("A/0/B/sig", 0, &ndim, shape, &data);
        assert(rc == 0);
        assert(ndim == 2);
        assert(shape[0] == (uint64_t)N);   // N logical elements, not physical slots
        for (int i = 0; i < N; ++i) {
            std::string got(data + i * shape[1]);
            assert(got == items[i] && "element mismatch");
        }
        free(data);
        db.close();
    }

    if (fs::exists(filename)) fs::remove(filename);
    std::cout << "OK: list of " << N << " elements round-trip correct\n";
}

int main() {
    // Case 1: mixed — one element >511B triggers span table
    {
        std::array<std::string, 3> v = {
            "hello",
            make_string(749),   // >511 → 2 slots
            "world"
        };
        write_and_read<3>(v, "test_span_mixed.pane");
    }
    // Case 2: all-compact — backward compatible, no span entries written
    {
        std::array<std::string, 3> v = {"alpha", "beta", "gamma"};
        write_and_read<3>(v, "test_span_compact.pane");
    }
    // Case 3: all elements >511B
    {
        std::array<std::string, 2> v = {
            make_string(749, 'x'),
            make_string(1100, 'y')
        };
        write_and_read<2>(v, "test_span_all_long.pane");
    }

    std::cout << "All span-table tests passed.\n";
    return 0;
}
