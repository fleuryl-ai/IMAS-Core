// @file  test_engine_write_read_string_512.cpp
// @brief Boundary test for the fixed-width (512-byte) data_raw_str column.
//
//   * round-trips strings of 0, 1, 300 and 511 (= STRING_MAX_LEN-1) bytes,
//     a multi-byte UTF-8 string, and a list of mixed-length strings;
//   * verifies a 512-byte (= STRING_MAX_LEN, i.e. 511 max + 1) and a 1000-byte
//     string are REJECTED with an exception (fail-fast, no silent truncation).
#include "panzerdb.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

#include "fixtures/panzerdb_helper.h"

int main() {
    std::cout << "=== TEST PanzerDB: fixed-width (512B) string column — boundary ===\n\n";
    const std::string filename   = "test_string_512.panzer";
    const std::string ofilename  = "test_string_512_overflow.panzer";
    for (const auto& f : {filename, ofilename})
        if (fs::exists(f)) fs::remove(f);

    // ===================================================================
    // Part A: valid lengths round-trip
    // ===================================================================
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);

        std::string s_1    = std::string(1,   'a');
        std::string s_300  = std::string(300, 'b');
        std::string s_511  = std::string(511, 'c');            // == STRING_MAX_LEN-1
        std::string utf8   = std::string("caf\xC3\xA9");       // "café", 5 UTF-8 bytes
        const char* p1    = s_1.c_str();
        const char* p300  = s_300.c_str();
        const char* p511  = s_511.c_str();
        const char* putf  = utf8.c_str();
        const char* pempty = "";

        db.writeData("s_1",    {}, &p1,    1);                  // 1 byte
        db.writeData("s_300",  {}, &p300,  1);                  // 300 bytes
        db.writeData("s_511",  {}, &p511,  1);                  // 511 bytes (max)
        db.writeData("s_empty",{}, &pempty,1);                  // 0 bytes
        db.writeData("s_utf8", {}, &putf,  1);                  // multibyte UTF-8
        const char* lst[3] = {"x", "yyyy", "zzzzzzz"};          // list of 3
        db.writeData("s_list", {3}, lst, 3);

        db.flush();                                             // commit (should succeed)
        db.close();
        std::cout << "  [OK] wrote strings of 0/1/300/511 bytes + UTF-8 + list (flush OK)\n";
    }

    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        std::vector<PanzerDB::Leaf> leaves = db.getLeaves();
        std::string r;

        const PanzerDB::Leaf* l;
        l = find_leaf(leaves, "s_1");     assert(l && "missing s_1");
        db.readTensor(*l, &r);            assert(r == std::string(1, 'a'));

        l = find_leaf(leaves, "s_300");   assert(l && "missing s_300");
        db.readTensor(*l, &r);            assert(r == std::string(300, 'b'));

        l = find_leaf(leaves, "s_511");   assert(l && "missing s_511");
        db.readTensor(*l, &r);            assert(r == std::string(511, 'c'));

        l = find_leaf(leaves, "s_empty"); assert(l && "missing s_empty");
        db.readTensor(*l, &r);            assert(r.empty());

        l = find_leaf(leaves, "s_utf8");  assert(l && "missing s_utf8");
        db.readTensor(*l, &r);            assert(r == std::string("caf\xC3\xA9"));

        l = find_leaf(leaves, "s_list", 0); assert(l && "missing s_list");
        assert(l->count == 3);
        std::vector<std::string> v(3);
        db.readTensor(*l, v.data());
        assert(v[0] == "x" && v[1] == "yyyy" && v[2] == "zzzzzzz");

        db.close();
        std::cout << "  [OK] all valid strings round-tripped (incl. 511 byte max + UTF-8 + list)\n";
    }

    // ===================================================================
    // Part B: over-length strings are rejected (fail-fast, no truncation)
    // ===================================================================
    // Overflow is rejected at WRITE time (before the value enters the append
    // buffer), so the object stays consistent and close() does not re-throw.
    auto must_throw = [&](const std::string& name, const std::string& value) {
        bool threw = false;
        try {
            PanzerDB db(ofilename, PanzerDB::OpenMode::WRITE, true);
            const char* p = value.c_str();
            db.writeData(name, {}, &p, 1);   // throws here (write-time length check)
            db.flush();
        } catch (const std::exception&) {
            threw = true;
        }
        assert(threw && "over-length string must throw at writeData()");
    };

    must_throw("s_512",  std::string(512,  'd'));   // == STRING_MAX_LEN  (511 max + 1)
    must_throw("s_1000", std::string(1000, 'z'));   // well over the cap
    std::cout << "  [OK] 512-byte and 1000-byte strings rejected with an exception\n";

    for (const auto& f : {filename, ofilename}) if (fs::exists(f)) fs::remove(f);

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}
