// @file  test_engine_write_read_string_512.cpp
// @brief Fixed-width (512-byte) data_raw_str column: short strings still
//        round-trip; long SCALAR strings are now chunked across multiple
//        512B slots (STRING_CHUNKED) and read back concatenated (swallowed,
//        no variable-length type -> stays SWMR-safe). Long LIST elements are
//        still rejected (scalars-only chunking scope).
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
    std::cout << "=== TEST PanzerDB: fixed-width (512B) column + STRING_CHUNKED scalars ===\n\n";
    const std::string filename   = "test_string_512.panzer";
    const std::string ofilename  = "test_string_512_overflow.panzer";
    for (const auto& f : {filename, ofilename})
        if (fs::exists(f)) fs::remove(f);

    // ===================================================================
    // Part A: short + long scalars, UTF-8, and a string list
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

        // Long scalars: each now stored as k 512B slots (STRING_CHUNKED).
        std::string s_749  = std::string(749,  'd');           // the reported schema doc length
        std::string s_1000 = std::string(1000, 'e');           // spans 2 slots
        std::string s_1298 = std::string(1298, 'f');           // max schema doc length, spans 3 slots
        for (char& c : s_749)  c = 'd';  // ensure uniform payload
        const char* p749  = s_749.c_str();
        const char* p1000 = s_1000.c_str();
        const char* p1298 = s_1298.c_str();

        // Give the long values a distinguishable first/last byte so a
        // truncated read (only first chunk) is clearly detectable.
        s_749.replace(0, 1, "?"); s_749.replace(748, 1, "!");
        s_1000.replace(0, 1, "?"); s_1000.replace(999, 1, "!");
        s_1298.replace(0, 1, "?"); s_1298.replace(1297, 1, "!");

        db.writeData("s_1",    {}, &p1,    1);                  // 1 byte
        db.writeData("s_300",  {}, &p300,  1);                  // 300 bytes
        db.writeData("s_511",  {}, &p511,  1);                  // 511 bytes (max single slot)
        db.writeData("s_empty",{}, &pempty,1);                  // 0 bytes
        db.writeData("s_utf8", {}, &putf,  1);                  // multibyte UTF-8
        const char* pL749  = s_749.c_str();
        const char* pL1000 = s_1000.c_str();
        const char* pL1298 = s_1298.c_str();
        db.writeData("s_749",  {}, &pL749,  1);                 // long scalar (chunked)
        db.writeData("s_1000", {}, &pL1000, 1);                 // long scalar (chunked)
        db.writeData("s_1298", {}, &pL1298, 1);                 // long scalar (chunked)
        const char* lst[3] = {"x", "yyyy", "zzzzzzz"};          // list of 3 (short)
        db.writeData("s_list", {3}, lst, 3);

        db.flush();                                             // commit (should succeed)
        db.close();
        std::cout << "  [OK] wrote short + long(749/1000/1298) scalars + UTF-8 + list\n";
    }

    {
        PanzerDB db(filename, PanzerDB::OpenMode::READ);
        std::vector<PanzerDB::Leaf> leaves = db.getLeaves();

        // Read a scalar string, chunk-aware: a STRING_CHUNKED leaf is joined
        // across its `count` slots; a short one is a single slot.
        auto read_scalar = [&](const std::string& name) -> std::string {
            const PanzerDB::Leaf* l = find_leaf(leaves, name);
            assert(l && "leaf missing");
            if (static_cast<uint64_t>(l->flags >> 4) ==
                static_cast<uint64_t>(PanzerDB::DataType::STRING_CHUNKED)) {
                std::vector<std::string> parts(l->count);
                db.readTensor(*l, parts.data());
                std::string out;
                for (const auto& p : parts) out += p;
                return out;
            }
            std::string out;
            db.readTensor(*l, &out);
            return out;
        };

        const auto expect = [&](const std::string& name, const std::string& expected) {
            std::string got = read_scalar(name);
            if (got != expected) {
                std::cerr << "  [FAIL] " << name << ": len got " << got.size()
                          << " expected " << expected.size() << "\n";
                std::exit(1);
            }
            std::cout << "  [OK] " << name << " round-tripped (" << got.size() << " bytes)\n";
        };

        expect("s_1",    std::string(1,   'a'));
        expect("s_300",  std::string(300, 'b'));
        expect("s_511",  std::string(511, 'c'));
        expect("s_empty", std::string());
        expect("s_utf8",  std::string("caf\xC3\xA9"));

        // Long scalars — re-build the exact expected payloads.
        std::string e749  = std::string(749,  'd'); e749.replace(0,1,"?"); e749.replace(748,1,"!");
        std::string e1000 = std::string(1000, 'e'); e1000.replace(0,1,"?"); e1000.replace(999,1,"!");
        std::string e1298 = std::string(1298, 'f'); e1298.replace(0,1,"?"); e1298.replace(1297,1,"!");
        expect("s_749",  e749);
        expect("s_1000", e1000);
        expect("s_1298", e1298);

        // List of short strings — count == number of elements.
        const PanzerDB::Leaf* l = find_leaf(leaves, "s_list", 0);
        assert(l && "missing s_list");
        assert(l->count == 3);
        std::vector<std::string> v(3);
        db.readTensor(*l, v.data());
        assert(v[0] == "x" && v[1] == "yyyy" && v[2] == "zzzzzzz");

        // Confirm the long ones really used STRING_CHUNKED (count > 1), the short
        // ones stayed a single STRING slot (count == 1).
        assert(find_leaf(leaves, "s_749") ->count == 2);   // 511 + 238
        assert(find_leaf(leaves, "s_1000")->count == 2);   // 511 + 489
        assert(find_leaf(leaves, "s_1298")->count == 3);   // 511*2 + 276
        assert(find_leaf(leaves, "s_511") ->count == 1);
        db.close();
        std::cout << "  [OK] all short + long scalars + list round-tripped (chunked counts correct)\n";
    }

    // ===================================================================
    // Part B: long LIST elements are STILL rejected (scalars-only chunking)
    // ===================================================================
    auto list_member_throws = [&](const std::vector<std::string>& list) {
        bool threw = false;
        try {
            PanzerDB db(ofilename, PanzerDB::OpenMode::WRITE, true);
            // Build the char* array and write it.
            std::vector<std::string> owned = list;
            std::vector<const char*> ptrs;
            for (auto& s : owned) ptrs.push_back(s.c_str());
            db.writeData("s_longlist", { (size_t)ptrs.size() }, ptrs.data(), ptrs.size());
            db.flush();
        } catch (const std::exception&) {
            threw = true;
        }
        assert(threw && "long list element must still throw (scalars-only chunking)");
    };

    list_member_throws({ "short", std::string(600, 'q') });   // one element > 511B
    std::cout << "  [OK] long LIST element still rejected (scalars-only scope)\n";

    for (const auto& f : {filename, ofilename}) if (fs::exists(f)) fs::remove(f);

    std::cout << "\nALL TESTS PASSED!\n";
    return 0;
}
