// @file  test_metadata_extractor_inttype.cpp
// @brief Regression: MetadataExtractor must return metadata for INT_* leaves,
//        matching FLT_/STR_/CPX_ handling. Covers the bug where
//        ids_properties/homogeneous_time (INT_0D) had all its @keys dropped.
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include "metadata_extractor.h"

namespace fs = std::filesystem;

static const char * XML_BODY = R"XML(
<?xml version="1.0"?>
<IDSs version="1.0" version_id="1" data_version="1">
  <IDS name="bench" version="1.0" version_id="1" data_version="1">
    <field name="ids_properties" data_type="structure" path="ids_properties"
           documentation="Container structure (should be excluded from extraction)"/>
    <field name="homogeneous_time" path="ids_properties/homogeneous_time"
           path_doc="ids_properties/homogeneous_time"
           data_type="INT_0D"
           type="constant"
           documentation="Homogeneous time flag"/>
    <field name="bench_signal" path="bench_signal"
           data_type="FLT_64"
           documentation="Bench float signal"
           units="eV"/>
    <field name="bench_str" path="bench_str"
           data_type="STR_0D"
           documentation="Bench string"
           type="constant"/>
  </IDS>
</IDSs>
)XML";

int main() {
    // Unique per-process scratch dir so parallel ctest runs don't clash.
    fs::path workdir = fs::temp_directory_path() / ("pz_meta_extract_" + std::to_string(getpid()));
    fs::remove_all(workdir);
    fs::create_directories(workdir);

    fs::path xml_file = workdir / "IDSDef.xml";
    {
        std::ofstream f(xml_file);
        f << XML_BODY;
    }

    MetadataExtractor ex(xml_file.string());
    auto md = ex.extract_metadata("bench");

    int failures = 0;

    auto expect = [&md, &failures](const std::string &key, const std::string &expected) {
        auto it = md.find(key);
        if (it == md.end()) {
            std::cerr << "[FAIL] missing expected key: " << key << "\n";
            ++failures;
            return;
        }
        if (it->second != expected) {
            std::cerr << "[FAIL] " << key << ": expected '" << expected << "' got '"
                      << it->second << "'\n";
            ++failures;
            return;
        }
        std::cout << "    [OK] " << key << " = '" << it->second << "'\n";
    };

    auto expectAbsent = [&md, &failures](const std::string &prefix) {
        for (const auto &kv : md) {
            if (kv.first.rfind(prefix, 0) == 0) {
                std::cerr << "[FAIL] should be absent but present: " << kv.first << "\n";
                ++failures;
                return;
            }
        }
        std::cout << "    [OK] no keys starting with '" << prefix << "'\n";
    };

    // --- INT_0D leaf — the case the user reported missing ---
    expect("ids_properties/homogeneous_time@data_type", "INT_0D");
    expect("ids_properties/homogeneous_time@type", "constant");
    expect("ids_properties/homogeneous_time@documentation", "Homogeneous time flag");
    expect("ids_properties/homogeneous_time@path_doc", "ids_properties/homogeneous_time");

    // --- FLT_64 leaf — regression: behaviour unchanged ---
    expect("bench_signal@data_type", "FLT_64");
    expect("bench_signal@units", "eV");
    expect("bench_signal@documentation", "Bench float signal");

    // --- STR_0D leaf — regression: behaviour unchanged ---
    expect("bench_str@data_type", "STR_0D");
    expect("bench_str@documentation", "Bench string");
    expect("bench_str@type", "constant");

    // --- Container node — must still be excluded (not a scalar leaf) ---
    expectAbsent("ids_properties@");   // no "ids_properties@..." entries for the structure container

    fs::remove_all(workdir);

    if (failures == 0) {
        std::cout << "\n[OK] Extractor INT_* regression test PASSED.\n";
        return 0;
    }
    std::cerr << "\n" << failures << " assertion(s) failed.\n";
    return 1;
}
