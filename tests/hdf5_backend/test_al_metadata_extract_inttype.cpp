// @file  test_al_metadata_extract_inttype.cpp
// @brief End-to-end regression: with IMAS_PREFIX set, verify the full
//        AL -> HDF5Writer_v2 -> PanzerDB path materialises metadata for INT_*
//        leaves (the previously missing ids_properties/homogeneous_time@data_type
//        and @type), that multi-level paths resolve, and — new — that metadata
//        values LONGER than the 512B fixed-width string slot (the schema's
//        749- and 1298-byte `documentation` strings) are swallowed via
//        STRING_CHUNKED slots and read back byte-for-byte (no ALBackendException).
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"

namespace fs = std::filesystem;

#define RESET ""
#define BOLD  ""
#define GREEN ""
#define RED   ""

const std::string IDS_NAME = "bench_int_meta";
const std::string URI      = "imas:hdf5?path=./test_db_metadata_extract_inttype";

// Exactly-n-byte doc: first 'A', last 'Z', middle 'x' -> a truncated read
// (missing a later chunk) is detectable. Chars are XML-safe.
static std::string makeDoc(size_t n) {
    std::string s(n, 'x');
    s.front() = 'A';
    s.back()  = 'Z';
    return s;
}

static std::string buildXML() {
    const std::string doc749  = makeDoc(749);
    const std::string doc1298 = makeDoc(1298);
    std::string x =
      "<?xml version=\"1.0\"?>\n"
      "<IDSs version=\"1.0\" version_id=\"1\" data_version=\"1\">\n"
      "  <IDS name=\"bench_int_meta\" version=\"1.0\" version_id=\"1\" data_version=\"1\">\n"
      "    <field name=\"ids_properties\" data_type=\"structure\" path=\"ids_properties\"\n"
      "           documentation=\"Container structure (not a leaf, should be excluded)\"/>\n"
      "    <field name=\"homogeneous_time\" path=\"ids_properties/homogeneous_time\"\n"
      "           path_doc=\"ids_properties/homogeneous_time\"\n"
      "           data_type=\"INT_0D\" type=\"constant\"\n"
      "           documentation=\"Homogeneous time flag\"/>\n"
      "    <field name=\"time\" path=\"time\" data_type=\"FLT_1D\" units=\"s\"/>\n"
      "    <field name=\"bench_signal\" path=\"bench_signal\"\n"
      "           data_type=\"FLT_64\" documentation=\"Bench float signal\" units=\"eV\"/>\n"
      "    <field name=\"long_doc_a\" path=\"long_doc_a\"\n"
      "           path_doc=\"long_doc_a\" data_type=\"INT_0D\" type=\"constant\"\n"
      "           documentation=\"" + doc749 + "\"/>\n"
      "    <field name=\"long_doc_b\" path=\"long_doc_b\"\n"
      "           path_doc=\"long_doc_b\" data_type=\"INT_0D\" type=\"constant\"\n"
      "           documentation=\"" + doc1298 + "\"/>\n"
      "  </IDS>\n"
      "</IDSs>\n";
    return x;
}

static std::string buildScratchPrefix() {
    const fs::path workdir = fs::temp_directory_path() /
                             ("pz_al_meta_" + std::to_string(getpid()));
    fs::remove_all(workdir);
    fs::create_directories(workdir / "include");
    {
        std::ofstream f(workdir / "include" / "IDSDef.xml");
        f << buildXML();
    }
    return workdir.string();
}

static void cleanupDb() {
    const size_t p = URI.find("path=");
    if (p != std::string::npos) fs::remove_all(URI.substr(p + 5));
}

int main() {
    try {
        const std::string prefix = buildScratchPrefix();
        setenv("IMAS_PREFIX", prefix.c_str(), 1);

        std::cout << BOLD << "\n=== AL e2e: INT_* metadata + long(>511B) docs ===\n"
                  << RESET;
        cleanupDb();

        // --- Phase 1: write (exercises the AL metadata / extraction path) ---
        {
            std::cout << "\n--- Phase 1: Write (IMAS_PREFIX set) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            OperationContext opCtx(&dataEntryCtx, IDS_NAME, "", WRITE_OP);
            backend.beginAction(&opCtx);

            // INT_0D leaf — the originally reported missing-metadatas case.
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "",
                              &homogeneous_time, alconst::integer_data, 0, nullptr);

            // FLT_1D time signal.
            int t_dim = 1;
            int t_size[] = {3};
            double tvals[] = {0.0, 0.1, 0.2};
            backend.writeData(&opCtx, "time", "time", tvals,
                              alconst::double_data, t_dim, t_size);

            // FLT_64 leaf.
            double v = 42.0;
            backend.writeData(&opCtx, "bench_signal", "",
                              &v, alconst::double_data, 0, nullptr);

            // Writing these INT_0D leaves triggers extraction of their LONG docs.
            int la = 0, lb = 0;
            backend.writeData(&opCtx, "long_doc_a", "", &la, alconst::integer_data, 0, nullptr);
            backend.writeData(&opCtx, "long_doc_b", "", &lb, alconst::integer_data, 0, nullptr);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Phase 1 (write) completed.\n" << RESET;
        }

        // --- Phase 2: read the metadata leaves back (incl. the long ones) ---
        {
            std::cout << "\n--- Phase 2: Read metadata leaves ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);
            OperationContext opCtx(&dataEntryCtx, IDS_NAME, "", READ_OP);
            backend.beginAction(&opCtx);

            auto readStr = [&backend, &opCtx](const std::string &path) -> std::string {
                void *data = nullptr;
                int type = alconst::char_data;
                int dim = 0;
                int size[H5S_MAX_RANK];
                backend.readData(&opCtx, path, "", &data, &type, &dim, size);
                if (data == nullptr || type != alconst::char_data) {
                    if (data) free(data);
                    return std::string();
                }
                std::string val(static_cast<const char *>(data));
                free(data);
                return val;
            };

            auto check = [&readStr](const std::string &path,
                                    const std::string &expected) {
                std::string got = readStr(path);
                if (got != expected) {
                    std::cerr << RED << "[FAIL] " << path << ": len got " << got.size()
                              << " expected " << expected.size() << "\n" << RESET;
                    std::exit(1);
                }
                std::cout << "    [OK] " << path << " (" << got.size() << " bytes)\n";
            };

            // INT_0D leaf — the originally reported missing metadata.
            check("ids_properties/homogeneous_time@data_type", "INT_0D");
            check("ids_properties/homogeneous_time@type", "constant");
            check("ids_properties/homogeneous_time@documentation", "Homogeneous time flag");
            check("ids_properties/homogeneous_time@path_doc", "ids_properties/homogeneous_time");

            // Regression: FLT_64 leaf still works.
            check("bench_signal@data_type", "FLT_64");
            check("bench_signal@units", "eV");
            check("bench_signal@documentation", "Bench float signal");

            // New: LONG documentation values (>511B) now stored via STRING_CHUNKED.
            check("long_doc_a@documentation", makeDoc(749));   // 2 slots
            check("long_doc_b@documentation", makeDoc(1298));  // 3 slots

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "\n[OK] Phase 2 (read) completed.\n" << RESET;
        }

        std::cout << BOLD << GREEN << "\nE2E INT_* metadata + long-doc test PASSED.\n" << RESET;
        fs::remove_all(prefix);
        cleanupDb();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << "\n";
        cleanupDb();
        return 1;
    }
}
