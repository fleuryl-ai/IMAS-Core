// @file  test_al_metadata_extract_inttype.cpp
// @brief End-to-end regression: with IMAS_PREFIX set, verify the full
//        AL -> HDF5Writer_v2 -> PanzerDB path materialises metadata for INT_*
//        leaves (the previously missing ids_properties/homogeneous_time@data_type
//        and @type), and that multi-level paths resolve correctly (the second bug:
//        "/" vs "&" namespace mismatch in the schema-path lookup).
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

static const char * XML_BODY = R"XML(
<?xml version="1.0"?>
<IDSs version="1.0" version_id="1" data_version="1">
  <IDS name="bench_int_meta" version="1.0" version_id="1" data_version="1">
    <field name="ids_properties" data_type="structure" path="ids_properties"
           documentation="Container structure (not a leaf, should be excluded)"/>
    <field name="homogeneous_time" path="ids_properties/homogeneous_time"
           path_doc="ids_properties/homogeneous_time"
           data_type="INT_0D"
           type="constant"
           documentation="Homogeneous time flag"/>
    <field name="time" path="time" data_type="FLT_1D" units="s"/>
    <field name="bench_signal" path="bench_signal"
           data_type="FLT_64"
           documentation="Bench float signal"
           units="eV"/>
  </IDS>
</IDSs>
)XML";

static std::string buildScratchPrefix() {
    const fs::path workdir = fs::temp_directory_path() /
                             ("pz_al_meta_" + std::to_string(getpid()));
    fs::remove_all(workdir);
    fs::create_directories(workdir / "include");
    {
        std::ofstream f(workdir / "include" / "IDSDef.xml");
        f << XML_BODY;
    }
    return workdir.string();
}

int main() {
    try {
        const std::string prefix = buildScratchPrefix();
        setenv("IMAS_PREFIX", prefix.c_str(), 1);

        std::cout << BOLD << "\n=== AL e2e: INT_* leaf metadata extraction ===\n"
                  << RESET;

        // --- cleanup any previous run ---
        fs::remove_all(URI.substr(URI.find("path=") + 5));

        // Phase 1: write IDS metadata-bearing nodes (exercises the AL extraction path).
        {
            std::cout << "\n--- Phase 1: Write (with IMAS_PREFIX set) ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, IDS_NAME, "", WRITE_OP);
            backend.beginAction(&opCtx);

            // INT_0D leaf — the reported case.
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

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Phase 1 (write) completed.\n" << RESET;
        }

        // Phase 2: read back the metadata leaves.
        {
            std::cout << "\n--- Phase 2: Read metadata leaves ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            OperationContext opCtx(&dataEntryCtx, IDS_NAME, "", READ_OP);
            backend.beginAction(&opCtx);

            auto check = [&backend, &opCtx](const std::string &path,
                                            const std::string &expected) {
                void *data = nullptr;
                int type = alconst::char_data;
                int dim = 0;
                int size[H5S_MAX_RANK];
                backend.readData(&opCtx, path, "", &data, &type, &dim, size);
                if (data == nullptr) {
                    std::cerr << RED << "[FAIL] " << path
                              << " not found (null)\n" << RESET;
                    return false;
                }
                std::string val(static_cast<const char *>(data));
                if (val != expected) {
                    std::cerr << RED << "[FAIL] " << path << ": expected '"
                              << expected << "' got '" << val << "'" << RESET << "\n";
                    free(data);
                    return false;
                }
                std::cout << "    [OK] " << path << " = '" << val << "'\n";
                free(data);
                return true;
            };

            bool ok = true;

            // INT_0D leaf — the case the user reported missing (data_type + type + doc)
            ok &= check("ids_properties/homogeneous_time@data_type", "INT_0D");
            ok &= check("ids_properties/homogeneous_time@type", "constant");
            ok &= check("ids_properties/homogeneous_time@documentation",
                        "Homogeneous time flag");
            ok &= check("ids_properties/homogeneous_time@path_doc",
                        "ids_properties/homogeneous_time");

            // Multi-level paths resolve (the second bug: "/" vs "&" lookup mismatch)
            ok &= check("ids_properties/homogeneous_time@data_type", "INT_0D");

            // Regression: FLT_64 leaf still works.
            ok &= check("bench_signal@data_type", "FLT_64");
            ok &= check("bench_signal@units", "eV");
            ok &= check("bench_signal@documentation", "Bench float signal");

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);

            if (!ok) {
                std::cerr << RED << "\nFAIL: one or more metadata leaves missing/wrong\n"
                          << RESET;
                fs::remove_all(prefix);
                return 1;
            }
            std::cout << GREEN << "\n[OK] Phase 2 (read) completed.\n" << RESET;
        }

        std::cout << BOLD << GREEN << "\nE2E INT_* leaf metadata test PASSED.\n" << RESET;
        fs::remove_all(prefix);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << "\n";
        return 1;
    }
}
