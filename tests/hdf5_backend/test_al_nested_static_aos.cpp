// tests/hdf5_backend/test_nested_static_aos.cpp
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

// Colors for debug
#define RESET   ""
#define BOLD    ""
#define GREEN   ""
#define RED     ""

const std::string URI = "imas:hdf5?path=./test_db_nested_static_aos";

int main() {
    try {
        std::cout << BOLD << "\n=== Test Nested Static AoS (source/ion/element) ===\n" << RESET;

        const int source_size = 2;
        const int ion_size = 2;
        const int element_size = 2;
        const int neutral_size = 2;

        // 1. Writing
        {
            std::cout << "\n--- Phase 1: Writing Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, FORCE_CREATE_PULSE);

            OperationContext opCtx(&dataEntryCtx, "core_sources", "", WRITE_OP);
            backend.beginAction(&opCtx);

            // Homogeneous time (required by some logic even if not used for static)
            int homogeneous_time = 1;
            backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

            // Static AoS 'source'
            ArraystructContext sourceCtx(&opCtx, "source", "");
            backend.beginArraystructAction(&sourceCtx, (int*)&source_size);

            for (int s = 0; s < source_size; ++s) {
                // Static AoS 'ion'
                ArraystructContext ionCtx(&sourceCtx, "ion", "");
                backend.beginArraystructAction(&ionCtx, (int*)&ion_size);

                for (int i = 0; i < ion_size; ++i) {
                    // Static AoS 'element'
                    ArraystructContext elementCtx(&ionCtx, "element", "");
                    backend.beginArraystructAction(&elementCtx, (int*)&element_size);

                    for (int e = 0; e < element_size; ++e) {
                        // Static 0D double 'a_stat'
                        double val = 100.0 + s * 100 + i * 10 + e;
                        backend.writeData(&elementCtx, "a_stat", "", &val, alconst::double_data, 0, nullptr);
                        
                        if (e < element_size - 1) elementCtx.nextIndex(1);
                    }
                    backend.endAction(&elementCtx);

                    if (i < ion_size - 1) ionCtx.nextIndex(1);
                }
                backend.endAction(&ionCtx);

                // Static AoS 'neutral'
                ArraystructContext neutralCtx(&sourceCtx, "neutral", "");
                backend.beginArraystructAction(&neutralCtx, (int*)&neutral_size);

                for (int n = 0; n < neutral_size; ++n) {
                    // Static 0D double 'a_stat' in neutral
                    // Using a different value formula to distinguish from ion/element/a_stat
                    double val = 500.0 + s * 100 + n;
                    backend.writeData(&neutralCtx, "a_stat", "", &val, alconst::double_data, 0, nullptr);
                    
                    if (n < neutral_size - 1) neutralCtx.nextIndex(1);
                }
                backend.endAction(&neutralCtx);

                if (s < source_size - 1) sourceCtx.nextIndex(1);
            }
            backend.endAction(&sourceCtx);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, FORCE_CREATE_PULSE);
            std::cout << GREEN << "[OK] Writing completed.\n" << RESET;
        }

        // 2. Reading / Validation
        {
            std::cout << "\n--- Phase 2: Reading Data ---\n";
            DataEntryContext dataEntryCtx(URI);
            HDF5Backend backend;
            backend.openPulse(&dataEntryCtx, OPEN_PULSE);

            OperationContext opCtx(&dataEntryCtx, "core_sources", "", READ_OP);
            backend.beginAction(&opCtx);

            // Read source
            ArraystructContext sourceCtx(&opCtx, "source", "");
            int s_size = 0;
            backend.beginArraystructAction(&sourceCtx, &s_size);
            assert(s_size == source_size);

            for (int s = 0; s < s_size; ++s) {
                // Read ion
                ArraystructContext ionCtx(&sourceCtx, "ion", "");
                int i_size = 0;
                backend.beginArraystructAction(&ionCtx, &i_size);
                assert(i_size == ion_size);

                for (int i = 0; i < i_size; ++i) {
                    // Read element
                    ArraystructContext elementCtx(&ionCtx, "element", "");
                    int e_size = 0;
                    backend.beginArraystructAction(&elementCtx, &e_size);
                    assert(e_size == element_size);

                    for (int e = 0; e < e_size; ++e) {
                        void* data = nullptr;
                        int type = alconst::double_data;
                        int dim = 0;
                        int size[H5S_MAX_RANK];

                        backend.readData(&elementCtx, "a_stat", "", &data, &type, &dim, size);
                        
                        assert(dim == 0);
                        double val = *(double*)data;
                        double expected = 100.0 + s * 100 + i * 10 + e;
                        
                        if (std::abs(val - expected) > 1e-9) {
                            std::cerr << RED << "Mismatch at s=" << s << ", i=" << i << ", e=" << e 
                                      << ": expected " << expected << ", got " << val << RESET << std::endl;
                            return 1;
                        }
                        free(data);

                        if (e < e_size - 1) elementCtx.nextIndex(1);
                    }
                    backend.endAction(&elementCtx);

                    if (i < i_size - 1) ionCtx.nextIndex(1);
                }
                backend.endAction(&ionCtx);

                // Read neutral
                ArraystructContext neutralCtx(&sourceCtx, "neutral", "");
                int n_size = 0;
                backend.beginArraystructAction(&neutralCtx, &n_size);
                assert(n_size == neutral_size);

                for (int n = 0; n < n_size; ++n) {
                    void* data = nullptr;
                    int type = alconst::double_data;
                    int dim = 0;
                    int size[H5S_MAX_RANK];

                    backend.readData(&neutralCtx, "a_stat", "", &data, &type, &dim, size);
                    
                    assert(dim == 0);
                    double val = *(double*)data;
                    double expected = 500.0 + s * 100 + n;
                    
                    if (std::abs(val - expected) > 1e-9) {
                        std::cerr << RED << "Mismatch for neutral a_stat at s=" << s << ", n=" << n 
                                  << ": expected " << expected << ", got " << val << RESET << std::endl;
                        return 1;
                    }
                    free(data);

                    if (n < n_size - 1) neutralCtx.nextIndex(1);
                }
                backend.endAction(&neutralCtx);

                if (s < s_size - 1) sourceCtx.nextIndex(1);
            }
            backend.endAction(&sourceCtx);

            backend.endAction(&opCtx);
            backend.closePulse(&dataEntryCtx, OPEN_PULSE);
            std::cout << GREEN << "[OK] Validation completed.\n" << RESET;
        }

    } catch (const std::exception& e) {
        std::cerr << RED << "Exception: " << e.what() << RESET << std::endl;
        return 1;
    }
    return 0;
}