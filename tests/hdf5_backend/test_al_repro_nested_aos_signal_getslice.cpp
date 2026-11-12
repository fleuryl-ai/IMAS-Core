// @file  test_al_repro_nested_aos_signal_getslice.cpp
// @brief A 1-D signal under a STATIC AoS `A` -> DYNAMIC (timed) AoS `B`,
//        inhomogeneous (ids_properties.homogeneous_time = 0). Base stored at
//        A/i/B/j/time. put() then get_slice of the signal.
#include "al_context.h"
#include "al_defs.h"
#include "hdf5_backend.h"
#include <hdf5.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

const std::string URI = "imas:hdf5?path=./test_db_repro_nested_aos_signal_getslice";

static const int A_SIZE  = 2;
static const int SPATIAL = 4;
static double base_time(double i, double j){ return 0.25*i + 0.5*j; }
static double sig_val(int i, int j, int x){ return 100.0*i + 10.0*j + x; }

static void write_fixture() {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, FORCE_CREATE_PULSE);
    OperationContext op(&dec, "magnetics", "", WRITE_OP);
    backend.beginAction(&op);
    int homogeneous_time = 0;
    backend.writeData(&op, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);
    ArraystructContext aCtx(&op, "A", "");
    int a_size = A_SIZE;
    backend.beginArraystructAction(&aCtx, &a_size);
    for (int i = 0; i < A_SIZE; ++i) {
        ArraystructContext bCtx(&aCtx, "B", "time");
        int b_size = -1;
        backend.beginArraystructAction(&bCtx, &b_size);   // dynamic size = SPATIAL samples
        b_size = SPATIAL;
        for (int j = 0; j < SPATIAL; ++j) {
            double tv = base_time(i, j);
            backend.writeData(&bCtx, "time", "time", &tv, alconst::double_data, 0, nullptr);
            std::vector<double> sig(1 /*scalar payload per j*/);
            sig[0] = sig_val(i, j, 0);
            backend.writeData(&bCtx, "sig", "time", sig.data(), alconst::double_data, 0, nullptr);
            if (j < SPATIAL - 1) bCtx.nextIndex(1);
        }
        backend.endAction(&bCtx);
        if (i < A_SIZE - 1) aCtx.nextIndex(1);
    }
    backend.endAction(&aCtx);
    backend.endAction(&op);
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
}

int main(){
    try {
        write_fixture();
        std::cout << "[fixture written; now reading a slice]\n";
        std::string result;
        try {
            HDF5Backend backend;
            DataEntryContext dec(URI);
            OperationContext op(&dec, "magnetics", READ_OP, alconst::slice_op, 0.5, alconst::closest_interp);
            backend.openPulse(&dec, OPEN_PULSE);
            backend.beginAction(&op);
            ArraystructContext aCtx(&op, "A", "");
            int an=0; backend.beginArraystructAction(&aCtx, &an);
            ArraystructContext bCtx(&aCtx, "B", "time");
            int bn=0; backend.beginArraystructAction(&bCtx, &bn);
            void* data=nullptr; int type=0, dim=0; int sz[H5S_MAX_RANK]={0};
            backend.readData(&bCtx, "sig", "time", &data, &type, &dim, sz);
            result = "OK — readData did NOT throw (no bug reproduced here)";
        } catch (const std::exception& e) {
            result = std::string("THREW: ") + e.what();
        }
        std::cout << result << "\n";
        if (result.find("Unexpected time vector with size 0") != std::string::npos)
            std::cout << ">>> BUG REPRODUCED (Unexpected time vector with size 0)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fixture/outer error: " << e.what() << "\n";
        return 1;
    }
}
