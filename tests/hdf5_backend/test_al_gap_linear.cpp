// tests/hdf5_backend/test_gap_linear_read.cpp
//
// Vérifie la lecture linear d'un signal dynamique présentant des "trous".
// Quand la slice demandée est absente, l'interpolation doit se faire
// entre les slices EXISTANTES les plus proches (la dernière <= t et la
// première >= t), jamais avec une slice arbitraire.
#include "al_context.h"
#include "al_defs.h"
#include "al_const.h"
#include "hdf5_backend.h"
#include <iostream>
#include <cmath>

const std::string URI = "imas:hdf5?path=./test_db_gap_linear";

static void write_fixture() {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, FORCE_CREATE_PULSE);
    OperationContext opCtx(&dec, "test_ids", "", WRITE_OP);
    backend.beginAction(&opCtx);

    int homogeneous_time = 0;
    backend.writeData(&opCtx, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);

    ArraystructContext ctxA(&opCtx, "A", "");
    int sizeA = 1;
    backend.beginArraystructAction(&ctxA, &sizeA);

    ArraystructContext ctxB(&ctxA, "B", "time");
    int sizeB = 5;
    backend.beginArraystructAction(&ctxB, &sizeB);

    for (int i = 0; i < 5; ++i) {
        double t = 0.1 * i;
        backend.writeData(&ctxB, "time", "", &t, alconst::double_data, 0, nullptr);
        if (i != 2)              // sigA: trou à la slice 2 (t=0.2)
            { double v = 100.0 + i * 10.0; backend.writeData(&ctxB, "sigA", "time", &v, alconst::double_data, 0, nullptr); }
        if (i != 1)              // sigB: trou à la slice 1
            { double v = 200.0 + i * 10.0; backend.writeData(&ctxB, "sigB", "time", &v, alconst::double_data, 0, nullptr); }
        { double v = 300.0 + i * 10.0; backend.writeData(&ctxB, "sigC", "time", &v, alconst::double_data, 0, nullptr); }
        if (i < 4) ctxB.nextIndex(1);
    }

    backend.endAction(&ctxB);
    backend.endAction(&ctxA);
    backend.endAction(&opCtx);
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
}

static double read_scalar(int interp, double req_time,
                          const char* sig, bool* available) {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, OPEN_PULSE);
    OperationContext opCtx(&dec, "test_ids", READ_OP, alconst::slice_op, req_time, interp);
    backend.beginAction(&opCtx);

    ArraystructContext ctxA(&opCtx, "A", "");
    int sizeA = 0;
    backend.beginArraystructAction(&ctxA, &sizeA);

    ArraystructContext ctxB(&ctxA, "B", "time");
    int sizeB = 0;
    backend.beginArraystructAction(&ctxB, &sizeB);

    void* data = nullptr;
    int type = alconst::double_data;
    int dim = 0;
    int size[1] = {0};
    int avail = backend.readData(&ctxB, sig, "time", &data, &type, &dim, size);
    if (available) *available = (avail == 0 ? false : true);

    double val = 0.0;
    if (data) { val = *(double*)data; free(data); }

    backend.endAction(&ctxB);
    backend.endAction(&ctxA);
    backend.endAction(&opCtx);
    backend.closePulse(&dec, OPEN_PULSE);
    return val;
}

static int nfail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; nfail++; } } while(0)

int main() {
    try {
        write_fixture();
        auto lin = [&](double t, const char* sig) { return read_scalar(alconst::linear_interp, t, sig, nullptr); };
        auto check = [&](double got, double exp, const char* what) {
            CHECK(std::abs(got - exp) < 1e-6, std::string(what) + ": got " + std::to_string(got) + " expected " + std::to_string(exp));
        };

        // ---- sigA: 100,110,(trou),130,140 (valeurs 100+i*10, trou à i=2) ----
        // t=0.05 -> entre 0 (100) et 0.1 (110) -> 105
        check(lin(0.05, "sigA"), 105.0, "linear sigA@0.05");
        // t=0.2 (TRU) -> entre 0.1 (110) et 0.3 (130) -> 120
        check(lin(0.2, "sigA"),  120.0, "linear sigA@0.2 (gap)");
        // t=0.25 -> entre 0.1 (110) et 0.3 (130) -> factor (0.25-0.1)/(0.3-0.1)=0.75 -> 110+0.75*20=125
        check(lin(0.25, "sigA"), 125.0, "linear sigA@0.25");
        // t=0.35-> entre 0.3 (130) et 0.4 (140) -> 135
        check(lin(0.35, "sigA"), 135.0, "linear sigA@0.35");
        // t=0.4 -> 140 (dernier, pas d'interp)
        check(lin(0.4, "sigA"),  140.0, "linear sigA@0.4");
        // t=0.0 -> 100 (premier)
        check(lin(0.0, "sigA"),  100.0, "linear sigA@0.0");

        // ---- sigB: 200,(trou@t=0.1),220,230,240 ----
        // Interpolation par temps entre slices existantes 0 (200) et 0.2 (220) :
        // t=0.05 -> factor 0.05/0.2=0.25 -> 200+0.25*20=205
        check(lin(0.05, "sigB"), 205.0, "linear sigB@0.05");
        // t=0.1 (TRU) -> factor 0.1/0.2=0.5 -> 200+0.5*20=210
        check(lin(0.1, "sigB"),  210.0, "linear sigB@0.1 (gap)");
        // t=0.15 -> factor 0.15/0.2=0.75 -> 200+0.75*20=215
        check(lin(0.15, "sigB"), 215.0, "linear sigB@0.15");
        // t=0.2 -> 220 (présente)
        check(lin(0.2, "sigB"),  220.0, "linear sigB@0.2");
        // t=0.3 -> 230 (présente)
        check(lin(0.3, "sigB"),  230.0, "linear sigB@0.3");

        // ---- sigC: présent partout ----
        check(lin(0.2, "sigC"),  320.0, "linear sigC@0.2");
        check(lin(0.15, "sigC"), 315.0, "linear sigC@0.15");

        // ---- Signal jamais écrit -> non disponible ----
        bool avail = false;
        read_scalar(alconst::linear_interp, 0.2, "sig_never", &avail);
        CHECK(!avail, "linear sig_never should be unavailable");

        if (nfail == 0) { std::cout << "[OK] gap linear read passed\n"; return 0; }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n"; return 1;
    }
    return 1;
}
