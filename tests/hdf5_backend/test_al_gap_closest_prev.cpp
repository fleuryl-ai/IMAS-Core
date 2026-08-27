// tests/hdf5_backend/test_gap_closest_prev_read.cpp
//
// Vérifie la lecture closest / previous d'un signal dynamique présentant des
// "trous" (certaines slices absentes). Quand la slice demandée est absente, la
// valeur retournée doit être celle d'une slice EXISTANTE (la plus proche / la
// dernière <= t), jamais d'une autre (cf. le bug du fallback sur slice 0).
#include "al_context.h"
#include "al_defs.h"
#include "al_const.h"
#include "hdf5_backend.h"
#include <iostream>
#include <cmath>

const std::string URI = "imas:hdf5?path=./test_db_gap_closest_prev";

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
        if (i != 2)              // sigA: trou à la slice 2
            { double v = 100.0 + i; backend.writeData(&ctxB, "sigA", "time", &v, alconst::double_data, 0, nullptr); }
        if (i != 1 && i != 3)    // sigB: trous aux slices 1 et 3
            { double v = 200.0 + i; backend.writeData(&ctxB, "sigB", "time", &v, alconst::double_data, 0, nullptr); }
        { double v = 300.0 + i; backend.writeData(&ctxB, "sigC", "time", &v, alconst::double_data, 0, nullptr); }
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
    double t_undef = alconst::undefined_time;
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
        auto check = [&](double got, double exp, const char* what) {
            CHECK(std::abs(got - exp) < 1e-9, std::string(what) + ": got " + std::to_string(got) + " expected " + std::to_string(exp));
        };

        // ---- CLOSEST ----
        // sigA trou à idx2 (t=0.2) : valeur la plus proche existante = slice 1 (101)
        check(read_scalar(alconst::closest_interp, 0.2, "sigA", nullptr), 101.0, "closest sigA@0.2");
        // sigC présent partout
        check(read_scalar(alconst::closest_interp, 0.2, "sigC", nullptr), 302.0, "closest sigC@0.2");
        // sigB trous à idx1 et idx3
        check(read_scalar(alconst::closest_interp, 0.1, "sigB", nullptr), 200.0, "closest sigB@0.1");
        check(read_scalar(alconst::closest_interp, 0.3, "sigB", nullptr), 202.0, "closest sigB@0.3");

        // ---- PREVIOUS (la dernière disponible <= t) ----
        check(read_scalar(alconst::previous_interp, 0.2, "sigA", nullptr), 101.0, "prev sigA@0.2");
        check(read_scalar(alconst::previous_interp, 0.3, "sigB", nullptr), 202.0, "prev sigB@0.3");
        check(read_scalar(alconst::previous_interp, 0.4, "sigB", nullptr), 204.0, "prev sigB@0.4 (present)");
        check(read_scalar(alconst::previous_interp, 0.1, "sigA", nullptr), 101.0, "prev sigA@0.1 (present)");

        // NOTE : `alconst::undefined_interp` (valeur 0) n'est pas autorisé en
        // slice_op par al_context.cpp ("Missing interpmode") : on ne le teste
        // donc pas ici. closest/previous suffisent à couvrir le cas "slice
        // absente -> valeur d'une slice existante la plus proche".

        // ---- Signal jamais écrit -> non disponible ----
        bool avail = false;
        read_scalar(alconst::closest_interp, 0.2, "sig_never", &avail);
        CHECK(!avail, "closest sig_never should be unavailable");
        avail = false;
        read_scalar(alconst::previous_interp, 0.2, "sig_never", &avail);
        CHECK(!avail, "previous sig_never should be unavailable");

        if (nfail == 0) { std::cout << "[OK] gap closest/previous read passed\n"; return 0; }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n"; return 1;
    }
    return 1;
}
