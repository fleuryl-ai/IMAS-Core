// tests/hdf5_backend/test_gap_timerange_read.cpp
//
// Vérifie la lecture time-range (par plage temporelle) d'un signal dynamique
// présentant des "trous". Chaque slice demandée doit être résolue vers une
// slice EXISTANTE (closest / linear), jamais vers une valeur arbitraire.
#include "al_context.h"
#include "al_defs.h"
#include "al_const.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>

const std::string URI = "imas:hdf5?path=./test_db_gap_timerange";

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
        { double v = 300.0 + i * 10.0; backend.writeData(&ctxB, "sigC", "time", &v, alconst::double_data, 0, nullptr); }
        if (i < 4) ctxB.nextIndex(1);
    }

    backend.endAction(&ctxB);
    backend.endAction(&ctxA);
    backend.endAction(&opCtx);
    backend.closePulse(&dec, FORCE_CREATE_PULSE);
}

// Lit la plage [tmin,tmax] (closest) d'un signal scalaire de B, renvoie les
// valeurs par slice (dans l'ordre). Retourne 0 si la donnée n'est disponible
// nulle part, sinon la liste.
static std::vector<double> read_range(int interp,
                                      double tmin, double tmax, const char* sig,
                                      bool* available_any) {
    std::vector<double> vals;
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, OPEN_PULSE);
    std::vector<double> dtime; // pas de resampling
    OperationContext opCtx(&dec, "test_ids", READ_OP, alconst::timerange_op, tmin, tmax, dtime, interp);
    backend.beginAction(&opCtx);

    ArraystructContext ctxA(&opCtx, "A", "");
    int sizeA = 0;
    backend.beginArraystructAction(&ctxA, &sizeA);

    ArraystructContext ctxB(&ctxA, "B", "time");
    int sizeB = 0;
    backend.beginArraystructAction(&ctxB, &sizeB);
    if (available_any && sizeB <= 0) *available_any = false;

    for (int i = 0; i < sizeB; ++i) {
        void* data = nullptr;
        int type = alconst::double_data;
        int dim = 0;
        int size[1] = {0};
        int avail = backend.readData(&ctxB, sig, "time", &data, &type, &dim, size);
        if (avail == 0) { if (available_any) *available_any = false; }
        if (data) { vals.push_back(*(double*)data); free(data); }
        if (i + 1 < sizeB) ctxB.nextIndex(1);
    }

    backend.endAction(&ctxB);
    backend.endAction(&ctxA);
    backend.endAction(&opCtx);
    backend.closePulse(&dec, OPEN_PULSE);
    return vals;
}

static int nfail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; nfail++; } } while(0)

int main() {
    try {
        write_fixture();

        // ---- sigA: 100,110,(trou@0.2),130,140 ----
        // Plage [0.0,0.4] closest : 5 slices
        std::vector<double> a = read_range(alconst::closest_interp, 0.0, 0.4, "sigA", nullptr);
        CHECK(a.size() == 5, "range sigA size (got " + std::to_string(a.size()) + ")");
        if (a.size() == 5) {
            CHECK(std::abs(a[0] - 100.0) < 1e-9, "range sigA[0]");
            CHECK(std::abs(a[1] - 110.0) < 1e-9, "range sigA[1]");
            CHECK(std::abs(a[2] - 110.0) < 1e-9, "range sigA[2] (trou -> closest 110)");
            CHECK(std::abs(a[3] - 130.0) < 1e-9, "range sigA[3]");
            CHECK(std::abs(a[4] - 140.0) < 1e-9, "range sigA[4]");
        }

        // ---- sigC: présent partout ----
        std::vector<double> c = read_range(alconst::closest_interp, 0.0, 0.4, "sigC", nullptr);
        CHECK(c.size() == 5, "range sigC size");
        if (c.size() == 5) {
            for (int i = 0; i < 5; ++i)
                CHECK(std::abs(c[i] - (300.0 + i * 10.0)) < 1e-9, "range sigC[" + std::to_string(i) + "]");
        }

        // ---- Signal jamais écrit -> non disponible ----
        bool any = true;
        read_range(alconst::closest_interp, 0.0, 0.4, "sig_never", &any);
        CHECK(!any, "range sig_never should be unavailable");

        if (nfail == 0) { std::cout << "[OK] gap timerange read passed\n"; return 0; }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n"; return 1;
    }
    return 1;
}
