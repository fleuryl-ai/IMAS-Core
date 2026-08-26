// tests/hdf5_backend/test_gap_homog_timerange_read.cpp
//
// Vérifie la lecture time-range (sans resampling) d'un signal dynamique
// dans un IDS à base de temps HOMOGÈNE (ids_properties&homogeneous_time = 1)
// présentant des "trous". Chaque slice de la plage demandée doit résolue vers
// une slice EXISTANTE (closest / linear), jamais vers une valeur décalée.
//
// Miroir du cas inhomogène test_gap_timerange_read.cpp, mais sans AoS dynamique
// (signal standalone à la racine sur la timebase racine "time") : c'est le
// chemin CASE 2 d'écriture + la lecture homogène time-range qu'on valide.
//
// Valeurs : sigA = 100+i (trou i=2), sigC = 300+i (plet), time = 0.1*i.
#include "al_context.h"
#include "al_defs.h"
#include "al_const.h"
#include "hdf5_backend.h"
#include <iostream>
#include <vector>
#include <cmath>

const std::string URI = "imas:hdf5?path=./test_db_gap_homog_timerange";

static void write_fixture() {
    // Phase init : creation de l'IDS + marqueur homogeneous_time = 1
    {
        HDF5Backend backend;
        DataEntryContext dec(URI);
        backend.openPulse(&dec, FORCE_CREATE_PULSE);
        OperationContext opInit(&dec, "test_ids", "", WRITE_OP);
        backend.beginAction(&opInit);
        int homogeneous_time = 1; // HOMOGÈNE
        backend.writeData(&opInit, "ids_properties/homogeneous_time", "", &homogeneous_time, alconst::integer_data, 0, nullptr);
        backend.endAction(&opInit);
        backend.closePulse(&dec, FORCE_CREATE_PULSE);
    }

    // 5 etapes, chacune un append. La timebase racine "time" est construite par
    // ces ajouts (1 point/etape) : [0.0, 0.1, 0.2, 0.3, 0.4].
    int n = 5;
    for (int i = 0; i < n; ++i) {
        HDF5Backend backend;
        DataEntryContext dec(URI);
        backend.openPulse(&dec, OPEN_PULSE);
        double t = 0.1 * i;
        double t_undef = alconst::undefined_time;
        int interp_undef = alconst::undefined_interp;
        OperationContext opCtx(&dec, "test_ids", WRITE_OP, alconst::slice_op, t_undef, interp_undef);
        backend.beginAction(&opCtx);

        backend.writeData(&opCtx, "time", "time", &t, alconst::double_data, 0, nullptr);
        if (i != 2)         { double v = 100.0 + i; backend.writeData(&opCtx, "sigA", "time", &v, alconst::double_data, 0, nullptr); } // trou i=2
        { double v = 300.0 + i; backend.writeData(&opCtx, "sigC", "time", &v, alconst::double_data, 0, nullptr); }                        // plet

        backend.endAction(&opCtx);
        backend.closePulse(&dec, OPEN_PULSE);
    }
}

// Lit la plage [tmin,tmax] (sans resampling) d'un signal scalaire de la
// racine, renvoie les valeurs par slice (dans l'ordre).
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

    void* data = nullptr;
    int type = alconst::double_data;
    int dim = 0;
    int size[1] = {0};
    int avail = backend.readData(&opCtx, sig, "time", &data, &type, &dim, size);
    bool have = (avail != 0 && data != nullptr);
    if (available_any) *available_any = have;
    if (have && dim >= 1) {
        for (int i = 0; i < size[0]; ++i) vals.push_back(((double*)data)[i]);
        free(data);
    }

    backend.endAction(&opCtx);
    backend.closePulse(&dec, OPEN_PULSE);
    return vals;
}

static int nfail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; nfail++; } } while(0)

int main() {
    try {
        write_fixture();

        auto check_vec = [&](const std::vector<double>& got, const std::vector<double>& exp, const char* what) {
            if (got.size() != exp.size()) {
                CHECK(false, std::string(what) + ": size got " + std::to_string(got.size()) + " expected " + std::to_string(exp.size()));
                return;
            }
            for (size_t i = 0; i < got.size(); ++i)
                CHECK(std::abs(got[i] - exp[i]) < 1e-9, std::string(what) + "[" + std::to_string(i) + "]: got " + std::to_string(got[i]) + " expected " + std::to_string(exp[i]));
        };

        // ---- sigA: 100,101,(trou@.2),103,104 ----
        // closest : chaque idx vers la slice existante la plus proche
        {
            std::vector<double> a = read_range(alconst::closest_interp, 0.0, 0.4, "sigA", nullptr);
            check_vec(a, {100.0, 101.0, 101.0, 103.0, 104.0}, "range closest sigA");
        }
        // linear   : interp entre slices EXISTANTES (trou@.2 -> 101..103 mids)
        {
            std::vector<double> a = read_range(alconst::linear_interp, 0.0, 0.4, "sigA", nullptr);
            check_vec(a, {100.0, 101.0, 102.0, 103.0, 104.0}, "range linear sigA");
        }

        // ---- sigC: plet ----
        {
            std::vector<double> c = read_range(alconst::closest_interp, 0.0, 0.4, "sigC", nullptr);
            check_vec(c, {300.0, 301.0, 302.0, 303.0, 304.0}, "range closest sigC");
        }

        // ---- Signal jamais écrit -> non disponible ----
        {
            bool any = true;
            std::vector<double> nv = read_range(alconst::closest_interp, 0.0, 0.4, "sig_never", &any);
            CHECK(!any && nv.empty(), "range sig_never should be unavailable");
        }

        if (nfail == 0) { std::cout << "[OK] gap homog timerange read passed\n"; return 0; }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n"; return 1;
    }
    return 1;
}
