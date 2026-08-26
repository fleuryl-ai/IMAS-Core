// tests/hdf5_backend/test_gap_homog_slice_read.cpp
//
// Vérifie la lecture slice (closest / previous / linear) d'un signal dynamique
// dans un IDS à base de temps HOMOGÈNE (ids_properties&homogeneous_time = 1)
// présentant des "trous" (certaines slices absentes).
//
// Le cas inhomogène (homogeneous_time = 0) est déjà couvert par
// test_gap_{closest_prev,linear,timerange}_read.cpp. Ici, il n'y a PAS d'AoS
// dynamique : le signal est une donnée dynamique standalone à la racine,
// rattachée à la timebase racine "time", écrite étape par étape (APPEND).
// C'est donc le chemin CASE 2 de writeDataSlicesImpl + la lecture homogène
// qu'on valide.
//
// Valeurs (base + i) : sigA = 100+i, sigB = 200+i, sigC = 300+i, time = 0.1*i
//   - sigA : trou à i=2   -> présent (0,100),(0.1,101),(0.3,103),(0.4,104)
//   - sigB : trous i=1,3  -> présent (0,200),(0.2,202),(0.4,204)
//   - sigC : plet         -> présent (0,300)..(0.4,304)
// Quand la slice demandée est absente, la valeur retournée doit être celle
// d'une slice EXISTANTE ; l'interpolation linear ne doit jamais passer par
// une slice absente.
#include "al_context.h"
#include "al_defs.h"
#include "al_const.h"
#include "hdf5_backend.h"
#include <iostream>
#include <cmath>

const std::string URI = "imas:hdf5?path=./test_db_gap_homog_slice";

static void write_fixture() {
    // Phase init : création de l'IDS + marqueur homogeneous_time = 1
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

    // 5 étapes temporelles, chacune un append (slice_op).
    // La timebase racine "time" est construite par ces ajouts (1 point/étape).
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

        // timebase racine : 1 point par étape
        backend.writeData(&opCtx, "time", "time", &t, alconst::double_data, 0, nullptr);

        if (i != 2)                  { double v = 100.0 + i; backend.writeData(&opCtx, "sigA", "time", &v, alconst::double_data, 0, nullptr); } // trou i=2
        if (i != 1 && i != 3)        { double v = 200.0 + i; backend.writeData(&opCtx, "sigB", "time", &v, alconst::double_data, 0, nullptr); } // trous i=1,3
        { double v = 300.0 + i;      backend.writeData(&opCtx, "sigC", "time", &v, alconst::double_data, 0, nullptr); }                        // plet

        backend.endAction(&opCtx);
        backend.closePulse(&dec, OPEN_PULSE);
    }
}

static double read_scalar(int interp, double req_time,
                          const char* sig, bool* available) {
    HDF5Backend backend;
    DataEntryContext dec(URI);
    backend.openPulse(&dec, OPEN_PULSE);
    double t_undef = alconst::undefined_time;
    OperationContext opCtx(&dec, "test_ids", READ_OP, alconst::slice_op, req_time, interp);
    backend.beginAction(&opCtx);

    void* data = nullptr;
    int type = alconst::double_data;
    int dim = 0;
    int size[1] = {0};
    int avail = backend.readData(&opCtx, sig, "time", &data, &type, &dim, size);
    if (available) *available = (avail == 0 ? false : true);

    double val = 0.0;
    if (data) { val = *(double*)data; free(data); }

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

        // ---- CLOSEST (slice absente -> la plus proche existante, tie -> temps le plus bas) ----
        check(read_scalar(alconst::closest_interp, 0.2, "sigA", nullptr), 101.0, "closest sigA@0.2 (trou -> t=0.1)");
        check(read_scalar(alconst::closest_interp, 0.3, "sigA", nullptr), 103.0, "closest sigA@0.3 (present)");
        check(read_scalar(alconst::closest_interp, 0.1, "sigB", nullptr), 200.0, "closest sigB@0.1 (trou -> t=0)");
        check(read_scalar(alconst::closest_interp, 0.3, "sigB", nullptr), 202.0, "closest sigB@0.3 (trou -> t=0.2)");
        check(read_scalar(alconst::closest_interp, 0.2, "sigC", nullptr), 302.0, "closest sigC@0.2 (plet)");

        // ---- PREVIOUS (la dernière disponible <= t) ----
        check(read_scalar(alconst::previous_interp, 0.2, "sigA", nullptr), 101.0, "prev sigA@0.2 (trou -> t=0.1)");
        check(read_scalar(alconst::previous_interp, 0.3, "sigB", nullptr), 202.0, "prev sigB@0.3 (trou -> t=0.2)");
        check(read_scalar(alconst::previous_interp, 0.4, "sigB", nullptr), 204.0, "prev sigB@0.4 (present)");
        check(read_scalar(alconst::previous_interp, 0.4, "sigC", nullptr), 304.0, "prev sigC@0.4 (present)");

        // ---- LINEAR (interp. entre slices EXISTANTES uniquement) ----
        // sigA@0.2 (trou) : inf=t=0.1(101), sup=t=0.3(103) -> f=0.5 -> 102
        check(read_scalar(alconst::linear_interp, 0.2, "sigA", nullptr), 102.0, "linear sigA@0.2 (gap)");
        // sigB@0.1 (trou) : inf=t=0(200), sup=t=0.2(202) -> f=0.5 -> 201
        check(read_scalar(alconst::linear_interp, 0.1, "sigB", nullptr), 201.0, "linear sigB@0.1 (gap)");
        // sigB@0.3 (trou) : inf=t=0.2(202), sup=t=0.4(204) -> f=0.5 -> 203
        check(read_scalar(alconst::linear_interp, 0.3, "sigB", nullptr), 203.0, "linear sigB@0.3 (gap)");
        // sigC@0.2 plet : pas de trou -> 302
        check(read_scalar(alconst::linear_interp, 0.2, "sigC", nullptr), 302.0, "linear sigC@0.2 (present)");

        // ---- Signal jamais écrit -> non disponible ----
        bool avail = false;
        read_scalar(alconst::closest_interp, 0.2, "sig_never", &avail);
        CHECK(!avail, "closest sig_never should be unavailable");
        avail = false;
        read_scalar(alconst::linear_interp, 0.2, "sig_never", &avail);
        CHECK(!avail, "linear sig_never should be unavailable");

        if (nfail == 0) { std::cout << "[OK] gap homog slice read passed\n"; return 0; }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n"; return 1;
    }
    return 1;
}
