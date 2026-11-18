// @file  test_index_lifetime_guard.cpp
// @brief ReadIndex / PanzerDB leaf-cache lifetime contract (to_improve.md point 6):
//        a shared ReadIndex must never expose Leaf*/string_view that the engine has
//        already destroyed. The generation counter makes the move visible, and
//        ensure_current() re-synchronises the index instead of dereferencing stale
//        leaves.
#include "panzerdb.h"
#include "iread_strategy.h"
#include <iostream>
#include <cassert>
#include <filesystem>
#include <memory>
#include <vector>

namespace fs = std::filesystem;

static size_t count_leaves(const std::vector<const PanzerDB::Leaf*>* found) {
    return found ? found->size() : 0;
}

int main() {
    std::cout << "=== TEST: ReadIndex stays in sync with the PanzerDB leaf cache ===\n\n";
    const std::string filename = "test_index_lifetime_guard.panzer";
    if (fs::exists(filename)) fs::remove(filename);

    // ------------------------------------------------------------------
    // 1. One time slice: A(static)/B(dynamic, "time") + signal at t=0
    // ------------------------------------------------------------------
    {
        PanzerDB db(filename, PanzerDB::OpenMode::WRITE, true);
        db.beginArray("A", 1);
        db.beginArray("B", "time");
        double t0 = 0.0, s0 = 100.0;
        db.writeDataSlices("time", {1}, &t0, 1, "");
        db.writeDataSlices("signal", {1}, &s0, 1, "time");
        db.endArray();
        db.endArray();
        db.close();
    }

    // ------------------------------------------------------------------
    // 2. An index is built over an APPEND engine (the SWMR-like situation:
    //    the engine keeps moving while readers hold leaves)
    // ------------------------------------------------------------------
    auto db = std::make_shared<PanzerDB>(filename, PanzerDB::OpenMode::APPEND);
    ReadIndex index(db);
    assert(index.is_current());
    const uint64_t gen0 = db->leaf_cache_generation();

    assert(count_leaves(index.find_path("A/0/B/signal")) == 1);
    assert(count_leaves(index.find_name("signal")) == 1);
    assert(index.find_path("nope/not/there") == nullptr);
    std::cout << "Index built at generation " << gen0
              << ", one leaf for A/0/B/signal.\n";

    // The Leaf* handed out must stay usable while the cache does not move.
    const PanzerDB::Leaf* leaf0 = index.find_name("signal")->at(0);
    assert(leaf0->path == "A/0/B/signal");

    // ------------------------------------------------------------------
    // 3. A new slice is written and committed: flush() only INVALIDATES the
    //    leaf cache, it does not destroy it -> nothing dangles, and the
    //    index is still considered current.
    // ------------------------------------------------------------------
    {
        db->beginArray("A", 1);
        db->beginArray("B", "time");
        double t1 = 0.1, s1 = 200.0;
        db->writeDataSlices("time", {1}, &t1, 1, "");
        db->writeDataSlices("signal", {1}, &s1, 1, "time");
        db->endArray();
        db->endArray();
        db->flush();
    }
    assert(db->leaf_cache_generation() == gen0);
    assert(index.is_current());
    assert(leaf0->path == "A/0/B/signal");          // still a live object
    assert(count_leaves(index.find_name("signal")) == 1);  // stale, not dangling
    std::cout << "flush() alone: generation unchanged, old leaves still valid.\n";

    // ------------------------------------------------------------------
    // 4. Re-reading the index destroys the old leaves: the generation moves,
    //    is_current() goes false and ensure_current() rebuilds the maps.
    // ------------------------------------------------------------------
    db->invalidateLeafCache();
    db->getLeaves();                                // rebuild -> leaves die here
    const uint64_t gen2 = db->leaf_cache_generation();
    assert(gen2 > gen0);
    assert(!index.is_current());
    assert(index.ensure_current());                 // rebuilt
    assert(!index.ensure_current());                // idempotent
    assert(index.is_current());
    assert(count_leaves(index.find_name("signal")) == 2);
    assert(count_leaves(index.find_path("A/0/B/signal")) == 2);
    std::cout << "After the rebuild: generation " << gen0 << " -> " << gen2
              << ", the index now sees both slices.\n";

    // The refreshed leaves are the NEW storage, and they read correctly.
    double out[64] = {0};
    for (const auto* leaf : *index.find_name("signal")) {
        db->readTensor(*leaf, out);
    }

    db->close();
    fs::remove(filename);
    std::cout << "\nAll leaf-lifetime guard tests passed!\n";
    return 0;
}
