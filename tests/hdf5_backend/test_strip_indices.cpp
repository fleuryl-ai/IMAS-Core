// test_strip_indices.cpp
// Pins the contract of PanzerDB::stripIndices, the AoS-index stripper used twice
// per node on the read path (schema lookup) and by the view publisher.
//
// It exists because the function was rewritten from a stringstream/getline loop
// into a single pass: the rewrite must be byte-for-byte equivalent, including the
// edge cases the stream loop produced by accident (empty segments dropped, so
// leading/duplicate slashes collapse; a path made only of indices becomes empty).

#include "panzerdb.h"

#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(const std::string& in, const std::string& expected) {
    const std::string got = PanzerDB::stripIndices(in);
    if (got != expected) {
        std::cout << "  FAIL: stripIndices(\"" << in << "\") = \"" << got
                  << "\", expected \"" << expected << "\"\n";
        ++failures;
    }
}

} // namespace

int main() {
    std::cout << "=== TEST PanzerDB::stripIndices ===\n";

    // Plain shapes.
    check("", "");
    check("signal", "signal");
    check("time_slice/3/profiles_1d/signal", "time_slice/profiles_1d/signal");
    check("core_profiles_3/0/12/x", "core_profiles_3/x");
    check("channel/0/n_e&data", "channel/n_e&data");

    // Multi-digit and zero-padded indices.
    check("time_slice/8149/signal", "time_slice/signal");
    check("a/007/b", "a/b");

    // A segment that only looks numeric-ish stays.
    check("a/1a/b", "a/1a/b");
    check("a/1_2/b", "a/1_2/b");
    check("2000Hz", "2000Hz");
    check("a/+/b", "a/+/b");
    check("a/-1/b", "a/-1/b");

    // Empty segments collapse (that is what the stream loop did: getline skips them).
    check("/", "");
    check("a//b", "a/b");
    check("/a/1/", "a");
    check("a/1/", "a");

    // Everything is an index -> empty result.
    check("0", "");
    check("0/1/2", "");
    check("//", "");

    // Non-ASCII bytes are not digits (the old loop relied on ::isdigit, which is
    // undefined for plain char here; ASCII-only is what the format can hold anyway).
    check("a/\xc3\xa9/b", "a/\xc3\xa9/b");

    if (failures == 0) {
        std::cout << "ALL PASSED\n";
        return 0;
    }
    std::cout << failures << " FAILURE(S)\n";
    return 1;
}
