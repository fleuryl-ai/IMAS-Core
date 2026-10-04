#include "aos_path_helpers.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

using std::string;
using std::vector;

static void check(const std::vector<std::string>& ctx_paths,
                  const std::string& in, const std::string& expected) {
    std::string got = sanitizeAosPath(ctx_paths, in);
    if (got != expected) {
        std::cerr << "FAIL: sanitizeAosPath(path=\"" << in << "\") -> \"" << got
                  << "\" (expected \"" << expected << "\")" << std::endl;
        assert(false);
    }
    std::cout << "PASSED: \"" << in << "\" -> \"" << got << "\"" << std::endl;
}

// Context chain of test_deep_chain: deepest level first (AL paths are cumulative)
static const std::vector<std::string> deep_chain = {"a/b/F", "a/b", "a"};

void test_special_cases() {
    std::cout << "Running test: test_special_cases" << std::endl;
    check(deep_chain, "", "");
    check(deep_chain, "/time", "time");
    check(deep_chain, "time", "time");
    check({}, "time", "time");
}

void test_relative_paths() {
    std::cout << "Running test: test_relative_paths" << std::endl;
    // Nothing to anchor: the levels stay '/'-separated (callers flatten them)
    check(deep_chain, "g/data", "g/data");
    check({}, "c/d/time", "c/d/time");
    check(deep_chain, "c/d/time", "c/d/time");
}

void test_absolute_through_context() {
    std::cout << "Running test: test_absolute_through_context" << std::endl;
    check(deep_chain, "a/b/F/g/data", "a&b&F/g&data");
    check(deep_chain, "a/b/F/time", "a&b&F/time");
    check(deep_chain, "/a/b/F/time", "/a&b&F/time");
    check(deep_chain, "a/b/F", "a&b&F");
}

void test_shallower_ancestor() {
    std::cout << "Running test: test_shallower_ancestor" << std::endl;
    // The path goes through an ancestor AoS, not through the deepest one
    check(deep_chain, "a/b/x/time", "a&b/x&time");
    check(deep_chain, "a/y", "a/y");
}

void test_repeated_segment_name() {
    std::cout << "Running test: test_repeated_segment_name" << std::endl;
    // Point 5 regression: the context path must be anchored at the head of the
    // path, a repeated name further down must not steal the match.
    check({"A/B", "A"}, "A/B/A/B/time", "A&B/A&B&time");
    check({"A/B/y/B/C", "A/B/y/B", "A/B/y", "A/B", "A"},
          "A/B/y/B/C/time", "A&B&y&B&C/time");
}

void test_no_partial_word_match() {
    std::cout << "Running test: test_no_partial_word_match" << std::endl;
    check({"a/b"}, "a/bc/time", "a/bc/time");
    check({"a/b"}, "xa/b/time", "xa/b/time");
    // Already flattened names are left untouched
    check({"equilibrium"}, "equilibrium&time", "equilibrium&time");
}

int main() {
    test_special_cases();
    test_relative_paths();
    test_absolute_through_context();
    test_shallower_ancestor();
    test_repeated_segment_name();
    test_no_partial_word_match();

    std::cout << "\nAll path sanitization tests passed!" << std::endl;
    return 0;
}
