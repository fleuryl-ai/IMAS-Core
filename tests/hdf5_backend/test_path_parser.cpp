#include "path_parser.h"
#include <iostream>
#include <cassert>

using namespace imas::direct_access;

void test_simple_path() {
    std::cout << "Running test: test_simple_path" << std::endl;
    PathParser parser("node1/node2/leaf");
    const auto& segments = parser.segments();
    assert(segments.size() == 3);
    assert(segments[0].node_name == "node1");
    assert(segments[0].selection == SelectionType::NONE);
    std::cout << "PASSED" << std::endl;
}

void test_path_with_index() {
    std::cout << "Running test: test_path_with_index" << std::endl;
    PathParser parser("parent[3]/child");
    const auto& segments = parser.segments();
    assert(segments.size() == 2);
    assert(segments[0].node_name == "parent");
    assert(segments[0].selection == SelectionType::INDEX);
    assert(segments[0].index == 3);
    std::cout << "PASSED" << std::endl;
}

void test_path_with_all_selection() {
    std::cout << "Running test: test_path_with_all_selection" << std::endl;
    PathParser parser("array[:]/data");
    const auto& segments = parser.segments();
    assert(segments.size() == 2);
    assert(segments[0].node_name == "array");
    assert(segments[0].selection == SelectionType::ALL);
    std::cout << "PASSED" << std::endl;
}

void test_path_with_full_slice() {
    std::cout << "Running test: test_path_with_full_slice" << std::endl;
    PathParser parser("time[10:20]");
    const auto& segments = parser.segments();
    assert(segments.size() == 1);
    assert(segments[0].node_name == "time");
    assert(segments[0].selection == SelectionType::SLICE);
    assert(segments[0].has_start && segments[0].start_index == 10);
    assert(segments[0].has_end && segments[0].end_index == 20);
    std::cout << "PASSED" << std::endl;
}

void test_path_with_open_start_slice() {
    std::cout << "Running test: test_path_with_open_start_slice" << std::endl;
    PathParser parser("signal[:50]");
    const auto& segments = parser.segments();
    assert(segments.size() == 1);
    assert(segments[0].node_name == "signal");
    assert(segments[0].selection == SelectionType::SLICE);
    assert(!segments[0].has_start);
    assert(segments[0].has_end && segments[0].end_index == 50);
    std::cout << "PASSED" << std::endl;
}

void test_path_with_open_end_slice() {
    std::cout << "Running test: test_path_with_open_end_slice" << std::endl;
    PathParser parser("data[100:]/value");
    const auto& segments = parser.segments();
    assert(segments.size() == 2);
    assert(segments[0].node_name == "data");
    assert(segments[0].selection == SelectionType::SLICE);
    assert(segments[0].has_start && segments[0].start_index == 100);
    assert(!segments[0].has_end);
    std::cout << "PASSED" << std::endl;
}

void test_complex_path_with_slices() {
    std::cout << "Running test: test_complex_path_with_slices" << std::endl;
    PathParser parser("A[0]/B[10:20]/C[:]/D[5:]");
    const auto& segments = parser.segments();
    assert(segments.size() == 4);
    assert(segments[0].selection == SelectionType::INDEX && segments[0].index == 0);
    assert(segments[1].selection == SelectionType::SLICE && segments[1].start_index == 10 && segments[1].end_index == 20);
    assert(segments[2].selection == SelectionType::ALL);
    assert(segments[3].selection == SelectionType::SLICE && segments[3].start_index == 5 && !segments[3].has_end);
    std::cout << "PASSED" << std::endl;
}


void test_invalid_path_syntax() {
    std::cout << "Running test: test_invalid_path_syntax" << std::endl;
    try {
        PathParser parser("A[3]B/C");
        assert(false);
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("Invalid path segment format") != std::string::npos);
    }
    std::cout << "PASSED" << std::endl;
}

void test_invalid_slice() {
    std::cout << "Running test: test_invalid_slice" << std::endl;
    try {
        PathParser parser("A[10:a]");
        assert(false);
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("Invalid slice format") != std::string::npos);
    }
    std::cout << "PASSED" << std::endl;
}


int main() {
    test_simple_path();
    test_path_with_index();
    test_path_with_all_selection();
    test_path_with_full_slice();
    test_path_with_open_start_slice();
    test_path_with_open_end_slice();
    test_complex_path_with_slices();
    test_invalid_path_syntax();
    test_invalid_slice();

    std::cout << "\nAll PathParser tests passed!" << std::endl;
    return 0;
}
