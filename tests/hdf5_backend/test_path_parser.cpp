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
    assert(segments[1].node_name == "node2");
    assert(segments[1].selection == SelectionType::NONE);
    assert(segments[2].node_name == "leaf");
    assert(segments[2].selection == SelectionType::NONE);
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
    assert(segments[1].node_name == "child");
    assert(segments[1].selection == SelectionType::NONE);
    std::cout << "PASSED" << std::endl;
}

void test_path_with_all_selection() {
    std::cout << "Running test: test_path_with_all_selection" << std::endl;
    PathParser parser("array[:]/data");
    const auto& segments = parser.segments();
    assert(segments.size() == 2);
    assert(segments[0].node_name == "array");
    assert(segments[0].selection == SelectionType::ALL);
    assert(segments[1].node_name == "data");
    assert(segments[1].selection == SelectionType::NONE);
    std::cout << "PASSED" << std::endl;
}

void test_complex_path() {
    std::cout << "Running test: test_complex_path" << std::endl;
    PathParser parser("A[0]/B[:]/C[42]/leaf");
    const auto& segments = parser.segments();
    assert(segments.size() == 4);
    assert(segments[0].node_name == "A");
    assert(segments[0].selection == SelectionType::INDEX);
    assert(segments[0].index == 0);
    assert(segments[1].node_name == "B");
    assert(segments[1].selection == SelectionType::ALL);
    assert(segments[2].node_name == "C");
    assert(segments[2].selection == SelectionType::INDEX);
    assert(segments[2].index == 42);
    assert(segments[3].node_name == "leaf");
    assert(segments[3].selection == SelectionType::NONE);
    std::cout << "PASSED" << std::endl;
}

void test_invalid_path_syntax() {
    std::cout << "Running test: test_invalid_path_syntax" << std::endl;
    try {
        PathParser parser("A[3]B/C");
        assert(false); // Doit échouer
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("Invalid path segment format") != std::string::npos);
    }
    std::cout << "PASSED" << std::endl;
}

void test_invalid_selection() {
    std::cout << "Running test: test_invalid_selection" << std::endl;
    try {
        PathParser parser("A[]/C");
        assert(false); // Doit échouer
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("Invalid selection format") != std::string::npos);
    }
    std::cout << "PASSED" << std::endl;
}

int main() {
    test_simple_path();
    test_path_with_index();
    test_path_with_all_selection();
    test_complex_path();
    test_invalid_path_syntax();
    test_invalid_selection();

    std::cout << "\nAll PathParser tests passed!" << std::endl;
    return 0;
}
