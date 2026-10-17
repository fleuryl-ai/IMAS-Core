/**
 * @file path_parser.h
 * @brief Grammar of the direct-access path syntax: an IMAS path whose segments may
 *        carry a bracketed selection (index, slice, or time window).
 *
 * Examples: "profiles_1d[:]/ion/z_ion", "core_profiles[time=2.7,interp=linear]/t_e",
 * "equilibrium/1/time_slice[0:50]/psi". The parser is purely syntactic: it turns the
 * string into PathSegment values and never touches storage. The selection kinds map
 * onto the PanzerDB read API (time index lookup, slice range, interpolated read).
 */
#ifndef PATH_PARSER_H
#define PATH_PARSER_H

#include <string>
#include <vector>
#include <stdexcept>

namespace imas {
namespace direct_access {

/**
 * @enum SelectionType
 * @brief Kind of restriction a bracketed selection applies to its node.
 *
 * One enum covers AoS instance selection and time selection; the fields of
 * PathSegment that are meaningful depend on which value is set.
 */
enum class SelectionType {
    NONE,   ///< No bracket: the whole node (all instances / all time steps).
    ALL,    ///< Explicit "[:]" — same as NONE, but stated by the caller.
    INDEX,  ///< "[3]" — one AoS instance index.
    SLICE,  ///< "[a:b]" (either bound optional) — a range of indices/time steps.
    TIME    ///< "[time=...]" — a time selection; one type covers a single time and a [t0:t1] window.
};

/**
 * @enum InterpolationMethod
 * @brief How to resolve a time selection that falls between stored time steps.
 */
enum class InterpolationMethod {
    NONE,    ///< No interpolation asked (e.g. a range selection, [time=1:2]).
    CLOSEST, ///< Take the nearest stored slice — the default for a single time ([time=2.7]).
    LINEAR   ///< Interpolate between the two bracketing slices ([time=2.7,interp=linear]).
};

/**
 * @struct PathSegment
 * @brief One '/'-separated node of a parsed path, with its optional selection.
 *
 * Which fields carry a value follows `selection`: INDEX uses `index`, SLICE uses
 * the `*_index`/`has_*` pair, TIME uses the `*_time`/`has_*_time` pair plus `interp`.
 * The unused fields keep their default and must not be read.
 */
struct PathSegment {
    std::string node_name;                        ///< Node name, without its bracket (may be empty for a leading selection).
    SelectionType selection = SelectionType::NONE;
    InterpolationMethod interp = InterpolationMethod::NONE;

    // Valid when selection == INDEX
    size_t index = 0;                             ///< The single requested instance index.

    // Valid when selection == SLICE
    size_t start_index = 0;                       ///< First index of the range (inclusive).
    size_t end_index = 0;                         ///< Last index of the range (exclusive, as used by the readers).
    bool has_start = false;                       ///< False -> the range starts at the beginning.
    bool has_end = false;                         ///< False -> the range runs to the end of the axis.

    // Valid when selection == TIME
    double start_time = 0.0;                      ///< Lower time bound.
    double end_time = 0.0;                        ///< Upper time bound.
    bool has_start_time = false;                  ///< False -> unbounded below ([time=:5]).
    bool has_end_time = false;                    ///< False -> unbounded above, and a lone [time=2.7] is a single instant.
};

/**
 * @class PathParser
 * @brief Splits a path into PathSegment values (see the grammar in path_parser.cpp).
 *
 * Node names accept word characters, '&', '@' and '-'; '@' is kept so metadata keys
 * ("t_e@units") stay one segment. Each bracket body is split on ',' and each token
 * classified: "time=..." -> TIME, "interp=linear|closest" -> interpolation,
 * ":" -> ALL, "a:b" -> SLICE, anything else -> INDEX (decimal; a non-numeric token
 * aborts with std::invalid_argument from std::stoul). A single [time=...] with no
 * upper bound and no explicit interp gets CLOSEST.
 */
class PathParser {
public:
    /**
     * @brief Parses `path` eagerly.
     * @param path The path to parse.
     * @throws std::runtime_error when a segment does not match the segment grammar.
     */
    explicit PathParser(const std::string& path);
    /**
     * @brief The parsed segments, in path order.
     * @return Reference to the internal segment vector (valid for this parser's lifetime).
     */
    const std::vector<PathSegment>& segments() const;
private:
    /**
     * @brief Runs the regex grammar over raw_path_, filling segments_.
     * @throws std::runtime_error on the first segment that cannot be matched.
     */
    void parse();
    std::string raw_path_;            ///< Unparsed input, kept for diagnostics.
    std::vector<PathSegment> segments_;   ///< Result of parse().
};

/**
 * @brief One-shot convenience wrapper around PathParser.
 * @param path The path to parse.
 * @return The segments, in path order.
 * @throws std::runtime_error when a segment does not match the grammar.
 */
inline std::vector<PathSegment> parse_path(const std::string& path) {
    PathParser parser(path);
    return parser.segments();
}

} // namespace direct_access
} // namespace imas

#endif // PATH_PARSER_H
