/**
 * @file direct_reader.h
 * @brief Entry point of the direct-access read path: a parsed path in, a
 *        TensorView out — bypassing the AL contexts entirely and reading
 *        PanzerDB leaves straight.
 *
 * This is the read side of imas::direct_access (see direct_access_api.h for the
 * public types, path_parser.h for the path grammar, and imas_h5read/ for the
 * MATLAB/Fortran front ends built on top of it).
 */
#ifndef DIRECT_READER_H
#define DIRECT_READER_H

#include "direct_access_api.h"
#include "path_parser.h" // PathSegment: the read() input type
#include <string>
#include <vector>
#include <hdf5.h>

namespace imas {
namespace direct_access {

/**
 * @class DirectReader
 * @brief Reads a tensor from a PanzerDB file given a parsed selection path.
 *
 * As implemented (direct_reader.cpp) this is a narrow proof-of-concept for the
 * direct-access API rather than a general reader: it opens "<ids_name>.h5", requires
 * the segments to contain one time-slice selection (SLICE or ALL) plus an
 * unselected "ion" level, and assembles [n_slices, n_ion] by reading the
 * "profiles_1d/<time>/ion/<i>/z_ion" scalar leaves one by one. Extending it means
 * resolving the AoS levels and the leaf path from the segments instead of the
 * hard-coded names.
 */
class DirectReader {
public:
    /**
     * @brief Binds the reader to one IDS.
     * @param ids_name IDS name; the file read is "<ids_name>.h5", next to the caller's
     *                 working directory (no pulse-layout resolution here).
     */
    explicit DirectReader(const std::string& ids_name);

    /**
     * @brief Destructor.
     */
    ~DirectReader();

    /**
     * @brief Reads the tensor selected by a parsed path.
     * @param segments Segments produced by PathParser/parse_path(), in path order.
     * @return A TensorView owning a freshly allocated buffer, with the time range as
     *         its first dimension. An empty view when the path is empty or the
     *         requested slice range is empty.
     * @throws std::runtime_error when the segments carry no slice/all selection, when
     *         the required static-AoS level is absent, or when a leaf read fails.
     * @note The PanzerDB engine is opened and closed inside this call, so there is no
     *       state to reuse across reads of the same IDS.
     */
    TensorView read(const std::vector<PathSegment>& segments);

private:
    std::string ids_name_;   // IDS name; ".h5" is appended to derive the file to open.
};

} // namespace direct_access
} // namespace imas

#endif // DIRECT_READER_H
