#ifndef DIRECT_READER_H
#define DIRECT_READER_H

#include "direct_access_api.h"
#include "path_parser.h" // Inclure pour PathSegment
#include <string>
#include <vector>
#include <hdf5.h>

namespace imas {
namespace direct_access {

/**
 * @class DirectReader
 * @brief Reads data directly from the storage backend.
 */
class DirectReader {
public:
    /**
     * @brief Constructeur.
     * @param ids_name The name of the IDS to read.
     */
    explicit DirectReader(const std::string& ids_name);

    /**
     * @brief Destructeur.
     */
    ~DirectReader();

    /**
     * @brief Reads a tensor based on a sequence of parsed path segments.
     * @param segments The path segments parsed by PathParser.
     * @return A TensorView holding the read data.
     */
    TensorView read(const std::vector<PathSegment>& segments);

private:
    std::string ids_name_;
};

} // namespace direct_access
} // namespace imas

#endif // DIRECT_READER_H
