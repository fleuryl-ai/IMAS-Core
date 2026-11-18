/**
 * @file metadata_extractor.h
 * @brief Reads the IMAS data dictionary (IDSDef.xml) and flattens the schema-level
 *        attributes of every scalar field into a flat "path@attribute" -> value map.
 *
 * The HDF5 v2 writer uses this to replay the dictionary metadata (units, doc,
 * coordinates, ...) as "@key" entries alongside the data, so a PanzerDB file can
 * be interpreted without opening the dictionary again. The map keys are schema
 * paths (no AoS instance indices), which is what PanzerDB::stripIndices() and
 * PanzerDB::readMetadata() resolve an instance path to.
 */
#ifndef METADATA_EXTRACTOR_H
#define METADATA_EXTRACTOR_H

#include <string>
#include <map>
#include "pugixml.hpp"

/**
 * @class MetadataExtractor
 * @brief One parsed IDSDef.xml, queried per IDS for its field metadata.
 *
 * The XML document is loaded once in the constructor and kept in memory; each
 * extract_metadata() call then walks only the requested IDS subtree. A failed
 * load is not fatal: the object stays usable and every query returns an empty
 * map (the writer simply stores no metadata).
 */
class MetadataExtractor {
public:
    /**
     * @brief Parses an IDS definition XML file.
     * @param xml_file_path Path to the IDSDef.xml file to load.
     * @note Parse errors are reported on stderr and leave the object in the
     *       "no metadata" state rather than throwing — a missing dictionary must
     *       not abort a conversion run.
     */
    explicit MetadataExtractor(const std::string& xml_file_path);

    /**
     * @brief Collects the metadata of every scalar field of one IDS.
     * @param ids_name IDS name as it appears in the IDSs/IDS@name attribute
     *                 (e.g. "core_profiles").
     * @return Map of "<schema path>@<attribute>" -> value, e.g.
     *         {"core_profiles/1/ion/temperature@units": "eV"}. Empty when the
     *         document failed to load or the IDS name is unknown.
     * @note Only scalar leaf kinds are visited (data_type INT_*, FLT_*, STR_*,
     *       CPX_*); structure/struct_array containers and logical/undefined
     *       fields are skipped. The attributes kept are path_doc, documentation,
     *       data_type, type, units, timebasepath and coordinate*.
     */
    std::map<std::string, std::string> extract_metadata(const std::string& ids_name);

private:
    /**
     * @brief Depth-first walk of one dictionary node, appending the metadata of
     *        the scalar fields found below it.
     * @param node     Current dictionary node (<IDS>, <structure>, <field>, ...).
     * @param metadata Output map filled in place (see extract_metadata()).
     */
    void traverse_nodes(pugi::xml_node node, std::map<std::string, std::string>& metadata);

    pugi::xml_document doc;       // Parsed dictionary, kept alive for the queries below.
    bool doc_loaded = false;      // False when the XML failed to parse; all queries then return empty.
};

#endif // METADATA_EXTRACTOR_H
