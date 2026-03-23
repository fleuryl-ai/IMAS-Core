#include "direct_reader.h"
#include "panzerdb.h"
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <cstring>

namespace imas {
namespace direct_access {

// --- Helper pour convertir les types ---
DataType convert_panzer_type_to_api(PanzerDB::DataType pz_type) {
    switch (pz_type) {
        case PanzerDB::DataType::FLOAT64: return DataType::DOUBLE;
        case PanzerDB::DataType::INT32: return DataType::INT32;
        case PanzerDB::DataType::COMPLEX128: return DataType::COMPLEX_DOUBLE;
        case PanzerDB::DataType::STRING: return DataType::STRING;
        default: return DataType::UNKNOWN;
    }
}

// --- Constructeur/Destructeur ---
DirectReader::DirectReader(const std::string& ids_name) : ids_name_(ids_name) {}
DirectReader::~DirectReader() {}

TensorView DirectReader::read(const std::vector<PathSegment>& segments) {
    if (segments.empty()) return TensorView();

    PanzerDB panzer_db(ids_name_ + ".h5", PanzerDB::OpenMode::READ);
    
    // 1. Analyser les segments pour trouver la slice et les AoS statiques
    const PathSegment* time_slice_segment = nullptr;
    const PathSegment* static_aos_segment = nullptr;

    for (const auto& segment : segments) {
        if (segment.selection == SelectionType::SLICE || segment.selection == SelectionType::ALL) {
            time_slice_segment = &segment;
        }
        if (segment.node_name == "ion" && segment.selection == SelectionType::NONE) {
            static_aos_segment = &segment;
        }
    }

    if (!time_slice_segment) throw std::runtime_error("Only time-sliced reads are supported for this implementation.");
    if (!static_aos_segment) throw std::runtime_error("Reading a full static AoS is required for this implementation.");

    // 2. Déterminer les dimensions
    size_t time_aos_size = panzer_db.getDynamicAOSSize("profiles_1d");
    int64_t start_idx = time_slice_segment->has_start ? time_slice_segment->start_index : 0;
    int64_t end_idx = time_slice_segment->has_end ? time_slice_segment->end_index : time_aos_size;
    if (end_idx > time_aos_size) end_idx = time_aos_size;
    int64_t total_slices_to_read = end_idx - start_idx;

    auto static_aos_shape = panzer_db.getAOSShape("profiles_1d/0/ion");
    size_t static_aos_count = static_aos_shape[0];

    if (total_slices_to_read <= 0) return TensorView();

    // 3. Allouer le buffer final
    std::vector<size_t> final_dims = {(size_t)total_slices_to_read, static_aos_count};
    std::unique_ptr<char[]> final_buffer = std::make_unique<char[]>(total_slices_to_read * static_aos_count * sizeof(double));

    // 4. Boucler et lire chaque scalaire
    for (int64_t t_iter = 0; t_iter < total_slices_to_read; ++t_iter) {
        int64_t current_time_index = start_idx + t_iter;
        for (size_t s_iter = 0; s_iter < static_aos_count; ++s_iter) {
            
            std::string full_path = "profiles_1d/" + std::to_string(current_time_index) + "/ion/" + std::to_string(s_iter) + "/z_ion";

            double* data_out = nullptr;
            uint64_t ndim = 0;
            uint64_t shape[6] = {0};

            int status = panzer_db.pz_readData_by_index(full_path.c_str(), -1, &ndim, shape, &data_out);
            
            if (status != 0 || !data_out) {
                throw std::runtime_error("Failed to read leaf at path: " + full_path);
            }
            
            size_t buffer_offset = (t_iter * static_aos_count + s_iter) * sizeof(double);
            memcpy(final_buffer.get() + buffer_offset, data_out, sizeof(double));
            
            free(data_out);
        }
    }
    
    return TensorView(std::move(final_buffer), final_dims, DataType::DOUBLE);
}

} // namespace direct_access
} // namespace imas
