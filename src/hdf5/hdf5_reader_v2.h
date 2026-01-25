#ifndef HDF5_READER_V2_H
#define HDF5_READER_V2_H 1

#include "hdf5_reader.h" // Include base class
#include <hdf5.h>
#include "iread_strategy.h" // Include strategy interface
#include "global_read_strategy.h" // Include default strategy
#include "slice_read_strategy.h" 
#include "timerange_read_strategy.h" 

#include "al_backend.h"
#include "data_interpolation.h"

#include <memory>
#include <vector>
#include <list>
#include <unordered_map>


class HDF5Reader_v2 : public HDF5Reader {

  private:

    /**
     * @brief Reads the homogeneous time property.
     * @param homogenenous_time Pointer to store the result.
     * @param gid HDF5 group ID.
     */
    void read_homogeneous_time(int* homogenenous_time, hid_t gid);

    /**
     * @brief Builds the path index for optimized lookups.
     */
    void build_path_index();
    
    std::unique_ptr<IReadStrategy> read_strategy; // Read strategy

  public:

    /**
     * @brief Constructor.
     * @param backend_version_ The version of the backend.
     */
     HDF5Reader_v2(std::string backend_version_);
    /**
     * @brief Destructor.
     */
    ~HDF5Reader_v2() override;

    /**
     * @brief Opens the HDF5 group for the specified IDS and initializes the read strategy.
     */
    void open_IDS_group(OperationContext * ctx, hid_t file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, std::string & files_directory, std::string & relative_file_path) override;
    /**
     * @brief Closes the HDF5 group associated with the context.
     */
    void close_group(OperationContext *ctx) override; 
    /**
     * @brief Closes the pulse file and releases resources.
     */
    void closePulse(DataEntryContext * ctx, int mode, hid_t *file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, int files_path_strategy, std::string & files_directory, std::string & relative_file_path) override;
    /**
     * @brief Reads N-Dimensional data from the IDS.
     * Delegates the read operation to the active read strategy.
     */
    int read_ND_Data(Context * ctx, std::string & att_name, std::string & timebasename, int *datatype, void **data, int *dim, int *size) override;
    /**
     * @brief Prepares reading of an Array of Structures (AoS).
     */
    void beginReadArraystructAction(ArraystructContext * ctx, int *size) override; 

    /**
     * @brief Finalizes the action on the current context.
     */
    void endAction(Context * ctx) override; 

    /**
     * @brief Returns the version of the HDF5 backend reader.
     */
    std::string getVersion() override;

};

#endif
