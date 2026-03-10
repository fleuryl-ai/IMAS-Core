#include "hdf5_backend.h"

#include <string.h>
#include <algorithm>
#include "hdf5_utils.h"
#include "hdf5_backend_factory.h"

HDF5Backend::HDF5Backend()
:  file_id(-1), pulseFilePath(""), opened_IDS_files()
{
    //H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    createBackendComponents(getVersion());
}

HDF5Backend::HDF5Backend(Backend * targetB)
{
    //H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
}

HDF5Backend::~HDF5Backend()
{
}

const int HDF5Backend::HDF5_BACKEND_VERSION_MAJOR = 2;
const int HDF5Backend::HDF5_BACKEND_VERSION_MINOR = 0;


void
 HDF5Backend::createBackendComponents(std::pair<int,int>  backend_version) {
    
    HDF5BackendFactory backendFactory(backend_version);
    hdf5Writer = backendFactory.createWriter();
    hdf5Reader = backendFactory.createReader();
    eventsHandler = backendFactory.createEventsHandler();
}

std::pair<int,int> HDF5Backend::getVersion(DataEntryContext *ctx)
{
  std::pair<int,int> version;

  if (getenv("IMAS_HDF5_BACKEND_VERSION") != NULL) {
      std::string env_version_str(getenv("IMAS_HDF5_BACKEND_VERSION"));
      size_t dot_pos = env_version_str.find('.');
      if (dot_pos != std::string::npos) {
        int major = std::stoi(env_version_str.substr(0, dot_pos));
        int minor = std::stoi(env_version_str.substr(dot_pos + 1));
        version = {major, minor};
      } else {
        version = {HDF5_BACKEND_VERSION_MAJOR, HDF5_BACKEND_VERSION_MINOR};
      }
    } else {
      version = {HDF5_BACKEND_VERSION_MAJOR, HDF5_BACKEND_VERSION_MINOR};
    }
  if(ctx!=NULL){ 

     std::string backend_version_from_file = "-1";
    try {
     
        files_path_strategy = HDF5Utils::MODIFIED_MDSPLUS_STRATEGY;
      
        //we call openPulse() which reads the backend version from the master file (no attempt for opening the master file will be performed if it is already opened) 
        HDF5Utils::openPulse(ctx, OPEN_PULSE, backend_version_from_file, &this->file_id, opened_IDS_files, files_path_strategy, files_directory, relative_file_path, this->pulseFilePath); 
      
        }
      catch (std::exception &e) {
            /*char error_message[200];
            sprintf(error_message, "Unable to get backend required version: %s\n", e.what());
            throw ALBackendException(error_message, LOG);*/
            //backend_version_from_file = -1;
      }
      if (!backend_version_from_file.empty() && backend_version_from_file != "-1") {
        version = HDF5BackendFactory::getRequiredVersion(backend_version_from_file);
        printf("Required backend components will be created for version: %d.%d\n", version.first, version.second);
      } 

      HDF5BackendFactory backendFactory(version);
      auto hdf5Reader_version = backendFactory.createReader();
      bool masterFileAlreadyOpened = (this->file_id != -1);
      if (!masterFileAlreadyOpened) //the master pulse file is closed only if it was already closed before to call the getVersion() method
        hdf5Reader_version->closePulse(ctx, OPEN_PULSE, &this->file_id, opened_IDS_files, files_path_strategy, files_directory, relative_file_path);
    }
  return version;
}

std::pair<int,int> HDF5Backend::getVersion() {
    return getVersion(NULL);;
}

void
 HDF5Backend::openPulse(DataEntryContext * ctx, int mode)
{
    access_mode = mode;

    std::pair<int,int> backend_version;
    
    files_path_strategy = HDF5Utils::MODIFIED_MDSPLUS_STRATEGY;
    std::string backend_version_str;

    switch (mode) {
        case OPEN_PULSE:
        case FORCE_OPEN_PULSE: 
            {
            backend_version = getVersion(ctx);
            backend_version_str = std::to_string(backend_version.first) + "." + std::to_string(backend_version.second);
            int status = HDF5Utils::openPulse(ctx, mode, backend_version_str, &this->file_id, opened_IDS_files, files_path_strategy, files_directory, relative_file_path, this->pulseFilePath); 
            if (status == -1) { //master file doesn't exist
                backend_version = getVersion();
                backend_version_str = std::to_string(backend_version.first) + "." + std::to_string(backend_version.second);
                HDF5Utils::createPulse(ctx, mode, backend_version_str, &this->file_id, opened_IDS_files, files_path_strategy, files_directory, relative_file_path, this->pulseFilePath);
            }
            break;
            }
        case CREATE_PULSE:
        case FORCE_CREATE_PULSE:
            backend_version = getVersion();
            backend_version_str = std::to_string(backend_version.first) + "." + std::to_string(backend_version.second);
            HDF5Utils::createPulse(ctx, mode, backend_version_str, &this->file_id, opened_IDS_files, files_path_strategy, files_directory, relative_file_path, this->pulseFilePath);
            break;
        default:
            throw ALBackendException("Mode not yet supported", LOG);
    }
    createBackendComponents(backend_version);
}

void HDF5Backend::closePulse(DataEntryContext * ctx, int mode)
{
    if (ctx == nullptr)
        throw ALBackendException("HDF5Backend: unexpected null context in HDF5Backend::closePulse()", LOG);
    if (access_mode == OPEN_PULSE || access_mode == FORCE_OPEN_PULSE) {
        hdf5Reader->closePulse(ctx, mode, &file_id, opened_IDS_files, files_path_strategy, files_directory, relative_file_path);
    } else if (access_mode == CREATE_PULSE || access_mode == FORCE_CREATE_PULSE) {
        hdf5Writer->closePulse(ctx, mode, &file_id, opened_IDS_files, files_path_strategy, files_directory, relative_file_path);
    }
    hdf5Writer->close_datasets();
    hdf5Reader->close_datasets();
}

void HDF5Backend::writeData(Context * ctx, std::string fieldname, std::string timebasename, void *data, int datatype, int dim, int *size)
{
    hdf5Writer->write_ND_Data(ctx, fieldname, timebasename, datatype, dim, size, data);
}

int HDF5Backend::readData(Context * ctx, std::string fieldname, std::string timebasename, void **data, int *datatype, int *dim, int *size)
{
    int dataAvailable = 0;      //not available by default
    dataAvailable = hdf5Reader->read_ND_Data(ctx, fieldname, timebasename, datatype, data, dim, size);
    return dataAvailable;
}


void HDF5Backend::deleteData(OperationContext * ctx, std::string path)
{
    if (file_id == -1) //master file is closed
        return;
    hdf5Writer->deleteData(ctx, this->file_id, opened_IDS_files, files_directory, relative_file_path);
}

void HDF5Backend::beginWriteArraystructAction(ArraystructContext * ctx, int *size)
{
    if (*size == 0)
        return;
    hdf5Writer->beginWriteArraystructAction(ctx, size);
}

void HDF5Backend::beginReadArraystructAction(ArraystructContext * ctx, int *size)
{
    hdf5Reader->beginReadArraystructAction(ctx, size);
}

void HDF5Backend::beginAction(OperationContext * ctx)
{
    eventsHandler->beginAction(ctx, file_id, opened_IDS_files, *hdf5Writer, *hdf5Reader, files_directory, relative_file_path, access_mode);
}

void HDF5Backend::endAction(Context * ctx)
{
    eventsHandler->endAction(ctx, file_id, *hdf5Writer, *hdf5Reader, opened_IDS_files);
}

void HDF5Backend::get_occurrences(Context* ctx, const  char* ids_name, int** occurrences_list, int* size)
{
    if (file_id == -1) //master file not opened
        throw ALBackendException("HDF5Backend: master file not opened while calling HDF5Backend::get_occurrences()", LOG); 
    hdf5Reader->get_occurrences(ids_name, occurrences_list, size, file_id);
}
