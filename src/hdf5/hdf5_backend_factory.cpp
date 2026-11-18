#include "hdf5_backend_factory.h"

#include "al_backend.h"

HDF5BackendFactory::HDF5BackendFactory(std::pair<int,int> backend_version):backend_version_(backend_version)
{
}

HDF5BackendFactory::~HDF5BackendFactory()
{
}

std::pair<int,int> HDF5BackendFactory::getRequiredVersion(const std::string &backend_version_from_file)
{
    if (backend_version_from_file.compare("1.0") == 0) {
        return {1, 0};
    } 
    else if (backend_version_from_file.compare("2.0") == 0) {
        return {2, 0};
    }
    else {
        std::string message("Unknown backend version: ");
        message += backend_version_from_file;
        throw ALBackendException(message, LOG);
    }
} 

std::unique_ptr < HDF5Writer > HDF5BackendFactory::createWriter()
{
    if (backend_version_.first == 1 && backend_version_.second == 0) {
        //printf("Creating HDF5Writer v1.0\n");
        std::unique_ptr < HDF5Writer > writer = std::unique_ptr < HDF5Writer > (new HDF5Writer(backend_version_));
        return writer;
    } 
    else if (backend_version_.first == 2 && backend_version_.second == 0) {
        //printf("Creating HDF5Writer_v2\n");
        std::unique_ptr < HDF5Writer_v2 > writer = std::unique_ptr < HDF5Writer_v2 > (new HDF5Writer_v2(backend_version_));
        return writer;
    }
    else {
        std::string message("No backend writer with version: ");
        message += std::to_string(backend_version_.first) + "." + std::to_string(backend_version_.second);
        throw ALBackendException(message, LOG);
    }

}

std::unique_ptr < HDF5Reader > HDF5BackendFactory::createReader()
{
    if (backend_version_.first == 1 && backend_version_.second == 0) {
        std::unique_ptr < HDF5Reader > reader = std::unique_ptr < HDF5Reader > (new HDF5Reader(backend_version_));
        return reader;
    } 
    else if (backend_version_.first == 2 && backend_version_.second == 0) {
        std::unique_ptr < HDF5Reader_v2 > reader = std::unique_ptr < HDF5Reader_v2 > (new HDF5Reader_v2(backend_version_));
        return reader;
    }
    else {
        std::string message("No backend reader with version: ");
        message += std::to_string(backend_version_.first) + "." + std::to_string(backend_version_.second);
        throw ALBackendException(message, LOG);
    }
}

std::unique_ptr < HDF5EventsHandler > HDF5BackendFactory::createEventsHandler()
{
    auto eventsHandler = std::unique_ptr < HDF5EventsHandler > (new HDF5EventsHandler());
    return eventsHandler;
}
