#include "hdf5_backend.h"
#include <string.h>
#include <algorithm>
#include "hdf5_utils.h"
#include "hdf5_backend_factory.h"
#include <iostream>

#define BACKEND_DEFAULT_WRITE_VERSION "BACKEND_DEFAULT_WRITE_VERSION"
#define BACKEND_ALLOW_AUTO_UPGRADE "BACKEND_ALLOW_AUTO_UPGRADE"
#define BACKEND_HDF5_FORCE_CONVERSION_TO_V1 "BACKEND_HDF5_FORCE_CONVERSION_TO_V1"


HDF5Backend::HDF5Backend()
:  file_id(-1), pulseFilePath(""), opened_IDS_files()
{
    createBackendComponents(getVersion());
}

HDF5Backend::HDF5Backend(Backend * targetB)
{
}

HDF5Backend::~HDF5Backend()
{
}

const int HDF5Backend::HDF5_BACKEND_VERSION_MAJOR = 2;
const int HDF5Backend::HDF5_BACKEND_VERSION_MINOR = 0;

void
 HDF5Backend::createBackendComponents(std::pair<int,int> backend_version) {
    //if (allowUpgrade())
    //    backend_version = std::make_pair(2,0);
    HDF5BackendFactory backendFactory(backend_version);
    hdf5Writer = backendFactory.createWriter();
    hdf5Reader = backendFactory.createReader();
    eventsHandler = backendFactory.createEventsHandler();
}

std::pair<int, int> HDF5Backend::parseVersion(const std::string& versionStr) {
    std::pair<int,int> version = {HDF5_BACKEND_VERSION_MAJOR, HDF5_BACKEND_VERSION_MINOR};
    size_t dot_pos = versionStr.find('.');
      if (dot_pos != std::string::npos) {
        int major = std::stoi(versionStr.substr(0, dot_pos));
        int minor = std::stoi(versionStr.substr(dot_pos + 1));
        version = {major, minor};
      }
    return version;
}

bool HDF5Backend::allowUpgrade() {
    char* v = std::getenv(BACKEND_ALLOW_AUTO_UPGRADE);  // Autorise-t-on la transformation physique du fichier de 1.0 vers 2.0 ?
    return v && std::string(v) == "true";
}

bool HDF5Backend::forceConversionToV1() {
    char* v = std::getenv(BACKEND_HDF5_FORCE_CONVERSION_TO_V1);  // Autorise-t-on la transformation physique du fichier de 2.0 vers 1.0 ?
    return v && std::string(v) == "true";
}

std::pair<int, int> HDF5Backend::getTargetVersion() {
    char* v = std::getenv(BACKEND_DEFAULT_WRITE_VERSION);
    return v ? parseVersion(v) : std::make_pair(2, 0);
}

std::pair<int,int> HDF5Backend::getVersion(DataEntryContext *ctx) {
    // --- DEBUG: Entrée de fonction ---
   
    if (ctx == NULL) {
        return getTargetVersion();
    } 

    bool masterFileAlreadyOpened = (this->file_id != -1);

    std::string physical_version_from_file = "-1";
    std::pair<int,int> physical_version = std::make_pair(-1, -1);

    try {
        
        //std::cout << "[DEBUG] Attempting to read version from file. Master file already opened: " << (masterFileAlreadyOpened ? "YES" : "NO") << std::endl;

        HDF5Utils::openPulse(ctx, OPEN_PULSE, physical_version_from_file, &this->file_id, opened_IDS_files, HDF5Utils::MODIFIED_MDSPLUS_STRATEGY, files_directory, relative_file_path, this->pulseFilePath); 
        
        physical_version = parseVersion(physical_version_from_file);
        //std::cout << "[DEBUG] Physical version detected: " << physical_version.first << "." << physical_version.second << " (string: " << physical_version_from_file << ")" << std::endl;

        if (!masterFileAlreadyOpened) {
            //std::cout << "[DEBUG] Closing pulse as it was temporary for version detection." << std::endl;
            HDF5BackendFactory backendFactory(physical_version);
            auto hdf5Reader_version = backendFactory.createReader();
            hdf5Reader_version->closePulse(ctx, OPEN_PULSE, &this->file_id, opened_IDS_files, files_path_strategy, files_directory, relative_file_path);
        } 
    }
    catch (std::exception &e) {
        //std::cerr << "[ERROR] Exception during version detection: " << e.what() << std::endl;
    }

    std::pair<int, int> m_identityVersion = getTargetVersion();
    //std::cout << "[DEBUG] Identity Version (Target): " << m_identityVersion.first << "." << m_identityVersion.second << std::endl;

    // --- INSTANCIATION DU READER ---
    if (physical_version != std::make_pair(-1, -1)) { //file exists
        //std::cout << "[DEBUG] Instantiating Reader with Physical major version." << physical_version.first << std::endl;
        HDF5BackendFactory backendFactory(physical_version);
        hdf5Reader = backendFactory.createReader();
    } 
    else {
        //std::cout << "[DEBUG] Instantiating Reader with major version." << physical_version.second  << std::endl;
        HDF5BackendFactory backendFactory(getTargetVersion());
        hdf5Reader = backendFactory.createReader();
    } 

    // --- LOGIQUE DE DÉCISION DU WRITER ---
    std::pair<int, int> writerVersion;
    std::pair<int, int> versionToReturn;

    switch (access_mode) {
        case alconst::open_pulse:
        case alconst::force_open_pulse:
            if (physical_version != getTargetVersion() && getTargetVersion() == std::make_pair(2, 0)){
                //printf("physical_version lower than backend version\n");
                if (allowUpgrade()){
                    //printf("Allowed to upgrade\n");
                    // Writing v2 tables into a v1 file is a rewrite, not an edit: the v1 datasets
                    // have to go first (beginAction drops them), and the master file has to stop
                    // claiming the pulse is v1 (closePulse re-annotates it). Without the second
                    // half, the next open reads that label, picks the v1 reader, and never sees
                    // what this session wrote.
                    upgrade_to_v2_ = true;
                    writerVersion = std::make_pair(2, 0);
                    versionToReturn = getTargetVersion();
                }
                else{
                    //printf("Not allowed to upgrade from physical major version %d\n", physical_version.first);
                    throw ALBackendException("Not allowed to upgrade to 2.0 from version 1.0 (BACKEND_ALLOW_AUTO_UPGRADE is false)");
                    //versionToReturn = physical_version; //error from LL no thrown ???
                } 
            } 
            else if (physical_version  == getTargetVersion() && getTargetVersion() == std::make_pair(2, 0)){
                //printf("physical_version same than backend version in 2.0\n");
                writerVersion = std::make_pair(2, 0);
                versionToReturn = getTargetVersion();
            } 
            else if (physical_version  == std::make_pair(1, 0) && getTargetVersion() == std::make_pair(1, 0)){
                //printf("physical_version same than backend version in 1.0\n");
                writerVersion = std::make_pair(1, 0);
                versionToReturn = getTargetVersion();
            } 
            else if (physical_version  == std::make_pair(2, 0) && getTargetVersion() == std::make_pair(1, 0)){
                // Reading v2 while writing v1 is not an edit of the same storage: the v2
                // tables have to go and the pulse re-annotated. Only allowed on purpose.
                if (!forceConversionToV1())
                    throw ALBackendException("Not allowed to downgrade from version 2.0 to version 1.0 "
                                             "(set BACKEND_HDF5_FORCE_CONVERSION_TO_V1=true to convert a pulse for testing).");
                // Nothing to install here: the reader above already follows the physical 2.0
                // layout, and the writer is the 1.0 one this process was built with. The flag
                // only records that this session is a rewrite, so the first write drops the
                // v2 storage and closePulse() rewrites the version header.
                conversion_to_v1_ = true;
                storage_rewritten_ = false;
            } 
            
        case alconst::create_pulse:
        case alconst::force_create_pulse:
            //printf("Returning version: %d.%d", getTargetVersion().first, getTargetVersion().second);
            return getTargetVersion();
            break;
    } 

    if (!masterFileAlreadyOpened){ 
        //printf("Setting writer with version: %d.%d", writerVersion.first, writerVersion.second);
        HDF5BackendFactory backendFactory(writerVersion);
        hdf5Writer = backendFactory.createWriter();
    } 

    return versionToReturn;
} 

std::pair<int,int> HDF5Backend::getVersion() {
    return getVersion(NULL);
}

void
 HDF5Backend::openPulse(DataEntryContext * ctx, int mode)
{
    access_mode = mode;
    // A rewrite belongs to one pulse session: getVersion() arms it, and the flags must not
    // leak into the next pulse this backend opens -- the next file may well match the target.
    conversion_to_v1_ = false;
    upgrade_to_v2_ = false;
    storage_rewritten_ = false;
    rewritten_IDS_.clear();

    std::pair<int,int> backend_version;
    
    files_path_strategy = HDF5Utils::MODIFIED_MDSPLUS_STRATEGY;
    std::string backend_version_str;

    switch (mode) {
        case OPEN_PULSE:
        case FORCE_OPEN_PULSE: 
            {
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
            //printf("Creating pulse file with version: %d.%d\n ", backend_version.first, backend_version.second);
            //if (allowUpgrade())
            //    backend_version = std::make_pair(2,0);
            backend_version_str = std::to_string(backend_version.first) + "." + std::to_string(backend_version.second);
            HDF5Utils::createPulse(ctx, mode, backend_version_str, &this->file_id, opened_IDS_files, files_path_strategy, files_directory, relative_file_path, this->pulseFilePath);
            break;
        default:
            throw ALBackendException("Mode not yet supported", LOG);
    }
}

void HDF5Backend::closePulse(DataEntryContext * ctx, int mode)
{
    if (ctx == nullptr)
        throw ALBackendException("HDF5Backend: unexpected null context in HDF5Backend::closePulse()", LOG);
    // A session that rewrote the pulse laid a layout down that the master file does not
    // advertise: it still carries the label of the layout that was there before. The IDS files
    // were recreated by the writer and carry their own header; the master is the one openPulse()
    // reads to pick a reader, so it has to follow -- in either direction, hence the target
    // version rather than a literal. Only once the session really wrote something: a session
    // that only read a v1 pulse with a 2.0 target leaves both files alone.
    if (storage_rewritten_ && file_id != -1) {
        const std::pair<int, int> written = getTargetVersion();
        HDF5Utils hdf5_utils;
        hdf5_utils.rewriteHeaderVersion(file_id,
                                        std::to_string(written.first) + "." + std::to_string(written.second));
    }
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
    // A session writing a layout other than the one on disk rewrites the IDS from scratch, and
    // the old storage has to go before the event handler opens the IDS file for this operation:
    // the writer -- v1 in a v2 to v1 session, v2 in an upgrade -- then lays the file down anew
    // (file, external link and its own version header, all by HDF5Writer::create_IDS_group)
    // instead of adding its datasets next to the previous layout's.
    if ((conversion_to_v1_ || upgrade_to_v2_) && ctx && ctx->getAccessmode() == WRITE_OP)
        dropIDSStorageForRewrite(ctx);
    eventsHandler->beginAction(ctx, file_id, opened_IDS_files, *hdf5Writer, *hdf5Reader, files_directory, relative_file_path, access_mode);
}

void HDF5Backend::dropIDSStorageForRewrite(OperationContext *ctx)
{
    std::string IDS_link_name = ctx->getDataobjectName();
    std::replace(IDS_link_name.begin(), IDS_link_name.end(), '/', '_');
    // Once per IDS, whatever the outcome below: a session writing two IDS of the same pulse has
    // to erase both of their files, not only the first one -- the second would otherwise end up
    // holding both layouts.
    if (!rewritten_IDS_.insert(IDS_link_name).second) return;
    storage_rewritten_ = true;
    if (file_id == -1) return;

    // The v2 read session keeps one engine + index alive across operations, with dataset
    // handles on /index and the raw tables; both readers and writer cache handles opened
    // through the IDS file. All of that has to be let go before the objects go away.
    hdf5Reader->release_session();
    hdf5Reader->close_datasets();
    hdf5Writer->close_datasets();
    hdf5Reader->close_group(ctx);
    hdf5Writer->close_group(ctx);

    HDF5Utils hdf5_utils;
    const std::string IDS_file = hdf5_utils.getIDSPulseFilePath(files_directory, relative_file_path, IDS_link_name);

    auto got = opened_IDS_files.find(IDS_link_name);
    if (got != opened_IDS_files.end() && got->second != -1) {
        hdf5_utils.closeIDSFile(got->second, IDS_link_name);
        got->second = -1;   // the writer reopens -- and now creates -- it under this name
    }
    // The file goes away entirely rather than having its objects unlinked one by one: HDF5 never
    // hands space back, so an emptied file would still weigh what the layout it is leaving cost
    // -- and comparing sizes is often the whole point of rewriting a pulse.
    if (H5Lexists(file_id, IDS_link_name.c_str(), H5P_DEFAULT) > 0)
        hdf5_utils.removeLinkFromMasterPulseFile(file_id, IDS_link_name);
    hdf5_utils.deleteIDSFile(IDS_file);
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
