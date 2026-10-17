#ifndef HDF5_BACKEND_FACTORY_H
#define HDF5_BACKEND_FACTORY_H 1


#include <hdf5.h>
#include "hdf5_writer.h"
#include "hdf5_reader.h"
#include "hdf5_events_handler.h"
#include "data_interpolation.h"
#include "hdf5_writer_v2.h"
#include "hdf5_reader_v2.h"


#include <memory>
#include <string>

/**
 * @file hdf5_backend_factory.h
 * @brief Instantiates the reader/writer pair matching the format version recorded
 *        in an existing file (or requested for a new one).
 *
 * Two on-disk layouts coexist: v1 (one HDF5 dataset per IMAS node, shapes stored
 * in companion datasets) and v2 (PanzerDB: columnar raw datasets plus a central
 * /index table). The backend version is read from the file's user block / version
 * attribute when opening, and the factory hands HDF5Backend the concrete classes
 * for that version. Adding a format feature therefore means adding a branch here,
 * not changing the v1/v2 classes.
 */

/**
 * @class HDF5BackendFactory
 * @brief Version-dispatching factory for the backend components.
 *
 * Holds only the version pair; the components themselves are returned as
 * unique_ptr over the abstract base classes, so HDF5Backend never names a
 * concrete version.
 */
class HDF5BackendFactory {
  private:
    std::pair<int,int> backend_version_;   // (major, minor) this factory instantiates, e.g. {2,0}.

  public:

    /**
     * @brief Binds the factory to one backend format version.
     * @param backend_version Version pair to instantiate (see getRequiredVersion()).
     */
    HDF5BackendFactory(std::pair<int,int> backend_version);
    ~HDF5BackendFactory();

  /**
   * @brief Creates the writer for the factory's version.
   * @return v1: HDF5Writer; v2: HDF5Writer_v2 (PanzerDB-backed).
   * @throws ALBackendException when no writer implements that version.
   */
    std::unique_ptr < HDF5Writer > createWriter();

  /**
   * @brief Creates the reader for the factory's version.
   * @return v1: HDF5Reader; v2: HDF5Reader_v2 (strategy-based, PanzerDB-backed).
   * @throws ALBackendException when no reader implements that version.
   */
    std::unique_ptr < HDF5Reader > createReader();

  /**
   * @brief Creates the action-event handler.
   * @return Always the version-independent HDF5EventsHandler: it drives the
   *         reader/writer through their base classes, so it needs no variant.
   */
    std::unique_ptr < HDF5EventsHandler > createEventsHandler();

  /**
   * @brief Maps the backend version string stored in a file onto a version pair.
   * @param backend_version_from_file Version text read from the file ("1.0" or "2.0").
   * @return {1,0} or {2,0} — the pair to hand to the constructor.
   * @throws ALBackendException on any unrecognised string, so a file written by a
   *         newer backend is rejected rather than misread.
   */
  static std::pair<int,int> getRequiredVersion(const std::string &backend_version_from_file);

};

#endif
