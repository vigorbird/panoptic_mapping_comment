#ifndef PANOPTIC_MAPPING_INPUT_POINTCLOUD_INTEGRATOR_FACTORY_H_
#define PANOPTIC_MAPPING_INPUT_POINTCLOUD_INTEGRATOR_FACTORY_H_

#include <memory>

#include "panoptic_mapping/input/pointcloud_integrator_base.h"

namespace panoptic_mapping {

/**
 * Class to produce pointcloud integrators
 */
class PointcloudIntegratorFactory {
 public:
  static std::unique_ptr<PointcloudIntegratorBase> create(const std::string &type);

 private:
  PointcloudIntegratorFactory() = default;
  ~PointcloudIntegratorFactory() = default;
};

}  // namespace panoptic_mapping

#endif //PANOPTIC_MAPPING_INPUT_POINTCLOUD_INTEGRATOR_FACTORY_H_
