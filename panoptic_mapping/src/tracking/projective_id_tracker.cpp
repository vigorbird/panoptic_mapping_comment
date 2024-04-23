#include "panoptic_mapping/tracking/projective_id_tracker.h"

#include <future>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "panoptic_mapping/common/index_getter.h"

namespace panoptic_mapping {

config_utilities::Factory::RegistrationRos<IDTrackerBase, ProjectiveIDTracker,
                                           std::shared_ptr<Globals>>
    ProjectiveIDTracker::registration_("projective");

void ProjectiveIDTracker::Config::checkParams() const {
  checkParamGT(rendering_threads, 0, "rendering_threads");
  checkParamNE(depth_tolerance, 0.f, "depth_tolerance");
  checkParamGT(rendering_subsampling, 0, "rendering_subsampling");
  checkParamConfig(renderer);
}

void ProjectiveIDTracker::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("depth_tolerance", &depth_tolerance);
  setupParam("tracking_metric", &tracking_metric);
  setupParam("match_acceptance_threshold", &match_acceptance_threshold);
  setupParam("use_class_data_for_matching", &use_class_data_for_matching);
  setupParam("use_approximate_rendering", &use_approximate_rendering);
  setupParam("rendering_subsampling", &rendering_subsampling);
  setupParam("min_allocation_size", &min_allocation_size);
  setupParam("rendering_threads", &rendering_threads);
  setupParam("renderer", &renderer);
}

ProjectiveIDTracker::ProjectiveIDTracker(const Config& config,
                                         std::shared_ptr<Globals> globals,
                                         bool print_config)
    : IDTrackerBase(std::move(globals)),
      config_(config.checkValid()),
      renderer_(config.renderer, globals_->camera()->getConfig(), false) {
  LOG_IF(INFO, config_.verbosity >= 1 && print_config) << "\n"
                                                       << config_.toString();
  //需要的输入
  addRequiredInputs({InputData::InputType::kColorImage,
                     InputData::InputType::kDepthImage,
                     InputData::InputType::kSegmentationImage,
                     InputData::InputType::kValidityImage});
}

//ProjectiveIDTracker::processInput 实现
void ProjectiveIDTracker::processInput(SubmapCollection* submaps,
                                       InputData* input) {
  CHECK_NOTNULL(submaps);
  CHECK_NOTNULL(input);
  CHECK(inputIsValid(*input));

  // Visualization.不用看没用！
  cv::Mat input_vis;
  std::unique_ptr<Timer> vis_timer;
  if (visualizationIsOn()) {
    vis_timer = std::make_unique<Timer>("visualization/tracking");
    Timer timer("visualization/tracking/input_image");
    input_vis = renderer_.colorIdImage(input->idImage());
    vis_timer->Pause();
  }

  // Render all submaps.
  Timer timer("tracking");
  auto t0 = std::chrono::high_resolution_clock::now();
  Timer detail_timer("tracking/compute_tracking_data");
  TrackingInfoAggregator tracking_data = computeTrackingData(submaps, input);//非常重要的函数！！！！！！ 就在这个文件中搜索即可！

  // Assign the input ids to tracks or allocate new maps.
  detail_timer = Timer("tracking/match_ids");
  std::unordered_map<int, int> input_to_output;//key= 语义标签，  value = submapid，表示对于空间中的这个语义物体到底哪个submap观测到它最多
  std::stringstream info;
  int n_matched = 0;
  int n_new = 0;
  Timer alloc_timer("tracking/allocate_submaps");
  alloc_timer.Pause();
  //2.遍历所有的语义标签
  for (const int input_id : tracking_data.getInputIDs()) {
    int submap_id;
    bool matched = false;
    float value;
    bool any_overlap;
    std::stringstream logging_details;

    // Find matches.
    //根据作者的所有yaml配置， use_class_data_for_matching都等于true
    if (config_.use_class_data_for_matching || config_.verbosity >= 4) {
      std::vector<std::pair<int, float>> ids_values;
      //2.1 整个代码就这里调用了 getAllMetrics函数
      any_overlap = tracking_data.getAllMetrics(input_id, //in 语义标签
                                                &ids_values,//out，first = submapid, second = iou,
                                                config_.tracking_metric);//in 根据作者的所有yaml配置， tracking_metric 都等于iOU
      // Check for classes if requested.
      if (config_.use_class_data_for_matching) {
        for (const auto& id_value : ids_values) {
          // These are ordered in decreasing overlap metric.
          //match_acceptance_threshold 在所有配置文件中都等于0.1
          if (id_value.second < config_.match_acceptance_threshold) {
            // No more matches possible.
            break;
          } else if (classesMatch(input_id,
                                  submaps->getSubmap(id_value.first).getClassID())) {//从gobal中查询是否之前的submap 类型id和当前的相同
            // Found the best match.
            matched = true;
            submap_id = id_value.first;//submapid
            value = id_value.second;//iou
            break;
          }
        }
      } else if (any_overlap && ids_values.front().second > config_.match_acceptance_threshold) {
        // Check only for the highest match.
        matched = true;
        submap_id = ids_values.front().first;
        value = ids_values.front().second;
      }

      // Print the matching statistics for all submaps.
      if (config_.verbosity >= 4) {
        logging_details << ". Overlap: ";
        for (const auto& id_value : ids_values) {
          logging_details << " " << id_value.first << "(" << std::fixed
                          << std::setprecision(2) << id_value.second << ")";
        }
      } else {
        logging_details << ". No overlap found.";
      }
    } else if (tracking_data.getHighestMetric(input_id, &submap_id, &value,
                                              config_.tracking_metric)) {//整个代码就这里调用了 getHighestMetric函数
      // Only consider the highest metric candidate.
      if (value > config_.match_acceptance_threshold) {
        matched = true;
      }
    }

    // Allocate new submap if necessary and store tracking info.
    //2.2 
    alloc_timer.Unpause();
    bool allocate_new_submap = tracking_data.getNumberOfInputPixels(input_id) >= config_.min_allocation_size;
    if (matched) {//如果新的观测和老的观测一致，还是之前的submap观测到这个物体最多
      n_matched++;
      input_to_output[input_id] = submap_id;
      submaps->getSubmapPtr(submap_id)->setWasTracked(true);//表示这个submap被成功track了
    } else if (allocate_new_submap) //如果这个语义物体观测到的像素坐标足够多了，并且和之前submap的观测不再一致了
    {
      n_new++;
      Submap* new_submap = allocateSubmap(input_id, submaps, input);
      if (new_submap) {
        input_to_output[input_id] = new_submap->getID();
      } else {
        input_to_output[input_id] = -1;
      }
    } else {
      // Ignore these.
      input_to_output[input_id] = -1;
    }
    alloc_timer.Pause();

    // Logging.
    if (config_.verbosity >= 3) {
      if (matched) {
        info << "\n  " << input_id << "->" << submap_id << " (" << std::fixed
             << std::setprecision(2) << value << ")";
      } else {
        if (allocate_new_submap) {
          info << "\n  " << input_id << "->" << input_to_output[input_id]
               << " [new]";
        } else {
          info << "\n  " << input_id << " [ignored]";
        }
        if (any_overlap) {
          info << " (" << std::fixed << std::setprecision(2) << value << ")";
        }
      }
      info << logging_details.str();
    }
  }//遍历完所有的语义id
  detail_timer.Stop();

  // Translate the id image.
  //将保存语义id的图像改为保存submap_id
  for (auto it = input->idImagePtr()->begin<int>(); it != input->idImagePtr()->end<int>(); ++it) {
    *it = input_to_output[*it];   
  }

  // Allocate free space map if required.
  alloc_timer.Unpause();
  //搜索 MonolithicFreespaceAllocator::allocateSubmap
  //在作者整个的配置文件里面都是使用的 freespace_allocator_ 类型 = MonolithicFreespaceAllocator
  freespace_allocator_->allocateSubmap(submaps, input);
  alloc_timer.Stop();

  // Finish.
  auto t1 = std::chrono::high_resolution_clock::now();
  timer.Stop();
  if (config_.verbosity >= 2) {
    LOG(INFO) << "Tracked IDs in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << "ms, " << n_matched << " matched, " << n_new
              << " newly allocated." << info.str();
  }

  // Publish Visualization if requested.
  if (visualizationIsOn()) {
    vis_timer->Unpause();
    Timer timer("visualization/tracking/rendered");
    if (config_.use_approximate_rendering) {
      rendered_vis_ = renderer_.colorIdImage(
          renderer_.renderActiveSubmapIDs(*submaps, input->T_M_C()));
    }
    timer = Timer("visualization/tracking/tracked");
    cv::Mat tracked_vis = renderer_.colorIdImage(input->idImage());
    timer.Stop();
    visualize(input_vis, "input");
    visualize(rendered_vis_, "rendered");
    visualize(input->colorImage(), "color");
    visualize(tracked_vis, "tracked");
    vis_timer->Stop();
  }
}// end ProjectiveIDTracker::processInput function





Submap* ProjectiveIDTracker::allocateSubmap(int input_id,
                                            SubmapCollection* submaps,
                                            InputData* input) {
  LabelEntry label;
  if (globals_->labelHandler()->segmentationIdExists(input_id)) {
    label = globals_->labelHandler()->getLabelEntry(input_id);
  }
  return submap_allocator_->allocateSubmap(submaps, input, input_id, label);//搜索 SemanticSubmapAllocator::allocateSubmap
}

//
bool ProjectiveIDTracker::classesMatch(int input_id, int submap_class_id) {
  if (!globals_->labelHandler()->segmentationIdExists(input_id)) {
    // Unknown ID.
    return false;
  }
  return globals_->labelHandler()->getClassID(input_id) == submap_class_id;
}

TrackingInfoAggregator ProjectiveIDTracker::computeTrackingData( SubmapCollection* submaps, InputData* input) {

  // Render each active submap in parallel to collect overlap statistics.
  //1.根据当前帧的位姿 然后判断哪些submap会被观测到
  const std::vector<int> visible_submaps = globals_->camera()->findVisibleSubmapIDs(*submaps, input->T_M_C());//整个代码就这里调用了 findVisibleSubmapIDs这函数

  // Make sure the meshes of all submaps are update for tracking.
  //2.遍历所有观测到的submap
  for (int submap_id : visible_submaps) {
    submaps->getSubmapPtr(submap_id)->updateMesh();
  }

  //3. 每个submap都对应一个trackinfo,，开启多线程更新每个观测到的submap里面每个语义标签对应多少个像素坐标
  // Render each submap in parallel.
  SubmapIndexGetter index_getter(visible_submaps);
  std::vector<std::future<std::vector<TrackingInfo>>> threads;
  TrackingInfoAggregator tracking_data;
  for (int i = 0; i < config_.rendering_threads; ++i) {
    threads.emplace_back(std::async(
        std::launch::async,
        [this, i, &tracking_data, &index_getter, submaps,input]() -> std::vector<TrackingInfo> {
            // Also process the input image.
            if (i == 0) {
              //只是更新了TrackingInfoAggregator中的total_input_count_这个变量，key=语义分割id， value = 多少个像素
              tracking_data.insertInputImage(  input->idImage(), input->depthImage(),
                                              globals_->camera()->getConfig(), config_.rendering_subsampling);
            }
            std::vector<TrackingInfo> result;
            int index;
            while (index_getter.getNextIndex(&index)) {
              //
              if (config_.use_approximate_rendering) {//默认应该是进入这个条件
                result.emplace_back(this->renderTrackingInfoApproximate(submaps->getSubmap(index), *input));
              } else {
                result.emplace_back(this->renderTrackingInfoVertices(submaps->getSubmap(index), *input));
              }
            }
            return result;
        }));
  }

  // Join all threads.
  std::vector<TrackingInfo> infos;
  for (auto& thread : threads) {
    for (const TrackingInfo& info : thread.get()) {
      infos.emplace_back(std::move(info));
    }
  }

  //4.本质上是为了更新这两个元素 total_rendered_count_ 和 overlap_
  //total_rendered_count_记录了在所有的submap中每个语义对应多少个像素需要被渲染
  //overlap_记录了每个语义标签对应的每个submap中，多少个像素被渲染
  tracking_data.insertTrackingInfos(infos);//整个代码就这里调用了 insertTrackingInfos 这个函数

  // Render the data if required.
  if (visualizationIsOn() && !config_.use_approximate_rendering) {
    Timer timer("visualization/tracking/rendered");
    cv::Mat vis =
        cv::Mat::ones(globals_->camera()->getConfig().height,
                      globals_->camera()->getConfig().width, CV_32SC1) *
        -1;
    for (const TrackingInfo& info : infos) {
      for (const Eigen::Vector2i& point : info.getPoints()) {
        vis.at<int>(point.y(), point.x()) = info.getSubmapID();
      }
    }
    rendered_vis_ = renderer_.colorIdImage(vis);
  }
  return tracking_data;
}// end computeTrackingData function!

//submap = 当前帧被观测到的submap
//input = 当前帧的深度图和语意图
//这个函数的主要目的是取出这个submap对应的所有voxel，然后将这些voxel投影到图像上，统计这个图像上，多少个像素属于同一个语义id
TrackingInfo ProjectiveIDTracker::renderTrackingInfoApproximate( const Submap& submap, 
                                                                 const InputData& input) const {
  // Approximate rendering by projecting the surface points of the submap into
  // the camera and fill in a patch of the size a voxel has (since there is 1
  // vertex per voxel).

  // Setup.
  const Camera& camera = *globals_->camera();
  TrackingInfo result(submap.getID(), camera.getConfig());
  //根据相机在submap下的位姿变换 和 submap在世界坐标系的位姿变换 得到相机在世界坐标系下的位姿变换
  const Transformation T_C_S = input.T_M_C().inverse() * submap.getT_M_S();
  const float size_factor_x = camera.getConfig().fx * submap.getTsdfLayer().voxel_size() / 2.f;
  const float size_factor_y = camera.getConfig().fy * submap.getTsdfLayer().voxel_size() / 2.f;
  const float block_size = submap.getTsdfLayer().block_size();
  const FloatingPoint block_diag_half = std::sqrt(3.0f) * block_size / 2.0f;
  const float depth_tolerance = config_.depth_tolerance > 0 ? config_.depth_tolerance: -config_.depth_tolerance * submap.getTsdfLayer().voxel_size();
  const cv::Mat& depth_image = input.depthImage();

  // Parse all blocks.
  //1.获取这个submap里面的所有voxel索引
  voxblox::BlockIndexList index_list;
  submap.getMeshLayer().getAllAllocatedMeshes(&index_list);
  for (const voxblox::BlockIndex& index : index_list) {

    //如果这个voxel不在视野范围内 则直接continue
    if (!camera.blockIsInViewFrustum(submap, index, T_C_S, block_size,
                                     block_diag_half)) {
      continue;
    }
    //获取这个voxel里面的mesh顶点
    for (const Point& vertex : submap.getMeshLayer().getMeshByIndex(index).vertices) {
      // Project vertex and check depth value.
      //将世界坐标系下的点变换到相机坐标系下
      const Point p_C = T_C_S * vertex;
      int u, v;
      //判断是否3D点能否投影到图像上
      if (!camera.projectPointToImagePlane(p_C, &u, &v)) {
        continue;
      }
      //判断这个点的深度和我们当前帧的测量深度差的不能太多
      if (std::abs(depth_image.at<float>(v, u) - p_C.z()) >= depth_tolerance) {
        continue;
      }

      // Compensate for vertex sparsity.
      const int size_x = std::ceil(size_factor_x / p_C.z());
      const int size_y = std::ceil(size_factor_y / p_C.z());
      result.insertRenderedPoint(u, v, size_x, size_y);//不是插入数据，本质上是更新一个voxel大小的空间体，对应需要渲染的图像范围
    }
  }//遍历完了所有voxel
  //更新tracking info 里面的counts_
  result.evaluate(input.idImage(), depth_image);//搜索 TrackingInfo::evaluate
  return result;

}//end function renderTrackingInfoApproximate



TrackingInfo ProjectiveIDTracker::renderTrackingInfoVertices(
    const Submap& submap, const InputData& input) const {
  TrackingInfo result(submap.getID());

  // Compute the maximum extent to lookup vertices.
  const Transformation T_C_S = input.T_M_C().inverse() * submap.getT_M_S();
  const Point origin_C = T_C_S * submap.getBoundingVolume().getCenter();
  std::vector<size_t> limits(4);  // x_min, x_max, y_min, y_max
  const Camera::Config& cam_config = globals_->camera()->getConfig();

  // NOTE(schmluk): Currently just iterate over the whole frame since the sphere
  // tangent computation was not robustly implemented.
  size_t subsampling_factor = 1;
  limits = {0u, static_cast<size_t>(cam_config.width), 0u,
            static_cast<size_t>(cam_config.height)};
  const Transformation T_S_C = T_C_S.inverse();
  const TsdfLayer& tsdf_layer = submap.getTsdfLayer();
  const float depth_tolerance =
      config_.depth_tolerance > 0
          ? config_.depth_tolerance
          : -config_.depth_tolerance * submap.getTsdfLayer().voxel_size();
  for (size_t u = limits[0]; u < limits[1];
       u += config_.rendering_subsampling) {
    for (size_t v = limits[2]; v < limits[3];
         v += config_.rendering_subsampling) {
      const float depth = input.depthImage().at<float>(v, u);
      if (depth < cam_config.min_range || depth > cam_config.max_range) {
        continue;
      }
      const cv::Vec3f& vertex = input.vertexMap().at<cv::Vec3f>(v, u);
      const Point P_S = T_S_C * Point(vertex[0], vertex[1], vertex[2]);
      const voxblox::BlockIndex block_index =
          tsdf_layer.computeBlockIndexFromCoordinates(P_S);
      const auto block = tsdf_layer.getBlockPtrByIndex(block_index);
      if (block) {
        const size_t voxel_index =
            block->computeLinearIndexFromCoordinates(P_S);
        const TsdfVoxel& voxel = block->getVoxelByLinearIndex(voxel_index);
        bool classes_match = true;
        if (submap.hasClassLayer()) {
          classes_match = submap.getClassLayer()
                              .getBlockConstPtrByIndex(block_index)
                              ->getVoxelByLinearIndex(voxel_index)
                              .belongsToSubmap();
        }
        if (voxel.weight > 1e-6 && std::abs(voxel.distance) < depth_tolerance) {
          result.insertVertexPoint(input.idImage().at<int>(v, u));
          if (visualizationIsOn()) {
            result.insertVertexVisualizationPoint(u, v);
          }
        }
      }
    }
  }
  return result;
}

}  // namespace panoptic_mapping
