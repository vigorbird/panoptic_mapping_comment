#include "panoptic_mapping/integrator/projection_interpolators.h"

#include <cmath>
#include <limits>
#include <unordered_map>

namespace panoptic_mapping {

config_utilities::Factory::Registration<InterpolatorBase, InterpolatorNearest>
    InterpolatorNearest::registration_("nearest");
config_utilities::Factory::Registration<InterpolatorBase, InterpolatorBilinear>
    InterpolatorBilinear::registration_("bilinear");
config_utilities::Factory::Registration<InterpolatorBase, InterpolatorAdaptive>
    InterpolatorAdaptive::registration_("adaptive");

int InterpolatorNearest::computeWeights(float u, float v,
                                        const Eigen::MatrixXf& range_image,
                                        const cv::Mat& id_image) {
  u_ = std::round(u);
  v_ = std::round(v);
  return id_image.at<int>(v_, u_);
}

float InterpolatorNearest::interpolateDepth(
    const Eigen::MatrixXf& range_image) {
  return range_image(v_, u_);
}

Color InterpolatorNearest::interpolateColor(const cv::Mat& color_image) {
  auto color_bgr = color_image.at<cv::Vec3b>(v_, u_);
  return Color(color_bgr[2], color_bgr[1], color_bgr[0]);
}

int InterpolatorBilinear::computeWeights(float u, float v,
                                         const Eigen::MatrixXf& range_image,
                                         const cv::Mat& id_image) {
  u_ = std::floor(u);
  v_ = std::floor(v);
  float du = u - static_cast<float>(u_);
  float dv = v - static_cast<float>(v_);
  weight_[0] = (1.f - du) * (1.f - dv);
  weight_[1] = (1.f - du) * dv;
  weight_[2] = du * (1.f - dv);
  weight_[3] = du * dv;
  std::unordered_map<int, float> ids;  // these are zero initialized by default.
  ids[id_image.at<int>(v_, u_)] += weight_[0];
  ids[id_image.at<int>(v_ + 1, u_)] += weight_[1];
  ids[id_image.at<int>(v_, u_ + 1)] += weight_[2];
  ids[id_image.at<int>(v_ + 1, u_ + 1)] += weight_[3];
  return std::max_element(std::begin(ids), std::end(ids),
                          [](const auto& p1, const auto& p2) {
                            return p1.second < p2.second;
                          })
      ->first;
}

float InterpolatorBilinear::interpolateDepth(
    const Eigen::MatrixXf& range_image) {
  return range_image(v_, u_) * weight_[0] +
         range_image(v_ + 1, u_) * weight_[1] +
         range_image(v_, u_ + 1) * weight_[2] +
         range_image(v_ + 1, u_ + 1) * weight_[3];
}

Color InterpolatorBilinear::interpolateColor(const cv::Mat& color_image) {
  Eigen::Vector3f color(0, 0, 0);
  auto c1 = color_image.at<cv::Vec3b>(v_, u_);
  auto c2 = color_image.at<cv::Vec3b>(v_ + 1, u_);
  auto c3 = color_image.at<cv::Vec3b>(v_, u_ + 1);
  auto c4 = color_image.at<cv::Vec3b>(v_ + 1, u_ + 1);
  for (size_t i = 0; i < 3; ++i) {
    color[i] = c1[i] * weight_[0] + c2[i] * weight_[1] + c3[i] * weight_[2] +
               c4[i] * weight_[3];
  }
  return Color(color[2], color[1], color[0]);  // bgr image
}

int InterpolatorAdaptive::computeWeights(float u, float v,
                                         const Eigen::MatrixXf& range_image,
                                         const cv::Mat& id_image) {
  constexpr float max_depth_difference = 0.2;  // m
  u_ = std::floor(u);
  v_ = std::floor(v);
  use_bilinear_ = true;

  // Check max depth difference.
  if (use_bilinear_) {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    for (size_t i = 0; i < 4; ++i) {
      const float depth = range_image(v_ + v_offset_[i], u_ + u_offset_[i]);
      if (depth > max) {
        max = depth;
      } else if (depth < min) {
        min = depth;
      }
    }
    if (max - min < max_depth_difference) {
      return InterpolatorBilinear::computeWeights(u, v, range_image, id_image);
    } else {
      use_bilinear_ = false;
    }
  }
  return id_image.at<int>(std::round(v), std::round(u));
}

float InterpolatorAdaptive::interpolateDepth(
    const Eigen::MatrixXf& range_image) {
  if (use_bilinear_) {
    return InterpolatorBilinear::interpolateDepth(range_image);
  }
  return range_image(v_, u_);
}

Color InterpolatorAdaptive::interpolateColor(const cv::Mat& color_image) {
  if (use_bilinear_) {
    return InterpolatorBilinear::interpolateColor(color_image);
  }
  auto color_bgr = color_image.at<cv::Vec3b>(v_, u_);
  return Color(color_bgr[2], color_bgr[1], color_bgr[0]);
}

}  // namespace panoptic_mapping
