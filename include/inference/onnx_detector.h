#pragma once

#include "common/types.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

namespace tracker {

class OnnxDetector {
public:
    OnnxDetector();
    ~OnnxDetector();

    // Non-copyable
    OnnxDetector(const OnnxDetector&) = delete;
    OnnxDetector& operator=(const OnnxDetector&) = delete;

    // Initialize ONNX Runtime session with CUDA EP.
    // model_path: path to .onnx file
    // input_size: model input dimension (e.g. 640)
    bool initialize(const char* model_path, int input_size,
                    float conf_threshold, int target_class);

    // Run inference on a BGRA image buffer.
    // Returns detections matching `target_class_`.
    std::vector<Detection> detect(const uint8_t* bgra_data,
                                  int img_width, int img_height,
                                  int img_stride);

private:
    // Pre-process: resize + normalize + HWC->CHW + BGR->RGB
    void preprocess(const uint8_t* bgra_data,
                    int img_width, int img_height, int img_stride,
                    std::vector<float>& blob);

    // Post-process YOLOv8 raw output -> detections
    std::vector<Detection> postprocess(const float* output_data,
                                       const std::vector<int64_t>& output_shape,
                                       int orig_w, int orig_h);

    // Non-Maximum Suppression
    static std::vector<Detection> nms(std::vector<Detection>& dets,
                                      float iou_threshold);

    int          input_size_      = 640;
    float        conf_threshold_  = 0.45f;
    int          target_class_    = 0;
    float        nms_threshold_   = 0.45f;

    std::unique_ptr<Ort::Env>             env_;
    std::unique_ptr<Ort::SessionOptions>  session_opts_;
    std::unique_ptr<Ort::Session>         session_;

    std::vector<const char*>  input_names_;
    std::vector<const char*>  output_names_;
    std::vector<std::string>  input_names_str_;
    std::vector<std::string>  output_names_str_;
};

} // namespace tracker
