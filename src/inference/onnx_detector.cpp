#include "inference/onnx_detector.h"

#ifdef _WIN32
#include <Windows.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace tracker {

OnnxDetector::OnnxDetector() = default;
OnnxDetector::~OnnxDetector() = default;

bool OnnxDetector::initialize(const char* model_path, int input_size,
                               float conf_threshold, int target_class) {
    input_size_     = input_size;
    conf_threshold_ = conf_threshold;
    target_class_   = target_class;

    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                          "RealtimeTracker");

        session_opts_ = std::make_unique<Ort::SessionOptions>();
        session_opts_->SetIntraOpNumThreads(1);
        session_opts_->SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
        // Append CUDA Execution Provider
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id                 = 0;
        cuda_opts.arena_extend_strategy     = 0;
        cuda_opts.gpu_mem_limit             = 0;  // unlimited
        cuda_opts.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchDefault;
        cuda_opts.do_copy_in_default_stream = 1;
        session_opts_->AppendExecutionProvider_CUDA(cuda_opts);
        std::printf("[OnnxDetector] CUDA Execution Provider enabled\n");
#endif

#ifdef USE_TENSORRT
        OrtTensorRTProviderOptions trt_opts;
        trt_opts.device_id             = 0;
        trt_opts.trt_max_workspace_size = static_cast<size_t>(1) << 30;
        trt_opts.trt_fp16_enable        = 1;
        session_opts_->AppendExecutionProvider_TensorRT(trt_opts);
        std::printf("[OnnxDetector] TensorRT Execution Provider enabled\n");
#endif

#ifdef _WIN32
        // Convert to wide string for Windows
        const int len = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, nullptr, 0);
        std::wstring wpath(static_cast<size_t>(len), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wpath.data(), len);
        session_ = std::make_unique<Ort::Session>(*env_, wpath.c_str(),
                                                   *session_opts_);
#else
        session_ = std::make_unique<Ort::Session>(*env_, model_path,
                                                   *session_opts_);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        // Input names
        const size_t num_inputs = session_->GetInputCount();
        input_names_str_.resize(num_inputs);
        input_names_.resize(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_str_[i] = name.get();
            input_names_[i]     = input_names_str_[i].c_str();
        }

        // Output names
        const size_t num_outputs = session_->GetOutputCount();
        output_names_str_.resize(num_outputs);
        output_names_.resize(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_str_[i] = name.get();
            output_names_[i]     = output_names_str_[i].c_str();
        }

        std::printf("[OnnxDetector] Model loaded: %s (%zu inputs, %zu outputs)\n",
                    model_path, num_inputs, num_outputs);
        return true;

    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "[OnnxDetector] ONNX Runtime error: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[OnnxDetector] Initialization error: %s\n", e.what());
        return false;
    } catch (...) {
        std::fprintf(stderr, "[OnnxDetector] Unknown initialization error\n");
        return false;
    }
}

void OnnxDetector::preprocess(const uint8_t* bgra_data,
                               int img_width, int img_height, int img_stride,
                               std::vector<float>& blob) {
    // Wrap BGRA data in a cv::Mat (no copy)
    cv::Mat bgra(img_height, img_width, CV_8UC4,
                 const_cast<uint8_t*>(bgra_data), img_stride);

    // BGRA -> RGB
    cv::Mat rgb;
    cv::cvtColor(bgra, rgb, cv::COLOR_BGRA2RGB);

    // Resize to model input size (letterbox)
    cv::Mat resized;
    const float scale = std::min(
        static_cast<float>(input_size_) / static_cast<float>(img_width),
        static_cast<float>(input_size_) / static_cast<float>(img_height)
    );
    const int new_w = static_cast<int>(std::round(static_cast<float>(img_width)  * scale));
    const int new_h = static_cast<int>(std::round(static_cast<float>(img_height) * scale));
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Pad to input_size_ x input_size_
    cv::Mat padded(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    const int dx = (input_size_ - new_w) / 2;
    const int dy = (input_size_ - new_h) / 2;
    resized.copyTo(padded(cv::Rect(dx, dy, new_w, new_h)));

    // HWC -> CHW, normalize to [0,1]
    blob.resize(static_cast<size_t>(3 * input_size_ * input_size_));
    const int area = input_size_ * input_size_;
    for (int y = 0; y < input_size_; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < input_size_; ++x) {
            const int idx = y * input_size_ + x;
            blob[0 * area + idx] = static_cast<float>(row[x * 3 + 0]) / 255.0f; // R
            blob[1 * area + idx] = static_cast<float>(row[x * 3 + 1]) / 255.0f; // G
            blob[2 * area + idx] = static_cast<float>(row[x * 3 + 2]) / 255.0f; // B
        }
    }
}

std::vector<Detection> OnnxDetector::detect(const uint8_t* bgra_data,
                                             int img_width, int img_height,
                                             int img_stride) {
    if (!session_) return {};

    try {
        // Pre-process
        std::vector<float> blob;
        preprocess(bgra_data, img_width, img_height, img_stride, blob);

        // Create input tensor
        std::array<int64_t, 4> input_shape = {1, 3, input_size_, input_size_};
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, blob.data(), blob.size(),
            input_shape.data(), input_shape.size());

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), &input_tensor, 1,
            output_names_.data(), output_names_.size());

        // Extract output
        const float* output_data = output_tensors[0].GetTensorData<float>();
        auto shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto output_shape = shape_info.GetShape();

        return postprocess(output_data, output_shape, img_width, img_height);
    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "[OnnxDetector] Inference error: %s\n", e.what());
        return {};
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[OnnxDetector] Detection error: %s\n", e.what());
        return {};
    }
}

std::vector<Detection> OnnxDetector::postprocess(
        const float* output_data,
        const std::vector<int64_t>& output_shape,
        int orig_w, int orig_h) {
    // YOLOv8 output shape: [1, num_classes+4, num_boxes]
    // Rows 0-3: cx, cy, w, h (in model input coordinates)
    // Rows 4+: class confidences
    std::vector<Detection> detections;

    if (output_shape.size() < 3) return detections;

    const int64_t num_features = output_shape[1]; // 4 + num_classes
    const int64_t num_boxes    = output_shape[2];
    const int64_t num_classes  = num_features - 4;

    if (num_classes <= 0) return detections;

    // Scale factors (letterbox -> original image)
    const float scale = std::min(
        static_cast<float>(input_size_) / static_cast<float>(orig_w),
        static_cast<float>(input_size_) / static_cast<float>(orig_h)
    );
    const float pad_x = (static_cast<float>(input_size_) - static_cast<float>(orig_w) * scale) * 0.5f;
    const float pad_y = (static_cast<float>(input_size_) - static_cast<float>(orig_h) * scale) * 0.5f;

    for (int64_t i = 0; i < num_boxes; ++i) {
        // Find max class score
        float max_score = 0.0f;
        int   max_cls   = 0;
        for (int64_t c = 0; c < num_classes; ++c) {
            const float score = output_data[(4 + c) * num_boxes + i];
            if (score > max_score) {
                max_score = score;
                max_cls   = static_cast<int>(c);
            }
        }

        if (max_score < conf_threshold_) continue;
        if (max_cls != target_class_)    continue;

        // Extract box in model coordinates
        float cx = output_data[0 * num_boxes + i];
        float cy = output_data[1 * num_boxes + i];
        float w  = output_data[2 * num_boxes + i];
        float h  = output_data[3 * num_boxes + i];

        // Remove padding and rescale to original image coordinates
        cx = (cx - pad_x) / scale;
        cy = (cy - pad_y) / scale;
        w  = w / scale;
        h  = h / scale;

        Detection det;
        det.x          = cx;
        det.y          = cy;
        det.w          = w;
        det.h          = h;
        det.confidence = max_score;
        det.class_id   = max_cls;
        detections.push_back(det);
    }

    return nms(detections, nms_threshold_);
}

std::vector<Detection> OnnxDetector::nms(std::vector<Detection>& dets,
                                          float iou_threshold) {
    if (dets.empty()) return {};

    // Sort by confidence descending
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);

        const float x1_a = dets[i].x - dets[i].w * 0.5f;
        const float y1_a = dets[i].y - dets[i].h * 0.5f;
        const float x2_a = dets[i].x + dets[i].w * 0.5f;
        const float y2_a = dets[i].y + dets[i].h * 0.5f;
        const float area_a = dets[i].w * dets[i].h;

        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;

            const float x1_b = dets[j].x - dets[j].w * 0.5f;
            const float y1_b = dets[j].y - dets[j].h * 0.5f;
            const float x2_b = dets[j].x + dets[j].w * 0.5f;
            const float y2_b = dets[j].y + dets[j].h * 0.5f;
            const float area_b = dets[j].w * dets[j].h;

            const float ix1 = std::max(x1_a, x1_b);
            const float iy1 = std::max(y1_a, y1_b);
            const float ix2 = std::min(x2_a, x2_b);
            const float iy2 = std::min(y2_a, y2_b);

            const float inter = std::max(0.0f, ix2 - ix1) *
                                std::max(0.0f, iy2 - iy1);
            const float iou = inter / (area_a + area_b - inter + 1e-6f);

            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

} // namespace tracker
