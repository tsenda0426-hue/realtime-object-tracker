// ====================================================================
//  RT-DETR Realtime Adaptive Input Correction System
//  Single-file architecture: DXGI + RT-DETR(ONNX/CUDA) + Kalman + ViGEm
//
//  4 concurrent modules:
//    Thread 1 - DXGI Desktop Duplication capture (center 400x400)
//    Thread 2 - RT-DETR ONNX Runtime inference (CUDA EP)
//    Thread 3 - Kalman Filter tracking / prediction
//    Thread 4 - 1000 Hz I/O loop (XInput passthrough + AI right-stick)
// ====================================================================

// ── Standard headers ────────────────────────────────────────────────
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// ── ONNX Runtime C++ API ────────────────────────────────────────────
#include <onnxruntime_cxx_api.h>

// ── OpenCV (resize / color conversion only) ─────────────────────────
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// ── Platform-specific ───────────────────────────────────────────────
#ifdef _WIN32
  #include <Windows.h>
  #include <d3d11.h>
  #include <dxgi1_2.h>
  #include <wrl/client.h>
  #include <Xinput.h>
  #include <ViGEm/Client.h>
#endif

// ====================================================================
//  Global shutdown flag
// ====================================================================
namespace {
std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running.store(false, std::memory_order_release);
}
} // namespace

// ====================================================================
//  Lock-free SPSC Ring Buffer (cache-line aligned, power-of-2)
// ====================================================================
template <typename T, std::size_t Capacity>
class RingBuffer {
    static_assert(Capacity > 0 && (Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");
public:
    bool try_push(T item) {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t next = (h + 1) & (Capacity - 1);
        if (next == tail_.load(std::memory_order_acquire)) return false;
        buf_[h] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    std::optional<T> try_pop() {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return std::nullopt;
        T item = std::move(buf_[t]);
        tail_.store((t + 1) & (Capacity - 1), std::memory_order_release);
        return item;
    }

private:
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
    std::array<T, Capacity>              buf_{};
};

// ====================================================================
//  Data types
// ====================================================================
using SteadyClock = std::chrono::steady_clock;
using TimePoint   = SteadyClock::time_point;

struct Detection {
    float x;           // bbox center x (pixels, capture coords)
    float y;           // bbox center y
    float w;           // bbox width
    float h;           // bbox height
    float confidence;
    int   class_id;
};

struct CapturedFrame {
    std::vector<uint8_t> pixels;  // BGRA 8-bit
    int      width   = 0;
    int      height  = 0;
    int      stride  = 0;        // bytes per row
    uint64_t frame_id = 0;
    TimePoint timestamp{};
};

struct DetectionResult {
    std::vector<Detection> detections;
    uint64_t               frame_id = 0;
    TimePoint              timestamp{};
};

struct GamepadState {
    uint16_t buttons       = 0;
    uint8_t  left_trigger  = 0;
    uint8_t  right_trigger = 0;
    int16_t  lx = 0, ly = 0;
    int16_t  rx = 0, ry = 0;
};

struct TrackedTarget {
    float x  = 0, y  = 0;
    float vx = 0, vy = 0;
    bool  valid = false;
    int   frames_lost = 0;
};

// ====================================================================
//  System configuration (all tuning constants in one place)
// ====================================================================
struct SystemConfig {
    // Capture region (centered on screen)
    int capture_width   = 400;
    int capture_height  = 400;
    int capture_offset_x = 0;   // computed at runtime
    int capture_offset_y = 0;

    // RT-DETR inference
    const char* model_path     = "rtdetr.onnx";
    int   infer_input_size     = 640;
    float confidence_threshold = 0.45f;
    int   target_class_id      = 0;      // COCO person=0

    // Kalman filter
    int   max_lost_frames  = 20;
    int   prediction_steps = 2;

    // I/O control loop
    float smoothing_alpha       = 0.35f;   // base EMA alpha
    float decay_exponent        = 2.0f;    // pow decay curve
    float max_correction_speed  = 25000.0f;
    float deadzone_radius       = 3.0f;
};

// ====================================================================
//  Module 1: DXGI Desktop Duplication Capture
// ====================================================================
class DxgiCapture {
public:
    DxgiCapture() = default;
    ~DxgiCapture() { shutdown(); }
    DxgiCapture(const DxgiCapture&) = delete;
    DxgiCapture& operator=(const DxgiCapture&) = delete;

    int screen_width()  const { return screen_w_; }
    int screen_height() const { return screen_h_; }

    bool initialize(int region_w, int region_h) {
        region_w_ = region_w;
        region_h_ = region_h;
#ifdef _WIN32
        if (!create_device())     return false;
        if (!create_duplication()) return false;

        // Center on screen
        offset_x_ = std::max(0, (screen_w_ - region_w_) / 2);
        offset_y_ = std::max(0, (screen_h_ - region_h_) / 2);

        if (!create_staging()) return false;

        std::printf("[DXGI] Screen %dx%d  region (%d,%d)+%dx%d\n",
                    screen_w_, screen_h_, offset_x_, offset_y_,
                    region_w_, region_h_);
        return true;
#else
        std::printf("[DXGI] Stub (non-Windows)\n");
        return true;
#endif
    }

    void shutdown() {
#ifdef _WIN32
        duplication_.Reset();
#endif
    }

    bool acquire_frame(CapturedFrame& out) {
#ifdef _WIN32
        if (!duplication_) {
            if (!create_duplication() || !create_staging()) return false;
        }

        Microsoft::WRL::ComPtr<IDXGIResource> res;
        DXGI_OUTDUPL_FRAME_INFO fi;
        HRESULT hr = duplication_->AcquireNextFrame(1, &fi, &res);

        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return false;
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            duplication_.Reset(); staging_.Reset();
            return false;
        }
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
            duplication_.Reset(); staging_.Reset();
            device_.Reset(); context_.Reset();
            create_device(); create_duplication(); create_staging();
            return false;
        }
        if (FAILED(hr)) { duplication_.Reset(); staging_.Reset(); return false; }

        Microsoft::WRL::ComPtr<ID3D11Texture2D> desktop;
        hr = res.As(&desktop);
        if (FAILED(hr)) { duplication_->ReleaseFrame(); return false; }

        D3D11_BOX box{};
        box.left   = static_cast<UINT>(offset_x_);
        box.top    = static_cast<UINT>(offset_y_);
        box.right  = static_cast<UINT>(offset_x_ + region_w_);
        box.bottom = static_cast<UINT>(offset_y_ + region_h_);
        box.front  = 0;
        box.back   = 1;
        context_->CopySubresourceRegion(staging_.Get(), 0, 0, 0, 0,
                                        desktop.Get(), 0, &box);

        D3D11_MAPPED_SUBRESOURCE mapped;
        hr = context_->Map(staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) { duplication_->ReleaseFrame(); return false; }

        out.width    = region_w_;
        out.height   = region_h_;
        out.stride   = static_cast<int>(mapped.RowPitch);
        out.frame_id = ++frame_counter_;
        out.timestamp = SteadyClock::now();

        const std::size_t bytes = static_cast<std::size_t>(out.stride) *
                                  static_cast<std::size_t>(region_h_);
        out.pixels.resize(bytes);
        std::memcpy(out.pixels.data(), mapped.pData, bytes);

        context_->Unmap(staging_.Get(), 0);
        duplication_->ReleaseFrame();
        return true;
#else
        (void)out;
        return false;
#endif
    }

private:
    int region_w_ = 400, region_h_ = 400;
    int offset_x_ = 0, offset_y_ = 0;
    int screen_w_ = 0, screen_h_ = 0;
    uint64_t frame_counter_ = 0;

#ifdef _WIN32
    Microsoft::WRL::ComPtr<ID3D11Device>          device_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext>    context_;
    Microsoft::WRL::ComPtr<IDXGIOutputDuplication> duplication_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D>        staging_;

    bool create_device() {
        D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL out_level;
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE,
                        nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                        levels, 1, D3D11_SDK_VERSION,
                        &device_, &out_level, &context_);
        if (FAILED(hr)) {
            std::fprintf(stderr, "[DXGI] D3D11CreateDevice failed: 0x%08lx\n", hr);
            return false;
        }
        return true;
    }

    bool create_duplication() {
        Microsoft::WRL::ComPtr<IDXGIDevice>  ddev;
        Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
        Microsoft::WRL::ComPtr<IDXGIOutput>  output;
        Microsoft::WRL::ComPtr<IDXGIOutput1> output1;

        if (FAILED(device_.As(&ddev)))            return false;
        if (FAILED(ddev->GetAdapter(&adapter)))   return false;
        if (FAILED(adapter->EnumOutputs(0, &output))) return false;

        DXGI_OUTPUT_DESC desc;
        output->GetDesc(&desc);
        screen_w_ = desc.DesktopCoordinates.right  - desc.DesktopCoordinates.left;
        screen_h_ = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;

        if (FAILED(output.As(&output1))) return false;
        HRESULT hr = output1->DuplicateOutput(device_.Get(), &duplication_);
        if (FAILED(hr)) {
            std::fprintf(stderr, "[DXGI] DuplicateOutput failed: 0x%08lx\n", hr);
            return false;
        }
        return true;
    }

    bool create_staging() {
        D3D11_TEXTURE2D_DESC td{};
        td.Width            = static_cast<UINT>(region_w_);
        td.Height           = static_cast<UINT>(region_h_);
        td.MipLevels        = 1;
        td.ArraySize        = 1;
        td.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage            = D3D11_USAGE_STAGING;
        td.CPUAccessFlags   = D3D11_CPU_ACCESS_READ;

        HRESULT hr = device_->CreateTexture2D(&td, nullptr, &staging_);
        if (FAILED(hr)) {
            std::fprintf(stderr, "[DXGI] CreateTexture2D staging: 0x%08lx\n", hr);
            return false;
        }
        return true;
    }
#endif
};

// ====================================================================
//  Module 2: RT-DETR ONNX Runtime Detector (CUDA EP)
//
//  RT-DETR output: NMS-free, Transformer-based detection head.
//  Supported output formats:
//    A) Ultralytics export:  [1, 300, 6]  = [x1,y1,x2,y2, score, class_id]
//    B) PaddleDetection:     [1, 300, 6]  = [class_id, score, x1,y1,x2,y2]
//    C) Raw logits:          [1, 300, 4+C] = [cx,cy,w,h, cls0..clsC-1]
//  Auto-detected at runtime from tensor shapes.
// ====================================================================
class RtDetrDetector {
public:
    RtDetrDetector() = default;
    ~RtDetrDetector() = default;
    RtDetrDetector(const RtDetrDetector&) = delete;
    RtDetrDetector& operator=(const RtDetrDetector&) = delete;

    bool initialize(const char* model_path, int input_size,
                    float conf_threshold, int target_class) {
        input_size_     = input_size;
        conf_threshold_ = conf_threshold;
        target_class_   = target_class;

        try {
            env_ = std::make_unique<Ort::Env>(
                ORT_LOGGING_LEVEL_WARNING, "RT_DETR_Tracker");

            opts_ = std::make_unique<Ort::SessionOptions>();
            opts_->SetIntraOpNumThreads(1);
            opts_->SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL);

            // ── CUDA Execution Provider (optimised) ─────────────
#ifdef USE_CUDA
            try {
                OrtCUDAProviderOptions cuda_opts{};
                cuda_opts.device_id                 = 0;
                cuda_opts.arena_extend_strategy     = 1; // kSameAsRequested
                cuda_opts.gpu_mem_limit             = 0; // unlimited
                cuda_opts.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchExhaustive;
                cuda_opts.do_copy_in_default_stream = 1;
                cuda_opts.has_user_compute_stream   = 0;
                cuda_opts.default_memory_arena_cfg  = nullptr;
                cuda_opts.tunable_op_enable         = 0;
                cuda_opts.tunable_op_tuning_enable  = 0;
                opts_->AppendExecutionProvider_CUDA(cuda_opts);
                std::printf("[RT-DETR] CUDA EP appended\n");
            } catch (const Ort::Exception& e) {
                std::fprintf(stderr,
                    "[RT-DETR] CUDA EP unavailable (%s), falling back to CPU\n",
                    e.what());
            }
#endif

#ifdef USE_TENSORRT
            try {
                OrtTensorRTProviderOptions trt{};
                trt.device_id              = 0;
                trt.trt_max_workspace_size = size_t(1) << 30;
                trt.trt_fp16_enable        = 1;
                opts_->AppendExecutionProvider_TensorRT(trt);
                std::printf("[RT-DETR] TensorRT EP appended\n");
            } catch (const Ort::Exception& e) {
                std::fprintf(stderr,
                    "[RT-DETR] TensorRT EP unavailable: %s\n", e.what());
            }
#endif

            // ── Load model ──────────────────────────────────────
#ifdef _WIN32
            const int len = MultiByteToWideChar(
                CP_UTF8, 0, model_path, -1, nullptr, 0);
            std::wstring wpath(static_cast<std::size_t>(len), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, model_path, -1,
                                wpath.data(), len);
            session_ = std::make_unique<Ort::Session>(
                *env_, wpath.c_str(), *opts_);
#else
            session_ = std::make_unique<Ort::Session>(
                *env_, model_path, *opts_);
#endif

            Ort::AllocatorWithDefaultOptions alloc;

            // Input names
            const std::size_t ni = session_->GetInputCount();
            in_names_str_.resize(ni);
            in_names_.resize(ni);
            for (std::size_t i = 0; i < ni; ++i) {
                auto n = session_->GetInputNameAllocated(i, alloc);
                in_names_str_[i] = n.get();
                in_names_[i]     = in_names_str_[i].c_str();
            }

            // Output names
            const std::size_t no = session_->GetOutputCount();
            out_names_str_.resize(no);
            out_names_.resize(no);
            for (std::size_t i = 0; i < no; ++i) {
                auto n = session_->GetOutputNameAllocated(i, alloc);
                out_names_str_[i] = n.get();
                out_names_[i]     = out_names_str_[i].c_str();
            }

            // Auto-detect input size from model
            if (ni > 0) {
                auto ti = session_->GetInputTypeInfo(0);
                auto si = ti.GetTensorTypeAndShapeInfo();
                auto shape = si.GetShape();
                if (shape.size() >= 4 && shape[2] > 0 && shape[3] > 0) {
                    int mh = static_cast<int>(shape[2]);
                    int mw = static_cast<int>(shape[3]);
                    if (mh == mw && mh != input_size_) {
                        std::printf("[RT-DETR] Auto-detected input %d "
                                    "(was %d)\n", mh, input_size_);
                        input_size_ = mh;
                    }
                }
                std::printf("[RT-DETR] Input shape: [");
                for (std::size_t s = 0; s < shape.size(); ++s)
                    std::printf("%lld%s",
                        static_cast<long long>(shape[s]),
                        s + 1 < shape.size() ? "," : "");
                std::printf("]\n");
            }

            std::printf("[RT-DETR] Model loaded: %s  (%zu in, %zu out)\n",
                        model_path, ni, no);

            // Print output shapes for diagnostics
            for (std::size_t i = 0; i < no; ++i) {
                auto ti = session_->GetOutputTypeInfo(i);
                auto si = ti.GetTensorTypeAndShapeInfo();
                auto shape = si.GetShape();
                std::printf("[RT-DETR] Output[%zu] \"%s\" shape=[",
                            i, out_names_[i]);
                for (std::size_t s = 0; s < shape.size(); ++s)
                    std::printf("%lld%s",
                        static_cast<long long>(shape[s]),
                        s + 1 < shape.size() ? "," : "");
                std::printf("]\n");
            }

            return true;
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr, "[RT-DETR] ONNX error: %s\n", e.what());
            return false;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[RT-DETR] Init error: %s\n", e.what());
            return false;
        }
    }

    std::vector<Detection> detect(const uint8_t* bgra,
                                  int img_w, int img_h, int img_stride) {
        if (!session_) return {};

        try {
            // ── Pre-process: BGRA→RGB, resize, normalize, HWC→CHW ──
            std::vector<float> blob;
            preprocess(bgra, img_w, img_h, img_stride, blob);

            std::array<int64_t, 4> in_shape = {
                1, 3, input_size_, input_size_};
            auto mem = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
                mem, blob.data(), blob.size(),
                in_shape.data(), in_shape.size());

            // ── Run ─────────────────────────────────────────────
            auto out_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                in_names_.data(), &in_tensor, 1,
                out_names_.data(), out_names_.size());

            // ── Post-process (RT-DETR: NMS-free) ────────────────
            return postprocess_rtdetr(out_tensors, img_w, img_h);
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr, "[RT-DETR] Inference: %s\n", e.what());
            return {};
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[RT-DETR] Detect: %s\n", e.what());
            return {};
        }
    }

private:
    int   input_size_      = 640;
    float conf_threshold_  = 0.45f;
    int   target_class_    = 0;

    std::unique_ptr<Ort::Env>            env_;
    std::unique_ptr<Ort::SessionOptions> opts_;
    std::unique_ptr<Ort::Session>        session_;

    std::vector<const char*>  in_names_;
    std::vector<const char*>  out_names_;
    std::vector<std::string>  in_names_str_;
    std::vector<std::string>  out_names_str_;

    // ── Pre-processing ──────────────────────────────────────────
    void preprocess(const uint8_t* bgra,
                    int w, int h, int stride,
                    std::vector<float>& blob) {
        cv::Mat src(h, w, CV_8UC4, const_cast<uint8_t*>(bgra), stride);
        cv::Mat rgb;
        cv::cvtColor(src, rgb, cv::COLOR_BGRA2RGB);

        // Letterbox resize
        const float scale = std::min(
            static_cast<float>(input_size_) / static_cast<float>(w),
            static_cast<float>(input_size_) / static_cast<float>(h));
        const int nw = static_cast<int>(std::round(
            static_cast<float>(w) * scale));
        const int nh = static_cast<int>(std::round(
            static_cast<float>(h) * scale));

        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

        cv::Mat padded(input_size_, input_size_, CV_8UC3,
                       cv::Scalar(114, 114, 114));
        const int dx = (input_size_ - nw) / 2;
        const int dy = (input_size_ - nh) / 2;
        resized.copyTo(padded(cv::Rect(dx, dy, nw, nh)));

        // HWC → CHW, normalise to [0,1]
        const int area = input_size_ * input_size_;
        blob.resize(static_cast<std::size_t>(3 * area));
        for (int y = 0; y < input_size_; ++y) {
            const auto* row = padded.ptr<uint8_t>(y);
            for (int x = 0; x < input_size_; ++x) {
                const int idx = y * input_size_ + x;
                blob[0 * area + idx] = static_cast<float>(row[x*3+0]) / 255.0f;
                blob[1 * area + idx] = static_cast<float>(row[x*3+1]) / 255.0f;
                blob[2 * area + idx] = static_cast<float>(row[x*3+2]) / 255.0f;
            }
        }
    }

    // ── RT-DETR post-processing (NMS-free) ──────────────────────
    //
    // Handles three common RT-DETR ONNX export layouts:
    //
    //  Format A  (Ultralytics)
    //    Single output [1, N, 6]:  x1, y1, x2, y2, score, class_id
    //    Coordinates are in original-image pixel space.
    //
    //  Format B  (PaddleDetection)
    //    Single output [1, N, 6]:  class_id, score, x1, y1, x2, y2
    //    Identified by column-0 values being small integers (class ids).
    //
    //  Format C  (Raw decoder logits, no built-in post-proc)
    //    Single output [1, N, 4+C]:  cx, cy, w, h, cls0 … clsC-1
    //    C > 2  ⇒  raw logits  (apply sigmoid to class scores).
    //
    // ────────────────────────────────────────────────────────────
    std::vector<Detection> postprocess_rtdetr(
            std::vector<Ort::Value>& outputs,
            int orig_w, int orig_h) {

        std::vector<Detection> dets;

        if (outputs.empty()) return dets;

        const float* data = outputs[0].GetTensorData<float>();
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        // Expect rank-3: [batch, num_queries, cols]
        if (shape.size() < 2) return dets;

        int64_t num_queries, cols;
        if (shape.size() == 3) {
            num_queries = shape[1];
            cols        = shape[2];
        } else {
            num_queries = shape[0];
            cols        = shape[1];
        }

        // Letterbox scale factors for coordinate mapping
        const float scale = std::min(
            static_cast<float>(input_size_) / static_cast<float>(orig_w),
            static_cast<float>(input_size_) / static_cast<float>(orig_h));
        const float pad_x = (static_cast<float>(input_size_) -
                             static_cast<float>(orig_w) * scale) * 0.5f;
        const float pad_y = (static_cast<float>(input_size_) -
                             static_cast<float>(orig_h) * scale) * 0.5f;

        if (cols == 6) {
            // Detect Format A vs B by inspecting the first row's column-0:
            // Format B has class_id (small int) in col 0.
            // Format A has x1 coordinate (typically > 1.0) in col 0.
            bool is_paddle = false;
            if (num_queries > 0) {
                float col0_val = data[0];
                float col1_val = data[1];
                // Paddle: col0 = class_id (0..999), col1 = score (0..1)
                // Ultra:  col0 = x1 (pixels), col1 = y1 (pixels)
                if (col0_val >= 0.0f && col0_val < 1000.0f &&
                    col1_val >= 0.0f && col1_val <= 1.0f &&
                    col0_val == std::floor(col0_val)) {
                    is_paddle = true;
                }
            }

            for (int64_t i = 0; i < num_queries; ++i) {
                const float* row = data + i * 6;

                float x1, y1, x2, y2, score;
                int cls;

                if (is_paddle) {
                    // Format B: [class_id, score, x1, y1, x2, y2]
                    cls   = static_cast<int>(row[0]);
                    score = row[1];
                    x1    = row[2];
                    y1    = row[3];
                    x2    = row[4];
                    y2    = row[5];
                } else {
                    // Format A: [x1, y1, x2, y2, score, class_id]
                    x1    = row[0];
                    y1    = row[1];
                    x2    = row[2];
                    y2    = row[3];
                    score = row[4];
                    cls   = static_cast<int>(row[5]);
                }

                if (score < conf_threshold_) continue;
                if (target_class_ >= 0 && cls != target_class_) continue;

                // Map from model-input coords to capture-region coords
                float cx = ((x1 + x2) * 0.5f - pad_x) / scale;
                float cy = ((y1 + y2) * 0.5f - pad_y) / scale;
                float bw = (x2 - x1) / scale;
                float bh = (y2 - y1) / scale;

                dets.push_back({cx, cy, bw, bh, score, cls});
            }
        } else if (cols > 6) {
            // Format C: [cx, cy, w, h, cls0 … clsC-1]  (raw logits)
            const int num_classes = static_cast<int>(cols - 4);

            for (int64_t i = 0; i < num_queries; ++i) {
                const float* row = data + i * cols;
                float raw_cx = row[0];
                float raw_cy = row[1];
                float raw_w  = row[2];
                float raw_h  = row[3];

                // Find best class via sigmoid
                int   best_cls   = 0;
                float best_score = -1e9f;
                for (int c = 0; c < num_classes; ++c) {
                    float logit = row[4 + c];
                    float sig = 1.0f / (1.0f + std::exp(-logit));
                    if (sig > best_score) {
                        best_score = sig;
                        best_cls   = c;
                    }
                }

                if (best_score < conf_threshold_) continue;
                if (target_class_ >= 0 && best_cls != target_class_) continue;

                // Raw coords might be normalised [0,1] or pixel-space
                float cx_px = raw_cx, cy_px = raw_cy;
                float w_px  = raw_w,  h_px  = raw_h;
                if (raw_cx <= 1.0f && raw_cy <= 1.0f &&
                    raw_w  <= 1.0f && raw_h  <= 1.0f) {
                    // Normalised → pixel
                    cx_px = raw_cx * static_cast<float>(input_size_);
                    cy_px = raw_cy * static_cast<float>(input_size_);
                    w_px  = raw_w  * static_cast<float>(input_size_);
                    h_px  = raw_h  * static_cast<float>(input_size_);
                }

                // Map from model-input to capture-region coords
                float out_cx = (cx_px - pad_x) / scale;
                float out_cy = (cy_px - pad_y) / scale;
                float out_w  = w_px / scale;
                float out_h  = h_px / scale;

                dets.push_back({out_cx, out_cy, out_w, out_h,
                                best_score, best_cls});
            }
        }

        // Sort by confidence descending → best detection first
        std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) {
                return a.confidence > b.confidence;
            });

        return dets;
    }
};

// ====================================================================
//  Module 3: Kalman Filter (6-state constant-acceleration model)
//  State: [x, y, vx, vy, ax, ay]   Measurement: [x, y]
// ====================================================================
class KalmanTracker {
public:
    static constexpr int N = 6;  // state dim
    static constexpr int M = 2;  // meas dim

    KalmanTracker() { reset(); }

    void reset() {
        initialised_ = false;
        lost_ = 0;
        std::memset(x_, 0, sizeof(x_));

        eye(P_);
        for (int i = 0; i < N; ++i) P_[i*N+i] = 100.0f;

        zero(Q_);
        Q_[0*N+0] = 0.5f;   Q_[1*N+1] = 0.5f;   // pos
        Q_[2*N+2] = 10.0f;  Q_[3*N+3] = 10.0f;   // vel
        Q_[4*N+4] = 50.0f;  Q_[5*N+5] = 50.0f;   // accel

        zero(R_);
        R_[0] = 2.0f;  R_[3] = 2.0f;
    }

    bool is_init() const { return initialised_; }
    int  lost()    const { return lost_; }
    void inc_lost()      { ++lost_; }
    void rst_lost()      { lost_ = 0; }

    void predict(double dt) {
        if (!initialised_) return;

        const float t  = static_cast<float>(dt);
        const float t2 = 0.5f * t * t;

        float F[N*N];
        eye(F);
        F[0*N+2] = t;  F[0*N+4] = t2;   // x  ← vx, ax
        F[1*N+3] = t;  F[1*N+5] = t2;   // y  ← vy, ay
        F[2*N+4] = t;                     // vx ← ax
        F[3*N+5] = t;                     // vy ← ay

        float xp[N];
        mul(F, x_, xp, N, N, 1);

        float Ft[N*N]; trans(F, Ft, N, N);
        float FP[N*N]; mul(F, P_, FP, N, N, N);
        float FPFt[N*N]; mul(FP, Ft, FPFt, N, N, N);
        add(FPFt, Q_, P_, N, N);
        std::memcpy(x_, xp, sizeof(x_));
    }

    void update(float mx, float my) {
        if (!initialised_) {
            std::memset(x_, 0, sizeof(x_));
            x_[0] = mx; x_[1] = my;
            eye(P_);
            for (int i = 0; i < N; ++i) P_[i*N+i] = 100.0f;
            initialised_ = true;
            return;
        }

        float H[M*N]; zero(H);
        H[0*N+0] = 1.0f;  H[1*N+1] = 1.0f;

        float z[M] = {mx, my};
        float Hx[M]; mul(H, x_, Hx, M, N, 1);
        float y[M]; sub(z, Hx, y, M, 1);

        float Ht[N*M]; trans(H, Ht, M, N);
        float HP[M*N]; mul(H, P_, HP, M, N, N);
        float HPHt[M*M]; mul(HP, Ht, HPHt, M, N, M);
        float S[M*M]; add(HPHt, R_, S, M, M);

        float Si[M*M];
        if (!inv2(S, Si)) return;

        float PHt[N*M]; mul(P_, Ht, PHt, N, N, M);
        float K[N*M]; mul(PHt, Si, K, N, M, M);

        float Ky[N]; mul(K, y, Ky, N, M, 1);
        for (int i = 0; i < N; ++i) x_[i] += Ky[i];

        float KH[N*N]; mul(K, H, KH, N, M, N);
        float I[N*N]; eye(I);
        float IKH[N*N]; sub(I, KH, IKH, N, N);
        float Pn[N*N]; mul(IKH, P_, Pn, N, N, N);
        std::memcpy(P_, Pn, sizeof(P_));
    }

    TrackedTarget state() const {
        return {x_[0], x_[1], x_[2], x_[3], initialised_, lost_};
    }

    TrackedTarget predict_ahead(int steps, double dt) const {
        if (!initialised_) return {0,0,0,0,false,lost_};
        const float t  = static_cast<float>(dt) * static_cast<float>(steps);
        const float t2 = 0.5f * t * t;
        return {
            x_[0] + x_[2]*t + x_[4]*t2,
            x_[1] + x_[3]*t + x_[5]*t2,
            x_[2] + x_[4]*t,
            x_[3] + x_[5]*t,
            true, lost_
        };
    }

private:
    float x_[N]{};
    float P_[N*N]{};
    float Q_[N*N]{};
    float R_[M*M]{};
    bool  initialised_ = false;
    int   lost_ = 0;

    // Tiny matrix helpers (row-major, stack-allocated)
    static void mul(const float* A, const float* B, float* C,
                    int m, int n, int p) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < p; ++j) {
                float s = 0;
                for (int k = 0; k < n; ++k) s += A[i*n+k]*B[k*p+j];
                C[i*p+j] = s;
            }
    }
    static void add(const float* A, const float* B, float* C,
                    int r, int c) {
        for (int i = 0; i < r*c; ++i) C[i] = A[i]+B[i];
    }
    static void sub(const float* A, const float* B, float* C,
                    int r, int c) {
        for (int i = 0; i < r*c; ++i) C[i] = A[i]-B[i];
    }
    static void trans(const float* A, float* At, int r, int c) {
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
                At[j*r+i] = A[i*c+j];
    }
    static bool inv2(const float* M, float* I) {
        float d = M[0]*M[3] - M[1]*M[2];
        if (std::fabs(d) < 1e-12f) return false;
        float id = 1.0f / d;
        I[0] =  M[3]*id; I[1] = -M[1]*id;
        I[2] = -M[2]*id; I[3] =  M[0]*id;
        return true;
    }
    static void eye(float* M) {
        std::memset(M, 0, sizeof(float)*N*N);
        for (int i = 0; i < N; ++i) M[i*N+i] = 1.0f;
    }
    static void zero(float* M) {
        std::memset(M, 0, sizeof(float)*N*N);
    }
};

// ====================================================================
//  Module 4: Gamepad Controller (XInput → ViGEm passthrough)
//
//  On startup, scans XInput ports 0-3 and locks the first physical pad.
//  Creates a virtual Xbox 360 pad via ViGEmBus.
//  The locked port prevents the virtual pad from feeding back into itself.
// ====================================================================
class GamepadController {
public:
    GamepadController() = default;
    ~GamepadController() { shutdown(); }
    GamepadController(const GamepadController&) = delete;
    GamepadController& operator=(const GamepadController&) = delete;

    int  locked_port() const { return locked_port_; }
    bool ready()       const { return ready_; }

    bool initialize() {
#ifdef _WIN32
        // Scan for physical pad
        XINPUT_STATE xs{};
        for (DWORD p = 0; p < XUSER_MAX_COUNT; ++p) {
            if (XInputGetState(p, &xs) == ERROR_SUCCESS) {
                locked_port_ = static_cast<int>(p);
                std::printf("[Gamepad] Physical pad on port %d\n",
                            locked_port_);
                break;
            }
        }
        if (locked_port_ < 0) {
            std::fprintf(stderr, "[Gamepad] No physical XInput pad found\n");
            return false;
        }

        // ViGEm client
        client_ = vigem_alloc();
        if (!client_) {
            std::fprintf(stderr, "[Gamepad] vigem_alloc failed\n");
            return false;
        }
        VIGEM_ERROR err = vigem_connect(client_);
        if (!VIGEM_SUCCESS(err)) {
            std::fprintf(stderr, "[Gamepad] vigem_connect: 0x%08x\n",
                         static_cast<unsigned>(err));
            vigem_free(client_); client_ = nullptr;
            return false;
        }

        // Virtual pad
        target_ = vigem_target_x360_alloc();
        if (!target_) {
            std::fprintf(stderr, "[Gamepad] x360_alloc failed\n");
            shutdown(); return false;
        }
        err = vigem_target_add(client_, target_);
        if (!VIGEM_SUCCESS(err)) {
            std::fprintf(stderr, "[Gamepad] target_add: 0x%08x\n",
                         static_cast<unsigned>(err));
            vigem_target_free(target_); target_ = nullptr;
            shutdown(); return false;
        }

        ready_ = true;
        std::printf("[Gamepad] Virtual Xbox 360 pad created\n");
        return true;
#else
        std::printf("[Gamepad] Stub (non-Windows)\n");
        return true;
#endif
    }

    void shutdown() {
#ifdef _WIN32
        if (target_ && client_) {
            vigem_target_remove(client_, target_);
            vigem_target_free(target_);
            target_ = nullptr;
        }
        if (client_) {
            vigem_disconnect(client_);
            vigem_free(client_);
            client_ = nullptr;
        }
        ready_ = false;
#endif
    }

    bool poll_physical(GamepadState& gs) {
#ifdef _WIN32
        if (locked_port_ < 0) return false;
        XINPUT_STATE xs{};
        if (XInputGetState(static_cast<DWORD>(locked_port_), &xs)
                != ERROR_SUCCESS) return false;
        gs.buttons       = xs.Gamepad.wButtons;
        gs.left_trigger  = xs.Gamepad.bLeftTrigger;
        gs.right_trigger = xs.Gamepad.bRightTrigger;
        gs.lx = xs.Gamepad.sThumbLX;
        gs.ly = xs.Gamepad.sThumbLY;
        gs.rx = xs.Gamepad.sThumbRX;
        gs.ry = xs.Gamepad.sThumbRY;
        return true;
#else
        gs = {};
        return false;
#endif
    }

    bool submit(const GamepadState& phys, int16_t ai_rx, int16_t ai_ry) {
#ifdef _WIN32
        if (!ready_ || !target_) return false;

        XUSB_REPORT rpt;
        XUSB_REPORT_INIT(&rpt);
        rpt.wButtons      = phys.buttons;
        rpt.bLeftTrigger  = phys.left_trigger;
        rpt.bRightTrigger = phys.right_trigger;
        rpt.sThumbLX      = phys.lx;
        rpt.sThumbLY      = phys.ly;

        // Right stick = physical + AI correction (clamped)
        const int32_t crx = static_cast<int32_t>(phys.rx) +
                            static_cast<int32_t>(ai_rx);
        const int32_t cry = static_cast<int32_t>(phys.ry) +
                            static_cast<int32_t>(ai_ry);
        rpt.sThumbRX = static_cast<SHORT>(
            std::clamp(crx, int32_t(-32768), int32_t(32767)));
        rpt.sThumbRY = static_cast<SHORT>(
            std::clamp(cry, int32_t(-32768), int32_t(32767)));

        return VIGEM_SUCCESS(
            vigem_target_x360_update(client_, target_, rpt));
#else
        (void)phys; (void)ai_rx; (void)ai_ry;
        return false;
#endif
    }

private:
    int  locked_port_ = -1;
    bool ready_       = false;
#ifdef _WIN32
    PVIGEM_CLIENT client_ = nullptr;
    PVIGEM_TARGET target_ = nullptr;
#endif
};

// ====================================================================
//  Thread 1: Screen capture (DXGI)
// ====================================================================
static void capture_thread(
        DxgiCapture& cap,
        RingBuffer<CapturedFrame, 4>& q) {
    std::printf("[T-Capture] Started\n");
    int errs = 0;

    while (g_running.load(std::memory_order_acquire)) {
        try {
            CapturedFrame f;
            if (cap.acquire_frame(f)) {
                q.try_push(std::move(f));
                errs = 0;
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        } catch (const std::exception& e) {
            if (++errs > 100) {
                std::fprintf(stderr, "[T-Capture] Fatal: %s\n", e.what());
                g_running.store(false, std::memory_order_release);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } catch (...) {
            if (++errs > 100) {
                g_running.store(false, std::memory_order_release);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    std::printf("[T-Capture] Stopped\n");
}

// ====================================================================
//  Thread 2: RT-DETR inference
// ====================================================================
static void inference_thread(
        RtDetrDetector& det,
        RingBuffer<CapturedFrame, 4>& frame_q,
        RingBuffer<DetectionResult, 4>& det_q) {
    std::printf("[T-Infer] Started\n");
    int errs = 0;

    while (g_running.load(std::memory_order_acquire)) {
        try {
            auto maybe = frame_q.try_pop();
            if (!maybe) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            auto& f = *maybe;
            auto dets = det.detect(f.pixels.data(),
                                   f.width, f.height, f.stride);

            DetectionResult r;
            r.detections = std::move(dets);
            r.frame_id   = f.frame_id;
            r.timestamp  = f.timestamp;
            det_q.try_push(std::move(r));
            errs = 0;
        } catch (const Ort::Exception& e) {
            if (++errs > 50) {
                std::fprintf(stderr, "[T-Infer] Fatal ONNX: %s\n", e.what());
                g_running.store(false, std::memory_order_release);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } catch (const std::exception& e) {
            if (++errs > 50) {
                std::fprintf(stderr, "[T-Infer] Fatal: %s\n", e.what());
                g_running.store(false, std::memory_order_release);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } catch (...) {
            if (++errs > 50) {
                g_running.store(false, std::memory_order_release);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    std::printf("[T-Infer] Stopped\n");
}

// ====================================================================
//  Thread 3: Tracking (Kalman predict / update)
//  Drains detection queue, feeds kalman, publishes predicted target.
// ====================================================================
struct PredictedTarget {
    std::atomic<float> px{0}, py{0};
    std::atomic<float> vx{0}, vy{0};
    std::atomic<bool>  valid{false};
    std::atomic<int>   lost{0};
};

static void tracking_thread(
        KalmanTracker& kf,
        RingBuffer<DetectionResult, 4>& det_q,
        PredictedTarget& target,
        const SystemConfig& cfg) {
    std::printf("[T-Track] Started\n");
    auto prev = SteadyClock::now();
    int errs = 0;

    while (g_running.load(std::memory_order_acquire)) {
        try {
            const auto now = SteadyClock::now();
            const double dt =
                std::chrono::duration<double>(now - prev).count();
            prev = now;
            const double sdt = (dt > 0 && dt < 0.1) ? dt : 0.001;

            // Drain queue, keep newest
            DetectionResult dr;
            bool has = false;
            while (auto m = det_q.try_pop()) { dr = std::move(*m); has = true; }

            if (has && !dr.detections.empty()) {
                kf.update(dr.detections[0].x, dr.detections[0].y);
                kf.rst_lost();
            } else {
                kf.inc_lost();
            }

            kf.predict(sdt);
            auto pred = kf.predict_ahead(cfg.prediction_steps, sdt);

            target.px.store(pred.x, std::memory_order_relaxed);
            target.py.store(pred.y, std::memory_order_relaxed);
            target.vx.store(pred.vx, std::memory_order_relaxed);
            target.vy.store(pred.vy, std::memory_order_relaxed);
            target.valid.store(pred.valid, std::memory_order_release);
            target.lost.store(kf.lost(), std::memory_order_relaxed);

            errs = 0;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        } catch (const std::exception& e) {
            if (++errs > 100) {
                std::fprintf(stderr, "[T-Track] Fatal: %s\n", e.what());
                g_running.store(false, std::memory_order_release);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } catch (...) {
            if (++errs > 100) {
                g_running.store(false, std::memory_order_release);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    std::printf("[T-Track] Stopped\n");
}

// ====================================================================
//  Thread 4: I/O loop @ 1000 Hz
//  Reads physical pad, applies AI correction on right stick, submits.
//  EMA smoothing + non-linear pow() decay curve.
// ====================================================================
static void io_thread(
        GamepadController& gp,
        const PredictedTarget& target,
        const SystemConfig& cfg) {
    std::printf("[T-IO] Started (1000 Hz)\n");

    const float base_alpha = cfg.smoothing_alpha;
    const float decay_exp  = cfg.decay_exponent;
    const float deadzone   = cfg.deadzone_radius;
    const float cx = static_cast<float>(cfg.capture_width)  * 0.5f;
    const float cy = static_cast<float>(cfg.capture_height) * 0.5f;

    float ema_rx = 0.0f, ema_ry = 0.0f;
    auto prev = SteadyClock::now();
    uint64_t tick = 0;
    int errs = 0;

    while (g_running.load(std::memory_order_acquire)) {
        try {
            const auto now = SteadyClock::now();
            const double dt =
                std::chrono::duration<double>(now - prev).count();
            prev = now;
            const double sdt = (dt > 0 && dt < 0.1) ? dt : 0.001;

            float raw_rx = 0.0f, raw_ry = 0.0f;
            float alpha = base_alpha;

            const bool  v    = target.valid.load(std::memory_order_acquire);
            const int   lost = target.lost.load(std::memory_order_relaxed);

            if (v && lost < cfg.max_lost_frames) {
                const float px = target.px.load(std::memory_order_relaxed);
                const float py = target.py.load(std::memory_order_relaxed);
                const float tvx = target.vx.load(std::memory_order_relaxed);
                const float tvy = target.vy.load(std::memory_order_relaxed);

                float ex = px - cx;
                float ey = py - cy;
                float dist = std::sqrt(ex*ex + ey*ey);

                if (dist > deadzone) {
                    float inv = 1.0f / dist;
                    float nx = ex * inv;
                    float ny = ey * inv;

                    float norm_dist = std::min(dist / cx, 1.0f);
                    float strength  = std::pow(norm_dist, decay_exp);

                    // Adaptive alpha: faster when target moves fast
                    float vmag = std::sqrt(tvx*tvx + tvy*tvy);
                    alpha = base_alpha +
                        (0.95f - base_alpha) * std::min(vmag / 500.0f, 1.0f);

                    raw_rx =  nx * strength * cfg.max_correction_speed *
                              static_cast<float>(sdt);
                    raw_ry = -ny * strength * cfg.max_correction_speed *
                              static_cast<float>(sdt);
                } else {
                    alpha = 0.95f; // fast decay to zero near centre
                }
            } else {
                alpha = 0.9f;
            }

            ema_rx = alpha * raw_rx + (1.0f - alpha) * ema_rx;
            ema_ry = alpha * raw_ry + (1.0f - alpha) * ema_ry;

            auto clamp16 = [](float v) -> int16_t {
                return static_cast<int16_t>(
                    std::clamp(v, -32768.0f, 32767.0f));
            };

            int16_t ai_rx = clamp16(ema_rx);
            int16_t ai_ry = clamp16(ema_ry);

            GamepadState phys{};
            gp.poll_physical(phys);
            gp.submit(phys, ai_rx, ai_ry);

            if ((tick++ & 511) == 0) {
                std::printf("[IO] corr=(%+6d,%+6d) alpha=%.3f lost=%d "
                            "dt=%.2fms\n",
                            ai_rx, ai_ry, alpha, lost, sdt*1000.0);
            }

            errs = 0;

            // Precise 1 ms sleep (busy-spin for remainder)
            const auto deadline = now + std::chrono::microseconds(1000);
            std::this_thread::sleep_for(std::chrono::microseconds(600));
            while (SteadyClock::now() < deadline) {
                // busy-wait for sub-ms precision
            }
        } catch (const std::exception& e) {
            if (++errs > 100) {
                std::fprintf(stderr, "[T-IO] Fatal: %s\n", e.what());
                g_running.store(false, std::memory_order_release);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } catch (...) {
            if (++errs > 100) {
                g_running.store(false, std::memory_order_release);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    std::printf("[T-IO] Stopped\n");
}

// ====================================================================
//  main() — top-level orchestrator with full exception safety
// ====================================================================
int main(int argc, char* argv[]) {
    try {
        std::signal(SIGINT,  signal_handler);
        std::signal(SIGTERM, signal_handler);

#ifdef _WIN32
        // Raise timer resolution to 1 ms for the I/O thread
        timeBeginPeriod(1);
#endif

        std::printf("===========================================================\n");
        std::printf("  RT-DETR Realtime Adaptive Input Correction System\n");
        std::printf("  Ultra-Low-Latency Pipeline  |  C++20  |  CUDA EP\n");
        std::printf("===========================================================\n\n");

        SystemConfig cfg;
        if (argc > 1) cfg.model_path = argv[1];

        // ── DXGI capture ────────────────────────────────────────
        std::printf("[Main] Init DXGI capture ...\n");
        DxgiCapture capture;
        if (!capture.initialize(cfg.capture_width, cfg.capture_height)) {
            std::fprintf(stderr, "[Main] DXGI init failed.\n");
            std::fprintf(stderr,
                "  Desktop Duplication API may not be available\n"
                "  (not supported over Remote Desktop / some VMs)\n");
            return 1;
        }
        if (capture.screen_width() > 0) {
            cfg.capture_offset_x =
                (capture.screen_width()  - cfg.capture_width)  / 2;
            cfg.capture_offset_y =
                (capture.screen_height() - cfg.capture_height) / 2;
        }

        // ── RT-DETR detector ────────────────────────────────────
        std::printf("[Main] Init RT-DETR inference engine ...\n");
        RtDetrDetector detector;
        if (!detector.initialize(cfg.model_path, cfg.infer_input_size,
                                 cfg.confidence_threshold,
                                 cfg.target_class_id)) {
            std::fprintf(stderr, "[Main] RT-DETR init failed.\n");
            std::fprintf(stderr, "  Model: %s\n", cfg.model_path);
            return 1;
        }

        // ── Gamepad controller ──────────────────────────────────
        std::printf("[Main] Init gamepad (XInput + ViGEm) ...\n");
        GamepadController gamepad;
        if (!gamepad.initialize()) {
            std::fprintf(stderr, "[Main] Gamepad init failed "
                                 "- detection-only mode\n");
        }

        // ── Kalman tracker ──────────────────────────────────────
        KalmanTracker kalman;
        PredictedTarget predicted;

        // ── Queues ──────────────────────────────────────────────
        RingBuffer<CapturedFrame,   4> frame_q;
        RingBuffer<DetectionResult, 4> det_q;

        // ── Launch 4 threads ────────────────────────────────────
        std::printf("[Main] Launching pipeline ...\n\n");

        std::thread t1(capture_thread,
                       std::ref(capture), std::ref(frame_q));
        std::thread t2(inference_thread,
                       std::ref(detector), std::ref(frame_q),
                       std::ref(det_q));
        std::thread t3(tracking_thread,
                       std::ref(kalman), std::ref(det_q),
                       std::ref(predicted), std::cref(cfg));
        std::thread t4(io_thread,
                       std::ref(gamepad), std::cref(predicted),
                       std::cref(cfg));

        // ── Main heartbeat ──────────────────────────────────────
        while (g_running.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            if (g_running.load(std::memory_order_acquire))
                std::printf("[Main] Running - Ctrl+C to stop\n");
        }

        // ── Shutdown ────────────────────────────────────────────
        std::printf("\n[Main] Shutting down ...\n");
        if (t1.joinable()) t1.join();
        if (t2.joinable()) t2.join();
        if (t3.joinable()) t3.join();
        if (t4.joinable()) t4.join();

        gamepad.shutdown();
        capture.shutdown();

#ifdef _WIN32
        timeEndPeriod(1);
#endif

        std::printf("[Main] Clean shutdown\n");
        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FATAL] %s\n", e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "[FATAL] Unknown exception\n");
        return 1;
    }
}
