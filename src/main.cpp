#include "common/types.h"
#include "common/ring_buffer.h"
#include "capture/dxgi_capture.h"
#include "inference/onnx_detector.h"
#include "tracking/kalman_tracker.h"
#include "input/gamepad_controller.h"

#include <atomic>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <csignal>
#include <mutex>
#include <exception>
#include <stdexcept>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace {

std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running.store(false, std::memory_order_release);
}

} // anonymous namespace

// =======================================================================
//  Thread 1: Screen Capture (DXGI) - Zero-latency GPU-direct capture
// =======================================================================
static void capture_thread(
        tracker::DxgiCapture& capture,
        tracker::RingBuffer<tracker::CapturedFrame, 4>& frame_queue,
        const tracker::SystemConfig& cfg) {
    (void)cfg;
    try {
        std::printf("[CaptureThread] Started\n");

        while (g_running.load(std::memory_order_acquire)) {
            tracker::CapturedFrame frame;
            if (capture.acquire_frame(frame)) {
                frame_queue.try_push(std::move(frame));
            } else {
                // Minimal yield (~5000 Hz poll rate) to stay ahead of vsync
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        std::printf("[CaptureThread] Stopped\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[CaptureThread] EXCEPTION: %s\n", e.what());
        g_running.store(false, std::memory_order_release);
    } catch (...) {
        std::fprintf(stderr, "[CaptureThread] UNKNOWN EXCEPTION\n");
        g_running.store(false, std::memory_order_release);
    }
}

// =======================================================================
//  Thread 2: AI Inference (ONNX Runtime / CUDA EP)
// =======================================================================
struct DetectionResult {
    std::vector<tracker::Detection> detections;
    uint64_t                        frame_id;
    tracker::TimePoint              timestamp;
};

static void inference_thread(
        tracker::OnnxDetector& detector,
        tracker::RingBuffer<tracker::CapturedFrame, 4>& frame_queue,
        tracker::RingBuffer<DetectionResult, 4>& detection_queue) {
    try {
        std::printf("[InferenceThread] Started\n");

        while (g_running.load(std::memory_order_acquire)) {
            auto maybe_frame = frame_queue.try_pop();
            if (!maybe_frame) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }

            auto& frame = *maybe_frame;

            // Detect with minimal copy: pass raw pixel buffer directly
            auto dets = detector.detect(frame.pixels.data(),
                                         frame.width, frame.height,
                                         frame.stride);

            DetectionResult result;
            result.detections = std::move(dets);
            result.frame_id   = frame.frame_id;
            result.timestamp   = frame.timestamp;
            detection_queue.try_push(std::move(result));
        }

        std::printf("[InferenceThread] Stopped\n");
    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "[InferenceThread] ONNX Runtime error: %s\n", e.what());
        g_running.store(false, std::memory_order_release);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[InferenceThread] EXCEPTION: %s\n", e.what());
        g_running.store(false, std::memory_order_release);
    } catch (...) {
        std::fprintf(stderr, "[InferenceThread] UNKNOWN EXCEPTION\n");
        g_running.store(false, std::memory_order_release);
    }
}

// =======================================================================
//  Thread 3: Tracking + Input Control
//  Adaptive EMA + non-linear decay for sub-millisecond response
// =======================================================================
static void control_thread(
        tracker::KalmanTracker& kalman,
        tracker::GamepadController& gamepad,
        tracker::RingBuffer<DetectionResult, 4>& detection_queue,
        const tracker::SystemConfig& cfg) {
    try {
        std::printf("[ControlThread] Started\n");

        const float base_alpha = cfg.smoothing_alpha;
        const float decay_exp  = cfg.decay_exponent;
        const float deadzone   = cfg.deadzone_radius;

        const float cx = static_cast<float>(cfg.capture_region.width)  * 0.5f;
        const float cy = static_cast<float>(cfg.capture_region.height) * 0.5f;

        float ema_rx = 0.0f;
        float ema_ry = 0.0f;

        auto prev_time = tracker::Clock::now();
        uint64_t tick = 0;

        // Adaptive alpha state: ramps up during fast target movement
        float adaptive_alpha = base_alpha;

        while (g_running.load(std::memory_order_acquire)) {
            const auto now = tracker::Clock::now();
            const double dt = std::chrono::duration<double>(now - prev_time).count();
            prev_time = now;

            // Clamp dt to prevent physics explosion after stalls
            const double safe_dt = (dt > 0.0 && dt < 0.1) ? dt : 0.001;

            // -- Drain detection queue: keep newest only --
            DetectionResult det_result;
            bool has_detection = false;
            while (auto maybe = detection_queue.try_pop()) {
                det_result    = std::move(*maybe);
                has_detection = true;
            }

            if (has_detection && !det_result.detections.empty()) {
                const auto& best = det_result.detections[0];
                kalman.update(best.x, best.y);
                kalman.reset_lost();
            } else {
                kalman.increment_lost();
            }

            // -- Kalman predict with safe dt --
            kalman.predict(safe_dt);
            auto predicted = kalman.get_prediction(cfg.prediction_steps, safe_dt);

            // -- Compute AI correction with adaptive response --
            float raw_rx = 0.0f;
            float raw_ry = 0.0f;

            if (predicted.valid && kalman.lost_frames() < cfg.max_lost_frames) {
                float ex = predicted.x - cx;
                float ey = predicted.y - cy;
                float dist = std::sqrt(ex * ex + ey * ey);

                if (dist > deadzone) {
                    float inv_dist = 1.0f / dist;
                    float nx = ex * inv_dist;
                    float ny = ey * inv_dist;

                    const float dist_max = cx;
                    float normalized_dist = std::min(dist / dist_max, 1.0f);
                    float strength = std::pow(normalized_dist, decay_exp);

                    // Adaptive alpha: increase responsiveness proportional
                    // to target velocity magnitude
                    float vel_mag = std::sqrt(predicted.vx * predicted.vx +
                                              predicted.vy * predicted.vy);
                    // Scale alpha from base up to 0.95 based on velocity
                    adaptive_alpha = base_alpha + (0.95f - base_alpha) *
                                     std::min(vel_mag / 500.0f, 1.0f);

                    raw_rx = nx * strength * cfg.max_correction_speed *
                             static_cast<float>(safe_dt);
                    raw_ry = -ny * strength * cfg.max_correction_speed *
                              static_cast<float>(safe_dt);
                } else {
                    // Inside deadzone: rapidly decay EMA to zero
                    adaptive_alpha = 0.95f;
                    // raw_rx/ry stay 0, EMA will converge to 0
                }
            } else {
                // No valid target: use high alpha to quickly zero out
                adaptive_alpha = 0.9f;
            }

            // Adaptive EMA smoothing
            ema_rx = adaptive_alpha * raw_rx + (1.0f - adaptive_alpha) * ema_rx;
            ema_ry = adaptive_alpha * raw_ry + (1.0f - adaptive_alpha) * ema_ry;

            auto clamp16 = [](float v) -> int16_t {
                return static_cast<int16_t>(
                    std::clamp(v, -32768.0f, 32767.0f));
            };

            int16_t ai_rx = clamp16(ema_rx);
            int16_t ai_ry = clamp16(ema_ry);

            // -- Gamepad pass-through + AI injection --
            tracker::GamepadState phys{};
            gamepad.poll_physical(phys);
            gamepad.submit_virtual(phys, ai_rx, ai_ry);

            // -- Telemetry --
            if ((tick++ % 500) == 0) {
                auto state = kalman.get_state();
                std::printf("[Control] pos=(%.1f,%.1f) vel=(%.1f,%.1f) "
                            "corr=(%d,%d) alpha=%.3f lost=%d dt=%.2fms\n",
                            state.x, state.y, state.vx, state.vy,
                            ai_rx, ai_ry, adaptive_alpha,
                            kalman.lost_frames(), safe_dt * 1000.0);
            }

            // ~2000 Hz control loop for sub-millisecond latency
            std::this_thread::sleep_for(std::chrono::microseconds(250));
        }

        std::printf("[ControlThread] Stopped\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ControlThread] EXCEPTION: %s\n", e.what());
        g_running.store(false, std::memory_order_release);
    } catch (...) {
        std::fprintf(stderr, "[ControlThread] UNKNOWN EXCEPTION\n");
        g_running.store(false, std::memory_order_release);
    }
}

// =======================================================================
//  main() - Hardened entry point: catches all exceptions
// =======================================================================
int main(int argc, char* argv[]) {
    try {
        std::signal(SIGINT,  signal_handler);
        std::signal(SIGTERM, signal_handler);

        std::printf("=======================================================\n");
        std::printf("  Realtime Object Tracker & Adaptive Input System\n");
        std::printf("  Ultra-Low-Latency Pipeline v2.0\n");
        std::printf("=======================================================\n\n");

        tracker::SystemConfig cfg;

        if (argc > 1) {
            cfg.model_path = argv[1];
        }

        // -- Initialize modules (each wrapped for clear error reporting) --
        std::printf("[Main] Initializing DXGI capture module...\n");
        tracker::DxgiCapture capture;
        if (!capture.initialize(cfg.capture_region)) {
            std::fprintf(stderr, "[Main] ERROR: DXGI capture initialization failed.\n");
            std::fprintf(stderr, "       Check that Desktop Duplication API is available.\n");
            std::fprintf(stderr, "       (Not supported over Remote Desktop / some VM environments)\n");
            return 1;
        }

        if (capture.screen_width() > 0) {
            cfg.capture_region.offset_x =
                (capture.screen_width()  - cfg.capture_region.width)  / 2;
            cfg.capture_region.offset_y =
                (capture.screen_height() - cfg.capture_region.height) / 2;
        }

        std::printf("[Main] Initializing ONNX Runtime inference engine...\n");
        tracker::OnnxDetector detector;
        if (!detector.initialize(cfg.model_path, cfg.infer_input_size,
                                  cfg.confidence_threshold, cfg.target_class_id)) {
            std::fprintf(stderr, "[Main] ERROR: ONNX detector initialization failed.\n");
            std::fprintf(stderr, "       Model path: %s\n", cfg.model_path);
            std::fprintf(stderr, "       Ensure the .onnx model file exists and CUDA drivers are installed.\n");
            return 1;
        }

        std::printf("[Main] Initializing gamepad controller (XInput + ViGEm)...\n");
        tracker::GamepadController gamepad;
        if (!gamepad.initialize()) {
            std::fprintf(stderr, "[Main] WARNING: Gamepad init failed - "
                                 "detection-only mode (no controller output)\n");
            std::fprintf(stderr, "       Install ViGEmBus driver: https://github.com/ViGEm/ViGEmBus/releases\n");
        }

        std::printf("[Main] Initializing Kalman tracker (6-state constant-acceleration)...\n");
        tracker::KalmanTracker kalman;

        tracker::RingBuffer<tracker::CapturedFrame, 4> frame_queue;
        tracker::RingBuffer<DetectionResult, 4>        detection_queue;

        std::printf("[Main] Launching pipeline threads...\n\n");

        std::thread t_capture(capture_thread,
                              std::ref(capture),
                              std::ref(frame_queue),
                              std::cref(cfg));

        std::thread t_inference(inference_thread,
                                std::ref(detector),
                                std::ref(frame_queue),
                                std::ref(detection_queue));

        std::thread t_control(control_thread,
                              std::ref(kalman),
                              std::ref(gamepad),
                              std::ref(detection_queue),
                              std::cref(cfg));

        // -- Main thread: status monitor --
        while (g_running.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            if (g_running.load(std::memory_order_acquire)) {
                std::printf("[Main] System running - press Ctrl+C to stop\n");
            }
        }

        std::printf("\n[Main] Shutting down...\n");

        if (t_capture.joinable())   t_capture.join();
        if (t_inference.joinable()) t_inference.join();
        if (t_control.joinable())   t_control.join();

        gamepad.shutdown();
        capture.shutdown();

        std::printf("[Main] Clean shutdown complete\n");
        return 0;

    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "[Main] ONNX Runtime fatal error: %s\n", e.what());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[Main] Fatal exception: %s\n", e.what());
        return 2;
    } catch (...) {
        std::fprintf(stderr, "[Main] Unknown fatal exception\n");
        return 2;
    }
}
