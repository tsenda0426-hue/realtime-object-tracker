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
//  Thread 1: Screen Capture (DXGI)
// =======================================================================
static void capture_thread(
        tracker::DxgiCapture& capture,
        tracker::RingBuffer<tracker::CapturedFrame, 4>& frame_queue,
        const tracker::SystemConfig& cfg) {
    (void)cfg;
    std::printf("[CaptureThread] Started\n");

    while (g_running.load(std::memory_order_acquire)) {
        tracker::CapturedFrame frame;
        if (capture.acquire_frame(frame)) {
            // Drop oldest if queue is full (triple-buffering style)
            frame_queue.try_push(frame);
        } else {
            // No new frame - yield briefly to avoid busy-spinning
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    std::printf("[CaptureThread] Stopped\n");
}

// =======================================================================
//  Thread 2: AI Inference (ONNX Runtime / CUDA)
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
    std::printf("[InferenceThread] Started\n");

    while (g_running.load(std::memory_order_acquire)) {
        auto maybe_frame = frame_queue.try_pop();
        if (!maybe_frame) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        auto& frame = *maybe_frame;
        auto dets = detector.detect(frame.pixels.data(),
                                     frame.width, frame.height,
                                     frame.stride);

        DetectionResult result;
        result.detections = std::move(dets);
        result.frame_id   = frame.frame_id;
        result.timestamp   = frame.timestamp;
        detection_queue.try_push(result);
    }

    std::printf("[InferenceThread] Stopped\n");
}

// =======================================================================
//  Thread 3: Tracking + Input Control (main control loop)
// =======================================================================
static void control_thread(
        tracker::KalmanTracker& kalman,
        tracker::GamepadController& gamepad,
        tracker::RingBuffer<DetectionResult, 4>& detection_queue,
        const tracker::SystemConfig& cfg) {
    std::printf("[ControlThread] Started\n");

    const float alpha     = cfg.smoothing_alpha;
    const float decay_exp = cfg.decay_exponent;
    const float deadzone  = cfg.deadzone_radius;

    // Screen center (target crosshair position)
    const float cx = static_cast<float>(cfg.capture_region.width)  * 0.5f;
    const float cy = static_cast<float>(cfg.capture_region.height) * 0.5f;

    // EMA-smoothed correction
    float ema_rx = 0.0f;
    float ema_ry = 0.0f;

    auto prev_time = tracker::Clock::now();

    uint64_t tick = 0;

    while (g_running.load(std::memory_order_acquire)) {
        const auto now = tracker::Clock::now();
        const double dt = std::chrono::duration<double>(now - prev_time).count();
        prev_time = now;

        // -- Process latest detections --
        DetectionResult det_result;
        bool has_detection = false;
        // Drain queue: keep only the newest result
        while (auto maybe = detection_queue.try_pop()) {
            det_result    = std::move(*maybe);
            has_detection = true;
        }

        if (has_detection && !det_result.detections.empty()) {
            // Use highest-confidence detection
            const auto& best = det_result.detections[0];
            kalman.update(best.x, best.y);
            kalman.reset_lost();
        } else {
            kalman.increment_lost();
        }

        // -- Kalman predict --
        kalman.predict(dt);
        auto predicted = kalman.get_prediction(cfg.prediction_steps, dt);

        // -- Compute AI correction --
        float raw_rx = 0.0f;
        float raw_ry = 0.0f;

        if (predicted.valid && kalman.lost_frames() < cfg.max_lost_frames) {
            // Error vector from screen center to predicted target
            float ex = predicted.x - cx;
            float ey = predicted.y - cy;
            float dist = std::sqrt(ex * ex + ey * ey);

            if (dist > deadzone) {
                // Normalize
                float nx = ex / dist;
                float ny = ey / dist;

                // Non-linear decay: output ~ dist^decay_exp / dist_max^decay_exp
                // Closer to target -> exponentially weaker output -> no overshoot
                const float dist_max = cx;  // max meaningful distance
                float normalized_dist = std::min(dist / dist_max, 1.0f);
                float strength = std::pow(normalized_dist, decay_exp);

                // Scale to stick range
                raw_rx = nx * strength * cfg.max_correction_speed *
                         static_cast<float>(dt);
                raw_ry = -ny * strength * cfg.max_correction_speed *
                          static_cast<float>(dt); // Y-axis inverted for stick
            }
        }

        // EMA smoothing
        ema_rx = alpha * raw_rx + (1.0f - alpha) * ema_rx;
        ema_ry = alpha * raw_ry + (1.0f - alpha) * ema_ry;

        // Clamp to int16 range
        auto clamp16 = [](float v) -> int16_t {
            return static_cast<int16_t>(
                std::clamp(v, -32768.0f, 32767.0f));
        };

        int16_t ai_rx = clamp16(ema_rx);
        int16_t ai_ry = clamp16(ema_ry);

        // -- Read physical pad & submit to virtual --
        tracker::GamepadState phys{};
        gamepad.poll_physical(phys);
        gamepad.submit_virtual(phys, ai_rx, ai_ry);

        // -- Telemetry (periodic) --
        if ((tick++ % 500) == 0) {
            auto state = kalman.get_state();
            std::printf("[Control] target=(%.1f,%.1f) vel=(%.1f,%.1f) "
                        "corr=(%d,%d) lost=%d dt=%.2fms\n",
                        state.x, state.y, state.vx, state.vy,
                        ai_rx, ai_ry, kalman.lost_frames(),
                        dt * 1000.0);
        }

        // Rate-limit control loop to ~1000 Hz
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }

    std::printf("[ControlThread] Stopped\n");
}

// =======================================================================
//  main()
// =======================================================================
int main(int argc, char* argv[]) {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::printf("=======================================================\n");
    std::printf("  Realtime Object Tracker & Adaptive Input System\n");
    std::printf("  Industrial / Military Grade Architecture\n");
    std::printf("=======================================================\n\n");

    tracker::SystemConfig cfg;

    // Allow model path override via CLI
    if (argc > 1) {
        cfg.model_path = argv[1];
    }

    // -- Initialize modules --
    std::printf("[Main] Initializing capture module...\n");
    tracker::DxgiCapture capture;
    if (!capture.initialize(cfg.capture_region)) {
        std::fprintf(stderr, "[Main] FATAL: Capture initialization failed\n");
        return 1;
    }

    // Update capture region to screen center if auto-detected
    if (capture.screen_width() > 0) {
        cfg.capture_region.offset_x =
            (capture.screen_width()  - cfg.capture_region.width)  / 2;
        cfg.capture_region.offset_y =
            (capture.screen_height() - cfg.capture_region.height) / 2;
    }

    std::printf("[Main] Initializing AI inference module...\n");
    tracker::OnnxDetector detector;
    if (!detector.initialize(cfg.model_path, cfg.infer_input_size,
                              cfg.confidence_threshold, cfg.target_class_id)) {
        std::fprintf(stderr, "[Main] FATAL: ONNX detector initialization failed\n");
        return 1;
    }

    std::printf("[Main] Initializing gamepad controller...\n");
    tracker::GamepadController gamepad;
    if (!gamepad.initialize()) {
        std::fprintf(stderr, "[Main] WARNING: Gamepad initialization failed - "
                             "running in detection-only mode\n");
    }

    std::printf("[Main] Initializing Kalman tracker...\n");
    tracker::KalmanTracker kalman;

    // -- Inter-thread queues (lock-free, power-of-2 capacity) --
    tracker::RingBuffer<tracker::CapturedFrame, 4> frame_queue;
    tracker::RingBuffer<DetectionResult, 4>        detection_queue;

    // -- Launch threads --
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
        std::printf("[Main] System running - press Ctrl+C to stop\n");
    }

    std::printf("\n[Main] Shutting down...\n");

    // -- Join threads --
    if (t_capture.joinable())   t_capture.join();
    if (t_inference.joinable()) t_inference.join();
    if (t_control.joinable())   t_control.join();

    // -- Cleanup --
    gamepad.shutdown();
    capture.shutdown();

    std::printf("[Main] Clean shutdown complete\n");
    return 0;
}
