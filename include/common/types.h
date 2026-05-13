#pragma once

#include <cstdint>
#include <chrono>
#include <vector>

namespace tracker {

struct Detection {
    float x;        // center x (pixels)
    float y;        // center y (pixels)
    float w;        // bbox width
    float h;        // bbox height
    float confidence;
    int   class_id;
};

struct TrackedTarget {
    float x;           // predicted center x
    float y;           // predicted center y
    float vx;          // velocity x
    float vy;          // velocity y
    bool  valid;       // whether prediction is trustworthy
    int   frames_lost; // frames since last detection
};

struct GamepadState {
    uint16_t buttons;
    uint8_t  left_trigger;
    uint8_t  right_trigger;
    int16_t  lx;   // left stick X
    int16_t  ly;   // left stick Y
    int16_t  rx;   // right stick X
    int16_t  ry;   // right stick Y
};

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Duration  = std::chrono::duration<double, std::milli>;

// capture region
struct CaptureRegion {
    int offset_x;
    int offset_y;
    int width;
    int height;
};

// system-wide config
struct SystemConfig {
    // capture
    CaptureRegion capture_region = {0, 0, 400, 400};
    int target_fps               = 144;

    // inference
    const char* model_path       = "yolov8n.onnx";
    int   infer_input_size       = 640;
    float confidence_threshold   = 0.45f;
    int   target_class_id        = 0;

    // tracking
    int   max_lost_frames        = 20;      // tolerate more lost frames
    int   prediction_steps       = 2;       // fewer steps = tighter to current

    // input control (tuned for sub-ms response)
    float smoothing_alpha        = 0.6f;    // base EMA alpha (adaptive range: 0.6-0.95)
    float decay_exponent         = 1.5f;    // gentler curve for smoother convergence
    float max_correction_speed   = 25000.0f; // aggressive stick deflection rate
    float deadzone_radius        = 3.0f;    // tighter deadzone for precision
};

} // namespace tracker
