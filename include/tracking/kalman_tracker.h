#pragma once

#include "common/types.h"

namespace tracker {

// 6-state Kalman Filter: [x, y, vx, vy, ax, ay]
// Measurement: [x, y]
// Predicts future position using constant-acceleration model.
class KalmanTracker {
public:
    KalmanTracker();

    void initialize(float x, float y);
    void reset();

    // Predict next state (call every frame)
    void predict(double dt);

    // Update with new measurement
    void update(float meas_x, float meas_y);

    // Get predicted position N steps ahead
    TrackedTarget get_prediction(int steps_ahead, double dt) const;

    // Get current state
    TrackedTarget get_state() const;

    bool is_initialized() const { return initialized_; }
    int  lost_frames()    const { return lost_frames_; }

    void increment_lost() { ++lost_frames_; }
    void reset_lost()     { lost_frames_ = 0; }

private:
    static constexpr int STATE_DIM = 6;  // x, y, vx, vy, ax, ay
    static constexpr int MEAS_DIM  = 2;  // x, y

    // State vector [x, y, vx, vy, ax, ay]
    float state_[STATE_DIM];

    // Covariance matrix (STATE_DIM x STATE_DIM, stored row-major)
    float P_[STATE_DIM * STATE_DIM];

    // Process noise
    float Q_[STATE_DIM * STATE_DIM];

    // Measurement noise
    float R_[MEAS_DIM * MEAS_DIM];

    bool initialized_ = false;
    int  lost_frames_  = 0;

    // Matrix helpers
    static void mat_multiply(const float* A, const float* B, float* C,
                             int m, int n, int p);
    static void mat_add(const float* A, const float* B, float* C, int rows, int cols);
    static void mat_subtract(const float* A, const float* B, float* C, int rows, int cols);
    static void mat_transpose(const float* A, float* At, int rows, int cols);
    static bool mat_invert_2x2(const float* M, float* Minv);
    static void mat_identity(float* M, int n);
    static void mat_zero(float* M, int rows, int cols);
};

} // namespace tracker
