#include "tracking/kalman_tracker.h"
#include <cstring>
#include <cmath>

namespace tracker {

// ── Matrix utility implementations ──────────────────────────────────
void KalmanTracker::mat_multiply(const float* A, const float* B, float* C,
                                  int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void KalmanTracker::mat_add(const float* A, const float* B, float* C,
                             int rows, int cols) {
    const int n = rows * cols;
    for (int i = 0; i < n; ++i) C[i] = A[i] + B[i];
}

void KalmanTracker::mat_subtract(const float* A, const float* B, float* C,
                                  int rows, int cols) {
    const int n = rows * cols;
    for (int i = 0; i < n; ++i) C[i] = A[i] - B[i];
}

void KalmanTracker::mat_transpose(const float* A, float* At,
                                   int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            At[j * rows + i] = A[i * cols + j];
}

bool KalmanTracker::mat_invert_2x2(const float* M, float* Minv) {
    const float det = M[0] * M[3] - M[1] * M[2];
    if (std::fabs(det) < 1e-12f) return false;
    const float inv_det = 1.0f / det;
    Minv[0] =  M[3] * inv_det;
    Minv[1] = -M[1] * inv_det;
    Minv[2] = -M[2] * inv_det;
    Minv[3] =  M[0] * inv_det;
    return true;
}

void KalmanTracker::mat_identity(float* M, int n) {
    std::memset(M, 0, sizeof(float) * static_cast<size_t>(n * n));
    for (int i = 0; i < n; ++i) M[i * n + i] = 1.0f;
}

void KalmanTracker::mat_zero(float* M, int rows, int cols) {
    std::memset(M, 0, sizeof(float) * static_cast<size_t>(rows * cols));
}

// ── Constructor ─────────────────────────────────────────────────────
KalmanTracker::KalmanTracker() {
    reset();
}

void KalmanTracker::reset() {
    initialized_ = false;
    lost_frames_  = 0;
    std::memset(state_, 0, sizeof(state_));

    mat_identity(P_, STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) P_[i * STATE_DIM + i] = 100.0f;

    // Process noise (tuned for pixel-level tracking at ~144fps)
    mat_zero(Q_, STATE_DIM, STATE_DIM);
    Q_[0 * STATE_DIM + 0] = 1.0f;    // x position
    Q_[1 * STATE_DIM + 1] = 1.0f;    // y position
    Q_[2 * STATE_DIM + 2] = 5.0f;    // vx
    Q_[3 * STATE_DIM + 3] = 5.0f;    // vy
    Q_[4 * STATE_DIM + 4] = 10.0f;   // ax
    Q_[5 * STATE_DIM + 5] = 10.0f;   // ay

    // Measurement noise
    mat_zero(R_, MEAS_DIM, MEAS_DIM);
    R_[0] = 4.0f;  // x measurement variance
    R_[3] = 4.0f;  // y measurement variance
}

void KalmanTracker::initialize(float x, float y) {
    reset();
    state_[0] = x;
    state_[1] = y;
    initialized_ = true;
}

void KalmanTracker::predict(double dt) {
    if (!initialized_) return;

    const float t  = static_cast<float>(dt);
    const float t2 = 0.5f * t * t;

    // State transition matrix F (constant acceleration model)
    // x  = x  + vx*dt + 0.5*ax*dt^2
    // y  = y  + vy*dt + 0.5*ay*dt^2
    // vx = vx + ax*dt
    // vy = vy + ay*dt
    // ax = ax
    // ay = ay
    float F[STATE_DIM * STATE_DIM];
    mat_identity(F, STATE_DIM);
    F[0 * STATE_DIM + 2] = t;    // x  ← vx
    F[0 * STATE_DIM + 4] = t2;   // x  ← ax
    F[1 * STATE_DIM + 3] = t;    // y  ← vy
    F[1 * STATE_DIM + 5] = t2;   // y  ← ay
    F[2 * STATE_DIM + 4] = t;    // vx ← ax
    F[3 * STATE_DIM + 5] = t;    // vy ← ay

    // x_pred = F * x
    float x_pred[STATE_DIM];
    mat_multiply(F, state_, x_pred, STATE_DIM, STATE_DIM, 1);

    // P_pred = F * P * F^T + Q
    float Ft[STATE_DIM * STATE_DIM];
    mat_transpose(F, Ft, STATE_DIM, STATE_DIM);

    float FP[STATE_DIM * STATE_DIM];
    mat_multiply(F, P_, FP, STATE_DIM, STATE_DIM, STATE_DIM);

    float FPFt[STATE_DIM * STATE_DIM];
    mat_multiply(FP, Ft, FPFt, STATE_DIM, STATE_DIM, STATE_DIM);

    mat_add(FPFt, Q_, P_, STATE_DIM, STATE_DIM);
    std::memcpy(state_, x_pred, sizeof(state_));
}

void KalmanTracker::update(float meas_x, float meas_y) {
    if (!initialized_) {
        initialize(meas_x, meas_y);
        return;
    }

    // Measurement matrix H: [1 0 0 0 0 0; 0 1 0 0 0 0]
    float H[MEAS_DIM * STATE_DIM];
    mat_zero(H, MEAS_DIM, STATE_DIM);
    H[0 * STATE_DIM + 0] = 1.0f;
    H[1 * STATE_DIM + 1] = 1.0f;

    // Innovation: y = z - H * x
    float z[MEAS_DIM] = {meas_x, meas_y};
    float Hx[MEAS_DIM];
    mat_multiply(H, state_, Hx, MEAS_DIM, STATE_DIM, 1);

    float y[MEAS_DIM];
    mat_subtract(z, Hx, y, MEAS_DIM, 1);

    // S = H * P * H^T + R
    float Ht[STATE_DIM * MEAS_DIM];
    mat_transpose(H, Ht, MEAS_DIM, STATE_DIM);

    float HP[MEAS_DIM * STATE_DIM];
    mat_multiply(H, P_, HP, MEAS_DIM, STATE_DIM, STATE_DIM);

    float S[MEAS_DIM * MEAS_DIM];
    float HPHt[MEAS_DIM * MEAS_DIM];
    mat_multiply(HP, Ht, HPHt, MEAS_DIM, STATE_DIM, MEAS_DIM);
    mat_add(HPHt, R_, S, MEAS_DIM, MEAS_DIM);

    // S_inv = inv(S) (2x2)
    float S_inv[MEAS_DIM * MEAS_DIM];
    if (!mat_invert_2x2(S, S_inv)) return;

    // K = P * H^T * S_inv
    float PHt[STATE_DIM * MEAS_DIM];
    mat_multiply(P_, Ht, PHt, STATE_DIM, STATE_DIM, MEAS_DIM);

    float K[STATE_DIM * MEAS_DIM];
    mat_multiply(PHt, S_inv, K, STATE_DIM, MEAS_DIM, MEAS_DIM);

    // x_new = x + K * y
    float Ky[STATE_DIM];
    mat_multiply(K, y, Ky, STATE_DIM, MEAS_DIM, 1);
    for (int i = 0; i < STATE_DIM; ++i) state_[i] += Ky[i];

    // P_new = (I - K * H) * P
    float KH[STATE_DIM * STATE_DIM];
    mat_multiply(K, H, KH, STATE_DIM, MEAS_DIM, STATE_DIM);

    float I[STATE_DIM * STATE_DIM];
    mat_identity(I, STATE_DIM);

    float IKH[STATE_DIM * STATE_DIM];
    mat_subtract(I, KH, IKH, STATE_DIM, STATE_DIM);

    float P_new[STATE_DIM * STATE_DIM];
    mat_multiply(IKH, P_, P_new, STATE_DIM, STATE_DIM, STATE_DIM);
    std::memcpy(P_, P_new, sizeof(P_));
}

TrackedTarget KalmanTracker::get_state() const {
    TrackedTarget t;
    t.x           = state_[0];
    t.y           = state_[1];
    t.vx          = state_[2];
    t.vy          = state_[3];
    t.valid       = initialized_;
    t.frames_lost = lost_frames_;
    return t;
}

TrackedTarget KalmanTracker::get_prediction(int steps_ahead, double dt) const {
    if (!initialized_) {
        return {0, 0, 0, 0, false, lost_frames_};
    }

    const float t  = static_cast<float>(dt) * static_cast<float>(steps_ahead);
    const float t2 = 0.5f * t * t;

    TrackedTarget pred;
    pred.x  = state_[0] + state_[2] * t + state_[4] * t2;
    pred.y  = state_[1] + state_[3] * t + state_[5] * t2;
    pred.vx = state_[2] + state_[4] * t;
    pred.vy = state_[3] + state_[5] * t;
    pred.valid       = true;
    pred.frames_lost = lost_frames_;
    return pred;
}

} // namespace tracker
