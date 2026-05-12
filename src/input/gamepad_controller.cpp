#include "input/gamepad_controller.h"
#include <cstdio>
#include <algorithm>

namespace tracker {

GamepadController::GamepadController() = default;

GamepadController::~GamepadController() {
    shutdown();
}

#ifdef _WIN32

bool GamepadController::initialize() {
    // ── Scan XInput ports 0-3 and lock the first connected pad ──
    XINPUT_STATE xi_state;
    for (DWORD port = 0; port < XUSER_MAX_COUNT; ++port) {
        if (XInputGetState(port, &xi_state) == ERROR_SUCCESS) {
            locked_port_ = static_cast<int>(port);
            std::printf("[Gamepad] Physical pad found on XInput port %d\n",
                        locked_port_);
            break;
        }
    }
    if (locked_port_ < 0) {
        std::fprintf(stderr, "[Gamepad] No physical XInput controller found\n");
        return false;
    }

    // ── Initialize ViGEmClient ──
    vigem_client_ = vigem_alloc();
    if (!vigem_client_) {
        std::fprintf(stderr, "[Gamepad] vigem_alloc failed\n");
        return false;
    }

    VIGEM_ERROR err = vigem_connect(vigem_client_);
    if (!VIGEM_SUCCESS(err)) {
        std::fprintf(stderr, "[Gamepad] vigem_connect failed: 0x%08x\n",
                     static_cast<unsigned>(err));
        vigem_free(vigem_client_);
        vigem_client_ = nullptr;
        return false;
    }

    // ── Create virtual Xbox 360 controller ──
    vigem_target_ = vigem_target_x360_alloc();
    if (!vigem_target_) {
        std::fprintf(stderr, "[Gamepad] vigem_target_x360_alloc failed\n");
        shutdown();
        return false;
    }

    err = vigem_target_add(vigem_client_, vigem_target_);
    if (!VIGEM_SUCCESS(err)) {
        std::fprintf(stderr, "[Gamepad] vigem_target_add failed: 0x%08x\n",
                     static_cast<unsigned>(err));
        vigem_target_free(vigem_target_);
        vigem_target_ = nullptr;
        shutdown();
        return false;
    }

    ready_ = true;
    std::printf("[Gamepad] Virtual Xbox 360 controller created successfully\n");
    return true;
}

void GamepadController::shutdown() {
    if (vigem_target_ && vigem_client_) {
        vigem_target_remove(vigem_client_, vigem_target_);
        vigem_target_free(vigem_target_);
        vigem_target_ = nullptr;
    }
    if (vigem_client_) {
        vigem_disconnect(vigem_client_);
        vigem_free(vigem_client_);
        vigem_client_ = nullptr;
    }
    ready_ = false;
}

bool GamepadController::poll_physical(GamepadState& state) {
    if (locked_port_ < 0) return false;

    XINPUT_STATE xi_state;
    DWORD result = XInputGetState(static_cast<DWORD>(locked_port_), &xi_state);
    if (result != ERROR_SUCCESS) return false;

    state.buttons       = xi_state.Gamepad.wButtons;
    state.left_trigger  = xi_state.Gamepad.bLeftTrigger;
    state.right_trigger = xi_state.Gamepad.bRightTrigger;
    state.lx            = xi_state.Gamepad.sThumbLX;
    state.ly            = xi_state.Gamepad.sThumbLY;
    state.rx            = xi_state.Gamepad.sThumbRX;
    state.ry            = xi_state.Gamepad.sThumbRY;
    return true;
}

bool GamepadController::submit_virtual(const GamepadState& physical,
                                        int16_t ai_rx, int16_t ai_ry) {
    if (!ready_ || !vigem_target_) return false;

    XUSB_REPORT report;
    XUSB_REPORT_INIT(&report);

    // Pass through all physical inputs
    report.wButtons      = physical.buttons;
    report.bLeftTrigger  = physical.left_trigger;
    report.bRightTrigger = physical.right_trigger;
    report.sThumbLX      = physical.lx;
    report.sThumbLY      = physical.ly;

    // Right stick: physical + AI correction (clamped to int16 range)
    const int32_t combined_rx = static_cast<int32_t>(physical.rx) +
                                static_cast<int32_t>(ai_rx);
    const int32_t combined_ry = static_cast<int32_t>(physical.ry) +
                                static_cast<int32_t>(ai_ry);

    report.sThumbRX = static_cast<SHORT>(
        std::clamp(combined_rx, static_cast<int32_t>(-32768),
                                static_cast<int32_t>(32767)));
    report.sThumbRY = static_cast<SHORT>(
        std::clamp(combined_ry, static_cast<int32_t>(-32768),
                                static_cast<int32_t>(32767)));

    VIGEM_ERROR err = vigem_target_x360_update(vigem_client_, vigem_target_,
                                                report);
    return VIGEM_SUCCESS(err);
}

#else // Non-Windows stubs

bool GamepadController::initialize() {
    std::printf("[Gamepad] Stub initialized (non-Windows build)\n");
    return true;
}

void GamepadController::shutdown() {}

bool GamepadController::poll_physical(GamepadState& state) {
    state = {};
    return false;
}

bool GamepadController::submit_virtual(const GamepadState& /*physical*/,
                                        int16_t /*ai_rx*/, int16_t /*ai_ry*/) {
    return false;
}

#endif // _WIN32

} // namespace tracker
