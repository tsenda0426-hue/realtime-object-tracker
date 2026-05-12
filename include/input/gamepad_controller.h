#pragma once

#include "common/types.h"
#include <cstdint>
#include <atomic>

#ifdef _WIN32
#include <Windows.h>
#include <Xinput.h>
#include <ViGEm/Client.h>
#endif

namespace tracker {

class GamepadController {
public:
    GamepadController();
    ~GamepadController();

    // Non-copyable
    GamepadController(const GamepadController&) = delete;
    GamepadController& operator=(const GamepadController&) = delete;

    // Scan XInput ports 0-3 and lock the first connected pad.
    // Create a virtual Xbox 360 controller via ViGEmBus.
    bool initialize();
    void shutdown();

    // Read physical gamepad state
    bool poll_physical(GamepadState& state);

    // Submit combined state to virtual gamepad.
    // physical: raw physical input
    // ai_rx, ai_ry: AI correction values for right stick (signed, -32768..32767 range)
    bool submit_virtual(const GamepadState& physical, int16_t ai_rx, int16_t ai_ry);

    int  locked_port() const { return locked_port_; }
    bool is_ready()    const { return ready_; }

private:
    int  locked_port_ = -1;
    bool ready_       = false;

#ifdef _WIN32
    PVIGEM_CLIENT  vigem_client_  = nullptr;
    PVIGEM_TARGET  vigem_target_  = nullptr;
#endif
};

} // namespace tracker
