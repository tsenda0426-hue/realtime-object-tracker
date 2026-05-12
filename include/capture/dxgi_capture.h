#pragma once

#include "common/types.h"
#include <vector>
#include <cstdint>
#include <atomic>

#ifdef _WIN32
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#endif

namespace tracker {

// Captured frame: raw BGRA pixel data for the requested region.
struct CapturedFrame {
    std::vector<uint8_t> pixels;  // BGRA 8-bit
    int      width;
    int      height;
    int      stride;             // bytes per row
    uint64_t frame_id;
    TimePoint timestamp;
};

class DxgiCapture {
public:
    DxgiCapture();
    ~DxgiCapture();

    // Non-copyable
    DxgiCapture(const DxgiCapture&) = delete;
    DxgiCapture& operator=(const DxgiCapture&) = delete;

    bool initialize(const CaptureRegion& region);
    void shutdown();

    // Acquires the next desktop frame. Returns false if no new frame.
    // The captured region is cropped to `region_` on the GPU side.
    bool acquire_frame(CapturedFrame& out);

    int screen_width()  const { return screen_w_; }
    int screen_height() const { return screen_h_; }

private:
    CaptureRegion region_{};
    int screen_w_ = 0;
    int screen_h_ = 0;
    uint64_t frame_counter_ = 0;

#ifdef _WIN32
    Microsoft::WRL::ComPtr<ID3D11Device>            device_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext>      context_;
    Microsoft::WRL::ComPtr<IDXGIOutputDuplication>   duplication_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D>          staging_tex_;

    bool create_device();
    bool create_duplication();
    bool create_staging_texture();
#endif
};

} // namespace tracker
