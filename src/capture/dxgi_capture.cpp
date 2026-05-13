#include "capture/dxgi_capture.h"
#include <cstring>
#include <cstdio>
#include <algorithm>

namespace tracker {

DxgiCapture::DxgiCapture() = default;
DxgiCapture::~DxgiCapture() { shutdown(); }

#ifdef _WIN32

bool DxgiCapture::create_device() {
    D3D_FEATURE_LEVEL feature_levels[] = {D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL feature_level_out;

    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        feature_levels,
        1,
        D3D11_SDK_VERSION,
        &device_,
        &feature_level_out,
        &context_
    );

    if (FAILED(hr)) {
        std::fprintf(stderr, "[DxgiCapture] D3D11CreateDevice failed: 0x%08lx\n", hr);
        return false;
    }
    return true;
}

bool DxgiCapture::create_duplication() {
    Microsoft::WRL::ComPtr<IDXGIDevice>  dxgi_device;
    Microsoft::WRL::ComPtr<IDXGIAdapter> dxgi_adapter;
    Microsoft::WRL::ComPtr<IDXGIOutput>  dxgi_output;
    Microsoft::WRL::ComPtr<IDXGIOutput1> dxgi_output1;

    HRESULT hr = device_.As(&dxgi_device);
    if (FAILED(hr)) return false;

    hr = dxgi_device->GetAdapter(&dxgi_adapter);
    if (FAILED(hr)) return false;

    hr = dxgi_adapter->EnumOutputs(0, &dxgi_output);
    if (FAILED(hr)) return false;

    // Get output (monitor) description for screen dimensions
    DXGI_OUTPUT_DESC out_desc;
    dxgi_output->GetDesc(&out_desc);
    screen_w_ = out_desc.DesktopCoordinates.right  - out_desc.DesktopCoordinates.left;
    screen_h_ = out_desc.DesktopCoordinates.bottom - out_desc.DesktopCoordinates.top;

    hr = dxgi_output.As(&dxgi_output1);
    if (FAILED(hr)) return false;

    hr = dxgi_output1->DuplicateOutput(device_.Get(), &duplication_);
    if (FAILED(hr)) {
        std::fprintf(stderr, "[DxgiCapture] DuplicateOutput failed: 0x%08lx\n", hr);
        return false;
    }

    return true;
}

bool DxgiCapture::create_staging_texture() {
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width            = static_cast<UINT>(region_.width);
    desc.Height           = static_cast<UINT>(region_.height);
    desc.MipLevels        = 1;
    desc.ArraySize        = 1;
    desc.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage            = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags   = D3D11_CPU_ACCESS_READ;

    HRESULT hr = device_->CreateTexture2D(&desc, nullptr, &staging_tex_);
    if (FAILED(hr)) {
        std::fprintf(stderr, "[DxgiCapture] CreateTexture2D (staging) failed: 0x%08lx\n", hr);
        return false;
    }
    return true;
}

bool DxgiCapture::initialize(const CaptureRegion& region) {
    region_ = region;

    if (!create_device())      return false;
    if (!create_duplication())  return false;

    // Center the capture region on the screen
    if (region_.offset_x == 0 && region_.offset_y == 0) {
        region_.offset_x = (screen_w_ - region_.width)  / 2;
        region_.offset_y = (screen_h_ - region_.height) / 2;
    }

    // Clamp
    region_.offset_x = std::clamp(region_.offset_x, 0, screen_w_ - region_.width);
    region_.offset_y = std::clamp(region_.offset_y, 0, screen_h_ - region_.height);

    if (!create_staging_texture()) return false;

    std::printf("[DxgiCapture] Initialized: screen=%dx%d, region=(%d,%d)+%dx%d\n",
                screen_w_, screen_h_,
                region_.offset_x, region_.offset_y,
                region_.width, region_.height);
    return true;
}

void DxgiCapture::shutdown() {
    if (duplication_) {
        duplication_.Reset();
    }
}

bool DxgiCapture::acquire_frame(CapturedFrame& out) {
    // If duplication was lost, attempt recovery
    if (!duplication_) {
        if (!create_duplication() || !create_staging_texture()) {
            return false;
        }
        std::printf("[DxgiCapture] Duplication recovered\n");
    }

    Microsoft::WRL::ComPtr<IDXGIResource> desktop_resource;
    DXGI_OUTDUPL_FRAME_INFO frame_info;

    HRESULT hr = duplication_->AcquireNextFrame(1, &frame_info, &desktop_resource);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        return false;
    }

    if (hr == DXGI_ERROR_ACCESS_LOST) {
        std::fprintf(stderr, "[DxgiCapture] Access lost - recovering\n");
        duplication_.Reset();
        staging_tex_.Reset();
        return false;
    }

    if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
        std::fprintf(stderr, "[DxgiCapture] Device lost (0x%08lx) - full reinit\n", hr);
        duplication_.Reset();
        staging_tex_.Reset();
        device_.Reset();
        context_.Reset();
        if (create_device() && create_duplication() && create_staging_texture()) {
            std::printf("[DxgiCapture] Device reinitialized\n");
        }
        return false;
    }

    if (FAILED(hr)) {
        std::fprintf(stderr, "[DxgiCapture] AcquireNextFrame: 0x%08lx\n", hr);
        duplication_.Reset();
        staging_tex_.Reset();
        return false;
    }

    // Get the desktop texture
    Microsoft::WRL::ComPtr<ID3D11Texture2D> desktop_tex;
    hr = desktop_resource.As(&desktop_tex);
    if (FAILED(hr)) {
        duplication_->ReleaseFrame();
        return false;
    }

    // Validate staging texture exists
    if (!staging_tex_) {
        duplication_->ReleaseFrame();
        return false;
    }

    // Copy the capture region from the desktop texture to our staging texture
    D3D11_BOX src_box;
    src_box.left   = static_cast<UINT>(region_.offset_x);
    src_box.top    = static_cast<UINT>(region_.offset_y);
    src_box.right  = static_cast<UINT>(region_.offset_x + region_.width);
    src_box.bottom = static_cast<UINT>(region_.offset_y + region_.height);
    src_box.front  = 0;
    src_box.back   = 1;

    context_->CopySubresourceRegion(
        staging_tex_.Get(), 0, 0, 0, 0,
        desktop_tex.Get(), 0, &src_box
    );

    // Map and read pixels
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = context_->Map(staging_tex_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        std::fprintf(stderr, "[DxgiCapture] Map staging failed: 0x%08lx\n", hr);
        duplication_->ReleaseFrame();
        return false;
    }

    out.width    = region_.width;
    out.height   = region_.height;
    out.stride   = static_cast<int>(mapped.RowPitch);
    out.frame_id = ++frame_counter_;
    out.timestamp = Clock::now();

    const size_t row_bytes = static_cast<size_t>(region_.width) * 4;
    out.pixels.resize(static_cast<size_t>(region_.height) * row_bytes);

    const uint8_t* src = static_cast<const uint8_t*>(mapped.pData);
    for (int y = 0; y < region_.height; ++y) {
        std::memcpy(out.pixels.data() + y * row_bytes,
                    src + y * mapped.RowPitch,
                    row_bytes);
    }
    out.stride = static_cast<int>(row_bytes);

    context_->Unmap(staging_tex_.Get(), 0);
    duplication_->ReleaseFrame();

    return true;
}

#else // Non-Windows stub

bool DxgiCapture::initialize(const CaptureRegion& region) {
    region_   = region;
    screen_w_ = 1920;
    screen_h_ = 1080;
    std::printf("[DxgiCapture] Stub initialized (non-Windows build)\n");
    return true;
}

void DxgiCapture::shutdown() {}

bool DxgiCapture::acquire_frame(CapturedFrame& /*out*/) {
    return false;
}

#endif // _WIN32

} // namespace tracker
