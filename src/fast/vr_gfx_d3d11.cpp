// SOH [VR] D3D11 leaf of the VR graphics seam (vr_gfx.h). Everything in the VR layer that names
// D3D11 lives here; vr_openxr.cpp stays API-neutral.
#if defined(ENABLE_DX11) && defined(ENABLE_VR)

#define NOMINMAX
#include <d3d11_1.h>
#include <dxgi.h>
#include <wrl/client.h>

#define XR_USE_GRAPHICS_API_D3D11
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <spdlog/spdlog.h>

#include "vr_gfx.h"
#include "fast/backends/gfx_direct3d_common.h"

using Microsoft::WRL::ComPtr;

namespace vrgfx {
namespace {

class D3D11Backend final : public Backend {
  public:
    explicit D3D11Backend(Fast::GfxRenderingAPIDX11* dx) : mDx(dx) {
        mDevice = dx->mDevice.Get();
        mContext = dx->mContext.Get();
        if (mContext) {
            mContext->QueryInterface(__uuidof(ID3D11DeviceContext1),
                                     reinterpret_cast<void**>(mContext1.GetAddressOf()));
        }
        mBinding.device = mDevice;
    }

    bool Valid() const {
        return mDevice != nullptr && mContext != nullptr;
    }

    const char* ApiName() const override {
        return "DirectX 11";
    }

    const char* RequiredInstanceExtension() const override {
        return XR_KHR_D3D11_ENABLE_EXTENSION_NAME;
    }

    bool CheckRequirements(XrInstance instance, XrSystemId system, std::string* why) override {
        PFN_xrGetD3D11GraphicsRequirementsKHR getReqs = nullptr;
        xrGetInstanceProcAddr(instance, "xrGetD3D11GraphicsRequirementsKHR",
                              reinterpret_cast<PFN_xrVoidFunction*>(&getReqs));
        XrGraphicsRequirementsD3D11KHR reqs = { XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR };
        if (getReqs == nullptr || XR_FAILED(getReqs(instance, system, &reqs))) {
            // Historically ignored; keep going and let xrCreateSession decide.
            spdlog::warn("[VR] xrGetD3D11GraphicsRequirementsKHR unavailable or failed");
            return true;
        }

        // Log-only checks (plan §11 item 7): the runtime's adapter vs ours, and the feature level.
        LogAdapters(reqs.adapterLuid);
        if (mDevice->GetFeatureLevel() < reqs.minFeatureLevel) {
            spdlog::warn("[VR] D3D11 feature level {:x} is below the runtime minimum {:x}",
                         (int)mDevice->GetFeatureLevel(), (int)reqs.minFeatureLevel);
        }
        (void)why;
        return true;
    }

    const void* SessionBinding() override {
        return &mBinding;
    }

    // The game outputs gamma-encoded (sRGB) colors. The swapchain must be created with an SRGB
    // format so the compositor decodes them correctly; a UNORM swapchain makes the compositor treat
    // gamma values as linear and re-encode them, washing the image out. Writes still go through a
    // UNORM view so the bits land in the texture verbatim.
    int64_t ChooseColorFormat(const std::vector<int64_t>& formats) override {
        int64_t chosen = formats.empty() ? DXGI_FORMAT_R8G8B8A8_UNORM : formats[0]; // fallback: first offered
        bool found = false;
        for (int64_t fmt : formats) {
            if (fmt == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) {
                chosen = fmt;
                found = true;
                break;
            }
        }
        if (!found) {
            // SRGB not available: UNORM as next best, accept the gamma mismatch.
            for (int64_t fmt : formats) {
                if (fmt == DXGI_FORMAT_R8G8B8A8_UNORM) {
                    chosen = fmt;
                    break;
                }
            }
        }
        // OpenXR D3D11 swapchain textures are allocated typeless, so views may use either variant
        // of the format family. The UNORM variant for RTVs/SRVs stores and reads the game's
        // already-gamma-encoded output without any extra conversion.
        mViewFormat = (chosen == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) ? DXGI_FORMAT_R8G8B8A8_UNORM
                                                                   : static_cast<DXGI_FORMAT>(chosen);
        spdlog::info("[VR] Swapchain format: {} (UNORM={}, SRGB={}), view format: {}", chosen,
                     (int)DXGI_FORMAT_R8G8B8A8_UNORM, (int)DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, (int)mViewFormat);
        return chosen;
    }

    bool Attach(Target t, XrSwapchain handle, uint32_t w, uint32_t h, int64_t fmt) override {
        auto& sc = mTargets[(int)t];
        Detach(t);
        sc.width = w;
        sc.height = h;

        uint32_t count = 0;
        xrEnumerateSwapchainImages(handle, 0, &count, nullptr);
        sc.images.resize(count, { XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR });
        xrEnumerateSwapchainImages(handle, count, &count,
                                   reinterpret_cast<XrSwapchainImageBaseHeader*>(sc.images.data()));
        sc.rtvs.resize(count);
        sc.dsvs.resize(count);
        sc.depth.resize(count);

        for (uint32_t i = 0; i < count; i++) {
            D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
            rtv_desc.Format = mViewFormat;
            rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            rtv_desc.Texture2D.MipSlice = 0;
            HRESULT hr = mDevice->CreateRenderTargetView(sc.images[i].texture, &rtv_desc, sc.rtvs[i].GetAddressOf());
            if (FAILED(hr)) {
                spdlog::error("[VR] Failed to create RTV for target {} image {} (0x{:08x})", (int)t, i, (uint32_t)hr);
                return false;
            }

            D3D11_TEXTURE2D_DESC depth_desc = {};
            depth_desc.Width = w;
            depth_desc.Height = h;
            depth_desc.MipLevels = 1;
            depth_desc.ArraySize = 1;
            depth_desc.Format = DXGI_FORMAT_D32_FLOAT;
            depth_desc.SampleDesc.Count = 1;
            depth_desc.Usage = D3D11_USAGE_DEFAULT;
            depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
            hr = mDevice->CreateTexture2D(&depth_desc, nullptr, sc.depth[i].GetAddressOf());
            if (FAILED(hr)) {
                spdlog::error("[VR] Failed to create depth texture for target {} image {} (0x{:08x})", (int)t, i,
                              (uint32_t)hr);
                return false;
            }

            D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
            dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
            dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
            dsv_desc.Texture2D.MipSlice = 0;
            hr = mDevice->CreateDepthStencilView(sc.depth[i].Get(), &dsv_desc, sc.dsvs[i].GetAddressOf());
            if (FAILED(hr)) {
                spdlog::error("[VR] Failed to create DSV for target {} image {} (0x{:08x})", (int)t, i, (uint32_t)hr);
                return false;
            }
        }
        (void)fmt;
        return true;
    }

    void Detach(Target t) override {
        auto& sc = mTargets[(int)t];
        sc.rtvs.clear();
        sc.dsvs.clear();
        sc.depth.clear();
        sc.images.clear();
        sc.width = sc.height = 0;
    }

    void BeginPass(Target t, uint32_t image, const float clearRgba[4]) override {
        auto& sc = mTargets[(int)t];
        if (image >= sc.rtvs.size()) {
            return;
        }
        ID3D11RenderTargetView* rtv = sc.rtvs[image].Get();
        ID3D11DepthStencilView* dsv = sc.dsvs[image].Get();
        mContext->OMSetRenderTargets(1, &rtv, dsv);
        mContext->ClearRenderTargetView(rtv, clearRgba);
        mContext->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);

        D3D11_VIEWPORT viewport = {};
        viewport.TopLeftX = 0;
        viewport.TopLeftY = 0;
        viewport.Width = static_cast<float>(sc.width);
        viewport.Height = static_cast<float>(sc.height);
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;
        mContext->RSSetViewports(1, &viewport);

        // The backend's viewport/scissor Y-flip needs the bound target's height.
        mDx->SetRenderTargetHeight((int32_t)sc.height);
    }

    void Rebind(Target t, uint32_t image) override {
        auto& sc = mTargets[(int)t];
        if (image >= sc.rtvs.size()) {
            return;
        }
        ID3D11RenderTargetView* rtv = sc.rtvs[image].Get();
        ID3D11DepthStencilView* dsv = sc.dsvs[image].Get();
        mContext->OMSetRenderTargets(1, &rtv, dsv);
        mDx->SetRenderTargetHeight((int32_t)sc.height);
    }

    void ClearDepth(Target t, uint32_t image) override {
        auto& sc = mTargets[(int)t];
        if (image >= sc.dsvs.size() || sc.dsvs[image] == nullptr) {
            return;
        }
        mContext->ClearDepthStencilView(sc.dsvs[image].Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    }

    void ClearColorRects(Target t, uint32_t image, const Rect* r, int n, const float rgbaPremul[4]) override {
        auto& sc = mTargets[(int)t];
        if (!mContext1 || image >= sc.rtvs.size() || n <= 0) {
            return;
        }
        D3D11_RECT rects[32];
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (r[i].w <= 0 || r[i].h <= 0) {
                continue;
            }
            rects[k++] = { (LONG)r[i].x, (LONG)r[i].y, (LONG)(r[i].x + r[i].w), (LONG)(r[i].y + r[i].h) };
            if (k == 32) {
                mContext1->ClearView(sc.rtvs[image].Get(), rgbaPremul, rects, k);
                k = 0;
            }
        }
        if (k > 0) {
            mContext1->ClearView(sc.rtvs[image].Get(), rgbaPremul, rects, k);
        }
    }

    // A swapchain image can't be read after release, so each pass copies its image here while still
    // acquired. Same family + size as the source, so a straight CopyResource is valid.
    void CopyToMirror(Target t, uint32_t image, int slot) override {
        auto& sc = mTargets[(int)t];
        if (slot < 0 || slot >= kMirrorCount || image >= sc.images.size() || sc.images[image].texture == nullptr) {
            return;
        }
        ID3D11Texture2D* src = sc.images[image].texture;
        auto& c = mMirror[slot];
        D3D11_TEXTURE2D_DESC sd;
        src->GetDesc(&sd);
        if (!c.tex || c.w != sd.Width || c.h != sd.Height) {
            D3D11_TEXTURE2D_DESC d = sd;
            d.MipLevels = 1;
            d.ArraySize = 1;
            d.SampleDesc.Count = 1;
            d.SampleDesc.Quality = 0;
            d.Usage = D3D11_USAGE_DEFAULT;
            d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            d.CPUAccessFlags = 0;
            d.MiscFlags = 0;
            c.srv.Reset();
            c.tex.Reset();
            c.w = c.h = 0;
            if (FAILED(mDevice->CreateTexture2D(&d, nullptr, c.tex.GetAddressOf()))) {
                // Non-fatal: the headset still renders, the companion window just won't show it.
                spdlog::warn("[VR] Failed to create desktop mirror texture (slot {})", slot);
                c.tex.Reset();
                return;
            }
            D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
            srv_desc.Format = mViewFormat;
            srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            srv_desc.Texture2D.MipLevels = 1;
            if (FAILED(mDevice->CreateShaderResourceView(c.tex.Get(), &srv_desc, c.srv.GetAddressOf()))) {
                spdlog::warn("[VR] Failed to create desktop mirror view (slot {})", slot);
                c.srv.Reset();
                c.tex.Reset();
                return;
            }
            c.w = sd.Width;
            c.h = sd.Height;
        }
        mContext->CopyResource(c.tex.Get(), src);
    }

    void* MirrorTextureId(int slot) override {
        if (slot < 0 || slot >= kMirrorCount) {
            return nullptr;
        }
        return mMirror[slot].srv.Get();
    }

    bool MirrorFlipV() const override {
        return false;
    }

    void Shutdown() override {
        for (int t = 0; t < (int)Target::Count; t++) {
            Detach((Target)t);
        }
        for (auto& c : mMirror) {
            c.srv.Reset();
            c.tex.Reset();
            c.w = c.h = 0;
        }
    }

  private:
    static std::string AdapterName(const wchar_t* w) {
        const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
        if (n <= 1) {
            return "unknown";
        }
        std::string s(n - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
        return s;
    }

    // The GPU the runtime requires next to the one our device runs on. A mismatch fails session
    // creation on most runtimes (XR_ERROR_GRAPHICS_DEVICE_INVALID) or shows black on others: typical
    // on laptops (integrated + dedicated GPU) and desktops with the integrated GPU still enabled.
    void LogAdapters(const LUID& required) {
        std::string ours = "unknown";
        std::string wanted = "unknown (not found among this machine's adapters)";
        LUID oursLuid = {};
        ComPtr<IDXGIDevice> dxgiDevice;
        ComPtr<IDXGIAdapter> adapter;
        if (SUCCEEDED(mDevice->QueryInterface(IID_PPV_ARGS(&dxgiDevice))) &&
            SUCCEEDED(dxgiDevice->GetAdapter(&adapter))) {
            DXGI_ADAPTER_DESC desc;
            if (SUCCEEDED(adapter->GetDesc(&desc))) {
                ours = AdapterName(desc.Description);
                oursLuid = desc.AdapterLuid;
            }
            ComPtr<IDXGIFactory1> factory;
            if (SUCCEEDED(adapter->GetParent(IID_PPV_ARGS(&factory)))) {
                ComPtr<IDXGIAdapter1> a;
                for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
                    DXGI_ADAPTER_DESC1 d;
                    if (SUCCEEDED(a->GetDesc1(&d)) && d.AdapterLuid.LowPart == required.LowPart &&
                        d.AdapterLuid.HighPart == required.HighPart) {
                        wanted = AdapterName(d.Description);
                        break;
                    }
                }
            }
        }
        if (oursLuid.LowPart == required.LowPart && oursLuid.HighPart == required.HighPart) {
            spdlog::info("[VR] GPU: {} (the adapter the headset runtime requires)", ours);
        } else {
            spdlog::warn("[VR] GPU MISMATCH: the game renders on '{}' but the headset runtime requires '{}'. Session "
                         "creation will likely fail or the headset stays black. Turn VR on and restart the game: with "
                         "VR on at launch the game creates its device on the headset's GPU.",
                         ours, wanted);
        }
    }

    struct TargetImages {
        uint32_t width = 0, height = 0;
        std::vector<XrSwapchainImageD3D11KHR> images;
        std::vector<ComPtr<ID3D11RenderTargetView>> rtvs;
        std::vector<ComPtr<ID3D11DepthStencilView>> dsvs;
        std::vector<ComPtr<ID3D11Texture2D>> depth;
    };
    struct MirrorCopy {
        ComPtr<ID3D11Texture2D> tex;
        ComPtr<ID3D11ShaderResourceView> srv;
        uint32_t w = 0, h = 0;
    };

    Fast::GfxRenderingAPIDX11* mDx;
    ID3D11Device* mDevice = nullptr;
    ID3D11DeviceContext* mContext = nullptr;
    ComPtr<ID3D11DeviceContext1> mContext1; // ClearView for the HUD backings
    XrGraphicsBindingD3D11KHR mBinding = { XR_TYPE_GRAPHICS_BINDING_D3D11_KHR };
    DXGI_FORMAT mViewFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    TargetImages mTargets[(int)Target::Count];
    MirrorCopy mMirror[kMirrorCount];
};

} // namespace

const char* D3D11InstanceExtension() {
    return XR_KHR_D3D11_ENABLE_EXTENSION_NAME;
}

bool D3D11RequiredAdapterLuid(XrInstance instance, XrSystemId system, uint64_t* luid) {
    PFN_xrGetD3D11GraphicsRequirementsKHR getReqs = nullptr;
    xrGetInstanceProcAddr(instance, "xrGetD3D11GraphicsRequirementsKHR",
                          reinterpret_cast<PFN_xrVoidFunction*>(&getReqs));
    XrGraphicsRequirementsD3D11KHR reqs = { XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR };
    if (getReqs == nullptr || XR_FAILED(getReqs(instance, system, &reqs))) {
        return false;
    }
    *luid = ((uint64_t)(uint32_t)reqs.adapterLuid.HighPart << 32) | reqs.adapterLuid.LowPart;
    return true;
}

std::unique_ptr<Backend> CreateD3D11(Fast::GfxRenderingAPI* rapi) {
    if (rapi == nullptr) {
        return nullptr;
    }
    auto b = std::make_unique<D3D11Backend>(static_cast<Fast::GfxRenderingAPIDX11*>(rapi));
    if (!b->Valid()) {
        spdlog::error("[VR] D3D11 device not available");
        return nullptr;
    }
    return b;
}

} // namespace vrgfx

#endif // ENABLE_DX11 && ENABLE_VR
