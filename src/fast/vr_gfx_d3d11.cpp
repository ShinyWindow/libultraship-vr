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

#include <array>

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

    bool ConfigureVisibilityMask(bool enable) override {
        mVisibilityMaskConfigured = false;
        mVisibilityMaskEyesForPass = 0x3u;
        for (auto& mask : mVisibilityMasks) {
            mask.vertexBuffer.Reset();
            mask.indexBuffer.Reset();
            mask.indexCount = 0;
        }
        if (!enable) {
            return true;
        }

        UINT support = 0;
        if (FAILED(mDevice->CheckFormatSupport(DXGI_FORMAT_D32_FLOAT_S8X24_UINT, &support)) ||
            (support & D3D11_FORMAT_SUPPORT_DEPTH_STENCIL) == 0) {
            spdlog::info("[VR] Hidden-area mask unavailable: D32_FLOAT_S8X24 depth/stencil is unsupported");
            return false;
        }
        if (!CreateVisibilityMaskPipeline()) {
            return false;
        }
        mVisibilityMaskConfigured = true;
        return true;
    }

    void SetVisibilityMask(uint32_t eye, const float* clipXy, uint32_t vertexCount, const uint32_t* indices,
                           uint32_t indexCount) override {
        if (eye >= mVisibilityMasks.size()) {
            return;
        }
        auto& mask = mVisibilityMasks[eye];
        mask.vertexBuffer.Reset();
        mask.indexBuffer.Reset();
        mask.indexCount = 0;
        if (!mVisibilityMaskConfigured || clipXy == nullptr || indices == nullptr || vertexCount == 0 ||
            indexCount == 0) {
            return;
        }

        D3D11_BUFFER_DESC vbDesc = {};
        vbDesc.ByteWidth = vertexCount * sizeof(float) * 2;
        vbDesc.Usage = D3D11_USAGE_IMMUTABLE;
        vbDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        D3D11_SUBRESOURCE_DATA vbData = {};
        vbData.pSysMem = clipXy;
        HRESULT hr = mDevice->CreateBuffer(&vbDesc, &vbData, mask.vertexBuffer.GetAddressOf());
        if (FAILED(hr)) {
            spdlog::warn("[VR] Hidden-area mask eye {} vertex upload failed (0x{:08x}); rendering the full eye", eye,
                         (uint32_t)hr);
            return;
        }

        D3D11_BUFFER_DESC ibDesc = {};
        ibDesc.ByteWidth = indexCount * sizeof(uint32_t);
        ibDesc.Usage = D3D11_USAGE_IMMUTABLE;
        ibDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
        D3D11_SUBRESOURCE_DATA ibData = {};
        ibData.pSysMem = indices;
        hr = mDevice->CreateBuffer(&ibDesc, &ibData, mask.indexBuffer.GetAddressOf());
        if (FAILED(hr)) {
            spdlog::warn("[VR] Hidden-area mask eye {} index upload failed (0x{:08x}); rendering the full eye", eye,
                         (uint32_t)hr);
            mask.vertexBuffer.Reset();
            return;
        }
        mask.indexCount = indexCount;
    }

    void SetVisibilityMaskEyesForPass(uint32_t eyeMask) override {
        mVisibilityMaskEyesForPass = eyeMask & 0x3u;
    }

    bool VisibilityMaskActive() const override {
        return mVisibilityMaskConfigured &&
               (mVisibilityMasks[0].indexCount != 0 || mVisibilityMasks[1].indexCount != 0);
    }

    bool Attach(Target t, XrSwapchain handle, uint32_t w, uint32_t h, int64_t fmt) override {
        auto& sc = mTargets[(int)t];
        Detach(t);
        sc.width = w;
        sc.height = h;
        const bool multiview = t == Target::EyeArray;
        const UINT layerCount = multiview ? 2u : 1u;
        const bool eyeTarget = t == Target::Eye0 || t == Target::Eye1 || multiview;
        const bool useVisibilityMask = eyeTarget && mVisibilityMaskConfigured;
        const DXGI_FORMAT depthFormat =
            useVisibilityMask ? DXGI_FORMAT_D32_FLOAT_S8X24_UINT : DXGI_FORMAT_D32_FLOAT;
        sc.hasStencil = useVisibilityMask;

        uint32_t count = 0;
        xrEnumerateSwapchainImages(handle, 0, &count, nullptr);
        sc.images.resize(count, { XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR });
        xrEnumerateSwapchainImages(handle, count, &count,
                                   reinterpret_cast<XrSwapchainImageBaseHeader*>(sc.images.data()));
        sc.rtvs.resize(count);
        sc.dsvs.resize(count);
        sc.depth.resize(count);
        if (multiview && useVisibilityMask) {
            sc.eyeDsvs.resize(count);
        }

        for (uint32_t i = 0; i < count; i++) {
            D3D11_TEXTURE2D_DESC color_desc = {};
            sc.images[i].texture->GetDesc(&color_desc);
            if (color_desc.ArraySize < layerCount) {
                spdlog::error("[VR] Target {} image {} has {} layers, but {} are required", (int)t, i,
                              color_desc.ArraySize, layerCount);
                return false;
            }

            D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
            rtv_desc.Format = mViewFormat;
            if (multiview) {
                // One view spans both OpenXR swapchain layers. The multiview geometry shader selects
                // the destination eye with SV_RenderTargetArrayIndex.
                rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
                rtv_desc.Texture2DArray.MipSlice = 0;
                rtv_desc.Texture2DArray.FirstArraySlice = 0;
                rtv_desc.Texture2DArray.ArraySize = layerCount;
            } else {
                rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
                rtv_desc.Texture2D.MipSlice = 0;
            }
            HRESULT hr = mDevice->CreateRenderTargetView(sc.images[i].texture, &rtv_desc, sc.rtvs[i].GetAddressOf());
            if (FAILED(hr)) {
                spdlog::error("[VR] Failed to create RTV for target {} image {} (0x{:08x})", (int)t, i, (uint32_t)hr);
                return false;
            }

            D3D11_TEXTURE2D_DESC depth_desc = {};
            depth_desc.Width = w;
            depth_desc.Height = h;
            depth_desc.MipLevels = 1;
            depth_desc.ArraySize = layerCount;
            depth_desc.Format = depthFormat;
            depth_desc.SampleDesc.Count = 1;
            depth_desc.Usage = D3D11_USAGE_DEFAULT;
            depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
            hr = mDevice->CreateTexture2D(&depth_desc, nullptr, sc.depth[i].GetAddressOf());
            if (FAILED(hr)) {
                if (useVisibilityMask) {
                    spdlog::warn("[VR] Hidden-area stencil depth allocation failed (0x{:08x}); retrying with the "
                                 "ordinary full-eye renderer",
                                 (uint32_t)hr);
                    mVisibilityMaskConfigured = false;
                    Detach(t);
                    return Attach(t, handle, w, h, fmt);
                }
                spdlog::error("[VR] Failed to create depth texture for target {} image {} (0x{:08x})", (int)t, i,
                              (uint32_t)hr);
                return false;
            }

            D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
            dsv_desc.Format = depthFormat;
            if (multiview) {
                dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
                dsv_desc.Texture2DArray.MipSlice = 0;
                dsv_desc.Texture2DArray.FirstArraySlice = 0;
                dsv_desc.Texture2DArray.ArraySize = layerCount;
            } else {
                dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
                dsv_desc.Texture2D.MipSlice = 0;
            }
            hr = mDevice->CreateDepthStencilView(sc.depth[i].Get(), &dsv_desc, sc.dsvs[i].GetAddressOf());
            if (FAILED(hr)) {
                if (useVisibilityMask) {
                    spdlog::warn("[VR] Hidden-area stencil view creation failed (0x{:08x}); retrying with the "
                                 "ordinary full-eye renderer",
                                 (uint32_t)hr);
                    mVisibilityMaskConfigured = false;
                    Detach(t);
                    return Attach(t, handle, w, h, fmt);
                }
                spdlog::error("[VR] Failed to create DSV for target {} image {} (0x{:08x})", (int)t, i, (uint32_t)hr);
                return false;
            }

            if (multiview && useVisibilityMask) {
                for (UINT eye = 0; eye < 2; ++eye) {
                    D3D11_DEPTH_STENCIL_VIEW_DESC eyeDsvDesc = {};
                    eyeDsvDesc.Format = depthFormat;
                    eyeDsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
                    eyeDsvDesc.Texture2DArray.MipSlice = 0;
                    eyeDsvDesc.Texture2DArray.FirstArraySlice = eye;
                    eyeDsvDesc.Texture2DArray.ArraySize = 1;
                    hr = mDevice->CreateDepthStencilView(sc.depth[i].Get(), &eyeDsvDesc,
                                                         sc.eyeDsvs[i][eye].GetAddressOf());
                    if (FAILED(hr)) {
                        spdlog::warn("[VR] Hidden-area eye-slice stencil view failed (0x{:08x}); retrying with the "
                                     "ordinary full-eye renderer",
                                     (uint32_t)hr);
                        mVisibilityMaskConfigured = false;
                        Detach(t);
                        return Attach(t, handle, w, h, fmt);
                    }
                }
            }
        }
        (void)fmt;
        return true;
    }

    void Detach(Target t) override {
        auto& sc = mTargets[(int)t];
        sc.rtvs.clear();
        sc.dsvs.clear();
        sc.eyeDsvs.clear();
        sc.depth.clear();
        sc.images.clear();
        sc.width = sc.height = 0;
        sc.hasStencil = false;
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
        const UINT clearFlags = D3D11_CLEAR_DEPTH | (sc.hasStencil ? D3D11_CLEAR_STENCIL : 0);
        mContext->ClearDepthStencilView(dsv, clearFlags, 1.0f, 0);

        D3D11_VIEWPORT viewport = {};
        viewport.TopLeftX = 0;
        viewport.TopLeftY = 0;
        viewport.Width = static_cast<float>(sc.width);
        viewport.Height = static_cast<float>(sc.height);
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;
        mContext->RSSetViewports(1, &viewport);

        const bool visibilityMask = DrawVisibilityMask(t, image);
        // The backend's viewport/scissor Y-flip needs the bound target's height.
        mDx->SetExternalRenderTarget((int32_t)sc.height, t == Target::EyeArray, visibilityMask);
    }

    void Rebind(Target t, uint32_t image) override {
        auto& sc = mTargets[(int)t];
        if (image >= sc.rtvs.size()) {
            return;
        }
        ID3D11RenderTargetView* rtv = sc.rtvs[image].Get();
        ID3D11DepthStencilView* dsv = sc.dsvs[image].Get();
        mContext->OMSetRenderTargets(1, &rtv, dsv);
        mDx->SetExternalRenderTarget((int32_t)sc.height, t == Target::EyeArray, HasVisibilityMask(t));
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
    // acquired. EyeArray is a two-layer texture; only layer 0 (the left eye) goes to the desktop.
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
        // Copying a two-layer source with CopyResource into the one-layer mirror is invalid. A
        // subresource copy works for both ordinary targets and EyeArray and preserves the raw bits.
        const UINT srcSubresource = D3D11CalcSubresource(0, 0, sd.MipLevels);
        D3D11_BOX source = { 0, 0, 0, sd.Width, sd.Height, 1 };
        mContext->CopySubresourceRegion(c.tex.Get(), 0, 0, 0, 0, src, srcSubresource, &source);
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

    bool SupportsMultiview() const override {
        // Feature level 10 includes geometry shaders and render-target arrays. The renderer uses a
        // geometry shader to duplicate each triangle into the two OpenXR array layers.
        return mDevice != nullptr && mDevice->GetFeatureLevel() >= D3D_FEATURE_LEVEL_10_0;
    }

    void TimingFrameBegin() override {
        ResolveTimingQueries();
        mActiveTiming = -1;
        for (size_t attempt = 0; attempt < mTiming.size(); ++attempt) {
            mTimingWrite = (mTimingWrite + 1) % mTiming.size();
            auto& slot = mTiming[mTimingWrite];
            if (slot.pending) {
                continue;
            }
            if (!EnsureTimingQueries(slot)) {
                return;
            }
            slot.phaseStarted.fill(false);
            mContext->Begin(slot.disjoint.Get());
            mContext->End(slot.frameStart.Get());
            mActiveTiming = static_cast<int>(mTimingWrite);
            return;
        }
    }

    void TimingPhaseBegin(TimingPhase phase) override {
        if (mActiveTiming < 0) return;
        const size_t p = static_cast<size_t>(phase);
        if (p >= static_cast<size_t>(TimingPhase::Count)) return;
        auto& slot = mTiming[mActiveTiming];
        mContext->End(slot.phaseStart[p].Get());
        slot.phaseStarted[p] = true;
    }

    void TimingPhaseEnd(TimingPhase phase) override {
        if (mActiveTiming < 0) return;
        const size_t p = static_cast<size_t>(phase);
        if (p >= static_cast<size_t>(TimingPhase::Count) || !mTiming[mActiveTiming].phaseStarted[p]) return;
        mContext->End(mTiming[mActiveTiming].phaseEnd[p].Get());
    }

    void TimingFrameEnd() override {
        if (mActiveTiming < 0) return;
        auto& slot = mTiming[mActiveTiming];
        mContext->End(slot.frameEnd.Get());
        mContext->End(slot.disjoint.Get());
        slot.pending = true;
        mActiveTiming = -1;
    }

    bool TimingGetLatest(float* eyes, float* hud, float* desktop, float* total) override {
        if (!mTimingValid) return false;
        if (eyes) *eyes = mLatestTiming[0];
        if (hud) *hud = mLatestTiming[1];
        if (desktop) *desktop = mLatestTiming[2];
        if (total) *total = mLatestTiming[3];
        return true;
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
        ConfigureVisibilityMask(false);
        mVisibilityMaskVs.Reset();
        mVisibilityMaskInputLayout.Reset();
        mVisibilityMaskDepthState.Reset();
        mVisibilityMaskRasterizerState.Reset();
    }

  private:
    struct TimingQueries {
        ComPtr<ID3D11Query> disjoint;
        ComPtr<ID3D11Query> frameStart;
        ComPtr<ID3D11Query> frameEnd;
        std::array<ComPtr<ID3D11Query>, static_cast<size_t>(TimingPhase::Count)> phaseStart;
        std::array<ComPtr<ID3D11Query>, static_cast<size_t>(TimingPhase::Count)> phaseEnd;
        std::array<bool, static_cast<size_t>(TimingPhase::Count)> phaseStarted{};
        bool pending = false;
    };

    bool EnsureTimingQueries(TimingQueries& slot) {
        if (slot.disjoint) return true;
        D3D11_QUERY_DESC desc = {};
        desc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
        if (FAILED(mDevice->CreateQuery(&desc, slot.disjoint.GetAddressOf()))) return false;
        desc.Query = D3D11_QUERY_TIMESTAMP;
        if (FAILED(mDevice->CreateQuery(&desc, slot.frameStart.GetAddressOf())) ||
            FAILED(mDevice->CreateQuery(&desc, slot.frameEnd.GetAddressOf()))) return false;
        for (size_t p = 0; p < static_cast<size_t>(TimingPhase::Count); ++p) {
            if (FAILED(mDevice->CreateQuery(&desc, slot.phaseStart[p].GetAddressOf())) ||
                FAILED(mDevice->CreateQuery(&desc, slot.phaseEnd[p].GetAddressOf()))) return false;
        }
        return true;
    }

    void ResolveTimingQueries() {
        for (auto& slot : mTiming) {
            if (!slot.pending) continue;
            D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint = {};
            if (mContext->GetData(slot.disjoint.Get(), &disjoint, sizeof(disjoint), D3D11_ASYNC_GETDATA_DONOTFLUSH) !=
                S_OK) continue;
            slot.pending = false;
            if (disjoint.Disjoint || disjoint.Frequency == 0) continue;
            UINT64 frameStart = 0, frameEnd = 0;
            if (mContext->GetData(slot.frameStart.Get(), &frameStart, sizeof(frameStart), 0) != S_OK ||
                mContext->GetData(slot.frameEnd.Get(), &frameEnd, sizeof(frameEnd), 0) != S_OK) continue;
            const double toMs = 1000.0 / static_cast<double>(disjoint.Frequency);
            for (size_t p = 0; p < static_cast<size_t>(TimingPhase::Count); ++p) {
                mLatestTiming[p] = 0.0f;
                if (!slot.phaseStarted[p]) continue;
                UINT64 start = 0, end = 0;
                if (mContext->GetData(slot.phaseStart[p].Get(), &start, sizeof(start), 0) == S_OK &&
                    mContext->GetData(slot.phaseEnd[p].Get(), &end, sizeof(end), 0) == S_OK && end >= start) {
                    mLatestTiming[p] = static_cast<float>((end - start) * toMs);
                }
            }
            mLatestTiming[3] = frameEnd >= frameStart ? static_cast<float>((frameEnd - frameStart) * toMs) : 0.0f;
            mTimingValid = true;
        }
    }

    bool CreateVisibilityMaskPipeline() {
        if (mVisibilityMaskVs && mVisibilityMaskInputLayout && mVisibilityMaskDepthState &&
            mVisibilityMaskRasterizerState) {
            return true;
        }

        static constexpr char kShader[] = R"(
struct VSInput {
    float2 position : POSITION;
};

float4 VSMain(VSInput input) : SV_Position {
    return float4(input.position, 0.0f, 1.0f);
}
)";

        HMODULE compilerModule = GetModuleHandleW(L"D3DCompiler_47.dll");
        auto compile = compilerModule != nullptr
                           ? reinterpret_cast<pD3DCompile>(GetProcAddress(compilerModule, "D3DCompile"))
                           : nullptr;
        if (compile == nullptr) {
            spdlog::warn("[VR] Hidden-area mask unavailable: D3DCompiler_47.dll is not loaded");
            return false;
        }

        ComPtr<ID3DBlob> shader;
        ComPtr<ID3DBlob> errors;
        HRESULT hr = compile(kShader, sizeof(kShader) - 1, "vr_visibility_mask", nullptr, nullptr, "VSMain",
                             "vs_4_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, shader.GetAddressOf(),
                             errors.GetAddressOf());
        if (FAILED(hr)) {
            const char* error = errors ? static_cast<const char*>(errors->GetBufferPointer()) : "unknown error";
            spdlog::warn("[VR] Hidden-area mask vertex shader compilation failed: {}", error);
            return false;
        }

        hr = mDevice->CreateVertexShader(shader->GetBufferPointer(), shader->GetBufferSize(), nullptr,
                                         mVisibilityMaskVs.GetAddressOf());
        if (FAILED(hr)) {
            spdlog::warn("[VR] Hidden-area mask vertex shader creation failed (0x{:08x})", (uint32_t)hr);
            return false;
        }
        D3D11_INPUT_ELEMENT_DESC element = { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
                                             D3D11_INPUT_PER_VERTEX_DATA, 0 };
        hr = mDevice->CreateInputLayout(&element, 1, shader->GetBufferPointer(), shader->GetBufferSize(),
                                        mVisibilityMaskInputLayout.GetAddressOf());
        if (FAILED(hr)) {
            spdlog::warn("[VR] Hidden-area mask input layout creation failed (0x{:08x})", (uint32_t)hr);
            return false;
        }

        D3D11_DEPTH_STENCIL_DESC depth = {};
        depth.DepthEnable = false;
        depth.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        depth.DepthFunc = D3D11_COMPARISON_ALWAYS;
        depth.StencilEnable = true;
        depth.StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK;
        depth.StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK;
        depth.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
        depth.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
        depth.FrontFace.StencilPassOp = D3D11_STENCIL_OP_REPLACE;
        depth.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
        depth.BackFace = depth.FrontFace;
        hr = mDevice->CreateDepthStencilState(&depth, mVisibilityMaskDepthState.GetAddressOf());
        if (FAILED(hr)) {
            spdlog::warn("[VR] Hidden-area mask stencil state creation failed (0x{:08x})", (uint32_t)hr);
            return false;
        }

        D3D11_RASTERIZER_DESC rasterizer = {};
        rasterizer.FillMode = D3D11_FILL_SOLID;
        rasterizer.CullMode = D3D11_CULL_NONE;
        rasterizer.DepthClipEnable = true;
        hr = mDevice->CreateRasterizerState(&rasterizer, mVisibilityMaskRasterizerState.GetAddressOf());
        if (FAILED(hr)) {
            spdlog::warn("[VR] Hidden-area mask rasterizer state creation failed (0x{:08x})", (uint32_t)hr);
            return false;
        }
        return true;
    }

    bool HasVisibilityMask(Target t) const {
        if (!mVisibilityMaskConfigured || mVisibilityMaskEyesForPass == 0 || !mTargets[(int)t].hasStencil) {
            return false;
        }
        if (t == Target::Eye0) {
            return (mVisibilityMaskEyesForPass & 0x1u) != 0 && mVisibilityMasks[0].indexCount != 0;
        }
        if (t == Target::Eye1) {
            return (mVisibilityMaskEyesForPass & 0x2u) != 0 && mVisibilityMasks[1].indexCount != 0;
        }
        if (t == Target::EyeArray) {
            return ((mVisibilityMaskEyesForPass & 0x1u) != 0 && mVisibilityMasks[0].indexCount != 0) ||
                   ((mVisibilityMaskEyesForPass & 0x2u) != 0 && mVisibilityMasks[1].indexCount != 0);
        }
        return false;
    }

    bool DrawVisibilityMask(Target t, uint32_t image) {
        auto& sc = mTargets[(int)t];
        if (!HasVisibilityMask(t) || image >= sc.dsvs.size() ||
            (t == Target::EyeArray && image >= sc.eyeDsvs.size())) {
            return false;
        }

        mContext->IASetInputLayout(mVisibilityMaskInputLayout.Get());
        mContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        mContext->VSSetShader(mVisibilityMaskVs.Get(), nullptr, 0);
        mContext->GSSetShader(nullptr, nullptr, 0);
        mContext->PSSetShader(nullptr, nullptr, 0);
        mContext->RSSetState(mVisibilityMaskRasterizerState.Get());
        mContext->OMSetDepthStencilState(mVisibilityMaskDepthState.Get(), 1);

        auto drawEye = [&](uint32_t eye, ID3D11DepthStencilView* dsv) {
            const auto& mask = mVisibilityMasks[eye];
            if ((mVisibilityMaskEyesForPass & (1u << eye)) == 0 || mask.indexCount == 0 || dsv == nullptr) {
                return;
            }
            mContext->OMSetRenderTargets(0, nullptr, dsv);
            ID3D11Buffer* vertexBuffer = mask.vertexBuffer.Get();
            const UINT stride = sizeof(float) * 2;
            const UINT offset = 0;
            mContext->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
            mContext->IASetIndexBuffer(mask.indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
            mContext->DrawIndexed(mask.indexCount, 0, 0);
        };

        if (t == Target::EyeArray) {
            drawEye(0, sc.eyeDsvs[image][0].Get());
            drawEye(1, sc.eyeDsvs[image][1].Get());
        } else {
            const uint32_t eye = t == Target::Eye1 ? 1u : 0u;
            drawEye(eye, sc.dsvs[image].Get());
        }

        ID3D11RenderTargetView* rtv = sc.rtvs[image].Get();
        mContext->OMSetRenderTargets(1, &rtv, sc.dsvs[image].Get());
        mDx->InvalidatePipelineStateAfterExternalDraw();
        return true;
    }

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
        bool hasStencil = false;
        std::vector<XrSwapchainImageD3D11KHR> images;
        std::vector<ComPtr<ID3D11RenderTargetView>> rtvs;
        std::vector<ComPtr<ID3D11DepthStencilView>> dsvs;
        std::vector<std::array<ComPtr<ID3D11DepthStencilView>, 2>> eyeDsvs;
        std::vector<ComPtr<ID3D11Texture2D>> depth;
    };
    struct VisibilityMask {
        ComPtr<ID3D11Buffer> vertexBuffer;
        ComPtr<ID3D11Buffer> indexBuffer;
        uint32_t indexCount = 0;
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
    bool mVisibilityMaskConfigured = false;
    uint32_t mVisibilityMaskEyesForPass = 0x3u;
    std::array<VisibilityMask, 2> mVisibilityMasks;
    ComPtr<ID3D11VertexShader> mVisibilityMaskVs;
    ComPtr<ID3D11InputLayout> mVisibilityMaskInputLayout;
    ComPtr<ID3D11DepthStencilState> mVisibilityMaskDepthState;
    ComPtr<ID3D11RasterizerState> mVisibilityMaskRasterizerState;
    std::array<TimingQueries, 8> mTiming;
    size_t mTimingWrite = 0;
    int mActiveTiming = -1;
    std::array<float, 4> mLatestTiming{};
    bool mTimingValid = false;
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
