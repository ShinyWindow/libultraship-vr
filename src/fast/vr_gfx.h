#pragma once

// SOH [VR] The graphics-API seam of the VR layer (VR-MULTI-API-PLAN.md §3.2).
//
// vr_openxr.cpp is the single OpenXR implementation (session, frame loop, poses, input, layers, HUD
// layout, mirror geometry) and never names a graphics API. Everything that does lives behind this
// interface, one leaf per API: vr_gfx_d3d11.cpp, vr_gfx_opengl.cpp, (later) vr_gfx_vulkan.cpp. Each
// leaf defines its own XR_USE_GRAPHICS_API_* before including openxr_platform.h.
//
// FROZEN after Phase 0: changes are additive, with a default implementation, and announced.
//
// Conventions every leaf meets (plan §3.4):
// - Orientation: XR images come out upright in the headset. Rects passed in are pixels with a
//   TOP-LEFT origin; a bottom-up API (GL) converts.
// - sRGB: the game's output is already gamma-encoded and is stored verbatim. Prefer an sRGB
//   swapchain format (the compositor decodes) written without re-encoding; fallback UNORM.
// - Depth: one private depth buffer per image, cleared to 1.0, never submitted.
// - Passes run only between xrWaitSwapchainImage and xrReleaseSwapchainImage of that image, and
//   every write is submitted by EndPass.
// - Mirror copies are taken while the image is still acquired, into textures the RUNNING ImGui
//   backend can sample, same verbatim bytes.

#include <openxr/openxr.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Fast {
class GfxRenderingAPI;
}

namespace vrgfx {

// Menu = the SoH (ImGui) menu panel, drawn by the running ImGui renderer backend into the bound
// image; Beam = the laser pointer's static gradient strip (filled once with ClearColorRects).
// (Menu/Beam added October 6, additive: the leaves size their per-target arrays by Count.)
enum class Target : int { Eye0, Eye1, Hud, Text, Screen, Menu, Beam, Count };

// Desktop mirror slots: 0 = left eye, 1 = HUD, 2 = text panel, 3 = flat-screen panel.
enum MirrorSlot : int { kMirrorEye = 0, kMirrorHud = 1, kMirrorText = 2, kMirrorScreen = 3, kMirrorCount = 4 };

struct Rect {
    int32_t x, y, w, h; // pixels, TOP-LEFT origin of the picture
};

class Backend {
  public:
    virtual ~Backend() = default;
    virtual const char* ApiName() const = 0; // "DirectX 11", "OpenGL", "Vulkan"

    // --- instance / session ---
    virtual const char* RequiredInstanceExtension() const = 0; // XR_KHR_D3D11_enable, ...
    virtual void AddInstanceExtensions(std::vector<const char*>& exts) {
    }
    // MUST call xrGet*GraphicsRequirementsKHR (the runtime refuses the session otherwise).
    // false = the running device can't drive this runtime; *why says so.
    virtual bool CheckRequirements(XrInstance instance, XrSystemId system, std::string* why) = 0;
    // XrGraphicsBinding*KHR chained into XrSessionCreateInfo::next. Valid until Shutdown.
    virtual const void* SessionBinding() = 0;

    // --- swapchains ---
    virtual int64_t ChooseColorFormat(const std::vector<int64_t>& offered) = 0;
    virtual XrSwapchainUsageFlags ExtraUsageFlags() const {
        return 0;
    }
    // Enumerate the swapchain's images and build a color target + private depth per image.
    virtual bool Attach(Target t, XrSwapchain sc, uint32_t w, uint32_t h, int64_t fmt) = 0;
    virtual void Detach(Target t) = 0;

    // --- passes ---
    // Bind, clear color + depth (1.0), full viewport, tell the renderer the target size/orientation.
    virtual void BeginPass(Target t, uint32_t image, const float clearRgba[4]) = 0;
    // Re-bind after the renderer drew into its own framebuffers mid-pass; no clear.
    virtual void Rebind(Target t, uint32_t image) = 0;
    // Mid-list Z fill.
    virtual void ClearDepth(Target t, uint32_t image) = 0;
    // Fill rects with one premultiplied colour (HUD backings, the test card). Doesn't change the
    // renderer's bound state as seen by the next draw.
    virtual void ClearColorRects(Target t, uint32_t image, const Rect* r, int n, const float rgbaPremul[4]) = 0;
    virtual void CopyToMirror(Target t, uint32_t image, int slot) = 0;
    // Leave the image releasable (Vulkan: layout + submit; D3D11/GL: nothing).
    virtual void EndPass(Target t, uint32_t image) {
    }

    // --- desktop mirror ---
    virtual void* MirrorTextureId(int slot) = 0; // ImTextureID for the running ImGui backend, or null
    virtual bool MirrorFlipV() const = 0;        // true when the copies are stored bottom-up (GL)

    // Layer submission: the core computes sub-image rects (wrist panels, text crop) top-left. True =
    // this runtime + API expects them bottom-left, and the core flips them before xrEndFrame.
    // (Added in Track GL; whether GL needs it is runtime-verified with the test card.)
    virtual bool SubImageYUp() const {
        return false;
    }

    virtual void Shutdown() = 0;
};

// The leaf for the running renderer, or null when that renderer has no VR leaf (compiled out or
// not supported). Never casts a renderer it wasn't built for.
std::unique_ptr<Backend> Create(int windowBackend, Fast::GfxRenderingAPI* rapi);

// Before the renderer exists (device creation, vr_probe_required_adapter): the instance extension a
// renderer's API needs, and the GPU the runtime requires for it as a 64-bit LUID
// ((HighPart << 32) | LowPart). Null / false when the API has no adapter requirement (OpenGL) or
// the runtime didn't say. Additive (added for the adapter fix, announced).
const char* InstanceExtensionFor(int windowBackend);
const char* ApiNameFor(int windowBackend); // Backend::ApiName() before the leaf exists ("DirectX 11", ...)
bool RequiredAdapterLuid(int windowBackend, XrInstance instance, XrSystemId system, uint64_t* luid);

// Leaf factories (each defined in its own leaf file, only when compiled in).
std::unique_ptr<Backend> CreateD3D11(Fast::GfxRenderingAPI* rapi);
std::unique_ptr<Backend> CreateOpenGL(Fast::GfxRenderingAPI* rapi);
const char* D3D11InstanceExtension();
bool D3D11RequiredAdapterLuid(XrInstance instance, XrSystemId system, uint64_t* luid);

} // namespace vrgfx
