// SOH [VR] Picks the VR graphics leaf for the running renderer (vr_gfx.h).
#ifdef ENABLE_VR
#include "vr_gfx.h"

#include "fast/Fast3dWindow.h"

namespace vrgfx {

std::unique_ptr<Backend> Create(int windowBackend, Fast::GfxRenderingAPI* rapi) {
    switch (windowBackend) {
#ifdef ENABLE_DX11
        case Fast::WindowBackend::FAST3D_DXGI_DX11:
            return CreateD3D11(rapi);
#endif
#ifdef ENABLE_OPENGL
        case Fast::WindowBackend::FAST3D_SDL_OPENGL:
            return CreateOpenGL(rapi);
#endif
        default:
            return nullptr;
    }
}

const char* InstanceExtensionFor(int windowBackend) {
    switch (windowBackend) {
#ifdef ENABLE_DX11
        case Fast::WindowBackend::FAST3D_DXGI_DX11:
            return D3D11InstanceExtension();
#endif
        default:
            return nullptr; // OpenGL: the context picks the GPU; nothing to steer before it exists
    }
}

const char* ApiNameFor(int windowBackend) {
    switch (windowBackend) {
        case Fast::WindowBackend::FAST3D_DXGI_DX11:
            return "DirectX 11";
        case Fast::WindowBackend::FAST3D_SDL_OPENGL:
            return "OpenGL";
        default:
            return "this renderer";
    }
}

bool RequiredAdapterLuid(int windowBackend, XrInstance instance, XrSystemId system, uint64_t* luid) {
    switch (windowBackend) {
#ifdef ENABLE_DX11
        case Fast::WindowBackend::FAST3D_DXGI_DX11:
            return D3D11RequiredAdapterLuid(instance, system, luid);
#endif
        default:
            return false;
    }
}

} // namespace vrgfx

#endif // ENABLE_VR
