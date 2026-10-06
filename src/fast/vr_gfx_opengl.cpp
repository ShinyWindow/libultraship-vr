// SOH [VR] OpenGL leaf of the VR graphics seam (vr_gfx.h): desktop GL now, GLES-ready (Track GL).
//
// Conventions (VR-MULTI-API-PLAN.md §3.4) as they land on GL:
// - Orientation: the renderer draws XR images like window fb 0 (invertY = false, bottom-up rows),
//   which is what the OpenXR GL binding expects. Rects from the core are top-left: y' = H - y - h.
// - sRGB: GL_SRGB8_ALPHA8 swapchain written with GL_FRAMEBUFFER_SRGB disabled = bytes stored
//   verbatim, the compositor decodes. Fallback GL_RGBA8 (gamma mismatch, logged).
// - Depth: a private GL_DEPTH_COMPONENT32F renderbuffer per image, cleared to 1.0.
// - The renderer caches GL state (scissor enable, blend, bound textures). Every GL call here that
//   changes state the renderer relies on is saved and restored (StateGuard).
// GLES (Track Q): everything outside the Win32 binding block compiles under USE_OPENGLES; see
// markdowns/VR-GLES-DELTAS.md for what an Android/EGL build still needs.
#if defined(ENABLE_OPENGL) && defined(ENABLE_VR)

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "fast/backends/gfx_opengl.h"

#if defined(_WIN32) && !defined(USE_OPENGLES)
#define VR_GL_WIN32_BINDING 1
#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_OPENGL
#endif
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "libultraship/bridge/consolevariablebridge.h"
#include "vr_gfx.h"

#ifndef GL_SRGB8_ALPHA8
#define GL_SRGB8_ALPHA8 0x8C43
#endif
#ifndef GL_FRAMEBUFFER_SRGB
#define GL_FRAMEBUFFER_SRGB 0x8DB9
#endif

namespace vrgfx {
namespace {

// Saves and restores the GL state a clear / blit / texture setup touches, so the renderer's own
// caches (mLastScissorEnabled, bound textures, depth mask) stay true.
struct StateGuard {
    GLint drawFb = 0, readFb = 0, rb = 0, activeTex = 0, tex2d = 0;
    GLint scissorBox[4] = {};
    GLboolean scissor = GL_FALSE, depthMask = GL_TRUE, colorMask[4] = { GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE };
    GLfloat clearColor[4] = {};
    GLfloat clearDepth = 1.0f;
    bool restoreFb;

    explicit StateGuard(bool restoreFramebuffers) : restoreFb(restoreFramebuffers) {
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFb);
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFb);
        glGetIntegerv(GL_RENDERBUFFER_BINDING, &rb);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTex);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex2d);
        scissor = glIsEnabled(GL_SCISSOR_TEST);
        glGetIntegerv(GL_SCISSOR_BOX, scissorBox);
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);
        glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);
        glGetFloatv(GL_COLOR_CLEAR_VALUE, clearColor);
        glGetFloatv(GL_DEPTH_CLEAR_VALUE, &clearDepth);
    }
    ~StateGuard() {
        if (restoreFb) {
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)drawFb);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)readFb);
        }
        glBindRenderbuffer(GL_RENDERBUFFER, (GLuint)rb);
        glActiveTexture((GLenum)activeTex);
        glBindTexture(GL_TEXTURE_2D, (GLuint)tex2d);
        if (scissor) {
            glEnable(GL_SCISSOR_TEST);
        } else {
            glDisable(GL_SCISSOR_TEST);
        }
        glScissor(scissorBox[0], scissorBox[1], scissorBox[2], scissorBox[3]);
        glDepthMask(depthMask);
        glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
        glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
#ifdef USE_OPENGLES
        glClearDepthf(clearDepth);
#else
        glClearDepth(clearDepth);
#endif
    }
};

static void ClearDepthTo1() {
#ifdef USE_OPENGLES
    glClearDepthf(1.0f);
#else
    glClearDepth(1.0);
#endif
}

class OpenGLBackend final : public Backend {
  public:
    explicit OpenGLBackend(Fast::GfxRenderingAPIOGL* gl) : mGl(gl) {
#ifdef VR_GL_WIN32_BINDING
        mBinding.hDC = wglGetCurrentDC();
        mBinding.hGLRC = wglGetCurrentContext();
#endif
    }

    bool Valid() const {
#ifdef VR_GL_WIN32_BINDING
        return mBinding.hDC != nullptr && mBinding.hGLRC != nullptr;
#else
        return false; // no OpenXR binding for this platform yet (EGL/Xlib: Track Q / issue #31)
#endif
    }

    const char* ApiName() const override {
        return "OpenGL";
    }

    const char* RequiredInstanceExtension() const override {
#ifdef VR_GL_WIN32_BINDING
        return XR_KHR_OPENGL_ENABLE_EXTENSION_NAME;
#else
        return "XR_KHR_opengl_es_enable";
#endif
    }

    bool CheckRequirements(XrInstance instance, XrSystemId system, std::string* why) override {
#ifdef VR_GL_WIN32_BINDING
        PFN_xrGetOpenGLGraphicsRequirementsKHR getReqs = nullptr;
        xrGetInstanceProcAddr(instance, "xrGetOpenGLGraphicsRequirementsKHR",
                              reinterpret_cast<PFN_xrVoidFunction*>(&getReqs));
        XrGraphicsRequirementsOpenGLKHR reqs = { XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR };
        if (getReqs == nullptr || XR_FAILED(getReqs(instance, system, &reqs))) {
            *why = "xrGetOpenGLGraphicsRequirementsKHR unavailable or failed";
            return false;
        }
        GLint major = 0, minor = 0;
        glGetIntegerv(GL_MAJOR_VERSION, &major);
        glGetIntegerv(GL_MINOR_VERSION, &minor);
        const XrVersion have = XR_MAKE_VERSION(major, minor, 0);
        const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
        spdlog::info("[VR] OpenGL {}.{} on {} ({}); runtime wants {}.{} to {}.{}", major, minor,
                     renderer ? renderer : "?", version ? version : "?", XR_VERSION_MAJOR(reqs.minApiVersionSupported),
                     XR_VERSION_MINOR(reqs.minApiVersionSupported), XR_VERSION_MAJOR(reqs.maxApiVersionSupported),
                     XR_VERSION_MINOR(reqs.maxApiVersionSupported));
        if (have < reqs.minApiVersionSupported) {
            *why = "the OpenGL context is older than the runtime's minimum";
            return false;
        }
        // Above the maximum is only "untested by the runtime": log, carry on.
        if (reqs.maxApiVersionSupported != 0 && have > reqs.maxApiVersionSupported) {
            spdlog::warn("[VR] OpenGL {}.{} is newer than the runtime says it was tested with", major, minor);
        }
        mHasCopyImage = GLEW_VERSION_4_3 || GLEW_ARB_copy_image;
        return true;
#else
        (void)instance;
        (void)system;
        *why = "no OpenXR OpenGL binding for this platform yet";
        return false;
#endif
    }

    const void* SessionBinding() override {
#ifdef VR_GL_WIN32_BINDING
        return &mBinding;
#else
        return nullptr;
#endif
    }

    int64_t ChooseColorFormat(const std::vector<int64_t>& formats) override {
        int64_t chosen = 0;
        for (int64_t f : formats) {
            if (f == GL_SRGB8_ALPHA8) {
                chosen = f;
                break;
            }
        }
        if (chosen == 0) {
            for (int64_t f : formats) {
                if (f == GL_RGBA8) {
                    chosen = f;
                    break;
                }
            }
        }
        if (chosen == 0) {
            chosen = formats.empty() ? GL_RGBA8 : formats[0];
        }
        if (chosen != GL_SRGB8_ALPHA8) {
            spdlog::warn("[VR] No GL_SRGB8_ALPHA8 swapchain format; using 0x{:x} (colours will look washed out)",
                         chosen);
        }
        spdlog::info("[VR] Swapchain format: 0x{:x} (SRGB8_ALPHA8=0x{:x}, RGBA8=0x{:x})", chosen, GL_SRGB8_ALPHA8,
                     GL_RGBA8);
        return chosen;
    }

    bool Attach(Target t, XrSwapchain handle, uint32_t w, uint32_t h, int64_t fmt) override {
#ifdef VR_GL_WIN32_BINDING
        Detach(t);
        auto& sc = mTargets[(int)t];
        sc.width = w;
        sc.height = h;

        uint32_t count = 0;
        xrEnumerateSwapchainImages(handle, 0, &count, nullptr);
        std::vector<XrSwapchainImageOpenGLKHR> images(count, { XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR });
        xrEnumerateSwapchainImages(handle, count, &count, reinterpret_cast<XrSwapchainImageBaseHeader*>(images.data()));

        StateGuard guard(true);
        sc.color.resize(count);
        sc.fbo.assign(count, 0);
        sc.depth.assign(count, 0);
        for (uint32_t i = 0; i < count; i++) {
            sc.color[i] = images[i].image;
            glGenRenderbuffers(1, &sc.depth[i]);
            glBindRenderbuffer(GL_RENDERBUFFER, sc.depth[i]);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, (GLsizei)w, (GLsizei)h);

            glGenFramebuffers(1, &sc.fbo[i]);
            glBindFramebuffer(GL_FRAMEBUFFER, sc.fbo[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sc.color[i], 0);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, sc.depth[i]);
            const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if (status != GL_FRAMEBUFFER_COMPLETE) {
                spdlog::error("[VR] GL framebuffer for target {} image {} incomplete (0x{:x})", (int)t, i, status);
                return false;
            }
        }
        (void)fmt;
        return true;
#else
        (void)t;
        (void)handle;
        (void)w;
        (void)h;
        (void)fmt;
        return false;
#endif
    }

    void Detach(Target t) override {
        auto& sc = mTargets[(int)t];
        if (HasContext()) {
            for (GLuint f : sc.fbo) {
                if (f != 0) {
                    glDeleteFramebuffers(1, &f);
                }
            }
            for (GLuint r : sc.depth) {
                if (r != 0) {
                    glDeleteRenderbuffers(1, &r);
                }
            }
        }
        sc.fbo.clear();
        sc.depth.clear();
        sc.color.clear(); // owned by the runtime
        sc.width = sc.height = 0;
    }

    void BeginPass(Target t, uint32_t image, const float clearRgba[4]) override {
        auto& sc = mTargets[(int)t];
        if (image >= sc.fbo.size()) {
            return;
        }
        // Binds the image and makes it the renderer's current framebuffer (invertY = false).
        mGl->SetExternalFramebuffer(sc.fbo[image], sc.width, sc.height);
#ifndef USE_OPENGLES
        // Writes to the sRGB image must not be re-encoded: the game's output is already gamma.
        glDisable(GL_FRAMEBUFFER_SRGB);
#endif
        {
            StateGuard guard(false);
            glDisable(GL_SCISSOR_TEST);
            glDepthMask(GL_TRUE);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glClearColor(clearRgba[0], clearRgba[1], clearRgba[2], clearRgba[3]);
            ClearDepthTo1();
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }
        glViewport(0, 0, (GLsizei)sc.width, (GLsizei)sc.height);
    }

    void Rebind(Target t, uint32_t image) override {
        auto& sc = mTargets[(int)t];
        if (image >= sc.fbo.size()) {
            return;
        }
        mGl->SetExternalFramebuffer(sc.fbo[image], sc.width, sc.height);
    }

    void ClearDepth(Target t, uint32_t image) override {
        auto& sc = mTargets[(int)t];
        if (image >= sc.fbo.size()) {
            return;
        }
        StateGuard guard(true);
        glBindFramebuffer(GL_FRAMEBUFFER, sc.fbo[image]);
        glDisable(GL_SCISSOR_TEST);
        glDepthMask(GL_TRUE); // a depth clear with the mask off silently does nothing
        ClearDepthTo1();
        glClear(GL_DEPTH_BUFFER_BIT);
    }

    void ClearColorRects(Target t, uint32_t image, const Rect* r, int n, const float rgbaPremul[4]) override {
        auto& sc = mTargets[(int)t];
        if (image >= sc.fbo.size() || n <= 0) {
            return;
        }
        StateGuard guard(true);
        glBindFramebuffer(GL_FRAMEBUFFER, sc.fbo[image]);
        glEnable(GL_SCISSOR_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glClearColor(rgbaPremul[0], rgbaPremul[1], rgbaPremul[2], rgbaPremul[3]);
        for (int i = 0; i < n; i++) {
            if (r[i].w <= 0 || r[i].h <= 0) {
                continue;
            }
            glScissor(r[i].x, (GLint)sc.height - r[i].y - r[i].h, r[i].w, r[i].h); // top-left -> GL
            glClear(GL_COLOR_BUFFER_BIT);
        }
    }

    void CopyToMirror(Target t, uint32_t image, int slot) override {
        auto& sc = mTargets[(int)t];
        if (slot < 0 || slot >= kMirrorCount || image >= sc.fbo.size()) {
            return;
        }
        auto& m = mMirror[slot];
        StateGuard guard(true);
        if (m.tex == 0 || m.w != sc.width || m.h != sc.height) {
            ReleaseMirror(m);
            glGenTextures(1, &m.tex);
            glBindTexture(GL_TEXTURE_2D, m.tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)sc.width, (GLsizei)sc.height, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
            m.w = sc.width;
            m.h = sc.height;
        }
#ifndef USE_OPENGLES
        if (mHasCopyImage) {
            // SRGB8_ALPHA8 and RGBA8 share a view class: a raw copy, bytes verbatim.
            glCopyImageSubData(sc.color[image], GL_TEXTURE_2D, 0, 0, 0, 0, m.tex, GL_TEXTURE_2D, 0, 0, 0, 0,
                               (GLsizei)sc.width, (GLsizei)sc.height, 1);
            return;
        }
#endif
        // Fallback (GL < 4.3, GLES 3): blit through a mirror FBO. GL_FRAMEBUFFER_SRGB is off, so the
        // blit doesn't convert either.
        if (m.fbo == 0) {
            glGenFramebuffers(1, &m.fbo);
        }
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m.fbo);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m.tex, 0);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, sc.fbo[image]);
        glDisable(GL_SCISSOR_TEST);
        glBlitFramebuffer(0, 0, (GLint)sc.width, (GLint)sc.height, 0, 0, (GLint)sc.width, (GLint)sc.height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    void* MirrorTextureId(int slot) override {
        if (slot < 0 || slot >= kMirrorCount || mMirror[slot].tex == 0) {
            return nullptr;
        }
        return reinterpret_cast<void*>(static_cast<uintptr_t>(mMirror[slot].tex));
    }

    bool MirrorFlipV() const override {
        return true; // GL textures are bottom-up; ImGui samples v = 0 at the top of the quad
    }

    bool SubImageYUp() const override {
        // Whether runtimes read GL sub-image rects bottom-left is unverified (plan §3.4): the wrist
        // panels and text crop answer it with the test card. Dev toggle until G4 decides.
        return CVarGetInteger("gVrGlSubImageYUp", 0) != 0;
    }

    void Shutdown() override {
        for (int t = 0; t < (int)Target::Count; t++) {
            Detach((Target)t);
        }
        for (auto& m : mMirror) {
            ReleaseMirror(m);
        }
    }

  private:
    struct TargetImages {
        uint32_t width = 0, height = 0;
        std::vector<GLuint> color; // runtime-owned swapchain textures
        std::vector<GLuint> fbo;
        std::vector<GLuint> depth;
    };
    struct MirrorCopy {
        GLuint tex = 0, fbo = 0;
        uint32_t w = 0, h = 0;
    };

    static bool HasContext() {
#ifdef VR_GL_WIN32_BINDING
        return wglGetCurrentContext() != nullptr;
#else
        return true;
#endif
    }

    void ReleaseMirror(MirrorCopy& m) {
        if (HasContext()) {
            if (m.tex != 0) {
                glDeleteTextures(1, &m.tex);
            }
            if (m.fbo != 0) {
                glDeleteFramebuffers(1, &m.fbo);
            }
        }
        m = {};
    }

    Fast::GfxRenderingAPIOGL* mGl;
#ifdef VR_GL_WIN32_BINDING
    XrGraphicsBindingOpenGLWin32KHR mBinding = { XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR };
#endif
    bool mHasCopyImage = false;
    TargetImages mTargets[(int)Target::Count];
    MirrorCopy mMirror[kMirrorCount];
};

} // namespace

std::unique_ptr<Backend> CreateOpenGL(Fast::GfxRenderingAPI* rapi) {
    if (rapi == nullptr) {
        return nullptr;
    }
    auto b = std::make_unique<OpenGLBackend>(static_cast<Fast::GfxRenderingAPIOGL*>(rapi));
    if (!b->Valid()) {
        spdlog::error("[VR] No current OpenGL context to bind OpenXR to");
        return nullptr;
    }
    return b;
}

} // namespace vrgfx

#endif // ENABLE_OPENGL && ENABLE_VR
