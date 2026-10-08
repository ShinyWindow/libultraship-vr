#ifdef ENABLE_OPENGL
#pragma once

#include "gfx_rendering_api.h"
#include "../interpreter.h"

#ifdef _MSC_VER
#include <SDL2/SDL.h>
// #define GL_GLEXT_PROTOTYPES 1
#include <GL/glew.h>
#elif FOR_WINDOWS
#include <GL/glew.h>
#include "SDL.h"
#define GL_GLEXT_PROTOTYPES 1
#include "SDL_opengl.h"
#elif __APPLE__
#include <SDL2/SDL.h>
#include <GL/glew.h>
#elif USE_OPENGLES
#include <SDL2/SDL.h>
#include <GLES3/gl3.h>
#else
#include <SDL2/SDL.h>
#define GL_GLEXT_PROTOTYPES 1
#include <SDL2/SDL_opengl.h>
#endif
namespace Fast {
// Uniform locations of one linked program, plus the values last uploaded to it. Uniform values live
// in the program object, so a value that hasn't changed since this program last drew needs no call
// (per draw this was 4-5 glUniform* calls, most of them repeats).
struct OglProgramUniforms {
    GLint frameCount = -1;
    GLint noiseScale = -1;
    GLint primDepth = -1;
    GLint textureWidth = -1;
    GLint textureHeight = -1;
    GLint textureFiltering = -1;
    GLint vrEyeMtx = -1; // SOH [VR] multiview programs only: per-layer clip transform

    bool primed = false; // false = nothing uploaded yet, every value goes up on the next draw
    uint32_t lastFrameCount = 0;
    float lastNoiseScale = 0.0f;
    float lastPrimDepth = 0.0f;
    GLint lastFiltering[2] = {};
    GLint lastWidth[2] = {};
    GLint lastHeight[2] = {};
    uint32_t lastEyeGen = 0;
};

struct ShaderProgram {
    GLuint openglProgramId;
    uint64_t shaderId0, shaderId1; // the combiner it was built for (to build variants later)
    uint8_t numInputs;
    bool usedTextures[SHADER_MAX_TEXTURES];
    uint8_t numFloats;
    GLint attribLocations[16];
    uint8_t attribSizes[16];
    uint8_t numAttribs;
    OglProgramUniforms uniforms;
    // SOH [VR] The same combiner built as a multiview program (layout(num_views = 2)), used while a
    // two-layer XR eye target is bound. Built on first use; attribute locations are bound to the
    // same indices in both programs, so one vertex layout serves either.
    GLuint multiviewProgramId;
    bool multiviewTried;
    OglProgramUniforms multiviewUniforms;
};

struct FramebufferOGL {
    uint32_t width, height;
    bool has_depth_buffer;
    uint32_t msaa_level;
    bool invertY;
    bool multiview; // SOH [VR] a two-layer XR eye target (draws use the multiview programs)

    GLuint fbo, clrbuf, clrbufMsaa, rbo;
};

struct TextureInfo {
    uint16_t width;
    uint16_t height;
    uint16_t filtering;
};

class GfxRenderingAPIOGL final : public GfxRenderingAPI {
  public:
    ~GfxRenderingAPIOGL() override = default;
    const char* GetName() override;
    int GetMaxTextureSize() override;
    GfxClipParameters GetClipParameters() override;
    void UnloadShader(ShaderProgram* oldPrg) override;
    void LoadShader(ShaderProgram* newPrg) override;
    ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) override;
    ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) override;
    void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) override;
    void ClearShaderCache() override;
    uint32_t NewTexture() override;
    void SelectTexture(int tile, uint32_t textureId) override;
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override;
    void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) override;
    void SetDepthTestAndMask(bool depth_test, bool z_upd) override;
    void SetCurrentPrimDepth(float depth) override;
    void SetZmodeDecal(bool decal) override;
    void SetViewport(int x, int y, int width, int height) override;
    void SetScissor(int x, int y, int width, int height) override;
    void SetUseAlpha(bool useAlpha) override;
    void DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) override;
    void Init() override;
    void OnResize() override;
    void StartFrame() override;
    void EndFrame() override;
    void FinishRender() override;
    int CreateFramebuffer() override;
    void UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                     bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                     bool can_extract_depth) override;
    void StartDrawToFramebuffer(int fbId, float noiseScale) override;
    void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0,
                         int dstX1, int dstY1) override;
    void ClearFramebuffer(bool color, bool depth) override;
    void ClearDepthRegion(int x, int y, int w, int h) override;
    void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) override;
    void ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) override;
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) override;
    void* GetFramebufferTextureId(int fbId) override;
    void SelectTextureFb(int fbId) override;
    void DeleteTexture(uint32_t texId) override;
    void SetTextureFilter(FilteringMode mode) override;
    FilteringMode GetTextureFilter() override;
    void SetSrgbMode() override;
    ImTextureID GetTextureById(int id) override;

    // SOH [VR] Make an OpenXR swapchain image's FBO (built by vr_gfx_opengl.cpp) the current
    // target: it gets a framebuffer slot of its own, so GetClipParameters (invertY = false, like
    // window fb 0), the decal depth bias and every mid-pass rebind see the right target.
    // multiview = the FBO is a two-layer eye target (vr_gfx_opengl.cpp, Target::EyeArray).
    void SetExternalFramebuffer(GLuint fbo, uint32_t width, uint32_t height, bool multiview = false);

    // SOH [VR] The multiview shader extension this context offers ("GL_OVR_multiview2" or
    // "GL_OVR_multiview"), or null. GLES only; desktop GL builds never use multiview.
    const char* MultiviewExtension() const {
        return mMultiviewExt;
    }

  private:
    std::string BuildFsShader(const CCFeatures& cc_features);
    void SetDrawUniforms(OglProgramUniforms& u, bool multiview);
    GLuint BuildProgram(const std::string& vs, const std::string& fs, const char* const* attribNames, size_t attribCount);
    bool EnsureMultiviewProgram(ShaderProgram* prg);
    bool UploadVertices(const float* buf, size_t floats, size_t strideFloats, GLint* firstVertex);

    std::vector<TextureInfo> textures;
    GLuint mCurrentTextureIds[SHADER_MAX_TEXTURES] = {};
    GLuint mLastBoundTextures[SHADER_MAX_TEXTURES] = {};
    uint8_t mCurrentTile;
    int8_t mLastActiveTexture = -1;
    int8_t mLastBlendEnabled = -1;
    int8_t mLastScissorEnabled = -1;

    std::map<std::pair<uint64_t, uint32_t>, ShaderProgram> mShaderProgramPool;
    ShaderProgram* mCurrentShaderProgram;
    ShaderProgram* mLastLoadedShader = nullptr;
    GLuint mBoundProgram = 0; // the program glUseProgram last bound (base or multiview variant)

    GLuint mOpenglVbo = 0;
#if defined(__APPLE__) || defined(USE_OPENGLES)
    GLuint mOpenglVao;
#endif

#ifdef USE_OPENGLES
    // Persistently mapped vertex ring (EXT_buffer_storage). glBufferData per draw re-allocates
    // (orphans) the buffer every time, which on Adreno cost ~13% of the game thread. The ring is
    // split into segments; a fence is dropped when writing leaves a segment and waited on (bounded)
    // before writing into it again. Draws land at a stride-aligned offset and are addressed with
    // glDrawArrays' first vertex, so the attribute pointers never move.
    static constexpr size_t kRingSegments = 8;
    static constexpr size_t kRingSegmentBytes = 8u << 20; // 8 MB each, 64 MB total
    GLuint mRingVbo = 0;
    uint8_t* mRingPtr = nullptr;
    size_t mRingHead = 0;    // byte offset of the next write
    size_t mRingSegment = 0; // segment mRingHead is in
    GLsync mRingFences[kRingSegments] = {};
    GLuint mFallbackVbo = 0; // glBufferData path for a draw whose segment is still busy
#endif
    const char* mMultiviewExt = nullptr;

    uint32_t mFrameCount = 0;

    std::vector<FramebufferOGL> mFrameBuffers;
    size_t mCurrentFrameBuffer = 0;
    float mCurrentNoiseScale = 0.0f;
    FilteringMode mCurrentFilterMode = FILTER_THREE_POINT;

    GLint mMaxMsaaLevel = 1;
    int mExternalFbSlot = -1;     // SOH [VR] mFrameBuffers slot for the XR target (SetExternalFramebuffer)
    int8_t mLastCoverageBlend = -1; // SOH [VR] which blend func is set: 0 = normal, 1 = VR quad coverage
    GLuint mPixelDepthRb = 0;
    GLuint mPixelDepthFb = 0;
    size_t mPixelDepthRbSize = 0;
};

} // namespace Fast
#endif
