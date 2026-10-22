#pragma once

#if defined(ENABLE_DX11) || defined(ENABLE_DX12)

#ifdef __cplusplus
#include "../interpreter.h"
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include "gfx_rendering_api.h"
#include "d3d11.h"
#include "d3dcompiler.h"

namespace Fast {

struct PerFrameCB {
    uint32_t noise_frame;
    float noise_scale;
    uint32_t padding[2]; // constant buffers must be multiples of 16 bytes in size
};

struct PerDrawCB {
    struct Texture {
        uint32_t width;
        uint32_t height;
        uint32_t linear_filtering;
        uint32_t padding;
    } mTextures[SHADER_MAX_TEXTURES];
};

struct PerPrimDepthCB {
    float prim_depth;
    float _pad[3]; // 16-byte CB alignment
};

// SOH [VR] Two clip-space transforms for D3D11 single-pass stereo. The CPU stores the matrices
// row-major; HLSL's default column-major interpretation plus mul(matrix, vector) performs the
// transpose needed to turn the engine's row-vector transform into a shader column-vector one.
struct PerVrEyeCB {
    float eye_mtx[2][16];
};

struct Coord {
    int x, y;
};

struct TextureData {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> resource_view;
    Microsoft::WRL::ComPtr<ID3D11SamplerState> sampler_state;
    uint32_t width;
    uint32_t height;
    bool linear_filtering;
};

struct FramebufferDX11 {
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> render_target_view;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilView> depth_stencil_view;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_stencil_srv;
    uint32_t texture_id;
    bool has_depth_buffer;
    uint32_t msaa_level;
};

struct ShaderProgramD3D11 {
    Microsoft::WRL::ComPtr<ID3D11VertexShader> vertex_shader;
    // SOH [VR] Duplicates each triangle into the two slices of an OpenXR EyeArray target. Kept
    // beside the combiner because its input/output signature contains that combiner's varyings.
    Microsoft::WRL::ComPtr<ID3D11GeometryShader> multiview_geometry_shader;
    // Retained only until the program is first used on EyeArray. This makes GS compilation fully
    // lazy, so flat/non-VR shader creation follows exactly its old VS+PS path.
    std::string multiview_shader_source;
    bool multiview_geometry_shader_tried = false;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> pixel_shader;
    Microsoft::WRL::ComPtr<ID3D11InputLayout> input_layout;
    Microsoft::WRL::ComPtr<ID3D11BlendState> blend_state;
    // SOH [VR] Same colour blend, but alpha accumulates coverage ("over") instead of being kept.
    // Used on transparent VR quad targets (text panel), where the compositor reads alpha; null when
    // the program doesn't blend.
    Microsoft::WRL::ComPtr<ID3D11BlendState> blend_state_coverage;

    uint64_t shader_id0;
    uint64_t shader_id1;
    uint8_t numInputs;
    uint8_t numFloats;
    bool usedTextures[SHADER_MAX_TEXTURES];
};

class GfxWindowBackendDXGI;

class GfxRenderingAPIDX11 final : public GfxRenderingAPI {
  public:
    GfxRenderingAPIDX11() = default;
    ~GfxRenderingAPIDX11() override;
    GfxRenderingAPIDX11(GfxWindowBackendDXGI* backend);
    const char* GetName() override;
    int GetMaxTextureSize() override;
    GfxClipParameters GetClipParameters() override;
    void UnloadShader(struct ShaderProgram* oldPrg) override;
    void LoadShader(struct ShaderProgram* newPrg) override;
    struct ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) override;
    struct ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) override;
    void ShaderGetInfo(struct ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) override;
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

    PFN_D3D11_CREATE_DEVICE mDX11CreateDevice;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> mContext;
    Microsoft::WRL::ComPtr<ID3D11Device> mDevice;
    GfxWindowBackendDXGI* mWindowBackend = nullptr;
    D3D_FEATURE_LEVEL mFeatureLevel;

    // SOH [VR] The VR layer binds OpenXR swapchain RTVs directly (outside the framebuffer map).
    // Tell the renderer the height needed by viewport/scissor Y-flips, whether draws must use the
    // layered geometry-shader path, and whether they must reject stencil-marked pixels hidden by
    // the headset lenses. Ordinary targets pass multiview=false, visibilityMask=false.
    void SetExternalRenderTarget(int32_t height, bool multiview, bool visibilityMask = false) {
        mRenderTargetHeight = height;
        if (mVrVisibilityMaskTarget != visibilityMask) {
            mVrVisibilityMaskTarget = visibilityMask;
            // Stencil participation is baked into the depth-stencil state. Force the first draw
            // after an eye/offscreen target transition to bind the matching cached variant.
            mBoundDepthStencilStateKey = 0xFFu;
        }
        if (mMultiviewTarget != multiview) {
            mMultiviewTarget = multiview;
            mVrEyeCbBound = false;
            if (!multiview && mContext) {
                // Do this at the target boundary, not on every ordinary draw. A flat-only run
                // never touches the geometry-shader stage.
                mContext->GSSetShader(nullptr, nullptr, 0);
                mLastGeometryShader = nullptr;
                mGeometryShaderStateKnown = true;
            } else {
                mGeometryShaderStateKnown = false;
                // A newly created OpenXR session can restart its generation counter, so force the
                // first EyeArray draw to upload rather than trusting a value from an old session.
                mVrEyeGeneration = UINT32_MAX;
            }
        }
    }

    // Kept for callers that only bind an ordinary single-layer target.
    void SetRenderTargetHeight(int32_t height) {
        SetExternalRenderTarget(height, false, false);
    }

    // A VR backend may issue a small native draw before the game's display list (for example, the
    // OpenXR hidden-area stencil mesh). Mark only the pipeline state it touches as unknown so the
    // next game draw restores its expected bindings without a destructive ClearState.
    void InvalidatePipelineStateAfterExternalDraw() {
        mLastShaderProgram = nullptr;
        mLastGeometryShader = nullptr;
        mGeometryShaderStateKnown = false;
        mVrEyeCbBound = false;
        mBoundDepthStencilStateKey = 0xFFu;
        mBoundRasterizerStateKey = ~uint64_t{ 0 };
        mLastPrimitaveTopology = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
    }

    // SOH [VR] Which GPU the device is created on, set before Init (Fast3dWindow::Init, only when VR
    // starts at launch): the adapter the headset runtime requires (LUID, (HighPart << 32) | LowPart,
    // 0 = unknown), else the high-performance GPU. Neither set = the system default, as before.
    uint64_t mPreferredAdapterLuid = 0;
    bool mPreferHighPerformance = false;

  private:
    bool EnsureMultiviewGeometryShader(struct ShaderProgramD3D11* prg);
    void CreateDepthStencilObjects(uint32_t width, uint32_t height, uint32_t msaa_count, ID3D11DepthStencilView** view,
                                   ID3D11ShaderResourceView** srv);

    HMODULE mDX11Module;

    HMODULE mCompilerModule;
    pD3DCompile mD3dCompile;

    uint32_t mMsaaNumQualityLevels[D3D11_MAX_MULTISAMPLE_SAMPLE_COUNT];

    // D3D11 state objects are immutable. Cache the small set of variants instead of asking the
    // driver to recreate identical objects whenever a VR target or display-list mode changes.
    std::array<Microsoft::WRL::ComPtr<ID3D11DepthStencilState>, 16> mDepthStencilStateCache;
    std::unordered_map<uint64_t, Microsoft::WRL::ComPtr<ID3D11RasterizerState>> mRasterizerStateCache;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> mUncachedDepthStencilState;
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> mUncachedRasterizerState;
    uint8_t mBoundDepthStencilStateKey = 0xFFu;
    uint64_t mBoundRasterizerStateKey = ~uint64_t{ 0 };
    bool mStateObjectCacheEnabled = false;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mVertexBuffer;
    // Ring-allocation cursor into mVertexBuffer, in bytes. Each DrawTriangles appends at the cursor
    // with MAP_WRITE_NO_OVERWRITE and only pays for a buffer rename (MAP_WRITE_DISCARD) when the
    // cursor wraps, instead of renaming on every single draw.
    uint32_t mVertexBufferOffset = 0;
    uint32_t mVertexBufferCapacity = 0;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mPerFrameCb;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mPerDrawCb;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mPerPrimDepthCb;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mPerVrEyeCb;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mCoordBuffer;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mCoordBufferSrv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mDepthValueOutputBuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mDepthValueOutputBufferCopy;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> mDepthValueOutputUav;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> mComputeShader;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> mComputeShaderMsaa;
    Microsoft::WRL::ComPtr<ID3DBlob> mComputeShaderMsaaBlob;
    size_t mCoordBufferSize;

#if DEBUG_D3D
    Microsoft::WRL::ComPtr<ID3D11Debug> debug;
#endif

    PerFrameCB mPerFrameCbData;
    PerDrawCB mPerDrawCbData;
    PerPrimDepthCB mPerPrimDepthCbData;
    PerVrEyeCB mPerVrEyeCbData = {};

    std::map<std::pair<uint64_t, uint32_t>, struct ShaderProgramD3D11> mShaderProgramPool;

    std::vector<struct TextureData> mTextures;
    int mCurrentTile;
    uint32_t mCurrentTextureIds[SHADER_MAX_TEXTURES] = {};

    std::vector<FramebufferDX11> mFrameBuffers;

    // Current state

    struct ShaderProgramD3D11* mShaderProgram;

    int32_t mRenderTargetHeight;
    int mCurrentFramebuffer;
    FilteringMode mCurrentFilterMode = FILTER_NONE;

    // Previous states (to prevent setting states needlessly)

    struct ShaderProgramD3D11* mLastShaderProgram = nullptr;
    ID3D11GeometryShader* mLastGeometryShader = nullptr;
    bool mGeometryShaderStateKnown = true;
    bool mMultiviewTarget = false;
    bool mVrVisibilityMaskTarget = false;
    bool mVrEyeCbBound = false;
    uint32_t mVrEyeGeneration = UINT32_MAX;
    Microsoft::WRL::ComPtr<ID3D11BlendState> mLastBlendState = nullptr;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mLastResourceViews[SHADER_MAX_TEXTURES] = { nullptr, nullptr };
    Microsoft::WRL::ComPtr<ID3D11SamplerState> mLastSamplerStates[SHADER_MAX_TEXTURES] = { nullptr, nullptr };

    D3D_PRIMITIVE_TOPOLOGY mLastPrimitaveTopology = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;

    // Cached staging texture for ReadFramebufferToCPU — avoids CreateTexture2D/Release per frame
    Microsoft::WRL::ComPtr<ID3D11Texture2D> mReadbackStaging;
    uint32_t mReadbackStagingW = 0;
    uint32_t mReadbackStagingH = 0;
};

std::string gfx_direct3d_common_build_shader(size_t& numFloats, const CCFeatures& cc_features,
                                             bool include_root_signature, bool three_point_filtering, bool use_srgb);
} // namespace Fast
#endif
#endif
