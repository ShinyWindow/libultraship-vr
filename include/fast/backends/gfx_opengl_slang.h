#ifdef ENABLE_OPENGL
#pragma once

// A runtime for RetroArch "slang" shader presets (.slangp) on OpenGL.
//
// The desktop builds run presets through librashader, which is a Rust library and can't be built for every platform
// libultraship runs on (the Nintendo Switch in particular). This runtime covers the OpenGL backend with no
// dependencies: the slang sources are Vulkan flavoured GLSL 450, which the runtime rewrites into plain GLSL at load
// time (uniform blocks become std140 uniform buffers, descriptor set and binding qualifiers are dropped, samplers
// and blocks are matched by name), so the driver compiles the shaders directly.
//
// Supported: #reference chains and parameter overrides in presets, per pass scaling (source, viewport, absolute,
// original), filtering and wrap modes, float and sRGB framebuffers, #pragma format, mipmapped inputs, LUT textures,
// pass aliases, PassOutput#, PassFeedback# and OriginalHistory# textures, and the standard uniform semantics.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "fast/backends/gfx_rendering_api.h"

namespace Fast {

class SlangFilterChain {
  public:
    SlangFilterChain();
    ~SlangFilterChain();
    SlangFilterChain(const SlangFilterChain&) = delete;
    SlangFilterChain& operator=(const SlangFilterChain&) = delete;

    // Loads a preset and compiles its passes on the current GL context. On failure error describes what went wrong.
    bool Load(const std::string& presetPath, std::string& error);

    // Runs the chain: sourceTexture (the finished game image) is filtered into the colour attachment of targetFbo,
    // filling its whole targetWidth x targetHeight. GL state that the chain touches is restored afterwards.
    bool Frame(uint32_t sourceTexture, uint32_t sourceWidth, uint32_t sourceHeight, uint32_t targetFbo,
               uint32_t targetWidth, uint32_t targetHeight, uint32_t frameCount);

    // Runtime parameters exposed by the preset, with initial values as the preset leaves them
    const std::vector<PostFilterParam>& GetParams() const;
    bool GetParam(const std::string& name, float& value) const;
    bool SetParam(const std::string& name, float value);

  private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace Fast
#endif
