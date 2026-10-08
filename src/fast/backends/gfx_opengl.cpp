#include "ship/window/Window.h"
#ifdef ENABLE_OPENGL

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <unordered_map>

#ifndef _LANGUAGE_C
#define _LANGUAGE_C
#endif

#ifdef __MINGW32__
#define FOR_WINDOWS 1
#else
#define FOR_WINDOWS 0
#endif

#include "fast/backends/gfx_opengl.h"
#include "ship/window/gui/Gui.h"
#include <prism/processor.h>
#include <fstream>
#include "ship/Context.h"
#include "ship/resource/factory/ShaderFactory.h"
#include "fast/interpreter.h"
#include "fast/vr_openxr.h"
#include "ship/config/ConsoleVariable.h"

namespace Fast {
int GfxRenderingAPIOGL::GetMaxTextureSize() {
    GLint max_texture_size;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
    return max_texture_size;
}

const char* GfxRenderingAPIOGL::GetName() {
    return "OpenGL";
}

GfxClipParameters GfxRenderingAPIOGL::GetClipParameters() {
    return { false, mFrameBuffers[mCurrentFrameBuffer].invertY };
}

static void VertexArraySetAttribs(ShaderProgram* prg) {
    size_t numFloats = prg->numFloats;
    size_t pos = 0;

    for (int i = 0; i < prg->numAttribs; i++) {
        if (prg->attribLocations[i] >= 0) {
            glEnableVertexAttribArray(prg->attribLocations[i]);
            glVertexAttribPointer(prg->attribLocations[i], prg->attribSizes[i], GL_FLOAT, GL_FALSE,
                                  numFloats * sizeof(float), (void*)(pos * sizeof(float)));
        }
        pos += prg->attribSizes[i];
    }
}

// Uploads only what changed since this program last drew (u belongs to the bound program).
void GfxRenderingAPIOGL::SetDrawUniforms(OglProgramUniforms& u, bool multiview) {
    const bool all = !u.primed;
    u.primed = true;

    if (all || u.lastFrameCount != mFrameCount) {
        glUniform1i(u.frameCount, (GLint)mFrameCount);
        u.lastFrameCount = mFrameCount;
    }
    if (all || u.lastNoiseScale != mCurrentNoiseScale) {
        glUniform1f(u.noiseScale, mCurrentNoiseScale);
        u.lastNoiseScale = mCurrentNoiseScale;
    }
    if (all || u.lastPrimDepth != mCurrentPrimDepth) {
        glUniform1f(u.primDepth, mCurrentPrimDepth);
        u.lastPrimDepth = mCurrentPrimDepth;
    }

    if (mCurrentShaderProgram->usedTextures[0] || mCurrentShaderProgram->usedTextures[1]) {
        const TextureInfo& t0 = textures[mCurrentTextureIds[0]];
        const TextureInfo& t1 = textures[mCurrentTextureIds[1]];
        const GLint filtering[2] = { t0.filtering, t1.filtering };
        const GLint width[2] = { t0.width, t1.width };
        const GLint height[2] = { t0.height, t1.height };
        if (all || memcmp(filtering, u.lastFiltering, sizeof(filtering)) != 0) {
            glUniform1iv(u.textureFiltering, 2, filtering);
            memcpy(u.lastFiltering, filtering, sizeof(filtering));
        }
        if (all || memcmp(width, u.lastWidth, sizeof(width)) != 0) {
            glUniform1iv(u.textureWidth, 2, width);
            memcpy(u.lastWidth, width, sizeof(width));
        }
        if (all || memcmp(height, u.lastHeight, sizeof(height)) != 0) {
            glUniform1iv(u.textureHeight, 2, height);
            memcpy(u.lastHeight, height, sizeof(height));
        }
    }

    // SOH [VR] Per-layer clip transforms (vr_get_multiview_eye_matrices): the generation changes
    // whenever the matrices do (every rendered frame, and when a draw switches between world and
    // eye-welded geometry).
    if (multiview) {
        uint32_t gen = 0;
        const float* m = vr_get_multiview_eye_matrices(&gen);
        if (all || gen != u.lastEyeGen) {
            glUniformMatrix4fv(u.vrEyeMtx, 2, GL_FALSE, m);
            u.lastEyeGen = gen;
        }
    }
}

static OglProgramUniforms LocateUniforms(GLuint program) {
    OglProgramUniforms u;
    u.frameCount = glGetUniformLocation(program, "frame_count");
    u.noiseScale = glGetUniformLocation(program, "noise_scale");
    u.primDepth = glGetUniformLocation(program, "prim_depth");
    u.textureWidth = glGetUniformLocation(program, "texture_width");
    u.textureHeight = glGetUniformLocation(program, "texture_height");
    u.textureFiltering = glGetUniformLocation(program, "texture_filtering");
    u.vrEyeMtx = glGetUniformLocation(program, "uVrEyeMtx");
    return u;
}

// Sampler units are fixed per name; set once on the (bound) program.
static void BindSamplerUnits(GLuint program, const CCFeatures& cc_features) {
    static const char* const kNames[6] = { "uTex0", "uTex1", "uTexMask0", "uTexMask1", "uTexBlend0", "uTexBlend1" };
    const bool used[6] = { cc_features.usedTextures[0], cc_features.usedTextures[1], cc_features.used_masks[0],
                           cc_features.used_masks[1],   cc_features.used_blend[0],   cc_features.used_blend[1] };
    for (int i = 0; i < 6; i++) {
        if (used[i]) {
            glUniform1i(glGetUniformLocation(program, kNames[i]), i);
        }
    }
}

void GfxRenderingAPIOGL::UnloadShader(ShaderProgram* old_prg) {
    if (old_prg != nullptr && old_prg == mLastLoadedShader) {
        for (unsigned int i = 0; i < old_prg->numAttribs; i++) {
            if (old_prg->attribLocations[i] >= 0) {
                glDisableVertexAttribArray(old_prg->attribLocations[i]);
            }
        }
        mLastLoadedShader = nullptr;
    }
}

void GfxRenderingAPIOGL::LoadShader(ShaderProgram* new_prg) {
    // if (!new_prg) return;
    mCurrentShaderProgram = new_prg;
    if (new_prg != mLastLoadedShader) {
        // Which program variant gets bound (base or multiview) depends on the target, so binding
        // and uniforms happen at draw time. The vertex layout is the same for both.
        VertexArraySetAttribs(new_prg);
        mLastLoadedShader = new_prg;
    }
}

#define RAND_NOISE "((random(vec3(floor(gl_FragCoord.xy * noise_scale), float(frame_count))) + 1.0) / 2.0)"

static const char* shader_item_to_str(uint32_t item, bool with_alpha, bool only_alpha, bool inputs_have_alpha,
                                      bool first_cycle, bool hint_single_element) {
    if (!only_alpha) {
        switch (item) {
            case SHADER_0:
                return with_alpha ? "vec4(0.0, 0.0, 0.0, 0.0)" : "vec3(0.0, 0.0, 0.0)";
            case SHADER_1:
                return with_alpha ? "vec4(1.0, 1.0, 1.0, 1.0)" : "vec3(1.0, 1.0, 1.0)";
            case SHADER_INPUT_1:
                return with_alpha || !inputs_have_alpha ? "vInput1" : "vInput1.rgb";
            case SHADER_INPUT_2:
                return with_alpha || !inputs_have_alpha ? "vInput2" : "vInput2.rgb";
            case SHADER_INPUT_3:
                return with_alpha || !inputs_have_alpha ? "vInput3" : "vInput3.rgb";
            case SHADER_INPUT_4:
                return with_alpha || !inputs_have_alpha ? "vInput4" : "vInput4.rgb";
            case SHADER_TEXEL0:
                return first_cycle ? (with_alpha ? "texVal0" : "texVal0.rgb")
                                   : (with_alpha ? "texVal1" : "texVal1.rgb");
            case SHADER_TEXEL0A:
                return first_cycle
                           ? (hint_single_element ? "texVal0.a"
                                                  : (with_alpha ? "vec4(texVal0.a, texVal0.a, texVal0.a, texVal0.a)"
                                                                : "vec3(texVal0.a, texVal0.a, texVal0.a)"))
                           : (hint_single_element ? "texVal1.a"
                                                  : (with_alpha ? "vec4(texVal1.a, texVal1.a, texVal1.a, texVal1.a)"
                                                                : "vec3(texVal1.a, texVal1.a, texVal1.a)"));
            case SHADER_TEXEL1A:
                return first_cycle
                           ? (hint_single_element ? "texVal1.a"
                                                  : (with_alpha ? "vec4(texVal1.a, texVal1.a, texVal1.a, texVal1.a)"
                                                                : "vec3(texVal1.a, texVal1.a, texVal1.a)"))
                           : (hint_single_element ? "texVal0.a"
                                                  : (with_alpha ? "vec4(texVal0.a, texVal0.a, texVal0.a, texVal0.a)"
                                                                : "vec3(texVal0.a, texVal0.a, texVal0.a)"));
            case SHADER_TEXEL1:
                return first_cycle ? (with_alpha ? "texVal1" : "texVal1.rgb")
                                   : (with_alpha ? "texVal0" : "texVal0.rgb");
            case SHADER_COMBINED:
                return with_alpha ? "texel" : "texel.rgb";
            case SHADER_NOISE:
                return with_alpha ? "vec4(" RAND_NOISE ", " RAND_NOISE ", " RAND_NOISE ", " RAND_NOISE ")"
                                  : "vec3(" RAND_NOISE ", " RAND_NOISE ", " RAND_NOISE ")";
        }
    } else {
        switch (item) {
            case SHADER_0:
                return "0.0";
            case SHADER_1:
                return "1.0";
            case SHADER_INPUT_1:
                return "vInput1.a";
            case SHADER_INPUT_2:
                return "vInput2.a";
            case SHADER_INPUT_3:
                return "vInput3.a";
            case SHADER_INPUT_4:
                return "vInput4.a";
            case SHADER_TEXEL0:
                return first_cycle ? "texVal0.a" : "texVal1.a";
            case SHADER_TEXEL0A:
                return first_cycle ? "texVal0.a" : "texVal1.a";
            case SHADER_TEXEL1A:
                return first_cycle ? "texVal1.a" : "texVal0.a";
            case SHADER_TEXEL1:
                return first_cycle ? "texVal1.a" : "texVal0.a";
            case SHADER_COMBINED:
                return "texel.a";
            case SHADER_NOISE:
                return RAND_NOISE;
        }
    }
    return "";
}

bool get_bool(prism::ContextTypes* value) {
    if (std::holds_alternative<int>(*value)) {
        return std::get<int>(*value) == 1;
    }
    return false;
}

prism::ContextTypes* append_formula(prism::ContextTypes* _, prism::ContextTypes* a_arg, prism::ContextTypes* a_single,
                                    prism::ContextTypes* a_mult, prism::ContextTypes* a_mix,
                                    prism::ContextTypes* a_with_alpha, prism::ContextTypes* a_only_alpha,
                                    prism::ContextTypes* a_alpha, prism::ContextTypes* a_first_cycle) {
    auto c = std::get<prism::MTDArray<int>>(*a_arg);
    bool do_single = get_bool(a_single);
    bool do_multiply = get_bool(a_mult);
    bool do_mix = get_bool(a_mix);
    bool with_alpha = get_bool(a_with_alpha);
    bool only_alpha = get_bool(a_only_alpha);
    bool opt_alpha = get_bool(a_alpha);
    bool first_cycle = get_bool(a_first_cycle);
    std::string out = "";
    if (do_single) {
        out += shader_item_to_str(c.at(only_alpha, 3), with_alpha, only_alpha, opt_alpha, first_cycle, false);
    } else if (do_multiply) {
        out += shader_item_to_str(c.at(only_alpha, 0), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += " * ";
        out += shader_item_to_str(c.at(only_alpha, 2), with_alpha, only_alpha, opt_alpha, first_cycle, true);
    } else if (do_mix) {
        out += "mix(";
        out += shader_item_to_str(c.at(only_alpha, 1), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += ", ";
        out += shader_item_to_str(c.at(only_alpha, 0), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += ", ";
        out += shader_item_to_str(c.at(only_alpha, 2), with_alpha, only_alpha, opt_alpha, first_cycle, true);
        out += ")";
    } else {
        out += "(";
        out += shader_item_to_str(c.at(only_alpha, 0), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += " - ";
        out += shader_item_to_str(c.at(only_alpha, 1), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += ") * ";
        out += shader_item_to_str(c.at(only_alpha, 2), with_alpha, only_alpha, opt_alpha, first_cycle, true);
        out += " + ";
        out += shader_item_to_str(c.at(only_alpha, 3), with_alpha, only_alpha, opt_alpha, first_cycle, false);
    }
    return new prism::ContextTypes{ out };
}

std::optional<std::string> opengl_include_fs(const std::string& path) {
    auto init = std::make_shared<Ship::ResourceInitData>();
    init->Type = (uint32_t)Ship::ResourceType::Shader;
    init->ByteOrder = Ship::Endianness::Native;
    init->Format = RESOURCE_FORMAT_BINARY;
    auto res = std::static_pointer_cast<Ship::Shader>(
        Ship::Context::GetRawInstance()->GetResourceManager()->LoadResource(path, true, init));
    if (res == nullptr) {
        return std::nullopt;
    }
    auto inc = static_cast<std::string*>(res->GetRawPointer());
    return *inc;
}

std::string GfxRenderingAPIOGL::BuildFsShader(const CCFeatures& cc_features) {
    prism::Processor processor;
    prism::ContextItems mContext = {
        { "VERTEX_SHADER", false },
        { "o_c", M_ARRAY(cc_features.c, int, 2, 2, 4) },
        { "o_alpha", cc_features.opt_alpha },
        { "o_fog", cc_features.opt_fog },
        { "o_texture_edge", cc_features.opt_texture_edge },
        { "o_noise", cc_features.opt_noise },
        { "o_2cyc", cc_features.opt_2cyc },
        { "o_alpha_threshold", cc_features.opt_alpha_threshold },
        { "o_invisible", cc_features.opt_invisible },
        { "o_grayscale", cc_features.opt_grayscale },
        { "o_prim_depth", cc_features.opt_prim_depth },
        { "o_textures", M_ARRAY(cc_features.usedTextures, bool, 2) },
        { "o_masks", M_ARRAY(cc_features.used_masks, bool, 2) },
        { "o_blend", M_ARRAY(cc_features.used_blend, bool, 2) },
        { "o_clamp", M_ARRAY(cc_features.clamp, bool, 2, 2) },
        { "o_inputs", cc_features.numInputs },
        { "o_do_mix", M_ARRAY(cc_features.do_mix, bool, 2, 2) },
        { "o_do_single", M_ARRAY(cc_features.do_single, bool, 2, 2) },
        { "o_do_multiply", M_ARRAY(cc_features.do_multiply, bool, 2, 2) },
        { "o_color_alpha_same", M_ARRAY(cc_features.color_alpha_same, bool, 2) },
        { "FILTER_THREE_POINT", FILTER_THREE_POINT },
        { "FILTER_LINEAR", FILTER_LINEAR },
        { "FILTER_NONE", FILTER_NONE },
        { "srgb_mode", mSrgbMode },
        { "SHADER_0", SHADER_0 },
        { "SHADER_INPUT_1", SHADER_INPUT_1 },
        { "SHADER_INPUT_2", SHADER_INPUT_2 },
        { "SHADER_INPUT_3", SHADER_INPUT_3 },
        { "SHADER_INPUT_4", SHADER_INPUT_4 },
        { "SHADER_INPUT_5", SHADER_INPUT_5 },
        { "SHADER_INPUT_6", SHADER_INPUT_6 },
        { "SHADER_INPUT_7", SHADER_INPUT_7 },
        { "SHADER_TEXEL0", SHADER_TEXEL0 },
        { "SHADER_TEXEL0A", SHADER_TEXEL0A },
        { "SHADER_TEXEL1", SHADER_TEXEL1 },
        { "SHADER_TEXEL1A", SHADER_TEXEL1A },
        { "SHADER_1", SHADER_1 },
        { "SHADER_COMBINED", SHADER_COMBINED },
        { "SHADER_NOISE", SHADER_NOISE },
        { "o_three_point_filtering", mCurrentFilterMode == FILTER_THREE_POINT },
        { "append_formula", (InvokeFunc)append_formula },
#ifdef __APPLE__
        { "GLSL_VERSION", "#version 410 core" },
        { "attr", "in" },
        { "opengles", false },
        { "core_opengl", true },
        { "texture", "texture" },
        { "vOutColor", "vOutColor" },
#elif defined(USE_OPENGLES)
        { "GLSL_VERSION", "#version 300 es\nprecision mediump float;" },
        { "attr", "in" },
        { "opengles", true },
        { "core_opengl", false },
        { "texture", "texture" },
        { "vOutColor", "vOutColor" },
#else
        { "GLSL_VERSION", "#version 130" },
        { "attr", "varying" },
        { "opengles", false },
        { "core_opengl", false },
        { "texture", "texture2D" },
        { "vOutColor", "gl_FragColor" },
#endif
    };
    processor.populate(mContext);
    auto init = std::make_shared<Ship::ResourceInitData>();
    init->Type = (uint32_t)Ship::ResourceType::Shader;
    init->ByteOrder = Ship::Endianness::Native;
    init->Format = RESOURCE_FORMAT_BINARY;
    const char* shaderName = Fast::gfx_get_shader(cc_features.shader_id);
    std::string path = "shaders/opengl/default.shader.glsl";

    if (nullptr != shaderName) {
        path = std::string(shaderName) + ".glsl";
    }

    auto res = static_pointer_cast<Ship::Shader>(
        Ship::Context::GetRawInstance()->GetResourceManager()->LoadResource(path, true, init));

    if (res == nullptr) {
        SPDLOG_ERROR("Failed to load default fragment shader, missing f3d.o2r?");
        abort();
    }

    auto shader = static_cast<std::string*>(res->GetRawPointer());
    processor.load(*shader);
    processor.bind_include_loader(opengl_include_fs);
    auto result = processor.process();
    // SPDLOG_INFO("=========== FRAGMENT SHADER ============");
    // SPDLOG_INFO(result);
    // SPDLOG_INFO("========================================");
    return result;
}

static size_t numFloats = 0;

static prism::ContextTypes* UpdateFloats(prism::ContextTypes* _, prism::ContextTypes* num) {
    numFloats += std::get<int>(*num);
    return nullptr;
}

static std::string BuildVsShader(const CCFeatures& cc_features) {
    numFloats = 4;
    prism::Processor processor;
    prism::ContextItems mContext = { { "VERTEX_SHADER", true },
                                     { "o_textures", M_ARRAY(cc_features.usedTextures, bool, 2) },
                                     { "o_clamp", M_ARRAY(cc_features.clamp, bool, 2, 2) },
                                     { "o_fog", cc_features.opt_fog },
                                     { "o_grayscale", cc_features.opt_grayscale },
                                     { "o_alpha", cc_features.opt_alpha },
                                     { "o_inputs", cc_features.numInputs },
                                     { "update_floats", (InvokeFunc)UpdateFloats },
#ifdef __APPLE__
                                     { "GLSL_VERSION", "#version 410 core" },
                                     { "attr", "in" },
                                     { "out", "out" },
                                     { "opengles", false }
#elif defined(USE_OPENGLES)
                                     { "GLSL_VERSION", "#version 300 es" },
                                     { "attr", "in" },
                                     { "out", "out" },
                                     { "opengles", true }
#else
                                     { "GLSL_VERSION", "#version 110" },
                                     { "attr", "attribute" },
                                     { "out", "varying" },
                                     { "opengles", false }
#endif
    };
    processor.populate(mContext);

    auto init = std::make_shared<Ship::ResourceInitData>();
    init->Type = (uint32_t)Ship::ResourceType::Shader;
    init->ByteOrder = Ship::Endianness::Native;
    init->Format = RESOURCE_FORMAT_BINARY;
    const char* shaderName = Fast::gfx_get_shader(cc_features.shader_id);
    std::string path = "shaders/opengl/default.shader.glsl";

    if (nullptr != shaderName) {
        path = std::string(shaderName) + ".glsl";
    }

    auto res = static_pointer_cast<Ship::Shader>(
        Ship::Context::GetRawInstance()->GetResourceManager()->LoadResource(path, true, init));

    if (res == nullptr) {
        SPDLOG_ERROR("Failed to load default vertex shader, missing f3d.o2r?");
        abort();
    }

    auto shader = static_cast<std::string*>(res->GetRawPointer());
    processor.load(*shader);
    processor.bind_include_loader(opengl_include_fs);
    auto result = processor.process();
    // SPDLOG_INFO("=========== VERTEX SHADER ============");
    // SPDLOG_INFO(result);
    // SPDLOG_INFO("========================================");
    return result;
}

void GfxRenderingAPIOGL::ClearShaderCache() {
    mShaderProgramPool.clear();
}

// The vertex attributes a combiner's shader takes, in VBO order. Every program built for the
// combiner binds attribute i to location i, so the base and multiview programs share one layout.
struct AttribLayout {
    char names[16][32];
    uint8_t sizes[16];
    size_t count = 0;

    void Add(const char* name, uint8_t size) {
        snprintf(names[count], sizeof(names[count]), "%s", name);
        sizes[count] = size;
        ++count;
    }
};

static AttribLayout BuildAttribLayout(const CCFeatures& cc_features) {
    AttribLayout l;
    l.Add("aVtxPos", 4);
    for (int i = 0; i < 2; i++) {
        if (cc_features.usedTextures[i]) {
            char name[32];
            snprintf(name, sizeof(name), "aTexCoord%d", i);
            l.Add(name, 2);
            for (int j = 0; j < 2; j++) {
                if (cc_features.clamp[i][j]) {
                    snprintf(name, sizeof(name), "aTexClamp%s%d", j == 0 ? "S" : "T", i);
                    l.Add(name, 1);
                }
            }
        }
    }
    if (cc_features.opt_fog) {
        l.Add("aFog", 4);
    }
    if (cc_features.opt_grayscale) {
        l.Add("aGrayscaleColor", 4);
    }
    for (int i = 0; i < cc_features.numInputs; i++) {
        char name[16];
        snprintf(name, sizeof(name), "aInput%d", i + 1);
        l.Add(name, cc_features.opt_alpha ? 4 : 3);
    }
    return l;
}

static GLuint CompileShaderStage(GLenum stage, const std::string& src, const char* what) {
    const GLchar* source = src.data();
    const GLint length = (GLint)src.size();
    GLuint shader = glCreateShader(stage);
    glShaderSource(shader, 1, &source, &length);
    glCompileShader(shader);
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char error_log[1024] = {};
        GLsizei len = 0;
        glGetShaderInfoLog(shader, sizeof(error_log) - 1, &len, error_log);
        SPDLOG_ERROR("{} shader compilation failed: {}", what, error_log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// Returns 0 when compiling or linking fails.
GLuint GfxRenderingAPIOGL::BuildProgram(const std::string& vs, const std::string& fs, const char* const* attribNames,
                                        size_t attribCount) {
    const GLuint vertex_shader = CompileShaderStage(GL_VERTEX_SHADER, vs, "Vertex");
    const GLuint fragment_shader = vertex_shader != 0 ? CompileShaderStage(GL_FRAGMENT_SHADER, fs, "Fragment") : 0;
    if (vertex_shader == 0 || fragment_shader == 0) {
        if (vertex_shader != 0) {
            glDeleteShader(vertex_shader);
        }
        return 0;
    }

    GLuint shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    for (size_t i = 0; i < attribCount; i++) {
        glBindAttribLocation(shader_program, (GLuint)i, attribNames[i]);
    }
    glLinkProgram(shader_program);
    glDeleteShader(vertex_shader); // flagged; freed with the program
    glDeleteShader(fragment_shader);

    GLint linked = 0;
    glGetProgramiv(shader_program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char error_log[1024] = {};
        GLsizei len = 0;
        glGetProgramInfoLog(shader_program, sizeof(error_log) - 1, &len, error_log);
        SPDLOG_ERROR("Shader program link failed: {}", error_log);
        glDeleteProgram(shader_program);
        return 0;
    }
    return shader_program;
}

ShaderProgram* GfxRenderingAPIOGL::CreateAndLoadNewShader(uint64_t shader_id0, uint64_t shader_id1) {
    CCFeatures cc_features;
    gfx_cc_get_features(shader_id0, shader_id1, &cc_features);
    const auto fs_buf = BuildFsShader(cc_features);
    const auto vs_buf = BuildVsShader(cc_features);

    const AttribLayout layout = BuildAttribLayout(cc_features);
    const char* attribNames[16];
    for (size_t i = 0; i < layout.count; i++) {
        attribNames[i] = layout.names[i];
    }
    const GLuint shader_program = BuildProgram(vs_buf, fs_buf, attribNames, layout.count);
    if (shader_program == 0) {
        abort();
    }

    struct ShaderProgram* prg = &mShaderProgramPool[std::make_pair(shader_id0, shader_id1)];
    for (size_t i = 0; i < layout.count; i++) {
        prg->attribLocations[i] = glGetAttribLocation(shader_program, layout.names[i]);
        prg->attribSizes[i] = layout.sizes[i];
    }

    prg->openglProgramId = shader_program;
    prg->shaderId0 = shader_id0;
    prg->shaderId1 = shader_id1;
    prg->numInputs = cc_features.numInputs;
    prg->usedTextures[0] = cc_features.usedTextures[0];
    prg->usedTextures[1] = cc_features.usedTextures[1];
    prg->usedTextures[2] = cc_features.used_masks[0];
    prg->usedTextures[3] = cc_features.used_masks[1];
    prg->usedTextures[4] = cc_features.used_blend[0];
    prg->usedTextures[5] = cc_features.used_blend[1];
    prg->numFloats = numFloats;
    prg->numAttribs = layout.count;
    prg->uniforms = LocateUniforms(shader_program);
    prg->multiviewProgramId = 0;
    prg->multiviewTried = false;
    prg->multiviewUniforms = {};

    LoadShader(prg);

    glUseProgram(shader_program);
    mBoundProgram = shader_program;
    BindSamplerUnits(shader_program, cc_features);

    return prg;
}

// SOH [VR] Rewrites a processed vertex shader into its multiview twin: the extension and
// layout(num_views = 2) right after #version, and gl_Position through the per-layer transform.
// Done on the processed text rather than as a template flag so custom shaders get it too and the
// APK's shaders don't depend on soh.o2r having been regenerated.
static bool MakeMultiviewVertexShader(std::string& vs, const char* ext) {
    const size_t version = vs.find("#version");
    const size_t eol = version == std::string::npos ? std::string::npos : vs.find('\n', version);
    const std::string from = "gl_Position = aVtxPos;";
    const size_t pos = vs.find(from);
    if (eol == std::string::npos || pos == std::string::npos) {
        return false;
    }
    vs.replace(pos, from.size(), "gl_Position = uVrEyeMtx[int(gl_ViewID_OVR)] * aVtxPos;");
    vs.insert(eol + 1, std::string("#extension ") + ext +
                           " : require\nlayout(num_views = 2) in;\nuniform mat4 uVrEyeMtx[2];\n");
    return true;
}

bool GfxRenderingAPIOGL::EnsureMultiviewProgram(ShaderProgram* prg) {
    if (prg->multiviewProgramId != 0) {
        return true;
    }
    if (prg->multiviewTried) {
        return false;
    }
    prg->multiviewTried = true;
    if (mMultiviewExt == nullptr) {
        return false;
    }

    CCFeatures cc_features;
    gfx_cc_get_features(prg->shaderId0, prg->shaderId1, &cc_features);
    const auto fs_buf = BuildFsShader(cc_features);
    auto vs_buf = BuildVsShader(cc_features);
    if (!MakeMultiviewVertexShader(vs_buf, mMultiviewExt)) {
        SPDLOG_ERROR("[VR] Multiview: vertex shader for combiner {:x}/{:x} has no gl_Position = aVtxPos; line",
                     prg->shaderId0, prg->shaderId1);
        return false;
    }
    const AttribLayout layout = BuildAttribLayout(cc_features);
    const char* attribNames[16];
    for (size_t i = 0; i < layout.count; i++) {
        attribNames[i] = layout.names[i];
    }
    const GLuint program = BuildProgram(vs_buf, fs_buf, attribNames, layout.count);
    if (program == 0) {
        SPDLOG_ERROR("[VR] Multiview: building the program for combiner {:x}/{:x} failed", prg->shaderId0,
                     prg->shaderId1);
        return false;
    }
    prg->multiviewProgramId = program;
    prg->multiviewUniforms = LocateUniforms(program);
    glUseProgram(program);
    mBoundProgram = program;
    BindSamplerUnits(program, cc_features);
    return true;
}

struct ShaderProgram* GfxRenderingAPIOGL::LookupShader(uint64_t shader_id0, uint64_t shader_id1) {
    auto it = mShaderProgramPool.find(std::make_pair(shader_id0, shader_id1));
    return it == mShaderProgramPool.end() ? nullptr : &it->second;
}

void GfxRenderingAPIOGL::ShaderGetInfo(struct ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) {
    *numInputs = prg->numInputs;
    usedTextures[0] = prg->usedTextures[0];
    usedTextures[1] = prg->usedTextures[1];
}

GLuint GfxRenderingAPIOGL::NewTexture() {
    GLuint ret;
    glGenTextures(1, &ret);
    textures.resize(std::max(textures.size(), (size_t)ret + 1));
    return ret;
}

void GfxRenderingAPIOGL::DeleteTexture(uint32_t texID) {
    glDeleteTextures(1, &texID);
}

void GfxRenderingAPIOGL::SelectTexture(int tile, GLuint texture_id) {
    if (mLastActiveTexture != tile) {
        mLastActiveTexture = tile;
        glActiveTexture(GL_TEXTURE0 + tile);
    }
    if (mLastBoundTextures[tile] != texture_id) {
        mLastBoundTextures[tile] = texture_id;
        glBindTexture(GL_TEXTURE_2D, texture_id);
    }
    mCurrentTextureIds[tile] = texture_id;
    mCurrentTile = tile;
}

void GfxRenderingAPIOGL::UploadTexture(const uint8_t* rgba32_buf, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        return;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba32_buf);
    textures[mCurrentTextureIds[mCurrentTile]].width = width;
    textures[mCurrentTextureIds[mCurrentTile]].height = height;
}

#ifdef USE_OPENGLES
#define GL_MIRROR_CLAMP_TO_EDGE 0x8743
#endif

static uint32_t gfx_cm_to_opengl(uint32_t val) {
    switch (val) {
        case G_TX_NOMIRROR | G_TX_CLAMP:
            return GL_CLAMP_TO_EDGE;
        case G_TX_MIRROR | G_TX_WRAP:
            return GL_MIRRORED_REPEAT;
        case G_TX_MIRROR | G_TX_CLAMP:
            return GL_MIRROR_CLAMP_TO_EDGE;
        case G_TX_NOMIRROR | G_TX_WRAP:
            return GL_REPEAT;
    }
    return 0;
}

void GfxRenderingAPIOGL::SetSamplerParameters(int tile, bool linear_filter, uint32_t cms, uint32_t cmt) {
    if (mLastActiveTexture != tile) {
        mLastActiveTexture = tile;
        glActiveTexture(GL_TEXTURE0 + tile);
    }
    const GLint filter = linear_filter && mCurrentFilterMode == FILTER_LINEAR ? GL_LINEAR : GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    textures[mCurrentTextureIds[tile]].filtering = !linear_filter ? FILTER_LINEAR : FILTER_THREE_POINT;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, gfx_cm_to_opengl(cms));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, gfx_cm_to_opengl(cmt));
}

void GfxRenderingAPIOGL::SetDepthTestAndMask(bool depth_test, bool z_upd) {
    mCurrentDepthTest = depth_test;
    mCurrentDepthMask = z_upd;
}

void GfxRenderingAPIOGL::SetCurrentPrimDepth(float depth) {
    if (depth != mCurrentPrimDepth) {
        mCurrentPrimDepth = depth;
        mPrimDepthDirty = true;
    }
}

void GfxRenderingAPIOGL::SetZmodeDecal(bool zmode_decal) {
    mCurrentZmodeDecal = zmode_decal;
}

void GfxRenderingAPIOGL::SetViewport(int x, int y, int width, int height) {
    glViewport(x, y, width, height);
}

void GfxRenderingAPIOGL::SetScissor(int x, int y, int width, int height) {
    glScissor(x, y, width, height);
}

void GfxRenderingAPIOGL::SetUseAlpha(bool use_alpha) {
    int8_t val = use_alpha ? 1 : 0;
    if (mLastBlendEnabled != val) {
        mLastBlendEnabled = val;
        if (use_alpha) {
            glEnable(GL_BLEND);
        } else {
            glDisable(GL_BLEND);
        }
    }
}

void GfxRenderingAPIOGL::DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) {
    if (mCurrentDepthTest != mLastDepthTest || mCurrentDepthMask != mLastDepthMask) {
        mLastDepthTest = mCurrentDepthTest;
        mLastDepthMask = mCurrentDepthMask;

        if (mCurrentDepthTest || mLastDepthMask) {
            glEnable(GL_DEPTH_TEST);
            glDepthMask(mLastDepthMask ? GL_TRUE : GL_FALSE);
            glDepthFunc(mCurrentDepthTest ? (mCurrentZmodeDecal ? GL_LEQUAL : GL_LESS) : GL_ALWAYS);
        } else {
            glDisable(GL_DEPTH_TEST);
        }
    }

    if (mCurrentZmodeDecal != mLastZmodeDecal) {
        mLastZmodeDecal = mCurrentZmodeDecal;
        if (mCurrentZmodeDecal) {
            // SSDB = SlopeScaledDepthBias 120 leads to -2 at 240p which is the same as N64 mode which has very little
            // fighting
            const int n64modeFactor = 120;
            const int noVanishFactor = 100;
            GLfloat SSDB = -2;
            switch (Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger(CVAR_Z_FIGHTING_MODE, 0)) {
                // scaled z-fighting (N64 mode like)
                case 1:
                    if (mFrameBuffers.size() >
                        mCurrentFrameBuffer) { // safety check for vector size can probably be removed
                        SSDB = -1.0f * (GLfloat)mFrameBuffers[mCurrentFrameBuffer].height / n64modeFactor;
                    }
                    break;
                // no vanishing paths
                case 2:
                    if (mFrameBuffers.size() >
                        mCurrentFrameBuffer) { // safety check for vector size can probably be removed
                        SSDB = -1.0f * (GLfloat)mFrameBuffers[mCurrentFrameBuffer].height / noVanishFactor;
                    }
                    break;
                // disabled
                case 0:
                default:
                    SSDB = -2;
            }
            glPolygonOffset(SSDB, -2);
            glEnable(GL_POLYGON_OFFSET_FILL);
        } else {
            glPolygonOffset(0, 0);
            glDisable(GL_POLYGON_OFFSET_FILL);
        }
    }

    // SOH [VR] The VR HUD and text quads are cleared to transparent and composited by alpha. The
    // normal blend func also applies SRC_ALPHA/ONE_MINUS_SRC_ALPHA to alpha, which leaves translucent
    // draws there under-covered (the compositor adds their colour onto the world). Standard "over"
    // on alpha makes the target premultiplied, as an XR quad layer expects. Same as
    // gfx_direct3d11.cpp's blend_state_coverage; checked per draw (eyes and quads share programs).
    const int8_t coverage = vr_wants_coverage_blend() ? 1 : 0;
    if (coverage != mLastCoverageBlend) {
        mLastCoverageBlend = coverage;
        if (coverage) {
            glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        } else {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
    }

    // SOH [VR] A two-layer XR eye target takes the combiner's multiview program; any other target
    // (offscreen framebuffers mid-pass included) the normal one. A single-view program can't draw
    // into a multiview framebuffer, so a combiner whose multiview build failed is skipped there.
    const bool multiview = mFrameBuffers[mCurrentFrameBuffer].multiview;
    GLuint program = mCurrentShaderProgram->openglProgramId;
    OglProgramUniforms* uniforms = &mCurrentShaderProgram->uniforms;
    if (multiview) {
        if (!EnsureMultiviewProgram(mCurrentShaderProgram)) {
            return;
        }
        program = mCurrentShaderProgram->multiviewProgramId;
        uniforms = &mCurrentShaderProgram->multiviewUniforms;
    }
    if (program != mBoundProgram) {
        glUseProgram(program);
        mBoundProgram = program;
    }
    SetDrawUniforms(*uniforms, multiview);

    // printf("flushing %d tris\n", buf_vbo_num_tris);
    GLint first = 0;
    const bool usedFallback = UploadVertices(buf_vbo, buf_vbo_len, mCurrentShaderProgram->numFloats, &first);
    glDrawArrays(GL_TRIANGLES, first, 3 * buf_vbo_num_tris);
#ifdef USE_OPENGLES
    if (usedFallback) {
        // Drew from the fallback buffer (see UploadVertices): point the attributes back at the ring.
        glBindBuffer(GL_ARRAY_BUFFER, mRingVbo);
        VertexArraySetAttribs(mCurrentShaderProgram);
    }
#endif
}

#ifdef USE_OPENGLES
// GLES 3 lists extensions one by one (glGetStringi).
static bool GlesHasExtension(const char* name) {
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++) {
        const char* e = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, (GLuint)i));
        if (e != nullptr && strcmp(e, name) == 0) {
            return true;
        }
    }
    return false;
}

#ifndef GL_MAP_PERSISTENT_BIT_EXT
#define GL_MAP_PERSISTENT_BIT_EXT 0x0040
#endif
#ifndef GL_MAP_COHERENT_BIT_EXT
#define GL_MAP_COHERENT_BIT_EXT 0x0080
#endif
typedef void (*PfnBufferStorageEXT)(GLenum target, GLsizeiptr size, const void* data, GLbitfield flags);

// Waits for the GPU to finish reading a ring segment. Bounded: a GPU that is genuinely that far
// behind gets the draw through the fallback buffer instead of a hang.
static bool WaitRingFence(GLsync& fence) {
    if (fence == nullptr) {
        return true;
    }
    GLenum r = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
    for (int i = 0; i < 20 && r == GL_TIMEOUT_EXPIRED; i++) {
        r = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 5000000); // 5 ms, 100 ms in all
    }
    if (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED) {
        glDeleteSync(fence);
        fence = nullptr;
        return true;
    }
    return false;
}
#endif

// Puts a draw's vertices where the bound attributes read them; *firstVertex = the glDrawArrays first
// vertex. True = sent through the fallback buffer (the caller re-points the attributes afterwards).
bool GfxRenderingAPIOGL::UploadVertices(const float* buf, size_t floats, size_t strideFloats, GLint* firstVertex) {
    const size_t bytes = floats * sizeof(float);
    *firstVertex = 0;
#ifdef USE_OPENGLES
    if (mRingPtr != nullptr) {
        const size_t stride = std::max<size_t>(strideFloats, 1) * sizeof(float);
        auto alignUp = [stride](size_t v) { return (v + stride - 1) / stride * stride; };
        size_t offset = alignUp(mRingHead);
        bool fits = offset + bytes <= (mRingSegment + 1) * kRingSegmentBytes;
        if (!fits && bytes + stride <= kRingSegmentBytes) {
            const size_t next = (mRingSegment + 1) % kRingSegments;
            if (WaitRingFence(mRingFences[next])) {
                // The segment being left is read by draws already queued: fence it.
                mRingFences[mRingSegment] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
                mRingSegment = next;
                offset = alignUp(next * kRingSegmentBytes);
                fits = true;
            }
        }
        if (fits) {
            memcpy(mRingPtr + offset, buf, bytes);
            mRingHead = offset + bytes;
            *firstVertex = (GLint)(offset / stride);
            return false;
        }
        glBindBuffer(GL_ARRAY_BUFFER, mFallbackVbo);
        VertexArraySetAttribs(mCurrentShaderProgram);
        glBufferData(GL_ARRAY_BUFFER, bytes, buf, GL_STREAM_DRAW);
        return true;
    }
#endif
    glBufferData(GL_ARRAY_BUFFER, bytes, buf, GL_STREAM_DRAW);
    return false;
}

void GfxRenderingAPIOGL::Init() {
#if !defined(__linux__) && !defined(__OpenBSD__)
    glewInit();
#endif

    glGenBuffers(1, &mOpenglVbo);
    glBindBuffer(GL_ARRAY_BUFFER, mOpenglVbo);

#if defined(__APPLE__) || defined(USE_OPENGLES)
    glGenVertexArrays(1, &mOpenglVao);
    glBindVertexArray(mOpenglVao);
#endif

#ifdef USE_OPENGLES
    // SOH [VR] Multiview shader extension (single-pass stereo, vr_gfx_opengl.cpp decides whether the
    // eyes use it) and the persistent vertex ring.
    mMultiviewExt = GlesHasExtension("GL_OVR_multiview2")  ? "GL_OVR_multiview2"
                    : GlesHasExtension("GL_OVR_multiview") ? "GL_OVR_multiview"
                                                           : nullptr;
    PfnBufferStorageEXT bufferStorage = nullptr;
    if (GlesHasExtension("GL_EXT_buffer_storage")) {
        bufferStorage = reinterpret_cast<PfnBufferStorageEXT>(SDL_GL_GetProcAddress("glBufferStorageEXT"));
    }
    if (bufferStorage != nullptr) {
        const size_t total = kRingSegments * kRingSegmentBytes;
        const GLbitfield flags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT | GL_MAP_COHERENT_BIT_EXT;
        glGenBuffers(1, &mRingVbo);
        glBindBuffer(GL_ARRAY_BUFFER, mRingVbo);
        bufferStorage(GL_ARRAY_BUFFER, (GLsizeiptr)total, nullptr, flags);
        mRingPtr = static_cast<uint8_t*>(glMapBufferRange(GL_ARRAY_BUFFER, 0, (GLsizeiptr)total, flags));
        if (mRingPtr == nullptr) {
            glBindBuffer(GL_ARRAY_BUFFER, mOpenglVbo);
            glDeleteBuffers(1, &mRingVbo);
            mRingVbo = 0;
        } else {
            mFallbackVbo = mOpenglVbo; // stays bound to the ring from here on
        }
    }
    SPDLOG_INFO("OpenGL ES: persistent vertex ring {}, multiview shaders {}", mRingPtr != nullptr ? "on" : "off",
                mMultiviewExt != nullptr ? mMultiviewExt : "unavailable");
#endif

#ifndef USE_OPENGLES // not supported on gles
    glEnable(GL_DEPTH_CLAMP);
#endif
    glDepthFunc(GL_LEQUAL);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    mFrameBuffers.resize(1); // for the default screen buffer

    glGenRenderbuffers(1, &mPixelDepthRb);
    glBindRenderbuffer(GL_RENDERBUFFER, mPixelDepthRb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 1, 1);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glGenFramebuffers(1, &mPixelDepthFb);
    glBindFramebuffer(GL_FRAMEBUFFER, mPixelDepthFb);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, mPixelDepthRb);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    mPixelDepthRbSize = 1;

    glGetIntegerv(GL_MAX_SAMPLES, &mMaxMsaaLevel);
}

void GfxRenderingAPIOGL::OnResize() {
}

void GfxRenderingAPIOGL::StartFrame() {
    mFrameCount++;
}

void GfxRenderingAPIOGL::EndFrame() {
    glFlush();
}

void GfxRenderingAPIOGL::FinishRender() {
}

int GfxRenderingAPIOGL::CreateFramebuffer() {
    GLuint clrbuf;
    glGenTextures(1, &clrbuf);
    glBindTexture(GL_TEXTURE_2D, clrbuf);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    GLuint clrbufMsaa;
    glGenRenderbuffers(1, &clrbufMsaa);

    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 1, 1);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    GLuint fbo;
    glGenFramebuffers(1, &fbo);

    size_t i = mFrameBuffers.size();
    mFrameBuffers.resize(i + 1);

    mFrameBuffers[i].fbo = fbo;
    mFrameBuffers[i].clrbuf = clrbuf;
    mFrameBuffers[i].clrbufMsaa = clrbufMsaa;
    mFrameBuffers[i].rbo = rbo;

    return i;
}

void GfxRenderingAPIOGL::UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                                     bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                                     bool can_extract_depth) {
    FramebufferOGL& fb = mFrameBuffers[fb_id];

    width = std::max(width, 1U);
    height = std::max(height, 1U);
    msaa_level = std::min(msaa_level, (uint32_t)mMaxMsaaLevel);

    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);

    if (fb_id != 0) {
        if (fb.width != width || fb.height != height || fb.msaa_level != msaa_level) {
            if (msaa_level <= 1) {
                glBindTexture(GL_TEXTURE_2D, fb.clrbuf);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
                glBindTexture(GL_TEXTURE_2D, 0);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fb.clrbuf, 0);
            } else {
                glBindRenderbuffer(GL_RENDERBUFFER, fb.clrbufMsaa);
                glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa_level, GL_RGB8, width, height);
                glBindRenderbuffer(GL_RENDERBUFFER, 0);
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, fb.clrbufMsaa);
            }
        }

        if (has_depth_buffer &&
            (fb.width != width || fb.height != height || fb.msaa_level != msaa_level || !fb.has_depth_buffer)) {
            glBindRenderbuffer(GL_RENDERBUFFER, fb.rbo);
            if (msaa_level <= 1) {
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
            } else {
                glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa_level, GL_DEPTH24_STENCIL8, width, height);
            }
            glBindRenderbuffer(GL_RENDERBUFFER, 0);
        }

        if (!fb.has_depth_buffer && has_depth_buffer) {
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, fb.rbo);
        } else if (fb.has_depth_buffer && !has_depth_buffer) {
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, 0);
        }
    }

    fb.width = width;
    fb.height = height;
    fb.has_depth_buffer = has_depth_buffer;
    fb.msaa_level = msaa_level;
    fb.invertY = opengl_invertY;
}

void GfxRenderingAPIOGL::StartDrawToFramebuffer(int fb_id, float noise_scale) {
    FramebufferOGL& fb = mFrameBuffers[fb_id];

    if (noise_scale != 0.0f) {
        mCurrentNoiseScale = 1.0f / noise_scale;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
    mCurrentFrameBuffer = fb_id;
}

void GfxRenderingAPIOGL::ClearFramebuffer(bool color, bool depth) {
    if (mLastScissorEnabled != 0) {
        mLastScissorEnabled = 0;
        glDisable(GL_SCISSOR_TEST);
    }
    glDepthMask(GL_TRUE);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear((color ? GL_COLOR_BUFFER_BIT : 0) | (depth ? GL_DEPTH_BUFFER_BIT : 0));
    glDepthMask(mCurrentDepthMask ? GL_TRUE : GL_FALSE);
    if (mLastScissorEnabled != 1) {
        mLastScissorEnabled = 1;
        glEnable(GL_SCISSOR_TEST);
    }
}

void GfxRenderingAPIOGL::ClearDepthRegion(int x, int y, int w, int h) {
    // Save current scissor state so callers don't need to manually invalidate.
    GLint prevScissor[4];
    GLboolean scissorWasEnabled = glIsEnabled(GL_SCISSOR_TEST);
    glGetIntegerv(GL_SCISSOR_BOX, prevScissor);

    glEnable(GL_SCISSOR_TEST);
    glScissor(x, y, w, h);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDepthMask(mCurrentDepthMask ? GL_TRUE : GL_FALSE);

    // Restore previous scissor state.
    glScissor(prevScissor[0], prevScissor[1], prevScissor[2], prevScissor[3]);
    if (!scissorWasEnabled) {
        glDisable(GL_SCISSOR_TEST);
    }
}

void GfxRenderingAPIOGL::ResolveMSAAColorBuffer(int fb_id_target, int fb_id_source) {
    FramebufferOGL& fb_dst = mFrameBuffers[fb_id_target];
    FramebufferOGL& fb_src = mFrameBuffers[fb_id_source];
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb_dst.fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb_src.fbo);

    // Disabled for blit
    if (mLastScissorEnabled != 0) {
        mLastScissorEnabled = 0;
        glDisable(GL_SCISSOR_TEST);
    }

    glBlitFramebuffer(0, 0, fb_src.width, fb_src.height, 0, 0, fb_dst.width, fb_dst.height, GL_COLOR_BUFFER_BIT,
                      GL_NEAREST);
    // SOH [VR] Rebind by FBO name: this passed the slot index, which bound the wrong FBO (or none).
    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffers[mCurrentFrameBuffer].fbo);

    if (mLastScissorEnabled != 1) {
        mLastScissorEnabled = 1;
        glEnable(GL_SCISSOR_TEST);
    }
}

void* GfxRenderingAPIOGL::GetFramebufferTextureId(int fb_id) {
    return (void*)(uintptr_t)mFrameBuffers[fb_id].clrbuf;
}

void GfxRenderingAPIOGL::SelectTextureFb(int fb_id) {
    // glDisable(GL_DEPTH_TEST);
    int tile = 0;
    GLuint texId = mFrameBuffers[fb_id].clrbuf;
    // Ensure the textures metadata vector can hold this FB texture handle.
    // FB color buffers are created outside NewTexture(), so the vector may
    // not have been resized for them yet.
    if (texId >= textures.size()) {
        textures.resize((size_t)texId + 1);
    }
    SelectTexture(tile, texId);
}

void GfxRenderingAPIOGL::CopyFramebuffer(int fb_dst_id, int fb_src_id, int srcX0, int srcY0, int srcX1, int srcY1,
                                         int dstX0, int dstY0, int dstX1, int dstY1) {
    if (fb_dst_id >= (int)mFrameBuffers.size() || fb_src_id >= (int)mFrameBuffers.size()) {
        return;
    }

    FramebufferOGL src = mFrameBuffers[fb_src_id];
    const FramebufferOGL& dst = mFrameBuffers[fb_dst_id];

    // Adjust y values for non-inverted source frame buffers because opengl uses bottom left for origin
    if (!src.invertY) {
        int temp = srcY1 - srcY0;
        srcY1 = src.height - srcY0;
        srcY0 = srcY1 - temp;
    }

    // Flip the y values
    if (src.invertY != dst.invertY) {
        std::swap(srcY0, srcY1);
    }

    // Disabled for blit
    if (mLastScissorEnabled != 0) {
        mLastScissorEnabled = 0;
        glDisable(GL_SCISSOR_TEST);
    }

    // For msaa enabled buffers we can't perform a scaled blit to a simple sample buffer
    // First do an unscaled blit to a msaa resolved buffer
    if (src.height != dst.height && src.width != dst.width && src.msaa_level > 1) {
        // Start with the main buffer (0) as the msaa resolved buffer
        int fb_resolve_id = 0;
        FramebufferOGL fb_resolve = mFrameBuffers[fb_resolve_id];

        // If the size doesn't match our source, then we need to use our separate color msaa resolved buffer (2)
        if (fb_resolve.height != src.height || fb_resolve.width != src.width) {
            fb_resolve_id = 2;
            fb_resolve = mFrameBuffers[fb_resolve_id];
        }

        glBindFramebuffer(GL_READ_FRAMEBUFFER, src.fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb_resolve.fbo);

        glBlitFramebuffer(0, 0, src.width, src.height, 0, 0, src.width, src.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        // Switch source buffer to the resolved sample
        fb_src_id = fb_resolve_id;
        src = fb_resolve;
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, src.fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst.fbo);

    // The 0 buffer is a double buffer so we need to choose the back to avoid imgui elements
    if (fb_src_id == 0) {
        glReadBuffer(GL_BACK);
    } else {
        glReadBuffer(GL_COLOR_ATTACHMENT0);
    }

    glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffers[mCurrentFrameBuffer].fbo);

    glReadBuffer(GL_BACK);

    if (mLastScissorEnabled != 1) {
        mLastScissorEnabled = 1;
        glEnable(GL_SCISSOR_TEST);
    }
}

void GfxRenderingAPIOGL::ReadFramebufferToCPU(int fb_id, uint32_t width, uint32_t height, uint16_t* rgba16_buf) {
    if (fb_id >= (int)mFrameBuffers.size()) {
        return;
    }

    // Read as RGBA8 (GL_UNSIGNED_BYTE) then convert to RGBA16 (5551).
    // GL_RGBA + GL_UNSIGNED_SHORT_5_5_5_1 writes 4 separate u16 components per pixel
    // (8 bytes) on some drivers (NVIDIA), not the packed 2 bytes the spec implies.
    // Reading as RGBA8 and converting matches the DX11 path's approach.
    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffers[fb_id].fbo);

    std::vector<uint8_t> rgba8(width * height * 4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba8.data());

    for (uint32_t i = 0; i < width * height; i++) {
        uint8_t r = (rgba8[i * 4 + 0] >> 3) & 0x1F;
        uint8_t g = (rgba8[i * 4 + 1] >> 3) & 0x1F;
        uint8_t b = (rgba8[i * 4 + 2] >> 3) & 0x1F;
        uint8_t a = rgba8[i * 4 + 3] ? 1 : 0;
        rgba16_buf[i] = (r << 11) | (g << 6) | (b << 1) | a;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffers[mCurrentFrameBuffer].fbo);
}

std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIOGL::GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) {
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res;

    FramebufferOGL& fb = mFrameBuffers[fb_id];

    // When looking up one value and the framebuffer is single-sampled, we can read pixels directly
    // Otherwise we need to blit first to a new buffer then read it
    if (coordinates.size() == 1 && fb.msaa_level <= 1) {
        uint32_t depth_stencil_value;
        glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
        int x = coordinates.begin()->first;
        int y = coordinates.begin()->second;
#ifndef USE_OPENGLES // not supported on gles. Runs fine without it, but this may cause issues
        glReadPixels(x, fb.invertY ? fb.height - y : y, 1, 1, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8,
                     &depth_stencil_value);
#endif
        res.emplace(*coordinates.begin(), (depth_stencil_value >> 18) << 2);
    } else {
        if (mPixelDepthRbSize < coordinates.size()) {
            // Resizing a renderbuffer seems broken with Intel's driver, so recreate one instead.
            glBindFramebuffer(GL_FRAMEBUFFER, mPixelDepthFb);
            glDeleteRenderbuffers(1, &mPixelDepthRb);
            glGenRenderbuffers(1, &mPixelDepthRb);
            glBindRenderbuffer(GL_RENDERBUFFER, mPixelDepthRb);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, coordinates.size(), 1);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, mPixelDepthRb);
            glBindRenderbuffer(GL_RENDERBUFFER, 0);

            mPixelDepthRbSize = coordinates.size();
        }

        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mPixelDepthFb);

        glDisable(GL_SCISSOR_TEST); // needed for the blit operation

        {
            size_t i = 0;
            for (const auto& coord : coordinates) {
                int x = coord.first;
                int y = coord.second;
                if (fb.invertY) {
                    y = fb.height - y;
                }
                glBlitFramebuffer(x, y, x + 1, y + 1, i, 0, i + 1, 1, GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT,
                                  GL_NEAREST);
                ++i;
            }
        }

        glBindFramebuffer(GL_READ_FRAMEBUFFER, mPixelDepthFb);
        std::vector<uint32_t> depth_stencil_values(coordinates.size());
#ifndef USE_OPENGLES // not supported on gles. Runs fine without it, but this may cause issues
        glReadPixels(0, 0, coordinates.size(), 1, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, depth_stencil_values.data());
#endif
        {
            size_t i = 0;
            for (const auto& coord : coordinates) {
                res.emplace(coord, (depth_stencil_values[i++] >> 18) << 2);
            }
        }
    }

    // SOH [VR] Rebind by FBO name, not slot index (see ResolveMSAAColorBuffer).
    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffers[mCurrentFrameBuffer].fbo);

    return res;
}

void GfxRenderingAPIOGL::SetTextureFilter(FilteringMode mode) {
    gfx_texture_cache_clear();
    mCurrentFilterMode = mode;
}

FilteringMode GfxRenderingAPIOGL::GetTextureFilter() {
    return mCurrentFilterMode;
}

void GfxRenderingAPIOGL::SetSrgbMode() {
    mSrgbMode = true;
}

ImTextureID GfxRenderingAPIOGL::GetTextureById(int id) {
    return reinterpret_cast<ImTextureID>(id);
}

// SOH [VR] The XR target is drawn like window fb 0 (invertY = false): GL images are bottom-up and
// the OpenXR GL binding expects exactly that. One slot, reused for every XR image (only the FBO
// name and size change); appended after the interpreter's framebuffers, so their ids never move.
void GfxRenderingAPIOGL::SetExternalFramebuffer(GLuint fbo, uint32_t width, uint32_t height, bool multiview) {
    if (mExternalFbSlot < 0) {
        mExternalFbSlot = (int)mFrameBuffers.size();
        mFrameBuffers.resize(mFrameBuffers.size() + 1);
    }
    FramebufferOGL& fb = mFrameBuffers[mExternalFbSlot];
    fb.fbo = fbo;
    fb.clrbuf = 0;
    fb.clrbufMsaa = 0;
    fb.rbo = 0;
    fb.width = width;
    fb.height = height;
    fb.msaa_level = 1;
    fb.has_depth_buffer = true;
    fb.invertY = false;
    fb.multiview = multiview;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    mCurrentFrameBuffer = (size_t)mExternalFbSlot;
}
} // namespace Fast
#endif

#pragma clang diagnostic pop
