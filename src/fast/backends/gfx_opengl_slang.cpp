#include "ship/window/Window.h"
#ifdef ENABLE_OPENGL

#include "fast/backends/gfx_opengl.h" // brings in the platform's GL header
#include "fast/backends/gfx_opengl_slang.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

#include <spdlog/spdlog.h>
#include <stb_image.h>

#ifndef GL_CLAMP_TO_BORDER
#define GL_CLAMP_TO_BORDER 0x812D
#endif
#ifndef GL_HALF_FLOAT
#define GL_HALF_FLOAT 0x140B
#endif
#ifndef GL_SRGB8_ALPHA8
#define GL_SRGB8_ALPHA8 0x8C43
#endif
#ifndef GL_RGBA16F
#define GL_RGBA16F 0x881A
#endif
#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif
#ifndef GL_R11F_G11F_B10F
#define GL_R11F_G11F_B10F 0x8C3A
#endif
#ifndef GL_RGB10_A2
#define GL_RGB10_A2 0x8059
#endif
#ifndef GL_RGBA16
#define GL_RGBA16 0x805B
#endif
#ifndef GL_INVALID_INDEX
#define GL_INVALID_INDEX 0xFFFFFFFFu
#endif

namespace Fast {
namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------------------------------------------
// Preset (.slangp) parsing
// ---------------------------------------------------------------------------------------------------------------

enum class ScaleType { Source, Viewport, Absolute, Original };
enum class WrapMode { ClampToBorder, ClampToEdge, Repeat, MirroredRepeat };

struct PresetPass {
    std::string shaderPath;
    bool filterLinear = false;
    WrapMode wrap = WrapMode::ClampToBorder;
    bool hasScale = false;
    ScaleType scaleTypeX = ScaleType::Source;
    ScaleType scaleTypeY = ScaleType::Source;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    bool floatFramebuffer = false;
    bool srgbFramebuffer = false;
    bool mipmapInput = false;
    std::string alias;
    uint32_t frameCountMod = 0;
};

struct PresetTexture {
    std::string name;
    std::string path;
    bool linear = false;
    WrapMode wrap = WrapMode::ClampToBorder;
    bool mipmap = false;
};

struct Preset {
    std::vector<PresetPass> passes;
    std::vector<PresetTexture> textures;
    std::vector<std::pair<std::string, float>> paramOverrides;
};

struct PresetValue {
    std::string value;
    fs::path baseDir; // paths are relative to the file that set the key
};
using PresetMap = std::map<std::string, PresetValue>;

std::string Trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool StartsWith(const std::string& s, const char* prefix) {
    return s.compare(0, strlen(prefix), prefix) == 0;
}

std::string Unquote(std::string s) {
    s = Trim(s);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        s = s.substr(1, s.size() - 2);
    }
    return s;
}

bool ReadPresetFile(const fs::path& path, PresetMap& out, std::string& error, int depth) {
    if (depth > 16) {
        error = "Preset references nest too deeply at " + path.string();
        return false;
    }
    std::ifstream file(path);
    if (!file) {
        error = "Could not open preset " + path.string();
        return false;
    }
    const fs::path dir = path.parent_path();
    std::string line;
    while (std::getline(file, line)) {
        std::string t = Trim(line);
        if (t.empty()) {
            continue;
        }
        if (StartsWith(t, "#reference")) {
            // The referenced preset is the base, keys that follow override it
            std::string ref = Unquote(t.substr(strlen("#reference")));
            if (!ReadPresetFile(dir / ref, out, error, depth + 1)) {
                return false;
            }
            continue;
        }
        if (t[0] == '#') {
            continue;
        }
        size_t eq = t.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string key = Trim(t.substr(0, eq));
        std::string value = Trim(t.substr(eq + 1));
        if (!value.empty() && value[0] == '"') {
            size_t end = value.find('"', 1);
            value = value.substr(1, end == std::string::npos ? std::string::npos : end - 1);
        } else {
            size_t hash = value.find('#');
            if (hash != std::string::npos) {
                value = Trim(value.substr(0, hash));
            }
        }
        out[key] = { value, dir };
    }
    return true;
}

bool ParseBool(const std::string& v) {
    return v == "true" || v == "1" || v == "True" || v == "TRUE";
}

WrapMode ParseWrap(const std::string& v) {
    if (v == "clamp_to_edge") {
        return WrapMode::ClampToEdge;
    }
    if (v == "repeat") {
        return WrapMode::Repeat;
    }
    if (v == "mirrored_repeat") {
        return WrapMode::MirroredRepeat;
    }
    return WrapMode::ClampToBorder;
}

ScaleType ParseScaleType(const std::string& v) {
    if (v == "viewport") {
        return ScaleType::Viewport;
    }
    if (v == "absolute") {
        return ScaleType::Absolute;
    }
    if (v == "original") {
        return ScaleType::Original;
    }
    return ScaleType::Source;
}

bool ParseFloat(const std::string& s, float& out) {
    char* end = nullptr;
    out = strtof(s.c_str(), &end);
    return end != nullptr && end != s.c_str() && *Trim(end).c_str() == '\0';
}

bool BuildPreset(const PresetMap& map, Preset& preset, std::string& error) {
    auto get = [&](const std::string& key) -> const PresetValue* {
        auto it = map.find(key);
        return it == map.end() ? nullptr : &it->second;
    };

    const PresetValue* shaders = get("shaders");
    if (shaders == nullptr) {
        error = "Preset has no 'shaders' entry";
        return false;
    }
    const int count = atoi(shaders->value.c_str());
    if (count <= 0 || count > 64) {
        error = "Preset has an invalid number of shaders";
        return false;
    }

    std::set<std::string> known = { "shaders", "parameters", "textures", "feedback_pass" };
    for (int i = 0; i < count; i++) {
        const std::string n = std::to_string(i);
        const PresetValue* shader = get("shader" + n);
        if (shader == nullptr) {
            error = "Preset is missing shader" + n;
            return false;
        }
        PresetPass pass;
        pass.shaderPath = (shader->baseDir / shader->value).lexically_normal().string();
        if (const PresetValue* v = get("filter_linear" + n)) {
            pass.filterLinear = ParseBool(v->value);
        }
        if (const PresetValue* v = get("wrap_mode" + n)) {
            pass.wrap = ParseWrap(v->value);
        }
        if (const PresetValue* v = get("scale_type" + n)) {
            pass.hasScale = true;
            pass.scaleTypeX = pass.scaleTypeY = ParseScaleType(v->value);
        }
        if (const PresetValue* v = get("scale_type_x" + n)) {
            pass.hasScale = true;
            pass.scaleTypeX = ParseScaleType(v->value);
        }
        if (const PresetValue* v = get("scale_type_y" + n)) {
            pass.hasScale = true;
            pass.scaleTypeY = ParseScaleType(v->value);
        }
        if (const PresetValue* v = get("scale" + n)) {
            pass.scaleX = pass.scaleY = strtof(v->value.c_str(), nullptr);
        }
        if (const PresetValue* v = get("scale_x" + n)) {
            pass.scaleX = strtof(v->value.c_str(), nullptr);
        }
        if (const PresetValue* v = get("scale_y" + n)) {
            pass.scaleY = strtof(v->value.c_str(), nullptr);
        }
        if (const PresetValue* v = get("float_framebuffer" + n)) {
            pass.floatFramebuffer = ParseBool(v->value);
        }
        if (const PresetValue* v = get("srgb_framebuffer" + n)) {
            pass.srgbFramebuffer = ParseBool(v->value);
        }
        if (const PresetValue* v = get("mipmap_input" + n)) {
            pass.mipmapInput = ParseBool(v->value);
        }
        if (const PresetValue* v = get("alias" + n)) {
            pass.alias = Trim(v->value);
        }
        if (const PresetValue* v = get("frame_count_mod" + n)) {
            pass.frameCountMod = (uint32_t)atoi(v->value.c_str());
        }
        for (const char* key : { "shader", "filter_linear", "wrap_mode", "scale_type", "scale_type_x", "scale_type_y",
                                 "scale", "scale_x", "scale_y", "float_framebuffer", "srgb_framebuffer",
                                 "mipmap_input", "alias", "frame_count_mod" }) {
            known.insert(key + n);
        }
        preset.passes.push_back(pass);
    }

    if (const PresetValue* textures = get("textures")) {
        std::stringstream ss(textures->value);
        std::string name;
        while (std::getline(ss, name, ';')) {
            name = Trim(name);
            if (name.empty()) {
                continue;
            }
            const PresetValue* path = get(name);
            if (path == nullptr) {
                error = "Preset texture " + name + " has no path";
                return false;
            }
            PresetTexture texture;
            texture.name = name;
            texture.path = (path->baseDir / path->value).lexically_normal().string();
            if (const PresetValue* v = get(name + "_linear")) {
                texture.linear = ParseBool(v->value);
            }
            if (const PresetValue* v = get(name + "_wrap_mode")) {
                texture.wrap = ParseWrap(v->value);
            }
            if (const PresetValue* v = get(name + "_mipmap")) {
                texture.mipmap = ParseBool(v->value);
            }
            for (const char* suffix : { "", "_linear", "_wrap_mode", "_mipmap" }) {
                known.insert(name + suffix);
            }
            preset.textures.push_back(texture);
        }
    }

    // Everything else that is a number is a parameter override
    for (const auto& [key, value] : map) {
        if (known.count(key)) {
            continue;
        }
        float f;
        if (ParseFloat(value.value, f)) {
            preset.paramOverrides.emplace_back(key, f);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Shader (.slang) loading and translation to plain GLSL
// ---------------------------------------------------------------------------------------------------------------

struct ShaderSource {
    std::string common;
    std::string vertex;
    std::string fragment;
    std::vector<PostFilterParam> params;
    std::string name;
    std::string format;
};

bool ParseParameterPragma(const std::string& line, PostFilterParam& param) {
    // #pragma parameter NAME "DESCRIPTION" initial minimum maximum [step]
    std::string rest = Trim(line.substr(line.find("parameter") + strlen("parameter")));
    size_t space = rest.find_first_of(" \t");
    if (space == std::string::npos) {
        return false;
    }
    param.name = rest.substr(0, space);
    rest = Trim(rest.substr(space));
    if (rest.empty() || rest[0] != '"') {
        return false;
    }
    size_t endQuote = rest.find('"', 1);
    if (endQuote == std::string::npos) {
        return false;
    }
    param.description = rest.substr(1, endQuote - 1);
    rest = rest.substr(endQuote + 1);
    size_t comment = rest.find("//");
    if (comment != std::string::npos) {
        rest = rest.substr(0, comment);
    }
    std::stringstream ss(rest);
    float values[4];
    int n = 0;
    while (n < 4 && ss >> values[n]) {
        n++;
    }
    if (n < 3) {
        return false;
    }
    param.initial = values[0];
    param.minimum = values[1];
    param.maximum = values[2];
    param.step = n == 4 ? values[3] : 0.1f * (values[2] - values[1]);
    return true;
}

bool ReadShaderFile(const fs::path& path, ShaderSource& out, int& stage, std::string& error, int depth) {
    if (depth > 16) {
        error = "Includes nest too deeply at " + path.string();
        return false;
    }
    std::ifstream file(path);
    if (!file) {
        error = "Could not open shader " + path.string();
        return false;
    }
    const fs::path dir = path.parent_path();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string t = Trim(line);
        if (StartsWith(t, "#include")) {
            std::string inc = Unquote(t.substr(strlen("#include")));
            size_t comment = inc.find("//");
            if (comment != std::string::npos) {
                inc = Trim(inc.substr(0, comment));
            }
            if (!ReadShaderFile(dir / inc, out, stage, error, depth + 1)) {
                return false;
            }
            continue;
        }
        if (StartsWith(t, "#version")) {
            continue;
        }
        if (StartsWith(t, "#pragma")) {
            const std::string pragma = Trim(t.substr(strlen("#pragma")));
            if (StartsWith(pragma, "stage")) {
                const std::string which = Trim(pragma.substr(strlen("stage")));
                stage = StartsWith(which, "vertex") ? 1 : StartsWith(which, "fragment") ? 2 : stage;
            } else if (StartsWith(pragma, "parameter")) {
                PostFilterParam param;
                if (ParseParameterPragma(pragma, param)) {
                    bool exists = false;
                    for (const PostFilterParam& p : out.params) {
                        exists |= p.name == param.name;
                    }
                    if (!exists) {
                        out.params.push_back(param);
                    }
                } else {
                    SPDLOG_WARN("Ignoring malformed parameter pragma in {}: {}", path.string(), t);
                }
            } else if (StartsWith(pragma, "name")) {
                out.name = Trim(pragma.substr(strlen("name")));
            } else if (StartsWith(pragma, "format")) {
                out.format = Trim(pragma.substr(strlen("format")));
            }
            // Other pragmas are RetroArch metadata the driver must not see
            continue;
        }
        std::string& section = stage == 1 ? out.vertex : stage == 2 ? out.fragment : out.common;
        section += line;
        section += '\n';
    }
    return true;
}

// Reflection of a uniform block, laid out per the std140 rules
struct BlockMember {
    std::string name;
    std::string type;
    uint32_t offset = 0;
};

struct BlockInfo {
    std::string blockName;    // the name as declared in the slang source
    std::string instanceName; // may be empty
    bool pushConstant = false;
    std::vector<BlockMember> members;
    uint32_t size = 0;
};

struct StageSource {
    std::string glsl;
    std::vector<BlockInfo> blocks;
    std::vector<std::string> samplers;
};

std::string StripComments(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (s.compare(i, 2, "//") == 0) {
            while (i < s.size() && s[i] != '\n') {
                i++;
            }
        } else if (s.compare(i, 2, "/*") == 0) {
            size_t end = s.find("*/", i + 2);
            i = end == std::string::npos ? s.size() : end + 2;
            out += ' ';
        } else {
            out += s[i++];
        }
    }
    return out;
}

bool Std140Layout(const std::string& type, uint32_t& size, uint32_t& align) {
    static const std::map<std::string, std::pair<uint32_t, uint32_t>> layouts = {
        { "float", { 4, 4 } },   { "int", { 4, 4 } },     { "uint", { 4, 4 } },    { "bool", { 4, 4 } },
        { "vec2", { 8, 8 } },    { "ivec2", { 8, 8 } },   { "uvec2", { 8, 8 } },   { "bvec2", { 8, 8 } },
        { "vec3", { 12, 16 } },  { "ivec3", { 12, 16 } }, { "uvec3", { 12, 16 } }, { "bvec3", { 12, 16 } },
        { "vec4", { 16, 16 } },  { "ivec4", { 16, 16 } }, { "uvec4", { 16, 16 } }, { "bvec4", { 16, 16 } },
        { "mat2", { 32, 16 } },  { "mat3", { 48, 16 } },  { "mat4", { 64, 16 } },  { "double", { 8, 8 } },
        { "dvec2", { 16, 16 } }, { "dvec4", { 32, 16 } },
    };
    auto it = layouts.find(type);
    if (it == layouts.end()) {
        return false;
    }
    size = it->second.first;
    align = it->second.second;
    return true;
}

void ParseBlockMembers(const std::string& body, BlockInfo& block) {
    uint32_t offset = 0;
    std::stringstream decls(StripComments(body));
    std::string decl;
    while (std::getline(decls, decl, ';')) {
        decl = Trim(decl);
        if (decl.empty()) {
            continue;
        }
        // "type a, b[4], c" possibly preceded by precision qualifiers
        for (char& c : decl) {
            if (c == ',' || c == '\n' || c == '\t') {
                c = ' ';
            }
        }
        std::stringstream tokens(decl);
        std::string type;
        while (tokens >> type) {
            if (type != "highp" && type != "mediump" && type != "lowp") {
                break;
            }
        }
        uint32_t size = 16, align = 16;
        if (!Std140Layout(type, size, align)) {
            SPDLOG_WARN("Unknown uniform block member type {} in block {}", type, block.blockName);
        }
        std::string name;
        while (tokens >> name) {
            uint32_t count = 1;
            size_t bracket = name.find('[');
            if (bracket != std::string::npos) {
                count = std::max(1, atoi(name.c_str() + bracket + 1));
                name = name.substr(0, bracket);
            }
            uint32_t memberSize = size, memberAlign = align;
            if (count > 1) {
                memberAlign = 16;
                memberSize = ((size + 15) / 16) * 16 * count;
            }
            offset = (offset + memberAlign - 1) / memberAlign * memberAlign;
            block.members.push_back({ name, type, offset });
            offset += memberSize;
        }
    }
    block.size = (offset + 15) / 16 * 16;
}

// Rewrites one stage of a slang shader into GLSL the driver can take, collecting what needs binding by name
StageSource TranslateStage(const std::string& source, bool vertexStage, int glslVersion, bool es) {
    StageSource out;
    const char* suffix = vertexStage ? "_vs" : "_fs";
    std::string glsl = source;

    // Uniform blocks: "layout(...) uniform Name { ... } instance;" -> std140 blocks with per-stage unique names, so
    // the vertex and fragment stages may declare different members without a link-time mismatch
    static const std::regex blockRegex(
        R"(layout\s*\(([^)]*)\)\s*uniform\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{([^}]*)\}\s*([A-Za-z_][A-Za-z0-9_]*)?\s*;)");
    std::string rebuilt;
    std::sregex_iterator it(glsl.begin(), glsl.end(), blockRegex), end;
    size_t last = 0;
    for (; it != end; ++it) {
        const std::smatch& m = *it;
        BlockInfo block;
        block.pushConstant = m[1].str().find("push_constant") != std::string::npos;
        block.blockName = m[2].str();
        block.instanceName = m[4].matched ? m[4].str() : "";
        ParseBlockMembers(m[3].str(), block);
        out.blocks.push_back(block);

        rebuilt += glsl.substr(last, m.position() - last);
        rebuilt += "layout(std140) uniform " + block.blockName + suffix + " {" + m[3].str() + "} " +
                   block.instanceName + ";";
        last = m.position() + m.length();
    }
    rebuilt += glsl.substr(last);
    glsl = rebuilt;

    // Drop Vulkan-only qualifiers from whatever layout() is left (samplers mostly)
    static const std::regex layoutRegex(R"(layout\s*\(([^)]*)\))");
    rebuilt.clear();
    last = 0;
    for (std::sregex_iterator lit(glsl.begin(), glsl.end(), layoutRegex); lit != end; ++lit) {
        const std::smatch& m = *lit;
        std::stringstream ss(m[1].str());
        std::string qualifier, kept;
        while (std::getline(ss, qualifier, ',')) {
            qualifier = Trim(qualifier);
            if (qualifier.empty() || qualifier == "push_constant" || StartsWith(qualifier, "set") ||
                StartsWith(qualifier, "binding")) {
                continue;
            }
            kept += (kept.empty() ? "" : ", ") + qualifier;
        }
        rebuilt += glsl.substr(last, m.position() - last);
        if (!kept.empty()) {
            rebuilt += "layout(" + kept + ")";
        }
        last = m.position() + m.length();
    }
    rebuilt += glsl.substr(last);
    glsl = rebuilt;

    // Explicit locations on varyings need GLSL 4.10, older drivers link them by name
    if (glslVersion < 410 && !es) {
        const std::regex varyingRegex(vertexStage ? R"(layout\s*\(\s*location\s*=\s*\d+\s*\)\s*((?:flat|smooth|noperspective)\s+)?out\b)"
                                                  : R"(layout\s*\(\s*location\s*=\s*\d+\s*\)\s*((?:flat|smooth|noperspective)\s+)?in\b)");
        glsl = std::regex_replace(glsl, varyingRegex, vertexStage ? "$1out" : "$1in");
    }

    static const std::regex samplerRegex(R"(uniform\s+sampler2D\s+([A-Za-z_][A-Za-z0-9_]*)\s*;)");
    for (std::sregex_iterator sit(glsl.begin(), glsl.end(), samplerRegex); sit != end; ++sit) {
        out.samplers.push_back((*sit)[1].str());
    }

    std::string header;
    if (es) {
        header = "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n";
    } else {
        header = "#version " + std::to_string(glslVersion) + "\n";
    }
    out.glsl = header + glsl;
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// GL objects
// ---------------------------------------------------------------------------------------------------------------

enum class UniformSemantic {
    MVP,
    OutputSize,
    FinalViewportSize,
    SourceSize,
    OriginalSize,
    FrameCount,
    FrameDirection,
    FrameTimeDelta,
    OriginalFPS,
    Rotation,
    TotalSubFrames,
    CurrentSubFrame,
    OriginalHistorySize, // index = history depth
    PassOutputSize,      // index = pass
    PassFeedbackSize,    // index = pass
    TextureSize,         // index = LUT
    Parameter,
    Unknown,
};

enum class TextureSemantic {
    Original,
    Source,
    OriginalHistory, // index = history depth
    PassOutput,      // index = pass
    PassFeedback,    // index = pass
    Texture,         // index = LUT
    Unknown,
};

struct UniformBinding {
    uint32_t offset;
    std::string type;
    UniformSemantic semantic;
    int index = 0;
    std::string parameter;
};

struct BlockBinding {
    bool active = false;
    GLuint buffer = 0;
    GLuint bindingPoint = 0;
    std::vector<uint8_t> data;
    std::vector<UniformBinding> uniforms;
};

struct TextureBinding {
    GLint unit;
    TextureSemantic semantic;
    int index = 0;
};

struct FormatInfo {
    GLenum internalFormat;
    GLenum format;
    GLenum type;
    bool srgb;
};

FormatInfo FormatFromName(const std::string& name, bool floatFramebuffer, bool srgbFramebuffer) {
    // Preset flags win over the shader's #pragma format, as in RetroArch
    if (srgbFramebuffer) {
        return { GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, true };
    }
    if (floatFramebuffer) {
        return { GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, false };
    }
    static const std::map<std::string, FormatInfo> formats = {
        { "R8G8B8A8_UNORM", { GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, false } },
        { "R8G8B8A8_SRGB", { GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, true } },
        { "R16G16B16A16_SFLOAT", { GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, false } },
        { "R32G32B32A32_SFLOAT", { GL_RGBA32F, GL_RGBA, GL_FLOAT, false } },
        { "R16G16B16A16_UNORM", { GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT, false } },
        { "A2B10G10R10_UNORM_PACK32", { GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, false } },
        { "B10G11R11_UFLOAT_PACK32", { GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, false } },
        { "R8_UNORM", { GL_R8, GL_RED, GL_UNSIGNED_BYTE, false } },
        { "R8G8_UNORM", { GL_RG8, GL_RG, GL_UNSIGNED_BYTE, false } },
        { "R16_SFLOAT", { GL_R16F, GL_RED, GL_HALF_FLOAT, false } },
        { "R16G16_SFLOAT", { GL_RG16F, GL_RG, GL_HALF_FLOAT, false } },
        { "R32_SFLOAT", { GL_R32F, GL_RED, GL_FLOAT, false } },
        { "R32G32_SFLOAT", { GL_RG32F, GL_RG, GL_FLOAT, false } },
    };
    auto it = formats.find(name);
    if (it != formats.end()) {
        return it->second;
    }
    if (!name.empty()) {
        SPDLOG_WARN("Unsupported shader framebuffer format {}, using RGBA8", name);
    }
    return { GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, false };
}

struct Pass {
    PresetPass config;
    std::string alias;
    FormatInfo format = { GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, false };
    GLuint program = 0;
    BlockBinding blocks[4]; // vertex UBO, fragment UBO, vertex push, fragment push
    std::vector<TextureBinding> textures;
    GLint maxUnit = -1;

    // Output (unused for the final pass, which draws to the caller's framebuffer)
    bool feedback = false; // keeps the previous frame's output for PassFeedback / <alias>Feedback
    GLuint texture[2] = { 0, 0 };
    GLuint fbo[2] = { 0, 0 };
    uint32_t width = 0;
    uint32_t height = 0;
};

struct LutTexture {
    PresetTexture config;
    GLuint texture = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

// Everything the chain changes is put back, the game renderer skips redundant state changes based on what it set
struct GLStateBackup {
    GLint program = 0, vao = 0, arrayBuffer = 0, uniformBuffer = 0, drawFbo = 0, readFbo = 0, activeTexture = 0;
    GLint viewport[4] = {}, scissor[4] = {};
    GLboolean blend = 0, depthTest = 0, scissorTest = 0, cullFace = 0, stencilTest = 0, polygonOffset = 0,
              srgb = 0, depthMask = 0;
    GLboolean colorMask[4] = {};
    std::vector<GLint> textures, samplers;
    GLint uniformBuffers[4] = {};

    void Save(int units) {
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &arrayBuffer);
        glGetIntegerv(GL_UNIFORM_BUFFER_BINDING, &uniformBuffer);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFbo);
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFbo);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
        glGetIntegerv(GL_VIEWPORT, viewport);
        glGetIntegerv(GL_SCISSOR_BOX, scissor);
        blend = glIsEnabled(GL_BLEND);
        depthTest = glIsEnabled(GL_DEPTH_TEST);
        scissorTest = glIsEnabled(GL_SCISSOR_TEST);
        cullFace = glIsEnabled(GL_CULL_FACE);
        stencilTest = glIsEnabled(GL_STENCIL_TEST);
        polygonOffset = glIsEnabled(GL_POLYGON_OFFSET_FILL);
#ifdef GL_FRAMEBUFFER_SRGB
        srgb = glIsEnabled(GL_FRAMEBUFFER_SRGB);
#endif
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);
        glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);
        textures.resize(units);
        samplers.resize(units);
        for (int i = 0; i < units; i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &textures[i]);
            glGetIntegerv(GL_SAMPLER_BINDING, &samplers[i]);
        }
        for (int i = 0; i < 4; i++) {
            glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, i, &uniformBuffers[i]);
        }
    }

    void Restore() {
        for (size_t i = 0; i < textures.size(); i++) {
            glActiveTexture(GL_TEXTURE0 + (GLenum)i);
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            glBindSampler((GLuint)i, samplers[i]);
        }
        for (int i = 0; i < 4; i++) {
            glBindBufferBase(GL_UNIFORM_BUFFER, i, uniformBuffers[i]);
        }
        glActiveTexture(activeTexture);
        glUseProgram(program);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, arrayBuffer);
        glBindBuffer(GL_UNIFORM_BUFFER, uniformBuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFbo);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
        glScissor(scissor[0], scissor[1], scissor[2], scissor[3]);
        auto set = [](GLenum cap, GLboolean on) { on ? glEnable(cap) : glDisable(cap); };
        set(GL_BLEND, blend);
        set(GL_DEPTH_TEST, depthTest);
        set(GL_SCISSOR_TEST, scissorTest);
        set(GL_CULL_FACE, cullFace);
        set(GL_STENCIL_TEST, stencilTest);
        set(GL_POLYGON_OFFSET_FILL, polygonOffset);
#ifdef GL_FRAMEBUFFER_SRGB
        set(GL_FRAMEBUFFER_SRGB, srgb);
#endif
        glDepthMask(depthMask);
        glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
    }
};

void WriteFloat(std::vector<uint8_t>& data, uint32_t offset, float value) {
    if (offset + sizeof(float) <= data.size()) {
        memcpy(data.data() + offset, &value, sizeof(float));
    }
}

void WriteUint(std::vector<uint8_t>& data, uint32_t offset, uint32_t value) {
    if (offset + sizeof(uint32_t) <= data.size()) {
        memcpy(data.data() + offset, &value, sizeof(uint32_t));
    }
}

void WriteScalar(std::vector<uint8_t>& data, const UniformBinding& u, double value) {
    if (u.type == "float") {
        WriteFloat(data, u.offset, (float)value);
    } else if (u.type == "int") {
        int32_t v = (int32_t)llround(value);
        WriteUint(data, u.offset, (uint32_t)v);
    } else if (u.type == "uint" || u.type == "bool") {
        WriteUint(data, u.offset, (uint32_t)llround(value));
    }
}

void WriteSize(std::vector<uint8_t>& data, const UniformBinding& u, uint32_t width, uint32_t height) {
    const float w = (float)std::max(width, 1u), h = (float)std::max(height, 1u);
    if (u.type == "vec4") {
        WriteFloat(data, u.offset, w);
        WriteFloat(data, u.offset + 4, h);
        WriteFloat(data, u.offset + 8, 1.0f / w);
        WriteFloat(data, u.offset + 12, 1.0f / h);
    } else if (u.type == "vec2") {
        WriteFloat(data, u.offset, w);
        WriteFloat(data, u.offset + 4, h);
    }
}

bool CompileShader(GLenum type, const std::string& source, GLuint& shader, std::string& log) {
    shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    if (length > 1) {
        log.resize(length);
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        log = Trim(log);
    }
    if (!ok) {
        glDeleteShader(shader);
        shader = 0;
        return false;
    }
    return true;
}

std::string NumberedSource(const std::string& source) {
    std::stringstream in(source), out;
    std::string line;
    int n = 1;
    while (std::getline(in, line)) {
        out << n++ << ": " << line << '\n';
    }
    return out.str();
}

bool ParseIndexed(const std::string& name, const char* prefix, int& index) {
    const size_t len = strlen(prefix);
    if (name.compare(0, len, prefix) != 0 || name.size() == len) {
        return false;
    }
    for (size_t i = len; i < name.size(); i++) {
        if (!isdigit((unsigned char)name[i])) {
            return false;
        }
    }
    index = atoi(name.c_str() + len);
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// The chain
// ---------------------------------------------------------------------------------------------------------------

struct SlangFilterChain::Impl {
    std::vector<Pass> passes;
    std::vector<LutTexture> luts;
    std::vector<PostFilterParam> params;
    std::map<std::string, float> paramValues;

    int glslVersion = 0;
    bool es = false;
    GLuint vao = 0;
    GLuint vbo = 0;
    std::map<uint32_t, GLuint> samplers;

    uint32_t sourceWidth = 0, sourceHeight = 0, targetWidth = 0, targetHeight = 0;
    int parity = 0; // which of the two feedback buffers is current

    int historyDepth = 0;
    std::vector<GLuint> historyTextures;
    std::vector<GLuint> historyFbos;
    GLuint sourceCopyFbo = 0;
    uint32_t historyFrame = 0;

    ~Impl() {
        Destroy();
    }

    void Destroy() {
        for (Pass& pass : passes) {
            if (pass.program) {
                glDeleteProgram(pass.program);
            }
            for (BlockBinding& block : pass.blocks) {
                if (block.buffer) {
                    glDeleteBuffers(1, &block.buffer);
                }
            }
            for (int i = 0; i < 2; i++) {
                if (pass.fbo[i]) {
                    glDeleteFramebuffers(1, &pass.fbo[i]);
                }
                if (pass.texture[i]) {
                    glDeleteTextures(1, &pass.texture[i]);
                }
            }
        }
        passes.clear();
        for (LutTexture& lut : luts) {
            if (lut.texture) {
                glDeleteTextures(1, &lut.texture);
            }
        }
        luts.clear();
        for (auto& [key, sampler] : samplers) {
            glDeleteSamplers(1, &sampler);
        }
        samplers.clear();
        if (!historyTextures.empty()) {
            glDeleteTextures((GLsizei)historyTextures.size(), historyTextures.data());
            glDeleteFramebuffers((GLsizei)historyFbos.size(), historyFbos.data());
        }
        historyTextures.clear();
        historyFbos.clear();
        if (sourceCopyFbo) {
            glDeleteFramebuffers(1, &sourceCopyFbo);
            sourceCopyFbo = 0;
        }
        if (vao) {
            glDeleteVertexArrays(1, &vao);
            vao = 0;
        }
        if (vbo) {
            glDeleteBuffers(1, &vbo);
            vbo = 0;
        }
        params.clear();
        paramValues.clear();
        sourceWidth = sourceHeight = targetWidth = targetHeight = 0;
    }

    GLuint GetSampler(bool linear, WrapMode wrap, bool mipmap) {
        const uint32_t key = (linear ? 1 : 0) | ((uint32_t)wrap << 1) | (mipmap ? 8 : 0);
        auto it = samplers.find(key);
        if (it != samplers.end()) {
            return it->second;
        }
        GLuint sampler = 0;
        glGenSamplers(1, &sampler);
        const GLint mag = linear ? GL_LINEAR : GL_NEAREST;
        const GLint min = mipmap ? (linear ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_NEAREST) : mag;
        GLint glWrap = GL_CLAMP_TO_BORDER;
        switch (wrap) {
            case WrapMode::ClampToEdge:
                glWrap = GL_CLAMP_TO_EDGE;
                break;
            case WrapMode::Repeat:
                glWrap = GL_REPEAT;
                break;
            case WrapMode::MirroredRepeat:
                glWrap = GL_MIRRORED_REPEAT;
                break;
            default:
                break;
        }
        glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, min);
        glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, mag);
        glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, glWrap);
        glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, glWrap);
        samplers[key] = sampler;
        return sampler;
    }

    int FindAlias(const std::string& name) const {
        for (size_t i = 0; i < passes.size(); i++) {
            if (!passes[i].alias.empty() && passes[i].alias == name) {
                return (int)i;
            }
        }
        return -1;
    }

    int FindLut(const std::string& name) const {
        for (size_t i = 0; i < luts.size(); i++) {
            if (luts[i].config.name == name) {
                return (int)i;
            }
        }
        return -1;
    }

    bool IsParameter(const std::string& name) const {
        return paramValues.count(name) != 0;
    }

    UniformBinding ResolveUniform(const BlockMember& member, size_t passIndex) {
        UniformBinding u;
        u.offset = member.offset;
        u.type = member.type;
        u.semantic = UniformSemantic::Unknown;
        const std::string& n = member.name;
        static const std::map<std::string, UniformSemantic> builtins = {
            { "MVP", UniformSemantic::MVP },
            { "OutputSize", UniformSemantic::OutputSize },
            { "FinalViewportSize", UniformSemantic::FinalViewportSize },
            { "SourceSize", UniformSemantic::SourceSize },
            { "OriginalSize", UniformSemantic::OriginalSize },
            { "FrameCount", UniformSemantic::FrameCount },
            { "FrameDirection", UniformSemantic::FrameDirection },
            { "FrameTimeDelta", UniformSemantic::FrameTimeDelta },
            { "OriginalFPS", UniformSemantic::OriginalFPS },
            { "Rotation", UniformSemantic::Rotation },
            { "TotalSubFrames", UniformSemantic::TotalSubFrames },
            { "CurrentSubFrame", UniformSemantic::CurrentSubFrame },
        };
        if (auto it = builtins.find(n); it != builtins.end()) {
            u.semantic = it->second;
            return u;
        }
        int index = 0;
        if (ParseIndexed(n, "OriginalHistorySize", index)) {
            u.semantic = UniformSemantic::OriginalHistorySize;
            u.index = index;
            historyDepth = std::max(historyDepth, index);
            return u;
        }
        if (ParseIndexed(n, "PassOutputSize", index) && index < (int)passes.size()) {
            u.semantic = UniformSemantic::PassOutputSize;
            u.index = index;
            return u;
        }
        if (ParseIndexed(n, "PassFeedbackSize", index) && index < (int)passes.size()) {
            u.semantic = UniformSemantic::PassFeedbackSize;
            u.index = index;
            passes[index].feedback = true;
            return u;
        }
        if (n.size() > 4 && n.compare(n.size() - 4, 4, "Size") == 0) {
            const std::string base = n.substr(0, n.size() - 4);
            if (int lut = FindLut(base); lut >= 0) {
                u.semantic = UniformSemantic::TextureSize;
                u.index = lut;
                return u;
            }
            if (int alias = FindAlias(base); alias >= 0) {
                u.semantic = UniformSemantic::PassOutputSize;
                u.index = alias;
                return u;
            }
            if (base.size() > 8 && base.compare(base.size() - 8, 8, "Feedback") == 0) {
                if (int alias = FindAlias(base.substr(0, base.size() - 8)); alias >= 0) {
                    u.semantic = UniformSemantic::PassFeedbackSize;
                    u.index = alias;
                    passes[alias].feedback = true;
                    return u;
                }
            }
        }
        if (IsParameter(n)) {
            u.semantic = UniformSemantic::Parameter;
            u.parameter = n;
            return u;
        }
        SPDLOG_WARN("Pass {}: uniform {} has no known meaning, it stays zero", passIndex, n);
        return u;
    }

    bool ResolveTexture(const std::string& name, TextureSemantic& semantic, int& index, size_t passIndex) {
        index = 0;
        if (name == "Original") {
            semantic = TextureSemantic::Original;
            return true;
        }
        if (name == "Source") {
            semantic = TextureSemantic::Source;
            return true;
        }
        if (ParseIndexed(name, "OriginalHistory", index)) {
            semantic = index == 0 ? TextureSemantic::Original : TextureSemantic::OriginalHistory;
            historyDepth = std::max(historyDepth, index);
            return true;
        }
        if (ParseIndexed(name, "PassOutput", index) && index < (int)passIndex) {
            semantic = TextureSemantic::PassOutput;
            return true;
        }
        if (ParseIndexed(name, "PassFeedback", index) && index < (int)passes.size()) {
            semantic = TextureSemantic::PassFeedback;
            passes[index].feedback = true;
            return true;
        }
        if (int lut = FindLut(name); lut >= 0) {
            semantic = TextureSemantic::Texture;
            index = lut;
            return true;
        }
        if (int alias = FindAlias(name); alias >= 0 && alias < (int)passIndex) {
            semantic = TextureSemantic::PassOutput;
            index = alias;
            return true;
        }
        if (name.size() > 8 && name.compare(name.size() - 8, 8, "Feedback") == 0) {
            if (int alias = FindAlias(name.substr(0, name.size() - 8)); alias >= 0) {
                semantic = TextureSemantic::PassFeedback;
                index = alias;
                passes[alias].feedback = true;
                return true;
            }
        }
        semantic = TextureSemantic::Unknown;
        return false;
    }

    bool BuildPass(size_t index, const ShaderSource& source, std::string& error) {
        Pass& pass = passes[index];
        const std::string common = source.common;
        StageSource vs = TranslateStage(common + source.vertex, true, glslVersion, es);
        StageSource fsrc = TranslateStage(common + source.fragment, false, glslVersion, es);

        const std::string where = "pass " + std::to_string(index) + " (" + pass.config.shaderPath + ")";
        GLuint vertex = 0, fragment = 0;
        std::string log;
        if (!CompileShader(GL_VERTEX_SHADER, vs.glsl, vertex, log)) {
            SPDLOG_ERROR("Vertex shader of {} failed to compile:\n{}\n{}", where, log, NumberedSource(vs.glsl));
            error = where + ": vertex shader failed to compile:\n" + log;
            return false;
        }
        if (!CompileShader(GL_FRAGMENT_SHADER, fsrc.glsl, fragment, log)) {
            SPDLOG_ERROR("Fragment shader of {} failed to compile:\n{}\n{}", where, log, NumberedSource(fsrc.glsl));
            glDeleteShader(vertex);
            error = where + ": fragment shader failed to compile:\n" + log;
            return false;
        }

        pass.program = glCreateProgram();
        glAttachShader(pass.program, vertex);
        glAttachShader(pass.program, fragment);
        glLinkProgram(pass.program);
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        GLint linked = GL_FALSE;
        glGetProgramiv(pass.program, GL_LINK_STATUS, &linked);
        if (!linked) {
            GLint length = 0;
            glGetProgramiv(pass.program, GL_INFO_LOG_LENGTH, &length);
            log.assign(std::max(length, 1), '\0');
            glGetProgramInfoLog(pass.program, length, nullptr, log.data());
            error = where + ": failed to link:\n" + Trim(log);
            SPDLOG_ERROR("{}", error);
            return false;
        }

        // Uniform blocks, one buffer per stage and kind so the stages may lay them out differently
        const struct {
            const std::vector<BlockInfo>& blocks;
            const char* suffix;
            bool push;
            int slot;
        } stages[] = {
            { vs.blocks, "_vs", false, 0 }, { fsrc.blocks, "_fs", false, 1 },
            { vs.blocks, "_vs", true, 2 },  { fsrc.blocks, "_fs", true, 3 },
        };
        for (const auto& stage : stages) {
            for (const BlockInfo& info : stage.blocks) {
                if (info.pushConstant != stage.push) {
                    continue;
                }
                BlockBinding& block = pass.blocks[stage.slot];
                const GLuint blockIndex = glGetUniformBlockIndex(pass.program, (info.blockName + stage.suffix).c_str());
                if (blockIndex == GL_INVALID_INDEX) {
                    continue; // the whole block went unused and was optimised away
                }
                GLint driverSize = 0;
                glGetActiveUniformBlockiv(pass.program, blockIndex, GL_UNIFORM_BLOCK_DATA_SIZE, &driverSize);
                block.active = true;
                block.bindingPoint = (GLuint)stage.slot;
                block.data.assign(std::max<uint32_t>(info.size, (uint32_t)driverSize), 0);
                glUniformBlockBinding(pass.program, blockIndex, block.bindingPoint);
                glGenBuffers(1, &block.buffer);
                glBindBuffer(GL_UNIFORM_BUFFER, block.buffer);
                glBufferData(GL_UNIFORM_BUFFER, block.data.size(), nullptr, GL_STREAM_DRAW);
                for (const BlockMember& member : info.members) {
                    block.uniforms.push_back(ResolveUniform(member, index));
                }
                break;
            }
        }

        // Samplers, bound to texture units in order of appearance
        glUseProgram(pass.program);
        std::set<std::string> seen;
        GLint unit = 0;
        std::vector<std::string> samplerNames = vs.samplers;
        samplerNames.insert(samplerNames.end(), fsrc.samplers.begin(), fsrc.samplers.end());
        for (const std::string& name : samplerNames) {
            if (!seen.insert(name).second) {
                continue;
            }
            const GLint location = glGetUniformLocation(pass.program, name.c_str());
            if (location < 0) {
                continue;
            }
            TextureBinding binding;
            binding.unit = unit;
            if (!ResolveTexture(name, binding.semantic, binding.index, index)) {
                SPDLOG_WARN("{}: texture {} has no known source", where, name);
            }
            glUniform1i(location, unit);
            pass.textures.push_back(binding);
            pass.maxUnit = unit;
            unit++;
        }
        glUseProgram(0);
        return true;
    }

    bool LoadLut(LutTexture& lut, std::string& error) {
        int w = 0, h = 0, channels = 0;
        stbi_uc* pixels = stbi_load(lut.config.path.c_str(), &w, &h, &channels, 4);
        if (pixels == nullptr) {
            error = "Could not load texture " + lut.config.path + ": " +
                    (stbi_failure_reason() ? stbi_failure_reason() : "unknown error");
            return false;
        }
        glGenTextures(1, &lut.texture);
        glBindTexture(GL_TEXTURE_2D, lut.texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        if (lut.config.mipmap) {
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        stbi_image_free(pixels);
        lut.width = (uint32_t)w;
        lut.height = (uint32_t)h;
        return true;
    }

    void AllocateTexture(GLuint& texture, GLuint& fbo, const FormatInfo& format, uint32_t width, uint32_t height) {
        if (texture == 0) {
            glGenTextures(1, &texture);
        }
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, format.internalFormat, width, height, 0, format.format, format.type, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
        if (fbo == 0) {
            glGenFramebuffers(1, &fbo);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            SPDLOG_ERROR("Shader pass framebuffer ({}x{}, format {:#x}) is incomplete", width, height,
                         format.internalFormat);
        }
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    // Pass output sizes follow the source and viewport sizes, so they are laid out again whenever those change
    void Resize(uint32_t srcW, uint32_t srcH, uint32_t dstW, uint32_t dstH) {
        sourceWidth = srcW;
        sourceHeight = srcH;
        targetWidth = dstW;
        targetHeight = dstH;

        uint32_t inputW = srcW, inputH = srcH;
        for (size_t i = 0; i < passes.size(); i++) {
            Pass& pass = passes[i];
            const bool last = i + 1 == passes.size();
            uint32_t w, h;
            if (last) {
                w = dstW;
                h = dstH;
            } else {
                auto scale = [&](ScaleType type, float factor, uint32_t input, uint32_t viewport, uint32_t original) {
                    switch (type) {
                        case ScaleType::Viewport:
                            return (uint32_t)std::lround(viewport * factor);
                        case ScaleType::Absolute:
                            return (uint32_t)std::lround(factor);
                        case ScaleType::Original:
                            return (uint32_t)std::lround(original * factor);
                        default:
                            return (uint32_t)std::lround(input * factor);
                    }
                };
                w = scale(pass.config.scaleTypeX, pass.config.scaleX, inputW, dstW, srcW);
                h = scale(pass.config.scaleTypeY, pass.config.scaleY, inputH, dstH, srcH);
            }
            w = std::max(w, 1u);
            h = std::max(h, 1u);
            if (!last && (pass.width != w || pass.height != h)) {
                AllocateTexture(pass.texture[0], pass.fbo[0], pass.format, w, h);
                if (pass.feedback) {
                    AllocateTexture(pass.texture[1], pass.fbo[1], pass.format, w, h);
                }
            }
            pass.width = w;
            pass.height = h;
            inputW = w;
            inputH = h;
        }

        if (historyDepth > 0) {
            const FormatInfo rgba = { GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, false };
            historyTextures.resize(historyDepth, 0);
            historyFbos.resize(historyDepth, 0);
            for (int i = 0; i < historyDepth; i++) {
                AllocateTexture(historyTextures[i], historyFbos[i], rgba, srcW, srcH);
            }
            if (sourceCopyFbo == 0) {
                glGenFramebuffers(1, &sourceCopyFbo);
            }
        }
    }

    // Sampling state of a texture depends on who consumes it: Source uses its pass' filter, Original the first
    // pass', and a pass' output the filter of the pass that comes after it
    void BindTexture(const TextureBinding& binding, size_t passIndex, GLuint sourceTexture) {
        GLuint texture = 0;
        const PresetPass* consumer = &passes[0].config;
        bool mipmap = false;
        switch (binding.semantic) {
            case TextureSemantic::Original:
                texture = sourceTexture;
                mipmap = passes[0].config.mipmapInput;
                break;
            case TextureSemantic::Source:
                texture = passIndex == 0 ? sourceTexture : passes[passIndex - 1].texture[OutputSlot(passIndex - 1)];
                consumer = &passes[passIndex].config;
                mipmap = consumer->mipmapInput;
                break;
            case TextureSemantic::OriginalHistory:
                if (binding.index >= 1 && binding.index <= historyDepth && historyFrame >= (uint32_t)binding.index) {
                    texture = historyTextures[(historyFrame - binding.index) % historyDepth];
                }
                break;
            case TextureSemantic::PassOutput: {
                const size_t next = std::min<size_t>(binding.index + 1, passes.size() - 1);
                texture = passes[binding.index].texture[OutputSlot(binding.index)];
                consumer = &passes[next].config;
                mipmap = consumer->mipmapInput && next == (size_t)binding.index + 1;
                break;
            }
            case TextureSemantic::PassFeedback: {
                const size_t next = std::min<size_t>(binding.index + 1, passes.size() - 1);
                texture = passes[binding.index].texture[OutputSlot(binding.index) ^ 1];
                consumer = &passes[next].config;
                break;
            }
            case TextureSemantic::Texture: {
                const LutTexture& lut = luts[binding.index];
                texture = lut.texture;
                glActiveTexture(GL_TEXTURE0 + binding.unit);
                glBindTexture(GL_TEXTURE_2D, texture);
                glBindSampler(binding.unit, GetSampler(lut.config.linear, lut.config.wrap, lut.config.mipmap));
                return;
            }
            default:
                break;
        }
        glActiveTexture(GL_TEXTURE0 + binding.unit);
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindSampler(binding.unit, GetSampler(consumer->filterLinear, consumer->wrap, mipmap));
    }

    int OutputSlot(size_t passIndex) const {
        return passes[passIndex].feedback ? parity : 0;
    }

    void FillBlock(BlockBinding& block, size_t passIndex, uint32_t frameCount, uint32_t inputW, uint32_t inputH) {
        const Pass& pass = passes[passIndex];
        for (const UniformBinding& u : block.uniforms) {
            switch (u.semantic) {
                case UniformSemantic::MVP:
                    if (u.type == "mat4") {
                        for (int i = 0; i < 16; i++) {
                            WriteFloat(block.data, u.offset + i * 4, i % 5 == 0 ? 1.0f : 0.0f);
                        }
                    }
                    break;
                case UniformSemantic::OutputSize:
                    WriteSize(block.data, u, pass.width, pass.height);
                    break;
                case UniformSemantic::FinalViewportSize:
                    WriteSize(block.data, u, targetWidth, targetHeight);
                    break;
                case UniformSemantic::SourceSize:
                    WriteSize(block.data, u, inputW, inputH);
                    break;
                case UniformSemantic::OriginalSize:
                case UniformSemantic::OriginalHistorySize:
                    WriteSize(block.data, u, sourceWidth, sourceHeight);
                    break;
                case UniformSemantic::FrameCount: {
                    uint32_t count = frameCount;
                    if (pass.config.frameCountMod > 0) {
                        count %= pass.config.frameCountMod;
                    }
                    WriteScalar(block.data, u, count);
                    break;
                }
                case UniformSemantic::FrameDirection:
                    WriteScalar(block.data, u, 1);
                    break;
                case UniformSemantic::FrameTimeDelta:
                    WriteScalar(block.data, u, 16667);
                    break;
                case UniformSemantic::OriginalFPS:
                    WriteScalar(block.data, u, 60.0);
                    break;
                case UniformSemantic::Rotation:
                    WriteScalar(block.data, u, 0);
                    break;
                case UniformSemantic::TotalSubFrames:
                case UniformSemantic::CurrentSubFrame:
                    WriteScalar(block.data, u, 1);
                    break;
                case UniformSemantic::PassOutputSize:
                case UniformSemantic::PassFeedbackSize:
                    WriteSize(block.data, u, passes[u.index].width, passes[u.index].height);
                    break;
                case UniformSemantic::TextureSize:
                    WriteSize(block.data, u, luts[u.index].width, luts[u.index].height);
                    break;
                case UniformSemantic::Parameter: {
                    auto it = paramValues.find(u.parameter);
                    WriteScalar(block.data, u, it != paramValues.end() ? it->second : 0.0f);
                    break;
                }
                default:
                    break;
            }
        }
        glBindBuffer(GL_UNIFORM_BUFFER, block.buffer);
        glBufferData(GL_UNIFORM_BUFFER, block.data.size(), block.data.data(), GL_STREAM_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, block.bindingPoint, block.buffer);
    }
};

SlangFilterChain::SlangFilterChain() : mImpl(std::make_unique<Impl>()) {
}

SlangFilterChain::~SlangFilterChain() = default;

bool SlangFilterChain::Load(const std::string& presetPath, std::string& error) {
    Impl& impl = *mImpl;
    impl.Destroy();

    GLint major = 0, minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
#ifdef USE_OPENGLES
    impl.es = true;
    impl.glslVersion = 300;
#else
    impl.es = false;
    impl.glslVersion = std::min(450, major * 100 + minor * 10);
    if (impl.glslVersion < 330) {
        error = "Shader presets need OpenGL 3.3, this context is " + std::to_string(major) + "." +
                std::to_string(minor);
        return false;
    }
#endif

    PresetMap map;
    Preset preset;
    if (!ReadPresetFile(fs::path(presetPath), map, error, 0) || !BuildPreset(map, preset, error)) {
        return false;
    }

    std::vector<ShaderSource> sources(preset.passes.size());
    impl.passes.resize(preset.passes.size());
    for (size_t i = 0; i < preset.passes.size(); i++) {
        int stage = 0;
        if (!ReadShaderFile(fs::path(preset.passes[i].shaderPath), sources[i], stage, error, 0)) {
            impl.Destroy();
            return false;
        }
        Pass& pass = impl.passes[i];
        pass.config = preset.passes[i];
        pass.alias = !pass.config.alias.empty() ? pass.config.alias : sources[i].name;
        pass.format =
            FormatFromName(sources[i].format, pass.config.floatFramebuffer, pass.config.srgbFramebuffer);
        for (const PostFilterParam& param : sources[i].params) {
            if (!impl.paramValues.count(param.name)) {
                impl.params.push_back(param);
                impl.paramValues[param.name] = param.initial;
            }
        }
    }
    for (const auto& [name, value] : preset.paramOverrides) {
        if (auto it = impl.paramValues.find(name); it != impl.paramValues.end()) {
            it->second = value;
        }
    }
    for (PostFilterParam& param : impl.params) {
        param.initial = impl.paramValues[param.name];
    }

    for (const PresetTexture& texture : preset.textures) {
        LutTexture lut;
        lut.config = texture;
        if (!impl.LoadLut(lut, error)) {
            impl.Destroy();
            return false;
        }
        impl.luts.push_back(lut);
    }

    for (size_t i = 0; i < impl.passes.size(); i++) {
        if (!impl.BuildPass(i, sources[i], error)) {
            impl.Destroy();
            return false;
        }
    }

    // Full screen quad: clip space position and texture coordinate, the texture's row 0 lands on row 0 of the target
    const float vertices[] = { -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 0.0f,
                               -1.0f, 1.0f,  0.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f };
    glGenVertexArrays(1, &impl.vao);
    glGenBuffers(1, &impl.vbo);
    glBindVertexArray(impl.vao);
    glBindBuffer(GL_ARRAY_BUFFER, impl.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (const void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (const void*)(2 * sizeof(float)));
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    SPDLOG_INFO("Loaded shader preset {} ({} passes, {} textures, {} parameters, GLSL {})", presetPath,
                impl.passes.size(), impl.luts.size(), impl.params.size(), impl.glslVersion);
    return true;
}

bool SlangFilterChain::Frame(uint32_t sourceTexture, uint32_t sourceWidth, uint32_t sourceHeight, uint32_t targetFbo,
                             uint32_t targetWidth, uint32_t targetHeight, uint32_t frameCount) {
    Impl& impl = *mImpl;
    if (impl.passes.empty() || sourceWidth == 0 || sourceHeight == 0 || targetWidth == 0 || targetHeight == 0) {
        return false;
    }

    int units = 0;
    for (const Pass& pass : impl.passes) {
        units = std::max(units, pass.maxUnit + 1);
    }
    GLStateBackup backup;
    backup.Save(units);

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glDepthMask(GL_FALSE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    if (sourceWidth != impl.sourceWidth || sourceHeight != impl.sourceHeight || targetWidth != impl.targetWidth ||
        targetHeight != impl.targetHeight) {
        impl.Resize(sourceWidth, sourceHeight, targetWidth, targetHeight);
    }

    // Keep the previous frames for OriginalHistory#
    if (impl.historyDepth > 0) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, impl.sourceCopyFbo);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sourceTexture, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, impl.historyFbos[impl.historyFrame % impl.historyDepth]);
        glBlitFramebuffer(0, 0, sourceWidth, sourceHeight, 0, 0, sourceWidth, sourceHeight, GL_COLOR_BUFFER_BIT,
                          GL_NEAREST);
    }

    glBindVertexArray(impl.vao);
    uint32_t inputW = sourceWidth, inputH = sourceHeight;
    for (size_t i = 0; i < impl.passes.size(); i++) {
        Pass& pass = impl.passes[i];
        const bool last = i + 1 == impl.passes.size();

        if (pass.config.mipmapInput) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D,
                          i == 0 ? sourceTexture : impl.passes[i - 1].texture[impl.OutputSlot(i - 1)]);
            glGenerateMipmap(GL_TEXTURE_2D);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, last ? targetFbo : pass.fbo[impl.OutputSlot(i)]);
        glViewport(0, 0, pass.width, pass.height);
#ifdef GL_FRAMEBUFFER_SRGB
        if (!last && pass.format.srgb) {
            glEnable(GL_FRAMEBUFFER_SRGB);
        } else {
            glDisable(GL_FRAMEBUFFER_SRGB);
        }
#endif
        glUseProgram(pass.program);
        for (BlockBinding& block : pass.blocks) {
            if (block.active) {
                impl.FillBlock(block, i, frameCount, inputW, inputH);
            }
        }
        for (const TextureBinding& binding : pass.textures) {
            impl.BindTexture(binding, i, sourceTexture);
        }
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        inputW = pass.width;
        inputH = pass.height;
    }

    impl.parity ^= 1;
    impl.historyFrame++;
    backup.Restore();
    return true;
}

const std::vector<PostFilterParam>& SlangFilterChain::GetParams() const {
    return mImpl->params;
}

bool SlangFilterChain::GetParam(const std::string& name, float& value) const {
    auto it = mImpl->paramValues.find(name);
    if (it == mImpl->paramValues.end()) {
        return false;
    }
    value = it->second;
    return true;
}

bool SlangFilterChain::SetParam(const std::string& name, float value) {
    auto it = mImpl->paramValues.find(name);
    if (it == mImpl->paramValues.end()) {
        return false;
    }
    it->second = value;
    return true;
}

} // namespace Fast
#endif
