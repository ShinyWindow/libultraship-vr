#define NOMINMAX

#include "fast/vr_openxr.h"
#include "fast/vr_hud_settings.h"
#include "vr_interface.h"

bool g_vr_hud_layout_pass = false;
bool g_vr_rect_world = false;
float g_vr_rect_world_mtx[4][4] = {};

#include <cstring>

// SOH [VR] The OpenXR core is API-neutral: every graphics-API call goes through the VrGfx leaf
// for the running renderer (vr_gfx.h; vr_gfx_d3d11.cpp, ...). ENABLE_VR = OpenXR is linked.
#ifdef ENABLE_VR

#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <memory>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h> // OpenXR environment diagnostics (registry, env); no graphics API here
#endif
#include <openxr/openxr.h>
#ifdef __ANDROID__
// Platform (not graphics-API) glue: the Android loader + instance need the JavaVM and activity.
#include <jni.h>
#include <SDL2/SDL_system.h>
#define XR_USE_PLATFORM_ANDROID
#include <openxr/openxr_platform.h>
#endif

#include <spdlog/spdlog.h>

#include "libultraship/bridge/consolevariablebridge.h"
#include "ship/Context.h"
#include "ship/window/Window.h"
#include "ship/window/MouseStateManager.h"
#include "ship/window/gui/Gui.h"
#include "ship/window/gui/GuiWindow.h"
#include "fast/Fast3dWindow.h"
#include "fast/interpreter.h"
#include "fast/vr_physics.h"
#include "vr_gfx.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>

// --------------------------------------------------------------------------
// Engine glue: the renderer is Fast::Interpreter owned by Fast3dWindow.
// --------------------------------------------------------------------------

Fast::Interpreter* vr_get_interpreter() {
    Ship::Context* ctx = Ship::Context::GetRawInstance();
    if (!ctx) {
        return nullptr;
    }
    auto wnd = std::dynamic_pointer_cast<Fast::Fast3dWindow>(ctx->GetWindow());
    if (!wnd) {
        return nullptr;
    }
    auto interp = wnd->GetInterpreterWeak().lock();
    return interp ? interp.get() : nullptr;
}

static int vr_running_window_backend() {
    Ship::Context* ctx = Ship::Context::GetRawInstance();
    auto wnd = ctx ? std::dynamic_pointer_cast<Fast::Fast3dWindow>(ctx->GetWindow()) : nullptr;
    return wnd ? wnd->GetWindowBackend() : 0;
}

// The interpreter sizes 2D/flat renders from mCurDimensions; point them at the given target so
// rectangles and the viewport fill the actual texture (replaces the old gfx_current_dimensions
// override in gfx_start_frame).
static void vr_apply_dimensions(uint32_t width, uint32_t height) {
    if (Fast::Interpreter* interp = vr_get_interpreter()) {
        interp->mCurDimensions.width = width;
        interp->mCurDimensions.height = height;
        interp->mCurDimensions.aspect_ratio = (float)width / (float)height;
    }
}

// Leave mCurDimensions at the eye size after any 2D pass (HUD quad, flat-screen panel), so it is
// the SAME at every Interpreter::StartFrame regardless of which passes a given frame ran.
// StartFrame reconfigures mGameFb from mCurDimensions unconditionally and re-scales every
// resizable framebuffer whenever it changed since last frame; letting the value alternate between
// eye size and a 2D target's size makes it tear down and reallocate eye-resolution textures
// several times per game tick. Cheap invariant, expensive to get wrong.
static void vr_restore_eye_dimensions();
static bool vr_hud_wrist_layout();
static int vr_hud_profile();
static float vr_hud_panel_f(int child, int hand, const char* field, float def);
// Wrist HUD canvas packing: native HUD px per layout unit (see the wrist layout section).
static constexpr float kWristPack = 0.5f;
static constexpr float kWristCanvasW = 160.0f; // native px per hand canvas
static bool vr_wrist_panel_pose(int hand, XrPosef* pose, float* mpu, float* w_units, float* h_units);
static void vr_hud_grab_update();
static void vr_draw_test_card(vrgfx::Target t, uint32_t image, uint32_t w, uint32_t h);
static void vr_refresh_rate_init();
static bool g_refresh_rate_retried = false; // one more rate query once frames are running

// --------------------------------------------------------------------------
// Internal state
// --------------------------------------------------------------------------

static struct {
    // OpenXR handles
    XrInstance instance;
    XrSystemId system_id;
    XrSession session;
    XrSpace local_space;
    XrSpace stage_space; // floor-level origin, used only for eye-height measurement (auto scale)

    // Auto world scale: scale = Link's eye height (game units, pushed by the game each frame) /
    // the player's physical eye height (meters, measured above the stage floor at calibration).
    // Recalibrated on first-person entry, manual recenter, and whenever Link's eye height changes
    // materially (child <-> adult).
    float link_eye_height_units;
    float auto_world_scale;
    float calibrated_link_eye_height;
    bool scale_calibrated;
    bool scale_recalibrate_requested;
    XrSessionState session_state;
    bool session_running;

    // View configuration
    uint32_t view_count;
    XrView views[2];
    XrViewConfigurationView config_views[2];

    // Graphics leaf for the running renderer: owns every API object (images, targets, depth,
    // mirror copies). The core keeps only the OpenXR handles and sizes.
    std::unique_ptr<vrgfx::Backend> gfx;

    // Swapchains (one per eye)
    struct EyeSwapchain {
        XrSwapchain handle;
        int64_t format;
        uint32_t width, height;
    } eye_swapchains[2];

    // Per-frame
    XrFrameState frame_state;
    bool frame_began;
    int current_eye; // 0/1, or kCenterEye during a multiview pass
    uint32_t refresh_rate;  // Cached headset refresh in Hz, derived from predictedDisplayPeriod
    uint32_t current_image_index[2]; // Acquired swapchain image index per eye

    // Cached per-frame matrices (row-major, row-vector convention). Index 2 (kCenterEye) is the
    // multiview pass's culling camera: see vr_build_center_view.
    float projection[3][4][4];
    float view[3][4][4];

    // Single-pass stereo (multiview, GLES): both eyes in one two-layer swapchain (eye_swapchains[0]
    // holds it; [1] keeps only the size), drawn by ONE interpreter run from the center camera. The
    // GPU moves each vertex from the center camera's clip space into each eye's with mv_eye_mtx
    // (one 4x4 per layer); eye-welded 2D (texrects, fills) uses identity instead (mv_welded).
    bool multiview;      // the session was created with the array swapchain
    bool multiview_pass; // inside vr_begin_stereo / vr_end_stereo
    bool mv_welded;
    uint32_t mv_gen;     // bumped whenever the matrices the renderer should use change
    float mv_eye_mtx[2][16];

    // Configuration
    float world_scale;       // N64 units per meter
    float near_clip;         // In game units
    float far_clip;          // In game units
    float resolution_scale;  // Multiplier on the runtime's recommended per-eye resolution

    // First-person camera
    bool first_person;        // When true, view is anchored to Link's head (game-driven)
    glm::vec3 anchor;         // Link's head this game frame, in game/world units (set by the game)
    glm::vec3 anchor_prev;    // Link's head the previous game frame (for sub-frame interpolation)
    bool anchor_initialized;  // False until the first anchor is pushed
    // Base yaw of the anchored playspace frame, stored as gamma = pi - cameraYaw (the world-to-
    // tracking rotation angle; 0 = world-aligned, which is first-person's frame). Third person
    // pushes the game camera's yaw each tick so facing tracking-forward looks where the game
    // camera looks; pitch/roll are intentionally NOT folded in (the horizon must stay level with
    // real gravity — a tilted horizon is instant motion sickness).
    float anchor_gamma;
    float anchor_gamma_prev;
    float interp_alpha;       // 0..1 blend between anchor_prev and anchor for the current render pass
    int16_t heading_offset;   // binang offset mapping HMD yaw -> game-world yaw (set at recenter)

    // Roomscale 6DOF: accumulated horizontal physical-walk displacement (game units, .x = world x,
    // .y = world z) that has been baked into Link's body position. The game advances it ONLY by the
    // body's collision-limited achieved move, and pushes anchor = bodyHead - roomscale_origin so the
    // eye stays continuous as the body slides under the head. See vr_roomscale_6dof plan.
    glm::vec2 roomscale_origin;

    // Motion controls (OpenXR action sets). hand index: 0 = left, 1 = right.
    XrActionSet action_set;
    XrPath hand_path[2];               // /user/hand/left, /user/hand/right
    XrAction grip_pose_action;         // POSE (per-hand subaction)
    XrAction aim_pose_action;          // POSE (per-hand subaction)
    XrAction trigger_action;           // FLOAT
    XrAction squeeze_action;           // FLOAT (grip)
    XrAction thumbstick_action;        // VECTOR2F
    XrAction thumbstick_click_action;  // BOOL
    XrAction primary_action;           // BOOL: A (right) / X (left)
    XrAction secondary_action;         // BOOL: B (right) / Y (left)
    XrAction menu_action;              // BOOL
    XrAction haptic_action;            // VIBRATION_OUTPUT (per-hand subaction)
    XrSpace grip_space[2];
    XrSpace aim_space[2];
    bool input_initialized;
    // Raw located view poses, preserved for compositor submission. The game-facing poses in `views`
    // get the artificial snap-turn applied; the compositor must instead see the physical head pose
    // the rendered image corresponds to (the turn is a world-space change, not a head-pose change).
    // submit_fov is latched alongside so a resubmitted (not re-rendered) eye image is described by
    // the frustum it was actually drawn with.
    XrPosef submit_pose[2];
    XrFovf submit_fov[2];

    // Frame plan for the current XR frame (see vr_set_frame_plan).
    bool plan_render_eyes;
    bool plan_render_hud;
    bool plan_present_desktop;
    // Per-frame controller state (raw, in OpenXR local space)
    bool hand_active[2];
    bool hand_tracked[2]; // position actually tracked this frame (not just valid: IMU-only drift is valid)
    XrPosef grip_pose[2];
    XrPosef grip_pose_raw[2]; // untouched by snap-turn; for compositor-space quads (hand HUD)
    XrPosef aim_pose[2];
    XrPosef aim_pose_raw[2]; // likewise: the menu laser is cast against a compositor quad
    float trigger_value[2];
    float squeeze_value[2];
    float thumbstick_x[2];
    float thumbstick_y[2];
    uint16_t buttons[2];               // VR_BTN_* bitmask per hand
    // Hand velocities this frame (RAW tracking space) from XrSpaceVelocity chained into the grip
    // locate. hand_vel_valid distinguishes "runtime reported them" from "left at zero" so the
    // physics layer knows when to fall back to finite-differencing.
    XrVector3f hand_lin_vel[2];
    XrVector3f hand_ang_vel[2];
    bool hand_vel_valid[2];

    // HUD overlay
    XrSpace view_space;
    struct EyeSwapchain hud_swapchain;
    uint32_t hud_image_index;
    void* hud_commands;
    bool rendering_hud;

    // Text panel: the message system (dialogue, signs, item text, the ocarina staff) on its own
    // quad that soft-follows in front of the player. Real metres in local_space, independent of
    // world scale and Link's age. text_commands is non-NULL only while a text box is showing.
    struct EyeSwapchain text_swapchain;
    uint32_t text_image_index;
    void* text_commands;
    float text_crop[4];       // u0, v0, u1, v1 of the 320x240 frame: the region the panel shows
    float text_layout[3];     // per-frame override (width m, distance m, height m); <= 0 = the setting
    bool rendering_text;
    bool text_has_image;      // rendered at least once since the text box opened
    bool text_was_visible;    // last frame's submit decision: a hidden -> shown edge snaps the panel
    glm::vec3 text_pos;       // panel centre, RAW local_space metres
    glm::vec3 text_vel;       // critically damped follow velocity
    glm::vec3 text_fwd;       // last valid level head-forward (fallback when looking straight up/down)
    bool text_following;
    // Wrist HUD layout. Per element (native 320x240 px): the extent it drew at on the TV layout this
    // pass, its held extent (grows at once, shrinks only after a smaller one has held for a while,
    // so a beating heart or a digit change never makes the layout twitch), its vanilla fade, and the
    // offset that moves it into its slot on its hand's canvas (computed from the held extents at
    // the end of each pass, applied on the next). Per hand: the laid-out block (the canvas region
    // its quad shows, backing included) and the block's fade.
    int hud_el;
    float hud_meas[VR_HUD_EL_COUNT][4];
    bool hud_meas_any[VR_HUD_EL_COUNT];
    uint8_t hud_meas_alpha[VR_HUD_EL_COUNT];
    float hud_rect[VR_HUD_EL_COUNT][4];
    bool hud_rect_valid[VR_HUD_EL_COUNT];
    int hud_shrink_passes[VR_HUD_EL_COUNT];
    int hud_absent_passes[VR_HUD_EL_COUNT];
    float hud_alpha[VR_HUD_EL_COUNT]; // 0..1, latched at the end of each HUD pass
    float hud_slot[VR_HUD_EL_COUNT][3]; // slot top-left (native px) + draw scale
    bool hud_placed[VR_HUD_EL_COUNT];
    bool hud_child; // Link's age, pushed by the game: picks the Adult / Child settings profile
    // In-headset HUD editing (gVrHud.GrabEdit): one hand grabs the OTHER wrist's panel (A/X +
    // grip: the whole panel, rigidly) or one element on it (B/Y + grip: dragged across the
    // panel's face). The edits are written live into the active profile's settings.
    struct {
        int grabber;       // hand holding something, -1 = none
        int owner;         // the panel's hand
        int element;       // VR_HUD_EL_* being dragged, -1 = the whole panel
        glm::vec3 rel_pos; // panel pose in the grabber's grip frame (panel grab)
        glm::quat rel_rot;
        XrPosef frame;     // panel pose frozen at grab start (element drag)
        float mpu;
        glm::vec2 start_local;
        float start_off[2];
        int hover[2];      // per panel: 0 none, 1 a hand is in reach, 2 grabbed
        bool swallow[2];   // hide this hand's buttons/grip/trigger from the game
    } grab;
    float hud_block[2][4]; // per hand canvas block, native px (x0, y0, x1, y1)
    bool hud_block_valid[2];
    float hud_block_alpha[2];
    // World-anchored HUD (world-space pause): game-world centre/yaw/size, converted per frame.
    bool hud_world;
    glm::vec3 hud_world_center;
    float hud_world_yaw;
    float hud_world_size[2];
    bool hud_world_pass; // this HUD pass draws for the world panel (no backings, no wrist latch)
    // Pause-frame canvas (vr_hud_world_canvas: width x, height x, element size x), latched per pass,
    // and each element's corner on it (x -1 left / +1 right, y +1 top / -1 bottom), chosen from
    // where it sat on the TV frame in the previous pass.
    float hud_world_canvas[3];
    float hud_world_meas[VR_HUD_EL_COUNT][4];
    bool hud_world_meas_any[VR_HUD_EL_COUNT];
    float hud_world_anchor[VR_HUD_EL_COUNT][2];
    bool hud_world_anchor_valid[VR_HUD_EL_COUNT];

    // Centre-eye pose this frame, RAW tracking space (no artificial turn): what the head really did.
    glm::vec3 head_pos_raw;
    glm::quat head_rot_raw;
    bool head_raw_valid;

    // Flat-screen mode: 2D contexts (file select, pause menu) render the whole frame onto a
    // world-locked floating panel instead of the stereo eyes. The last-rendered world frame keeps
    // being submitted behind it with its original pose, so it stays frozen-but-head-tracked.
    bool flat_screen;
    bool flat_screen_prev;
    XrPosef flat_pose; // panel pose in local_space (RAW tracking coords — quads bypass the snap-turn)
    struct EyeSwapchain screen_swapchain;
    uint32_t screen_image_index;
    bool rendering_screen;   // currently rendering into the screen swapchain (vs the HUD's)
    bool eyes_ever_rendered;   // don't submit the projection layer before its swapchains have content
    bool hud_ever_rendered;    // likewise for the HUD quad — its swapchain starts uninitialised
    bool screen_ever_rendered; // and for the flat-screen panel

    // SoH (ImGui) menu panel and its laser pointer (see "SoH menu panel" below). The panel image is
    // drawn by the running ImGui renderer backend (Fast3dGui) between vr_begin_menu/vr_end_menu;
    // the beam strip is a static gradient filled once at session start.
    struct EyeSwapchain menu_swapchain;
    uint32_t menu_image_index;
    bool menu_ever_rendered;
    struct EyeSwapchain beam_swapchain;
    bool beam_ready;

    // Desktop mirror: copies of the left eye and of the headset's QUAD layers (HUD / wrist panels,
    // text panel, flat-screen panel), owned by the graphics leaf (vrgfx mirror slots). A swapchain
    // image can't be read after release, so each pass copies its image while still acquired. The
    // quad layers are composited by the runtime, so the eye mirror never contains them; the
    // companion window redraws each one through the left eye instead (vr_get_mirror_quads).
    // The quads submitted this frame (pose in RAW local space, as the compositor got them).
    struct MirrorQuad {
        int src; // 0 HUD, 1 text, 2 flat-screen panel
        XrPosef pose;
        XrExtent2Df size;
        XrRect2Di rect;
    } mirror_quads[8];
    int mirror_quad_count;

    bool initialized;
    // Runtime VR<->flat toggle. `initialized` means the OpenXR session exists; `enabled` means the
    // mod is actively driving it. Disabled-with-session = flat gameplay with the headset idle,
    // ready to resume instantly. All game-facing predicates check both.
    bool enabled;
    bool reenable_fixup; // one-shot: re-zero roomscale on the first located frame after re-enable

    // Headset presence (XR_EXT_user_presence proximity-sensor events). When the extension is
    // unavailable the feature is inert: user_present stays true and doffing changes nothing.
    bool user_presence_supported;
    bool user_present;

    // Alyx-style in-wall view fade: when the player's physical head is inside geometry, the WORLD
    // layer fades toward black at the compositor (XR_KHR_composition_layer_color_scale_bias) —
    // the head is never pushed back. Target set by the game per tick; smoothed per XR frame.
    bool color_scale_supported;
    // XR_EXT_performance_settings (Meta's runtimes): CPU clock hint, gVrHighCpuClock.
    bool perf_settings_supported;
    int perf_cpu_level_applied; // last XrPerfSettingsLevelEXT sent, -1 = none this session
    // XR_FB_display_refresh_rate (Meta's runtimes): the rates this headset offers, and the request
    // last sent for gVrRefreshRate (Hz, 0 = the headset's default). -1 = none this session.
    bool refresh_rate_supported;
    std::vector<float> refresh_rates;
    float refresh_rate_applied;
    float view_fade_target;
    float view_fade_current;

    // Connection lifecycle (vr_apply_mode_request). A session the runtime lost or ended is only
    // flagged where it is noticed and torn down at the next tick boundary, where no display list or
    // XR frame is in flight; a lost one then reconnects, an ended one turns VR off. A vr_init failure
    // worth retrying (headset not detected yet) is retried for a while instead of turning VR off.
    bool session_lost;       // LOSS_PENDING, instance loss, or a call returned *_LOST
    bool session_exited;     // EXITING: the runtime ended the session (e.g. the user quit it)
    bool init_retryable;     // the last vr_init failure may succeed later; the instance is kept
    // Some runtimes / streaming bridges report "not worn" for a headset that is (proximity sensor
    // disabled or not forwarded). "Not worn" only drops VR once "worn" has been seen this session.
    bool presence_confirmed;
} xr = {};

static constexpr int kCenterEye = 2;

static void vr_restore_eye_dimensions() {
    const auto& sc = xr.eye_swapchains[0];
    if (sc.width > 0 && sc.height > 0) {
        vr_apply_dimensions(sc.width, sc.height);
    }
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

static bool xr_check(XrResult result, const char* msg) {
    if (XR_SUCCEEDED(result)) return true;
    if (result == XR_ERROR_SESSION_LOST || result == XR_ERROR_INSTANCE_LOST) {
        xr.session_lost = true;
    }
    if (xr.instance != XR_NULL_HANDLE) {
        char buf[XR_MAX_RESULT_STRING_SIZE];
        xrResultToString(xr.instance, result, buf);
        spdlog::error("[VR] {} failed: {}", msg, buf);
    } else {
        spdlog::error("[VR] {} failed: XrResult {}", msg, static_cast<int>(result));
    }
    return false;
}

// Build an asymmetric projection matrix from XrFovf.
// Build asymmetric projection from XrFovf.
// Output is row-major for the engine's row-vector convention (clip = v * P).
// This is the TRANSPOSE of the standard column-vector OpenGL projection.
static void build_projection_matrix(const XrFovf& fov, float near_z, float far_z, float out[4][4]) {
    float left = tanf(fov.angleLeft);
    float right = tanf(fov.angleRight);
    float up = tanf(fov.angleUp);
    float down = tanf(fov.angleDown);

    float width = right - left;
    float height = up - down;
    float depth = far_z - near_z;

    memset(out, 0, sizeof(float) * 16);

    // Row-vector convention (transposed from column-vector):
    out[0][0] = 2.0f / width;
    out[1][1] = 2.0f / height;
    out[2][0] = (right + left) / width;
    out[2][1] = (up + down) / height;
    out[2][2] = -(far_z + near_z) / depth;
    out[2][3] = -1.0f;
    out[3][2] = -(2.0f * far_z * near_z) / depth;
}

// Convert XrPosef to a view matrix (inverse of the pose).
// Applies world_scale to translation.
// Output is row-major for row-vector convention.
static void pose_to_view_matrix(const XrPosef& pose, float world_scale, float out[4][4]) {
    glm::quat q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    glm::mat4 rotation = glm::mat4_cast(q);
    glm::vec3 pos(pose.position.x * world_scale, pose.position.y * world_scale, pose.position.z * world_scale);

    glm::mat4 transform = glm::translate(glm::mat4(1.0f), pos) * rotation;
    glm::mat4 view = glm::inverse(transform);

    // GLM is column-major and column-vector (v' = M * v).
    // Engine uses row-vector (v' = v * M^T), so we need the mathematical transpose.
    // GLM: view[col][row], so view[r][c] = element(c, r) = transposed element(r, c).
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = view[r][c];
}

// --------------------------------------------------------------------------
// Single-pass stereo: the center camera and the per-layer transforms
// --------------------------------------------------------------------------

static void mat4_from_float(double out[4][4], const float in[4][4]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = in[r][c];
}

static void mat4_mul_d(double res[4][4], const double a[4][4], const double b[4][4]) {
    double tmp[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            tmp[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
    memcpy(res, tmp, sizeof(tmp));
}

// Gauss-Jordan with partial pivoting. False = singular.
static bool mat4_inverse_d(const double m[4][4], double out[4][4]) {
    double a[4][8];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            a[r][c] = m[r][c];
            a[r][c + 4] = (r == c) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < 4; col++) {
        int pivot = col;
        for (int r = col + 1; r < 4; r++) {
            if (fabs(a[r][col]) > fabs(a[pivot][col])) pivot = r;
        }
        if (fabs(a[pivot][col]) < 1e-12) return false;
        if (pivot != col) {
            for (int c = 0; c < 8; c++) std::swap(a[col][c], a[pivot][c]);
        }
        const double inv = 1.0 / a[col][col];
        for (int c = 0; c < 8; c++) a[col][c] *= inv;
        for (int r = 0; r < 4; r++) {
            if (r == col) continue;
            const double f = a[r][col];
            if (f == 0.0) continue;
            for (int c = 0; c < 8; c++) a[r][c] -= f * a[col][c];
        }
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = a[r][c + 4];
    return true;
}

// Per-layer transforms for the multiview programs: clip_eye = clip_center * inv(VP_c) * VP_e. The
// anchored view the interpreter uses is A * view for every camera with the same A (vr_get_view_
// matrix), so inv(A v_c P_c) A v_e P_e = inv(P_c) inv(v_c) v_e P_e: the anchor (big game-unit
// translations) cancels, and the raw per-frame matrices give the exact answer. Stored row-major,
// uploaded untransposed, which is what GLSL's  M * v  needs for the engine's row-vector  v * C.
static void vr_update_multiview_matrices() {
    double vpc[4][4], v[4][4], p[4][4], inv[4][4];
    mat4_from_float(v, xr.view[kCenterEye]);
    mat4_from_float(p, xr.projection[kCenterEye]);
    mat4_mul_d(vpc, v, p);
    const bool ok = mat4_inverse_d(vpc, inv);
    for (int eye = 0; eye < 2; eye++) {
        double vpe[4][4], c[4][4];
        mat4_from_float(v, xr.view[eye]);
        mat4_from_float(p, xr.projection[eye]);
        mat4_mul_d(vpe, v, p);
        if (ok) {
            mat4_mul_d(c, inv, vpe);
        }
        for (int r = 0; r < 4; r++)
            for (int col = 0; col < 4; col++)
                xr.mv_eye_mtx[eye][r * 4 + col] = ok ? (float)c[r][col] : (r == col ? 1.0f : 0.0f);
    }
    xr.mv_gen++;
}

// The multiview pass runs the display list ONCE, so the interpreter's CPU-side clip rejection,
// backface culling and fog need one camera whose view contains everything either eye sees: the
// center camera. Orientation = the eyes' average; field of view = the union of both eyes' edge
// directions in that frame; apex pulled back along its forward axis until both eye positions are
// inside. A frustum is a convex cone, so containing an eye's apex and all of its edge directions
// means containing its whole frustum: nothing either eye can see is rejected. (The pull-back is
// about half the IPD over the tangent of the outer half-angle: ~3 cm on a Quest 3.) Backface
// culling decides from this viewpoint for both eyes; a triangle within a degree or two of edge-on
// can disagree with one eye, which at that angle covers next to no pixels.
static void vr_build_center_view() {
    if (!xr.multiview) {
        return;
    }
    glm::quat q[2];
    glm::vec3 e[2];
    for (int i = 0; i < 2; i++) {
        const XrPosef& pose = xr.views[i].pose;
        q[i] = glm::quat(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        e[i] = glm::vec3(pose.position.x, pose.position.y, pose.position.z);
    }
    if (glm::dot(q[0], q[1]) < 0.0f) {
        q[1] = -q[1];
    }
    glm::quat qc = q[0] + q[1];
    const float qlen = glm::length(qc);
    if (qlen < 1e-6f) {
        return;
    }
    qc *= (1.0f / qlen);
    const glm::mat3 R = glm::mat3_cast(qc);
    const glm::mat3 Rt = glm::transpose(R);
    const glm::vec3 mid = 0.5f * (e[0] + e[1]);

    float minX = -1e-3f, maxX = 1e-3f, minY = -1e-3f, maxY = 1e-3f;
    for (int i = 0; i < 2; i++) {
        const XrFovf& f = xr.views[i].fov;
        const glm::mat3 Re = glm::mat3_cast(q[i]);
        const float tx[2] = { tanf(f.angleLeft), tanf(f.angleRight) };
        const float ty[2] = { tanf(f.angleDown), tanf(f.angleUp) };
        for (float x : tx) {
            for (float y : ty) {
                const glm::vec3 d = Rt * (Re * glm::vec3(x, y, -1.0f));
                if (d.z > -1e-4f) {
                    continue; // an edge at or past 90 degrees off the center axis: not a real headset
                }
                minX = std::min(minX, d.x / -d.z);
                maxX = std::max(maxX, d.x / -d.z);
                minY = std::min(minY, d.y / -d.z);
                maxY = std::max(maxY, d.y / -d.z);
            }
        }
    }

    // Apex at local (0, 0, back) (forward is -Z). An eye at local p sits inside when its tangents
    // from the apex, p.xy / (back - p.z), are within the union.
    float back = 0.0f;
    for (int i = 0; i < 2; i++) {
        const glm::vec3 p = Rt * (e[i] - mid);
        auto need = [](float c, float lo, float hi) { return c > 0.0f ? c / hi : (c < 0.0f ? c / lo : 0.0f); };
        back = std::max(back, p.z + std::max(need(p.x, minX, maxX), need(p.y, minY, maxY)));
    }
    back = back * 1.02f + 0.001f;

    const glm::vec3 pc = mid + R * glm::vec3(0.0f, 0.0f, back);
    XrPosef center;
    center.position = { pc.x, pc.y, pc.z };
    center.orientation = { qc.x, qc.y, qc.z, qc.w };
    XrFovf fov;
    fov.angleLeft = atanf(minX);
    fov.angleRight = atanf(maxX);
    fov.angleDown = atanf(minY);
    fov.angleUp = atanf(maxY);
    pose_to_view_matrix(center, xr.world_scale, xr.view[kCenterEye]);
    // The far plane moves back with the apex, so the eyes' far planes stay inside it.
    build_projection_matrix(fov, xr.near_clip, xr.far_clip + 2.0f * back * xr.world_scale,
                            xr.projection[kCenterEye]);
    vr_update_multiview_matrices();
}

// --------------------------------------------------------------------------
// OpenXR session state event handling
// --------------------------------------------------------------------------

static const char* vr_session_state_name(XrSessionState s) {
    switch (s) {
        case XR_SESSION_STATE_IDLE: return "IDLE";
        case XR_SESSION_STATE_READY: return "READY";
        case XR_SESSION_STATE_SYNCHRONIZED: return "SYNCHRONIZED";
        case XR_SESSION_STATE_VISIBLE: return "VISIBLE";
        case XR_SESSION_STATE_FOCUSED: return "FOCUSED";
        case XR_SESSION_STATE_STOPPING: return "STOPPING";
        case XR_SESSION_STATE_LOSS_PENDING: return "LOSS_PENDING";
        case XR_SESSION_STATE_EXITING: return "EXITING";
        default: return "UNKNOWN";
    }
}

static void handle_session_state_change(XrSessionState new_state) {
    // Logged in full: a session stuck short of VISIBLE (or bouncing out of it) is the first thing to
    // read in a "the game runs but the headset shows nothing" report.
    spdlog::info("[VR] Session state: {} -> {}", vr_session_state_name(xr.session_state),
                 vr_session_state_name(new_state));
    xr.session_state = new_state;

    switch (new_state) {
        case XR_SESSION_STATE_READY: {
            xr.presence_confirmed = false;
            XrSessionBeginInfo begin_info = { XR_TYPE_SESSION_BEGIN_INFO };
            begin_info.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
            if (xr_check(xrBeginSession(xr.session, &begin_info), "xrBeginSession")) {
                xr.session_running = true;
                spdlog::info("[VR] Session started");
            }
            break;
        }
        case XR_SESSION_STATE_STOPPING: {
            xr.session_running = false;
            xr_check(xrEndSession(xr.session), "xrEndSession");
            spdlog::info("[VR] Session stopped");
            break;
        }
        case XR_SESSION_STATE_LOSS_PENDING:
        case XR_SESSION_STATE_EXITING:
            xr.session_running = false;
            if (new_state == XR_SESSION_STATE_LOSS_PENDING) {
                xr.session_lost = true;
            } else {
                xr.session_exited = true;
            }
            // Torn down at the next tick boundary (vr_apply_mode_request); frames stop meanwhile.
            break;
        default:
            break;
    }
}

static void vr_reset_snap_turn(); // defined with the snap-turn state below

// Per-hand thumbstick suppression for modal hand gestures (Alyx-style item selector) —
// applied at the source in update_input, so every stick consumer inherits it.
static bool g_stick_suppressed[2] = { false, false };
// A game-side menu that owns the right stick while rendering in stereo (the world-space pause
// menu: the stick is the C-stick there) turns artificial turning off without touching the stick.
static bool g_turn_suppressed = false;

// Physical climbing view lock (vr_set_climb_view_lock). The game moves Link's body opposite to the
// gripping hand once per 20 Hz tick; interpolating the anchor between those ticks would leave the
// view a tick behind the arm (the world "rubber-bands" behind every pull). While locked, the anchor
// is the game's CURRENT one plus the hand's motion since the tick sampled it (ref), reversed and
// restricted to the directions the game moves the body (in the wall plane; vertical only on a
// ladder): the gripping hand stays exactly where it took hold and the view follows the arm at
// headset rate. At the next tick the game's body move absorbs that motion and ref re-bases.
static int g_climb_hand = -1;
static glm::vec3 g_climb_ref(0.0f);
static glm::vec3 g_climb_normal(0.0f);
static bool g_climb_lateral = false;
// Letting go: the locked view runs in real time, the interpolated anchor a 20 Hz tick behind, so
// switching straight back would hop the view. Instead the view blends from where it was at the
// release to the interpolated anchor over kClimbHandoffSec (smoothstep).
static bool g_climb_handoff_active = false;
static glm::vec3 g_climb_handoff_anchor(0.0f);
static std::chrono::steady_clock::time_point g_climb_handoff_t0;
static constexpr float kClimbHandoffSec = 0.12f;

// Limits from the game (vr_set_climb_view_limits): how far the body can still go this tick along
// the wall (t), up (y) and out from it (n) — the view never runs past what the next tick's
// collision will allow (no overshoot-and-snap-back against the top, an edge, the floor, a ceiling
// or the closest the view may come to the wall).
static bool g_climb_lim_valid = false;
static glm::vec3 g_climb_lim_out(0.0f); // n: horizontal, out of the wall
static glm::vec3 g_climb_lim_lo(0.0f);  // allowed travel in -t, -y, -n (>= 0)
static glm::vec3 g_climb_lim_hi(0.0f);  // allowed travel in +t, +y, +n (>= 0)
// Tracking guards: the gripping hand losing positional tracking (IMU-only drift is still "valid"),
// jumping (tracking reacquired, a glitch) or a system recenter must not move the view. The view then
// holds its last offset until the game re-bases (it reads vr_consume_climb_discontinuity).
static bool g_climb_frozen = false;
static bool g_climb_discontinuity = false;
static glm::vec3 g_climb_last_body(0.0f);
static glm::vec3 g_climb_prev_live(0.0f);
static bool g_climb_prev_valid = false;
static constexpr float kClimbJumpMeters = 0.15f; // in one headset frame: never a real hand

static glm::vec3 vr_climb_locked_anchor() {
    const XrVector3f& p = xr.grip_pose[g_climb_hand].position;
    const glm::vec3 live = glm::vec3(p.x, p.y, p.z) * xr.world_scale;
    const bool tracked = xr.hand_tracked[g_climb_hand];
    if (!tracked) {
        g_climb_frozen = true;
        g_climb_discontinuity = true;
    } else if (g_climb_prev_valid) {
        const glm::vec3 jump = live - g_climb_prev_live;
        const float lim = kClimbJumpMeters * xr.world_scale;
        if (glm::dot(jump, jump) > lim * lim) {
            g_climb_frozen = true;
            g_climb_discontinuity = true;
        }
    }
    g_climb_prev_live = live;
    g_climb_prev_valid = tracked;
    if (g_climb_frozen) {
        return xr.anchor + g_climb_last_body;
    }

    glm::vec3 d = live - g_climb_ref;
    d -= g_climb_normal * glm::dot(d, g_climb_normal);
    if (!g_climb_lateral) {
        d.x = d.z = 0.0f;
    }
    glm::vec3 body = -d;
    if (g_climb_lim_valid) {
        const glm::vec3 n = g_climb_lim_out;
        const glm::vec3 t(-n.z, 0.0f, n.x);
        const float bt = glm::clamp(glm::dot(body, t), -g_climb_lim_lo.x, g_climb_lim_hi.x);
        const float by = glm::clamp(body.y, -g_climb_lim_lo.y, g_climb_lim_hi.y);
        const float bn = glm::clamp(glm::dot(body, n), -g_climb_lim_lo.z, g_climb_lim_hi.z);
        body = t * bt + glm::vec3(0.0f, by, 0.0f) + n * bn;
    }
    g_climb_last_body = body;
    return xr.anchor + body;
}

// The anchor every game-facing composition uses this render pass (camera, hands, quads, sim).
static glm::vec3 vr_anchor_now() {
    if (g_climb_hand >= 0 && xr.first_person && xr.hand_active[g_climb_hand]) {
        return vr_climb_locked_anchor();
    }
    const glm::vec3 base = glm::mix(xr.anchor_prev, xr.anchor, xr.interp_alpha);
    if (g_climb_handoff_active) {
        const float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - g_climb_handoff_t0).count();
        const glm::vec3 gap = base - g_climb_handoff_anchor;
        if (t >= kClimbHandoffSec || !xr.first_person || glm::dot(gap, gap) > 200.0f * 200.0f) {
            g_climb_handoff_active = false;
        } else {
            const float k = t / kClimbHandoffSec;
            return glm::mix(g_climb_handoff_anchor, base, k * k * (3.0f - 2.0f * k));
        }
    }
    return base;
}

static void poll_events() {
    XrEventDataBuffer event = { XR_TYPE_EVENT_DATA_BUFFER };
    while (xrPollEvent(xr.instance, &event) == XR_SUCCESS) {
        if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            auto* state_event = reinterpret_cast<XrEventDataSessionStateChanged*>(&event);
            handle_session_state_change(state_event->state);
        } else if (event.type == XR_TYPE_EVENT_DATA_USER_PRESENCE_CHANGED_EXT) {
            // Proximity sensor: headset donned/doffed. The runtime also posts the initial state
            // right after the session starts.
            auto* presence_event = reinterpret_cast<XrEventDataUserPresenceChangedEXT*>(&event);
            xr.user_present = (presence_event->isUserPresent == XR_TRUE);
            if (xr.user_present) {
                xr.presence_confirmed = true;
                spdlog::info("[VR] Headset donned");
            } else if (xr.presence_confirmed) {
                spdlog::info("[VR] Headset doffed");
            } else {
                spdlog::info("[VR] Runtime reports the headset not worn before ever reporting it worn; "
                             "staying in VR (proximity sensor may be disabled or not forwarded)");
            }
        } else if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
            spdlog::warn("[VR] OpenXR instance loss pending (runtime shutting down or restarting)");
            xr.session_lost = true;
        } else if (event.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) {
            // The user triggered the runtime's built-in recenter: LOCAL space reorients so their
            // CURRENT physical facing becomes the new neutral. Because our view frames compose
            // "base yaw + head offset from neutral", this inherently realigns the player — in
            // third person, neutral = the chase camera's facing, so after a recenter they are
            // looking at Link from directly behind the camera again. We just have to drop every
            // piece of state expressed in the OLD space coordinates.
            auto* space_event = reinterpret_cast<XrEventDataReferenceSpaceChangePending*>(&event);
            if (space_event->referenceSpaceType == XR_REFERENCE_SPACE_TYPE_LOCAL) {
                spdlog::info("[VR] System recenter — realigning playspace");
                vr_reset_snap_turn();                    // accumulated turn was in old-space coords
                vr_reset_roomscale();                    // old-space origin would read as a huge lean
                if (g_climb_hand >= 0) {                 // a climbing hand's ref is in the old space
                    g_climb_frozen = true;
                    g_climb_discontinuity = true;
                }
                xr.scale_recalibrate_requested = true;   // player is standing normally right now
                xr.flat_screen_prev = false;             // re-place the menu panel in the new space
            }
        }
        event = { XR_TYPE_EVENT_DATA_BUFFER };
    }
}

// --------------------------------------------------------------------------
// Motion controls: OpenXR action-set setup + per-frame sync
// --------------------------------------------------------------------------

// Create the gameplay action set, controller pose + input actions, suggest bindings for the common
// runtimes, attach to the session, and create per-hand pose spaces. Called once during vr_init after
// the reference space exists. Optional: on failure motion controls are disabled but the HMD works.
static bool setup_input() {
    XrActionSetCreateInfo set_ci = { XR_TYPE_ACTION_SET_CREATE_INFO };
    strcpy(set_ci.actionSetName, "gameplay");
    strcpy(set_ci.localizedActionSetName, "Gameplay");
    if (!xr_check(xrCreateActionSet(xr.instance, &set_ci, &xr.action_set), "xrCreateActionSet")) {
        return false;
    }

    xrStringToPath(xr.instance, "/user/hand/left", &xr.hand_path[0]);
    xrStringToPath(xr.instance, "/user/hand/right", &xr.hand_path[1]);

    auto make_action = [&](const char* name, const char* localized, XrActionType type, XrAction* out) -> bool {
        XrActionCreateInfo ci = { XR_TYPE_ACTION_CREATE_INFO };
        strcpy(ci.actionName, name);
        strcpy(ci.localizedActionName, localized);
        ci.actionType = type;
        ci.countSubactionPaths = 2;
        ci.subactionPaths = xr.hand_path;
        return xr_check(xrCreateAction(xr.action_set, &ci, out), "xrCreateAction");
    };

    bool ok = true;
    ok &= make_action("grip_pose", "Grip Pose", XR_ACTION_TYPE_POSE_INPUT, &xr.grip_pose_action);
    ok &= make_action("aim_pose", "Aim Pose", XR_ACTION_TYPE_POSE_INPUT, &xr.aim_pose_action);
    ok &= make_action("trigger", "Trigger", XR_ACTION_TYPE_FLOAT_INPUT, &xr.trigger_action);
    ok &= make_action("squeeze", "Squeeze", XR_ACTION_TYPE_FLOAT_INPUT, &xr.squeeze_action);
    ok &= make_action("thumbstick", "Thumbstick", XR_ACTION_TYPE_VECTOR2F_INPUT, &xr.thumbstick_action);
    ok &= make_action("thumbstick_click", "Thumbstick Click", XR_ACTION_TYPE_BOOLEAN_INPUT,
                      &xr.thumbstick_click_action);
    ok &= make_action("primary", "Primary Button", XR_ACTION_TYPE_BOOLEAN_INPUT, &xr.primary_action);
    ok &= make_action("secondary", "Secondary Button", XR_ACTION_TYPE_BOOLEAN_INPUT, &xr.secondary_action);
    ok &= make_action("menu", "Menu", XR_ACTION_TYPE_BOOLEAN_INPUT, &xr.menu_action);
    ok &= make_action("haptic", "Haptic Feedback", XR_ACTION_TYPE_VIBRATION_OUTPUT, &xr.haptic_action);
    if (!ok) return false;

    auto path = [&](const char* s) -> XrPath {
        XrPath p = XR_NULL_PATH;
        xrStringToPath(xr.instance, s, &p);
        return p;
    };
    auto suggest = [&](const char* profile, std::vector<XrActionSuggestedBinding> binds) {
        XrInteractionProfileSuggestedBinding sb = { XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING };
        sb.interactionProfile = path(profile);
        sb.suggestedBindings = binds.data();
        sb.countSuggestedBindings = static_cast<uint32_t>(binds.size());
        xr_check(xrSuggestInteractionProfileBindings(xr.instance, &sb), "xrSuggestInteractionProfileBindings");
    };

    // Oculus Touch (Quest / Rift) — the most common.
    suggest("/interaction_profiles/oculus/touch_controller",
            { { xr.grip_pose_action, path("/user/hand/left/input/grip/pose") },
              { xr.grip_pose_action, path("/user/hand/right/input/grip/pose") },
              { xr.aim_pose_action, path("/user/hand/left/input/aim/pose") },
              { xr.aim_pose_action, path("/user/hand/right/input/aim/pose") },
              { xr.trigger_action, path("/user/hand/left/input/trigger/value") },
              { xr.trigger_action, path("/user/hand/right/input/trigger/value") },
              { xr.squeeze_action, path("/user/hand/left/input/squeeze/value") },
              { xr.squeeze_action, path("/user/hand/right/input/squeeze/value") },
              { xr.thumbstick_action, path("/user/hand/left/input/thumbstick") },
              { xr.thumbstick_action, path("/user/hand/right/input/thumbstick") },
              { xr.thumbstick_click_action, path("/user/hand/left/input/thumbstick/click") },
              { xr.thumbstick_click_action, path("/user/hand/right/input/thumbstick/click") },
              { xr.primary_action, path("/user/hand/left/input/x/click") },
              { xr.primary_action, path("/user/hand/right/input/a/click") },
              { xr.secondary_action, path("/user/hand/left/input/y/click") },
              { xr.secondary_action, path("/user/hand/right/input/b/click") },
              { xr.menu_action, path("/user/hand/left/input/menu/click") },
              { xr.haptic_action, path("/user/hand/left/output/haptic") },
              { xr.haptic_action, path("/user/hand/right/output/haptic") } });

    // Valve Index.
    suggest("/interaction_profiles/valve/index_controller",
            { { xr.grip_pose_action, path("/user/hand/left/input/grip/pose") },
              { xr.grip_pose_action, path("/user/hand/right/input/grip/pose") },
              { xr.aim_pose_action, path("/user/hand/left/input/aim/pose") },
              { xr.aim_pose_action, path("/user/hand/right/input/aim/pose") },
              { xr.trigger_action, path("/user/hand/left/input/trigger/value") },
              { xr.trigger_action, path("/user/hand/right/input/trigger/value") },
              { xr.squeeze_action, path("/user/hand/left/input/squeeze/value") },
              { xr.squeeze_action, path("/user/hand/right/input/squeeze/value") },
              { xr.thumbstick_action, path("/user/hand/left/input/thumbstick") },
              { xr.thumbstick_action, path("/user/hand/right/input/thumbstick") },
              { xr.thumbstick_click_action, path("/user/hand/left/input/thumbstick/click") },
              { xr.thumbstick_click_action, path("/user/hand/right/input/thumbstick/click") },
              { xr.primary_action, path("/user/hand/left/input/a/click") },
              { xr.primary_action, path("/user/hand/right/input/a/click") },
              { xr.secondary_action, path("/user/hand/left/input/b/click") },
              { xr.secondary_action, path("/user/hand/right/input/b/click") },
              { xr.haptic_action, path("/user/hand/left/output/haptic") },
              { xr.haptic_action, path("/user/hand/right/output/haptic") } });

    // KHR simple controller — universal fallback (pose + select + menu only).
    suggest("/interaction_profiles/khr/simple_controller",
            { { xr.grip_pose_action, path("/user/hand/left/input/grip/pose") },
              { xr.grip_pose_action, path("/user/hand/right/input/grip/pose") },
              { xr.aim_pose_action, path("/user/hand/left/input/aim/pose") },
              { xr.aim_pose_action, path("/user/hand/right/input/aim/pose") },
              { xr.primary_action, path("/user/hand/left/input/select/click") },
              { xr.primary_action, path("/user/hand/right/input/select/click") },
              { xr.menu_action, path("/user/hand/left/input/menu/click") },
              { xr.menu_action, path("/user/hand/right/input/menu/click") },
              { xr.haptic_action, path("/user/hand/left/output/haptic") },
              { xr.haptic_action, path("/user/hand/right/output/haptic") } });

    XrSessionActionSetsAttachInfo attach = { XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO };
    attach.countActionSets = 1;
    attach.actionSets = &xr.action_set;
    if (!xr_check(xrAttachSessionActionSets(xr.session, &attach), "xrAttachSessionActionSets")) {
        return false;
    }

    for (int h = 0; h < 2; h++) {
        XrActionSpaceCreateInfo as_ci = { XR_TYPE_ACTION_SPACE_CREATE_INFO };
        as_ci.poseInActionSpace = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
        as_ci.subactionPath = xr.hand_path[h];
        as_ci.action = xr.grip_pose_action;
        xr_check(xrCreateActionSpace(xr.session, &as_ci, &xr.grip_space[h]), "xrCreateActionSpace (grip)");
        as_ci.action = xr.aim_pose_action;
        xr_check(xrCreateActionSpace(xr.session, &as_ci, &xr.aim_space[h]), "xrCreateActionSpace (aim)");
    }

    xr.input_initialized = true;
    spdlog::info("[VR] Motion-control input initialized");
    return true;
}

// Sync controller actions and locate the hand poses each frame. Called from vr_begin_frame after the
// views are located, with the same predicted display time. Safe to call before the session is focused
// (everything reads inactive -> zeros).
static void update_input() {
    if (!xr.input_initialized) return;

    XrActiveActionSet active = { xr.action_set, XR_NULL_PATH };
    XrActionsSyncInfo sync = { XR_TYPE_ACTIONS_SYNC_INFO };
    sync.countActiveActionSets = 1;
    sync.activeActionSets = &active;
    if (!XR_SUCCEEDED(xrSyncActions(xr.session, &sync))) {
        return;
    }

    const XrTime t = xr.frame_state.predictedDisplayTime;

    for (int h = 0; h < 2; h++) {
        const XrPath hp = xr.hand_path[h];

        // Chain a velocity request into the grip locate: the runtime returns filtered + predicted
        // hand velocities for free — far cleaner than differentiating poses ourselves. Consumed by
        // the physical-combat layer (vr_physics) for swing speed and throw velocity.
        XrSpaceVelocity vel = { XR_TYPE_SPACE_VELOCITY };
        XrSpaceLocation loc = { XR_TYPE_SPACE_LOCATION };
        loc.next = &vel;
        xrLocateSpace(xr.grip_space[h], xr.local_space, t, &loc);
        const bool valid = (loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
                           (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT);
        xr.hand_active[h] = valid;
        xr.hand_tracked[h] = valid && (loc.locationFlags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT);
        if (valid) {
            xr.grip_pose[h] = loc.pose;
        }
        xr.hand_vel_valid[h] = valid && (vel.velocityFlags & XR_SPACE_VELOCITY_LINEAR_VALID_BIT);
        xr.hand_lin_vel[h] = xr.hand_vel_valid[h] ? vel.linearVelocity : XrVector3f{ 0.0f, 0.0f, 0.0f };
        xr.hand_ang_vel[h] = (valid && (vel.velocityFlags & XR_SPACE_VELOCITY_ANGULAR_VALID_BIT))
                                 ? vel.angularVelocity
                                 : XrVector3f{ 0.0f, 0.0f, 0.0f };

        XrSpaceLocation aloc = { XR_TYPE_SPACE_LOCATION };
        xrLocateSpace(xr.aim_space[h], xr.local_space, t, &aloc);
        if ((aloc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
            (aloc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
            xr.aim_pose[h] = aloc.pose;
            xr.aim_pose_raw[h] = aloc.pose;
        }

        auto get_float = [&](XrAction a) -> float {
            XrActionStateGetInfo gi = { XR_TYPE_ACTION_STATE_GET_INFO };
            gi.action = a;
            gi.subactionPath = hp;
            XrActionStateFloat st = { XR_TYPE_ACTION_STATE_FLOAT };
            if (XR_SUCCEEDED(xrGetActionStateFloat(xr.session, &gi, &st)) && st.isActive) {
                return st.currentState;
            }
            return 0.0f;
        };
        auto get_bool = [&](XrAction a) -> bool {
            XrActionStateGetInfo gi = { XR_TYPE_ACTION_STATE_GET_INFO };
            gi.action = a;
            gi.subactionPath = hp;
            XrActionStateBoolean st = { XR_TYPE_ACTION_STATE_BOOLEAN };
            if (XR_SUCCEEDED(xrGetActionStateBoolean(xr.session, &gi, &st)) && st.isActive) {
                return st.currentState == XR_TRUE;
            }
            return false;
        };

        xr.trigger_value[h] = get_float(xr.trigger_action);
        xr.squeeze_value[h] = get_float(xr.squeeze_action);

        XrActionStateGetInfo gi = { XR_TYPE_ACTION_STATE_GET_INFO };
        gi.action = xr.thumbstick_action;
        gi.subactionPath = hp;
        XrActionStateVector2f vst = { XR_TYPE_ACTION_STATE_VECTOR2F };
        if (XR_SUCCEEDED(xrGetActionStateVector2f(xr.session, &gi, &vst)) && vst.isActive) {
            xr.thumbstick_x[h] = vst.currentState.x;
            xr.thumbstick_y[h] = vst.currentState.y;
        } else {
            xr.thumbstick_x[h] = xr.thumbstick_y[h] = 0.0f;
        }
        // Modal hand gestures (the Alyx-style item selector) suppress a hand's stick at the
        // SOURCE, so every consumer — movement, artificial turning, stick C-buttons — inherits
        // it: holding a stick-click gesture cannot steer, turn or fire items.
        if (g_stick_suppressed[h]) {
            xr.thumbstick_x[h] = xr.thumbstick_y[h] = 0.0f;
        }

        // Bitmask. Analog trigger/grip are thresholded so they also read as digital buttons.
        uint16_t b = 0;
        if (xr.trigger_value[h] > 0.6f) b |= (1 << 0);            // VR_BTN_TRIGGER
        if (xr.squeeze_value[h] > 0.6f) b |= (1 << 1);            // VR_BTN_GRIP
        if (get_bool(xr.primary_action)) b |= (1 << 2);          // VR_BTN_PRIMARY
        if (get_bool(xr.secondary_action)) b |= (1 << 3);        // VR_BTN_SECONDARY
        if (get_bool(xr.thumbstick_click_action)) b |= (1 << 4); // VR_BTN_THUMBCLICK
        if (get_bool(xr.menu_action)) b |= (1 << 5);             // VR_BTN_MENU
        xr.buttons[h] = b;
    }
}

// --------------------------------------------------------------------------
// SoH menu panel: the ImGui settings menu in the headset, driven by a controller laser
// --------------------------------------------------------------------------
// While the SoH menu is visible and VR is running, ImGui lays out for this panel instead of the
// desktop window and draws into the menu swapchain (Fast3dGui), shown as a world-locked quad put in
// front of the gaze when the menu opens (RAW local_space metres, like every compositor quad). A
// controller ray is the mouse: the hit is the cursor, the trigger the left button, the stick the
// wheel; the hand whose trigger was pulled last points. The game sees neutral controller input
// while the panel is up, and each input stays hidden after it closes until it is released, so the
// click that closed the menu can't swing a sword.
// HOLDING the off hand's thumbstick click (left; right when left-handed) opens and closes it. A TAP
// keeps doing what it is bound to (START, pause, by default): the game gets that tap as a short
// pulse when the stick is released, since at press time it can't yet be told apart from the start
// of a hold. (The item selector's default hold is the SWORD hand's stick click: no overlap.)

static constexpr uint32_t kMenuPanelW = 1600; // ImGui display size of the panel, px
static constexpr uint32_t kMenuPanelH = 1000;
static constexpr float kMenuHoldSec = 0.6f;
static constexpr float kMenuTapPulseSec = 0.12f; // over two 20 Hz game ticks: padmgr can't miss it

static struct {
    bool open; // latched once per XR frame (vr_menu_update)
    XrPosef pose;
    float size[2]; // metres
    // Off-hand stick click: hold = toggle, tap = pulse to the game.
    bool btn_prev;
    bool btn_pending;
    double btn_down_t;
    double pulse_until;
    // Pointer.
    int hand; // -1 = not chosen yet (dominant hand on open)
    bool hit;
    float px, py; // panel pixels
    float hit_dist; // metres along the ray
    bool down;
    float wheel; // ImGui wheel units accumulated since the last vr_menu_take_pointer
    // After closing: inputs that were held stay hidden from the game until released.
    uint16_t latch_buttons[2];
    bool latch_trigger[2], latch_squeeze[2], latch_stick[2];
} g_menu = { false, {}, {}, false, false, 0.0, 0.0, -1 };

static std::shared_ptr<Ship::GuiWindow> vr_menu_window() {
    Ship::Context* ctx = Ship::Context::GetRawInstance();
    if (ctx == nullptr || ctx->GetWindow() == nullptr || ctx->GetWindow()->GetGui() == nullptr) {
        return nullptr;
    }
    return ctx->GetWindow()->GetGui()->GetMenu();
}

// Level, gaze-facing placement in front of the head (yaw only, like the flat-screen panel).
static void vr_menu_place() {
    const XrPosef& vp = xr.views[0].pose; // raw: the turn is applied later in vr_begin_frame
    const glm::vec3 head(0.5f * (xr.views[0].pose.position.x + xr.views[1].pose.position.x),
                         0.5f * (xr.views[0].pose.position.y + xr.views[1].pose.position.y),
                         0.5f * (xr.views[0].pose.position.z + xr.views[1].pose.position.z));
    const glm::quat ho(vp.orientation.w, vp.orientation.x, vp.orientation.y, vp.orientation.z);
    glm::vec3 fwd = ho * glm::vec3(0.0f, 0.0f, -1.0f);
    fwd.y = 0.0f;
    const float len = glm::length(fwd);
    fwd = (len > 1e-4f) ? fwd / len : glm::vec3(0.0f, 0.0f, -1.0f);
    const float dist = std::clamp(CVarGetFloat("gVrMenuDistance", 1.0f), 0.4f, 3.0f);
    const float width = std::clamp(CVarGetFloat("gVrMenuWidth", 1.2f), 0.4f, 3.0f);
    const float height = std::clamp(CVarGetFloat("gVrMenuHeight", -0.1f), -1.0f, 1.0f);
    const glm::vec3 pos = head + fwd * dist + glm::vec3(0.0f, height, 0.0f);
    const float qyaw = atan2f(-fwd.x, -fwd.z); // the quad's front (+Z) faces back at the player
    const glm::quat q = glm::angleAxis(qyaw, glm::vec3(0.0f, 1.0f, 0.0f));
    g_menu.pose.position = { pos.x, pos.y, pos.z };
    g_menu.pose.orientation = { q.x, q.y, q.z, q.w };
    g_menu.size[0] = width;
    g_menu.size[1] = width * (float)kMenuPanelH / (float)kMenuPanelW;
}

// The pointing hand's ray (RAW aim pose) against the panel's front face.
static void vr_menu_cast(int hand) {
    g_menu.hit = false;
    if (hand < 0 || !xr.hand_active[hand]) {
        return;
    }
    const XrPosef& a = xr.aim_pose_raw[hand];
    const glm::vec3 o(a.position.x, a.position.y, a.position.z);
    const glm::vec3 d = glm::quat(a.orientation.w, a.orientation.x, a.orientation.y, a.orientation.z) *
                        glm::vec3(0.0f, 0.0f, -1.0f);
    const glm::vec3 c(g_menu.pose.position.x, g_menu.pose.position.y, g_menu.pose.position.z);
    const glm::quat pq(g_menu.pose.orientation.w, g_menu.pose.orientation.x, g_menu.pose.orientation.y,
                       g_menu.pose.orientation.z);
    const glm::vec3 n = pq * glm::vec3(0.0f, 0.0f, 1.0f);
    const float denom = glm::dot(d, n);
    if (denom > -1e-4f) {
        return; // parallel, or pointing at the back
    }
    const float t = glm::dot(c - o, n) / denom;
    if (t <= 0.0f) {
        return;
    }
    const glm::vec3 local = glm::inverse(pq) * (o + d * t - c);
    const float u = local.x / g_menu.size[0] + 0.5f;
    const float v = 0.5f - local.y / g_menu.size[1];
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return;
    }
    g_menu.hit = true;
    g_menu.hit_dist = t;
    g_menu.px = u * (float)kMenuPanelW;
    g_menu.py = v * (float)kMenuPanelH;
}

// Once per XR frame, right after update_input: before anything (snap turn, the game) reads input.
static void vr_menu_update() {
    const double now = (double)xr.frame_state.predictedDisplayTime * 1e-9;
    float dt = xr.frame_state.predictedDisplayPeriod > 0
                   ? (float)((double)xr.frame_state.predictedDisplayPeriod * 1e-9)
                   : 1.0f / (float)vr_get_refresh_rate();
    dt = std::min(dt, 1.0f / 30.0f);
    const std::shared_ptr<Ship::GuiWindow> menu = vr_menu_window();

    // Off-hand stick click: hold toggles the panel; a tap reaches the game as a pulse on release.
    const int mh = CVarGetInteger("gVrLeftHanded", 0) ? 1 : 0;
    const bool btn = (xr.buttons[mh] & VR_BTN_THUMBCLICK) != 0;
    bool toggle = false;
    if (btn && !g_menu.btn_prev) {
        g_menu.btn_pending = true;
        g_menu.btn_down_t = now;
    }
    if (g_menu.btn_pending && btn && now - g_menu.btn_down_t >= kMenuHoldSec) {
        g_menu.btn_pending = false;
        toggle = true;
    } else if (g_menu.btn_pending && !btn) {
        g_menu.btn_pending = false;
        if (g_menu.open) {
            toggle = true; // a tap also closes it
        } else {
            g_menu.pulse_until = now + kMenuTapPulseSec;
        }
    }
    g_menu.btn_prev = btn;
    xr.buttons[mh] &= ~VR_BTN_THUMBCLICK;
    if (now < g_menu.pulse_until) {
        xr.buttons[mh] |= VR_BTN_THUMBCLICK;
    }
    if (toggle && menu != nullptr) {
        menu->ToggleVisibility();
        Ship::Context::GetRawInstance()->GetWindow()->GetMouseStateManager()->UpdateMouseCapture();
        vr_trigger_haptic(mh, 0.4f, 0.0f, 30.0f);
    }

    // Open = the menu is visible (Esc on the desktop opens it too).
    const bool was_open = g_menu.open;
    g_menu.open = menu != nullptr && menu->IsVisible();
    if (g_menu.open && !was_open) {
        vr_menu_place();
        xr.menu_ever_rendered = false; // not the last opening's image: wait for this one's first frame
        g_menu.down = false;
        g_menu.wheel = 0.0f;
        g_menu.hand = CVarGetInteger("gVrLeftHanded", 0) ? 0 : 1;
    }
    if (!g_menu.open && was_open) {
        for (int h = 0; h < 2; h++) {
            g_menu.latch_buttons[h] = xr.buttons[h];
            g_menu.latch_trigger[h] = xr.trigger_value[h] > 0.2f;
            g_menu.latch_squeeze[h] = xr.squeeze_value[h] > 0.2f;
            g_menu.latch_stick[h] = fabsf(xr.thumbstick_x[h]) > 0.25f || fabsf(xr.thumbstick_y[h]) > 0.25f;
        }
        g_menu.down = false;
    }

    if (g_menu.open) {
        // The pointer follows the hand that pulls its trigger.
        for (int h = 0; h < 2; h++) {
            if (h != g_menu.hand && !g_menu.down && xr.trigger_value[h] > 0.55f && xr.hand_active[h]) {
                g_menu.hand = h;
            }
        }
        vr_menu_cast(g_menu.hand);
        const float trig = g_menu.hand >= 0 ? xr.trigger_value[g_menu.hand] : 0.0f;
        if (!g_menu.down && trig > 0.55f) {
            g_menu.down = true;
            if (g_menu.hit) {
                vr_trigger_haptic(g_menu.hand, 0.25f, 0.0f, 15.0f);
            }
        } else if (g_menu.down && trig < 0.35f) {
            g_menu.down = false;
        }
        // Scroll by time, not per frame: full deflection = 12 wheel units a second.
        const float sy = g_menu.hand >= 0 ? xr.thumbstick_y[g_menu.hand] : 0.0f;
        if (fabsf(sy) > 0.2f) {
            g_menu.wheel += (sy > 0.0f ? 1.0f : -1.0f) * (fabsf(sy) - 0.2f) / 0.8f * 12.0f * dt;
        }
    }

    // What the game sees.
    for (int h = 0; h < 2; h++) {
        if (g_menu.open) {
            xr.buttons[h] = 0;
            xr.trigger_value[h] = xr.squeeze_value[h] = 0.0f;
            xr.thumbstick_x[h] = xr.thumbstick_y[h] = 0.0f;
            continue;
        }
        g_menu.latch_buttons[h] &= xr.buttons[h];
        xr.buttons[h] &= ~g_menu.latch_buttons[h];
        if (g_menu.latch_trigger[h]) {
            g_menu.latch_trigger[h] = xr.trigger_value[h] > 0.2f;
            xr.trigger_value[h] = 0.0f;
            xr.buttons[h] &= ~VR_BTN_TRIGGER;
        }
        if (g_menu.latch_squeeze[h]) {
            g_menu.latch_squeeze[h] = xr.squeeze_value[h] > 0.2f;
            xr.squeeze_value[h] = 0.0f;
            xr.buttons[h] &= ~VR_BTN_GRIP;
        }
        if (g_menu.latch_stick[h]) {
            g_menu.latch_stick[h] = fabsf(xr.thumbstick_x[h]) > 0.25f || fabsf(xr.thumbstick_y[h]) > 0.25f;
            xr.thumbstick_x[h] = xr.thumbstick_y[h] = 0.0f;
        }
    }
}

// The beam strip, once per session: transparent at the top (the far end, +Y of the quad), fading in
// toward the hand. Built from ClearColorRects (premultiplied), so every backend gets it for free.
static void vr_menu_fill_beam() {
    auto& sc = xr.beam_swapchain;
    if (sc.handle == XR_NULL_HANDLE) {
        return;
    }
    XrSwapchainImageAcquireInfo acquire_info = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
    uint32_t image = 0;
    if (!xr_check(xrAcquireSwapchainImage(sc.handle, &acquire_info, &image), "xrAcquireSwapchainImage (beam)")) {
        return;
    }
    XrSwapchainImageWaitInfo wait_info = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
    wait_info.timeout = XR_INFINITE_DURATION;
    xr_check(xrWaitSwapchainImage(sc.handle, &wait_info), "xrWaitSwapchainImage (beam)");
    const float clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    xr.gfx->BeginPass(vrgfx::Target::Beam, image, clear);
    constexpr int kBands = 16;
    for (int i = 0; i < kBands; i++) {
        const float a = 0.85f * (float)(i + 1) / (float)kBands; // band 0 = top = far end
        const float rgba[4] = { 0.55f * a, 0.85f * a, 1.0f * a, a };
        const vrgfx::Rect r = { 0, (int32_t)(sc.height * i / kBands), (int32_t)sc.width,
                                (int32_t)(sc.height / kBands) };
        xr.gfx->ClearColorRects(vrgfx::Target::Beam, image, &r, 1, rgba);
    }
    xr.gfx->EndPass(vrgfx::Target::Beam, image);
    XrSwapchainImageReleaseInfo release_info = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    xr.beam_ready =
        xr_check(xrReleaseSwapchainImage(sc.handle, &release_info), "xrReleaseSwapchainImage (beam)");
}

bool vr_menu_panel_active() {
    return xr.initialized && xr.enabled && xr.session_running && g_menu.open;
}

void vr_menu_panel_size(int* w, int* h) {
    *w = (int)kMenuPanelW;
    *h = (int)kMenuPanelH;
}

void vr_menu_take_pointer(VrMenuPointer* out) {
    out->valid = g_menu.open && g_menu.hit;
    out->x = g_menu.px;
    out->y = g_menu.py;
    out->down = g_menu.down;
    out->wheel = g_menu.wheel;
    out->hand = g_menu.hand;
    g_menu.wheel = 0.0f;
}

// The running ImGui renderer backend draws the panel between these (Fast3dGui). The image is
// cleared to transparent; ImGui's blending leaves premultiplied colour, which is what the
// compositor expects. Afterwards the window framebuffer is bound again for the desktop's copy.
bool vr_begin_menu() {
    if (!xr.initialized || !xr.gfx || xr.menu_swapchain.handle == XR_NULL_HANDLE) {
        return false;
    }
    auto& sc = xr.menu_swapchain;
    XrSwapchainImageAcquireInfo acquire_info = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
    if (!xr_check(xrAcquireSwapchainImage(sc.handle, &acquire_info, &xr.menu_image_index),
                  "xrAcquireSwapchainImage (menu)")) {
        return false;
    }
    XrSwapchainImageWaitInfo wait_info = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
    wait_info.timeout = XR_INFINITE_DURATION;
    xr_check(xrWaitSwapchainImage(sc.handle, &wait_info), "xrWaitSwapchainImage (menu)");
    const float clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    xr.gfx->BeginPass(vrgfx::Target::Menu, xr.menu_image_index, clear);
    return true;
}

void vr_end_menu() {
    if (!xr.initialized || !xr.gfx) {
        return;
    }
    vr_draw_test_card(vrgfx::Target::Menu, xr.menu_image_index, xr.menu_swapchain.width, xr.menu_swapchain.height);
    xr.gfx->EndPass(vrgfx::Target::Menu, xr.menu_image_index);
    XrSwapchainImageReleaseInfo release_info = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    if (xr_check(xrReleaseSwapchainImage(xr.menu_swapchain.handle, &release_info), "xrReleaseSwapchainImage (menu)")) {
        xr.menu_ever_rendered = true;
    }
    if (Fast::Interpreter* interp = vr_get_interpreter()) {
        interp->GetCurrentRenderingAPI()->StartDrawToFramebuffer(0, 0.0f);
    }
}

// --------------------------------------------------------------------------
// Artificial snap-turn: an accumulated world-space yaw (rotation + the translation that keeps the
// pivot fixed) applied to every game-facing pose. Never applied to compositor-submitted poses.
// --------------------------------------------------------------------------

static glm::quat g_turn_rot(1.0f, 0.0f, 0.0f, 0.0f);
static glm::vec3 g_turn_off(0.0f);

// A SNAP waits for the next game tick (vr_commit_pending_snap_turn, from VR_GameTickBegin). The
// tick records its display lists once, culled against the head pose it read, and they are then
// shown for every sub-frame until the next tick; a snap applied mid-tick showed those lists turned
// 45 degrees, with the edge on the turned-toward side culled away for up to 50 ms. Committing at
// the tick boundary lets that tick cull for the new heading. Smooth turning steps a degree or two
// per frame, well inside the culling padding, so it still applies per frame.
static float g_snap_pending_deg = 0.0f;
static int g_snap_pending_frames = 0;

// Drop the accumulated artificial turn (system recenter: it was expressed in old-space coords).
static void vr_reset_snap_turn() {
    g_turn_rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    g_turn_off = glm::vec3(0.0f);
    g_snap_pending_deg = 0.0f;
    g_snap_pending_frames = 0;
}

// The world-space pause frame's canvas relative to the TV frame it replaces: { width x, height x,
// element size x } (gVrPauseHudWidth / Height / Scale, percent). 1, 1, 1 = vanilla's framing; the
// width defaults to 1.3 (user, October 5) so the left and right groups spread off the pages.
static void vr_hud_world_canvas(float out[3]) {
    out[0] = std::clamp(CVarGetFloat("gVrPauseHudWidth", 130.0f), 50.0f, 400.0f) / 100.0f;
    out[1] = std::clamp(CVarGetFloat("gVrPauseHudHeight", 100.0f), 50.0f, 400.0f) / 100.0f;
    out[2] = std::clamp(CVarGetFloat("gVrPauseHudScale", 100.0f), 25.0f, 300.0f) / 100.0f;
}

static XrPosef apply_turn(const XrPosef& p) {
    const glm::vec3 pos = g_turn_rot * glm::vec3(p.position.x, p.position.y, p.position.z) + g_turn_off;
    const glm::quat q =
        g_turn_rot * glm::quat(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
    XrPosef out;
    out.position = { pos.x, pos.y, pos.z };
    out.orientation = { q.x, q.y, q.z, q.w };
    return out;
}

// Rotate the world by `degrees_right` (positive = player turns right) about the vertical axis
// through the player's current head position. Pivoting on the head keeps the player in place —
// any other pivot would translate them sideways as they turn. Called with this frame's raw views
// located but not yet turn-adjusted. Serves BOTH turn styles: one 45-degree call per flick for
// snap, one sub-degree call per frame for smooth (the pivot re-derives from the live head each
// call, so continuous turning stays centered on the player).
static void vr_apply_snap_turn(float degrees_right) {
    const float rad = degrees_right * (3.14159265358979323846f / 180.0f);
    // Right-handed yaw about +Y turns left, so turning right is the negative angle.
    const glm::quat r = glm::angleAxis(-rad, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 raw_center =
        0.5f * (glm::vec3(xr.views[0].pose.position.x, xr.views[0].pose.position.y, xr.views[0].pose.position.z) +
                glm::vec3(xr.views[1].pose.position.x, xr.views[1].pose.position.y, xr.views[1].pose.position.z));
    const glm::vec3 pivot = g_turn_rot * raw_center + g_turn_off; // where the head currently appears
    g_turn_rot = glm::normalize(r * g_turn_rot);
    g_turn_off = r * (g_turn_off - pivot) + pivot;
}

// Commit a latched snap at the game-tick boundary. By now vr_begin_frame has already turned this
// frame's poses, so the pivot is the TURNED centre eye, and every game-facing pose the tick is
// about to read (views, grips, aims, the view matrices) gets the same rotation about it — the tick
// then sees one consistent, already-turned frame. Re-checks the turn gates: a context that took
// the turn away since the stick latched (pause, flat screen, third person) drops the snap.
void vr_commit_pending_snap_turn() {
    if (g_snap_pending_deg == 0.0f) {
        return;
    }
    const float deg = g_snap_pending_deg;
    g_snap_pending_deg = 0.0f;
    g_snap_pending_frames = 0;
    if (!xr.initialized || xr.flat_screen || !xr.first_person || g_turn_suppressed) {
        return;
    }
    const float rad = deg * (3.14159265358979323846f / 180.0f);
    const glm::quat r = glm::angleAxis(-rad, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 pivot =
        0.5f * (glm::vec3(xr.views[0].pose.position.x, xr.views[0].pose.position.y, xr.views[0].pose.position.z) +
                glm::vec3(xr.views[1].pose.position.x, xr.views[1].pose.position.y, xr.views[1].pose.position.z));
    g_turn_rot = glm::normalize(r * g_turn_rot);
    g_turn_off = r * (g_turn_off - pivot) + pivot;

    auto rotate_about_pivot = [&r, &pivot](XrPosef& p) {
        const glm::vec3 pos = r * (glm::vec3(p.position.x, p.position.y, p.position.z) - pivot) + pivot;
        const glm::quat q = glm::normalize(r * glm::quat(p.orientation.w, p.orientation.x, p.orientation.y,
                                                         p.orientation.z));
        p.position = { pos.x, pos.y, pos.z };
        p.orientation = { q.x, q.y, q.z, q.w };
    };
    for (int eye = 0; eye < 2; eye++) {
        rotate_about_pivot(xr.views[eye].pose);
        pose_to_view_matrix(xr.views[eye].pose, xr.world_scale, xr.view[eye]);
    }
    vr_build_center_view();
    for (int h = 0; h < 2; h++) {
        if (xr.hand_active[h]) {
            rotate_about_pivot(xr.grip_pose[h]);
            rotate_about_pivot(xr.aim_pose[h]);
        }
    }
}

// Lock-on framing request from the game: the world direction of the current lock-on target, and
// how long the request stays live without a refresh. The game pushes this once per 20 Hz tick, so
// the TTL only has to outlast a tick or two — it exists so that a game state which stops updating
// (cutscene, menu, scene unload) can never leave the world creeping around on a stale target.
static int16_t g_lockon_yaw = 0;
static float g_lockon_ttl = 0.0f;
static constexpr float kLockOnTtlSeconds = 0.15f;

// The requested bearing arrives at the 20 Hz game tick but is consumed at headset rate, so it is a
// STAIRCASE: while circling a target the bearing can sweep ~100 deg/s, which lands as a ~5 degree
// jump every tick. Chasing that directly is what made the view stutter — the slew limiter ran at
// full speed for ~40 ms eating each step, then sat still for the rest of the tick. So the input is
// low-passed into a continuous bearing first, and the tracker below is proportional rather than
// bang-bang. Together they turn a stepped input into steady motion at the true sweep rate.
// Invalidated whenever the request drops, so acquiring a new target starts from where you are
// looking instead of sweeping in from the last target's bearing.
static float g_lockon_smoothed_deg = 0.0f;
static bool g_lockon_smooth_valid = false;
static constexpr float kLockOnInputTau = 0.04f;  // seconds; smooths the 20 Hz staircase
static constexpr float kLockOnTrackTau = 0.04f;  // seconds; tracker stiffness near the target

// Signed degrees in (-180, 180]. Bearings wrap, and every difference here has to take the short
// way around or the view would unwind the long way through a heading crossing.
static float vr_wrap180(float deg) {
    deg = fmodf(deg + 180.0f, 360.0f);
    if (deg < 0.0f) {
        deg += 360.0f;
    }
    return deg - 180.0f;
}

// One-shot "face this way" (vr_request_face_yaw): an instant artificial turn, consumed by the next
// frame's turn block, that leaves the player's heading on the requested game yaw.
static bool g_face_yaw_pending = false;
static int16_t g_face_yaw = 0;

void vr_set_climb_view_limits(const float wall_out[3], const float lo[3], const float hi[3]) {
    if (wall_out == nullptr || lo == nullptr || hi == nullptr) {
        g_climb_lim_valid = false;
        return;
    }
    glm::vec3 n(wall_out[0], 0.0f, wall_out[2]);
    const float len = glm::length(n);
    if (len < 1e-4f) {
        g_climb_lim_valid = false;
        return;
    }
    g_climb_lim_out = n / len;
    g_climb_lim_lo = glm::max(glm::vec3(lo[0], lo[1], lo[2]), glm::vec3(0.0f));
    g_climb_lim_hi = glm::max(glm::vec3(hi[0], hi[1], hi[2]), glm::vec3(0.0f));
    g_climb_lim_valid = true;
}

bool vr_consume_climb_discontinuity() {
    const bool d = g_climb_discontinuity;
    g_climb_discontinuity = false;
    return d;
}

void vr_request_face_yaw(int16_t yaw_binang) {
    g_face_yaw = yaw_binang;
    g_face_yaw_pending = true;
}

void vr_set_lockon_yaw(int16_t yaw_binang, bool active) {
    g_lockon_yaw = yaw_binang;
    g_lockon_ttl = active ? kLockOnTtlSeconds : 0.0f;
    if (!active) {
        g_lockon_smooth_valid = false;
    }
}

// The heading the player WILL have this frame: the raw HMD forward carried through the turn
// accumulated so far. vr_get_heading_yaw reads xr.views, which are still raw at the point the turn
// block runs (apply_turn happens further down), so the turn has to be composed in by hand here.
// Matches vr_get_heading_yaw's convention exactly, so "the target is dead
// ahead" means the same thing to the framing code and to Link's steering.
static bool vr_pending_heading_yaw(int16_t* out) {
    const XrQuaternionf& q = xr.views[0].pose.orientation;
    const glm::quat gq = g_turn_rot * glm::quat(q.w, q.x, q.y, q.z);
    const glm::vec3 fwd = gq * glm::vec3(0.0f, 0.0f, -1.0f);
    if (fwd.x * fwd.x + fwd.z * fwd.z <= 1e-6f) {
        return false; // looking straight up or down: heading is degenerate, correct nothing
    }
    const float yaw = atan2f(fwd.x, fwd.z);
    *out = static_cast<int16_t>(yaw * (32768.0f / 3.14159265358979323846f));
    return true;
}

// --------------------------------------------------------------------------
// OpenXR environment diagnostics
// --------------------------------------------------------------------------
// Most "VR won't start" / "the headset stays black" reports come down to the machine's OpenXR
// setup, which other VR games often never touch (OpenVR titles bypass it entirely): which runtime
// the loader is pointed at, and which API layers it injects into every OpenXR app. All of it goes
// to the log, so one log file from a player identifies their setup.

#ifdef _WIN32
static std::string vr_narrow(const wchar_t* w) {
    if (w == nullptr || *w == L'\0') {
        return std::string();
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) {
        return std::string();
    }
    std::string s(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

// The runtime manifest the loader will use: XR_RUNTIME_JSON overrides the registry.
static std::string vr_active_runtime_path() {
    char env[1024];
    const DWORD env_len = GetEnvironmentVariableA("XR_RUNTIME_JSON", env, sizeof(env));
    if (env_len > 0 && env_len < sizeof(env)) {
        return std::string(env) + " (XR_RUNTIME_JSON override)";
    }
    wchar_t buf[1024];
    DWORD size = sizeof(buf);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Khronos\\OpenXR\\1", L"ActiveRuntime", RRF_RT_REG_SZ, nullptr,
                     buf, &size) == ERROR_SUCCESS) {
        return vr_narrow(buf);
    }
    return "none registered";
}

// Implicit layers are registered as values named by their manifest path; DWORD 0 = enabled.
static int vr_log_implicit_layers(HKEY root, const char* root_name) {
    HKEY key;
    if (RegOpenKeyExW(root, L"SOFTWARE\\Khronos\\OpenXR\\1\\ApiLayers\\Implicit", 0, KEY_READ, &key) != ERROR_SUCCESS) {
        return 0;
    }
    int count = 0;
    for (DWORD i = 0;; i++) {
        wchar_t name[1024];
        DWORD name_len = 1024;
        DWORD type = 0;
        DWORD data = 1;
        DWORD data_size = sizeof(data);
        const LONG r = RegEnumValueW(key, i, name, &name_len, nullptr, &type, reinterpret_cast<BYTE*>(&data),
                                     &data_size);
        if (r == ERROR_NO_MORE_ITEMS) {
            break;
        }
        if (r != ERROR_SUCCESS) {
            continue;
        }
        spdlog::info("[VR]   implicit layer ({}): {} [{}]", root_name, vr_narrow(name),
                     (type == REG_DWORD && data == 0) ? "enabled" : "disabled");
        count++;
    }
    RegCloseKey(key);
    return count;
}
#else
// Android: the loader asks the system runtime broker (Meta Horizon OS); there is no registry.
static std::string vr_active_runtime_path() {
    return "the system runtime";
}
#endif

// Before xrCreateInstance: where the loader will go, and what it will load on the way. Once per
// active runtime, not on every reconnect attempt.
static void vr_log_openxr_environment() {
    static std::string s_logged_runtime;
    const std::string runtime = vr_active_runtime_path();
    if (runtime == s_logged_runtime) {
        return;
    }
    s_logged_runtime = runtime;
    spdlog::info("[VR] OpenXR active runtime: {}", runtime);
#ifdef _WIN32
    spdlog::info("[VR] OpenXR implicit API layers (loaded into every OpenXR app while enabled):");
    const int implicit = vr_log_implicit_layers(HKEY_LOCAL_MACHINE, "machine") +
                         vr_log_implicit_layers(HKEY_CURRENT_USER, "user");
    if (implicit == 0) {
        spdlog::info("[VR]   (none)");
    }
#endif
    uint32_t n = 0;
    if (XR_SUCCEEDED(xrEnumerateApiLayerProperties(0, &n, nullptr)) && n > 0) {
        std::vector<XrApiLayerProperties> layers(n, { XR_TYPE_API_LAYER_PROPERTIES });
        if (XR_SUCCEEDED(xrEnumerateApiLayerProperties(n, &n, layers.data()))) {
            for (uint32_t i = 0; i < n; i++) {
                spdlog::info("[VR]   available API layer: {} v{} - {}", layers[i].layerName, layers[i].layerVersion,
                             layers[i].description);
            }
        }
    }
}

static void vr_log_instance_properties() {
    XrInstanceProperties p = { XR_TYPE_INSTANCE_PROPERTIES };
    if (XR_SUCCEEDED(xrGetInstanceProperties(xr.instance, &p))) {
        spdlog::info("[VR] OpenXR runtime: {} {}.{}.{}", p.runtimeName, XR_VERSION_MAJOR(p.runtimeVersion),
                     XR_VERSION_MINOR(p.runtimeVersion), XR_VERSION_PATCH(p.runtimeVersion));
    }
}

static void vr_log_system_properties() {
    XrSystemProperties p = { XR_TYPE_SYSTEM_PROPERTIES };
    if (XR_SUCCEEDED(xrGetSystemProperties(xr.instance, xr.system_id, &p))) {
        spdlog::info("[VR] Headset: {} (vendor 0x{:04x}), max {} layers, max swapchain {}x{}, "
                     "orientation tracking {}, position tracking {}",
                     p.systemName, p.vendorId, p.graphicsProperties.maxLayerCount,
                     p.graphicsProperties.maxSwapchainImageWidth, p.graphicsProperties.maxSwapchainImageHeight,
                     p.trackingProperties.orientationTracking ? "yes" : "no",
                     p.trackingProperties.positionTracking ? "yes" : "no");
    }
}

// --------------------------------------------------------------------------
// Lifecycle
// --------------------------------------------------------------------------

bool vr_backend_supported(int window_backend) {
    // One table (plan 3.5): the renderers with a VR graphics leaf (vr_gfx_*.cpp).
    switch (window_backend) {
#ifdef ENABLE_DX11
        case Fast::WindowBackend::FAST3D_DXGI_DX11:
            return true;
#endif
#ifdef ENABLE_OPENGL
        case Fast::WindowBackend::FAST3D_SDL_OPENGL:
            return true;
#endif
        default:
            return false; // Metal: no OpenXR runtime on macOS.
    }
}

static bool vr_instance_init();
static bool vr_session_init();

bool vr_init() {
    // The graphics leaf for the running renderer. Only a supported backend is ever handed to a
    // leaf, so no leaf casts a renderer it wasn't built for.
    xr.gfx.reset();
    const int backend = vr_running_window_backend();
    if (!vr_backend_supported(backend)) {
        spdlog::error("[VR] VR is not available on the running renderer (backend id {})", backend);
        return false;
    }
    Fast::Interpreter* interp = vr_get_interpreter();
    xr.gfx = vrgfx::Create(backend, interp ? interp->GetCurrentRenderingAPI() : nullptr);
    if (!xr.gfx) {
        spdlog::error("[VR] No VR graphics backend for the running renderer");
        return false;
    }
    spdlog::info("[VR] Graphics backend: {}", xr.gfx->ApiName());

    // Default configuration
    xr.world_scale = 35.0f;
    // ~11 cm at the default scale. At 10 units the near plane reached past Link's wall radius
    // (child 14, adult 18 from the eye at a wall), so pressing against a wall showed through it.
    // Fog keeps its original near-10 depth curve via vr_get_fog_ndc_z.
    xr.near_clip = 4.0f;
    xr.far_clip = 30000.0f;

    // Default the frame plan to "do everything"; the window layer overrides it per frame. Without
    // this the zero-initialised flags would suppress every pass until the first vr_set_frame_plan.
    xr.plan_render_eyes = true;
    xr.plan_render_hud = true;
    xr.plan_present_desktop = true;
    xr.roomscale_origin = glm::vec2(0.0f);

    // Per-eye resolution multiplier. Runtimes (esp. SteamVR) often bake a supersampling
    // factor into the "recommended" size, so each eye can be 1.4-2x the panel resolution.
    // This is the main GPU-cost lever in VR; drop below 1.0 to trade sharpness for framerate.
    // Tunable via the gVrResolutionScale CVar (takes effect on next vr_init).
    xr.resolution_scale = CVarGetFloat("gVrResolutionScale", 1.0f);
    if (xr.resolution_scale < 0.1f) xr.resolution_scale = 0.1f;
    if (xr.resolution_scale > 2.0f) xr.resolution_scale = 2.0f;

    // Sane default until the first frame is located and we can read the true display period.
    xr.refresh_rate = 90;

    if (!vr_instance_init()) {
        xr.gfx->Shutdown();
        xr.gfx.reset();
        return false;
    }
    if (!vr_session_init()) {
        if (xr.gfx) {
            xr.gfx->Shutdown();
            xr.gfx.reset();
        }
        return false;
    }
    return true;
}

#ifdef __ANDROID__
// Chained into XrInstanceCreateInfo (XR_KHR_android_create_instance). The activity is a global
// ref held for the process: the runtime keeps using it for the instance's lifetime.
static XrInstanceCreateInfoAndroidKHR s_android_instance_ci = { XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR };

// XR_KHR_loader_init_android: the Android loader can't find the runtime until it has the JavaVM
// and an application context, so this runs before the first OpenXR call (extension enumeration).
static bool vr_android_loader_init() {
    static bool s_done = false;
    if (s_done) {
        return true;
    }
    JNIEnv* env = static_cast<JNIEnv*>(SDL_AndroidGetJNIEnv());
    jobject activity = static_cast<jobject>(SDL_AndroidGetActivity());
    JavaVM* vm = nullptr;
    if (env == nullptr || activity == nullptr || env->GetJavaVM(&vm) != JNI_OK || vm == nullptr) {
        spdlog::error("[VR] No JavaVM / activity for the OpenXR loader");
        return false;
    }
    jobject activity_ref = env->NewGlobalRef(activity);
    env->DeleteLocalRef(activity);

    PFN_xrInitializeLoaderKHR init_loader = nullptr;
    if (XR_FAILED(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR",
                                        reinterpret_cast<PFN_xrVoidFunction*>(&init_loader))) ||
        init_loader == nullptr) {
        spdlog::error("[VR] The OpenXR loader has no xrInitializeLoaderKHR");
        return false;
    }
    XrLoaderInitInfoAndroidKHR li = { XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR };
    li.applicationVM = vm;
    li.applicationContext = activity_ref;
    if (!xr_check(init_loader(reinterpret_cast<const XrLoaderInitInfoBaseHeaderKHR*>(&li)), "xrInitializeLoaderKHR")) {
        return false;
    }
    s_android_instance_ci.applicationVM = vm;
    s_android_instance_ci.applicationActivity = activity_ref;
    s_done = true;
    return true;
}
#endif

// Create xr.instance with the renderer API's extension (+ the backend's extras) and the optional
// extensions the runtime offers. Shared by vr_instance_init and the pre-device adapter probe.
static bool vr_create_instance(const char* required_ext, const char* api_name, const std::vector<const char*>& extra) {
#ifdef __ANDROID__
    if (!vr_android_loader_init()) {
        return false;
    }
#endif
    // Optional extensions are enabled only when the runtime offers them; the backend's graphics
    // extension is required, and checked rather than assumed.
    vr_log_openxr_environment();
    xr.user_presence_supported = false;
    xr.color_scale_supported = false;
    xr.perf_settings_supported = false;
    xr.refresh_rate_supported = false;
    {
        bool required_offered = false;
        uint32_t ext_count = 0;
        xrEnumerateInstanceExtensionProperties(nullptr, 0, &ext_count, nullptr);
        std::vector<XrExtensionProperties> ext_props(ext_count, { XR_TYPE_EXTENSION_PROPERTIES });
        xrEnumerateInstanceExtensionProperties(nullptr, ext_count, &ext_count, ext_props.data());
        for (const auto& p : ext_props) {
            if (strcmp(p.extensionName, XR_EXT_USER_PRESENCE_EXTENSION_NAME) == 0) {
                xr.user_presence_supported = true;
            }
            if (strcmp(p.extensionName, XR_KHR_COMPOSITION_LAYER_COLOR_SCALE_BIAS_EXTENSION_NAME) == 0) {
                xr.color_scale_supported = true;
            }
            if (strcmp(p.extensionName, XR_EXT_PERFORMANCE_SETTINGS_EXTENSION_NAME) == 0) {
                xr.perf_settings_supported = true;
            }
            if (strcmp(p.extensionName, XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME) == 0) {
                xr.refresh_rate_supported = true;
            }
            if (strcmp(p.extensionName, required_ext) == 0) {
                required_offered = true;
            }
        }
        // ext_count 0 = no runtime / loader failure: let xrCreateInstance report it as before.
        if (ext_count > 0 && !required_offered) {
            spdlog::error("[VR] The OpenXR runtime does not offer {}: it can't run VR on {}", required_ext, api_name);
            return false;
        }
    }

    std::vector<const char*> extensions = { required_ext };
    extensions.insert(extensions.end(), extra.begin(), extra.end());
    if (xr.user_presence_supported) {
        extensions.push_back(XR_EXT_USER_PRESENCE_EXTENSION_NAME);
    }
    if (xr.color_scale_supported) {
        extensions.push_back(XR_KHR_COMPOSITION_LAYER_COLOR_SCALE_BIAS_EXTENSION_NAME);
    }
    if (xr.perf_settings_supported) {
        extensions.push_back(XR_EXT_PERFORMANCE_SETTINGS_EXTENSION_NAME);
    }
    if (xr.refresh_rate_supported) {
        extensions.push_back(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME);
    }
#ifdef __ANDROID__
    extensions.push_back(XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME);
#endif

    XrInstanceCreateInfo instance_ci = { XR_TYPE_INSTANCE_CREATE_INFO };
#ifdef __ANDROID__
    instance_ci.next = &s_android_instance_ci;
#endif
    strcpy(instance_ci.applicationInfo.applicationName, "Ship of Harkinian VR");
    instance_ci.applicationInfo.applicationVersion = 1;
    strcpy(instance_ci.applicationInfo.engineName, "libultraship");
    instance_ci.applicationInfo.engineVersion = 1;
    instance_ci.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
    instance_ci.enabledExtensionCount = (uint32_t)extensions.size();
    instance_ci.enabledExtensionNames = extensions.data();

    if (!xr_check(xrCreateInstance(&instance_ci, &xr.instance), "xrCreateInstance")) {
        xr.instance = XR_NULL_HANDLE;
        spdlog::error("[VR] Failed to create an OpenXR instance with the active runtime ({}). Start your headset's "
                      "software (SteamVR, Meta Quest Link, Virtual Desktop, ...) and make sure it is set as the "
                      "OpenXR runtime.",
                      vr_active_runtime_path());
        return false;
    }
    vr_log_instance_properties();
    return true;
}

// xrGetSystem on xr.instance. FORM_FACTOR_UNAVAILABLE = the runtime is up but the headset isn't
// (still starting, Link not connected, asleep): the spec expects apps to retry, so the instance is
// kept and xr.init_retryable set. Any other failure destroys the instance.
static bool vr_get_system() {
    XrSystemGetInfo system_info = { XR_TYPE_SYSTEM_GET_INFO };
    system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    const XrResult system_result = xrGetSystem(xr.instance, &system_info, &xr.system_id);
    if (system_result == XR_ERROR_FORM_FACTOR_UNAVAILABLE) {
        xr.init_retryable = true;
        return false;
    }
    if (!xr_check(system_result, "xrGetSystem")) {
        xrDestroyInstance(xr.instance);
        xr.instance = XR_NULL_HANDLE;
        return false;
    }
    return true;
}

// Before the renderer creates its device: the GPU the runtime requires (vr_openxr.h). The instance
// stays alive for vr_init, which reuses it (the probe enables exactly what vr_instance_init would:
// the leaf's extension, no backend extras (the D3D11 leaf adds none), and the same optional ones).
bool vr_probe_required_adapter(int window_backend, uint64_t* luid) {
    const char* ext = vrgfx::InstanceExtensionFor(window_backend);
    if (!vr_backend_supported(window_backend) || ext == nullptr) {
        return false;
    }
    if (xr.instance == XR_NULL_HANDLE && !vr_create_instance(ext, vrgfx::ApiNameFor(window_backend), {})) {
        return false;
    }
    xr.init_retryable = false;
    if (!vr_get_system()) {
        spdlog::info("[VR] Headset not available at startup: the renderer picks the high-performance GPU");
        return false;
    }
    if (!vrgfx::RequiredAdapterLuid(window_backend, xr.instance, xr.system_id, luid)) {
        return false;
    }
    spdlog::info("[VR] Headset runtime requires adapter LUID {:08x}:{:08x}; the renderer will create its device there",
                 (uint32_t)(*luid >> 32), (uint32_t)(*luid & 0xffffffffu));
    return true;
}

// Instance + system + graphics requirements (plan 3.5). Kept apart from the session so a
// renderer that must create its device THROUGH OpenXR (Vulkan) can run this before its own Init.
static bool vr_instance_init() {
    xr.user_present = true; // assume worn until the runtime says otherwise
    xr.view_fade_target = xr.view_fade_current = 0.0f;
    xr.init_retryable = false;
    xr.session_lost = false;
    xr.session_exited = false;

    // An instance kept from an attempt whose headset wasn't available yet, or from the pre-device
    // adapter probe, is reused: re-creating it every retry would re-launch some runtimes.
    if (xr.instance == XR_NULL_HANDLE) {
        std::vector<const char*> extra;
        xr.gfx->AddInstanceExtensions(extra);
        if (!vr_create_instance(xr.gfx->RequiredInstanceExtension(), xr.gfx->ApiName(), extra)) {
            return false;
        }
    }

    if (!vr_get_system()) {
        return false;
    }
    vr_log_system_properties();

    // --- Graphics requirements (the runtime refuses the session unless they were queried) ---
    std::string why;
    if (!xr.gfx->CheckRequirements(xr.instance, xr.system_id, &why)) {
        spdlog::error("[VR] {} can't drive this OpenXR runtime: {}", xr.gfx->ApiName(), why);
        xrDestroyInstance(xr.instance);
        xr.instance = XR_NULL_HANDLE;
        return false;
    }
    return true;
}

// One XR swapchain of the given size + the backend's per-image color/depth targets for it.
// The core keeps only the handle and size; the backend owns every API object.
static bool vr_create_swapchain(vrgfx::Target t, decltype(xr.hud_swapchain)& sc, uint32_t w, uint32_t h, int64_t fmt,
                                const char* label, uint32_t array_size = 1) {
    sc.width = w;
    sc.height = h;
    sc.format = fmt;

    XrSwapchainCreateInfo swapchain_ci = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
    swapchain_ci.usageFlags =
        XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT | xr.gfx->ExtraUsageFlags();
    swapchain_ci.format = fmt;
    swapchain_ci.sampleCount = 1;
    swapchain_ci.width = w;
    swapchain_ci.height = h;
    swapchain_ci.faceCount = 1;
    swapchain_ci.arraySize = array_size;
    swapchain_ci.mipCount = 1;

    const std::string what = std::string("xrCreateSwapchain (") + label + ")";
    if (!xr_check(xrCreateSwapchain(xr.session, &swapchain_ci, &sc.handle), what.c_str())) {
        return false;
    }
    if (!xr.gfx->Attach(t, sc.handle, w, h, fmt)) {
        spdlog::error("[VR] {} swapchain: building render targets failed", label);
        return false;
    }
    uint32_t image_count = 0;
    xrEnumerateSwapchainImages(sc.handle, 0, &image_count, nullptr);
    spdlog::info("[VR] {} swapchain: {}x{}, {} images", label, w, h, image_count);
    return true;
}

// Binding, session, spaces, swapchains.
static bool vr_session_init() {
    // --- Create Session ---
    XrSessionCreateInfo session_ci = { XR_TYPE_SESSION_CREATE_INFO };
    session_ci.next = xr.gfx->SessionBinding();
    session_ci.systemId = xr.system_id;
    if (!xr_check(xrCreateSession(xr.instance, &session_ci, &xr.session), "xrCreateSession")) {
        xrDestroyInstance(xr.instance);
        xr.instance = XR_NULL_HANDLE;
        return false;
    }

    // --- Create Reference Space ---
    XrReferenceSpaceCreateInfo space_ci = { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
    space_ci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    space_ci.poseInReferenceSpace = { { 0, 0, 0, 1 }, { 0, 0, 0 } }; // Identity
    if (!xr_check(xrCreateReferenceSpace(xr.session, &space_ci, &xr.local_space), "xrCreateReferenceSpace")) {
        xrDestroySession(xr.session);
        xrDestroyInstance(xr.instance);
        return false;
    }

    // --- Create STAGE space (floor-level origin) for physical eye-height measurement ---
    // Used only to calibrate auto world scale: everything else stays in LOCAL. Optional — if the
    // runtime doesn't offer STAGE (no floor calibration), auto scale stays unavailable and the
    // manual gVrWorldScale value is used.
    xr.stage_space = XR_NULL_HANDLE;
    {
        uint32_t space_count = 0;
        xrEnumerateReferenceSpaces(xr.session, 0, &space_count, nullptr);
        std::vector<XrReferenceSpaceType> space_types(space_count);
        xrEnumerateReferenceSpaces(xr.session, space_count, &space_count, space_types.data());
        for (XrReferenceSpaceType t : space_types) {
            if (t == XR_REFERENCE_SPACE_TYPE_STAGE) {
                XrReferenceSpaceCreateInfo stage_ci = { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
                stage_ci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
                stage_ci.poseInReferenceSpace = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
                if (!XR_SUCCEEDED(xrCreateReferenceSpace(xr.session, &stage_ci, &xr.stage_space))) {
                    xr.stage_space = XR_NULL_HANDLE;
                }
                break;
            }
        }
        spdlog::info("[VR] Stage (floor) space {} — auto world scale {}",
                     xr.stage_space != XR_NULL_HANDLE ? "available" : "unavailable",
                     xr.stage_space != XR_NULL_HANDLE ? "enabled" : "disabled");
    }

    // Motion-control input (controller poses + buttons). Optional — the HMD works without it, so a
    // failure here just leaves input_initialized false and the VR_Get*Hand/Button APIs return empty.
    setup_input();

    // --- Enumerate View Configuration ---
    uint32_t view_count = 0;
    xrEnumerateViewConfigurationViews(xr.instance, xr.system_id,
                                      XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &view_count, nullptr);
    if (view_count != 2) {
        spdlog::error("[VR] Expected 2 views for stereo, got {}", view_count);
        vr_shutdown();
        return false;
    }
    xr.view_count = 2;
    xr.config_views[0] = { XR_TYPE_VIEW_CONFIGURATION_VIEW };
    xr.config_views[1] = { XR_TYPE_VIEW_CONFIGURATION_VIEW };
    xrEnumerateViewConfigurationViews(xr.instance, xr.system_id,
                                      XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                      2, &view_count, xr.config_views);

    spdlog::info("[VR] Recommended render resolution: {}x{} per eye",
                 xr.config_views[0].recommendedImageRectWidth,
                 xr.config_views[0].recommendedImageRectHeight);

    // --- Enumerate Swapchain Formats ---
    uint32_t format_count = 0;
    xrEnumerateSwapchainFormats(xr.session, 0, &format_count, nullptr);
    std::vector<int64_t> formats(format_count);
    xrEnumerateSwapchainFormats(xr.session, format_count, &format_count, formats.data());
    if (formats.empty()) {
        spdlog::error("[VR] The runtime offers no swapchain formats");
        vr_shutdown();
        return false;
    }
    // The backend applies the sRGB rule (vr_gfx.h): an sRGB swapchain the compositor decodes,
    // written verbatim; UNORM fallback with a gamma mismatch.
    const int64_t chosen_format = xr.gfx->ChooseColorFormat(formats);

    // --- Create Swapchains (one per eye, or one two-layer swapchain for single-pass stereo) ---
    uint32_t eye_w[2], eye_h[2];
    for (uint32_t eye = 0; eye < 2; eye++) {
        // Apply the resolution multiplier, then clamp to what the runtime allows.
        uint32_t scaled_w = (uint32_t)lroundf(xr.config_views[eye].recommendedImageRectWidth * xr.resolution_scale);
        uint32_t scaled_h = (uint32_t)lroundf(xr.config_views[eye].recommendedImageRectHeight * xr.resolution_scale);
        if (scaled_w < 1) scaled_w = 1;
        if (scaled_h < 1) scaled_h = 1;
        if (scaled_w > xr.config_views[eye].maxImageRectWidth) scaled_w = xr.config_views[eye].maxImageRectWidth;
        if (scaled_h > xr.config_views[eye].maxImageRectHeight) scaled_h = xr.config_views[eye].maxImageRectHeight;
        eye_w[eye] = scaled_w;
        eye_h[eye] = scaled_h;

        spdlog::info("[VR] Eye {} render resolution: {}x{} (recommended {}x{}, scale {:.2f})", eye, scaled_w, scaled_h,
                     xr.config_views[eye].recommendedImageRectWidth, xr.config_views[eye].recommendedImageRectHeight,
                     xr.resolution_scale);
    }

    // Single-pass stereo (gVrMultiview, takes effect when the session starts): both eyes in one
    // two-layer swapchain at the larger eye's size. Any failure falls back to two swapchains.
#ifdef __ANDROID__
    constexpr int kMultiviewDefault = 1;
#else
    constexpr int kMultiviewDefault = 0;
#endif
    xr.multiview = false;
    if (CVarGetInteger("gVrMultiview", kMultiviewDefault) && xr.gfx->SupportsMultiview()) {
        auto& sc = xr.eye_swapchains[0];
        const uint32_t w = std::max(eye_w[0], eye_w[1]);
        const uint32_t h = std::max(eye_h[0], eye_h[1]);
        if (vr_create_swapchain(vrgfx::Target::EyeArray, sc, w, h, chosen_format, "eyes (multiview)", 2)) {
            xr.eye_swapchains[1] = sc;
            xr.eye_swapchains[1].handle = XR_NULL_HANDLE; // size only; the layers live in [0]
            xr.multiview = true;
            vr_build_center_view();
        } else {
            spdlog::warn("[VR] Multiview eye swapchain failed; rendering the eyes in two passes");
            xr.gfx->Detach(vrgfx::Target::EyeArray);
            if (sc.handle != XR_NULL_HANDLE) {
                xrDestroySwapchain(sc.handle);
                sc.handle = XR_NULL_HANDLE;
            }
        }
    }
    spdlog::info("[VR] Stereo rendering: {}", xr.multiview ? "single pass (multiview)" : "one pass per eye");

    for (uint32_t eye = 0; eye < 2 && !xr.multiview; eye++) {
        if (!vr_create_swapchain(eye == 0 ? vrgfx::Target::Eye0 : vrgfx::Target::Eye1, xr.eye_swapchains[eye],
                                 eye_w[eye], eye_h[eye], chosen_format, eye == 0 ? "eye 0" : "eye 1")) {
            vr_shutdown();
            return false;
        }
    }

    // --- Create VIEW reference space (head-locked, for HUD overlay) ---
    XrReferenceSpaceCreateInfo view_space_ci = { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
    view_space_ci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    view_space_ci.poseInReferenceSpace = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
    if (!xr_check(xrCreateReferenceSpace(xr.session, &view_space_ci, &xr.view_space), "xrCreateReferenceSpace (VIEW)")) {
        vr_shutdown();
        return false;
    }

    // --- HUD swapchain (2048x1536, 4:3) ---
    // Twice the old 1024x768: the wrist layout packs each hand's canvas at half scale (twice the
    // room for moved / enlarged elements), and this keeps those elements as sharp as before.
    // --- Flat-screen swapchain (whole-frame panel for 2D contexts: N64 logo, title) ---
    // --- Text-panel swapchain. 4:3 like the HUD, so the interpreter's 2D aspect handling is the
    // identity and every texrect lands exactly where vanilla put it; the layer then crops its
    // imageRect to the text box. Higher resolution than the HUD: this one is read word by word.
    if (!vr_create_swapchain(vrgfx::Target::Hud, xr.hud_swapchain, 2048, 1536, chosen_format, "HUD") ||
        !vr_create_swapchain(vrgfx::Target::Screen, xr.screen_swapchain, 1280, 960, chosen_format, "screen") ||
        !vr_create_swapchain(vrgfx::Target::Text, xr.text_swapchain, 1280, 960, chosen_format, "text") ||
        !vr_create_swapchain(vrgfx::Target::Menu, xr.menu_swapchain, kMenuPanelW, kMenuPanelH, chosen_format,
                             "menu") ||
        !vr_create_swapchain(vrgfx::Target::Beam, xr.beam_swapchain, 8, 256, chosen_format, "beam")) {
        vr_shutdown();
        return false;
    }
    vr_menu_fill_beam();

    // Initialize views
    xr.views[0] = { XR_TYPE_VIEW };
    xr.views[1] = { XR_TYPE_VIEW };

    xr.grab.grabber = -1;
    xr.perf_cpu_level_applied = -1;
    g_refresh_rate_retried = false;
    vr_refresh_rate_init();
    xr.initialized = true;
    xr.enabled = true;
    spdlog::info("[VR] OpenXR initialized successfully");
    return true;
}

void vr_shutdown() {
    xr.initialized = false;
    xr.session_running = false;
    vrphys_reset();

    // The backend's views/depth/mirror copies go first, then the swapchains they were built on.
    if (xr.gfx) {
        xr.gfx->Shutdown();
        xr.gfx.reset();
    }
    for (auto* sc : { &xr.eye_swapchains[0], &xr.eye_swapchains[1], &xr.hud_swapchain, &xr.screen_swapchain,
                      &xr.text_swapchain, &xr.menu_swapchain, &xr.beam_swapchain }) {
        if (sc->handle != XR_NULL_HANDLE) {
            xrDestroySwapchain(sc->handle);
            sc->handle = XR_NULL_HANDLE;
        }
    }
    xr.menu_ever_rendered = false;
    xr.beam_ready = false;
    xr.refresh_rates.clear();
    xr.hud_commands = nullptr;
    xr.text_commands = nullptr;
    xr.text_has_image = false;
    xr.text_was_visible = false;
    xr.head_raw_valid = false;
    xr.eyes_ever_rendered = false;
    xr.multiview = false;
    xr.multiview_pass = false;
    xr.flat_screen = false;
    xr.flat_screen_prev = false;

    xr.mirror_quad_count = 0;
    if (xr.view_space != XR_NULL_HANDLE) {
        xrDestroySpace(xr.view_space);
        xr.view_space = XR_NULL_HANDLE;
    }
    if (xr.stage_space != XR_NULL_HANDLE) {
        xrDestroySpace(xr.stage_space);
        xr.stage_space = XR_NULL_HANDLE;
    }
    xr.scale_calibrated = false;
    xr.scale_recalibrate_requested = false;
    if (xr.local_space != XR_NULL_HANDLE) {
        xrDestroySpace(xr.local_space);
        xr.local_space = XR_NULL_HANDLE;
    }
    if (xr.session != XR_NULL_HANDLE) {
        xrDestroySession(xr.session);
        xr.session = XR_NULL_HANDLE;
    }
    if (xr.instance != XR_NULL_HANDLE) {
        xrDestroyInstance(xr.instance);
        xr.instance = XR_NULL_HANDLE;
    }

    spdlog::info("[VR] OpenXR shut down");
}

// --------------------------------------------------------------------------
// Per-frame
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Frame plan + timing
// --------------------------------------------------------------------------

void vr_set_frame_plan(bool render_eyes, bool render_hud, bool present_desktop) {
    xr.plan_render_eyes = render_eyes;
    xr.plan_render_hud = render_hud;
    xr.plan_present_desktop = present_desktop;
}

bool vr_should_render_eyes() { return xr.plan_render_eyes; }
bool vr_should_render_hud() { return xr.plan_render_hud; }
bool vr_should_present_desktop() { return xr.plan_present_desktop; }

namespace {

VrFrameStats g_stats = {};
// Rate counters: count events, convert to Hz once a second so the readout is stable.
int g_eye_pass_count = 0;
int g_frame_count = 0;
std::chrono::steady_clock::time_point g_rate_epoch = std::chrono::steady_clock::now();

// Exponential smoothing. Frame times at 120 Hz are noisy enough that an unsmoothed readout is
// unreadable; ~0.05 settles in well under a second while still showing spikes.
inline void smooth(float& acc, float sample) {
    acc += (sample - acc) * 0.05f;
}

} // namespace

void vr_report_frame_times(float eyes_ms, float hud_ms, float desktop_ms, float frame_ms, bool rendered_eyes) {
    // Only fold an eye-pass sample in on frames that actually ran one, or the average decays toward
    // zero on reprojection-only frames and stops meaning "cost of an eye pass".
    if (rendered_eyes) {
        smooth(g_stats.eyes_ms, eyes_ms);
        g_eye_pass_count++;
    }
    if (hud_ms > 0.0f) {
        smooth(g_stats.hud_ms, hud_ms);
    }
    if (desktop_ms > 0.0f) {
        smooth(g_stats.desktop_ms, desktop_ms);
    }
    smooth(g_stats.frame_ms, frame_ms);
    g_frame_count++;

    const auto now = std::chrono::steady_clock::now();
    const float elapsed = std::chrono::duration<float>(now - g_rate_epoch).count();
    if (elapsed >= 1.0f) {
        g_stats.eye_hz = g_eye_pass_count / elapsed;
        g_stats.frame_hz = g_frame_count / elapsed;
        g_eye_pass_count = 0;
        g_frame_count = 0;
        g_rate_epoch = now;
    }
}

void vr_report_game_tick_ms(float tick_ms) {
    smooth(g_stats.tick_ms, tick_ms);
}

void vr_get_frame_stats(VrFrameStats* out) {
    if (out != nullptr) {
        *out = g_stats;
    }
}

// Close a frame opened by xrBeginFrame without submitting anything. The spec requires an xrEndFrame
// for every xrBeginFrame, with zero layers when shouldRender is false. A runtime that waits for the
// app's first ended frame before taking the session out of SYNCHRONIZED (where shouldRender is
// false) would otherwise never make it visible: the app shows as running, the headset shows nothing.
static void vr_end_frame_empty() {
    xr.frame_began = false;
    XrFrameEndInfo end_info = { XR_TYPE_FRAME_END_INFO };
    end_info.displayTime = xr.frame_state.predictedDisplayTime;
    end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    end_info.layerCount = 0;
    end_info.layers = nullptr;
    xr_check(xrEndFrame(xr.session, &end_info), "xrEndFrame (no layers)");
}

// CPU clock hint (gVrHighCpuClock, default on). The game thread is CPU-bound in busy scenes (Hyrule
// Field on a Quest 3: GPU ~25% busy while frames take 20-40 ms), and Meta's runtime raises the CPU
// clock only after frames have already been missed (log, October 6: 1382 MHz at the start of a dip,
// 1920 MHz a second later). Sustained high keeps it up, at some battery and heat cost; off hands the
// choice back to the runtime (sustained low, its default). Sent only when the setting changes.
static void vr_apply_perf_settings() {
    if (!xr.perf_settings_supported || xr.session == XR_NULL_HANDLE) {
        return;
    }
    const XrPerfSettingsLevelEXT want = CVarGetInteger("gVrHighCpuClock", 1) ? XR_PERF_SETTINGS_LEVEL_SUSTAINED_HIGH_EXT
                                                                            : XR_PERF_SETTINGS_LEVEL_SUSTAINED_LOW_EXT;
    if ((int)want == xr.perf_cpu_level_applied) {
        return;
    }
    const bool nothing_to_undo = xr.perf_cpu_level_applied < 0 && want == XR_PERF_SETTINGS_LEVEL_SUSTAINED_LOW_EXT;
    xr.perf_cpu_level_applied = (int)want; // failed or not, don't retry every frame
    if (nothing_to_undo) {
        return;
    }
    PFN_xrPerfSettingsSetPerformanceLevelEXT set_level = nullptr;
    xrGetInstanceProcAddr(xr.instance, "xrPerfSettingsSetPerformanceLevelEXT",
                          reinterpret_cast<PFN_xrVoidFunction*>(&set_level));
    if (set_level != nullptr &&
        xr_check(set_level(xr.session, XR_PERF_SETTINGS_DOMAIN_CPU_EXT, want), "xrPerfSettingsSetPerformanceLevelEXT")) {
        spdlog::info("[VR] CPU performance level: {}",
                     want == XR_PERF_SETTINGS_LEVEL_SUSTAINED_HIGH_EXT ? "sustained high" : "sustained low");
    }
}

// Headset refresh rate (gVrRefreshRate, Hz; 0 = the headset's default). Meta's runtimes run an
// immersive app at the rate it requests through XR_FB_display_refresh_rate, else their default
// (72 Hz on Quest). The rates offered are read once per session (vr_refresh_rate_init) and logged
// with the current one; a request is sent only when the setting changes, and only for an offered
// rate. The game's pacing follows the real rate on its own (predictedDisplayPeriod).
static void vr_refresh_rate_init() {
    xr.refresh_rates.clear();
    xr.refresh_rate_applied = -1.0f;
    if (!xr.refresh_rate_supported) {
        spdlog::info("[VR] Display refresh rate: the runtime offers no choice (no XR_FB_display_refresh_rate)");
        return;
    }
    PFN_xrEnumerateDisplayRefreshRatesFB enumerate = nullptr;
    PFN_xrGetDisplayRefreshRateFB get_rate = nullptr;
    xrGetInstanceProcAddr(xr.instance, "xrEnumerateDisplayRefreshRatesFB",
                          reinterpret_cast<PFN_xrVoidFunction*>(&enumerate));
    xrGetInstanceProcAddr(xr.instance, "xrGetDisplayRefreshRateFB", reinterpret_cast<PFN_xrVoidFunction*>(&get_rate));
    if (enumerate == nullptr) {
        spdlog::warn("[VR] Display refresh rate: xrEnumerateDisplayRefreshRatesFB not found");
        return;
    }
    uint32_t count = 0;
    XrResult res = enumerate(xr.session, 0, &count, nullptr);
    if (XR_FAILED(res) || count == 0) {
        spdlog::warn("[VR] Display refresh rate: enumerating failed (XrResult {}, {} rates)", (int)res, count);
        return;
    }
    xr.refresh_rates.resize(count);
    res = enumerate(xr.session, count, &count, xr.refresh_rates.data());
    if (XR_FAILED(res)) {
        spdlog::warn("[VR] Display refresh rate: enumerating failed (XrResult {})", (int)res);
        xr.refresh_rates.clear();
        return;
    }
    xr.refresh_rates.resize(count);
    std::string list;
    for (float hz : xr.refresh_rates) {
        list += (list.empty() ? "" : ", ") + std::to_string((int)(hz + 0.5f));
    }
    float current = 0.0f;
    if (get_rate != nullptr) {
        get_rate(xr.session, &current);
    }
    spdlog::info("[VR] Display refresh rates offered: {} Hz; current {:.0f} Hz", list, current);
}

static void vr_apply_refresh_rate() {
    if (xr.refresh_rates.empty() && xr.refresh_rate_supported && !g_refresh_rate_retried) {
        // Some runtimes only answer once the session is running.
        g_refresh_rate_retried = true;
        vr_refresh_rate_init();
    }
    if (xr.refresh_rates.empty() || xr.session == XR_NULL_HANDLE) {
        return;
    }
    float want = (float)CVarGetInteger("gVrRefreshRate", 0);
    if (want > 0.0f) {
        // Only an offered rate (the setting may come from another headset's config).
        bool offered = false;
        for (float hz : xr.refresh_rates) {
            offered |= fabsf(hz - want) < 0.5f;
            if (fabsf(hz - want) < 0.5f) {
                want = hz;
            }
        }
        if (!offered) {
            want = 0.0f;
        }
    }
    if (want == xr.refresh_rate_applied) {
        return;
    }
    const bool nothing_to_undo = xr.refresh_rate_applied < 0.0f && want == 0.0f;
    xr.refresh_rate_applied = want; // failed or not, don't retry every frame
    if (nothing_to_undo) {
        return;
    }
    PFN_xrRequestDisplayRefreshRateFB request = nullptr;
    xrGetInstanceProcAddr(xr.instance, "xrRequestDisplayRefreshRateFB", reinterpret_cast<PFN_xrVoidFunction*>(&request));
    // 0 = no preference: the runtime goes back to its default.
    if (request != nullptr && xr_check(request(xr.session, want), "xrRequestDisplayRefreshRateFB")) {
        if (want > 0.0f) {
            spdlog::info("[VR] Requested a {:.0f} Hz display", want);
        } else {
            spdlog::info("[VR] Display refresh rate back to the headset's default");
        }
    }
}

int vr_get_supported_refresh_rates(float* out, int max) {
    int n = 0;
    for (float hz : xr.refresh_rates) {
        if (n < max) {
            out[n++] = hz;
        }
    }
    return n;
}

bool vr_begin_frame() {
    if (!xr.initialized || !xr.enabled) return false;

    poll_events();

    if (!xr.session_running) return false;

    // Wait for the runtime to signal it's ready for a new frame. Time spent blocked here is spare
    // headroom — if it trends to zero we are no longer keeping up with the headset.
    xr.frame_state = { XR_TYPE_FRAME_STATE };
    XrFrameWaitInfo wait_info = { XR_TYPE_FRAME_WAIT_INFO };
    const auto wait_start = std::chrono::steady_clock::now();
    const XrResult wait_result = xrWaitFrame(xr.session, &wait_info, &xr.frame_state);
    smooth(g_stats.wait_ms, std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - wait_start).count());
    if (!xr_check(wait_result, "xrWaitFrame")) {
        return false;
    }

    // Derive the headset refresh rate from the nominal display period (nanoseconds). This paces
    // the game's fixed-timestep logic via the interpolation system (see GetInterpolationFPS).
    if (xr.frame_state.predictedDisplayPeriod > 0) {
        uint32_t hz = (uint32_t)(1.0e9 / (double)xr.frame_state.predictedDisplayPeriod + 0.5);
        if (hz >= 30 && hz <= 1000) {
            xr.refresh_rate = hz;
        }
    }

    XrFrameBeginInfo begin_info = { XR_TYPE_FRAME_BEGIN_INFO };
    if (!xr_check(xrBeginFrame(xr.session, &begin_info), "xrBeginFrame")) {
        return false;
    }
    xr.frame_began = true;

    // The caller only ends frames this returns true for, so every early-out past xrBeginFrame closes
    // the frame itself (vr_end_frame_empty).
    if (!xr.frame_state.shouldRender) {
        vr_end_frame_empty();
        return false;
    }

    // Live-tunable world scale (game units per real-world meter). Higher = the world feels smaller;
    // together with the game-unit head offsets this fully controls perceived height above the ground.
    {
        float ws = CVarGetFloat("gVrWorldScale", 35.0f);
        if (ws < 5.0f) ws = 5.0f;
        if (ws > 200.0f) ws = 200.0f;
        xr.world_scale = ws;
    }

    // Locate views (get per-eye pose and FOV)
    XrViewState view_state = { XR_TYPE_VIEW_STATE };
    XrViewLocateInfo view_locate_info = { XR_TYPE_VIEW_LOCATE_INFO };
    view_locate_info.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    view_locate_info.displayTime = xr.frame_state.predictedDisplayTime;
    view_locate_info.space = xr.local_space;

    uint32_t view_count = 2;
    XrResult result = xrLocateViews(xr.session, &view_locate_info, &view_state, 2, &view_count, xr.views);
    if (!XR_SUCCEEDED(result)) {
        spdlog::warn("[VR] xrLocateViews failed");
        vr_end_frame_empty();
        return false;
    }

    // Sync controllers + locate hand poses for this frame (motion controls).
    update_input();
    // The SoH menu panel takes its pointer input first and hides it from everything after.
    vr_menu_update();
    vr_apply_perf_settings();
    vr_apply_refresh_rate();

    // Auto world scale calibration: measure the player's physical eye height above the real floor
    // (STAGE space) and derive the scale that puts their eyes exactly at Link's eyes — which also
    // makes the game ground coincide with the real floor. Runs when requested (first-person entry,
    // manual recenter) or when Link's eye height changes materially (child <-> adult swap).
    if (CVarGetInteger("gVrAutoWorldScale", 1) && xr.stage_space != XR_NULL_HANDLE &&
        xr.link_eye_height_units > 1.0f) {
        const bool eye_height_changed =
            xr.scale_calibrated &&
            fabsf(xr.link_eye_height_units - xr.calibrated_link_eye_height) > 2.0f;
        if (xr.scale_recalibrate_requested || !xr.scale_calibrated || eye_height_changed) {
            XrSpaceLocation stage_loc = { XR_TYPE_SPACE_LOCATION };
            if (XR_SUCCEEDED(xrLocateSpace(xr.local_space, xr.stage_space, xr.frame_state.predictedDisplayTime,
                                           &stage_loc)) &&
                (stage_loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT)) {
                const float head_local_y = 0.5f * (xr.views[0].pose.position.y + xr.views[1].pose.position.y);
                const float eye_height_m = head_local_y + stage_loc.pose.position.y;
                // Sanity window: reject crouched/mistracked measurements rather than producing a
                // wild scale; the previous calibration (or the manual CVar) stays in effect.
                if (eye_height_m > 0.9f && eye_height_m < 2.4f) {
                    xr.auto_world_scale = xr.link_eye_height_units / eye_height_m;
                    xr.auto_world_scale = fminf(fmaxf(xr.auto_world_scale, 10.0f), 100.0f);
                    xr.calibrated_link_eye_height = xr.link_eye_height_units;
                    xr.scale_calibrated = true;
                    xr.scale_recalibrate_requested = false;
                    spdlog::info("[VR] World scale calibrated: {:.1f} units/m (eye {:.0f} units / {:.2f} m)",
                                 xr.auto_world_scale, xr.link_eye_height_units, eye_height_m);
                }
            }
        }
        if (xr.scale_calibrated) {
            xr.world_scale = xr.auto_world_scale;
        }
    }

    // Re-enable fixup: the player may have physically moved while playing flat, so map their
    // CURRENT position to Link's current body — otherwise the stale roomscale origin makes Link
    // glide off to wherever the player wandered. Needs this frame's freshly located views.
    if (xr.reenable_fixup) {
        xr.reenable_fixup = false;
        vr_reset_roomscale();
    }

    // Flat-screen panel placement: on entering a 2D context, drop the panel in front of the
    // player's current gaze. Uses the RAW located pose — quad layers are submitted in local_space
    // and never include the artificial snap-turn.
    if (xr.flat_screen && !xr.flat_screen_prev) {
        const XrPosef& vp = xr.views[0].pose;
        const glm::vec3 head(0.5f * (xr.views[0].pose.position.x + xr.views[1].pose.position.x),
                             0.5f * (xr.views[0].pose.position.y + xr.views[1].pose.position.y),
                             0.5f * (xr.views[0].pose.position.z + xr.views[1].pose.position.z));
        const glm::quat ho(vp.orientation.w, vp.orientation.x, vp.orientation.y, vp.orientation.z);
        glm::vec3 fwd = ho * glm::vec3(0.0f, 0.0f, -1.0f);
        fwd.y = 0.0f;
        const float len = glm::length(fwd);
        fwd = (len > 1e-4f) ? fwd / len : glm::vec3(0.0f, 0.0f, -1.0f);
        float dist = CVarGetFloat("gVrScreenDistance", 2.2f);
        if (dist < 0.5f) dist = 0.5f;
        const glm::vec3 pos = head + fwd * dist;
        // Yaw-only orientation, the quad's front (+Z) facing back at the player.
        const float qyaw = atan2f(-fwd.x, -fwd.z);
        const glm::quat q = glm::angleAxis(qyaw, glm::vec3(0.0f, 1.0f, 0.0f));
        xr.flat_pose.position = { pos.x, pos.y, pos.z };
        xr.flat_pose.orientation = { q.x, q.y, q.z, q.w };
    }
    xr.flat_screen_prev = xr.flat_screen;

    // Artificial turning (right stick X), the two styles every VR title offers: SNAP latches a
    // discrete turn on a threshold crossing (the stick must return to center before the next
    // snap fires); SMOOTH yaws continuously at headset rate while the stick is deflected past
    // the deadzone — analog by default (deflection past the deadzone scales the rate,
    // re-normalized so full tilt = full speed), constant-rate if preferred. Both run through
    // the same head-pivot turn accumulation, so the physics sim and every game-facing pose
    // compose identically. Suspended in flat-screen mode (right stick navigates menus) and in
    // third person (the stock game owns the camera and the right stick is pure C-buttons).
    if (xr.input_initialized && !xr.flat_screen && xr.first_person && !g_turn_suppressed &&
        CVarGetInteger("gVrSnapTurnOn", 1)) {
        static int snap_latch = 0;
        const float sx = xr.thumbstick_x[1];
        if (CVarGetInteger("gVrTurnStyle", 0) == 1) {
            const float dead = CVarGetFloat("gVrSmoothTurnDeadzone", 0.25f);
            const float mag = fabsf(sx);
            if (mag > dead) {
                float frac = 1.0f;
                if (CVarGetInteger("gVrSmoothTurnAnalog", 1)) {
                    frac = (mag - dead) / (1.0f - dead);
                }
                float dt = xr.frame_state.predictedDisplayPeriod > 0
                               ? (float)((double)xr.frame_state.predictedDisplayPeriod * 1e-9)
                               : 1.0f / (float)vr_get_refresh_rate();
                if (dt > 1.0f / 30.0f) {
                    dt = 1.0f / 30.0f; // a frame hitch must not lurch the world around
                }
                vr_apply_snap_turn((sx > 0.0f ? 1.0f : -1.0f) * frac *
                                   CVarGetFloat("gVrSmoothTurnSpeed", 120.0f) * dt);
            }
            snap_latch = 0;
        } else if (snap_latch == 0 && fabsf(sx) > 0.6f) {
            snap_latch = (sx > 0.0f) ? 1 : -1;
            // Latched, committed at the next game tick (see g_snap_pending_deg).
            g_snap_pending_deg += snap_latch * CVarGetFloat("gVrSnapTurnDegrees", 45.0f);
            g_snap_pending_frames = 0;
        } else if (snap_latch != 0 && fabsf(sx) < 0.3f) {
            snap_latch = 0;
        }
    }
    // Backstop: a game that stops ticking (load hitch, a state that never calls VR_GameTickBegin)
    // must not swallow the snap. Past ~0.2 s it applies here like it always used to.
    if (g_snap_pending_deg != 0.0f && ++g_snap_pending_frames > (int)(0.2f * (float)vr_get_refresh_rate())) {
        const float deg = g_snap_pending_deg;
        g_snap_pending_deg = 0.0f;
        g_snap_pending_frames = 0;
        if (xr.input_initialized && !xr.flat_screen && xr.first_person && !g_turn_suppressed) {
            vr_apply_snap_turn(deg);
        }
    }

    // Lock-on framing (Legaiaflame's Lock On): ease the world so the lock-on target stays in front
    // of the player. Runs AFTER artificial turning so it corrects against the turn the player just
    // asked for rather than fighting a stale heading. The deadzone is the whole design: inside that
    // cone the world is left completely alone, so glancing around costs nothing and there is no
    // constant micro-rotation to make anyone sick; only the excess past the cone is taken out, and
    // never faster than the configured rate. Yaw only — pitch and roll are the player's alone.
    if (xr.input_initialized && !xr.flat_screen && xr.first_person && g_lockon_ttl > 0.0f) {
        float dt = xr.frame_state.predictedDisplayPeriod > 0
                       ? (float)((double)xr.frame_state.predictedDisplayPeriod * 1e-9)
                       : 1.0f / (float)vr_get_refresh_rate();
        if (dt > 1.0f / 30.0f) {
            dt = 1.0f / 30.0f; // a frame hitch must not lurch the world around
        }
        g_lockon_ttl -= dt;

        int16_t heading = 0;
        if (vr_pending_heading_yaw(&heading)) {
            // Low-pass the stepped request into a continuous bearing. On the first frame of a lock
            // it is adopted outright — easing in from a stale bearing would swing the view through
            // an arc the player never asked for.
            const float target_deg = (float)g_lockon_yaw * (180.0f / 32768.0f);
            if (!g_lockon_smooth_valid) {
                g_lockon_smoothed_deg = target_deg;
                g_lockon_smooth_valid = true;
            } else {
                g_lockon_smoothed_deg = vr_wrap180(
                    g_lockon_smoothed_deg + vr_wrap180(target_deg - g_lockon_smoothed_deg) *
                                                (1.0f - expf(-dt / kLockOnInputTau)));
            }

            const float err_deg =
                vr_wrap180(g_lockon_smoothed_deg - (float)heading * (180.0f / 32768.0f));
            const float dead = CVarGetFloat("gVrLockOnDeadzone", 0.0f);
            float excess = 0.0f;
            if (err_deg > dead) {
                excess = err_deg - dead;
            } else if (err_deg < -dead) {
                excess = err_deg + dead;
            }
            if (excess != 0.0f) {
                // Slew-limited far away, eased close in. The proportional term is what removes the
                // stutter: the rotation rate becomes a function of how far off the target is, so a
                // steadily sweeping bearing produces steady motion instead of full-speed bursts
                // separated by dead stops. The speed slider stays a hard ceiling, which is what
                // keeps ACQUIRING a target (a 170 degree error) from whipping the view around.
                excess *= (1.0f - expf(-dt / kLockOnTrackTau));
                const float max_step = CVarGetFloat("gVrLockOnTurnSpeed", 120.0f) * dt;
                if (excess > max_step) {
                    excess = max_step;
                } else if (excess < -max_step) {
                    excess = -max_step;
                }
                // NEGATED, and the sign matters more than it looks: vr_apply_snap_turn takes
                // degrees to the player's RIGHT, and a right turn DECREASES the game's binang yaw
                // (yaw 0 faces +Z and increases toward +X, which is the player's left in this
                // frame). Closing a positive error therefore needs a negative right-turn. Get this
                // backwards and the loop becomes positive feedback: it drives the error away from
                // zero until it parks at the opposite fixed point, leaving the target exactly
                // behind the player's head.
                vr_apply_snap_turn(-excess);
            }
        }
    }

    // A one-shot "face this way" from the game (getting onto a ladder from above: turn to face it).
    // Same head-pivot rotation and the same sign convention as the lock-on correction above.
    if (g_face_yaw_pending) {
        g_face_yaw_pending = false;
        int16_t heading = 0;
        if (xr.input_initialized && !xr.flat_screen && xr.first_person && vr_pending_heading_yaw(&heading)) {
            const float err_deg = (float)(int16_t)(g_face_yaw - heading) * (180.0f / 32768.0f);
            vr_apply_snap_turn(-err_deg);
        }
    }

    // Apply the accumulated snap-turn to every game-facing pose, preserving the raw view poses for
    // layer submission in vr_end_frame. Hand poses are only adjusted when freshly located this frame
    // (a stale pose already carries the previous turn and would be double-rotated). The submit
    // pose/FOV are refreshed ONLY on frames whose eye images we are about to redraw: in flat-screen
    // mode, and on stereo-divisor reprojection frames, the projection layer keeps re-submitting the
    // last world frame described by the frustum it was rendered from, so the compositor reprojects
    // it correctly instead of stretching it onto a pose it never matched.
    const bool refresh_submit = !xr.flat_screen && xr.plan_render_eyes;
    // Centre eye in RAW tracking space, every frame (submit_pose only refreshes on redraw frames):
    // the soft-follow text panel is placed from where the head physically is.
    if ((view_state.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) &&
        (view_state.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT)) {
        const XrPosef& p0 = xr.views[0].pose;
        const XrPosef& p1 = xr.views[1].pose;
        glm::quat q0(p0.orientation.w, p0.orientation.x, p0.orientation.y, p0.orientation.z);
        glm::quat q1(p1.orientation.w, p1.orientation.x, p1.orientation.y, p1.orientation.z);
        if (glm::dot(q0, q1) < 0.0f) {
            q1 = -q1;
        }
        xr.head_rot_raw = glm::normalize(q0 + q1);
        xr.head_pos_raw = 0.5f * glm::vec3(p0.position.x + p1.position.x, p0.position.y + p1.position.y,
                                           p0.position.z + p1.position.z);
        xr.head_raw_valid = true;
    }
    for (int eye = 0; eye < 2; eye++) {
        if (refresh_submit || !xr.eyes_ever_rendered) {
            xr.submit_pose[eye] = xr.views[eye].pose;
            xr.submit_fov[eye] = xr.views[eye].fov;
        }
        xr.views[eye].pose = apply_turn(xr.views[eye].pose);
    }
    for (int h = 0; h < 2; h++) {
        if (xr.hand_active[h]) {
            // Keep the RAW tracking-space grip for compositor quads (hand-attached HUD): quad
            // layers are composed against live tracking and must not carry the artificial turn.
            xr.grip_pose_raw[h] = xr.grip_pose[h];
            xr.grip_pose[h] = apply_turn(xr.grip_pose[h]);
            xr.aim_pose[h] = apply_turn(xr.aim_pose[h]);
        }
    }

    // In-headset HUD editing (grab a wrist panel / element with the other hand). What it uses is
    // hidden from the game, which reads these values on its next tick.
    vr_hud_grab_update();
    for (int h = 0; h < 2; h++) {
        if (xr.grab.swallow[h]) {
            if (xr.grab.grabber == h) {
                xr.buttons[h] = 0;
                xr.squeeze_value[h] = 0.0f;
                xr.trigger_value[h] = 0.0f;
            } else {
                xr.buttons[h] &= ~((1 << 2) | (1 << 3)); // A/B or X/Y
            }
        }
    }

    // Physical-combat substrate: push this frame's RAW hand kinematics, then integrate one step
    // with the frame's world context (snap turn + the same blended anchor vr_get_hand_pose uses,
    // current because vr_set_interp_alpha runs just before vr_begin_frame). Raw in, context
    // alongside: hand history stays continuous across snap turns instead of spiking.
    {
        for (int h = 0; h < 2; h++) {
            if (!xr.hand_active[h]) {
                continue;
            }
            const XrPosef& rp = xr.grip_pose_raw[h];
            const float pos[3] = { rp.position.x, rp.position.y, rp.position.z };
            const float quat[4] = { rp.orientation.x, rp.orientation.y, rp.orientation.z, rp.orientation.w };
            const float lv[3] = { xr.hand_lin_vel[h].x, xr.hand_lin_vel[h].y, xr.hand_lin_vel[h].z };
            const float av[3] = { xr.hand_ang_vel[h].x, xr.hand_ang_vel[h].y, xr.hand_ang_vel[h].z };
            vrphys_push_hand_sample(h, pos, quat, lv, av, xr.hand_vel_valid[h],
                                    (uint64_t)xr.frame_state.predictedDisplayTime);
        }
        const float dt = xr.frame_state.predictedDisplayPeriod > 0
                             ? (float)((double)xr.frame_state.predictedDisplayPeriod * 1e-9)
                             : 1.0f / (float)vr_get_refresh_rate();
        const float turn_quat[4] = { g_turn_rot.x, g_turn_rot.y, g_turn_rot.z, g_turn_rot.w };
        const float turn_off[3] = { g_turn_off.x, g_turn_off.y, g_turn_off.z };
        const glm::vec3 anchor = (xr.first_person && xr.anchor_initialized)
                                     ? vr_anchor_now()
                                     : glm::vec3(0.0f);
        const float anchor_units[3] = { anchor.x, anchor.y, anchor.z };
        vrphys_step(dt, turn_quat, turn_off, anchor_units, xr.world_scale, xr.first_person);

        // Fire the contact haptics the sim just produced — same frame, zero game-tick latency.
        VrPhysHapticReq reqs[8];
        const int nreq = vrphys_take_haptic_requests(reqs, 8);
        for (int i = 0; i < nreq; i++) {
            vr_trigger_haptic(reqs[i].hand, reqs[i].amplitude01, reqs[i].freq_hz, reqs[i].duration_ms);
        }
    }

    // Build matrices for each eye
    for (int eye = 0; eye < 2; eye++) {
        build_projection_matrix(xr.views[eye].fov, xr.near_clip, xr.far_clip, xr.projection[eye]);
        pose_to_view_matrix(xr.views[eye].pose, xr.world_scale, xr.view[eye]);
    }
    vr_build_center_view();

    return true;
}

// Soft-follow placement of the text panel, in RAW local_space metres (a compositor quad must not
// carry the artificial turn, and real metres keep it the same size whatever the world scale or
// Link's age). The panel sits level at gVrTextDistance along the head's LEVEL forward,
// gVrTextHeight below/above the eyes, always facing the head about the vertical axis only — no
// pitch, no roll. It snaps into place when a text box opens; after that it stays put while the head
// looks around inside a deadzone, and once the player turns (or walks, or stands up) past it, it
// glides to the new front with a critically damped follow and stops dead when it arrives — no
// overshoot, and no perpetual drift from tracking noise.
static XrPosef text_panel_place(bool snap, float dt) {
    const glm::quat& hq = xr.head_rot_raw;
    const glm::vec3 head = xr.head_pos_raw;

    // Level forward that stays defined looking straight up or down: blend the forward vector's
    // horizontal part with the up vector's (which tips toward/away from the view direction as
    // the head pitches). Exact for pure pitch; roll only nudges it.
    const glm::vec3 f = hq * glm::vec3(0.0f, 0.0f, -1.0f);
    const glm::vec3 u = hq * glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 fh = glm::vec3(f.x, 0.0f, f.z) * u.y - glm::vec3(u.x, 0.0f, u.z) * f.y;
    if (glm::length(fh) > 1e-4f) {
        xr.text_fwd = glm::normalize(fh);
    } else if (glm::length(xr.text_fwd) < 0.5f) {
        xr.text_fwd = glm::vec3(0.0f, 0.0f, -1.0f);
    }

    const float dist = std::clamp(xr.text_layout[1] > 0.0f ? xr.text_layout[1] : CVarGetFloat("gVrTextDistance", 1.4f),
                                  0.4f, 6.0f);
    const float lift = xr.text_layout[0] > 0.0f ? xr.text_layout[2] : CVarGetFloat("gVrTextHeight", -0.15f);
    const glm::vec3 target = head + xr.text_fwd * dist + glm::vec3(0.0f, lift, 0.0f);

    if (snap) {
        xr.text_pos = target;
        xr.text_vel = glm::vec3(0.0f);
        xr.text_following = false;
    } else {
        const glm::vec3 to = xr.text_pos - head;
        const glm::vec3 to_h(to.x, 0.0f, to.z);
        const float to_h_len = glm::length(to_h);
        const float cos_err = to_h_len > 1e-4f ? glm::dot(to_h / to_h_len, xr.text_fwd) : -1.0f;
        const float err_deg = acosf(std::clamp(cos_err, -1.0f, 1.0f)) * (180.0f / 3.14159265358979323846f);
        const float gap = glm::length(target - xr.text_pos);

        if (err_deg > 100.0f || gap > 2.0f) {
            // Recentre, a teleport in tracking space, or a full turn-around: gliding a panel through
            // or around the player's head reads worse than simply re-placing it.
            xr.text_pos = target;
            xr.text_vel = glm::vec3(0.0f);
            xr.text_following = false;
        } else {
            if (!xr.text_following &&
                (err_deg > CVarGetFloat("gVrTextFollowDeg", 20.0f) || fabsf(xr.text_pos.y - target.y) > 0.15f ||
                 fabsf(to_h_len - dist) > 0.35f)) {
                xr.text_following = true;
            }
            if (xr.text_following) {
                // Critically damped spring toward the (moving) target: smooth start, no overshoot.
                const float tau = std::max(CVarGetFloat("gVrTextFollowSpeed", 0.25f), 0.02f);
                const float omega = 2.0f / tau;
                const float x = omega * dt;
                const float decay = 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
                const glm::vec3 change = xr.text_pos - target;
                const glm::vec3 temp = (xr.text_vel + omega * change) * dt;
                xr.text_vel = (xr.text_vel - omega * temp) * decay;
                xr.text_pos = target + (change + temp) * decay;
                // Arrived (2 cm is under a degree at reading distance): stop dead, so the panel is
                // perfectly still while reading instead of chasing head-tracking noise.
                if (glm::length(target - xr.text_pos) < 0.02f) {
                    xr.text_vel = glm::vec3(0.0f);
                    xr.text_following = false;
                }
            }
        }
    }

    // Face the head about the vertical axis only (level panel, level text). A quad's visible face
    // is its +Z; rotating +Z by yaw about Y gives (sin yaw, 0, cos yaw).
    const glm::vec3 back = head - xr.text_pos;
    const float yaw = atan2f(back.x, back.z);
    const glm::quat q = glm::angleAxis(yaw, glm::vec3(0.0f, 1.0f, 0.0f));

    XrPosef pose;
    pose.position = { xr.text_pos.x, xr.text_pos.y, xr.text_pos.z };
    pose.orientation = { q.x, q.y, q.z, q.w };
    return pose;
}

// A wrist panel's quad: pose in RAW local space (grip anchor + the profile's offset/rotation),
// metres per layout unit, and card size in layout units. false if the panel isn't shown.
static bool vr_wrist_panel_pose(int hand, XrPosef* pose, float* mpu, float* w_units, float* h_units) {
    if (!xr.hand_active[hand] || !xr.hud_block_valid[hand] || xr.hud_block_alpha[hand] <= 0.0f) {
        return false;
    }
    const float kDeg = 3.14159265358979323846f / 180.0f;
    const int child = vr_hud_profile();
    const VrHudPanelDefaults& pd = kVrHudPanelDefaults[child][hand];
    const float* b = xr.hud_block[hand];

    // Panel pose in the grip frame (cm, degrees): yaw about the grip's up, then pitch, then roll
    // about the panel's own normal.
    const XrPosef& gp = xr.grip_pose_raw[hand];
    const glm::quat gq(gp.orientation.w, gp.orientation.x, gp.orientation.y, gp.orientation.z);
    const glm::vec3 off(vr_hud_panel_f(child, hand, "X", pd.x), vr_hud_panel_f(child, hand, "Y", pd.y),
                        vr_hud_panel_f(child, hand, "Z", pd.z));
    const glm::vec3 c = glm::vec3(gp.position.x, gp.position.y, gp.position.z) + gq * (off * 0.01f);
    const glm::quat q =
        gq * glm::angleAxis(vr_hud_panel_f(child, hand, "Yaw", pd.yaw) * kDeg, glm::vec3(0.0f, 1.0f, 0.0f)) *
        glm::angleAxis(vr_hud_panel_f(child, hand, "Pitch", pd.pitch) * kDeg, glm::vec3(1.0f, 0.0f, 0.0f)) *
        glm::angleAxis(vr_hud_panel_f(child, hand, "Roll", pd.roll) * kDeg, glm::vec3(0.0f, 0.0f, 1.0f));
    pose->position = { c.x, c.y, c.z };
    pose->orientation = { q.x, q.y, q.z, q.w };
    // 0.8 mm per layout unit at 100%; the block is in native px, kWristPack px per unit.
    *mpu = 0.0008f * std::clamp(vr_hud_panel_f(child, hand, "Scale", pd.scale), 10.0f, 1000.0f) / 100.0f;
    *w_units = (b[2] - b[0]) / kWristPack;
    *h_units = (b[3] - b[1]) / kWristPack;
    return true;
}

static void vr_hud_set_f(const char* name, float v) {
    CVarSetFloat(name, v);
}

// A point in a panel's plane coordinates (metres from its centre, +x right, +y up).
static glm::vec2 vr_panel_local(const XrPosef& pose, const glm::vec3& w) {
    const glm::quat q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    const glm::vec3 l = glm::inverse(q) * (w - glm::vec3(pose.position.x, pose.position.y, pose.position.z));
    return glm::vec2(l.x, l.y);
}

// Distance from a point to a panel quad (0 when right on its face).
static float vr_panel_distance(const XrPosef& pose, float w, float h, const glm::vec3& p) {
    const glm::quat q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    const glm::vec3 l = glm::inverse(q) * (p - glm::vec3(pose.position.x, pose.position.y, pose.position.z));
    const float dx = std::max(fabsf(l.x) - w * 0.5f, 0.0f);
    const float dy = std::max(fabsf(l.y) - h * 0.5f, 0.0f);
    return sqrtf(dx * dx + dy * dy + l.z * l.z);
}

// In-headset HUD editing. Runs every XR frame after input is read and the raw grips are known,
// before the game reads input for its next tick, so what it swallows never reaches gameplay:
//   A / X + grip near the OTHER wrist's panel: grab the panel. It follows the grabbing hand
//     rigidly; its pose relative to its own wrist is written to the profile every frame.
//   B / Y + grip: grab the element nearest the hand on that panel and drag it across the face.
// While a hand is in reach of a panel its A/B/X/Y are swallowed; while it holds something all of
// its buttons, grip and trigger are. Release grip to drop (settings saved).
static void vr_hud_grab_update() {
    auto& g = xr.grab;
    g.hover[0] = g.hover[1] = 0;
    g.swallow[0] = g.swallow[1] = false;
    const bool enabled = CVarGetInteger("gVrHud.GrabEdit", 1) != 0 && vr_hud_wrist_layout() && !xr.hud_world &&
                         !xr.flat_screen && xr.hud_ever_rendered && xr.hud_commands != nullptr;
    if (!enabled) {
        g.grabber = -1;
        return;
    }
    const int child = vr_hud_profile();
    const float kRadToDeg = 180.0f / 3.14159265358979323846f;
    char name[96];

    // Holding: update, or drop on grip release.
    if (g.grabber >= 0) {
        const int h = g.grabber;
        const int o = g.owner;
        if (!xr.hand_active[h] || !xr.hand_active[o] || xr.squeeze_value[h] < 0.35f) {
            vr_trigger_haptic(h, 0.3f, 0.0f, 25.0f);
            g.grabber = -1;
            CVarSave();
            return;
        }
        g.swallow[h] = true;
        g.hover[o] = 2;
        const XrPosef& gp = xr.grip_pose_raw[h];
        const glm::quat gq(gp.orientation.w, gp.orientation.x, gp.orientation.y, gp.orientation.z);
        const glm::vec3 gpos(gp.position.x, gp.position.y, gp.position.z);
        if (g.element < 0) {
            // Panel: new world pose = grabber * rel, written back relative to the owner's grip.
            const glm::vec3 wp = gpos + gq * g.rel_pos;
            const glm::quat wq = glm::normalize(gq * g.rel_rot);
            const XrPosef& op = xr.grip_pose_raw[o];
            const glm::quat oq(op.orientation.w, op.orientation.x, op.orientation.y, op.orientation.z);
            const glm::quat oinv = glm::inverse(oq);
            const glm::vec3 off = oinv * (wp - glm::vec3(op.position.x, op.position.y, op.position.z)) * 100.0f;
            // local = RotY(yaw) RotX(pitch) RotZ(roll); R[r][c] = m[c][r].
            const glm::mat3 m = glm::mat3_cast(glm::normalize(oinv * wq));
            const float pitch = asinf(std::clamp(-m[2][1], -1.0f, 1.0f));
            const float roll = atan2f(m[0][1], m[1][1]);
            const float yaw = atan2f(m[2][0], m[2][2]);
            const char* fields[6] = { "X", "Y", "Z", "Yaw", "Pitch", "Roll" };
            const float vals[6] = { off.x, off.y, off.z, yaw * kRadToDeg, pitch * kRadToDeg, roll * kRadToDeg };
            for (int i = 0; i < 6; i++) {
                VrHudPanelCVar(name, sizeof(name), child, o, fields[i]);
                vr_hud_set_f(name, vals[i]);
            }
        } else {
            // Element: drag in the panel plane frozen at grab start, in layout units (y down).
            const glm::vec2 l = vr_panel_local(g.frame, gpos);
            const VrHudElementDesc* d = VrHudElementById(g.element);
            if (d != nullptr && g.mpu > 0.0f) {
                VrHudElementCVar(name, sizeof(name), child, d->key, "X");
                vr_hud_set_f(name, g.start_off[0] + (l.x - g.start_local.x) / g.mpu);
                VrHudElementCVar(name, sizeof(name), child, d->key, "Y");
                vr_hud_set_f(name, g.start_off[1] - (l.y - g.start_local.y) / g.mpu);
            }
        }
        return;
    }

    // Not holding: hover / grab start. A hand only reaches for the OTHER wrist's panel.
    static bool s_grip_was[2] = { false, false };
    for (int h = 0; h < 2; h++) {
        const int o = 1 - h;
        const bool grip = xr.squeeze_value[h] > 0.7f;
        const bool grip_edge = grip && !s_grip_was[h];
        s_grip_was[h] = grip;
        if (!xr.hand_active[h]) {
            continue;
        }
        XrPosef pose;
        float mpu, wu, hu;
        if (!vr_wrist_panel_pose(o, &pose, &mpu, &wu, &hu)) {
            continue;
        }
        const XrPosef& gp = xr.grip_pose_raw[h];
        const glm::vec3 gpos(gp.position.x, gp.position.y, gp.position.z);
        if (vr_panel_distance(pose, wu * mpu, hu * mpu, gpos) > 0.12f) {
            continue;
        }
        g.hover[o] = std::max(g.hover[o], 1);
        const bool mod_panel = (xr.buttons[h] & (1 << 2)) != 0;   // A (right) / X (left)
        const bool mod_element = (xr.buttons[h] & (1 << 3)) != 0; // B (right) / Y (left)
        if (mod_panel || mod_element) {
            g.swallow[h] = true;
        }
        if (!grip_edge || !(mod_panel || mod_element)) {
            continue;
        }

        const glm::quat gq(gp.orientation.w, gp.orientation.x, gp.orientation.y, gp.orientation.z);
        g.owner = o;
        g.element = -1;
        if (mod_panel) {
            const glm::quat pq(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
            const glm::vec3 pp(pose.position.x, pose.position.y, pose.position.z);
            g.rel_pos = glm::inverse(gq) * (pp - gpos);
            g.rel_rot = glm::normalize(glm::inverse(gq) * pq);
        } else {
            // The element under (or nearest) the hand, in card units (origin top-left, y down).
            const glm::vec2 l = vr_panel_local(pose, gpos);
            const float ux = l.x / mpu + wu * 0.5f;
            const float uy = hu * 0.5f - l.y / mpu;
            float best = 30.0f; // units
            for (int el = 0; el < VR_HUD_EL_COUNT; el++) {
                const VrHudElementDesc* d = VrHudElementById(el);
                if (d == nullptr || d->hand != o || !xr.hud_placed[el] || !xr.hud_rect_valid[el]) {
                    continue;
                }
                const float x0 = (xr.hud_slot[el][0] - o * kWristCanvasW) / kWristPack;
                const float y0 = xr.hud_slot[el][1] / kWristPack;
                const float s = xr.hud_slot[el][2] / kWristPack;
                const float x1 = x0 + (xr.hud_rect[el][2] - xr.hud_rect[el][0]) * s;
                const float y1 = y0 + (xr.hud_rect[el][3] - xr.hud_rect[el][1]) * s;
                const float dx = std::max(std::max(x0 - ux, ux - x1), 0.0f);
                const float dy = std::max(std::max(y0 - uy, uy - y1), 0.0f);
                const float dist = sqrtf(dx * dx + dy * dy);
                if (dist < best) {
                    best = dist;
                    g.element = el;
                }
            }
            if (g.element < 0) {
                continue;
            }
            const VrHudElementDesc* d = VrHudElementById(g.element);
            g.frame = pose;
            g.mpu = mpu;
            g.start_local = l;
            VrHudElementCVar(name, sizeof(name), child, d->key, "X");
            g.start_off[0] = CVarGetFloat(name, d->x[child]);
            VrHudElementCVar(name, sizeof(name), child, d->key, "Y");
            g.start_off[1] = CVarGetFloat(name, d->y[child]);
        }
        g.grabber = h;
        g.swallow[h] = true;
        g.hover[o] = 2;
        vr_trigger_haptic(h, 0.6f, 0.0f, 40.0f);
        return;
    }
}

// A game-world pose (position in game units, facing = RotateY(yaw)) as a RAW tracking-space pose
// for a compositor quad: the exact inverse of vr_get_head_matrix's composition
// (world = anchor + unyaw(gamma) * turned * world_scale, turned = turn * raw), with the same
// interpolated anchor and gamma this frame renders with, so the quad sits on the world geometry.
static XrPosef game_world_to_raw(const glm::vec3& p, float yaw) {
    glm::vec3 anchor(0.0f);
    float g = 0.0f;
    if (xr.anchor_initialized) {
        anchor = vr_anchor_now();
        g = xr.anchor_gamma_prev + (xr.anchor_gamma - xr.anchor_gamma_prev) * xr.interp_alpha;
    }
    const float cg = cosf(g), sg = sinf(g);
    const glm::mat3 unyaw(glm::vec3(cg, 0.0f, sg), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(-sg, 0.0f, cg));
    const glm::mat3 reyaw = glm::transpose(unyaw);
    const float ws = xr.world_scale > 0.0f ? xr.world_scale : 1.0f;
    const glm::quat turn_inv = glm::inverse(g_turn_rot);

    const glm::vec3 turned = reyaw * (p - anchor) / ws;
    const glm::vec3 raw = turn_inv * (turned - g_turn_off);
    const glm::quat q_world = glm::angleAxis(yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::quat q_raw = glm::normalize(turn_inv * glm::quat_cast(reyaw) * q_world);

    XrPosef pose;
    pose.position = { raw.x, raw.y, raw.z };
    pose.orientation = { q_raw.x, q_raw.y, q_raw.z, q_raw.w };
    return pose;
}

void vr_end_frame() {
    if (!xr.frame_began) return;
    xr.frame_began = false;

    XrCompositionLayerProjectionView projection_views[2] = {};
    for (int eye = 0; eye < 2; eye++) {
        projection_views[eye] = { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW };
        // Submit the RAW physical pose, not the turn-adjusted one — reprojection must compare
        // against where the player's head actually is, or the compositor would fight the snap turn.
        projection_views[eye].pose = xr.submit_pose[eye];
        projection_views[eye].fov = xr.submit_fov[eye];
        // Multiview: both eyes are layers of the one swapchain in eye_swapchains[0].
        projection_views[eye].subImage.swapchain = xr.multiview ? xr.eye_swapchains[0].handle
                                                                : xr.eye_swapchains[eye].handle;
        projection_views[eye].subImage.imageRect.offset = { 0, 0 };
        projection_views[eye].subImage.imageRect.extent = {
            static_cast<int32_t>(xr.eye_swapchains[eye].width),
            static_cast<int32_t>(xr.eye_swapchains[eye].height)
        };
        projection_views[eye].subImage.imageArrayIndex = xr.multiview ? (uint32_t)eye : 0;
    }

    XrCompositionLayerProjection projection_layer = { XR_TYPE_COMPOSITION_LAYER_PROJECTION };
    projection_layer.space = xr.local_space;
    projection_layer.viewCount = 2;
    projection_layer.views = projection_views;

    // Alyx-style in-wall fade: darken the WORLD layer at the compositor while the player's head is
    // inside geometry (the head is never pushed back — golden rule of VR cameras). Smoothed here
    // at XR frame rate so the 20 Hz game-side target reads as a clean ~100 ms fade. Menus/HUD
    // layers stay at full brightness.
    XrCompositionLayerColorScaleBiasKHR color_scale = { XR_TYPE_COMPOSITION_LAYER_COLOR_SCALE_BIAS_KHR };
    {
        const float step = 0.12f;
        if (xr.view_fade_current < xr.view_fade_target) {
            xr.view_fade_current = fminf(xr.view_fade_current + step, xr.view_fade_target);
        } else {
            xr.view_fade_current = fmaxf(xr.view_fade_current - step, xr.view_fade_target);
        }
        if (xr.color_scale_supported && xr.view_fade_current > 0.001f) {
            const float s = 1.0f - xr.view_fade_current;
            color_scale.colorScale = { s, s, s, 1.0f };
            color_scale.colorBias = { 0.0f, 0.0f, 0.0f, 0.0f };
            color_scale.next = nullptr;
            projection_layer.next = &color_scale;
        }
    }

    // Classic HUD quad layer (alpha-blended): the whole HUD head-locked in front of the eyes. The
    // hand-attached classic modes were removed; the wrist layout below replaces them.
    XrCompositionLayerQuad hud_layer = { XR_TYPE_COMPOSITION_LAYER_QUAD };
    hud_layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    hud_layer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    hud_layer.subImage.swapchain = xr.hud_swapchain.handle;
    hud_layer.subImage.imageRect.offset = { 0, 0 };
    hud_layer.subImage.imageRect.extent = {
        static_cast<int32_t>(xr.hud_swapchain.width),
        static_cast<int32_t>(xr.hud_swapchain.height)
    };
    hud_layer.subImage.imageArrayIndex = 0;

    hud_layer.space = xr.view_space;
    hud_layer.pose = { { 0, 0, 0, 1 },
                       { CVarGetFloat("gVrHudOffX", 0.0f), CVarGetFloat("gVrHudOffY", 0.0f),
                         -CVarGetFloat("gVrHudDistance", 2.0f) } };
    float hud_width = CVarGetFloat("gVrHudSize", 1.5f);
    if (hud_width < 0.05f) {
        hud_width = 0.05f;
    }
    hud_layer.size = { hud_width, hud_width * 0.75f }; // 4:3, matching the HUD swapchain

    // Wrist layout: crops of the same HUD image, one stack per controller — vitals on the LEFT,
    // buttons over the minimap on the RIGHT (physical hands by design, not handedness). Each crop
    // is sized at a fixed metres-per-native-pixel, so every element keeps the same real size
    // whatever the crop; each panel is placed on its grip by the per-profile gVrHud.<Adult|Child>.
    // <Left|Right>Panel.* settings (vr_hud_settings.h). A hand that isn't tracked shows
    // nothing (no head-locked fallback), a group that's faded out or not drawn drops out of its stack.
    // World-space pause: the whole HUD image on one world-anchored quad framing the front page
    // (VR_SetHudWorldPanel); replaces both the wrist stacks and the classic quad meanwhile.
    XrCompositionLayerQuad hud_world_layer = { XR_TYPE_COMPOSITION_LAYER_QUAD };
    const bool hud_world = xr.hud_world && xr.hud_ever_rendered && xr.hud_commands != nullptr;
    if (hud_world) {
        const float ws = xr.world_scale > 0.0f ? xr.world_scale : 1.0f;
        hud_world_layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
        hud_world_layer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        hud_world_layer.space = xr.local_space;
        hud_world_layer.subImage.swapchain = xr.hud_swapchain.handle;
        hud_world_layer.subImage.imageRect.offset = { 0, 0 };
        hud_world_layer.subImage.imageRect.extent = { (int32_t)xr.hud_swapchain.width,
                                                      (int32_t)xr.hud_swapchain.height };
        hud_world_layer.subImage.imageArrayIndex = 0;
        hud_world_layer.pose = game_world_to_raw(xr.hud_world_center, xr.hud_world_yaw);
        float canvas[3];
        vr_hud_world_canvas(canvas);
        hud_world_layer.size = { std::max(xr.hud_world_size[0] * canvas[0] / ws, 0.01f),
                                 std::max(xr.hud_world_size[1] * canvas[1] / ws, 0.01f) };
    }

    XrCompositionLayerQuad wrist_layers[2];
    uint32_t wrist_count = 0;
    const bool wrist_layout = vr_hud_wrist_layout();
    if (wrist_layout && !xr.hud_world && xr.hud_ever_rendered && xr.hud_commands != nullptr) {
        const auto& sc = xr.hud_swapchain;
        const float sx = sc.width / 320.0f, sy = sc.height / 240.0f;
        for (int hand = 0; hand < 2; hand++) {
            if (!xr.hand_active[hand] || !xr.hud_block_valid[hand] || xr.hud_block_alpha[hand] <= 0.0f) {
                continue;
            }
            const float* b = xr.hud_block[hand];
            XrPosef pose;
            float mpu, w_units, h_units;
            if (!vr_wrist_panel_pose(hand, &pose, &mpu, &w_units, &h_units)) {
                continue;
            }

            XrCompositionLayerQuad& L = wrist_layers[wrist_count++];
            L = { XR_TYPE_COMPOSITION_LAYER_QUAD };
            L.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
            L.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
            L.space = xr.local_space;
            L.subImage.swapchain = sc.handle;
            L.subImage.imageRect.offset = { (int32_t)(b[0] * sx), (int32_t)(b[1] * sy) };
            L.subImage.imageRect.extent = { std::max((int32_t)((b[2] - b[0]) * sx + 0.5f), 1),
                                            std::max((int32_t)((b[3] - b[1]) * sy + 0.5f), 1) };
            L.subImage.imageArrayIndex = 0;
            L.pose = pose;
            L.size = { std::max(w_units * mpu, 0.005f), std::max(h_units * mpu, 0.005f) };
        }
    }

    // Flat-screen quad (world-locked panel with the whole 2D frame: file select, pause menu)
    XrCompositionLayerQuad screen_layer = { XR_TYPE_COMPOSITION_LAYER_QUAD };
    screen_layer.space = xr.local_space;
    screen_layer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    screen_layer.subImage.swapchain = xr.screen_swapchain.handle;
    screen_layer.subImage.imageRect.offset = { 0, 0 };
    screen_layer.subImage.imageRect.extent = {
        static_cast<int32_t>(xr.screen_swapchain.width),
        static_cast<int32_t>(xr.screen_swapchain.height)
    };
    screen_layer.subImage.imageArrayIndex = 0;
    screen_layer.pose = xr.flat_pose;
    {
        float sw = CVarGetFloat("gVrScreenSize", 2.4f);
        if (sw < 0.5f) sw = 0.5f;
        screen_layer.size = { sw, sw * 0.75f }; // 4:3, matching the swapchain
    }

    // Text panel quad: only while a text box is showing (the game hands over a list only then, and
    // never in flat-screen contexts, where text stays in the panel's frame), and only once this
    // box's image exists. The imageRect is cropped to the region the game reported (the text box,
    // plus the staff while the ocarina is out), so the panel IS the box: its width is the setting,
    // its height follows the crop's aspect.
    XrCompositionLayerQuad text_layer = { XR_TYPE_COMPOSITION_LAYER_QUAD };
    const bool text_visible = xr.text_commands != nullptr && xr.text_has_image && xr.head_raw_valid;
    if (text_visible) {
        const auto& sc = xr.text_swapchain;
        const float fw = (float)sc.width;
        const float fh = (float)sc.height;
        int32_t x0 = (int32_t)(std::clamp(xr.text_crop[0], 0.0f, 1.0f) * fw);
        int32_t y0 = (int32_t)(std::clamp(xr.text_crop[1], 0.0f, 1.0f) * fh);
        int32_t x1 = (int32_t)(std::clamp(xr.text_crop[2], 0.0f, 1.0f) * fw + 0.5f);
        int32_t y1 = (int32_t)(std::clamp(xr.text_crop[3], 0.0f, 1.0f) * fh + 0.5f);
        if (x1 - x0 < 8 || y1 - y0 < 8) { // degenerate report: show the whole frame
            x0 = 0;
            y0 = 0;
            x1 = (int32_t)sc.width;
            y1 = (int32_t)sc.height;
        }

        const float dt = xr.frame_state.predictedDisplayPeriod > 0
                             ? std::min((float)((double)xr.frame_state.predictedDisplayPeriod * 1e-9), 1.0f / 30.0f)
                             : 1.0f / (float)vr_get_refresh_rate();
        const float tw = std::clamp(xr.text_layout[0] > 0.0f ? xr.text_layout[0] : CVarGetFloat("gVrTextWidth", 0.9f),
                                    0.2f, 5.0f);

        text_layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
        text_layer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        text_layer.space = xr.local_space;
        text_layer.subImage.swapchain = sc.handle;
        text_layer.subImage.imageRect.offset = { x0, y0 };
        text_layer.subImage.imageRect.extent = { x1 - x0, y1 - y0 };
        text_layer.subImage.imageArrayIndex = 0;
        text_layer.pose = text_panel_place(!xr.text_was_visible, dt);
        text_layer.size = { tw, tw * (float)(y1 - y0) / (float)(x1 - x0) };
    }
    xr.text_was_visible = text_visible;

    // SoH menu panel, over everything (it is modal while open), and its laser beam: a thin strip
    // from the pointing hand to the hit point (or 2.5 m out), turned about its own axis to face
    // the head so it reads as a line from any angle.
    XrCompositionLayerQuad menu_layer = { XR_TYPE_COMPOSITION_LAYER_QUAD };
    XrCompositionLayerQuad beam_layer = { XR_TYPE_COMPOSITION_LAYER_QUAD };
    const bool menu_visible = g_menu.open && xr.menu_ever_rendered;
    bool beam_visible = false;
    if (menu_visible) {
        menu_layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
        menu_layer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        menu_layer.space = xr.local_space;
        menu_layer.subImage.swapchain = xr.menu_swapchain.handle;
        menu_layer.subImage.imageRect.offset = { 0, 0 };
        menu_layer.subImage.imageRect.extent = { (int32_t)xr.menu_swapchain.width, (int32_t)xr.menu_swapchain.height };
        menu_layer.pose = g_menu.pose;
        menu_layer.size = { g_menu.size[0], g_menu.size[1] };

        const int bh = g_menu.hand;
        if (xr.beam_ready && bh >= 0 && xr.hand_active[bh] && xr.head_raw_valid) {
            const XrPosef& a = xr.aim_pose_raw[bh];
            const glm::vec3 o(a.position.x, a.position.y, a.position.z);
            const glm::vec3 d = glm::normalize(
                glm::quat(a.orientation.w, a.orientation.x, a.orientation.y, a.orientation.z) *
                glm::vec3(0.0f, 0.0f, -1.0f));
            const float len = g_menu.hit ? std::max(g_menu.hit_dist, 0.02f) : 2.5f;
            const glm::vec3 mid = o + d * (0.5f * len);
            glm::vec3 z = xr.head_pos_raw - mid;
            z -= glm::dot(z, d) * d;
            if (glm::length(z) < 1e-4f) {
                z = glm::abs(d.y) < 0.9f ? glm::cross(d, glm::vec3(0.0f, 1.0f, 0.0f))
                                         : glm::cross(d, glm::vec3(1.0f, 0.0f, 0.0f));
            }
            z = glm::normalize(z);
            const glm::vec3 x = glm::cross(d, z);
            const glm::quat q = glm::quat_cast(glm::mat3(x, d, z));
            beam_layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
            beam_layer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
            beam_layer.space = xr.local_space;
            beam_layer.subImage.swapchain = xr.beam_swapchain.handle;
            beam_layer.subImage.imageRect.offset = { 0, 0 };
            beam_layer.subImage.imageRect.extent = { (int32_t)xr.beam_swapchain.width,
                                                     (int32_t)xr.beam_swapchain.height };
            beam_layer.pose.position = { mid.x, mid.y, mid.z };
            beam_layer.pose.orientation = { q.x, q.y, q.z, q.w };
            beam_layer.size = { 0.006f, len };
            beam_visible = true;
        }
    }

    // Assemble layers back-to-front. The projection (world) layer is only submitted once its
    // swapchains have ever been rendered (at boot we go straight into flat-screen file select).
    const XrCompositionLayerBaseHeader* layers[10];
    uint32_t layer_count = 0;
    if (xr.eyes_ever_rendered) {
        layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection_layer);
    }
    // Desktop mirror: remember every quad exactly as the compositor gets it.
    xr.mirror_quad_count = 0;
    auto mirror_rec = [&](const XrCompositionLayerQuad& L, int src) {
        if (xr.mirror_quad_count < 8) {
            auto& m = xr.mirror_quads[xr.mirror_quad_count++];
            m.src = src;
            m.pose = L.pose;
            m.size = L.size;
            m.rect = L.subImage.imageRect;
        }
    };
    if (xr.flat_screen && xr.screen_ever_rendered) {
        layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&screen_layer);
        mirror_rec(screen_layer, 2);
    }
    // Same guard as the projection layer: the HUD quad's swapchain is uninitialised until the
    // first HUD pass, and with per-tick HUD rendering that may be a few frames in. Also skip while
    // the game has detached the overlay (hud_commands NULL — flat-screen contexts route it into
    // the panel instead), so a stale HUD image doesn't float over the pause menu.
    if (hud_world) {
        layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&hud_world_layer);
        mirror_rec(hud_world_layer, 0);
    } else if (xr.hud_world) {
        // Panel requested but no HUD image this frame: show nothing rather than the wrists.
    } else if (wrist_layout) {
        for (uint32_t i = 0; i < wrist_count; i++) {
            layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&wrist_layers[i]);
            mirror_rec(wrist_layers[i], 0);
        }
    } else if (xr.hud_ever_rendered && xr.hud_commands != nullptr) {
        layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&hud_layer);
        if (hud_layer.space == xr.local_space) {
            mirror_rec(hud_layer, 0);
        } else {
            // Head-locked classic HUD (view space): the same offset from the mirrored eye,
            // expressed in local space.
            XrCompositionLayerQuad L = hud_layer;
            const XrPosef& e = xr.submit_pose[0];
            const glm::quat eq(e.orientation.w, e.orientation.x, e.orientation.y, e.orientation.z);
            const glm::vec3 p = glm::vec3(e.position.x, e.position.y, e.position.z) +
                                eq * glm::vec3(L.pose.position.x, L.pose.position.y, L.pose.position.z);
            const glm::quat lq(L.pose.orientation.w, L.pose.orientation.x, L.pose.orientation.y,
                               L.pose.orientation.z);
            const glm::quat q = eq * lq;
            L.pose.position = { p.x, p.y, p.z };
            L.pose.orientation = { q.x, q.y, q.z, q.w };
            mirror_rec(L, 0);
        }
    }
    // Text last: it is what the player is reading, so it wins over the HUD if they ever overlap.
    if (text_visible) {
        layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&text_layer);
        mirror_rec(text_layer, 1);
    }
    // The menu (no mirror record: the desktop draws the same ImGui frame itself), beam on top.
    if (menu_visible) {
        layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&menu_layer);
    }
    if (beam_visible) {
        layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&beam_layer);
    }

    // Sub-image rects are computed top-left (and recorded that way for the mirror above). A backend
    // whose runtime reads them bottom-left (vr_gfx.h SubImageYUp) gets them flipped here.
    if (xr.gfx && xr.gfx->SubImageYUp()) {
        for (uint32_t i = 0; i < layer_count; i++) {
            if (layers[i]->type != XR_TYPE_COMPOSITION_LAYER_QUAD) {
                continue;
            }
            auto* q = const_cast<XrCompositionLayerQuad*>(reinterpret_cast<const XrCompositionLayerQuad*>(layers[i]));
            const XrSwapchain h = q->subImage.swapchain;
            const uint32_t img_h = h == xr.text_swapchain.handle     ? xr.text_swapchain.height
                                   : h == xr.screen_swapchain.handle ? xr.screen_swapchain.height
                                   : h == xr.menu_swapchain.handle   ? xr.menu_swapchain.height
                                   : h == xr.beam_swapchain.handle   ? xr.beam_swapchain.height
                                                                     : xr.hud_swapchain.height;
            q->subImage.imageRect.offset.y =
                (int32_t)img_h - (q->subImage.imageRect.offset.y + q->subImage.imageRect.extent.height);
        }
    }

    XrFrameEndInfo end_info = { XR_TYPE_FRAME_END_INFO };
    end_info.displayTime = xr.frame_state.predictedDisplayTime;
    end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;

    if (xr.frame_state.shouldRender) {
        end_info.layerCount = layer_count;
        end_info.layers = layers;
    } else {
        end_info.layerCount = 0;
        end_info.layers = nullptr;
    }

    xr_check(xrEndFrame(xr.session, &end_info), "xrEndFrame");
}

// --------------------------------------------------------------------------
// Per-eye
// --------------------------------------------------------------------------

static vrgfx::Target vr_eye_target(int eye) {
    if (xr.multiview) {
        return vrgfx::Target::EyeArray;
    }
    return eye == 0 ? vrgfx::Target::Eye0 : vrgfx::Target::Eye1;
}

// The acquired image of the eye pass in progress (a multiview pass has one image for both eyes).
static uint32_t vr_current_eye_image() {
    return xr.current_image_index[xr.multiview_pass ? 0 : (xr.current_eye & 1)];
}

// Dev test card (gVrGfxTestCard, VR-MULTI-API-PLAN.md 9.1), drawn into a target after the game:
// corner patches TL red / TR green / BL blue / BR white, a 16-step grey ramp (0..255) along the
// top, and a white arrow pointing up in the centre. One look answers orientation (incl. the wrist
// sub-rects and the text crop), gamma (the ramp vs the same card on another API) and mirror flip.
// Built from ClearColorRects only, so every backend gets it for free.
static void vr_draw_test_card(vrgfx::Target t, uint32_t image, uint32_t w, uint32_t h) {
    if (!CVarGetInteger("gVrGfxTestCard", 0) || w < 32 || h < 32) {
        return;
    }
    const int32_t W = (int32_t)w, H = (int32_t)h;
    auto fill = [&](int32_t x, int32_t y, int32_t rw, int32_t rh, float r, float g, float b) {
        const vrgfx::Rect rc = { x, y, rw, rh };
        const float c[4] = { r, g, b, 1.0f };
        xr.gfx->ClearColorRects(t, image, &rc, 1, c);
    };
    const int32_t p = std::min(W, H) / 8;
    fill(0, 0, p, p, 1.0f, 0.0f, 0.0f);         // top-left red
    fill(W - p, 0, p, p, 0.0f, 1.0f, 0.0f);     // top-right green
    fill(0, H - p, p, p, 0.0f, 0.0f, 1.0f);     // bottom-left blue
    fill(W - p, H - p, p, p, 1.0f, 1.0f, 1.0f); // bottom-right white
    for (int i = 0; i < 16; i++) {
        const int32_t x0 = p + (W - 2 * p) * i / 16;
        const int32_t x1 = p + (W - 2 * p) * (i + 1) / 16;
        const float v = (float)(i * 17) / 255.0f; // byte values 0, 17, ..., 255 (stored verbatim)
        fill(x0, 0, x1 - x0, p / 2, v, v, v);
    }
    const int32_t cx = W / 2, cy = H / 2, sz = std::min(W, H) / 3;
    const int32_t rows = 12;
    for (int k = 0; k < rows; k++) { // arrow head: tip at the top, widening downwards
        const int32_t rw = std::max<int32_t>(1, sz * (k + 1) / rows);
        fill(cx - rw / 2, cy - sz / 2 + (sz / 2) * k / rows, rw, std::max<int32_t>(1, (sz / 2) / rows + 1), 1.0f,
             1.0f, 1.0f);
    }
    fill(cx - sz / 10, cy, sz / 5, sz / 2, 1.0f, 1.0f, 1.0f); // stem
}

void vr_begin_eye(int eye) {
    if (!xr.initialized) return;
    xr.current_eye = eye;
    xr.eyes_ever_rendered = true;

    auto& sc = xr.eye_swapchains[eye];

    // Acquire swapchain image
    XrSwapchainImageAcquireInfo acquire_info = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
    uint32_t image_index = 0;
    xr_check(xrAcquireSwapchainImage(sc.handle, &acquire_info, &image_index), "xrAcquireSwapchainImage");
    xr.current_image_index[eye] = image_index;

    // Wait for it to be ready
    XrSwapchainImageWaitInfo wait_info = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
    wait_info.timeout = XR_INFINITE_DURATION;
    xr_check(xrWaitSwapchainImage(sc.handle, &wait_info), "xrWaitSwapchainImage");

    // Bind, clear (opaque black, depth 1.0), full viewport, renderer told the target size.
    const float clear_color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    xr.gfx->BeginPass(vr_eye_target(eye), image_index, clear_color);
    // Interpreter renders at the eye texture's size (replaces the old gfx_start_frame override)
    vr_apply_dimensions(sc.width, sc.height);
}

void vr_end_eye(int eye) {
    if (!xr.initialized) return;

    // Grab the left eye for the desktop mirror while its swapchain image is still acquired — once
    // released below, the runtime owns the texture again and it's no longer safe to read. Skipped
    // on frames the companion window isn't presenting: this is a full-eye-resolution CopyResource
    // (tens of MB) and nothing would consume the result.
    auto& sc = xr.eye_swapchains[eye];
    vr_draw_test_card(vr_eye_target(eye), xr.current_image_index[eye], sc.width, sc.height);
    if (eye == 0 && xr.plan_present_desktop) {
        vr_capture_mirror();
    }
    xr.gfx->EndPass(vr_eye_target(eye), xr.current_image_index[eye]);

    XrSwapchainImageReleaseInfo release_info = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    xr_check(xrReleaseSwapchainImage(sc.handle, &release_info), "xrReleaseSwapchainImage");
}

// Single-pass stereo: one interpreter run draws both layers of the eye swapchain from the center
// camera (vr_build_center_view); the renderer's multiview programs place each vertex per layer.
bool vr_multiview_enabled() {
    return xr.initialized && xr.multiview;
}

void vr_begin_stereo() {
    if (!xr.initialized || !xr.multiview) return;
    xr.current_eye = kCenterEye;
    xr.multiview_pass = true;
    xr.eyes_ever_rendered = true;
    xr.mv_welded = false;
    xr.mv_gen++;

    auto& sc = xr.eye_swapchains[0];
    XrSwapchainImageAcquireInfo acquire_info = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
    uint32_t image_index = 0;
    xr_check(xrAcquireSwapchainImage(sc.handle, &acquire_info, &image_index), "xrAcquireSwapchainImage");
    xr.current_image_index[0] = xr.current_image_index[1] = image_index;

    XrSwapchainImageWaitInfo wait_info = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
    wait_info.timeout = XR_INFINITE_DURATION;
    xr_check(xrWaitSwapchainImage(sc.handle, &wait_info), "xrWaitSwapchainImage");

    const float clear_color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    xr.gfx->BeginPass(vrgfx::Target::EyeArray, image_index, clear_color);
    vr_apply_dimensions(sc.width, sc.height);
}

void vr_end_stereo() {
    if (!xr.initialized || !xr.multiview_pass) return;
    auto& sc = xr.eye_swapchains[0];
    const uint32_t image = xr.current_image_index[0];
    vr_draw_test_card(vrgfx::Target::EyeArray, image, sc.width, sc.height);
    if (xr.plan_present_desktop) {
        vr_capture_mirror();
    }
    xr.gfx->EndPass(vrgfx::Target::EyeArray, image);

    XrSwapchainImageReleaseInfo release_info = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    xr_check(xrReleaseSwapchainImage(sc.handle, &release_info), "xrReleaseSwapchainImage");
    xr.multiview_pass = false;
    xr.current_eye = 0;
}

bool vr_is_multiview_pass() {
    return xr.multiview_pass;
}

bool vr_multiview_eye_welded() {
    return xr.mv_welded;
}

// Eye-welded 2D (texrects and fills that cover each eye's own view) must land identically on both
// layers: identity transforms. The interpreter flushes before switching.
void vr_set_multiview_eye_welded(bool welded) {
    if (welded != xr.mv_welded) {
        xr.mv_welded = welded;
        xr.mv_gen++;
    }
}

const float* vr_get_multiview_eye_matrices(uint32_t* generation) {
    static const float kIdentity[2][16] = {
        { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
    };
    if (generation != nullptr) {
        *generation = xr.mv_gen;
    }
    return xr.mv_welded ? &kIdentity[0][0] : &xr.mv_eye_mtx[0][0];
}

// --------------------------------------------------------------------------
// Matrix queries
// --------------------------------------------------------------------------

void vr_get_projection_matrix(int eye, float out[4][4]) {
    memcpy(out, xr.projection[eye], sizeof(float) * 16);
}

// Fog depth (z/w) for a stereo-pass vertex, from its clip-space 1/w alone. The game's fog
// coefficients (gSPFogPosition) act on z/w, which depends on the near plane; VR fog was tuned
// with the eye projection at near 10. Reproduce that near-10 z/w so moving the real near plane
// closer leaves fog exactly as it was.
void vr_get_fog_ndc_z_params(float* a, float* b) {
    const float n = 10.0f;
    const float f = xr.far_clip;
    *a = (f + n) / (f - n);
    *b = 2.0f * f * n / (f - n);
}

void vr_get_view_matrix(int eye, float out[4][4]) {
    const float (*v)[4] = xr.view[eye];
    if (!xr.anchor_initialized) {
        memcpy(out, v, sizeof(float) * 16);
        return;
    }
    // Anchored view = T(-anchor) * RotY(gamma) * view  (row-vector convention). The anchor is the
    // game-world point the playspace is glued to: Link's head in first person, the chase camera's
    // position in third person. Gamma is the base yaw of that frame: 0 in first person (world-
    // aligned, orientation pure-HMD), pi - cameraYaw in third person (facing tracking-forward
    // looks where the stock camera looks). Pitch/roll are never folded in — the horizon stays
    // level with real gravity. Anchor is in game units, matching the world_scale-scaled HMD
    // translation.
    // The game pushes anchor + gamma at 20 fps; interpolate both to this render sub-frame with the
    // same alpha the engine uses for everything else, so the camera tracks the smooth world.
    const glm::vec3 a = vr_anchor_now();
    const float g = xr.anchor_gamma_prev + (xr.anchor_gamma - xr.anchor_gamma_prev) * xr.interp_alpha;
    const float cg = cosf(g), sg = sinf(g);

    // C = RowRotY(g) * v: rotate the (translated) world into the yawed playspace frame before the
    // raw HMD view. RowRotY rows: [c 0 -s; 0 1 0; s 0 c] — rotates a direction's yaw additively.
    float C[4][4];
    for (int c = 0; c < 4; c++) {
        C[0][c] = cg * v[0][c] - sg * v[2][c];
        C[1][c] = v[1][c];
        C[2][c] = sg * v[0][c] + cg * v[2][c];
        C[3][c] = v[3][c];
    }
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 4; c++) {
            out[r][c] = C[r][c];
        }
    }
    const float ax = a.x, ay = a.y, az = a.z;
    for (int c = 0; c < 4; c++) {
        out[3][c] = C[3][c] - ax * C[0][c] - ay * C[1][c] - az * C[2][c];
    }
}

// Camera pose in game-world coords, matching the rendered (anchored) HMD view. The game feeds this
// into its own View (eye/lookAt/up) so frustum culling, audio panning and projected-position math
// align with what the player sees. Rendering is untouched (gfx_pc builds clip from the per-eye VR
// matrices and skips the game's lookAt in first-person).
//
// Derivation, no matrix inversion required: the rendered view maps a world point p to view space as
// R^-1 * (p - pos - anchor) (see vr_get_view_matrix), so the camera-to-world transform is
// translate(anchor + pos) * R. Hence eye = anchor + pos and the world forward/up are the HMD
// orientation's basis vectors (view space looks down -Z). Because the returned (eye, fwd, up) triple
// is self-consistent, feeding it back through the game's guLookAt reproduces the rendered view
// matrix exactly, sidestepping the OpenXR<->game axis-sign pitfalls that bit heading. Uses the
// center eye (average of the two eye poses). pos is scaled by world_scale to game units.
void vr_get_camera_pose(float eye[3], float fwd[3], float up[3]) {
    // Sensible identity defaults if a frame hasn't been located yet.
    eye[0] = eye[1] = eye[2] = 0.0f;
    fwd[0] = 0.0f; fwd[1] = 0.0f; fwd[2] = -1.0f;
    up[0] = 0.0f; up[1] = 1.0f; up[2] = 0.0f;
    if (!xr.initialized) return;

    // Center-eye position (midpoint of the two eyes), scaled to game units.
    glm::vec3 pos = 0.5f *
        (glm::vec3(xr.views[0].pose.position.x, xr.views[0].pose.position.y, xr.views[0].pose.position.z) +
         glm::vec3(xr.views[1].pose.position.x, xr.views[1].pose.position.y, xr.views[1].pose.position.z));
    pos *= xr.world_scale;

    // Center orientation: hemisphere-aligned, normalized average of the two eye quaternions (they're
    // near-identical, so an nlerp at 0.5 is plenty for culling). Bail to defaults if unset.
    const XrQuaternionf& q0r = xr.views[0].pose.orientation;
    const XrQuaternionf& q1r = xr.views[1].pose.orientation;
    glm::quat q0(q0r.w, q0r.x, q0r.y, q0r.z);
    glm::quat q1(q1r.w, q1r.x, q1r.y, q1r.z);
    if (glm::dot(q0, q1) < 0.0f) q1 = -q1;
    glm::quat q = q0 + q1;
    float qlen = glm::length(q);
    if (qlen < 1e-6f) return;
    q *= (1.0f / qlen);
    glm::mat3 R = glm::mat3_cast(q);

    // eye = anchor (Link's head in first person, chase camera in third) + the HMD's positional
    // offset (the world point the render maps to the view origin). Use the latest pushed anchor;
    // for a per-game-frame culling query, sub-frame interpolation isn't needed.
    glm::vec3 anchor = xr.anchor_initialized ? xr.anchor : glm::vec3(0.0f);
    glm::vec3 f = R * glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 u = R * glm::vec3(0.0f, 1.0f, 0.0f);

    // Rotate the HMD-frame vectors from the yawed playspace frame into the world (inverse of the
    // view matrix's RowRotY(gamma)): additive yaw by -gamma.
    if (xr.anchor_gamma != 0.0f) {
        const float cg = cosf(xr.anchor_gamma), sg = sinf(xr.anchor_gamma);
        auto unyaw = [cg, sg](glm::vec3 v) {
            return glm::vec3(v.x * cg - v.z * sg, v.y, v.x * sg + v.z * cg);
        };
        pos = unyaw(pos);
        f = unyaw(f);
        u = unyaw(u);
    }
    glm::vec3 e = anchor + pos;

    eye[0] = e.x; eye[1] = e.y; eye[2] = e.z;
    fwd[0] = f.x; fwd[1] = f.y; fwd[2] = f.z;
    up[0] = u.x; up[1] = u.y; up[2] = u.z;
}

// Vertical FOV (degrees) for the game's culling frustum, sized to cover the whole VR view. The
// game's native ~60-degree fovy is far narrower than the binocular VR field, so reusing it would
// cull geometry that's actually visible at the periphery (the very pop-in this fixes). Take the
// wider eye's vertical FOV and pad it generously; the culling projection derives its horizontal
// extent from the game's aspect ratio, so a wide fovy widens horizontal coverage too. Over-wide
// just draws slightly more geometry; too-narrow re-introduces edge pop-in, so we err wide.
float vr_get_culling_fovy() {
    const float kDefault = 100.0f;
    if (!xr.initialized) return kDefault;
    float vfov = 0.0f; // radians
    for (int eye = 0; eye < 2; eye++) {
        float v = xr.views[eye].fov.angleUp - xr.views[eye].fov.angleDown;
        if (v > vfov) vfov = v;
    }
    if (vfov <= 0.0f) return kDefault;
    const float kPad = 1.3f; // +30% headroom so nothing visible is culled
    float deg = glm::degrees(vfov) * kPad;
    if (deg < 90.0f) deg = 90.0f;
    if (deg > 160.0f) deg = 160.0f;
    return deg;
}

// --------------------------------------------------------------------------
// Roomscale 6DOF (physical walking moves Link's body, collision-swept)
// --------------------------------------------------------------------------

// Center-eye position (midpoint of the two eyes), scaled to game units. Zero if not located yet.
static glm::vec3 center_eye_pos_scaled() {
    if (!xr.initialized) return glm::vec3(0.0f);
    glm::vec3 pos = 0.5f *
        (glm::vec3(xr.views[0].pose.position.x, xr.views[0].pose.position.y, xr.views[0].pose.position.z) +
         glm::vec3(xr.views[1].pose.position.x, xr.views[1].pose.position.y, xr.views[1].pose.position.z));
    return pos * xr.world_scale;
}

// How far Link's body should try to move this frame to sit back under the head: the current head
// horizontal offset (game units) minus the displacement already baked into the body. The game
// rate-limits this, collision-sweeps it, and reports the achieved amount via the call below.
void vr_get_roomscale_desired(float out[2]) {
    glm::vec3 head = center_eye_pos_scaled();
    out[0] = head.x - xr.roomscale_origin.x;
    out[1] = head.z - xr.roomscale_origin.y;
}

// Advance the baked-in origin by the body's ACHIEVED horizontal move (collision-limited). Advancing
// by the achieved amount (not the desired amount) is what leaves blocked motion as a head-lean.
void vr_add_roomscale_displacement(float dx, float dz) {
    xr.roomscale_origin.x += dx;
    xr.roomscale_origin.y += dz;
}

// The baked-in origin (.x = world x, .y = world z), so the game can push anchor = bodyHead - origin.
void vr_get_roomscale_origin(float out[2]) {
    out[0] = xr.roomscale_origin.x;
    out[1] = xr.roomscale_origin.y;
}

// Re-zero roomscale so the player's current physical position maps to Link's current body position
// (desired -> 0, no body jerk). Called on recenter / first-person enable / scene change.
void vr_reset_roomscale() {
    glm::vec3 head = center_eye_pos_scaled();
    xr.roomscale_origin.x = head.x;
    xr.roomscale_origin.y = head.z;
}

// Clamp the head-lean — how far the camera sits horizontally from Link's body — to max_units, by
// advancing the baked-in origin toward the current head offset. The controller only ever moves the
// (collision-bounded) body, so this is what stops a large physical head offset (or wall-blocked
// motion) from floating the camera far past Link / out of bounds. max_units <= 0 disables it.
void vr_clamp_roomscale_lean(float max_units) {
    if (max_units <= 0.0f || !xr.initialized) return;
    glm::vec3 head = center_eye_pos_scaled();
    glm::vec2 residual(head.x - xr.roomscale_origin.x, head.z - xr.roomscale_origin.y);
    float len = glm::length(residual);
    if (len > max_units && len > 1e-4f) {
        glm::vec2 clamped = residual * (max_units / len);
        xr.roomscale_origin.x = head.x - clamped.x;
        xr.roomscale_origin.y = head.z - clamped.y;
    }
}

// --------------------------------------------------------------------------
// State queries
// --------------------------------------------------------------------------

bool vr_is_initialized() {
    // The mod-wide predicate: every VR hook in the game and interpreter gates on this, so the
    // enabled flag toggles the entire mod as one unit.
    return xr.initialized && xr.enabled;
}

// vr_init retry window (vr_apply_mode_request): a headset that isn't available yet, or a session
// being re-established after a loss, gets kVrRetryWindow of attempts before VR is turned off.
static struct {
    bool active;
    bool reconnect; // re-establishing a lost session: every failure is worth retrying
    std::chrono::steady_clock::time_point deadline;
    std::chrono::steady_clock::time_point next;
} g_vr_retry;
static constexpr auto kVrRetryInterval = std::chrono::seconds(2);
static constexpr auto kVrRetryWindow = std::chrono::seconds(60);

// Drop an instance kept between retries (vr_instance_init keeps it while the headset is missing).
static void vr_release_kept_instance() {
    if (!xr.initialized && xr.instance != XR_NULL_HANDLE) {
        xrDestroyInstance(xr.instance);
        xr.instance = XR_NULL_HANDLE;
    }
}

// Latch a pending VR<->flat mode request (CVar gVrEnabled). MUST be called at a game-tick boundary
// only (graph.c, before the tick's display list is built): a DL built for one mode must never be
// interpreted in the other, and toggling between xrBeginFrame/xrEndFrame would corrupt the session.
bool vr_can_disable() {
#ifdef __ANDROID__
    return false; // standalone headset: there is no flat screen to fall back to
#else
    return true;
#endif
}

void vr_apply_mode_request() {
    // Standalone headsets always want VR: a saved gVrEnabled 0 (from an older build, or a config
    // copied from a PC) is turned back on, so nothing keyed on the CVar (settings pages, the game's
    // VR checks) believes VR is off while it runs. Off would leave nothing visible.
    if (!vr_can_disable() && CVarGetInteger("gVrEnabled", 1) == 0) {
        CVarSetInteger("gVrEnabled", 1);
        CVarSave();
    }
    const bool want = CVarGetInteger("gVrEnabled", 1) != 0;

    // A session the runtime lost or ended (flagged by the event pump or a *_LOST result) is torn
    // down here, where no display list or XR frame is in flight. A lost one reconnects through the
    // retry window below; an ended one (the player quit the runtime) turns VR off.
    if (xr.initialized && (xr.session_lost || xr.session_exited)) {
        const bool reconnect = !xr.session_exited;
        spdlog::warn("[VR] OpenXR session {}: shutting it down{}", reconnect ? "lost" : "ended by the runtime",
                     reconnect ? " and reconnecting" : ", VR off");
        vr_shutdown();
        if (reconnect || !vr_can_disable()) {
            g_vr_retry.reconnect = true;
        } else {
            CVarSetInteger("gVrEnabled", 0);
        }
        return;
    }

    if (want && !xr.initialized) {
        // VR runs only on renderers with a VR graphics leaf (vr_backend_supported: DirectX 11 and
        // OpenGL, not Metal). Anywhere else init is impossible: don't try, and leave the CVar alone.
        if (!vr_backend_supported(vr_running_window_backend())) {
            return;
        }

        // Lazy init: launching with VR off never touches OpenXR (no SteamVR popup); the session is
        // created the first time the player toggles in. A headset that isn't available yet, or a
        // lost session being re-established, is retried every kVrRetryInterval for kVrRetryWindow
        // with the checkbox left on; only then, or on a failure retrying can't fix, does the CVar
        // flip back so the UI reflects reality.
        const auto now = std::chrono::steady_clock::now();
        if (g_vr_retry.active && now < g_vr_retry.next) {
            return;
        }
        if (vr_init()) {
            if (g_vr_retry.active) {
                spdlog::info("[VR] Headset connected");
            }
            g_vr_retry = {};
            xr.reenable_fixup = true;
            return;
        }
        if (xr.init_retryable || g_vr_retry.reconnect) {
            if (!g_vr_retry.active) {
                g_vr_retry.active = true;
                g_vr_retry.deadline = now + kVrRetryWindow;
                spdlog::warn("[VR] {}; retrying every {} s for up to {} s",
                             xr.init_retryable ? "Headset not detected yet (put it on, wake it, or connect Link)"
                                               : "Reconnecting to the OpenXR runtime",
                             (long long)kVrRetryInterval.count(), (long long)kVrRetryWindow.count());
            }
            if (now < g_vr_retry.deadline || !vr_can_disable()) {
                g_vr_retry.next = now + kVrRetryInterval;
                return;
            }
            spdlog::error("[VR] No headset after {} s; VR turned off", (long long)kVrRetryWindow.count());
        }
        g_vr_retry = {};
        vr_release_kept_instance();
        CVarSetInteger("gVrEnabled", 0);
        return;
    }

    if (!xr.initialized) {
        // VR switched off while not running: stop waiting and drop an instance kept for the retry.
        if (g_vr_retry.active) {
            spdlog::info("[VR] VR turned off; stopped waiting for the headset");
        }
        g_vr_retry = {};
        vr_release_kept_instance();
        return;
    }

    // Headset presence folds into the desired mode: doffing the headset auto-drops to flat play,
    // donning it again auto-resumes — unless the player opted to stay in VR (gVrStayOnDoff), the
    // runtime can't report presence, or VR is manually off anyway. Because this recomputes every
    // tick from (CVar, presence), manual F9 stays sticky while presence flips are symmetric.
    // "Not worn" only counts once the runtime has reported "worn" this session: a disabled or
    // unforwarded proximity sensor reads "not worn" forever and would keep VR off for good.
    const bool stay_on_doff = CVarGetInteger("gVrStayOnDoff", 0) != 0;
    const bool effective = want && (xr.user_present || stay_on_doff || !xr.user_presence_supported ||
                                    !xr.presence_confirmed || !vr_can_disable());

    if (effective && !xr.enabled) {
        xr.enabled = true;
        xr.reenable_fixup = true;
    } else if (!effective && xr.enabled) {
        xr.enabled = false;
        // The overlay DL pointer goes stale immediately (graph.c stops re-arming it in flat mode).
        xr.hud_commands = nullptr;
        xr.text_commands = nullptr;
        xr.text_has_image = false;
        // Stale hand kinematics/sim state must not leak across a disable -> re-enable gap.
        vrphys_reset();
    } else if (!xr.enabled) {
        // Session alive but idle: keep pumping the event loop at tick rate so the runtime sees us
        // as responsive AND so the "headset donned" presence event can arrive to resume VR.
        poll_events();
    }
}

int vr_get_current_eye() {
    return xr.current_eye;
}

void vr_get_recommended_resolution(uint32_t* width, uint32_t* height) {
    if (xr.initialized) {
        // Return the actual (scaled) swapchain size, not the raw recommendation, so the
        // engine's render dimensions match the viewport bound in vr_begin_eye().
        *width = xr.eye_swapchains[0].width;
        *height = xr.eye_swapchains[0].height;
    }
}

uint32_t vr_get_refresh_rate() {
    return xr.refresh_rate ? xr.refresh_rate : 90;
}

float vr_get_world_scale() {
    return xr.world_scale;
}

void vr_set_world_scale(float units_per_meter) {
    xr.world_scale = units_per_meter;
}

// --------------------------------------------------------------------------
// First-person camera
// --------------------------------------------------------------------------

void vr_set_first_person(bool enabled) {
    xr.first_person = enabled;
    if (enabled) {
        // First person is world-aligned: no base yaw.
        xr.anchor_gamma = xr.anchor_gamma_prev = 0.0f;
    }
}

// Third person: base yaw of the playspace = the game camera's facing (binang, game convention:
// yaw 0 faces +Z, dir = (sin, 0, cos)). Facing straight ahead in the headset then looks where the
// stock camera looks — including cutscene shots. Pushed once per game tick after the anchor.
void vr_set_camera_yaw(int16_t yaw_binang) {
    const float kPi = 3.14159265358979323846f;
    const float cam_yaw = (float)yaw_binang * (kPi / 32768.0f);
    // World-to-tracking rotation: tracking-forward (world dir(pi) when gamma=0) must map to the
    // camera's dir(cam_yaw), so rotate by gamma = pi - cam_yaw.
    float next = kPi - cam_yaw;
    xr.anchor_gamma_prev = xr.anchor_gamma;
    xr.anchor_gamma = next;
    // Interpolate across ticks via the shortest arc; snap on camera cuts (> 90 deg in one tick,
    // e.g. cutscene shot changes) so the view doesn't smear through the swing.
    float delta = next - xr.anchor_gamma_prev;
    while (delta > kPi) delta -= 2.0f * kPi;
    while (delta < -kPi) delta += 2.0f * kPi;
    if (delta > 0.5f * kPi || delta < -0.5f * kPi) {
        xr.anchor_gamma_prev = next;
    } else {
        // Keep prev within one wrap of next so the view-matrix lerp stays shortest-arc.
        xr.anchor_gamma_prev = next - delta;
    }
}

bool vr_is_first_person() {
    return xr.first_person;
}

void vr_set_camera_anchor(float x, float y, float z) {
    const glm::vec3 next(x, y, z);
    if (!xr.anchor_initialized) {
        xr.anchor = xr.anchor_prev = next;
        xr.anchor_initialized = true;
        return;
    }
    xr.anchor_prev = xr.anchor;
    xr.anchor = next;
    // Snap (skip interpolation) across large jumps like scene loads / warps, so the camera doesn't
    // smear across the cut. Normal movement is only a few units per game frame.
    const float kSnapDist = 200.0f;
    const glm::vec3 delta = next - xr.anchor_prev;
    if (glm::dot(delta, delta) > kSnapDist * kSnapDist) {
        xr.anchor_prev = next;
    }
}

void vr_set_interp_alpha(float alpha) {
    xr.interp_alpha = alpha;
}

// --------------------------------------------------------------------------
// Motion controls: accessors (hand: 0 = left, 1 = right)
// --------------------------------------------------------------------------

// Controller grip pose in game-world coords, composed the SAME way as the camera eye: world pos =
// anchor + grip_position * world_scale (interpolated anchor, so hands track the smoothly-rendered
// body), orientation = the controller orientation in the game-world frame (the same basis the camera
// uses). The game pushes the combined anchor (bodyHead - roomscale_origin) via vr_set_camera_anchor,
// so hands are automatically consistent with the eye + roomscale. out_quat is x,y,z,w. Returns false
// (and identity) if the hand isn't tracked.
// Effective grip pose for game-facing consumers: normally the (snap-turn-adjusted) controller
// grip, but while the held-object sim owns this hand, the SIMULATED grip pose instead — that is
// what makes the rendered hand/weapon (and every collider the game derives from the hand matrix)
// press against surfaces and lag with inertia. Sim state is raw tracking space, so the artificial
// turn is applied here exactly like everything else game-facing.
static XrPosef vr_effective_grip_pose(int hand) {
    float sp[3];
    float sq[4];
    if (vrphys_get_hand_sim_pose_raw(hand, sp, sq)) {
        XrPosef p;
        p.position = { sp[0], sp[1], sp[2] };
        p.orientation = { sq[0], sq[1], sq[2], sq[3] };
        return apply_turn(p);
    }
    return xr.grip_pose[hand];
}

bool vr_get_hand_pose(int hand, float out_pos[3], float out_quat[4]) {
    if (hand < 0 || hand > 1 || !xr.initialized || !xr.input_initialized || !xr.hand_active[hand]) {
        out_pos[0] = out_pos[1] = out_pos[2] = 0.0f;
        out_quat[0] = out_quat[1] = out_quat[2] = 0.0f;
        out_quat[3] = 1.0f;
        return false;
    }
    const XrPosef p = vr_effective_grip_pose(hand);
    const glm::vec3 anchor = (xr.first_person && xr.anchor_initialized)
                                 ? vr_anchor_now()
                                 : glm::vec3(0.0f);
    out_pos[0] = anchor.x + p.position.x * xr.world_scale;
    out_pos[1] = anchor.y + p.position.y * xr.world_scale;
    out_pos[2] = anchor.z + p.position.z * xr.world_scale;
    out_quat[0] = p.orientation.x;
    out_quat[1] = p.orientation.y;
    out_quat[2] = p.orientation.z;
    out_quat[3] = p.orientation.w;
    return true;
}

// Controller AIM ray in game-world coords: origin + unit forward direction. The aim pose is the
// runtime's calibrated pointing ray for the controller (subtly different from the grip pose —
// tuned per device so "where you point" matches player intent). Same anchor + world_scale
// composition as the grip pose, and it carries the snap-turn like everything game-facing. The
// game converts the direction to its own binang conventions (Math_Atan2S) so engine-specific
// pitch/yaw sign conventions stay in engine code. False (and forward = -Z) if untracked.
bool vr_get_aim_ray(int hand, float out_pos[3], float out_dir[3]) {
    out_pos[0] = out_pos[1] = out_pos[2] = 0.0f;
    out_dir[0] = 0.0f;
    out_dir[1] = 0.0f;
    out_dir[2] = -1.0f;
    if (hand < 0 || hand > 1 || !xr.initialized || !xr.input_initialized || !xr.hand_active[hand]) {
        return false;
    }
    const XrPosef& p = xr.aim_pose[hand];
    const glm::vec3 anchor = (xr.first_person && xr.anchor_initialized)
                                 ? vr_anchor_now()
                                 : glm::vec3(0.0f);
    glm::quat q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);

    // Player-tunable calibration: angle offsets (degrees, applied in the aim frame — pitch about
    // the ray's own X, yaw about its Y) and a positional offset (meters in the aim frame, scaled
    // to game units) so the launch point can sit exactly where the weapon's muzzle/pouch looks.
    const float kDeg = 3.14159265358979323846f / 180.0f;
    const float calPitch = CVarGetFloat("gVrAimCalPitch", 0.0f);
    const float calYaw = CVarGetFloat("gVrAimCalYaw", 0.0f);
    if (calPitch != 0.0f || calYaw != 0.0f) {
        q = q * glm::angleAxis(calYaw * kDeg, glm::vec3(0.0f, 1.0f, 0.0f)) *
            glm::angleAxis(calPitch * kDeg, glm::vec3(1.0f, 0.0f, 0.0f));
    }
    // Z negated: the CVar is "meters forward along the ray", and OpenXR aim forward is -Z.
    const glm::vec3 off(CVarGetFloat("gVrAimOffX", 0.0f), CVarGetFloat("gVrAimOffY", 0.0f),
                        -CVarGetFloat("gVrAimOffZ", 0.0f));
    const glm::vec3 posOff = q * (off * xr.world_scale);

    out_pos[0] = anchor.x + p.position.x * xr.world_scale + posOff.x;
    out_pos[1] = anchor.y + p.position.y * xr.world_scale + posOff.y;
    out_pos[2] = anchor.z + p.position.z * xr.world_scale + posOff.z;
    const glm::vec3 d = q * glm::vec3(0.0f, 0.0f, -1.0f); // OpenXR aim forward is -Z
    out_dir[0] = d.x;
    out_dir[1] = d.y;
    out_dir[2] = d.z;
    return true;
}

bool vr_is_hand_active(int hand) {
    return (hand >= 0 && hand <= 1) && xr.input_initialized && xr.hand_active[hand];
}

uint16_t vr_get_controller_buttons(int hand) {
    if (hand < 0 || hand > 1 || !xr.input_initialized) return 0;
    return xr.buttons[hand];
}

void vr_get_thumbstick(int hand, float* x, float* y) {
    if (hand < 0 || hand > 1 || !xr.input_initialized) {
        *x = *y = 0.0f;
        return;
    }
    *x = xr.thumbstick_x[hand];
    *y = xr.thumbstick_y[hand];
}

void vr_set_stick_suppressed(int hand, bool suppressed) {
    if (hand >= 0 && hand <= 1) {
        g_stick_suppressed[hand] = suppressed;
    }
}

void vr_set_turn_suppressed(bool suppressed) {
    g_turn_suppressed = suppressed;
}

void vr_set_climb_view_lock(int hand, const float ref_units[3], const float wall_normal[3], bool lateral) {
    if (hand < 0 || hand > 1 || ref_units == nullptr) {
        if (g_climb_hand >= 0 && xr.first_person && xr.hand_active[g_climb_hand]) {
            g_climb_handoff_anchor = vr_climb_locked_anchor();
            g_climb_handoff_t0 = std::chrono::steady_clock::now();
            g_climb_handoff_active = true;
        }
        g_climb_hand = -1;
        g_climb_lim_valid = false;
        g_climb_frozen = false;
        g_climb_prev_valid = false;
        return;
    }
    g_climb_handoff_active = false;
    if (g_climb_hand != hand) {
        g_climb_prev_valid = false;
        g_climb_last_body = glm::vec3(0.0f);
    }
    // A fresh ref from the game (it re-bases after a discontinuity) releases a frozen view.
    g_climb_frozen = false;
    g_climb_hand = hand;
    g_climb_ref = glm::vec3(ref_units[0], ref_units[1], ref_units[2]);
    g_climb_normal = wall_normal != nullptr ? glm::vec3(wall_normal[0], wall_normal[1], wall_normal[2])
                                            : glm::vec3(0.0f);
    const float len = glm::length(g_climb_normal);
    g_climb_normal = len > 1e-4f ? g_climb_normal / len : glm::vec3(0.0f);
    g_climb_lateral = lateral;
}

bool vr_get_hand_tracked(int hand, float out_units[3]) {
    out_units[0] = out_units[1] = out_units[2] = 0.0f;
    if (hand < 0 || hand > 1 || !xr.initialized || !xr.input_initialized || !xr.hand_active[hand] ||
        !xr.hand_tracked[hand]) {
        return false;
    }
    const XrVector3f& p = xr.grip_pose[hand].position;
    out_units[0] = p.x * xr.world_scale;
    out_units[1] = p.y * xr.world_scale;
    out_units[2] = p.z * xr.world_scale;
    return true;
}

float vr_get_trigger(int hand) {
    if (hand < 0 || hand > 1 || !xr.input_initialized) return 0.0f;
    return xr.trigger_value[hand];
}

float vr_get_grip(int hand) {
    if (hand < 0 || hand > 1 || !xr.input_initialized) return 0.0f;
    return xr.squeeze_value[hand];
}

// One-shot controller vibration through the haptic output action. xrApplyHapticFeedback is not
// frame-scoped and the whole pipeline is single-threaded, so game-tick code calls this directly —
// no queue needed. While the session isn't focused the runtime just ignores it.
void vr_trigger_haptic(int hand, float amplitude01, float freq_hz, float duration_ms) {
    if (hand < 0 || hand > 1 || !xr.initialized || !xr.enabled || !xr.input_initialized) {
        return;
    }
    XrHapticActionInfo info = { XR_TYPE_HAPTIC_ACTION_INFO };
    info.action = xr.haptic_action;
    info.subactionPath = xr.hand_path[hand];
    XrHapticVibration vib = { XR_TYPE_HAPTIC_VIBRATION };
    vib.amplitude = fminf(fmaxf(amplitude01, 0.0f), 1.0f);
    vib.frequency = freq_hz > 0.0f ? freq_hz : XR_FREQUENCY_UNSPECIFIED;
    vib.duration = duration_ms > 0.0f ? (XrDuration)((double)duration_ms * 1.0e6) : XR_MIN_HAPTIC_DURATION;
    xrApplyHapticFeedback(xr.session, &info, reinterpret_cast<const XrHapticBaseHeader*>(&vib));
}

// Live hand-matrix registry: maps each frame's hand limb Mtx* to its controller index so gfx_pc can
// substitute a fresh controller pose per eye, bypassing the game-rate interpolation that makes the
// hands judder (the camera is smooth for the same reason — it's replaced live per eye). g_hand_scale
// is Link's model scale, folded into the hand matrix so the live-replaced hand renders at full size.
static std::unordered_map<const void*, int> g_hand_mtx_registry;
static float g_hand_scale = 1.0f;
static bool g_hand_mirror[2] = { false, false }; // per controller hand: reflect the hand geometry
                                                 // (flip handedness) when it drives Link's
                                                 // opposite-side hand model

// Hand draw matrix (model-local -> game-world) in the engine's row-vector MtxF layout, for pinning
// Link's hand limb to the controller. Same world position as vr_get_hand_pose (anchor + grip_pos *
// world_scale, so hands stay consistent with the camera + roomscale), orientation = controller
// orientation * a tunable calibration (gVrHandCal* CVars, degrees) so the held item lines up with the
// real controller. Does NOT include Link's model scale — the game applies actor.scale afterward.
// Layout matches pose_to_view_matrix (out[r][c] = glm column r, row c). false (+ identity) if untracked.
bool vr_get_hand_matrix(int hand, float out[4][4]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = (r == c) ? 1.0f : 0.0f;
    if (hand < 0 || hand > 1 || !xr.initialized || !xr.input_initialized || !xr.hand_active[hand]) {
        return false;
    }
    const XrPosef p = vr_effective_grip_pose(hand);
    const glm::vec3 anchor = (xr.first_person && xr.anchor_initialized)
                                 ? vr_anchor_now()
                                 : glm::vec3(0.0f);
    const glm::vec3 world_pos(anchor.x + p.position.x * xr.world_scale,
                              anchor.y + p.position.y * xr.world_scale,
                              anchor.z + p.position.z * xr.world_scale);
    glm::quat q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
    const float kDeg = 3.14159265358979323846f / 180.0f;
    // Mirror axis: which model-local axis the reflection negates. The hand meshes' fingers/grip run
    // along model +X (the sword blade extends along hand-space +X, see the melee weapon tip/base in
    // z_player_lib.c), so the left<->right symmetry plane must KEEP X and flip the thumb axis —
    // default Z. Reflecting X itself (old default) turns the mesh inside-out instead of opposite-handed.
    int axis = CVarGetInteger("gVrHandMirrorAxis", 2);
    if (axis < 0 || axis > 2) axis = 2;
    const bool mirrored = g_hand_mirror[hand];
    // Calibration (model rest pose -> controller grip frame). Defaults were hand-tuned in-headset
    // against the MIRRORED sword hand on the right controller, which uses the mirror-conjugate of
    // these values (F * cal * F: the Euler component about the mirror axis is preserved, the other
    // two are negated). An UNMIRRORED hand also uses the conjugate regardless of controller: OpenXR
    // grip frames are defined per-hand (palm-relative), so mirror-symmetric physical poses report
    // the same orientation — an unreflected mesh attaches with the same rotation on either side.
    glm::vec3 calDeg(CVarGetFloat("gVrHandCalPitch", 88.0f), CVarGetFloat("gVrHandCalYaw", -100.0f),
                     CVarGetFloat("gVrHandCalRoll", 80.0f));
    // Positional offset (real cm) in the controller grip frame, so the hand mesh can be nudged
    // to sit naturally on the controller; the conjugate reflects it (negate the mirror-axis component).
    glm::vec3 off(CVarGetFloat("gVrHandOffX", 0.0f), CVarGetFloat("gVrHandOffY", 6.3f),
                  CVarGetFloat("gVrHandOffZ", 0.0f));
    if (hand == 0 && CVarGetInteger("gVrHandLOverride", 1)) {
        // Fully independent left-controller tuning (values used literally, no conjugation).
        calDeg = glm::vec3(CVarGetFloat("gVrHandLCalPitch", -149.0f), CVarGetFloat("gVrHandLCalYaw", 76.0f),
                           CVarGetFloat("gVrHandLCalRoll", 30.0f));
        off = glm::vec3(CVarGetFloat("gVrHandLOffX", 0.0f), CVarGetFloat("gVrHandLOffY", 6.3f),
                        CVarGetFloat("gVrHandLOffZ", 0.0f));
    } else if (hand == 1 || !mirrored) {
        for (int k = 0; k < 3; k++) {
            if (k != axis) calDeg[k] = -calDeg[k];
        }
        off[axis] = -off[axis];
    }
    // The offsets are real centimetres (how the hand sits on the controller is about the player's
    // grip, not the game world), so they hold at any world scale.
    off *= 0.01f * xr.world_scale;
    glm::quat cal = glm::quat(calDeg * kDeg);
    // Mirror = reflect the chosen model-local axis to flip the hand's handedness (the game also
    // inverts back-face culling for it).
    glm::vec3 sc(g_hand_scale);
    if (mirrored) {
        sc[axis] = -sc[axis];
    }
    glm::mat4 m = glm::translate(glm::mat4(1.0f), world_pos + q * off) * glm::mat4_cast(q * cal) *
                  glm::scale(glm::mat4(1.0f), sc);
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = m[r][c];
    return true;
}

// Folds Link's model scale into the live hand matrix (the game sets this to actor.scale each frame).
void vr_set_hand_scale(float s) {
    g_hand_scale = s;
}

// Reflect that hand's geometry to flip its apparent handedness (set when the controller drives
// Link's opposite-side hand model). The game must also invert back-face culling for a mirrored hand.
void vr_set_hand_mirror(int hand, bool mirror) {
    if (hand >= 0 && hand <= 1) {
        g_hand_mirror[hand] = mirror;
    }
}

// The game tags each hand limb's per-frame Mtx* (register) and clears the registry each game frame;
// gfx_pc calls vr_lookup_hand_matrix per eye and, on a hit, uses the LIVE controller pose. addr is the
// limb's Mtx pointer, matching gfx_sp_matrix's addr argument.
void vr_register_hand_matrix(const void* mtx, int hand) {
    if (mtx) g_hand_mtx_registry[mtx] = hand;
}

// Live hand-CHILD matrices (bow/slingshot string): registered with a hand-LOCAL transform
// extracted game-side against the same 20 Hz hand snapshot the matrix was built from; lookup
// returns (live hand pose) x (local), welding derived geometry to the live-rendered hand
// instead of letting it trail at game rate.
struct HandChildMtx {
    int hand;
    float local[4][4]; // MtxF layout: [column][component]
};
static std::unordered_map<const void*, HandChildMtx> g_hand_child_registry;

void vr_register_hand_child_matrix(const void* mtx, int hand, const float* local_mf16) {
    if (mtx && local_mf16) {
        HandChildMtx& e = g_hand_child_registry[mtx];
        e.hand = hand;
        memcpy(e.local, local_mf16, sizeof(e.local));
    }
}

// The rendered head (center eye) in game-world coords, as a model matrix in the hand matrix's
// layout: +X right, +Y up, -Z forward. Exactly the inverse of vr_get_view_matrix's anchoring
// (interpolated anchor + gamma, this frame's located views), so geometry composed onto it holds
// perfectly still in front of the eyes. false (+ identity) before the first located frame.
bool vr_get_head_matrix(float out[4][4]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = (r == c) ? 1.0f : 0.0f;
    if (!xr.initialized) {
        return false;
    }
    const XrQuaternionf& q0r = xr.views[0].pose.orientation;
    const XrQuaternionf& q1r = xr.views[1].pose.orientation;
    glm::quat q0(q0r.w, q0r.x, q0r.y, q0r.z);
    glm::quat q1(q1r.w, q1r.x, q1r.y, q1r.z);
    if (glm::dot(q0, q1) < 0.0f) q1 = -q1;
    glm::quat q = q0 + q1;
    const float qlen = glm::length(q);
    if (qlen < 1e-6f) {
        return false;
    }
    q *= (1.0f / qlen);
    glm::vec3 pos = 0.5f * (glm::vec3(xr.views[0].pose.position.x, xr.views[0].pose.position.y,
                                      xr.views[0].pose.position.z) +
                            glm::vec3(xr.views[1].pose.position.x, xr.views[1].pose.position.y,
                                      xr.views[1].pose.position.z));
    pos *= xr.world_scale;
    glm::vec3 anchor(0.0f);
    float g = 0.0f;
    if (xr.anchor_initialized) {
        anchor = vr_anchor_now();
        g = xr.anchor_gamma_prev + (xr.anchor_gamma - xr.anchor_gamma_prev) * xr.interp_alpha;
    }
    // Undo the view's RowRotY(gamma): v -> (x cg - z sg, y, x sg + z cg) (see vr_get_camera_pose).
    const float cg = cosf(g), sg = sinf(g);
    const glm::mat3 unyaw(glm::vec3(cg, 0.0f, sg), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(-sg, 0.0f, cg));
    const glm::mat4 m = glm::translate(glm::mat4(1.0f), anchor + unyaw * pos) * glm::mat4(unyaw * glm::mat3_cast(q));
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = m[r][c];
    return true;
}

// Live head-CHILD matrices (a worn item, the Lens of Truth aperture): registered with a
// head-LOCAL transform; lookup returns (rendered head) x (local) per eye, so the geometry is
// glued to the face at render rate instead of trailing the head at game rate.
static std::unordered_map<const void*, HandChildMtx> g_head_child_registry;

void vr_register_head_child_matrix(const void* mtx, const float* local_mf16) {
    if (mtx && local_mf16) {
        HandChildMtx& e = g_head_child_registry[mtx];
        e.hand = -1;
        memcpy(e.local, local_mf16, sizeof(e.local));
    }
}

// The playspace frame in game-world coords (MtxF layout): origin at the anchor this render pass
// uses (the same interpolated anchor the camera and hands compose with), axes = the playspace's
// (world-aligned in first person; turned by gamma in third). Anything placed in it moves with
// locomotion exactly as the hands and the view do, at render rate.
bool vr_get_playspace_matrix(float out[4][4]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = (r == c) ? 1.0f : 0.0f;
    if (!xr.initialized || !xr.anchor_initialized) {
        return false;
    }
    const glm::vec3 anchor = vr_anchor_now();
    const float g = xr.anchor_gamma_prev + (xr.anchor_gamma - xr.anchor_gamma_prev) * xr.interp_alpha;
    const float cg = cosf(g), sg = sinf(g);
    // Columns of unyaw (see vr_get_head_matrix), then the translation.
    out[0][0] = cg;
    out[0][2] = sg;
    out[2][0] = -sg;
    out[2][2] = cg;
    out[3][0] = anchor.x;
    out[3][1] = anchor.y;
    out[3][2] = anchor.z;
    return true;
}

// Live playspace-CHILD matrices (the item selector's compass): registered with a playspace-LOCAL
// transform; lookup returns (playspace now) x (local) per eye, so UI left floating where the hand
// was rides joystick locomotion exactly like the hands, instead of through game-rate interpolation.
static std::unordered_map<const void*, HandChildMtx> g_playspace_child_registry;

void vr_register_playspace_child_matrix(const void* mtx, const float* local_mf16) {
    if (mtx && local_mf16) {
        HandChildMtx& e = g_playspace_child_registry[mtx];
        e.hand = -1;
        memcpy(e.local, local_mf16, sizeof(e.local));
    }
}

void vr_clear_hand_matrices() {
    g_hand_mtx_registry.clear();
    g_hand_child_registry.clear();
    g_head_child_registry.clear();
    g_playspace_child_registry.clear();
}

// out = parent COMPOSED WITH local (local applied to vertices first); MtxF [column][component].
static void vr_compose_child(const float parent[4][4], const float local[4][4], float out[4][4]) {
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            out[c][r] = parent[0][r] * local[c][0] + parent[1][r] * local[c][1] + parent[2][r] * local[c][2] +
                        parent[3][r] * local[c][3];
        }
    }
}

bool vr_lookup_hand_matrix(const void* mtx, float out[4][4]) {
    // Hot path: gfx_sp_matrix calls this for EVERY G_MTX command, per eye — thousands per frame,
    // against registries that hold a handful of entries. Skip the hashes entirely when empty
    // (which is every command outside Link's hands, and every frame with hands disabled).
    if (!g_hand_mtx_registry.empty()) {
        auto it = g_hand_mtx_registry.find(mtx);
        if (it != g_hand_mtx_registry.end()) {
            return vr_get_hand_matrix(it->second, out);
        }
    }
    if (!g_hand_child_registry.empty()) {
        auto it = g_hand_child_registry.find(mtx);
        if (it != g_hand_child_registry.end()) {
            float hm[4][4];
            if (!vr_get_hand_matrix(it->second.hand, hm)) {
                return false;
            }
            vr_compose_child(hm, it->second.local, out);
            return true;
        }
    }
    if (!g_head_child_registry.empty()) {
        auto it = g_head_child_registry.find(mtx);
        if (it != g_head_child_registry.end()) {
            float head[4][4];
            if (!vr_get_head_matrix(head)) {
                return false;
            }
            vr_compose_child(head, it->second.local, out);
            return true;
        }
    }
    if (!g_playspace_child_registry.empty()) {
        auto it = g_playspace_child_registry.find(mtx);
        if (it != g_playspace_child_registry.end()) {
            float space[4][4];
            if (!vr_get_playspace_matrix(space)) {
                return false;
            }
            vr_compose_child(space, it->second.local, out);
            return true;
        }
    }
    return false;
}

int16_t vr_get_head_yaw() {
    if (!xr.initialized) return 0;
    // Heading (yaw around the Y axis) of the headset, extracted from the HMD orientation.
    const XrQuaternionf& q = xr.views[0].pose.orientation;
    const float yaw = atan2f(2.0f * (q.w * q.y + q.x * q.z), 1.0f - 2.0f * (q.y * q.y + q.z * q.z));
    // Convert radians -> binary angle (binang): pi maps to 0x8000.
    const float kPi = 3.14159265358979323846f;
    return static_cast<int16_t>(yaw / kPi * 32768.0f);
}

int16_t vr_get_heading_yaw() {
    if (!xr.initialized) return 0;
    // Steering must match what the player SEES. The first-person view is composed from the raw HMD
    // orientation — no recenter rotation is ever applied to the view — so the game-world look
    // direction is fully determined by the HMD pose alone. Derive the heading from the HMD forward
    // vector projected onto the horizontal plane: atan2(fx, fz) IS the game binang yaw (game yaw 0
    // faces +Z; movement applies sin->x, cos->z). This replaces the old Euler-angle extraction +
    // fudge constants, which skewed steering when the head pitched (up to ~15 deg looking down) and
    // added a recenter offset the view never used — the "walking sideways" bug: movement rotated
    // away from the look direction by (linkYaw - headYaw) captured at an arbitrary moment.
    static int16_t s_last_heading = 0;
    const XrQuaternionf& q = xr.views[0].pose.orientation;
    const glm::quat gq(q.w, q.x, q.y, q.z);
    const glm::vec3 fwd = gq * glm::vec3(0.0f, 0.0f, -1.0f);
    if (fwd.x * fwd.x + fwd.z * fwd.z > 1e-6f) {
        const float yaw = atan2f(fwd.x, fwd.z);
        s_last_heading = static_cast<int16_t>(yaw * (32768.0f / 3.14159265358979323846f));
    } // else: looking straight up/down, heading is degenerate — hold the last stable value
    return s_last_heading;
}

void vr_recenter_heading(int16_t link_yaw) {
    // Intentionally does NOT capture a steering offset anymore: the view never applies a recenter
    // rotation, so steering must not either — any captured offset rotates movement away from the
    // look direction (the old "walking sideways" bug). The game still calls this alongside
    // VR_ResetRoomscale when first-person (re)starts; there is simply nothing to do for heading.
    // It IS the moment to re-measure the player's physical eye height though: recentering is the
    // player's "I'm standing normally now" declaration, so auto world scale recalibrates here.
    (void)link_yaw;
    xr.heading_offset = 0;
    xr.scale_recalibrate_requested = true;
}

// Link's standing eye height in game units (Player_GetHeight + the player's tuned head offset),
// pushed by the game every first-person frame. Feeds auto world scale; a material change
// (child <-> adult) triggers automatic recalibration.
void vr_set_link_eye_height(float units) {
    xr.link_eye_height_units = units;
}

// In-wall view fade target (0 = clear, 1 = black), set by the game per tick from how deep the
// camera sits beyond solid geometry. Smoothed and applied at the compositor in vr_end_frame.
void vr_set_view_fade(float fade) {
    xr.view_fade_target = fminf(fmaxf(fade, 0.0f), 1.0f);
}

void vr_clear_current_eye_depth() {
    if (!xr.initialized || !xr.frame_began || xr.rendering_screen || xr.rendering_text || xr.rendering_hud) {
        return;
    }
    xr.gfx->ClearDepth(vr_eye_target(xr.current_eye), vr_current_eye_image());
}

void vr_rebind_current_eye_target() {
    if (!xr.initialized || !xr.frame_began) return;

    // Restore whichever target is ACTUALLY being rendered: the flat-screen panel or the HUD when a
    // 2D pass is active (the pause menu runs framebuffer copies mid-pass — blindly rebinding an eye
    // here used to dump the inventory into the stale right-eye image), else the current eye.
    if (xr.rendering_screen) {
        xr.gfx->Rebind(vrgfx::Target::Screen, xr.screen_image_index);
    } else if (xr.rendering_text) {
        xr.gfx->Rebind(vrgfx::Target::Text, xr.text_image_index);
    } else if (xr.rendering_hud) {
        xr.gfx->Rebind(vrgfx::Target::Hud, xr.hud_image_index);
    } else {
        xr.gfx->Rebind(vr_eye_target(xr.current_eye), vr_current_eye_image());
    }
}

// --------------------------------------------------------------------------
// HUD overlay
// --------------------------------------------------------------------------

void vr_set_hud_commands(void* commands) { xr.hud_commands = commands; }
void* vr_get_hud_commands() { return xr.hud_commands; }

// Wrist layout (default) vs the classic single head-locked HUD quad.
static bool vr_hud_wrist_layout() {
    return CVarGetInteger("gVrHudLayout", 0) == 0;
}

// Which wrist an element lives on: 0 = left controller, 1 = right, -1 = none.
static int vr_hud_el_hand(int el) {
    switch (el) {
        case VR_HUD_EL_HEARTS:
        case VR_HUD_EL_MAGIC:
        case VR_HUD_EL_RUPEES:
        case VR_HUD_EL_KEYS:
        case VR_HUD_EL_TIMER:
        case VR_HUD_EL_GAME_TIMER:
            return 0;
        case VR_HUD_EL_BTN_B:
        case VR_HUD_EL_BTN_A:
        case VR_HUD_EL_BTN_C_LEFT:
        case VR_HUD_EL_BTN_C_DOWN:
        case VR_HUD_EL_BTN_C_RIGHT:
        case VR_HUD_EL_BTN_C_UP:
        case VR_HUD_EL_BTN_DPAD:
        case VR_HUD_EL_MOUNT:
        case VR_HUD_EL_MINIMAP:
            return 1;
        default:
            return -1;
    }
}

// The wrist canvases: each hand gets its own region of the HUD image (left half / right half of
// the 320x240 frame) and its elements are laid out there in rows, top to bottom, left to right,
// each row's elements centred on the row's height; then every element moves by its own offset
// and size from the settings (vr_hud_settings.h, per Adult / Child profile). -1 ends a row, -2
// ends the hand.
//   LEFT : hearts / magic / rupees, keys / timers
//   RIGHT: the B / A / C button cluster in vanilla's arrangement / D-pad, carrots-or-archery / minimap
// Layout runs in LAYOUT UNITS (1 = a native HUD pixel at 100% size); the canvas is packed into
// the image at kWristPack native px per unit, which doubles the room per hand (the HUD swapchain
// doubled in resolution to match, so elements are exactly as sharp as before).
static const int kWristRows[2][16] = {
    { VR_HUD_EL_HEARTS, -1, VR_HUD_EL_MAGIC, -1, VR_HUD_EL_RUPEES, VR_HUD_EL_KEYS, -1, VR_HUD_EL_TIMER,
      VR_HUD_EL_GAME_TIMER, -2 },
    { -3, VR_HUD_EL_BTN_DPAD, VR_HUD_EL_MOUNT, -1, VR_HUD_EL_MINIMAP, -2 }, // -3 = the button cluster
};
static constexpr float kWristCanvasH = 240.0f;
static constexpr float kWristRegionW = kWristCanvasW / kWristPack; // layout units per hand
static constexpr float kWristRegionH = kWristCanvasH / kWristPack;
static constexpr float kWristGap = 3.0f;                    // between auto-laid-out elements, units

// The settings profile in effect: Link's age, unless the menu forces one for preview.
static int vr_hud_profile() {
    const int preview = CVarGetInteger("gVrHud.Preview", 0);
    if (preview == 1) {
        return 0;
    }
    if (preview == 2) {
        return 1;
    }
    return xr.hud_child ? 1 : 0;
}

static float vr_hud_panel_f(int child, int hand, const char* field, float def) {
    char name[96];
    VrHudPanelCVar(name, sizeof(name), child, hand, field);
    return CVarGetFloat(name, def);
}

void vr_set_hud_child(bool child) {
    xr.hud_child = child;
}

bool vr_get_hud_child() {
    return xr.hud_child;
}

void vr_hud_marker(uint32_t w1) {
    int el = (int)(w1 & 0xFF);
    if (el >= VR_HUD_EL_COUNT) {
        el = VR_HUD_EL_NONE;
    }
    xr.hud_el = el;
    const uint8_t alpha = (uint8_t)((w1 >> 16) & 0xFF);
    if (alpha > xr.hud_meas_alpha[el]) {
        xr.hud_meas_alpha[el] = alpha;
    }
}

void vr_set_hud_world_panel(bool enabled, const float center[3], float yaw, float width, float height) {
    xr.hud_world = enabled;
    if (enabled) {
        xr.hud_world_center = glm::vec3(center[0], center[1], center[2]);
        xr.hud_world_yaw = yaw;
        xr.hud_world_size[0] = width;
        xr.hud_world_size[1] = height;
    }
}

// out = { s, ox, oy, sy }: the interpreter writes x' = s x + ox w, y' = sy y + oy w (clip space),
// which maps the element's TV extent origin r0 to its slot A (native px, y down) at scale s:
// X' = A + (X - r0) s with X = (ndc + 1) 160, Y = (1 - ndc) 120. sy = s except on the pause frame.
bool vr_hud_tri(const float ndc[6], float out[4]) {
    const int el = xr.hud_el;
    out[0] = out[3] = 1.0f;
    out[1] = out[2] = 0.0f;
    if (xr.hud_world_pass) {
        // The pause frame: every HUD element (vitals, buttons with Return / Save / Decide, the
        // equip fly-in), measured only to know which corner it belongs to; the wrists keep their
        // layout. The canvas can be wider / taller than the TV frame and the elements bigger or
        // smaller (vr_hud_world_canvas): each element keeps its distance from its own corner
        // (scaled with the element size), so it neither stretches nor drifts off that corner.
        if (el == VR_HUD_EL_NONE || el == VR_HUD_EL_OTHER) {
            return false;
        }
        float cx = 0.0f, cy = 0.0f;
        float* m = xr.hud_world_meas[el];
        for (int i = 0; i < 3; i++) {
            const float x = ndc[i * 2], y = ndc[i * 2 + 1];
            cx += x / 3.0f;
            cy += y / 3.0f;
            if (!xr.hud_world_meas_any[el]) {
                m[0] = m[2] = x;
                m[1] = m[3] = y;
                xr.hud_world_meas_any[el] = true;
            } else {
                m[0] = std::min(m[0], x);
                m[1] = std::min(m[1], y);
                m[2] = std::max(m[2], x);
                m[3] = std::max(m[3], y);
            }
        }
        // The button cluster (B, A, START, the C buttons) is one group on the TV: pinned together to
        // the top right, B / A / START keep their place just left of the C buttons. By centre alone
        // they sat left of the middle and went to the top left, onto the hearts and magic.
        const bool buttonCluster = el == VR_HUD_EL_BTN_A || el == VR_HUD_EL_BTN_B || el == VR_HUD_EL_BTN_START ||
                                   (el >= VR_HUD_EL_BTN_C_LEFT && el <= VR_HUD_EL_BTN_C_UP);
        float ax, ay;
        if (el == VR_HUD_EL_PAUSE_FX || buttonCluster) {
            ax = ay = 1.0f; // the fly-in lands on the C buttons (top right); never flips mid-flight
        } else if (xr.hud_world_anchor_valid[el]) {
            ax = xr.hud_world_anchor[el][0];
            ay = xr.hud_world_anchor[el][1];
        } else {
            ax = cx < 0.0f ? -1.0f : 1.0f; // first pass it draws: this triangle decides
            ay = cy < 0.0f ? -1.0f : 1.0f;
        }
        const float* c = xr.hud_world_canvas;
        out[0] = c[2] / c[0];
        out[3] = c[2] / c[1];
        out[1] = ax * (1.0f - out[0]);
        out[2] = ay * (1.0f - out[3]);
        return true;
    }
    if (vr_hud_el_hand(el) < 0) {
        return false;
    }
    // Entirely off-frame (SoH's "hidden" cosmetic position is -9999; rects skip clip rejection):
    // invisible anyway, and it must not stretch the element's extent to the frame edge.
    if ((ndc[0] < -1.0f && ndc[2] < -1.0f && ndc[4] < -1.0f) || (ndc[0] > 1.0f && ndc[2] > 1.0f && ndc[4] > 1.0f) ||
        (ndc[1] < -1.0f && ndc[3] < -1.0f && ndc[5] < -1.0f) || (ndc[1] > 1.0f && ndc[3] > 1.0f && ndc[5] > 1.0f)) {
        return false;
    }
    // Measure on the TV layout (native px, the untouched position).
    float* m = xr.hud_meas[el];
    for (int i = 0; i < 3; i++) {
        const float x = std::clamp((ndc[i * 2] + 1.0f) * 160.0f, 0.0f, 320.0f);
        const float y = std::clamp((1.0f - ndc[i * 2 + 1]) * 120.0f, 0.0f, 240.0f);
        if (!xr.hud_meas_any[el]) {
            m[0] = m[2] = x;
            m[1] = m[3] = y;
            xr.hud_meas_any[el] = true;
        } else {
            m[0] = std::min(m[0], x);
            m[1] = std::min(m[1], y);
            m[2] = std::max(m[2], x);
            m[3] = std::max(m[3], y);
        }
    }
    // Draw it in its slot. Not placed (first pass it ever drew, or hidden in the settings): no draw.
    if (!xr.hud_placed[el] || !xr.hud_rect_valid[el]) {
        return false;
    }
    const float s = xr.hud_slot[el][2];
    const float* r0 = xr.hud_rect[el];
    out[0] = out[3] = s;
    out[1] = (xr.hud_slot[el][0] - r0[0] * s) / 160.0f + s - 1.0f;
    out[2] = 1.0f - s - (xr.hud_slot[el][1] - r0[1] * s) / 120.0f;
    return true;
}

// Fold this pass's measurements into the held extents, then lay both canvases out for the next pass.
static void vr_hud_latch_and_layout() {
    for (int el = 0; el < VR_HUD_EL_COUNT; el++) {
        if (vr_hud_el_hand(el) < 0) {
            continue;
        }
        if (!xr.hud_meas_any[el]) {
            // Not drawn: keep its slot for a while (the Navi prompt blinks, a one-tick gap mustn't
            // reflow the layout), then drop it out of the layout.
            if (++xr.hud_absent_passes[el] > 30) {
                xr.hud_rect_valid[el] = false;
                xr.hud_alpha[el] = 0.0f;
            }
            continue;
        }
        xr.hud_absent_passes[el] = 0;
        xr.hud_alpha[el] = xr.hud_meas_alpha[el] / 255.0f;

        const float* meas = xr.hud_meas[el];
        float* r = xr.hud_rect[el];
        if (!xr.hud_rect_valid[el]) {
            memcpy(r, meas, sizeof(float) * 4);
            xr.hud_rect_valid[el] = true;
            xr.hud_shrink_passes[el] = 0;
            continue;
        }
        const bool grows = meas[0] < r[0] || meas[1] < r[1] || meas[2] > r[2] || meas[3] > r[3];
        const float slack = 1.5f;
        const bool smaller = meas[0] > r[0] + slack || meas[1] > r[1] + slack || meas[2] < r[2] - slack ||
                             meas[3] < r[3] - slack;
        if (grows) {
            r[0] = std::min(r[0], meas[0]);
            r[1] = std::min(r[1], meas[1]);
            r[2] = std::max(r[2], meas[2]);
            r[3] = std::max(r[3], meas[3]);
            xr.hud_shrink_passes[el] = 0;
        } else if (smaller) {
            // ~1.5 s of HUD passes at 20 Hz: a lost heart container or 3 -> 2 digit rupees settles,
            // the low-health heart beat never gets the chance.
            if (++xr.hud_shrink_passes[el] > 30) {
                memcpy(r, meas, sizeof(float) * 4);
                xr.hud_shrink_passes[el] = 0;
            }
        } else {
            xr.hud_shrink_passes[el] = 0;
        }
    }

    const int child = vr_hud_profile();
    for (int hand = 0; hand < 2; hand++) {
        const VrHudPanelDefaults& pd = kVrHudPanelDefaults[child][hand];
        const float pad = std::clamp(vr_hud_panel_f(child, hand, "Padding", pd.padding), 0.0f, 100.0f);

        // Per element settings for this hand: size, offset, visibility.
        float el_scale[VR_HUD_EL_COUNT];
        float el_off[VR_HUD_EL_COUNT][2];
        bool el_show[VR_HUD_EL_COUNT];
        for (int el = 0; el < VR_HUD_EL_COUNT; el++) {
            el_scale[el] = 1.0f;
            el_off[el][0] = el_off[el][1] = 0.0f;
            el_show[el] = true;
            const VrHudElementDesc* d = VrHudElementById(el);
            if (d == nullptr) {
                continue;
            }
            char name[96];
            VrHudElementCVar(name, sizeof(name), child, d->key, "Scale");
            el_scale[el] = std::clamp(CVarGetFloat(name, d->scale[child]), 10.0f, 500.0f) / 100.0f;
            VrHudElementCVar(name, sizeof(name), child, d->key, "X");
            el_off[el][0] = CVarGetFloat(name, d->x[child]);
            VrHudElementCVar(name, sizeof(name), child, d->key, "Y");
            el_off[el][1] = CVarGetFloat(name, d->y[child]);
            VrHudElementCVar(name, sizeof(name), child, d->key, "Show");
            el_show[el] = CVarGetInteger(name, d->show[child]) != 0;
        }

        // Auto layout in layout units (rows, wrapping at the region edge), then each element's
        // own offset on top; everything is kept inside this hand's region of the image.
        float y = pad;
        float max_x = 0.0f, max_y = 0.0f;
        float block_alpha = 0.0f;
        bool any = false;
        int row[16];
        int row_n = 0;
        auto el_w = [&](int el) { return (xr.hud_rect[el][2] - xr.hud_rect[el][0]) * el_scale[el]; };
        auto el_h = [&](int el) { return (xr.hud_rect[el][3] - xr.hud_rect[el][1]) * el_scale[el]; };
        auto flush_row = [&]() {
            if (row_n == 0) {
                return;
            }
            float row_h = 0.0f;
            for (int k = 0; k < row_n; k++) {
                row_h = std::max(row_h, el_h(row[k]));
            }
            float x = pad;
            for (int k = 0; k < row_n; k++) {
                const int el = row[k];
                const float w = el_w(el);
                const float h = el_h(el);
                float fx = x + el_off[el][0];
                float fy = y + (row_h - h) * 0.5f + el_off[el][1];
                fx = std::clamp(fx, 0.0f, std::max(kWristRegionW - w, 0.0f));
                fy = std::clamp(fy, 0.0f, std::max(kWristRegionH - h, 0.0f));
                xr.hud_slot[el][0] = hand * kWristCanvasW + fx * kWristPack;
                xr.hud_slot[el][1] = fy * kWristPack;
                xr.hud_slot[el][2] = el_scale[el] * kWristPack;
                xr.hud_placed[el] = true;
                max_x = std::max(max_x, fx + w);
                max_y = std::max(max_y, fy + h);
                x += w + kWristGap;
            }
            y += row_h + kWristGap;
            row_n = 0;
        };

        // The B / A / C cluster: every button keeps its TV position relative to the others
        // (vanilla's arrangement, as in SoH's HUD editor). All of them sit at their constant +
        // the same right-edge shift (OTRGetDimensionFromRightEdge with the eye's aspect, which is
        // what the game laid them out with), so one fixed reference moves the cluster rigidly;
        // labels changing width never shift it. 8 units of room to the left for the B label.
        auto place_cluster = [&]() {
            static const int kCluster[6] = { VR_HUD_EL_BTN_B, VR_HUD_EL_BTN_A, VR_HUD_EL_BTN_C_UP,
                                             VR_HUD_EL_BTN_C_LEFT, VR_HUD_EL_BTN_C_DOWN, VR_HUD_EL_BTN_C_RIGHT };
            const auto& eye0 = xr.eye_swapchains[0];
            const float aspect = (eye0.width > 0 && eye0.height > 0) ? (float)eye0.width / eye0.height : 1.0f;
            const float shift = 120.0f * aspect - 160.0f;
            const float ref_x = kVrHudClusterRefX + shift - 8.0f;
            const float ref_y = kVrHudClusterRefY - 2.0f;
            float bottom = y;
            bool placed_any = false;
            for (int el : kCluster) {
                if (!xr.hud_rect_valid[el] || !el_show[el]) {
                    xr.hud_placed[el] = false;
                    continue;
                }
                const float w = el_w(el);
                const float h = el_h(el);
                float fx = pad + (xr.hud_rect[el][0] - ref_x) + el_off[el][0];
                float fy = y + (xr.hud_rect[el][1] - ref_y) + el_off[el][1];
                fx = std::clamp(fx, 0.0f, std::max(kWristRegionW - w, 0.0f));
                fy = std::clamp(fy, 0.0f, std::max(kWristRegionH - h, 0.0f));
                xr.hud_slot[el][0] = hand * kWristCanvasW + fx * kWristPack;
                xr.hud_slot[el][1] = fy * kWristPack;
                xr.hud_slot[el][2] = el_scale[el] * kWristPack;
                xr.hud_placed[el] = true;
                max_x = std::max(max_x, fx + w);
                max_y = std::max(max_y, fy + h);
                bottom = std::max(bottom, fy + h);
                placed_any = true;
                any = true;
                block_alpha = std::max(block_alpha, xr.hud_alpha[el]);
            }
            if (placed_any) {
                y = bottom + kWristGap;
            }
        };

        for (int k = 0; k < 16 && kWristRows[hand][k] != -2; k++) {
            const int el = kWristRows[hand][k];
            if (el == -1) {
                flush_row();
                continue;
            }
            if (el == -3) {
                flush_row();
                place_cluster();
                continue;
            }
            if (!xr.hud_rect_valid[el] || !el_show[el]) {
                xr.hud_placed[el] = false;
                continue;
            }
            float row_w = pad;
            for (int j = 0; j < row_n; j++) {
                row_w += el_w(row[j]) + kWristGap;
            }
            if (row_n > 0 && row_w + el_w(el) > kWristRegionW - pad) {
                flush_row();
            }
            row[row_n++] = el;
            any = true;
            block_alpha = std::max(block_alpha, xr.hud_alpha[el]);
        }
        flush_row();

        // The card: fitted to the elements plus padding, or the size set in the settings.
        float card_w = vr_hud_panel_f(child, hand, "Width", pd.width);
        float card_h = vr_hud_panel_f(child, hand, "Height", pd.height);
        if (card_w <= 0.0f) {
            card_w = max_x + pad;
        }
        if (card_h <= 0.0f) {
            card_h = max_y + pad;
        }
        card_w = std::clamp(card_w, 4.0f, kWristRegionW);
        card_h = std::clamp(card_h, 4.0f, kWristRegionH);

        xr.hud_block_valid[hand] = any;
        if (any) {
            xr.hud_block[hand][0] = hand * kWristCanvasW;
            xr.hud_block[hand][1] = 0.0f;
            xr.hud_block[hand][2] = hand * kWristCanvasW + card_w * kWristPack;
            xr.hud_block[hand][3] = card_h * kWristPack;
        }
        xr.hud_block_alpha[hand] = block_alpha;
    }
}

bool vr_wants_coverage_blend() {
    return xr.rendering_text || (xr.rendering_hud && !xr.rendering_screen);
}

void vr_begin_hud() {
    if (!xr.initialized) return;
    xr.rendering_hud = true;
    xr.hud_ever_rendered = true;

    auto& sc = xr.hud_swapchain;
    XrSwapchainImageAcquireInfo acquire_info = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
    uint32_t image_index = 0;
    xr_check(xrAcquireSwapchainImage(sc.handle, &acquire_info, &image_index), "xrAcquireSwapchainImage (HUD)");
    xr.hud_image_index = image_index;

    XrSwapchainImageWaitInfo wait_info = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
    wait_info.timeout = XR_INFINITE_DURATION;
    xr_check(xrWaitSwapchainImage(sc.handle, &wait_info), "xrWaitSwapchainImage (HUD)");

    const float clear_color[4] = { 0.0f, 0.0f, 0.0f, 0.0f }; // Transparent
    xr.gfx->BeginPass(vrgfx::Target::Hud, image_index, clear_color);

    // Wrist layout: paint each hand canvas's dark backing (premultiplied translucent black, faded
    // with its elements) over the laid-out block BEFORE the elements draw into their slots on it.
    xr.hud_world_pass = xr.hud_world;
    g_vr_hud_layout_pass = xr.hud_world_pass || vr_hud_wrist_layout();
    xr.hud_el = VR_HUD_EL_NONE;
    if (xr.hud_world_pass) {
        vr_hud_world_canvas(xr.hud_world_canvas);
        for (int el = 0; el < VR_HUD_EL_COUNT; el++) {
            xr.hud_world_meas_any[el] = false;
        }
    }
    if (g_vr_hud_layout_pass && !xr.hud_world_pass) {
        const int child = vr_hud_profile();
        const float sx = sc.width / 320.0f, sy = sc.height / 240.0f;
        for (int hand = 0; hand < 2; hand++) {
            const float backing = std::clamp(
                vr_hud_panel_f(child, hand, "Backing", kVrHudPanelDefaults[child][hand].backing) / 100.0f, 0.0f, 1.0f);
            if (!xr.hud_block_valid[hand] || backing <= 0.0f) {
                continue;
            }
            const float* b = xr.hud_block[hand];
            const int32_t x0 = (int32_t)(b[0] * sx), y0 = (int32_t)(b[1] * sy);
            const vrgfx::Rect rect = { x0, y0, (int32_t)(b[2] * sx + 0.5f) - x0, (int32_t)(b[3] * sy + 0.5f) - y0 };
            // In-headset editing: tint the card while a hand is in reach (blue) / holding it.
            const float a = std::max(backing * xr.hud_block_alpha[hand], xr.grab.hover[hand] ? 0.6f : 0.0f);
            const float tint = xr.grab.hover[hand] == 2 ? 0.35f : (xr.grab.hover[hand] == 1 ? 0.18f : 0.0f);
            const float dark[4] = { 0.0f, tint * 0.4f * a, tint * a, a };
            xr.gfx->ClearColorRects(vrgfx::Target::Hud, image_index, &rect, 1, dark);
        }
        for (int el = 0; el < VR_HUD_EL_COUNT; el++) {
            xr.hud_meas_any[el] = false;
            xr.hud_meas_alpha[el] = 0;
        }
    }

    vr_apply_dimensions(sc.width, sc.height);
}

void vr_end_hud() {
    if (!xr.initialized) return;
    xr.rendering_hud = false;
    if (g_vr_hud_layout_pass) {
        g_vr_hud_layout_pass = false;
        if (!xr.hud_world_pass) {
            vr_hud_latch_and_layout();
        } else {
            // Each element's corner for the next pass, from the centre of where it drew on the TV
            // frame this pass (an element that didn't draw keeps its last corner).
            for (int el = 0; el < VR_HUD_EL_COUNT; el++) {
                if (xr.hud_world_meas_any[el]) {
                    const float* m = xr.hud_world_meas[el];
                    xr.hud_world_anchor[el][0] = (m[0] + m[2]) < 0.0f ? -1.0f : 1.0f;
                    xr.hud_world_anchor[el][1] = (m[1] + m[3]) < 0.0f ? -1.0f : 1.0f;
                    xr.hud_world_anchor_valid[el] = true;
                }
            }
        }
    }
    xr.hud_world_pass = false;

    vr_draw_test_card(vrgfx::Target::Hud, xr.hud_image_index, xr.hud_swapchain.width, xr.hud_swapchain.height);
    xr.gfx->CopyToMirror(vrgfx::Target::Hud, xr.hud_image_index, vrgfx::kMirrorHud); // desktop mirror
    xr.gfx->EndPass(vrgfx::Target::Hud, xr.hud_image_index);

    XrSwapchainImageReleaseInfo release_info = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    xr_check(xrReleaseSwapchainImage(xr.hud_swapchain.handle, &release_info), "xrReleaseSwapchainImage (HUD)");

    vr_restore_eye_dimensions();
}

bool vr_is_rendering_hud() { return xr.rendering_hud; }

bool vr_is_rendering_screen() { return xr.rendering_screen; }

// --------------------------------------------------------------------------
// Text panel (message system on its own soft-follow quad)
// --------------------------------------------------------------------------

void vr_set_text_commands(void* commands, const float crop[4]) {
    if (commands == nullptr) {
        // Box closed: the next one starts from a blank image and snaps into place.
        xr.text_has_image = false;
    }
    xr.text_commands = commands;
    for (int i = 0; i < 4; i++) {
        xr.text_crop[i] = crop[i];
    }
}

void* vr_get_text_commands() { return xr.text_commands; }

void vr_set_text_panel_layout(float width_m, float distance_m, float height_m) {
    xr.text_layout[0] = width_m;
    xr.text_layout[1] = distance_m;
    xr.text_layout[2] = height_m;
}

// Same 2D pass as the HUD (rendering_hud keeps the game's flat projection), into the text swapchain.
void vr_begin_text() {
    if (!xr.initialized) return;
    xr.rendering_hud = true;
    xr.rendering_text = true;
    xr.text_has_image = true;

    auto& sc = xr.text_swapchain;
    XrSwapchainImageAcquireInfo acquire_info = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
    uint32_t image_index = 0;
    xr_check(xrAcquireSwapchainImage(sc.handle, &acquire_info, &image_index), "xrAcquireSwapchainImage (text)");
    xr.text_image_index = image_index;

    XrSwapchainImageWaitInfo wait_info = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
    wait_info.timeout = XR_INFINITE_DURATION;
    xr_check(xrWaitSwapchainImage(sc.handle, &wait_info), "xrWaitSwapchainImage (text)");

    const float clear_color[4] = { 0.0f, 0.0f, 0.0f, 0.0f }; // Transparent
    xr.gfx->BeginPass(vrgfx::Target::Text, image_index, clear_color);
    vr_apply_dimensions(sc.width, sc.height);
}

void vr_end_text() {
    if (!xr.initialized) return;
    xr.rendering_hud = false;
    xr.rendering_text = false;

    vr_draw_test_card(vrgfx::Target::Text, xr.text_image_index, xr.text_swapchain.width, xr.text_swapchain.height);
    xr.gfx->CopyToMirror(vrgfx::Target::Text, xr.text_image_index, vrgfx::kMirrorText); // desktop mirror
    xr.gfx->EndPass(vrgfx::Target::Text, xr.text_image_index);

    XrSwapchainImageReleaseInfo release_info = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    xr_check(xrReleaseSwapchainImage(xr.text_swapchain.handle, &release_info), "xrReleaseSwapchainImage (text)");

    vr_restore_eye_dimensions();
}

bool vr_is_rendering_text() { return xr.rendering_text; }

// --------------------------------------------------------------------------
// Flat-screen mode (whole frame on a floating panel: file select, pause menu)
// --------------------------------------------------------------------------

void vr_set_flat_screen(bool enabled) {
    xr.flat_screen = enabled;
}

bool vr_get_flat_screen() {
    return xr.initialized && xr.enabled && xr.flat_screen;
}

// Render the game's full frame into the screen swapchain. Reuses the HUD's "2D rendering" flag so
// gfx_pc uses the normal flat projection instead of the per-eye VR overrides.
void vr_begin_screen() {
    if (!xr.initialized) return;
    xr.rendering_hud = true; // gfx_pc's "2D target" flag: use the flat projection, not the VR eyes
    xr.rendering_screen = true;
    xr.screen_ever_rendered = true;

    auto& sc = xr.screen_swapchain;
    XrSwapchainImageAcquireInfo acquire_info = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
    uint32_t image_index = 0;
    xr_check(xrAcquireSwapchainImage(sc.handle, &acquire_info, &image_index), "xrAcquireSwapchainImage (screen)");
    xr.screen_image_index = image_index;

    XrSwapchainImageWaitInfo wait_info = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
    wait_info.timeout = XR_INFINITE_DURATION;
    xr_check(xrWaitSwapchainImage(sc.handle, &wait_info), "xrWaitSwapchainImage (screen)");

    const float clear_color[4] = { 0.0f, 0.0f, 0.0f, 1.0f }; // Opaque black
    xr.gfx->BeginPass(vrgfx::Target::Screen, image_index, clear_color);
    vr_apply_dimensions(sc.width, sc.height);
}

void vr_end_screen() {
    if (!xr.initialized) return;
    xr.rendering_hud = false;
    xr.rendering_screen = false;

    vr_draw_test_card(vrgfx::Target::Screen, xr.screen_image_index, xr.screen_swapchain.width,
                      xr.screen_swapchain.height);
    xr.gfx->CopyToMirror(vrgfx::Target::Screen, xr.screen_image_index, vrgfx::kMirrorScreen); // desktop mirror
    xr.gfx->EndPass(vrgfx::Target::Screen, xr.screen_image_index);

    XrSwapchainImageReleaseInfo release_info = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    xr_check(xrReleaseSwapchainImage(xr.screen_swapchain.handle, &release_info), "xrReleaseSwapchainImage (screen)");

    vr_restore_eye_dimensions();
}

// Dimensions of whichever 2D target is currently being rendered (HUD quad or the flat-screen
// panel), so gfx_pc sizes the frame to the actual texture instead of assuming the HUD's.
void vr_get_2d_target_size(uint32_t* w, uint32_t* h) {
    if (xr.rendering_screen) {
        *w = xr.screen_swapchain.width;
        *h = xr.screen_swapchain.height;
    } else if (xr.rendering_text) {
        *w = xr.text_swapchain.width;
        *h = xr.text_swapchain.height;
    } else {
        *w = xr.hud_swapchain.width;
        *h = xr.hud_swapchain.height;
    }
}

// --------------------------------------------------------------------------
// Desktop mirror
// --------------------------------------------------------------------------

void vr_capture_mirror() {
    if (!xr.initialized || !xr.gfx) return;
    xr.gfx->CopyToMirror(vr_eye_target(0), xr.current_image_index[0], vrgfx::kMirrorEye); // layer 0 when multiview
}

bool vr_get_mirror_flip_v() {
    return xr.gfx && xr.gfx->MirrorFlipV();
}

// The headset's quad layers as the desktop mirror (the left eye) sees them: each quad as a grid of
// projected points (0..1 across the mirror image) with its texture and UV rect, so the companion
// window can draw it in perspective over the mirror (it is a compositor layer the eye image never
// contains). Projected through the pose + FOV the mirrored eye image was rendered with.
int vr_get_mirror_quads(VrMirrorQuad* out, int max) {
    if (!xr.initialized || !xr.enabled || !xr.gfx) {
        return 0;
    }
    const bool flip_v = xr.gfx->MirrorFlipV();
    const XrPosef& e = xr.submit_pose[0];
    const XrFovf& f = xr.submit_fov[0];
    const glm::quat eq_inv =
        glm::inverse(glm::quat(e.orientation.w, e.orientation.x, e.orientation.y, e.orientation.z));
    const glm::vec3 ep(e.position.x, e.position.y, e.position.z);
    const float tl = tanf(f.angleLeft), tr = tanf(f.angleRight), tu = tanf(f.angleUp), td = tanf(f.angleDown);
    if (tr - tl < 1e-4f || tu - td < 1e-4f) {
        return 0;
    }
    int n = 0;
    for (int i = 0; i < xr.mirror_quad_count && n < max; i++) {
        const auto& m = xr.mirror_quads[i];
        // src 0 HUD, 1 text, 2 flat-screen panel -> mirror slots 1..3; the copies are the size of
        // their swapchain.
        static const int kSlot[3] = { vrgfx::kMirrorHud, vrgfx::kMirrorText, vrgfx::kMirrorScreen };
        const auto* src_sc = m.src == 0 ? &xr.hud_swapchain : (m.src == 1 ? &xr.text_swapchain : &xr.screen_swapchain);
        void* tex = (m.src >= 0 && m.src < 3) ? xr.gfx->MirrorTextureId(kSlot[m.src]) : nullptr;
        const float cw = (float)src_sc->width, ch = (float)src_sc->height;
        if (tex == nullptr || cw <= 0.0f || ch <= 0.0f) {
            continue;
        }
        VrMirrorQuad& o = out[n++];
        o.srv = tex;
        o.uv[0] = (float)m.rect.offset.x / cw;
        o.uv[1] = (float)m.rect.offset.y / ch;
        o.uv[2] = (float)(m.rect.offset.x + m.rect.extent.width) / cw;
        o.uv[3] = (float)(m.rect.offset.y + m.rect.extent.height) / ch;
        if (flip_v) { // copies stored bottom-up (GL): same picture, mirrored V
            o.uv[1] = 1.0f - o.uv[1];
            o.uv[3] = 1.0f - o.uv[3];
        }
        const glm::quat q(m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z);
        const glm::vec3 p(m.pose.position.x, m.pose.position.y, m.pose.position.z);
        for (int ix = 0; ix < VR_MIRROR_GRID; ix++) {
            for (int iy = 0; iy < VR_MIRROR_GRID; iy++) {
                // Image top-left = the quad's (-x, +y) corner.
                const float sx = (float)ix / (VR_MIRROR_GRID - 1) - 0.5f;
                const float sy = 0.5f - (float)iy / (VR_MIRROR_GRID - 1);
                const glm::vec3 w = p + q * glm::vec3(sx * m.size.width, sy * m.size.height, 0.0f);
                const glm::vec3 d = eq_inv * (w - ep);
                if (d.z > -0.01f) {
                    o.ok[ix][iy] = false;
                    continue;
                }
                const float tx = d.x / -d.z, ty = d.y / -d.z;
                o.pt[ix][iy][0] = (tx - tl) / (tr - tl);
                o.pt[ix][iy][1] = (tu - ty) / (tu - td);
                o.ok[ix][iy] = true;
            }
        }
    }
    return n;
}

void* vr_get_mirror_texture_id() {
    return xr.gfx ? xr.gfx->MirrorTextureId(vrgfx::kMirrorEye) : nullptr;
}

#else // !ENABLE_VR

// Stubs for builds without OpenXR
Fast::Interpreter* vr_get_interpreter() { return nullptr; }
bool vr_init() { return false; }
void vr_apply_mode_request() {}
void vr_shutdown() {}
bool vr_begin_frame() { return false; }
void vr_end_frame() {}
void vr_set_frame_plan(bool, bool, bool) {}
bool vr_should_render_eyes() { return true; }
bool vr_should_render_hud() { return true; }
bool vr_should_present_desktop() { return true; }
void vr_report_frame_times(float, float, float, float, bool) {}
void vr_report_game_tick_ms(float) {}
void vr_commit_pending_snap_turn() {}
void vr_get_frame_stats(struct VrFrameStats* out) {
    if (out != nullptr) {
        *out = {};
    }
}
void vr_begin_eye(int) {}
void vr_end_eye(int) {}
bool vr_multiview_enabled() { return false; }
void vr_begin_stereo() {}
void vr_end_stereo() {}
bool vr_is_multiview_pass() { return false; }
bool vr_multiview_eye_welded() { return false; }
void vr_set_multiview_eye_welded(bool) {}
const float* vr_get_multiview_eye_matrices(uint32_t* generation) {
    static const float kIdentity[32] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                                         1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    if (generation != nullptr) {
        *generation = 0;
    }
    return kIdentity;
}
void vr_get_projection_matrix(int, float out[4][4]) { memset(out, 0, sizeof(float) * 16); }
void vr_get_fog_ndc_z_params(float* a, float* b) { *a = *b = 0.0f; }
void vr_get_view_matrix(int, float out[4][4]) { memset(out, 0, sizeof(float) * 16); }
bool vr_is_initialized() { return false; }
int vr_get_current_eye() { return 0; }
void vr_get_recommended_resolution(uint32_t* w, uint32_t* h) { *w = 0; *h = 0; }
uint32_t vr_get_refresh_rate() { return 90; }
float vr_get_world_scale() { return 1.0f; }
void vr_set_world_scale(float) {}
void vr_set_link_eye_height(float) {}
void vr_set_view_fade(float) {}
void vr_set_first_person(bool) {}
bool vr_is_first_person() { return false; }
void vr_set_camera_anchor(float, float, float) {}
void vr_set_camera_yaw(int16_t) {}
void vr_get_camera_pose(float eye[3], float fwd[3], float up[3]) {
    eye[0] = eye[1] = eye[2] = 0.0f;
    fwd[0] = 0.0f; fwd[1] = 0.0f; fwd[2] = -1.0f;
    up[0] = 0.0f; up[1] = 1.0f; up[2] = 0.0f;
}
float vr_get_culling_fovy() { return 100.0f; }
void vr_get_roomscale_desired(float out[2]) { out[0] = out[1] = 0.0f; }
void vr_add_roomscale_displacement(float, float) {}
void vr_get_roomscale_origin(float out[2]) { out[0] = out[1] = 0.0f; }
void vr_reset_roomscale() {}
void vr_clamp_roomscale_lean(float) {}
bool vr_get_aim_ray(int, float out_pos[3], float out_dir[3]) {
    out_pos[0] = out_pos[1] = out_pos[2] = 0.0f;
    out_dir[0] = 0.0f;
    out_dir[1] = 0.0f;
    out_dir[2] = -1.0f;
    return false;
}
bool vr_get_hand_pose(int, float out_pos[3], float out_quat[4]) {
    out_pos[0] = out_pos[1] = out_pos[2] = 0.0f;
    out_quat[0] = out_quat[1] = out_quat[2] = 0.0f;
    out_quat[3] = 1.0f;
    return false;
}
bool vr_is_hand_active(int) { return false; }
uint16_t vr_get_controller_buttons(int) { return 0; }
void vr_get_thumbstick(int, float* x, float* y) { *x = *y = 0.0f; }
void vr_set_turn_suppressed(bool) {}
void vr_set_stick_suppressed(int, bool) {}
void vr_set_climb_view_lock(int, const float*, const float*, bool) {}
void vr_request_face_yaw(int16_t) {}
void vr_set_climb_view_limits(const float*, const float*, const float*) {}
bool vr_consume_climb_discontinuity() { return false; }
bool vr_get_hand_tracked(int, float out_units[3]) {
    out_units[0] = out_units[1] = out_units[2] = 0.0f;
    return false;
}
float vr_get_trigger(int) { return 0.0f; }
float vr_get_grip(int) { return 0.0f; }
bool vr_get_hand_matrix(int, float out[4][4]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = (r == c) ? 1.0f : 0.0f;
    return false;
}
void vr_set_hand_scale(float) {}
void vr_set_hand_mirror(int, bool) {}
void vr_trigger_haptic(int, float, float, float) {}
void vr_register_hand_matrix(const void*, int) {}
void vr_register_hand_child_matrix(const void*, int, const float*) {}
bool vr_get_head_matrix(float out[4][4]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = (r == c) ? 1.0f : 0.0f;
    return false;
}
void vr_register_head_child_matrix(const void*, const float*) {}
bool vr_get_playspace_matrix(float out[4][4]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = (r == c) ? 1.0f : 0.0f;
    return false;
}
void vr_register_playspace_child_matrix(const void*, const float*) {}
void vr_clear_hand_matrices() {}
bool vr_lookup_hand_matrix(const void*, float out[4][4]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r][c] = (r == c) ? 1.0f : 0.0f;
    return false;
}
int16_t vr_get_head_yaw() { return 0; }
int16_t vr_get_heading_yaw() { return 0; }
void vr_set_lockon_yaw(int16_t, bool) {}
void vr_recenter_heading(int16_t) {}
void vr_set_interp_alpha(float) {}
void vr_rebind_current_eye_target() {}
void vr_clear_current_eye_depth() {}
void vr_set_hud_commands(void*) {}
void* vr_get_hud_commands() { return nullptr; }
void vr_begin_hud() {}
void vr_end_hud() {}
bool vr_is_rendering_hud() { return false; }
void vr_set_text_commands(void*, const float*) {}
void* vr_get_text_commands() { return nullptr; }
void vr_set_text_panel_layout(float, float, float) {}
void vr_begin_text() {}
void vr_end_text() {}
bool vr_is_rendering_text() { return false; }
bool vr_menu_panel_active() { return false; }
bool vr_can_disable() { return true; }
int vr_get_supported_refresh_rates(float*, int) { return 0; }
void vr_menu_panel_size(int* w, int* h) { *w = 0; *h = 0; }
void vr_menu_take_pointer(VrMenuPointer* out) { *out = {}; }
bool vr_begin_menu() { return false; }
void vr_end_menu() {}
bool vr_wants_coverage_blend() { return false; }
void vr_hud_marker(uint32_t) {}
void vr_set_hud_world_panel(bool, const float*, float, float, float) {}
bool vr_hud_tri(const float*, float* out) { out[0] = out[3] = 1.0f; out[1] = out[2] = 0.0f; return true; }
void vr_set_hud_child(bool) {}
bool vr_get_hud_child() { return false; }
bool vr_is_rendering_screen() { return false; }
void vr_set_flat_screen(bool) {}
bool vr_get_flat_screen() { return false; }
void vr_begin_screen() {}
void vr_end_screen() {}
void vr_get_2d_target_size(uint32_t* w, uint32_t* h) { *w = 1024; *h = 768; }
void vr_capture_mirror() {}
void* vr_get_mirror_texture_id() { return nullptr; }
bool vr_get_mirror_flip_v() { return false; }
bool vr_backend_supported(int) { return false; }
bool vr_probe_required_adapter(int, uint64_t*) { return false; }
int vr_get_mirror_quads(VrMirrorQuad*, int) { return 0; }

#endif
