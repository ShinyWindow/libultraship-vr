#define NOMINMAX

#include "fast/vr_openxr.h"

#ifdef ENABLE_DX11

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <chrono>
#include <unordered_map>

#include <d3d11.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#define XR_USE_GRAPHICS_API_D3D11
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <spdlog/spdlog.h>

#include "libultraship/bridge/consolevariablebridge.h"
#include "ship/Context.h"
#include "fast/Fast3dWindow.h"
#include "fast/interpreter.h"
#include "fast/backends/gfx_direct3d_common.h"
#include "fast/vr_physics.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>

// --------------------------------------------------------------------------
// Engine glue (post-GfxPC-refactor): the renderer is Fast::Interpreter owned by Fast3dWindow, and
// the D3D11 device/context live as public members on GfxRenderingAPIDX11. These shims keep the
// pre-refactor accessor names used throughout this file.
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

static Fast::GfxRenderingAPIDX11* vr_dx11() {
    Fast::Interpreter* interp = vr_get_interpreter();
    if (!interp) {
        return nullptr;
    }
    return static_cast<Fast::GfxRenderingAPIDX11*>(interp->GetCurrentRenderingAPI());
}

static void* gfx_d3d11_get_device() {
    auto* dx = vr_dx11();
    return dx ? dx->mDevice.Get() : nullptr;
}

static void* gfx_d3d11_get_context() {
    auto* dx = vr_dx11();
    return dx ? dx->mContext.Get() : nullptr;
}

static void gfx_d3d11_set_render_target_height(uint32_t height) {
    if (auto* dx = vr_dx11()) {
        dx->SetRenderTargetHeight((int32_t)height);
    }
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

    // Swapchains (one per eye)
    struct EyeSwapchain {
        XrSwapchain handle;
        int64_t format;
        uint32_t width, height;
        std::vector<XrSwapchainImageD3D11KHR> images;
        std::vector<ComPtr<ID3D11RenderTargetView>> rtvs;
        std::vector<ComPtr<ID3D11DepthStencilView>> dsvs;
        std::vector<ComPtr<ID3D11Texture2D>> depth_textures;
    } eye_swapchains[2];

    // Per-frame
    XrFrameState frame_state;
    bool frame_began;
    int current_eye;
    uint32_t refresh_rate;  // Cached headset refresh in Hz, derived from predictedDisplayPeriod
    uint32_t current_image_index[2]; // Acquired swapchain image index per eye

    // Cached per-frame matrices (row-major, row-vector convention)
    float projection[2][4][4];
    float view[2][4][4];

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
    XrPosef grip_pose[2];
    XrPosef grip_pose_raw[2]; // untouched by snap-turn; for compositor-space quads (hand HUD)
    XrPosef aim_pose[2];
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

    // Desktop mirror: a copy of the left eye for display in the companion window. We can't sample the
    // swapchain image directly at present time (the runtime owns it once released), so the left eye is
    // copied here each frame while still acquired.
    ComPtr<ID3D11Texture2D> mirror_texture;
    ComPtr<ID3D11ShaderResourceView> mirror_srv;

    // D3D11 cached pointers
    ID3D11Device* d3d_device;
    ID3D11DeviceContext* d3d_context;

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
    float view_fade_target;
    float view_fade_current;
} xr = {};

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
// OpenXR session state event handling
// --------------------------------------------------------------------------

static void handle_session_state_change(XrSessionState new_state) {
    xr.session_state = new_state;

    switch (new_state) {
        case XR_SESSION_STATE_READY: {
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
            xr.initialized = false;
            spdlog::warn("[VR] Session lost or exiting");
            break;
        default:
            break;
    }
}

static void vr_reset_snap_turn(); // defined with the snap-turn state below

// Per-hand thumbstick suppression for modal hand gestures (Alyx-style item selector) —
// applied at the source in update_input, so every stick consumer inherits it.
static bool g_stick_suppressed[2] = { false, false };

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
            spdlog::info("[VR] Headset {}", xr.user_present ? "donned" : "doffed");
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
// Artificial snap-turn: an accumulated world-space yaw (rotation + the translation that keeps the
// pivot fixed) applied to every game-facing pose. Never applied to compositor-submitted poses.
// --------------------------------------------------------------------------

static glm::quat g_turn_rot(1.0f, 0.0f, 0.0f, 0.0f);
static glm::vec3 g_turn_off(0.0f);

// Drop the accumulated artificial turn (system recenter: it was expressed in old-space coords).
static void vr_reset_snap_turn() {
    g_turn_rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    g_turn_off = glm::vec3(0.0f);
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
// Matches vr_get_heading_yaw's convention exactly, manual offset included, so "the target is dead
// ahead" means the same thing to the framing code and to Link's steering.
static bool vr_pending_heading_yaw(int16_t* out) {
    const XrQuaternionf& q = xr.views[0].pose.orientation;
    const glm::quat gq = g_turn_rot * glm::quat(q.w, q.x, q.y, q.z);
    const glm::vec3 fwd = gq * glm::vec3(0.0f, 0.0f, -1.0f);
    if (fwd.x * fwd.x + fwd.z * fwd.z <= 1e-6f) {
        return false; // looking straight up or down: heading is degenerate, correct nothing
    }
    const float yaw = atan2f(fwd.x, fwd.z);
    const int16_t manual = static_cast<int16_t>(CVarGetInteger("gVrHeadingManualOffset", 0));
    *out = static_cast<int16_t>(static_cast<int16_t>(yaw * (32768.0f / 3.14159265358979323846f)) + manual);
    return true;
}

// --------------------------------------------------------------------------
// Lifecycle
// --------------------------------------------------------------------------

bool vr_init() {
    // Get D3D11 device
    xr.d3d_device = static_cast<ID3D11Device*>(gfx_d3d11_get_device());
    xr.d3d_context = static_cast<ID3D11DeviceContext*>(gfx_d3d11_get_context());
    if (!xr.d3d_device || !xr.d3d_context) {
        spdlog::error("[VR] D3D11 device not available");
        return false;
    }

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

    // --- Create Instance ---
    // Optional extensions are enabled only when the runtime offers them.
    xr.user_presence_supported = false;
    xr.user_present = true; // assume worn until the runtime says otherwise
    xr.color_scale_supported = false;
    xr.view_fade_target = xr.view_fade_current = 0.0f;
    {
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
        }
    }

    const char* extensions[3] = { XR_KHR_D3D11_ENABLE_EXTENSION_NAME };
    uint32_t extension_count = 1;
    if (xr.user_presence_supported) {
        extensions[extension_count++] = XR_EXT_USER_PRESENCE_EXTENSION_NAME;
    }
    if (xr.color_scale_supported) {
        extensions[extension_count++] = XR_KHR_COMPOSITION_LAYER_COLOR_SCALE_BIAS_EXTENSION_NAME;
    }

    XrInstanceCreateInfo instance_ci = { XR_TYPE_INSTANCE_CREATE_INFO };
    strcpy(instance_ci.applicationInfo.applicationName, "Ship of Harkinian VR");
    instance_ci.applicationInfo.applicationVersion = 1;
    strcpy(instance_ci.applicationInfo.engineName, "libultraship");
    instance_ci.applicationInfo.engineVersion = 1;
    instance_ci.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
    instance_ci.enabledExtensionCount = extension_count;
    instance_ci.enabledExtensionNames = extensions;

    if (!xr_check(xrCreateInstance(&instance_ci, &xr.instance), "xrCreateInstance")) {
        spdlog::error("[VR] Failed to create OpenXR instance. Make sure SteamVR is running.");
        return false;
    }

    // --- Get System ---
    XrSystemGetInfo system_info = { XR_TYPE_SYSTEM_GET_INFO };
    system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    if (!xr_check(xrGetSystem(xr.instance, &system_info, &xr.system_id), "xrGetSystem")) {
        xrDestroyInstance(xr.instance);
        xr.instance = XR_NULL_HANDLE;
        return false;
    }

    // --- Check D3D11 graphics requirements ---
    PFN_xrGetD3D11GraphicsRequirementsKHR xrGetD3D11GraphicsRequirementsKHR = nullptr;
    xrGetInstanceProcAddr(xr.instance, "xrGetD3D11GraphicsRequirementsKHR",
                          reinterpret_cast<PFN_xrVoidFunction*>(&xrGetD3D11GraphicsRequirementsKHR));

    XrGraphicsRequirementsD3D11KHR gfx_requirements = { XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR };
    if (xrGetD3D11GraphicsRequirementsKHR) {
        xrGetD3D11GraphicsRequirementsKHR(xr.instance, xr.system_id, &gfx_requirements);
    }

    // --- Create Session ---
    XrGraphicsBindingD3D11KHR d3d_binding = { XR_TYPE_GRAPHICS_BINDING_D3D11_KHR };
    d3d_binding.device = xr.d3d_device;

    XrSessionCreateInfo session_ci = { XR_TYPE_SESSION_CREATE_INFO };
    session_ci.next = &d3d_binding;
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

    // The game outputs gamma-encoded (sRGB) colors. The swapchain must be created with an
    // SRGB format so the compositor decodes them correctly; a UNORM swapchain makes the
    // compositor treat gamma values as linear and re-encode them, washing the image out.
    // Writes still go through a UNORM view (below) so the bits land in the texture verbatim.
    int64_t chosen_format = formats[0]; // fallback to first supported
    for (int64_t fmt : formats) {
        if (fmt == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) {
            chosen_format = fmt;
            break;
        }
    }
    if (chosen_format != DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) {
        // SRGB not available — pick UNORM as next best, accept the gamma mismatch for now
        for (int64_t fmt : formats) {
            if (fmt == DXGI_FORMAT_R8G8B8A8_UNORM) {
                chosen_format = fmt;
                break;
            }
        }
    }
    // OpenXR D3D11 swapchain textures are allocated typeless, so views may use either
    // variant of the format family. Using the UNORM variant for RTVs/SRVs stores and
    // reads the game's already-gamma-encoded output without any extra conversion.
    DXGI_FORMAT view_format = (chosen_format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB)
                                  ? DXGI_FORMAT_R8G8B8A8_UNORM
                                  : static_cast<DXGI_FORMAT>(chosen_format);
    spdlog::info("[VR] Swapchain format: {} (UNORM={}, SRGB={}), view format: {}",
                 chosen_format, (int)DXGI_FORMAT_R8G8B8A8_UNORM, (int)DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                 (int)view_format);

    // --- Create Swapchains (one per eye) ---
    for (uint32_t eye = 0; eye < 2; eye++) {
        auto& sc = xr.eye_swapchains[eye];

        // Apply the resolution multiplier, then clamp to what the runtime allows.
        uint32_t scaled_w = (uint32_t)lroundf(xr.config_views[eye].recommendedImageRectWidth * xr.resolution_scale);
        uint32_t scaled_h = (uint32_t)lroundf(xr.config_views[eye].recommendedImageRectHeight * xr.resolution_scale);
        if (scaled_w < 1) scaled_w = 1;
        if (scaled_h < 1) scaled_h = 1;
        if (scaled_w > xr.config_views[eye].maxImageRectWidth) scaled_w = xr.config_views[eye].maxImageRectWidth;
        if (scaled_h > xr.config_views[eye].maxImageRectHeight) scaled_h = xr.config_views[eye].maxImageRectHeight;

        sc.width = scaled_w;
        sc.height = scaled_h;
        sc.format = chosen_format;

        spdlog::info("[VR] Eye {} render resolution: {}x{} (recommended {}x{}, scale {:.2f})", eye, sc.width, sc.height,
                     xr.config_views[eye].recommendedImageRectWidth, xr.config_views[eye].recommendedImageRectHeight,
                     xr.resolution_scale);

        XrSwapchainCreateInfo swapchain_ci = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
        swapchain_ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
        swapchain_ci.format = chosen_format;
        swapchain_ci.sampleCount = 1;
        swapchain_ci.width = sc.width;
        swapchain_ci.height = sc.height;
        swapchain_ci.faceCount = 1;
        swapchain_ci.arraySize = 1;
        swapchain_ci.mipCount = 1;

        if (!xr_check(xrCreateSwapchain(xr.session, &swapchain_ci, &sc.handle), "xrCreateSwapchain")) {
            vr_shutdown();
            return false;
        }

        // Enumerate swapchain images
        uint32_t image_count = 0;
        xrEnumerateSwapchainImages(sc.handle, 0, &image_count, nullptr);
        sc.images.resize(image_count, { XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR });
        xrEnumerateSwapchainImages(sc.handle, image_count, &image_count,
                                   reinterpret_cast<XrSwapchainImageBaseHeader*>(sc.images.data()));

        // Create RTVs and depth resources for each swapchain image
        sc.rtvs.resize(image_count);
        sc.dsvs.resize(image_count);
        sc.depth_textures.resize(image_count);

        for (uint32_t i = 0; i < image_count; i++) {
            // RTV
            D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
            rtv_desc.Format = view_format;
            rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            rtv_desc.Texture2D.MipSlice = 0;
            HRESULT hr = xr.d3d_device->CreateRenderTargetView(
                sc.images[i].texture, &rtv_desc, sc.rtvs[i].GetAddressOf());
            if (FAILED(hr)) {
                spdlog::error("[VR] Failed to create RTV for eye {} image {}", eye, i);
                vr_shutdown();
                return false;
            }

            // Depth texture
            D3D11_TEXTURE2D_DESC depth_desc = {};
            depth_desc.Width = sc.width;
            depth_desc.Height = sc.height;
            depth_desc.MipLevels = 1;
            depth_desc.ArraySize = 1;
            depth_desc.Format = DXGI_FORMAT_D32_FLOAT;
            depth_desc.SampleDesc.Count = 1;
            depth_desc.Usage = D3D11_USAGE_DEFAULT;
            depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;

            hr = xr.d3d_device->CreateTexture2D(&depth_desc, nullptr, sc.depth_textures[i].GetAddressOf());
            if (FAILED(hr)) {
                spdlog::error("[VR] Failed to create depth texture for eye {} image {}", eye, i);
                vr_shutdown();
                return false;
            }

            // DSV
            D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
            dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
            dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
            dsv_desc.Texture2D.MipSlice = 0;

            hr = xr.d3d_device->CreateDepthStencilView(
                sc.depth_textures[i].Get(), &dsv_desc, sc.dsvs[i].GetAddressOf());
            if (FAILED(hr)) {
                spdlog::error("[VR] Failed to create DSV for eye {} image {}", eye, i);
                vr_shutdown();
                return false;
            }
        }

        spdlog::info("[VR] Eye {} swapchain: {}x{}, {} images", eye, sc.width, sc.height, image_count);
    }

    // --- Create desktop mirror texture (a copy of the left eye, shown in the companion window) ---
    {
        const auto& eye0 = xr.eye_swapchains[0];
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = eye0.width;
        desc.Height = eye0.height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = view_format;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        HRESULT hr = xr.d3d_device->CreateTexture2D(&desc, nullptr, xr.mirror_texture.ReleaseAndGetAddressOf());
        if (SUCCEEDED(hr)) {
            D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
            srv_desc.Format = view_format;
            srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            srv_desc.Texture2D.MipLevels = 1;
            hr = xr.d3d_device->CreateShaderResourceView(xr.mirror_texture.Get(), &srv_desc,
                                                         xr.mirror_srv.ReleaseAndGetAddressOf());
        }
        if (FAILED(hr)) {
            // Non-fatal: the headset still renders, the companion window just won't show the mirror.
            spdlog::warn("[VR] Failed to create desktop mirror texture; companion window will be blank");
            xr.mirror_texture.Reset();
            xr.mirror_srv.Reset();
        } else {
            spdlog::info("[VR] Desktop mirror texture: {}x{}", eye0.width, eye0.height);
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

    // --- Create HUD swapchain (1024x768, 4:3) ---
    {
        auto& sc = xr.hud_swapchain;
        sc.width = 1024;
        sc.height = 768;
        sc.format = chosen_format;

        XrSwapchainCreateInfo swapchain_ci = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
        swapchain_ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
        swapchain_ci.format = chosen_format;
        swapchain_ci.sampleCount = 1;
        swapchain_ci.width = sc.width;
        swapchain_ci.height = sc.height;
        swapchain_ci.faceCount = 1;
        swapchain_ci.arraySize = 1;
        swapchain_ci.mipCount = 1;

        if (!xr_check(xrCreateSwapchain(xr.session, &swapchain_ci, &sc.handle), "xrCreateSwapchain (HUD)")) {
            vr_shutdown();
            return false;
        }

        uint32_t image_count = 0;
        xrEnumerateSwapchainImages(sc.handle, 0, &image_count, nullptr);
        sc.images.resize(image_count, { XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR });
        xrEnumerateSwapchainImages(sc.handle, image_count, &image_count,
                                   reinterpret_cast<XrSwapchainImageBaseHeader*>(sc.images.data()));

        sc.rtvs.resize(image_count);
        sc.dsvs.resize(image_count);
        sc.depth_textures.resize(image_count);

        for (uint32_t i = 0; i < image_count; i++) {
            D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
            rtv_desc.Format = view_format;
            rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            xr.d3d_device->CreateRenderTargetView(sc.images[i].texture, &rtv_desc, sc.rtvs[i].GetAddressOf());

            D3D11_TEXTURE2D_DESC depth_desc = {};
            depth_desc.Width = sc.width;
            depth_desc.Height = sc.height;
            depth_desc.MipLevels = 1;
            depth_desc.ArraySize = 1;
            depth_desc.Format = DXGI_FORMAT_D32_FLOAT;
            depth_desc.SampleDesc.Count = 1;
            depth_desc.Usage = D3D11_USAGE_DEFAULT;
            depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
            xr.d3d_device->CreateTexture2D(&depth_desc, nullptr, sc.depth_textures[i].GetAddressOf());

            D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
            dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
            dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
            xr.d3d_device->CreateDepthStencilView(sc.depth_textures[i].Get(), &dsv_desc, sc.dsvs[i].GetAddressOf());
        }
        spdlog::info("[VR] HUD swapchain: {}x{}, {} images", sc.width, sc.height, image_count);
    }

    // --- Create flat-screen swapchain (whole-frame panel for 2D contexts: file select, pause) ---
    {
        auto& sc = xr.screen_swapchain;
        sc.width = 1280;
        sc.height = 960;
        sc.format = chosen_format;

        XrSwapchainCreateInfo swapchain_ci = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
        swapchain_ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
        swapchain_ci.format = chosen_format;
        swapchain_ci.sampleCount = 1;
        swapchain_ci.width = sc.width;
        swapchain_ci.height = sc.height;
        swapchain_ci.faceCount = 1;
        swapchain_ci.arraySize = 1;
        swapchain_ci.mipCount = 1;

        if (!xr_check(xrCreateSwapchain(xr.session, &swapchain_ci, &sc.handle), "xrCreateSwapchain (screen)")) {
            vr_shutdown();
            return false;
        }

        uint32_t image_count = 0;
        xrEnumerateSwapchainImages(sc.handle, 0, &image_count, nullptr);
        sc.images.resize(image_count, { XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR });
        xrEnumerateSwapchainImages(sc.handle, image_count, &image_count,
                                   reinterpret_cast<XrSwapchainImageBaseHeader*>(sc.images.data()));

        sc.rtvs.resize(image_count);
        sc.dsvs.resize(image_count);
        sc.depth_textures.resize(image_count);

        for (uint32_t i = 0; i < image_count; i++) {
            D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
            rtv_desc.Format = view_format;
            rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            xr.d3d_device->CreateRenderTargetView(sc.images[i].texture, &rtv_desc, sc.rtvs[i].GetAddressOf());

            D3D11_TEXTURE2D_DESC depth_desc = {};
            depth_desc.Width = sc.width;
            depth_desc.Height = sc.height;
            depth_desc.MipLevels = 1;
            depth_desc.ArraySize = 1;
            depth_desc.Format = DXGI_FORMAT_D32_FLOAT;
            depth_desc.SampleDesc.Count = 1;
            depth_desc.Usage = D3D11_USAGE_DEFAULT;
            depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
            xr.d3d_device->CreateTexture2D(&depth_desc, nullptr, sc.depth_textures[i].GetAddressOf());

            D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
            dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
            dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
            xr.d3d_device->CreateDepthStencilView(sc.depth_textures[i].Get(), &dsv_desc, sc.dsvs[i].GetAddressOf());
        }
        spdlog::info("[VR] Screen swapchain: {}x{}, {} images", sc.width, sc.height, image_count);
    }

    // Initialize views
    xr.views[0] = { XR_TYPE_VIEW };
    xr.views[1] = { XR_TYPE_VIEW };

    xr.initialized = true;
    xr.enabled = true;
    spdlog::info("[VR] OpenXR initialized successfully");
    return true;
}

void vr_shutdown() {
    xr.initialized = false;
    xr.session_running = false;
    vrphys_reset();

    for (uint32_t eye = 0; eye < 2; eye++) {
        auto& sc = xr.eye_swapchains[eye];
        sc.rtvs.clear();
        sc.dsvs.clear();
        sc.depth_textures.clear();
        sc.images.clear();
        if (sc.handle != XR_NULL_HANDLE) {
            xrDestroySwapchain(sc.handle);
            sc.handle = XR_NULL_HANDLE;
        }
    }

    {
        auto& sc = xr.hud_swapchain;
        sc.rtvs.clear(); sc.dsvs.clear(); sc.depth_textures.clear(); sc.images.clear();
        if (sc.handle != XR_NULL_HANDLE) { xrDestroySwapchain(sc.handle); sc.handle = XR_NULL_HANDLE; }
    }

    {
        auto& sc = xr.screen_swapchain;
        sc.rtvs.clear(); sc.dsvs.clear(); sc.depth_textures.clear(); sc.images.clear();
        if (sc.handle != XR_NULL_HANDLE) { xrDestroySwapchain(sc.handle); sc.handle = XR_NULL_HANDLE; }
    }
    xr.eyes_ever_rendered = false;
    xr.flat_screen = false;
    xr.flat_screen_prev = false;

    xr.mirror_srv.Reset();
    xr.mirror_texture.Reset();
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

    if (!xr.frame_state.shouldRender) {
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
        return false;
    }

    // Sync controllers + locate hand poses for this frame (motion controls).
    update_input();

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
    if (xr.input_initialized && !xr.flat_screen && xr.first_person && CVarGetInteger("gVrSnapTurnOn", 1)) {
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
            vr_apply_snap_turn(snap_latch * CVarGetFloat("gVrSnapTurnDegrees", 45.0f));
        } else if (snap_latch != 0 && fabsf(sx) < 0.3f) {
            snap_latch = 0;
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

    // Apply the accumulated snap-turn to every game-facing pose, preserving the raw view poses for
    // layer submission in vr_end_frame. Hand poses are only adjusted when freshly located this frame
    // (a stale pose already carries the previous turn and would be double-rotated). The submit
    // pose/FOV are refreshed ONLY on frames whose eye images we are about to redraw: in flat-screen
    // mode, and on stereo-divisor reprojection frames, the projection layer keeps re-submitting the
    // last world frame described by the frustum it was rendered from, so the compositor reprojects
    // it correctly instead of stretching it onto a pose it never matched.
    const bool refresh_submit = !xr.flat_screen && xr.plan_render_eyes;
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
                                     ? glm::mix(xr.anchor_prev, xr.anchor, xr.interp_alpha)
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

    return true;
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
        projection_views[eye].subImage.swapchain = xr.eye_swapchains[eye].handle;
        projection_views[eye].subImage.imageRect.offset = { 0, 0 };
        projection_views[eye].subImage.imageRect.extent = {
            static_cast<int32_t>(xr.eye_swapchains[eye].width),
            static_cast<int32_t>(xr.eye_swapchains[eye].height)
        };
        projection_views[eye].subImage.imageArrayIndex = 0;
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

    // HUD quad layer (alpha-blended). Attachment via gVrHudAttach: 0 = head-locked (classic),
    // 1/2 = pinned to the left/right controller like a wrist panel. Hand modes use the RAW grip
    // pose in local_space (compositor quads must not carry the artificial snap-turn) and fall back
    // to head-locked while that hand is untracked.
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

    float hud_width;
    const int hud_attach = CVarGetInteger("gVrHudAttach", 0);
    const int hud_hand = hud_attach - 1;
    if ((hud_attach == 1 || hud_attach == 2) && xr.hand_active[hud_hand]) {
        const float kDeg = 3.14159265358979323846f / 180.0f;
        const XrPosef& gp = xr.grip_pose_raw[hud_hand];
        const glm::quat gq(gp.orientation.w, gp.orientation.x, gp.orientation.y, gp.orientation.z);
        // Positional offset in the grip frame (meters), mirrored in X for the right hand so one
        // tuning works symmetrically on either side.
        glm::vec3 off(CVarGetFloat("gVrHudHandOffX", 0.0f), CVarGetFloat("gVrHudHandOffY", 0.10f),
                      CVarGetFloat("gVrHudHandOffZ", -0.08f));
        if (hud_hand == 1) {
            off.x = -off.x;
        }
        const glm::vec3 p = glm::vec3(gp.position.x, gp.position.y, gp.position.z) + gq * off;
        // Tilt about the grip X axis so the panel faces the player's eyes at a natural wrist angle.
        const glm::quat q =
            gq * glm::angleAxis(CVarGetFloat("gVrHudHandPitch", -40.0f) * kDeg, glm::vec3(1.0f, 0.0f, 0.0f));
        hud_layer.space = xr.local_space;
        hud_layer.pose.position = { p.x, p.y, p.z };
        hud_layer.pose.orientation = { q.x, q.y, q.z, q.w };
        hud_width = CVarGetFloat("gVrHudHandSize", 0.35f);
    } else {
        hud_layer.space = xr.view_space;
        hud_layer.pose = { { 0, 0, 0, 1 },
                           { CVarGetFloat("gVrHudOffX", 0.0f), CVarGetFloat("gVrHudOffY", 0.0f),
                             -CVarGetFloat("gVrHudDistance", 2.0f) } };
        hud_width = CVarGetFloat("gVrHudSize", 1.5f);
    }
    if (hud_width < 0.05f) {
        hud_width = 0.05f;
    }
    hud_layer.size = { hud_width, hud_width * 0.75f }; // 4:3, matching the HUD swapchain

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

    // Assemble layers back-to-front. The projection (world) layer is only submitted once its
    // swapchains have ever been rendered (at boot we go straight into flat-screen file select).
    const XrCompositionLayerBaseHeader* layers[3];
    uint32_t layer_count = 0;
    if (xr.eyes_ever_rendered) {
        layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection_layer);
    }
    if (xr.flat_screen && xr.screen_ever_rendered) {
        layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&screen_layer);
    }
    // Same guard as the projection layer: the HUD quad's swapchain is uninitialised until the
    // first HUD pass, and with per-tick HUD rendering that may be a few frames in. Also skip while
    // the game has detached the overlay (hud_commands NULL — flat-screen contexts route it into
    // the panel instead), so a stale HUD image doesn't float over the pause menu.
    if (xr.hud_ever_rendered && xr.hud_commands != nullptr) {
        layers[layer_count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&hud_layer);
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

    // Bind render target
    ID3D11RenderTargetView* rtv = sc.rtvs[image_index].Get();
    ID3D11DepthStencilView* dsv = sc.dsvs[image_index].Get();
    xr.d3d_context->OMSetRenderTargets(1, &rtv, dsv);

    // Clear
    float clear_color[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    xr.d3d_context->ClearRenderTargetView(rtv, clear_color);
    xr.d3d_context->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);

    // Set viewport
    D3D11_VIEWPORT viewport = {};
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    viewport.Width = static_cast<float>(sc.width);
    viewport.Height = static_cast<float>(sc.height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    xr.d3d_context->RSSetViewports(1, &viewport);

    // Tell D3D11 backend the render target height so viewport Y-flip works correctly
    gfx_d3d11_set_render_target_height(sc.height);
    // Interpreter renders at the eye texture's size (replaces the old gfx_start_frame override)
    vr_apply_dimensions(sc.width, sc.height);
}

void vr_end_eye(int eye) {
    if (!xr.initialized) return;

    // Grab the left eye for the desktop mirror while its swapchain image is still acquired — once
    // released below, the runtime owns the texture again and it's no longer safe to read. Skipped
    // on frames the companion window isn't presenting: this is a full-eye-resolution CopyResource
    // (tens of MB) and nothing would consume the result.
    if (eye == 0 && xr.plan_present_desktop) {
        vr_capture_mirror();
    }

    auto& sc = xr.eye_swapchains[eye];
    XrSwapchainImageReleaseInfo release_info = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    xr_check(xrReleaseSwapchainImage(sc.handle, &release_info), "xrReleaseSwapchainImage");
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
    const glm::vec3 a = glm::mix(xr.anchor_prev, xr.anchor, xr.interp_alpha);
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

// Latch a pending VR<->flat mode request (CVar gVrEnabled). MUST be called at a game-tick boundary
// only (graph.c, before the tick's display list is built): a DL built for one mode must never be
// interpreted in the other, and toggling between xrBeginFrame/xrEndFrame would corrupt the session.
void vr_apply_mode_request() {
    const bool want = CVarGetInteger("gVrEnabled", 1) != 0;

    if (want && !xr.initialized) {
        // Lazy init: launching with VR off never touches OpenXR (no SteamVR popup); the session is
        // created the first time the player toggles in.
        if (vr_init()) {
            xr.reenable_fixup = true;
        } else {
            // No runtime/headset available — flip the CVar back so the UI reflects reality.
            CVarSetInteger("gVrEnabled", 0);
        }
        return;
    }

    if (!xr.initialized) {
        return;
    }

    // Headset presence folds into the desired mode: doffing the headset auto-drops to flat play,
    // donning it again auto-resumes — unless the player opted to stay in VR (gVrStayOnDoff), the
    // runtime can't report presence, or VR is manually off anyway. Because this recomputes every
    // tick from (CVar, presence), manual F9 stays sticky while presence flips are symmetric.
    const bool stay_on_doff = CVarGetInteger("gVrStayOnDoff", 0) != 0;
    const bool effective = want && (xr.user_present || stay_on_doff || !xr.user_presence_supported);

    if (effective && !xr.enabled) {
        xr.enabled = true;
        xr.reenable_fixup = true;
    } else if (!effective && xr.enabled) {
        xr.enabled = false;
        // The overlay DL pointer goes stale immediately (graph.c stops re-arming it in flat mode).
        xr.hud_commands = nullptr;
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
                                 ? glm::mix(xr.anchor_prev, xr.anchor, xr.interp_alpha)
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
                                 ? glm::mix(xr.anchor_prev, xr.anchor, xr.interp_alpha)
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
                                 ? glm::mix(xr.anchor_prev, xr.anchor, xr.interp_alpha)
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
        anchor = glm::mix(xr.anchor_prev, xr.anchor, xr.interp_alpha);
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

void vr_clear_hand_matrices() {
    g_hand_mtx_registry.clear();
    g_hand_child_registry.clear();
    g_head_child_registry.clear();
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
    const int16_t manual = static_cast<int16_t>(CVarGetInteger("gVrHeadingManualOffset", 0));
    return static_cast<int16_t>(s_last_heading + manual);
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

void vr_rebind_current_eye_target() {
    if (!xr.initialized || !xr.frame_began) return;

    // Restore whichever target is ACTUALLY being rendered: the flat-screen panel or the HUD when a
    // 2D pass is active (the pause menu runs framebuffer copies mid-pass — blindly rebinding an eye
    // here used to dump the inventory into the stale right-eye image), else the current eye.
    ID3D11RenderTargetView* rtv;
    ID3D11DepthStencilView* dsv;
    uint32_t height;
    if (xr.rendering_screen) {
        auto& sc = xr.screen_swapchain;
        rtv = sc.rtvs[xr.screen_image_index].Get();
        dsv = sc.dsvs[xr.screen_image_index].Get();
        height = sc.height;
    } else if (xr.rendering_hud) {
        auto& sc = xr.hud_swapchain;
        rtv = sc.rtvs[xr.hud_image_index].Get();
        dsv = sc.dsvs[xr.hud_image_index].Get();
        height = sc.height;
    } else {
        auto& sc = xr.eye_swapchains[xr.current_eye];
        uint32_t idx = xr.current_image_index[xr.current_eye];
        rtv = sc.rtvs[idx].Get();
        dsv = sc.dsvs[idx].Get();
        height = sc.height;
    }
    xr.d3d_context->OMSetRenderTargets(1, &rtv, dsv);
    gfx_d3d11_set_render_target_height(height);
}

// --------------------------------------------------------------------------
// HUD overlay
// --------------------------------------------------------------------------

void vr_set_hud_commands(void* commands) { xr.hud_commands = commands; }
void* vr_get_hud_commands() { return xr.hud_commands; }

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

    ID3D11RenderTargetView* rtv = sc.rtvs[image_index].Get();
    ID3D11DepthStencilView* dsv = sc.dsvs[image_index].Get();
    xr.d3d_context->OMSetRenderTargets(1, &rtv, dsv);

    float clear_color[] = { 0.0f, 0.0f, 0.0f, 0.0f }; // Transparent
    xr.d3d_context->ClearRenderTargetView(rtv, clear_color);
    xr.d3d_context->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);

    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(sc.width);
    viewport.Height = static_cast<float>(sc.height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    xr.d3d_context->RSSetViewports(1, &viewport);

    gfx_d3d11_set_render_target_height(sc.height);
    vr_apply_dimensions(sc.width, sc.height);
}

void vr_end_hud() {
    if (!xr.initialized) return;
    xr.rendering_hud = false;

    XrSwapchainImageReleaseInfo release_info = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    xr_check(xrReleaseSwapchainImage(xr.hud_swapchain.handle, &release_info), "xrReleaseSwapchainImage (HUD)");

    vr_restore_eye_dimensions();
}

bool vr_is_rendering_hud() { return xr.rendering_hud; }

bool vr_is_rendering_screen() { return xr.rendering_screen; }

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

    ID3D11RenderTargetView* rtv = sc.rtvs[image_index].Get();
    ID3D11DepthStencilView* dsv = sc.dsvs[image_index].Get();
    xr.d3d_context->OMSetRenderTargets(1, &rtv, dsv);

    float clear_color[] = { 0.0f, 0.0f, 0.0f, 1.0f }; // Opaque black
    xr.d3d_context->ClearRenderTargetView(rtv, clear_color);
    xr.d3d_context->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);

    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(sc.width);
    viewport.Height = static_cast<float>(sc.height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    xr.d3d_context->RSSetViewports(1, &viewport);

    gfx_d3d11_set_render_target_height(sc.height);
    vr_apply_dimensions(sc.width, sc.height);
}

void vr_end_screen() {
    if (!xr.initialized) return;
    xr.rendering_hud = false;
    xr.rendering_screen = false;

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
    } else {
        *w = xr.hud_swapchain.width;
        *h = xr.hud_swapchain.height;
    }
}

// --------------------------------------------------------------------------
// Desktop mirror
// --------------------------------------------------------------------------

void vr_capture_mirror() {
    if (!xr.initialized || !xr.mirror_texture) return;

    auto& sc = xr.eye_swapchains[0];
    uint32_t idx = xr.current_image_index[0];
    if (idx >= sc.images.size()) return;

    ID3D11Texture2D* src = sc.images[idx].texture;
    if (src) {
        // Mirror texture was created with the same format/size as the eye swapchain image, so a
        // straight resource copy is valid (no shader blit needed).
        xr.d3d_context->CopyResource(xr.mirror_texture.Get(), src);
    }
}

void* vr_get_mirror_texture_id() {
    return xr.mirror_srv.Get();
}

#else // !ENABLE_DX11

// Stubs for non-D3D11 builds
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
void vr_get_frame_stats(struct VrFrameStats* out) {
    if (out != nullptr) {
        *out = {};
    }
}
void vr_begin_eye(int) {}
void vr_end_eye(int) {}
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
void vr_set_hud_commands(void*) {}
void* vr_get_hud_commands() { return nullptr; }
void vr_begin_hud() {}
void vr_end_hud() {}
bool vr_is_rendering_hud() { return false; }
bool vr_is_rendering_screen() { return false; }
void vr_set_flat_screen(bool) {}
bool vr_get_flat_screen() { return false; }
void vr_begin_screen() {}
void vr_end_screen() {}
void vr_get_2d_target_size(uint32_t* w, uint32_t* h) { *w = 1024; *h = 768; }
void vr_capture_mirror() {}
void* vr_get_mirror_texture_id() { return nullptr; }

#endif
