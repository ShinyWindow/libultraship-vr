#pragma once

#include <stdbool.h>
#include <stdint.h>

namespace Fast {
class Interpreter;
}

// The Interpreter this VR layer renders through (null before the window exists). Exposed so the
// interpreter/window hooks and the VR layer agree on one lookup path.
Fast::Interpreter* vr_get_interpreter();

// Lifecycle
// Whether VR can run on a renderer (Fast::WindowBackend id): one table for everything that decides
// whether VR may start (eager init, vr_apply_mode_request, the game's renderer popups).
bool vr_backend_supported(int window_backend);
// Call BEFORE the renderer creates its device, only when VR will start at launch: asks the OpenXR
// runtime which GPU the headset needs (D3D11: the adapter LUID, (HighPart << 32) | LowPart) so the
// device is created there. False when unknown (headset not available yet, API without an adapter
// requirement). Leaves the instance alive for vr_init to reuse.
bool vr_probe_required_adapter(int window_backend, uint64_t* luid);
bool vr_init();
void vr_shutdown();
// Latch a pending VR<->flat mode request (CVar gVrEnabled). Call ONLY at a game-tick boundary,
// before the tick's display list is built. Lazily creates the OpenXR session on first enable.
void vr_apply_mode_request();

// Per-frame
bool vr_begin_frame();
void vr_end_frame();

// Frame plan. The window layer decides once per XR frame — BEFORE vr_begin_frame, which latches
// the submit poses from it — which of the expensive per-frame jobs actually run. A skipped eye or
// HUD pass keeps its previous swapchain image and is resubmitted against the pose/FOV it was
// rendered with, so the compositor reprojects it onto the live head pose; head tracking stays at
// full headset rate while the world content updates at the reduced rate. This is the same
// mechanism flat-screen mode uses to keep the frozen world head-tracked behind the menu panel.
void vr_set_frame_plan(bool render_eyes, bool render_hud, bool present_desktop);
bool vr_should_render_eyes();
bool vr_should_render_hud();
bool vr_should_present_desktop();

// Per-frame timing, in milliseconds, exponentially smoothed. Fed by the window layer (which is the
// only place that sees the whole frame) and by vr_begin_frame for the xrWaitFrame block. Displayed
// by the VR Settings performance section; purely diagnostic.
struct VrFrameStats {
    float wait_ms;    // blocked inside xrWaitFrame — this is spare headroom, higher is better
    float eyes_ms;    // both eye display-list passes
    float hud_ms;     // HUD quad pass
    float desktop_ms; // companion window: ImGui + mirror blit + Present
    float frame_ms;   // whole DrawAndRunGraphicsCommands
    float tick_ms;    // GameState_Update, per 20 Hz game tick
    float eye_hz;     // eye passes actually rendered per second
    float frame_hz;   // XR frames submitted per second
};
void vr_report_frame_times(float eyes_ms, float hud_ms, float desktop_ms, float frame_ms, bool rendered_eyes);
void vr_report_game_tick_ms(float tick_ms);
// Commit a snap turn latched by the right stick; called at the start of each game tick so the
// tick culls and records for the turned heading.
void vr_commit_pending_snap_turn();
void vr_get_frame_stats(struct VrFrameStats* out);

// Per-eye
void vr_begin_eye(int eye);
void vr_end_eye(int eye);

// Matrix queries (used by gfx_pc.cpp matrix injection)
void vr_get_projection_matrix(int eye, float out[4][4]);
void vr_get_view_matrix(int eye, float out[4][4]);
// Fog z/w as a near-10 eye projection would produce it: z/w = a - b * (1/w). Keeps fog independent
// of the real near plane.
void vr_get_fog_ndc_z_params(float* a, float* b);

// State queries
bool vr_is_initialized();
int vr_get_current_eye();
void vr_get_recommended_resolution(uint32_t* width, uint32_t* height);
// Headset display refresh rate in Hz (e.g. 72/90/120). Used to pace the game's fixed-timestep
// logic via the interpolation system. Returns a sane default before the first frame is located.
uint32_t vr_get_refresh_rate();
// Display refresh rates the headset offers (XR_FB_display_refresh_rate: Meta's runtimes), read at
// session start; 0 where the runtime doesn't offer a choice. gVrRefreshRate (Hz, 0 = the headset's
// default) picks one of them.
int vr_get_supported_refresh_rates(float* out, int max);
float vr_get_world_scale();
void vr_set_world_scale(float units_per_meter);
// Link's standing eye height in game units, pushed each first-person frame. Auto world scale
// derives units/meter from this and the player's measured physical eye height (STAGE floor).
void vr_set_link_eye_height(float units);
// Alyx-style in-wall fade target (0 clear .. 1 black); world layer only, smoothed per XR frame.
void vr_set_view_fade(float fade);

// First-person camera
void vr_set_first_person(bool enabled);
bool vr_is_first_person();
void vr_set_camera_anchor(float x, float y, float z);
// Third person: base yaw of the playspace frame = the game camera's facing (binang). Facing
// tracking-forward in the headset then looks where the stock camera looks. Reset by
// vr_set_first_person(true).
void vr_set_camera_yaw(int16_t yaw_binang);
int16_t vr_get_head_yaw();

// Camera unification (Phase 3): report the rendered HMD pose to the game so its CPU-side camera
// systems (frustum culling, audio panning, projected-position/LOD) agree with what the player sees.
// vr_get_camera_pose returns the center-eye pose in game-world coords (eye position + forward/up
// unit direction vectors). vr_get_culling_fovy returns a vertical FOV (degrees) wide enough to cover
// the whole binocular VR view, so peripheral geometry isn't culled. Only meaningful while
// first-person is active.
void vr_get_camera_pose(float eye[3], float fwd[3], float up[3]);
float vr_get_culling_fovy();

// Roomscale 6DOF (physical translation moves Link's body, collision-swept). roomscale_origin is the
// horizontal physical-walk displacement (game units) baked into the body; the game advances it only
// by the body's collision-limited achieved move. See vr_roomscale_6dof plan.
void vr_get_roomscale_desired(float out[2]);
void vr_add_roomscale_displacement(float dx, float dz);
void vr_get_roomscale_origin(float out[2]);
void vr_reset_roomscale();
// Bound how far the camera may sit from Link's body horizontally; first person passes the body-move
// deadzone, so physical motion the body couldn't follow is discarded. <= 0 disables.
void vr_clamp_roomscale_lean(float max_units);

// Motion controls (OpenXR action sets). hand: 0 = left, 1 = right. Hand poses are in game-world
// coords (same anchor + world_scale transform as the camera); out_quat is x,y,z,w. See the
// vr_motion_controls plan.
bool vr_get_hand_pose(int hand, float out_pos[3], float out_quat[4]);
// Controller aim ray (the runtime's calibrated pointing pose) in game-world coords:
// origin + unit forward. For weapon aiming. False (and dir = -Z) if untracked.
bool vr_get_aim_ray(int hand, float out_pos[3], float out_dir[3]);
bool vr_is_hand_active(int hand);
uint16_t vr_get_controller_buttons(int hand);
void vr_get_thumbstick(int hand, float* x, float* y);
// Modal hand gestures (Alyx-style item selector): while suppressed, this hand's thumbstick
// reads centered at the source — movement, turning and stick C-buttons all inherit it.
void vr_set_stick_suppressed(int hand, bool suppressed);
// Artificial turning (snap/smooth) off while a stereo game menu owns the right stick.
void vr_set_turn_suppressed(bool suppressed);
// Physical climbing: see VR_SetClimbViewLock / VR_GetHandTracked (vr_interface.h).
void vr_set_climb_view_lock(int hand, const float ref_units[3], const float wall_normal[3], bool lateral);
bool vr_get_hand_tracked(int hand, float out_units[3]);
void vr_request_face_yaw(int16_t yaw_binang);
void vr_set_climb_view_limits(const float wall_out[3], const float lo[3], const float hi[3]);
bool vr_consume_climb_discontinuity();
float vr_get_trigger(int hand);
float vr_get_grip(int hand);
// One-shot controller vibration through the OpenXR haptic action. amplitude 0..1, freq_hz <= 0 =
// runtime default, duration in milliseconds (clamped up to the runtime minimum). Fire-and-forget
// and not frame-scoped, so game-tick code may call it directly; no-op while input is inactive.
void vr_trigger_haptic(int hand, float amplitude01, float freq_hz, float duration_ms);
// Hand draw matrix (model-local -> game-world, engine row-vector MtxF layout) for pinning Link's hand
// limb to the controller. Includes Link's model scale (set via vr_set_hand_scale). False if untracked.
bool vr_get_hand_matrix(int hand, float out[4][4]);

// Live hand rendering: the game tags each hand limb's per-frame Mtx* (register) + clears the registry
// each frame; gfx_pc calls vr_lookup_hand_matrix per eye and substitutes the LIVE controller pose so
// the hands track at headset rate instead of the interpolated game rate. vr_set_hand_scale folds in
// Link's model scale so the live-replaced hand renders at the right size.
void vr_set_hand_scale(float s);
void vr_set_hand_mirror(int hand, bool mirror);
void vr_register_hand_matrix(const void* mtx, int hand);
// Hand-CHILD matrix: substituted with (live hand pose) x (local_mf16, MtxF layout) — for
// geometry derived from the hand at 20 Hz that must track the live hand (bowstring).
void vr_register_hand_child_matrix(const void* mtx, int hand, const float* local_mf16);
// The rendered center-eye pose as a model matrix (same layout; +X right, +Y up, -Z forward), and the
// head-CHILD registry: substituted with (rendered head) x (local_mf16) per eye — worn items. Cleared
// with the hand registries.
bool vr_get_head_matrix(float out[4][4]);
void vr_register_head_child_matrix(const void* mtx, const float* local_mf16);
// The playspace frame (origin = this render pass's interpolated anchor, playspace axes) and its
// child registry: substituted with (playspace now) x (local_mf16) per eye. Cleared with the hand
// registries.
bool vr_get_playspace_matrix(float out[4][4]);
void vr_register_playspace_child_matrix(const void* mtx, const float* local_mf16);
void vr_clear_hand_matrices();
bool vr_lookup_hand_matrix(const void* mtx, float out[4][4]);

// HMD-driven heading (Phase 2)
int16_t vr_get_heading_yaw();
void vr_set_lockon_yaw(int16_t yaw_binang, bool active);
void vr_recenter_heading(int16_t link_yaw);

// Sub-frame interpolation factor (0..1) for the current render pass, so the camera anchor can be
// interpolated between game frames in lockstep with the rest of the interpolated world.
void vr_set_interp_alpha(float alpha);

// Render target rebind (called when sub-framebuffer operations restore the main target)
void vr_rebind_current_eye_target();
// Clear the depth of the eye image currently being rendered (stereo passes only). The interpreter
// calls it for a full-screen Z fill issued mid-list: a game resetting depth for content that must
// not be occluded by the world (the world-space pause menu's 3D Link).
void vr_clear_current_eye_depth();

// HUD overlay (rendered to a separate quad layer in front of the user)
void vr_set_hud_commands(void* commands);
void* vr_get_hud_commands();
void vr_begin_hud();
void vr_end_hud();
bool vr_is_rendering_hud();
bool vr_is_rendering_screen();

// Text panel: the message system's display list (text boxes, ocarina staff) on its own quad that
// soft-follows in front of the player in real metres. commands NULL = no text box showing (panel
// hidden; the next non-NULL snaps it into place). crop = u0, v0, u1, v1 of the 320x240 frame to
// show. Rendered once per frame like the HUD, as a 2D pass (vr_is_rendering_hud() is true).
void vr_set_text_commands(void* commands, const float crop[4]);
void* vr_get_text_commands();
void vr_set_text_panel_layout(float width_m, float distance_m, float height_m);
void vr_begin_text();
void vr_end_text();
bool vr_is_rendering_text();

// SoH (ImGui) menu panel. Active while the menu is visible and VR is running: ImGui then lays out
// for a panel of vr_menu_panel_size() px (Fast3dGui) and draws each frame between vr_begin_menu()
// and vr_end_menu() into the panel's image, a world-locked quad in front of the player. Holding the
// left thumbstick click (right when left-handed) toggles the menu; the game sees neutral controller input while it is open.
// The pointer is a controller ray: x, y in panel px (valid = the ray hits the panel), down = the
// trigger (with hysteresis), wheel = ImGui wheel units accumulated since the previous call.
struct VrMenuPointer {
    bool valid;
    float x, y;
    bool down;
    float wheel;
    int hand;
};
bool vr_menu_panel_active();
// False on standalone headsets (Android): VR can't be switched off there (no flat screen exists);
// the VR Mode checkbox and F9 are hidden / ignored, and doffing never drops to flat.
bool vr_can_disable();
void vr_menu_panel_size(int* w, int* h);
void vr_menu_take_pointer(VrMenuPointer* out);
bool vr_begin_menu();
void vr_end_menu();
// A 2D pass into a transparent quad target (HUD or text): the backend accumulates alpha coverage.
bool vr_wants_coverage_blend();

// HUD elements (see VR_HUD_MARKER in vr_interface.h). g_vr_hud_layout_pass is true only during a
// HUD pass that filters/lays out elements (wrist layout, or the world-space pause frame); the
// interpreter then asks vr_hud_tri, per triangle, whether to draw it, and gets { s, ox, oy, sy }:
// clip-space x' = s x + ox w, y' = sy y + oy w moves and sizes it into its slot (wrists: sy = s;
// pause frame: pinned to its corner of the resizable canvas). A plain global so the per-triangle
// check costs a load when it's off.
extern bool g_vr_hud_layout_pass;
void vr_hud_marker(uint32_t w1);
bool vr_hud_tri(const float ndc_xy[6], float out_scale_offset[4]);
// Link's age, for the Adult / Child wrist HUD settings profile (vr_hud_settings.h).
void vr_set_hud_child(bool child);
bool vr_get_hud_child();
// World-anchored HUD quad (world-space pause): see VR_SetHudWorldPanel.
void vr_set_hud_world_panel(bool enabled, const float center[3], float yaw, float width, float height);
// Texture rectangles in stereo passes (world-space file select): see VR_SetRectWorldPanel. Plain
// globals so the per-rect check costs a load when it's off.
extern bool g_vr_rect_world;
extern float g_vr_rect_world_mtx[4][4];

// Flat-screen mode: 2D contexts (file select, pause) render the whole frame to a world-locked
// floating panel instead of the stereo eyes; the frozen world stays behind it, still head-tracked.
void vr_set_flat_screen(bool enabled);
bool vr_get_flat_screen();
void vr_begin_screen();
void vr_end_screen();
void vr_get_2d_target_size(uint32_t* w, uint32_t* h);

// Desktop mirror: copy the rendered left eye into a sampleable texture so the companion window can
// display what the headset sees (and ImGui can composite the menu on top of it). vr_capture_mirror
// must run while the left-eye swapchain image is still acquired (i.e. before vr_end_eye(0) releases
// it). vr_get_mirror_texture_id returns the SRV as an ImTextureID-compatible pointer, or null if the
// mirror isn't available.
void vr_capture_mirror();
void* vr_get_mirror_texture_id();
// True when the mirror copies are stored bottom-up (OpenGL): draw the eye mirror with V flipped.
// vr_get_mirror_quads already flips its own UVs.
bool vr_get_mirror_flip_v();

// The headset's quad layers (HUD / wrist panels, text panel, flat-screen panel, pause HUD frame)
// as the mirrored left eye sees them, for the companion window to draw over the mirror: each quad
// is a VR_MIRROR_GRID^2 grid of points in 0..1 mirror-image coordinates (ok = in front of the eye)
// plus its texture (an ImTextureID) and UV rect (u0, v0, u1, v1). Returns how many were filled.
#define VR_MIRROR_GRID 9
struct VrMirrorQuad {
    void* srv;
    float uv[4];
    float pt[VR_MIRROR_GRID][VR_MIRROR_GRID][2];
    bool ok[VR_MIRROR_GRID][VR_MIRROR_GRID];
};
int vr_get_mirror_quads(VrMirrorQuad* out, int max);
