#pragma once

#include <stdbool.h>
#include <stdint.h>

// --------------------------------------------------------------------------
// VR physical-combat substrate: headset-rate hand kinematics (velocity + pose history) and the
// spring-damper "virtual held object" simulation. Pure math, platform-independent — compiled on
// every backend, but only ever FED by the OpenXR layer (vr_openxr.cpp) while a session is live;
// with no feeder every query returns false/empty, so callers need no platform guards.
//
// Frames and units:
//  - Feeder inputs are RAW OpenXR tracking space, meters (pre snap-turn). The artificial snap
//    turn and the camera anchor are pushed as per-step context instead of being baked into the
//    samples, so hand history stays continuous across a snap turn (the turn is a world-space
//    change, not a hand motion) and never produces a phantom velocity spike.
//  - Game-facing outputs compose exactly like vr_get_hand_pose: world = anchor +
//    (turn_rot * raw + turn_off) * world_scale. Directions/velocities are turn-rotated but stay
//    in physical meters/second — thresholds tuned in m/s hold regardless of world scale or
//    Link's age.
//  - The whole engine pipeline is single-threaded (game tick and every XR sub-frame run on one
//    call stack), so this module needs no locking: ticks push/pull strictly between steps.
// --------------------------------------------------------------------------

// One game-facing hand sample (one XR frame). Field-for-field mirror of VrHandSample in
// vr_interface.h (kept as separate types so the C ABI header stays self-contained).
struct VrPhysHandSample {
    float pos_units[3];   // game-world position, game units (anchor at sample time folded in)
    float quat[4];        // game-facing orientation (snap turn applied), x,y,z,w
    float lin_vel_mps[3]; // linear velocity, game-facing direction, physical m/s
    float ang_vel_rps[3]; // angular velocity, rad/s, game-facing frame
    uint64_t time_ns;     // XR predicted display time of the sample
};

// ---- Feeder API (vr_openxr.cpp only) ----

// Drop all state: session init/shutdown, and any context where stale kinematics could leak
// across (mode re-enable). Also deactivates every held-object slot.
void vrphys_reset();

// One raw grip sample for this XR frame. vel_valid says the runtime filled XrSpaceVelocity
// (linear); when false the step falls back to finite-differencing consecutive samples.
void vrphys_push_hand_sample(int hand, const float pos_m[3], const float quat_xyzw[4],
                             const float lin_vel_mps[3], const float ang_vel_rps[3], bool vel_valid,
                             uint64_t time_ns);

// Integrate one XR frame: convert this frame's pushed samples into game-facing history using the
// frame's world context, then advance the held-object springs. turn_* is the accumulated
// artificial snap turn (raw tracking -> game-facing), anchor is the sub-frame-blended camera
// anchor in game units, world_scale is game units per meter.
void vrphys_step(float dt_s, const float turn_quat_xyzw[4], const float turn_off_m[3],
                 const float anchor_units[3], float world_scale, bool first_person);

// ---- Game-facing queries (exposed through vr_interface) ----

// Latest hand velocity: linear m/s + angular rad/s, game-facing frame. False while untracked
// (no sample recently) or before any sample arrived.
bool vrphys_get_hand_velocity(int hand, float out_lin_mps[3], float out_ang_rps[3]);

// Copy up to max_samples unread history samples (oldest first) and mark them read. Returns the
// count written. Sized for one 20 Hz game tick of 120 Hz XR frames with slack; on overflow the
// oldest unread samples are dropped.
int vrphys_get_hand_path(int hand, VrPhysHandSample* out, int max_samples);

// ---- Held-object simulation slots ----
// One slot per concurrently-simulated held thing. A slot is active while it has a descriptor;
// pass NULL to deactivate. On activation the object snaps to its hand target (no fly-in).
// Spring-damper toward the primary hand's grip pose (or, two-handed, that pose re-aimed along
// the line between both hands), resolved against the pushed contact primitives.

enum VrPhysSlotId {
    VRPHYS_SLOT_WEAPON = 0, // main-hand melee weapon
    VRPHYS_SLOT_SHIELD = 1, // off-hand shield
    VRPHYS_SLOT_PROP = 2,   // carried prop (pot/rock/bomb)
    VRPHYS_SLOT_BOW = 3,    // bow/slingshot body (string hand handled separately)
    VRPHYS_SLOT_COUNT = 4,
};

struct VrPhysObjectDesc {
    int primary_hand;           // 0 = left, 1 = right
    int secondary_hand;         // -1 = one-handed; else the hand gripping grip_local_secondary_m
    float lin_freq_hz;          // position spring natural frequency (14+ = near-1:1, 3-4 = heavy)
    float lin_zeta;             // position damping ratio (>= 1.0 keeps it overshoot-free)
    float ang_freq_hz;          // orientation spring natural frequency
    float ang_zeta;             // orientation damping ratio
    float max_accel_mps2;       // linear acceleration clamp; <= 0 = unclamped
    float grip_local_root_m[3]; // held segment (handle end -> business end) in grip-local meters;
    float grip_local_tip_m[3];  //   this is what the contact solve sweeps
    bool contact_enabled;       // resolve the segment against the pushed contact primitives
    float friction;             // tangential damping per contacting step (0..1)
    // Tuning; <= 0 means "use the built-in default".
    float blade_radius_m;    // collision thickness — how far the segment rests from a surface
    float touch_tolerance_m; // treat as touching (impact sfx/haptics) within this distance
    float max_ang_accel;     // angular acceleration clamp, rad/s^2
    bool pivot_only;         // contacts rotate the object about the grip, never translate it
    float grip_local_edge_m[3]; // flat-blade half-width offset, grip-local meters; 0 = round
    float tip_taper_frac;       // trailing fraction of the length that tapers to the point
    // Above this mid-blade speed (m/s) contacts disengage entirely: a committed swing cuts
    // THROUGH instead of snagging, while a gentle touch still rests on the surface. 0 = the
    // blade never passes through. Re-engages at 70% of the threshold (hysteresis).
    float passthrough_speed_mps;
    // Resistance while cutting THROUGH something (passthrough active and the blade overlaps
    // geometry): fraction of the blade's remaining catch-up distance RETAINED per 90 Hz step
    // (frame-rate normalized). 0 = clean cut, 0.55 = the blade visibly drags through flesh
    // and catches up on exit. Separate coefficients for enemy bodies and world geometry.
    float cut_drag_flesh;
    float cut_drag_world;
    // Cosmetic weight lag: the SERVED (rendered) pose trails the hand's rotation by this many
    // seconds during fast swings and snaps back slightly underdamped — the "weight wiggle".
    // Physics, contacts and damage never lag; at rest the offset is exactly zero. <= 0 = off.
    float visual_lag_s;
    float visual_snap_hz; // catch-up spring frequency; <= 0.5 uses the built-in 5 Hz
    // Two-handed hold (secondary_hand >= 0): where the second hand grips, in the PRIMARY grip's
    // local frame (meters) — a point on the handle. The target orientation is re-aimed so the
    // direction to this point follows the line between the two hands (roll about that line stays
    // with the primary wrist), and the secondary hand's served pose is pinned to this point on
    // the simulated object, so the rendered off hand holds the handle and lags with it.
    float grip_local_secondary_m[3];
    // Gravity on the object's centre of mass (grip-local meters): the orientation target sags
    // under it against the angular spring, so a heavy head droops in a loose grip. Scale 0 = off,
    // 1 = real g. Only the collision-free target sees it; contacts are unaffected.
    float grip_local_com_m[3];
    float gravity_scale;
};

void vrphys_set_object(int slot, const VrPhysObjectDesc* desc_or_null);
bool vrphys_is_object_active(int slot);

// Simulated object pose in game-facing coords (same composition as the hand queries): position in
// game units, velocities in physical m/s / rad/s. False while the slot is inactive.
bool vrphys_get_object_pose(int slot, float out_pos_units[3], float out_quat_xyzw[4],
                            float out_lin_vel_mps[3], float out_ang_vel_rps[3]);

// Sim grip pose in RAW tracking space (meters), for the OpenXR layer's pose routing: while a slot
// owns a hand, the rendered hand (and everything the game derives from the hand matrix) follows
// the simulated object instead of the raw controller. False when no active slot binds this hand.
bool vrphys_get_hand_sim_pose_raw(int hand, float out_pos_m[3], float out_quat_xyzw[4]);

// Active contacts from the latest sim step (up to 4): positions in world units, unit normals in
// the game-facing frame, 3 floats each. Debug/inspection.
int vrphys_get_object_contacts(int slot, float* out_pos_units_xyz, float* out_normals_xyz, int max_contacts);

// Flight recorder (diagnostics): while enabled, every sim step of the weapon slot is captured
// into a ring holding the most recent ~20 s. Enabling clears it; vrphys_log_write dumps CSV and
// empties the ring, returning the record count (0 = nothing captured, -1 = file open failed).
// --- Visual-mesh collision (experimental) ---
// Triangles harvested from the renderer itself: the game marks a world-space region of
// interest around the blade each tick, and the interpreter feeds every triangle it draws
// inside that region (world units) into a per-frame buffer that the sim merges with the
// game-pushed prims. Mask ON while drawing things the blade must not collide with (the
// player's own hands/weapon, the sword trail).
void vrphys_mesh_set_region(const float center_units[3], float radius_units, bool enabled);
void vrphys_mesh_mask(bool masked);
void vrphys_mesh_set_flesh(bool flesh); // tris harvested while set carry the flesh material id
bool vrphys_mesh_collecting();
void vrphys_mesh_consider_tri(const float a[3], const float b[3], const float c[3]);
// The harvested tris the solver actually used last step (world units, 9 floats per tri).
int vrphys_mesh_get_debug_tris(float* out_xyz9_per_tri, int max_tris);

void vrphys_log_set_enabled(bool enabled);
int vrphys_log_count();
int vrphys_log_write(const char* path);

// ---- Contact primitives (pushed per game tick, WORLD space / game units) ----
// The blade only collides with what the game pushes here: level geometry planes and the colliders
// of things a blade shouldn't pass through. Converted to tracking space per step with the live
// frame context, so they stay put while Link moves within the tick.

enum VrPhysPrimType {
    VRPHYS_PRIM_SPHERE = 0,  // a = center, radius
    VRPHYS_PRIM_CAPSULE = 1, // a..b = segment, radius
    VRPHYS_PRIM_PLANE = 2,   // a = point on plane, b = unit normal (radius unused)
    VRPHYS_PRIM_TRI = 3,     // a,b,c = triangle vertices — BOUNDED level geometry
};

struct VrPhysContactPrim {
    int type;
    float a[3];
    float b[3];
    float c[3];
    float radius;
    int id; // opaque game-side tag (kind + material), echoed back in contact events
};

#define VRPHYS_MAX_CONTACT_PRIMS 48
void vrphys_set_contact_prims(const VrPhysContactPrim* prims, int count);

// ---- Blade path + contact events (produced per sim step, drained per game tick) ----

struct VrPhysBladeSample {
    float root_units[3];  // world units
    float tip_units[3];   // world units
    float mid_vel_mps[3]; // sim velocity of the blade midpoint, game-facing frame, m/s
    float grip_units[3];  // sim grip position, world units
    float quat[4];        // sim orientation, game-facing frame (x,y,z,w) — with grip_units, lets
                          //   the game rebuild ANY grip-local point at this sample
    uint64_t time_ns;
};
int vrphys_get_blade_path(int slot, VrPhysBladeSample* out, int max_samples); // drains

enum VrPhysEventType {
    VRPHYS_EV_CONTACT_BEGIN = 0,
    VRPHYS_EV_CONTACT_END = 1,
};

struct VrPhysEvent {
    int type;
    int slot;
    int prim_id;        // the contacted primitive's game-side tag
    float pos_units[3]; // contact point, world units
    float normal[3];    // contact normal, game-facing frame
    float impact_mps;   // approach speed at contact begin
    uint64_t time_ns;
};
int vrphys_drain_events(VrPhysEvent* out, int max_events); // drains

// Haptic requests emitted by the contact solve (impact feel with zero game-tick latency); the
// OpenXR layer drains and fires these right after each step.
struct VrPhysHapticReq {
    int hand;
    float amplitude01;
    float freq_hz;
    float duration_ms;
};
int vrphys_take_haptic_requests(VrPhysHapticReq* out, int max);
