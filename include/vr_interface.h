#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool VR_IsInitialized();
// Latch a pending VR<->flat mode toggle (CVar gVrEnabled). The game calls this once per game tick,
// BEFORE building the tick's display list, so a DL built for one mode is never drawn in the other.
void VR_ApplyModeRequest(void);
void VR_SetOverlayDisplayList(void* commands);

// Bracket the game's fixed-timestep logic update so the VR performance readout can separate it
// from render cost. Game logic runs once per 20 Hz tick on the same thread as the render passes,
// so its cost comes straight out of that tick's render budget; seeing it split out is the whole
// point. No-ops when VR is inactive.
void VR_GameTickBegin(void);
void VR_GameTickEnd(void);

// Flat-screen (2D) contexts — file select, pause menu. While enabled, the frame renders onto a
// world-locked floating panel placed in front of the player (instead of the stereo world), and the
// last world frame stays frozen-but-head-tracked behind it. The game sets this every frame.
void VR_SetFlatScreen(bool enabled);
bool VR_IsFlatScreen(void);

// First-person camera (game-side integration).
// The game pushes Link's head position each frame as the world-space anchor;
// the VR layer composes the view as anchor + HMD offset/orientation.
void VR_SetFirstPerson(bool enabled);
bool VR_GetFirstPerson(void);
void VR_SetCameraAnchor(float x, float y, float z);
// Link's standing eye height in game units, pushed each first-person frame. With auto world scale
// on, the VR layer derives units/meter from this and the player's real measured eye height, so the
// game ground matches the physical floor and child/adult swaps rescale automatically.
void VR_SetLinkEyeHeight(float units);
// Alyx-style comfort fade: 0 = clear, 1 = world layer black (menus/HUD unaffected). The game sets
// this each tick from how far the player's physical head sits beyond solid geometry.
void VR_SetViewFade(float fade);
// Third person: the game camera's facing (binang yaw) becomes the playspace's base orientation,
// so looking straight ahead in the headset looks where the stock camera looks (cutscenes too).
void VR_SetCameraYaw(int16_t yaw_binang);

// HMD yaw as a binary angle (binang), for driving gameplay heading. (Phase 2)
int16_t VR_GetHeadYaw(void);

// HMD-driven heading. VR_GetHeadingYaw returns the head yaw mapped into game-world space
// (recenter offset + invert/manual-offset CVars applied) for use as Link's steering yaw.
// VR_RecenterHeading captures the offset so that the player's current physical facing maps
// to linkYaw (the player's current in-game facing).
int16_t VR_GetHeadingYaw(void);
void    VR_RecenterHeading(int16_t linkYaw);

// Lock-on framing (Legaiaflame's Lock On). The game pushes the world direction of its lock-on
// target every tick; the playspace then eases so that direction stays within a deadzone cone of
// where the player is actually looking. A headset's orientation can never be overridden, so this
// turns the WORLD under the player instead — the same head-pivot rotation artificial turning uses,
// applied at headset rate so it glides. Inside the cone nothing happens at all, which is what
// keeps free look intact; only the excess past it is corrected, and never faster than the
// configured rate. The request expires shortly after the game stops refreshing it, so a state that
// never runs the update (cutscene, menu, unload) can't leave the world slowly rotating.
void VR_SetLockOnYaw(int16_t yawBinang, bool active);

// Camera unification (Phase 3). VR_GetCameraPose returns the rendered HMD pose in game-world coords
// (eye position + forward/up unit vectors); VR_GetCullingFovy returns a vertical FOV (degrees) wide
// enough to cover the binocular VR view. The game feeds these into its View so CPU-side systems
// (frustum culling, audio panning, projected-position/LOD) match what the player sees. Rendering is
// unaffected. Only meaningful while first-person is active.
void  VR_GetCameraPose(float eye[3], float fwd[3], float up[3]);
float VR_GetCullingFovy(void);

// Roomscale 6DOF (physical walking moves Link's body, collision-swept). The game reads the desired
// per-frame body move (VR_GetRoomscaleDesired), collision-sweeps it, reports the achieved amount
// (VR_AddRoomscaleDisplacement), and pushes anchor = bodyHead - VR_GetRoomscaleOrigin. VR_ResetRoomscale
// re-zeros so the current physical position maps to Link's current body (recenter / scene / enable).
void VR_GetRoomscaleDesired(float out[2]);
void VR_AddRoomscaleDisplacement(float dx, float dz);
void VR_GetRoomscaleOrigin(float out[2]);
void VR_ResetRoomscale(void);
// Bound how far the camera may sit from Link's body horizontally (so the controller-driven camera
// stays within Link's collision; only physical 6DOF lean uses this slack). <= 0 disables.
void VR_ClampRoomscaleLean(float max_units);

// --- Motion controls ---
// Hand index.
#define VR_HAND_LEFT  0
#define VR_HAND_RIGHT 1
// Controller button bitmask (VR_GetControllerButton, per hand). Face buttons are per-hand:
// PRIMARY = A (right) / X (left); SECONDARY = B (right) / Y (left). Analog trigger/grip are also
// thresholded into the TRIGGER/GRIP bits so they read as digital buttons.
#define VR_BTN_TRIGGER    (1 << 0)
#define VR_BTN_GRIP       (1 << 1)
#define VR_BTN_PRIMARY    (1 << 2)
#define VR_BTN_SECONDARY  (1 << 3)
#define VR_BTN_THUMBCLICK (1 << 4)
#define VR_BTN_MENU       (1 << 5)

// Controller grip pose in game-world coords (eye/anchor frame): pos in game units, quat is x,y,z,w.
// Returns false (and identity) if that hand isn't tracked. Buttons/sticks/trigger/grip per hand.
bool     VR_GetHandPose(int hand, float pos[3], float quat[4]);
// Modal hand gestures (the Alyx-style item selector): while suppressed, a hand's thumbstick
// reads as centered at the SOURCE — movement, artificial turning and stick C-buttons all
// inherit it, so holding a stick-click gesture can't steer, turn or fire items.
void     VR_SetStickSuppressed(int hand, int32_t suppressed);
// Controller aim ray (runtime-calibrated pointing pose) in game-world coords: origin + unit
// forward direction. This is the ray for weapon aiming (slingshot/bow/hookshot).
bool     VR_GetAimRay(int hand, float pos[3], float dir[3]);
bool     VR_IsHandActive(int hand);
uint16_t VR_GetControllerButton(int hand);
void     VR_GetThumbstick(int hand, float* x, float* y);
float    VR_GetTrigger(int hand);
float    VR_GetGrip(int hand);
// Hand draw matrix (model-local -> game-world, engine MtxF layout) for pinning Link's hand limb to the
// controller. Includes Link's model scale (set via VR_SetHandScale). False if untracked.
bool     VR_GetHandMatrix(int hand, float out[4][4]);

// --- Physical combat substrate ---
// Contract version of the physical-combat interface between the game and this library. Bump on any
// breaking change to these types/functions; the game asserts equality at init so a stale submodule
// build fails loudly instead of subtly misbehaving.
#define VR_PHYS_INTERFACE_VERSION 15
int32_t VR_PhysGetInterfaceVersion(void);

// Latest hand velocity: linear in physical meters/second (independent of world scale and Link's
// age — tune gameplay thresholds in real m/s), angular in radians/second, directions in the same
// game-facing frame as VR_GetHandPose (snap turn applied). Runtime-reported (filtered + predicted)
// where supported, else finite-differenced from the pose history. False while the hand is untracked.
bool VR_GetHandVelocity(int hand, float linVelMps[3], float angVelRps[3]);

// Headset-rate hand pose history: one sample per XR frame, oldest first, drained — each sample is
// returned exactly once. pos is in game-world units with the camera anchor at SAMPLE time folded in
// (same composition as VR_GetHandPose); velocity in physical m/s. Call once per game tick with
// maxSamples >= 16 (a 20 Hz tick spans ~6 XR frames at 120 Hz; 16 leaves slack for hitches).
typedef struct VrHandSample {
    float pos[3];       // game-world units
    float quat[4];      // x,y,z,w, game-facing frame
    float linVelMps[3]; // physical m/s, game-facing direction (runtime-filtered; free of
                        // artificial locomotion, snap turns and camera-anchor motion)
    float angVelRps[3]; // angular velocity, rad/s, game-facing frame (same filtering) — combine
                        // as v + w x r for the velocity of any point on a held object
    uint64_t timeNs;    // XR predicted display time
} VrHandSample;
int32_t VR_GetHandPath(int hand, VrHandSample* out, int32_t maxSamples);

// One-shot controller vibration. amplitude 0..1, freqHz <= 0 = runtime default, duration in
// milliseconds (clamped up to the runtime minimum). Safe to call any time; no-op when VR is off.
void VR_TriggerHaptic(int hand, float amplitude01, float freqHz, float durationMs);

// Live world scale in game units per real-world meter (includes auto height calibration), for
// converting between the physical-combat APIs' meter-based values and world units.
float VR_GetWorldScale(void);

// --- Held-object simulation (spring-damper "virtual held object" with contact) ---
// One slot per concurrently-simulated held thing. Push a descriptor every game tick while held
// (soh owns all tuning); pass NULL to release the slot. While a slot owns a hand, the rendered
// hand — and everything the game derives from the hand matrix, colliders included — follows the
// SIMULATED pose: the object lags with inertia and stops/bounces on the pushed contact
// primitives while the real hand keeps going.

#define VR_PHYS_SLOT_WEAPON 0
#define VR_PHYS_SLOT_SHIELD 1
#define VR_PHYS_SLOT_PROP 2
#define VR_PHYS_SLOT_BOW 3

typedef struct VrHeldObjectDesc {
    int primaryHand;         // VR_HAND_LEFT / VR_HAND_RIGHT
    int secondaryHand;       // -1 = one-handed
    float linFreqHz;         // position spring frequency (14+ = near-1:1, 3-4 = heavy)
    float linZeta;           // position damping ratio (>= 1 = no overshoot)
    float angFreqHz;         // orientation spring frequency
    float angZeta;           // orientation damping ratio
    float maxAccelMps2;      // linear acceleration clamp; <= 0 = unclamped
    float gripLocalRootM[3]; // held segment (handle end -> business end), grip-local meters
    float gripLocalTipM[3];
    int contactEnabled;      // nonzero: resolve the segment against the contact primitives
    float friction;          // tangential damping while contacting (0..1)
    // Contact tuning; <= 0 means "use the library default".
    float bladeRadiusM;     // collision thickness: how far the object rests off a surface
    float touchToleranceM;  // counts as touching, for impact effects
    float maxAngAccel;      // angular acceleration ceiling, rad/s^2 (stability backstop)
    int32_t pivotOnly;      // nonzero: contacts may only ROTATE the object about the grip —
                            // the grip position always tracks the hand (never pushed back)
    // Flat-blade cross-section. With a nonzero edge vector the collider is a RECTANGLE (two
    // long edges + spine, each bladeRadiusM thick) instead of one round capsule, tapering to
    // the tip point over the last tipTaperFrac of the length.
    float gripLocalEdgeM[3]; // half-width offset (grip-local meters); zero = round blade
    float tipTaperFrac;      // 0 = square tip, 0.2 = pointed over the last 20% of the blade
    float passthroughSpeedMps; // fast swings above this mid-blade speed cut THROUGH geometry
                               // instead of snagging; gentle contact still rests. 0 = never.
    float cutDragFlesh;        // resistance while cutting through enemy bodies (0..1 retained
                               // per 90Hz step: 0 = clean cut, ~0.55 = heavy flesh drag)
    float cutDragWorld;        // same, for world geometry the swing passes through
    float visualLagS;          // cosmetic weight lag: the RENDERED pose trails the hand's
                               // rotation by this many seconds on fast swings, then snaps back
                               // with a little overshoot. Physics/damage never lag. <= 0 = off.
    float visualSnapHz;        // catch-up spring frequency; <= 0.5 uses the built-in 5 Hz
    // Two-handed hold (secondaryHand >= 0): the point on the handle the second hand grips, in the
    // PRIMARY grip's local frame (meters). The object is re-aimed so that point follows the line
    // between the hands (roll stays with the primary wrist), and the second hand's served pose —
    // VR_GetHandPose/VR_GetHandMatrix, so Link's rendered off hand — is pinned onto it.
    float gripLocalSecondaryM[3];
    // Gravity on the centre of mass (grip-local meters): the orientation target sags under it
    // against the angular spring, so a heavy head droops in a loose grip. 0 = off, 1 = real g.
    float gripLocalComM[3];
    float gravityScale;
} VrHeldObjectDesc;
void VR_PhysSetObject(int slot, const VrHeldObjectDesc* descOrNull);

// EXPERIMENTAL visual-mesh collision: while enabled, the renderer harvests every triangle it
// draws within radiusUnits of centerUnits (world space, game units) and the sim collides the
// blade against the nearest of them — the geometry you SEE, animated enemies included, instead
// of the simplified collision mesh. Call once per game tick (center follows the weapon hand).
// Mask ON around draws the blade must ignore (the player's own arms/weapon, the sword trail).
void VR_PhysSetMeshRegion(const float centerUnits[3], float radiusUnits, int32_t enabled);
void VR_PhysMeshMask(int32_t masked);
// Debug: the harvested triangles the solver collided against last step (world units, 9 floats
// per triangle). Returns the count written (up to maxTris).
int32_t VR_PhysGetMeshDebugTris(float* outXyz9PerTri, int32_t maxTris);

// Contact primitives, world space / game units, re-pushed each game tick: level geometry planes
// and the colliders a blade must not pass through (armor, shields). Max 24; excess dropped.
#define VR_PHYS_PRIM_SPHERE 0  // a = center, radius
#define VR_PHYS_PRIM_CAPSULE 1 // a..b segment, radius
#define VR_PHYS_PRIM_PLANE 2   // a = point on plane, b = unit normal
#define VR_PHYS_PRIM_TRI 3     // a,b,c = triangle vertices (BOUNDED — use for level geometry so
                               //   constraints end exactly where the polygon ends: ledges, corners)
#define VR_PHYS_MAX_CONTACT_PRIMS 48
typedef struct VrContactPrim {
    int type;
    float a[3];
    float b[3];
    float c[3];
    float radius;
    int32_t id; // opaque game-side tag (e.g. kind + surface material), echoed back in events
} VrContactPrim;
void VR_PhysSetContactPrims(const VrContactPrim* prims, int32_t count);

// The simulated blade's path, one sample per XR frame (world units), oldest first, drained —
// build swept damage colliders from THIS (not the raw hand path) while the sim owns the weapon,
// or a blade stopped at a wall would still deal damage along the hand's ghost trajectory.
typedef struct VrBladeSample {
    float root[3];      // world units
    float tip[3];       // world units
    float midVelMps[3]; // sim velocity of the blade midpoint, m/s (locomotion-free)
    float gripPos[3];   // sim grip position, world units
    float quat[4];      // sim orientation, game-facing frame (x,y,z,w): with gripPos, rebuilds any
                        //   grip-local point at this sample (a hammer head's real cross-section)
    uint64_t timeNs;
} VrBladeSample;
int32_t VR_PhysGetBladePath(int slot, VrBladeSample* out, int32_t maxSamples);

// Contact events (begin/end), oldest first, drained once per game tick — drive impact SFX and
// spark effects from these. The matching haptics fire VR-side with zero tick latency.
#define VR_PHYS_EV_CONTACT_BEGIN 0
#define VR_PHYS_EV_CONTACT_END 1
typedef struct VrContactEvent {
    int type;
    int slot;
    int32_t primId;  // the contacted primitive's id tag (what the blade actually touched)
    float pos[3];    // contact point, world units
    float normal[3]; // contact normal, game-facing frame
    float impactMps; // approach speed at contact begin
    uint64_t timeNs;
} VrContactEvent;
int32_t VR_PhysDrainEvents(VrContactEvent* out, int32_t maxEvents);

// Debug: the blade's active contacts from the latest sim step (up to 4). outPosXYZ/outNormalXYZ
// receive 3 floats per contact (world units / game-facing unit normals). Returns the count.
int32_t VR_PhysGetContacts(int slot, float* outPosXYZ, float* outNormalXYZ, int32_t maxContacts);

// Flight recorder for diagnosing held-object behavior. While enabled every sim step is captured
// (ring buffer, most recent ~20 s). Write dumps CSV and empties the ring; it returns the record
// count, or -1 if the file could not be opened.
void VR_PhysLogSetEnabled(bool enabled);
int32_t VR_PhysLogCount(void);
int32_t VR_PhysLogWrite(const char* path);

// Live hand rendering. Set the model scale (Link's actor.scale) each frame; clear the hand-matrix
// registry each frame, then tag each hand limb's Mtx* so gfx_pc replaces it with the live controller
// pose per eye (full headset rate, no game-rate judder).
void     VR_SetHandScale(float s);
// Reflect that hand's geometry to flip handedness (when a controller drives Link's opposite-side hand
// model). Per hand: reflecting also mirrors held items' face designs (e.g. the shield crest), so the
// shield hand typically stays unmirrored. The game must also invert back-face culling for a mirrored
// hand. Axis via gVrHandMirrorAxis.
void     VR_SetHandMirror(int hand, bool mirror);
void     VR_RegisterHandMatrix(const void* mtx, int hand);
// A matrix DERIVED from a hand at 20 Hz (bow/slingshot string): substituted per frame with
// (live hand pose) x (localMf16, MtxF layout), so it stays welded to the live-rendered hand.
void     VR_RegisterHandChildMatrix(const void* mtx, int hand, const float* localMf16);
void     VR_ClearHandMatrices(void);

#ifdef __cplusplus
}
#endif
