#include "fast/vr_physics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Self-contained vec3/quat helpers (floats, x/y/z/w quats). Deliberately NOT glm: this file is
// compiled on every platform, and glm only resolves transitively on the Windows/vcpkg setup that
// builds the OpenXR layer. The handful of operations a spring-damper needs fits in a page.
namespace {

struct V3 {
    float x, y, z;
};
struct Q4 {
    float x, y, z, w;
};

constexpr V3 kV3Zero = { 0.0f, 0.0f, 0.0f };
constexpr Q4 kQIdent = { 0.0f, 0.0f, 0.0f, 1.0f };

inline V3 v3(const float* p) {
    return { p[0], p[1], p[2] };
}
inline void v3_store(const V3& v, float* out) {
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
}
inline V3 add(const V3& a, const V3& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}
inline V3 sub(const V3& a, const V3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}
inline V3 mul(const V3& a, float s) {
    return { a.x * s, a.y * s, a.z * s };
}
inline float len(const V3& a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
inline float vdot(const V3& a, const V3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline V3 vcross(const V3& a, const V3& b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
inline float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}
// Closest point to p on segment ab.
inline V3 closest_on_seg(const V3& p, const V3& a, const V3& b) {
    const V3 ab = sub(b, a);
    const float len2 = vdot(ab, ab);
    if (len2 < 1e-12f) {
        return a;
    }
    float t = vdot(sub(p, a), ab) / len2;
    t = clamp01(t);
    return add(a, mul(ab, t));
}
// Closest points between segments a0a1 and b0b1 (out_a on the first, out_b on the second).
inline void closest_seg_seg(const V3& a0, const V3& a1, const V3& b0, const V3& b1, V3& out_a, V3& out_b) {
    const V3 d1 = sub(a1, a0);
    const V3 d2 = sub(b1, b0);
    const V3 r = sub(a0, b0);
    const float A = vdot(d1, d1), E = vdot(d2, d2), F = vdot(d2, r);
    float s, t;
    if (A < 1e-12f && E < 1e-12f) {
        s = t = 0.0f;
    } else if (A < 1e-12f) {
        s = 0.0f;
        t = clamp01(F / E);
    } else {
        const float C = vdot(d1, r);
        if (E < 1e-12f) {
            t = 0.0f;
            s = clamp01(-C / A);
        } else {
            const float B = vdot(d1, d2);
            const float denom = A * E - B * B;
            s = (denom > 1e-12f) ? clamp01((B * F - C * E) / denom) : 0.0f;
            t = clamp01((B * s + F) / E);
            s = clamp01((B * t - C) / A);
        }
    }
    out_a = add(a0, mul(d1, s));
    out_b = add(b0, mul(d2, t));
}
// Point inside triangle (p assumed near the tri's plane; n = unit normal).
inline bool point_in_tri(const V3& p, const V3& a, const V3& b, const V3& c, const V3& n) {
    if (vdot(vcross(sub(b, a), sub(p, a)), n) < 0.0f) {
        return false;
    }
    if (vdot(vcross(sub(c, b), sub(p, b)), n) < 0.0f) {
        return false;
    }
    if (vdot(vcross(sub(a, c), sub(p, c)), n) < 0.0f) {
        return false;
    }
    return true;
}
// Closest point on triangle abc to p (n = unit normal).
inline V3 closest_on_tri(const V3& p, const V3& a, const V3& b, const V3& c, const V3& n) {
    const V3 proj = sub(p, mul(n, vdot(sub(p, a), n)));
    if (point_in_tri(proj, a, b, c, n)) {
        return proj;
    }
    const V3 e0 = closest_on_seg(p, a, b);
    const V3 e1 = closest_on_seg(p, b, c);
    const V3 e2 = closest_on_seg(p, c, a);
    const float d0 = vdot(sub(p, e0), sub(p, e0));
    const float d1 = vdot(sub(p, e1), sub(p, e1));
    const float d2 = vdot(sub(p, e2), sub(p, e2));
    if (d0 <= d1 && d0 <= d2) {
        return e0;
    }
    return (d1 <= d2) ? e1 : e2;
}
// Contact between the blade segment r0r1 and BOUNDED triangle abc. Always fills pen (negative =
// separation), n and cp when it returns true; false only for degenerate geometry. Handles both
// proximity (closest pair — edge contacts give naturally rounded ledge-lip normals) and a clean
// crossing through the interior (pushed back toward the grip side).
inline bool seg_tri_contact(const V3& r0, const V3& r1, const V3& a, const V3& b, const V3& c,
                            float blade_radius, float& out_pen, V3& out_n, V3& out_cp) {
    const V3 tn_raw = vcross(sub(b, a), sub(c, a));
    const float tn_len = len(tn_raw);
    if (tn_len < 1e-9f) {
        return false;
    }
    const V3 tn = mul(tn_raw, 1.0f / tn_len);
    const float sd0 = vdot(sub(r0, a), tn);
    const float sd1 = vdot(sub(r1, a), tn);
    if ((sd0 > 0.0f) != (sd1 > 0.0f)) {
        const float t = sd0 / (sd0 - sd1);
        const V3 x = add(r0, mul(sub(r1, r0), t));
        if (point_in_tri(x, a, b, c, tn)) {
            const float side = sd0 >= 0.0f ? 1.0f : -1.0f;
            const float deep = side > 0.0f ? -sd1 : sd1; // far endpoint's depth beyond the plane
            out_n = mul(tn, side);
            out_pen = blade_radius + (deep > 0.0f ? deep : 0.0f);
            out_cp = x;
            return true;
        }
    }
    float best_d2 = 1e18f;
    V3 best_pb = r0;
    V3 best_pt = a;
    const V3 ends[2] = { r0, r1 };
    for (int i = 0; i < 2; i++) {
        const V3 pt = closest_on_tri(ends[i], a, b, c, tn);
        const float d2 = vdot(sub(ends[i], pt), sub(ends[i], pt));
        if (d2 < best_d2) {
            best_d2 = d2;
            best_pb = ends[i];
            best_pt = pt;
        }
    }
    const V3 ea[3] = { a, b, c };
    const V3 eb[3] = { b, c, a };
    for (int i = 0; i < 3; i++) {
        V3 pb, pt;
        closest_seg_seg(r0, r1, ea[i], eb[i], pb, pt);
        const float d2 = vdot(sub(pb, pt), sub(pb, pt));
        if (d2 < best_d2) {
            best_d2 = d2;
            best_pb = pb;
            best_pt = pt;
        }
    }
    const float dist = sqrtf(best_d2);
    if (dist < 1e-5f) {
        return false;
    }
    out_n = mul(sub(best_pb, best_pt), 1.0f / dist);
    out_pen = blade_radius - dist;
    out_cp = best_pt;
    return true;
}

inline Q4 q4(const float* p) {
    return { p[0], p[1], p[2], p[3] };
}
inline void q4_store(const Q4& q, float* out) {
    out[0] = q.x;
    out[1] = q.y;
    out[2] = q.z;
    out[3] = q.w;
}
inline Q4 qmul(const Q4& a, const Q4& b) {
    return { a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
             a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
inline Q4 qconj(const Q4& q) {
    return { -q.x, -q.y, -q.z, q.w };
}
inline Q4 qnorm(const Q4& q) {
    const float m = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (m < 1e-8f) {
        return kQIdent;
    }
    return { q.x / m, q.y / m, q.z / m, q.w / m };
}
inline V3 qrot(const Q4& q, const V3& v) {
    // v' = v + 2*qv x (qv x v + w*v)
    const V3 qv = { q.x, q.y, q.z };
    const V3 t = { 2.0f * (qv.y * v.z - qv.z * v.y), 2.0f * (qv.z * v.x - qv.x * v.z),
                   2.0f * (qv.x * v.y - qv.y * v.x) };
    return { v.x + q.w * t.x + (qv.y * t.z - qv.z * t.y), v.y + q.w * t.y + (qv.z * t.x - qv.x * t.z),
             v.z + q.w * t.z + (qv.x * t.y - qv.y * t.x) };
}
// Rotation error target * q^-1 as a scaled-axis vector (radians), shortest arc.
inline V3 q_error_vec(const Q4& target, const Q4& q) {
    Q4 e = qmul(target, qconj(q));
    if (e.w < 0.0f) { // shortest arc
        e = { -e.x, -e.y, -e.z, -e.w };
    }
    const float sin_half = sqrtf(e.x * e.x + e.y * e.y + e.z * e.z);
    if (sin_half < 1e-6f) {
        return kV3Zero;
    }
    const float w = e.w > 1.0f ? 1.0f : e.w;
    const float angle = 2.0f * atan2f(sin_half, w);
    return mul({ e.x / sin_half, e.y / sin_half, e.z / sin_half }, angle);
}
// Integrate orientation by angular velocity (rad/s) over dt: q += 0.5*dt*(0,omega)*q, renormalized.
#ifdef VRPHYS_TRACE_CORR
int g_vrphys_trace = 0;
#endif
inline Q4 q_integrate(const Q4& q, const V3& omega, float dt) {
    const Q4 og = { omega.x, omega.y, omega.z, 0.0f };
    const Q4 dq = qmul(og, q);
    return qnorm({ q.x + 0.5f * dt * dq.x, q.y + 0.5f * dt * dq.y, q.z + 0.5f * dt * dq.z,
                   q.w + 0.5f * dt * dq.w });
}
// Shortest-arc rotation taking unit vector a onto unit vector b.
inline Q4 q_from_to(const V3& a, const V3& b) {
    const float d = vdot(a, b);
    if (d < -0.9999f) { // antiparallel: half turn about any axis perpendicular to a
        V3 axis = vcross(a, { 1.0f, 0.0f, 0.0f });
        if (len(axis) < 1e-3f) {
            axis = vcross(a, { 0.0f, 1.0f, 0.0f });
        }
        axis = mul(axis, 1.0f / len(axis));
        return { axis.x, axis.y, axis.z, 0.0f };
    }
    const V3 c = vcross(a, b);
    return qnorm({ c.x, c.y, c.z, 1.0f + d });
}

// --------------------------------------------------------------------------
// State
// --------------------------------------------------------------------------

// A hand is reported untracked after this many steps without a fresh sample (~1/4 s at 120 Hz).
constexpr int kInactiveAfterSteps = 30;
constexpr int kHistoryCap = 64;

struct HandState {
    // Pending raw sample for this frame (set by push, consumed by step).
    bool pending;
    V3 pend_pos_m;
    Q4 pend_quat;
    V3 pend_lin_vel; // runtime-reported, raw tracking space
    V3 pend_ang_vel;
    bool pend_vel_valid;
    uint64_t pend_time_ns;

    // Previous raw sample (for the finite-difference velocity fallback).
    bool have_prev;
    V3 prev_pos_m;
    uint64_t prev_time_ns;
    V3 fallback_vel; // smoothed finite-difference velocity, raw tracking space

    // Latest converted (game-facing) kinematics.
    bool have_latest;
    V3 latest_lin_mps;
    V3 latest_ang_rps;
    int steps_since_sample;

    // Unread game-facing history ring, oldest first.
    VrPhysHandSample ring[kHistoryCap];
    int ring_head; // index of oldest unread
    int ring_count;
};

constexpr int kBladeRingCap = 32;
constexpr int kEventCap = 16;
constexpr int kHapticCap = 8;
constexpr float kBladeRadiusM = 0.012f;    // collision thickness of the held segment
constexpr float kDeepRecognizeM = 0.35f;   // how far behind a one-sided face still counts as
                                           // "inside the solid" (capped so far-side polys of
                                           // thin walls can't push the wrong way)
constexpr float kCorrectionCapM = 0.15f;   // max positional correction per contact per step
// Contacts on the hidden side of a face are only trusted this close to the face's boundary
// edge if the neighbouring face says the blade is genuinely inside the solid. Nearer than
// this to a convex edge (a ledge lip), a behind-the-plane point usually means the blade is
// pivoting OVER the edge, and ejecting it through the face is the classic "ghost collision".
constexpr float kLipGuardM = 0.06f;
constexpr float kTouchToleranceM = 0.008f; // counts as "touching" for impact sfx/haptics

struct SlotState {
    bool active;
    VrPhysObjectDesc desc;
    bool state_valid; // pos/quat snapped to the hand at least once since activation
    // Simulated state, RAW tracking space (meters) — same frame the hand samples arrive in, so a
    // snap turn (context change) never kicks the spring.
    V3 pos_m;
    V3 vel_mps; // finite-difference, for reporting only — never integrated against contacts
    Q4 quat;
    V3 ang_vel_rps;
    // The spring's goal pose. This is driven ONLY by the hand and never sees collision, so it
    // cannot be destabilized by contacts; the object then moves toward it without penetrating.
    V3 tgt_pos_m;
    V3 tgt_vel_mps;
    Q4 tgt_quat;
    V3 tgt_ang_vel_rps;
    // Cosmetic weight-lag ("Ancient Dungeon wiggle"): a raw-space axis-angle offset applied to
    // the SERVED (rendered) pose only — vrphys_get_hand_sim_pose_raw. It trails the hand's
    // angular velocity during fast swings and snaps back with a slightly underdamped spring.
    // Physics never sees it: contacts, the blade ring and every solver read use sl.quat.
    V3 vis_off;
    V3 vis_off_vel;
    // Two-handed this step: the secondary hand is tracked and far enough from the grip for the
    // line between the hands to define an aim. Gates the served off-hand pose and its haptics.
    bool two_hand;
    // Contact + output state
    bool in_contact;
    bool passthrough; // fast-swing state: contacts disengaged until the swing slows down
    // Active contacts from the latest solve (RAW space): the spring projects its drive off these
    // normals next step, and the debug overlay draws them.
    V3 contact_pts[4];
    V3 contact_ns[4];
    float contact_pens[4];
    int contact_ids[4];
    int contact_count;
    VrPhysBladeSample blade_ring[kBladeRingCap];
    int blade_head;
    int blade_count;
};

// Contact primitives: world-space as pushed (game units), converted to raw tracking meters each
// step with that step's context.
VrPhysContactPrim g_prims[VRPHYS_MAX_CONTACT_PRIMS];
int g_prim_count = 0;

// ---- Visual-mesh harvest (experimental) ----
// World-unit triangles captured by the interpreter while it draws, double-buffered per XR
// frame: the sim (which steps BEFORE the frame draws) reads the buffer the renderer just
// finished, and the renderer fills the other side.
struct MeshTri {
    V3 a, b, c;
    int id; // material tag: wall by default, flesh while the flesh marker is set
};
constexpr int kMeshCap = 4096;
constexpr int kMeshSelect = 32; // nearest tris fed to the solver each step
MeshTri g_mesh_buf[2][kMeshCap];
int g_mesh_count[2] = { 0, 0 };
int g_mesh_write = 0;
bool g_mesh_enabled = false;
bool g_mesh_masked = false;
bool g_mesh_flesh = false;
V3 g_mesh_center = kV3Zero; // world units
float g_mesh_radius = 0.0f;
// The tris actually fed to the solver last step (world units), for the debug overlay.
MeshTri g_mesh_dbg[kMeshSelect];
int g_mesh_dbg_count = 0;

VrPhysEvent g_events[kEventCap];
int g_event_count = 0;
VrPhysHapticReq g_haptics[kHapticCap];
int g_haptic_count = 0;
uint64_t g_last_time_ns = 0; // newest hand-sample time, stamped onto sim outputs

struct Context {
    bool valid;
    Q4 turn_rot;
    V3 turn_off_m;
    V3 anchor_units;
    float world_scale;
    bool first_person;
};

HandState g_hands[2];
SlotState g_slots[VRPHYS_SLOT_COUNT];
Context g_ctx;

// --------------------------------------------------------------------------
// Flight recorder: a ring of per-sim-step samples for the weapon slot, dumped to CSV on demand.
// Captures the pose BEFORE and AFTER the contact solve separately, so a jitter cycle can be
// attributed precisely — spring driving into the surface, contacts over-correcting out, or the
// contact SET itself churning (the prim hash catches discovery instability at the 20 Hz tick).
// --------------------------------------------------------------------------

constexpr int kLogCap = 2400; // ~20-27 s of headset-rate capture

struct LogRec {
    uint64_t t;
    float dt;
    float hand[3];      // spring target (raw controller), meters
    float pos_spring[3];// pose after the spring, before contacts
    float quat_spring[4];
    float pos_final[3]; // pose after the contact solve
    float quat_final[4];
    float vel[3];
    float ang[3];
    float root[3]; // blade endpoints, world units
    float tip[3];
    int nc;
    int nprims;
    uint32_t prim_hash;
    struct {
        float n[3];
        float p[3];
        float pen;
        int id;
    } c[4];
};

LogRec* g_log = nullptr;
int g_log_head = 0;
int g_log_count = 0;
bool g_log_enabled = false;
uint32_t g_prim_hash = 0;

void log_push(const LogRec& r) {
    if (!g_log_enabled) {
        return;
    }
    if (g_log == nullptr) {
        g_log = (LogRec*)malloc(sizeof(LogRec) * kLogCap);
        if (g_log == nullptr) {
            g_log_enabled = false;
            return;
        }
    }
    g_log[(g_log_head + g_log_count) % kLogCap] = r;
    if (g_log_count == kLogCap) {
        g_log_head = (g_log_head + 1) % kLogCap; // ring: keep the most recent window
    } else {
        g_log_count++;
    }
}

inline V3 to_world_units(const V3& raw_m, const Context& c) {
    const V3 turned = add(qrot(c.turn_rot, raw_m), c.turn_off_m);
    return add(c.anchor_units, mul(turned, c.world_scale));
}

} // namespace

// --------------------------------------------------------------------------
// Feeder API
// --------------------------------------------------------------------------

void vrphys_reset() {
    memset(g_hands, 0, sizeof(g_hands));
    memset(g_slots, 0, sizeof(g_slots));
    memset(&g_ctx, 0, sizeof(g_ctx));
    g_prim_count = 0;
    g_event_count = 0;
    g_haptic_count = 0;
    g_last_time_ns = 0;
    for (HandState& h : g_hands) {
        h.steps_since_sample = kInactiveAfterSteps;
    }
}

void vrphys_push_hand_sample(int hand, const float pos_m[3], const float quat_xyzw[4],
                             const float lin_vel_mps[3], const float ang_vel_rps[3], bool vel_valid,
                             uint64_t time_ns) {
    if (hand < 0 || hand > 1) {
        return;
    }
    HandState& h = g_hands[hand];
    if (time_ns > g_last_time_ns) {
        g_last_time_ns = time_ns;
    }
    h.pending = true;
    h.pend_pos_m = v3(pos_m);
    h.pend_quat = q4(quat_xyzw);
    h.pend_lin_vel = v3(lin_vel_mps);
    h.pend_ang_vel = v3(ang_vel_rps);
    h.pend_vel_valid = vel_valid;
    h.pend_time_ns = time_ns;
}

void vrphys_step(float dt_s, const float turn_quat_xyzw[4], const float turn_off_m[3],
                 const float anchor_units[3], float world_scale, bool first_person) {
    // Clamp dt into the plausible XR frame window so a hitch (or a runtime reporting a zero
    // period) can't destabilize the springs.
    if (dt_s < 1.0f / 144.0f) {
        dt_s = 1.0f / 144.0f;
    } else if (dt_s > 1.0f / 45.0f) {
        dt_s = 1.0f / 45.0f;
    }

    g_ctx.valid = true;
    g_ctx.turn_rot = q4(turn_quat_xyzw);
    g_ctx.turn_off_m = v3(turn_off_m);
    g_ctx.anchor_units = v3(anchor_units);
    g_ctx.world_scale = world_scale;
    g_ctx.first_person = first_person;

    // Convert this frame's pushed samples into game-facing history.
    for (HandState& h : g_hands) {
        if (!h.pending) {
            if (h.steps_since_sample < kInactiveAfterSteps) {
                h.steps_since_sample++;
            }
            continue;
        }
        h.pending = false;
        h.steps_since_sample = 0;

        // Raw-space linear velocity: prefer the runtime's filtered+predicted report; fall back to
        // finite-differencing consecutive raw samples (lightly smoothed — 120 Hz differences are
        // noisy) on runtimes that leave XrSpaceVelocity empty.
        V3 raw_vel;
        if (h.pend_vel_valid) {
            raw_vel = h.pend_lin_vel;
            h.fallback_vel = raw_vel;
        } else if (h.have_prev && h.pend_time_ns > h.prev_time_ns) {
            const float sample_dt = (float)((double)(h.pend_time_ns - h.prev_time_ns) * 1e-9);
            if (sample_dt > 1e-4f && sample_dt < 0.1f) {
                const V3 inst = mul(sub(h.pend_pos_m, h.prev_pos_m), 1.0f / sample_dt);
                h.fallback_vel = add(mul(h.fallback_vel, 0.4f), mul(inst, 0.6f));
            }
            raw_vel = h.fallback_vel;
        } else {
            raw_vel = kV3Zero;
        }
        h.have_prev = true;
        h.prev_pos_m = h.pend_pos_m;
        h.prev_time_ns = h.pend_time_ns;

        h.have_latest = true;
        h.latest_lin_mps = qrot(g_ctx.turn_rot, raw_vel);
        h.latest_ang_rps = qrot(g_ctx.turn_rot, h.pend_ang_vel);

        VrPhysHandSample s;
        v3_store(to_world_units(h.pend_pos_m, g_ctx), s.pos_units);
        q4_store(qmul(g_ctx.turn_rot, h.pend_quat), s.quat);
        v3_store(h.latest_lin_mps, s.lin_vel_mps);
        v3_store(h.latest_ang_rps, s.ang_vel_rps);
        s.time_ns = h.pend_time_ns;
        if (h.ring_count == kHistoryCap) { // full: drop the oldest unread sample
            h.ring_head = (h.ring_head + 1) % kHistoryCap;
            h.ring_count--;
        }
        h.ring[(h.ring_head + h.ring_count) % kHistoryCap] = s;
        h.ring_count++;
    }

    // Convert this tick's contact primitives into raw tracking space with the fresh context
    // (they are pushed in world space so they stay planted while Link moves within the tick).
    struct RawPrim {
        int type;
        V3 a;
        V3 b;
        V3 c;
        float radius_m;
        int id;
    };
    RawPrim raw_prims[VRPHYS_MAX_CONTACT_PRIMS + kMeshSelect];
    const Q4 turn_inv = qconj(g_ctx.turn_rot);
    const float inv_scale = 1.0f / (g_ctx.world_scale > 1.0f ? g_ctx.world_scale : 35.0f);
    auto world_to_raw = [&](const V3& w) -> V3 {
        return qrot(turn_inv, sub(mul(sub(w, g_ctx.anchor_units), inv_scale), g_ctx.turn_off_m));
    };
    auto raw_to_world = [&](const V3& p) -> V3 {
        // Must mirror to_world_units exactly: the turn offset applies AFTER the turn rotation.
        // Rotating (p + off) instead displaces the result by R*off - off, which is nonzero the
        // moment any snap turn has accumulated — and this position ranks the visual-mesh tris,
        // so the error selected constraints around a phantom blade.
        return add(g_ctx.anchor_units,
                   mul(add(qrot(g_ctx.turn_rot, p), g_ctx.turn_off_m), 1.0f / inv_scale));
    };
    for (int i = 0; i < g_prim_count; i++) {
        raw_prims[i].type = g_prims[i].type;
        raw_prims[i].a = world_to_raw(v3(g_prims[i].a));
        raw_prims[i].b = (g_prims[i].type == VRPHYS_PRIM_PLANE) ? qrot(turn_inv, v3(g_prims[i].b))
                                                                : world_to_raw(v3(g_prims[i].b));
        raw_prims[i].c = (g_prims[i].type == VRPHYS_PRIM_TRI) ? world_to_raw(v3(g_prims[i].c)) : kV3Zero;
        raw_prims[i].radius_m = g_prims[i].radius * inv_scale;
        raw_prims[i].id = g_prims[i].id;
    }
    int n_work = g_prim_count;

    // Visual-mesh merge: read the tri buffer the renderer finished last frame, flip the write
    // side for the upcoming draw, and append the nearest tris to the blade. Ranked by centroid
    // distance to the blade segment minus tri radius (conservative), in world units.
    if (g_mesh_enabled) {
        const int read_side = g_mesh_write;
        g_mesh_write ^= 1;
        g_mesh_count[g_mesh_write] = 0;
        g_mesh_masked = false; // never let a lost pop-marker mask a whole frame
        g_mesh_flesh = false;
        const SlotState& wsl = g_slots[VRPHYS_SLOT_WEAPON];
        if (wsl.active && wsl.state_valid && g_mesh_count[read_side] > 0) {
            const V3 br = raw_to_world(add(wsl.pos_m, qrot(wsl.quat, v3(wsl.desc.grip_local_root_m))));
            const V3 bt = raw_to_world(add(wsl.pos_m, qrot(wsl.quat, v3(wsl.desc.grip_local_tip_m))));
            struct Sel {
                float rank;
                int idx;
            };
            // Stickiness, the other collision-gather lesson: prefer the tris chosen LAST step
            // so the constraint set stays put instead of reshuffling every frame (a member
            // that blinks out for one step lets the target sink, and the snap-back is the
            // churn jitter). Last step's picks are in g_mesh_dbg; match by centroid.
            const int prev_count = g_mesh_dbg_count;
            V3 prev_cen[kMeshSelect];
            for (int p = 0; p < prev_count; p++) {
                prev_cen[p] =
                    mul(add(add(g_mesh_dbg[p].a, g_mesh_dbg[p].b), g_mesh_dbg[p].c), 1.0f / 3.0f);
            }
            const float sticky_bias = 8.0f; // world units of rank preference
            // Both eyes render (and harvest) the same geometry, so every unique tri appears
            // TWICE in the buffer. Rank 2x the target into the candidate list, then emit the
            // nearest kMeshSelect UNIQUE tris below — deduping only at emission would let each
            // eye-duplicate burn a selection slot and halve the real constraint coverage.
            constexpr int kMeshCand = kMeshSelect * 2;
            Sel sel[kMeshCand];
            int nsel = 0;
            for (int i = 0; i < g_mesh_count[read_side]; i++) {
                const MeshTri& mt = g_mesh_buf[read_side][i];
                const V3 cen = mul(add(add(mt.a, mt.b), mt.c), 1.0f / 3.0f);
                const float tri_r =
                    fmaxf(len(sub(mt.a, cen)), fmaxf(len(sub(mt.b, cen)), len(sub(mt.c, cen))));
                float d = len(sub(cen, closest_on_seg(cen, br, bt))) - tri_r;
                for (int p = 0; p < prev_count; p++) {
                    if (len(sub(cen, prev_cen[p])) < 0.5f) {
                        d -= sticky_bias;
                        break;
                    }
                }
                if (nsel == kMeshCand && d >= sel[nsel - 1].rank) {
                    continue;
                }
                int at = (nsel < kMeshCand) ? nsel++ : kMeshCand - 1;
                while (at > 0 && sel[at - 1].rank > d) {
                    sel[at] = sel[at - 1];
                    at--;
                }
                sel[at] = { d, i };
            }
            g_mesh_dbg_count = 0;
            for (int s = 0; s < nsel && g_mesh_dbg_count < kMeshSelect &&
                            n_work < (int)(sizeof(raw_prims) / sizeof(raw_prims[0]));
                 s++) {
                const MeshTri& mt = g_mesh_buf[read_side][sel[s].idx];
                // Emit unique tris only: the second eye's copy of an already-emitted tri is
                // skipped here, and the 2x candidate pool above keeps the next-nearest unique
                // tri available in its place.
                const V3 cen = mul(add(add(mt.a, mt.b), mt.c), 1.0f / 3.0f);
                bool dup = false;
                for (int q = 0; q < g_mesh_dbg_count; q++) {
                    const V3 qcen = mul(add(add(g_mesh_dbg[q].a, g_mesh_dbg[q].b), g_mesh_dbg[q].c),
                                        1.0f / 3.0f);
                    if (len(sub(cen, qcen)) < 0.05f && len(sub(mt.a, g_mesh_dbg[q].a)) < 0.05f) {
                        dup = true;
                        break;
                    }
                }
                if (dup) {
                    continue;
                }
                RawPrim& rp = raw_prims[n_work++];
                rp.type = VRPHYS_PRIM_TRI;
                rp.a = world_to_raw(mt.a);
                rp.b = world_to_raw(mt.b);
                rp.c = world_to_raw(mt.c);
                rp.radius_m = 0.0f;
                rp.id = mt.id; // wall by default, flesh for enemy/NPC body tris
                g_mesh_dbg[g_mesh_dbg_count++] = mt;
            }
        }
    }

    // Advance the held-object springs toward their hand targets (raw tracking space).
    constexpr float kTwoPi = 6.28318530717958647692f;
    constexpr float kMaxAngAccel = 3000.0f; // rad/s^2 — backstop only (implicit = stable)
    for (SlotState& sl : g_slots) {
        if (!sl.active) {
            continue;
        }
        const int hand = sl.desc.primary_hand;
        if (hand < 0 || hand > 1) {
            continue;
        }
        const HandState& h = g_hands[hand];
        if (!h.have_prev || h.steps_since_sample >= kInactiveAfterSteps) {
            // Untracked hand: hold pose, bleed velocity so the object doesn't drift away.
            sl.vel_mps = mul(sl.vel_mps, 0.9f);
            sl.ang_vel_rps = mul(sl.ang_vel_rps, 0.9f);
            sl.vis_off = mul(sl.vis_off, 0.9f);
            sl.vis_off_vel = kV3Zero;
            continue;
        }
        const V3 target_pos = h.prev_pos_m; // latest raw sample
        Q4 target_quat = h.pend_quat.w == 0.0f && h.pend_quat.x == 0.0f && h.pend_quat.y == 0.0f &&
                                 h.pend_quat.z == 0.0f
                             ? kQIdent
                             : qnorm(h.pend_quat);
        const V3 target_vel = h.pend_vel_valid ? h.pend_lin_vel : h.fallback_vel;
        // Angular velocity the target is moving with — the damping term matches it, and the
        // pass-through speed and weight lag read it. One-handed that is the wrist's own.
        V3 target_ang_vel = h.pend_ang_vel;

        // ---- Two-handed aim ----
        // The primary grip stays nailed to its hand; the object is re-aimed so the secondary grip
        // point lies along the line between the hands. Only the handle DIRECTION comes from the
        // second hand — the roll about the handle stays with the primary wrist, so twisting the
        // lead hand still turns the head. The target's angular velocity is assembled the same way
        // from the runtime's smooth hand velocities (never finite-differenced positions, which
        // would amplify tracking noise across the short span between the hands): the primary
        // wrist's roll about the handle plus the handle's swing from the hands' relative motion.
        sl.two_hand = false;
        {
            const int hand2 = sl.desc.secondary_hand;
            const V3 grip2_local = v3(sl.desc.grip_local_secondary_m);
            const float grip2_len = len(grip2_local);
            if (hand2 >= 0 && hand2 <= 1 && hand2 != hand && grip2_len > 0.02f) {
                const HandState& h2 = g_hands[hand2];
                const V3 span = sub(h2.prev_pos_m, target_pos);
                const float span_len = len(span);
                // Hands closer than a few cm give no usable aim: stay one-handed that step.
                if (h2.have_prev && h2.steps_since_sample < kInactiveAfterSteps && span_len > 0.04f) {
                    const V3 d_want = mul(span, 1.0f / span_len);
                    const V3 d_cur = qrot(target_quat, mul(grip2_local, 1.0f / grip2_len));
                    target_quat = qnorm(qmul(q_from_to(d_cur, d_want), target_quat));
                    const V3 v2 = h2.pend_vel_valid ? h2.pend_lin_vel : h2.fallback_vel;
                    const V3 swing = mul(vcross(d_want, sub(v2, target_vel)), 1.0f / span_len);
                    const V3 roll = mul(d_want, vdot(h.pend_ang_vel, d_want));
                    target_ang_vel = add(roll, swing);
                    sl.two_hand = true;
                }
            }
        }

        if (!sl.state_valid) { // activation: appear in the hand, not fly in from the origin
            sl.state_valid = true;
            sl.pos_m = sl.tgt_pos_m = target_pos;
            sl.vel_mps = sl.tgt_vel_mps = target_vel;
            sl.quat = sl.tgt_quat = qnorm(target_quat);
            sl.ang_vel_rps = sl.tgt_ang_vel_rps = kV3Zero;
            sl.vis_off = sl.vis_off_vel = kV3Zero;
            continue;
        }

        // ------------------------------------------------------------------
        // POSITION-BASED held-object update.
        //
        // Forces and contacts are adversaries: a spring pushes the object into a surface with
        // force proportional to how far the hand has travelled past it, contacts push back, and
        // any imbalance between the two is stored as velocity that rings. Pressing harder simply
        // scales the fight up — which is exactly the "push too hard and it shakes" failure.
        //
        // So the spring no longer moves the object. It advances a TARGET pose that is driven by
        // the hand alone and never sees collision (unconditionally stable by construction). The
        // object then walks toward that target in small non-penetrating substeps, sliding along
        // whatever it touches. Nothing accumulates at a contact, so an unreachable target just
        // means the blade rests against the surface, however hard the hand pushes.
        // ------------------------------------------------------------------

        // --- Target pose: implicit spring toward the hand (collision-free). ---
        const float wl = kTwoPi * (sl.desc.lin_freq_hz > 0.1f ? sl.desc.lin_freq_hz : 0.1f);
        const float k_lin = wl * wl;
        const float c_lin = 2.0f * sl.desc.lin_zeta * wl;
        V3 tv = mul(add(add(sl.tgt_vel_mps, mul(sub(target_pos, sl.tgt_pos_m), dt_s * k_lin)),
                        mul(target_vel, dt_s * c_lin)),
                    1.0f / (1.0f + dt_s * c_lin + dt_s * dt_s * k_lin));
        if (sl.desc.max_accel_mps2 > 0.0f) {
            const V3 dv = sub(tv, sl.tgt_vel_mps);
            const float dvm = len(dv);
            const float dvmax = sl.desc.max_accel_mps2 * dt_s;
            if (dvm > dvmax) {
                tv = add(sl.tgt_vel_mps, mul(dv, dvmax / dvm));
            }
        }
        sl.tgt_vel_mps = tv;
        sl.tgt_pos_m = add(sl.tgt_pos_m, mul(sl.tgt_vel_mps, dt_s));

        const float wa = kTwoPi * (sl.desc.ang_freq_hz > 0.1f ? sl.desc.ang_freq_hz : 0.1f);
        const float k_ang = wa * wa;
        const float c_ang = 2.0f * sl.desc.ang_zeta * wa;
        // Gravity on the centre of mass, as a point mass at r: angular acceleration r x g / |r|^2.
        // Raw tracking space is Y-up (the snap turn is a yaw), so g is -Y here. It enters the
        // implicit update as an external acceleration: the spring settles where k * sag balances
        // it, so the droop is gentle with a firm grip and pronounced with a soft one.
        V3 grav_alpha = kV3Zero;
        if (sl.desc.gravity_scale > 0.0f) {
            const V3 com = qrot(sl.tgt_quat, v3(sl.desc.grip_local_com_m));
            const float com2 = vdot(com, com);
            if (com2 > 1e-4f) {
                const V3 g = { 0.0f, -9.81f * sl.desc.gravity_scale, 0.0f };
                grav_alpha = mul(vcross(com, g), 1.0f / com2);
            }
        }
        V3 twv = mul(add(add(sl.tgt_ang_vel_rps,
                             mul(add(mul(q_error_vec(target_quat, sl.tgt_quat), k_ang), grav_alpha), dt_s)),
                         mul(target_ang_vel, dt_s * c_ang)),
                     1.0f / (1.0f + dt_s * c_ang + dt_s * dt_s * k_ang));
        const float max_ang = sl.desc.max_ang_accel > 0.0f ? sl.desc.max_ang_accel : kMaxAngAccel;
        {
            const V3 dw = sub(twv, sl.tgt_ang_vel_rps);
            const float dwm = len(dw);
            const float dwmax = max_ang * dt_s;
            if (dwm > dwmax) {
                twv = add(sl.tgt_ang_vel_rps, mul(dw, dwmax / dwm));
            }
        }
        sl.tgt_ang_vel_rps = twv;
        sl.tgt_quat = q_integrate(sl.tgt_quat, sl.tgt_ang_vel_rps, dt_s);

        // Flight recorder: the pose the spring asked for, before collision has a say.
        const V3 log_pos_spring = sl.tgt_pos_m;
        const Q4 log_quat_spring = sl.tgt_quat;
        const V3 log_hand_target = target_pos;

        const V3 prev_pos = sl.pos_m;
        const Q4 prev_quat = sl.quat;

        // --- Move the object toward the target without ever penetrating. ---
        const V3 root_local = v3(sl.desc.grip_local_root_m);
        const V3 tip_local = v3(sl.desc.grip_local_tip_m);
        const bool has_segment = len(sub(tip_local, root_local)) > 1e-4f;
        const float blade_len = len(sub(tip_local, root_local));

        // Blade cross-section (grip-local). With a width vector the collider is a FLAT BLADE:
        // the two long edges plus the spine, each blade_r thick, with the edges converging to
        // the tip point over the last tip_taper_frac of the length. The flat side rests flat
        // (both edges touch), edge-on contact bites like an edge, and glancing stabs slide
        // off the point. Without a width vector this reduces to the single round capsule.
        V3 seg_root[5];
        V3 seg_tip[5];
        int nsegs = 0;
        if (has_segment) {
            seg_root[nsegs] = root_local;
            seg_tip[nsegs++] = tip_local;
            const V3 half_w = v3(sl.desc.grip_local_edge_m);
            if (len(half_w) > 1e-4f) {
                const float taper = clamp01(sl.desc.tip_taper_frac);
                const V3 taper_pt = add(root_local, mul(sub(tip_local, root_local), 1.0f - taper));
                for (int e = 0; e < 2; e++) {
                    const V3 off = (e == 0) ? half_w : mul(half_w, -1.0f);
                    seg_root[nsegs] = add(root_local, off);
                    seg_tip[nsegs++] = add(taper_pt, off);
                    if (taper > 0.01f) { // edge corner -> the point
                        seg_root[nsegs] = add(taper_pt, off);
                        seg_tip[nsegs++] = tip_local;
                    }
                }
            }
        }
        const float blade_r = sl.desc.blade_radius_m > 0.0f ? sl.desc.blade_radius_m : kBladeRadiusM;
        const float touch_m =
            sl.desc.touch_tolerance_m > 0.0f ? sl.desc.touch_tolerance_m : kTouchToleranceM;
        sl.contact_count = 0;

        float max_impact = 0.0f;
        int ntouch = 0;
        V3 ev_n = { 0.0f, 1.0f, 0.0f };
        V3 ev_p = sl.pos_m;
        int ev_id = 0;

        // Fast-swing pass-through: a committed swing (mid-blade speed above the threshold,
        // from the HAND's velocities — the collision-free intent, so contact cannot gate its
        // own release) cuts through geometry instead of snagging; damage comes from the swept
        // quads either way. Hysteresis: re-engage only once the swing decays to 70%.
        if (sl.desc.passthrough_speed_mps > 0.0f && has_segment) {
            const V3 mid_r = qrot(target_quat, mul(add(root_local, tip_local), 0.5f));
            const float sp = len(add(target_vel, vcross(target_ang_vel, mid_r)));
            if (sl.passthrough) {
                if (sp < sl.desc.passthrough_speed_mps * 0.7f) {
                    sl.passthrough = false;
                }
            } else if (sp > sl.desc.passthrough_speed_mps) {
                sl.passthrough = true;
            }
        } else {
            sl.passthrough = false;
        }

        if (!sl.desc.contact_enabled || !has_segment || n_work == 0 || sl.passthrough) {
            // Cut resistance: while a committed swing is passing THROUGH something, the blade
            // is held back — it covers only part of its catch-up distance to the hand each
            // step while overlapping, then catches up cleanly on exit. Frame-rate normalized
            // (coefficients are "fraction retained per 90 Hz step"). Flesh drags harder than
            // world geometry, and the cut rumbles continuously in the hand.
            float drag = 0.0f;
            if (sl.passthrough && (sl.desc.cut_drag_flesh > 0.0f || sl.desc.cut_drag_world > 0.0f)) {
                const V3 r0 = add(sl.pos_m, qrot(sl.quat, root_local));
                const V3 r1 = add(sl.pos_m, qrot(sl.quat, tip_local));
                for (int i = 0; i < n_work; i++) {
                    const RawPrim& pr = raw_prims[i];
                    bool overlap = false;
                    if (pr.type == VRPHYS_PRIM_TRI) {
                        float pen;
                        V3 n, cp;
                        if (seg_tri_contact(r0, r1, pr.a, pr.b, pr.c, blade_r, pen, n, cp)) {
                            overlap = pen > -0.01f;
                        }
                    } else if (pr.type == VRPHYS_PRIM_CAPSULE) {
                        V3 pa, pb;
                        closest_seg_seg(r0, r1, pr.a, pr.b, pa, pb);
                        overlap = len(sub(pa, pb)) < blade_r + pr.radius_m + 0.01f;
                    } else if (pr.type == VRPHYS_PRIM_SPHERE) {
                        overlap = len(sub(closest_on_seg(pr.a, r0, r1), pr.a)) <
                                  blade_r + pr.radius_m + 0.01f;
                    }
                    if (overlap) {
                        const int kind = pr.id >> 12;
                        const float k = (kind == 3) ? clamp01(sl.desc.cut_drag_flesh)
                                                    : clamp01(sl.desc.cut_drag_world);
                        drag = fmaxf(drag, k);
                    }
                }
            }
            if (drag > 0.0f) {
                const float retention = powf(drag, dt_s * 90.0f);
                sl.pos_m = add(sl.tgt_pos_m, mul(sub(sl.pos_m, sl.tgt_pos_m), retention));
                const V3 aerr = q_error_vec(sl.tgt_quat, sl.quat);
                sl.quat = q_integrate(sl.quat, mul(aerr, 1.0f - retention), 1.0f);
                for (int hi = 0; hi < (sl.two_hand ? 2 : 1) && g_haptic_count < kHapticCap; hi++) {
                    VrPhysHapticReq& hr = g_haptics[g_haptic_count++];
                    hr.hand = hi == 0 ? sl.desc.primary_hand : sl.desc.secondary_hand;
                    hr.amplitude01 = clamp01(0.2f + 0.5f * drag);
                    hr.freq_hz = 0.0f;
                    hr.duration_ms = 25.0f; // re-armed every step: reads as continuous
                }
            } else {
                sl.pos_m = sl.tgt_pos_m;
                sl.quat = sl.tgt_quat;
            }
        } else {
            // Resolve the blade out of everything it overlaps at its CURRENT pose, moving along
            // each contact normal. Pure geometry: no impulses, no stored energy, so this can
            // neither ring nor be fought by the spring. Lever-arm aware, so a tip contact mostly
            // rotates the blade while the grip keeps tracking the hand.
            const float inertia = fmaxf(0.02f, blade_len * blade_len / 3.0f);
            // Triangle adjacency, resolved on the fly against the (small) pushed prim set.
            // Game collision meshes are sealed, so a face's boundary edge normally has a
            // neighbouring face sharing both vertices; which side of that neighbour a point
            // falls on distinguishes "inside the solid" from "hanging past a convex edge".
            auto same_vert = [](const V3& a, const V3& b) { return len(sub(a, b)) < 1e-3f; };
            auto find_shared_tri = [&](int self, const V3& e0, const V3& e1) -> int {
                for (int j = 0; j < n_work; j++) {
                    if (j == self || raw_prims[j].type != VRPHYS_PRIM_TRI) {
                        continue;
                    }
                    const V3 vv[3] = { raw_prims[j].a, raw_prims[j].b, raw_prims[j].c };
                    int h0 = -1, h1 = -1;
                    for (int k = 0; k < 3; k++) {
                        if (same_vert(vv[k], e0)) h0 = k;
                        if (same_vert(vv[k], e1)) h1 = k;
                    }
                    if (h0 >= 0 && h1 >= 0 && h0 != h1) {
                        return j;
                    }
                }
                return -1;
            };
            // True when a behind-the-plane contact of tri `self` at on-plane point `cpp` is
            // within the lip guard of a boundary edge AND the probe point (the deep part of
            // the blade) is on the OUTSIDE of the face sharing that edge: the blade is
            // wrapping a convex lip, not stabbing the face interior. No neighbour found =>
            // treat as solid (never open a hole on incomplete adjacency).
            auto convex_lip_exempt = [&](int self, const V3& cpp, const V3& probe,
                                         float guard) -> bool {
                const RawPrim& t = raw_prims[self];
                const V3 vv[3] = { t.a, t.b, t.c };
                float best = 1e9f;
                int be = -1;
                for (int e = 0; e < 3; e++) {
                    const V3 q = closest_on_seg(cpp, vv[e], vv[(e + 1) % 3]);
                    const float d = len(sub(cpp, q));
                    if (d < best) {
                        best = d;
                        be = e;
                    }
                }
                if (be < 0 || best > guard) {
                    return false;
                }
                const int nb = find_shared_tri(self, vv[be], vv[(be + 1) % 3]);
                if (nb < 0) {
                    return false;
                }
                const RawPrim& q = raw_prims[nb];
                const V3 qn_raw = vcross(sub(q.b, q.a), sub(q.c, q.a));
                const float qn_l = len(qn_raw);
                if (qn_l < 1e-9f) {
                    return false;
                }
                return vdot(sub(probe, q.a), mul(qn_raw, 1.0f / qn_l)) > 1e-4f;
            };
            auto depenetrate = [&](int iterations, bool record) {
                for (int it = 0; it < iterations; it++) {
                    bool moved = false;
                    for (int i = 0; i < n_work; i++) {
                        const RawPrim& pr = raw_prims[i];
                        V3 tn = { 0.0f, 1.0f, 0.0f };
                        if (pr.type == VRPHYS_PRIM_TRI) {
                            const V3 tn_raw = vcross(sub(pr.b, pr.a), sub(pr.c, pr.a));
                            const float tn_l = len(tn_raw);
                            if (tn_l < 1e-9f) {
                                continue;
                            }
                            tn = mul(tn_raw, 1.0f / tn_l);
                        }
                        constexpr int kSamples = 5;
                        // Per blade segment (spine + flat-blade edges); within each:
                        // m: 0..kSamples-1 point samples, kSamples = interior crossing,
                        // kSamples+1..kSamples+3 = segment vs the tri's three edges
                        // (the continuous contact that lets the blade rest on and pivot
                        // around a ledge lip between point samples).
                        for (int si = 0; si < nsegs; si++)
                        for (int m = 0; m <= kSamples + 3; m++) {
                            const V3 rootNow = add(sl.pos_m, qrot(sl.quat, seg_root[si]));
                            const V3 tipNow = add(sl.pos_m, qrot(sl.quat, seg_tip[si]));
                            if (len(sub(tipNow, rootNow)) < 1e-5f) {
                                continue; // degenerate (zero-taper corner segment)
                            }
                            float pen;
                            V3 n;
                            V3 cp;
                            if (m < kSamples) {
                                const float t = (float)m / (float)(kSamples - 1);
                                const V3 p = add(rootNow, mul(sub(tipNow, rootNow), t));
                                if (pr.type == VRPHYS_PRIM_PLANE) {
                                    const float sd = vdot(sub(p, pr.a), pr.b);
                                    pen = blade_r - sd;
                                    n = pr.b;
                                    cp = sub(p, mul(n, sd));
                                } else if (pr.type == VRPHYS_PRIM_TRI) {
                                    const float sd = vdot(sub(p, pr.a), tn);
                                    if (sd < 0.0f) {
                                        if (sd < -kDeepRecognizeM) {
                                            continue;
                                        }
                                        const V3 proj = sub(p, mul(tn, sd));
                                        if (!point_in_tri(proj, pr.a, pr.b, pr.c, tn)) {
                                            continue;
                                        }
                                        if (convex_lip_exempt(i, proj, p, kLipGuardM)) {
                                            continue;
                                        }
                                        pen = blade_r - sd;
                                        n = tn;
                                        cp = proj;
                                    } else {
                                        const V3 pt = closest_on_tri(p, pr.a, pr.b, pr.c, tn);
                                        const V3 d = sub(p, pt);
                                        const float dist = len(d);
                                        if (dist < 1e-5f) {
                                            continue;
                                        }
                                        pen = blade_r - dist;
                                        n = mul(d, 1.0f / dist);
                                        cp = pt;
                                    }
                                } else {
                                    const V3 on_prim = (pr.type == VRPHYS_PRIM_CAPSULE)
                                                           ? closest_on_seg(p, pr.a, pr.b)
                                                           : pr.a;
                                    const V3 d = sub(p, on_prim);
                                    const float dist = len(d);
                                    if (dist < 1e-5f) {
                                        continue;
                                    }
                                    pen = (pr.radius_m + blade_r) - dist;
                                    n = mul(d, 1.0f / dist);
                                    cp = add(on_prim, mul(n, pr.radius_m));
                                }
                            } else if (m == kSamples) {
                                if (pr.type != VRPHYS_PRIM_TRI) {
                                    continue;
                                }
                                const float sd0 = vdot(sub(rootNow, pr.a), tn);
                                const float sd1 = vdot(sub(tipNow, pr.a), tn);
                                if ((sd0 > 0.0f) == (sd1 > 0.0f)) {
                                    continue;
                                }
                                const float t = sd0 / (sd0 - sd1);
                                const V3 x = add(rootNow, mul(sub(tipNow, rootNow), t));
                                if (!point_in_tri(x, pr.a, pr.b, pr.c, tn)) {
                                    continue;
                                }
                                const float side = sd0 >= 0.0f ? 1.0f : -1.0f;
                                const V3 deep_end = (side > 0.0f) ? tipNow : rootNow;
                                const V3 proj_deep =
                                    sub(deep_end, mul(tn, vdot(sub(deep_end, pr.a), tn)));
                                if (!point_in_tri(proj_deep, pr.a, pr.b, pr.c, tn) &&
                                    convex_lip_exempt(i, proj_deep, deep_end, 1e9f)) {
                                    continue;
                                }
                                const float deep = side > 0.0f ? -sd1 : sd1;
                                pen = blade_r + (deep > 0.0f ? deep : 0.0f);
                                n = mul(tn, side);
                                cp = x;
                            } else {
                                const int e = m - kSamples - 1;
                                if (pr.type != VRPHYS_PRIM_TRI) {
                                    if (e != 0 ||
                                        (pr.type != VRPHYS_PRIM_CAPSULE && pr.type != VRPHYS_PRIM_SPHERE)) {
                                        continue;
                                    }
                                    // Continuous closest-pair contact for capsules/spheres: the
                                    // five point samples leave ~len/4 gaps, and a thin bone
                                    // capsule (skeleton-fitted enemy limb) slips clean between
                                    // them. The closest pair between the blade segment and the
                                    // capsule axis never misses.
                                    V3 pb, pc2;
                                    if (pr.type == VRPHYS_PRIM_CAPSULE) {
                                        closest_seg_seg(rootNow, tipNow, pr.a, pr.b, pb, pc2);
                                    } else {
                                        pc2 = pr.a;
                                        pb = closest_on_seg(pr.a, rootNow, tipNow);
                                    }
                                    const V3 d = sub(pb, pc2);
                                    const float dist = len(d);
                                    if (dist < 1e-5f) {
                                        continue;
                                    }
                                    pen = (pr.radius_m + blade_r) - dist;
                                    n = mul(d, 1.0f / dist);
                                    cp = add(pc2, mul(n, pr.radius_m));
                                } else {
                                    const V3 vv[3] = { pr.a, pr.b, pr.c };
                                    V3 pb, pt;
                                    closest_seg_seg(rootNow, tipNow, vv[e], vv[(e + 1) % 3], pb, pt);
                                    const V3 d = sub(pb, pt);
                                    const float dist = len(d);
                                    if (dist < 1e-5f) {
                                        continue;
                                    }
                                    n = mul(d, 1.0f / dist);
                                    // Only outward-facing edge contact: a normal with no
                                    // component along the face normal would drag the blade
                                    // through the face.
                                    if (vdot(n, tn) < 0.02f) {
                                        continue;
                                    }
                                    pen = blade_r - dist;
                                    cp = pt;
                                }
                            }

                            if (record && pen > -touch_m) {
                                ntouch++;
                                if (pen > -touch_m && sl.contact_count < 4) {
                                    bool dup = false;
                                    for (int c = 0; c < sl.contact_count; c++) {
                                        if (vdot(sl.contact_ns[c], n) > 0.98f) {
                                            dup = true;
                                            break;
                                        }
                                    }
                                    if (!dup) {
                                        sl.contact_pts[sl.contact_count] = cp;
                                        sl.contact_ns[sl.contact_count] = n;
                                        sl.contact_pens[sl.contact_count] = pen;
                                        sl.contact_ids[sl.contact_count] = pr.id;
                                        sl.contact_count++;
                                    }
                                }
                                ev_n = n;
                                ev_p = cp;
                                ev_id = pr.id;
                            }
                            if (pen <= 0.0f) {
                                continue;
                            }
                            moved = true;
#ifdef VRPHYS_TRACE_CORR
                            if (g_vrphys_trace) {
                                printf("    CORR prim=%d(id %d) m=%d pen=%.4f n=(%.2f,%.2f,%.2f) cp=(%.3f,%.3f,%.3f)\n",
                                       i, pr.id, m, pen, n.x, n.y, n.z, cp.x, cp.y, cp.z);
                            }
#endif

                            const V3 r = sub(cp, sl.pos_m);
                            const V3 rxn = vcross(r, n);
                            float depth = pen;
                            if (depth > kCorrectionCapM) {
                                depth = kCorrectionCapM;
                            }
                            if (sl.desc.pivot_only) {
                                // The grip is nailed to the hand: resolve by rotating about it
                                // only. A contact needs a usable lever (|r x n| = the arm the
                                // rotation acts through); without one no orientation change can
                                // clear it, so let it clip rather than churn the blade.
                                const float denom = vdot(rxn, rxn);
                                const float min_lever = 0.15f * blade_len;
                                if (denom < min_lever * min_lever) {
                                    continue;
                                }
                                V3 dtheta = mul(rxn, depth / denom);
                                const float dtl = len(dtheta);
                                if (dtl > 0.15f) {
                                    dtheta = mul(dtheta, 0.15f / dtl);
                                }
                                sl.quat = q_integrate(sl.quat, dtheta, 1.0f);
                            } else {
                                // Correction split between translation and rotation by the lever
                                // arm. Full correction: substeps keep each one tiny.
                                const float k_n = 1.0f + vdot(rxn, rxn) / inertia;
                                const float lambda = depth / k_n;
                                sl.pos_m = add(sl.pos_m, mul(n, lambda));
                                V3 dtheta = mul(rxn, lambda / inertia);
                                const float dtl = len(dtheta);
                                if (dtl > 0.2f) {
                                    dtheta = mul(dtheta, 0.2f / dtl);
                                }
                                sl.quat = q_integrate(sl.quat, dtheta, 1.0f);
                            }
                        }
                    }
                    if (!moved) {
                        break;
                    }
                }
            };

            // Walk toward the target in steps no larger than the blade's own thickness, so the
            // blade can never jump through a surface and every correction stays shallow.
            V3 dpos = sub(sl.tgt_pos_m, sl.pos_m);
            V3 drot = q_error_vec(sl.tgt_quat, sl.quat);
            const float travel = len(dpos) + len(drot) * blade_len;
            int nsub = (int)(travel / fmaxf(blade_r, 0.002f)) + 1;
            // With swing-through DISABLED nothing else stops a fast blade, so the
            // substep-per-blade-radius guarantee must hold at any speed or thin walls tunnel;
            // with it enabled, contacts disengage above the threshold long before 8 substeps
            // stop being enough.
            const int max_sub = sl.desc.passthrough_speed_mps > 0.0f ? 8 : 32;
            if (nsub > max_sub) {
                nsub = max_sub;
            }
            const float inv = 1.0f / (float)nsub;
            const V3 dpos_step = mul(dpos, inv);
            const V3 drot_step = mul(drot, inv);
            for (int sIdx = 0; sIdx < nsub; sIdx++) {
                sl.pos_m = add(sl.pos_m, dpos_step);
                sl.quat = q_integrate(sl.quat, drot_step, 1.0f);
                depenetrate(3, sIdx == nsub - 1);
            }

            // Friction: drag at the recorded contacts. Each touching contact gives back a
            // fraction of the tangential distance its blade material point slid this step,
            // applied as the same grip-pivot rotation the normal corrections use — so the
            // blade angle "sticks" and trails while scraping along a surface, but the grip
            // itself never resists the hand. Position-based (a fraction of displacement, not
            // a force), so it cannot ring any more than the depenetration can.
            const float fric = clamp01(sl.desc.friction);
            if (sl.desc.pivot_only && fric > 0.0f && sl.contact_count > 0) {
                bool dragged = false;
                for (int c = 0; c < sl.contact_count; c++) {
                    if (sl.contact_pens[c] <= -touch_m) {
                        continue; // recorded in the touch band but not actually pressing
                    }
                    const V3 cp = sl.contact_pts[c];
                    const V3 n = sl.contact_ns[c];
                    // Slide = the HAND-driven motion of the contact point this step (target
                    // velocities, not the blade's own). Measuring the blade's actual motion
                    // here feeds the friction rotation back into next step's "slide" and winds
                    // the angle up until geometry saturates it — the drag must oppose only
                    // what the player is doing, not what the solver did.
                    const V3 r_cp = sub(cp, sl.pos_m);
                    const V3 v_pt = add(sl.tgt_vel_mps, vcross(sl.tgt_ang_vel_rps, r_cp));
                    const V3 disp = mul(v_pt, dt_s);
                    const V3 slide = sub(disp, mul(n, vdot(disp, n)));
                    const float mag = len(slide);
                    if (mag < 1e-6f) {
                        continue;
                    }
                    const V3 t = mul(slide, 1.0f / mag);
                    const V3 r = sub(cp, sl.pos_m);
                    const V3 lever = vcross(r, t);
                    const float denom = vdot(lever, lever);
                    const float min_lever = 0.15f * blade_len;
                    if (denom < min_lever * min_lever) {
                        continue;
                    }
                    V3 dtheta = mul(lever, -fric * mag / denom);
                    const float dtl = len(dtheta);
                    if (dtl > 0.05f) {
                        dtheta = mul(dtheta, 0.05f / dtl);
                    }
                    sl.quat = q_integrate(sl.quat, dtheta, 1.0f);
                    dragged = true;
                }
                if (dragged) {
                    depenetrate(2, false); // the drag rotation may have re-pressed a surface
                }
            }
        }

        // Velocities are DERIVED, never integrated — nothing to store, nothing to ring.
        sl.vel_mps = mul(sub(sl.pos_m, prev_pos), 1.0f / dt_s);
        sl.ang_vel_rps = mul(q_error_vec(sl.quat, prev_quat), 1.0f / dt_s);

        // ---- Cosmetic weight lag (the Ancient Dungeon trick) ----
        // The RENDERED pose trails the hand's rotation by visual_lag_s seconds during a fast
        // swing and snaps back through a slightly underdamped spring — the little overshoot
        // when the swing stops IS the "weight wiggle". Purely visual: the offset is applied
        // only in vrphys_get_hand_sim_pose_raw (the served/rendered pose); contacts, the
        // blade ring and damage all live on sl.quat and never lag. At rest the offset decays
        // to exactly zero, so the sword stays parented to the hand.
        if (sl.desc.visual_lag_s > 0.0f) {
            V3 tgt = mul(target_ang_vel, -sl.desc.visual_lag_s);
            constexpr float kVisOffCap = 0.4f; // rad — a lag pose, not a detached sword
            const float tm = len(tgt);
            if (tm > kVisOffCap) {
                tgt = mul(tgt, kVisOffCap / tm);
            }
            const float w0 =
                kTwoPi * (sl.desc.visual_snap_hz > 0.5f ? sl.desc.visual_snap_hz : 5.0f);
            constexpr float kVisZeta = 0.55f; // underdamped on purpose: the overshoot sells it
            // Semi-implicit Euler: stable for w0*dt <= 2 (snap <= ~20 Hz at 72 Hz refresh).
            const V3 acc = sub(mul(sub(tgt, sl.vis_off), w0 * w0),
                               mul(sl.vis_off_vel, 2.0f * kVisZeta * w0));
            sl.vis_off_vel = add(sl.vis_off_vel, mul(acc, dt_s));
            sl.vis_off = add(sl.vis_off, mul(sl.vis_off_vel, dt_s));
            const float om = len(sl.vis_off);
            if (om > kVisOffCap) {
                sl.vis_off = mul(sl.vis_off, kVisOffCap / om);
            }
        } else {
            sl.vis_off = kV3Zero;
            sl.vis_off_vel = kV3Zero;
        }

        // Impact strength = how much of the motion the surface actually refused this step.
        if (ntouch > 0) {
            if (sl.desc.pivot_only) {
                max_impact = len(q_error_vec(sl.tgt_quat, sl.quat)) * blade_len / dt_s * 0.25f;
            } else if (len(sub(sl.tgt_pos_m, sl.pos_m)) > 0.0f) {
                max_impact = len(sub(mul(sub(sl.pos_m, prev_pos), 1.0f / dt_s),
                                     mul(sub(sl.tgt_pos_m, prev_pos), 1.0f / dt_s)));
            }
        }

        if (ntouch > 0) {
            if (!sl.in_contact) {
                sl.in_contact = true;
                if (g_event_count < kEventCap) {
                    VrPhysEvent& ev = g_events[g_event_count++];
                    ev.type = VRPHYS_EV_CONTACT_BEGIN;
                    ev.slot = static_cast<int>(&sl - &g_slots[0]);
                    ev.prim_id = ev_id;
                    v3_store(to_world_units(ev_p, g_ctx), ev.pos_units);
                    v3_store(qrot(g_ctx.turn_rot, ev_n), ev.normal);
                    ev.impact_mps = max_impact;
                    ev.time_ns = g_last_time_ns;
                }
                // Both hands feel it when both are on the handle.
                for (int hi = 0; hi < (sl.two_hand ? 2 : 1) && g_haptic_count < kHapticCap; hi++) {
                    VrPhysHapticReq& hr = g_haptics[g_haptic_count++];
                    hr.hand = hi == 0 ? sl.desc.primary_hand : sl.desc.secondary_hand;
                    hr.amplitude01 = clamp01(0.3f + max_impact * 0.18f);
                    hr.freq_hz = 0.0f;
                    hr.duration_ms = 30.0f + (max_impact > 4.0f ? 4.0f : max_impact) * 20.0f;
                }
            }
        } else if (sl.in_contact) {
            sl.in_contact = false;
            if (g_event_count < kEventCap) {
                VrPhysEvent& ev = g_events[g_event_count++];
                ev.type = VRPHYS_EV_CONTACT_END;
                ev.slot = static_cast<int>(&sl - &g_slots[0]);
                ev.prim_id = ev_id;
                v3_store(to_world_units(ev_p, g_ctx), ev.pos_units);
                v3_store(qrot(g_ctx.turn_rot, ev_n), ev.normal);
                ev.impact_mps = 0.0f;
                ev.time_ns = g_last_time_ns;
            }
        }

        // ---- Blade path sample (world units), for the game's swept damage colliders ----
        if (has_segment) {
            const V3 root = add(sl.pos_m, qrot(sl.quat, root_local));
            const V3 tip = add(sl.pos_m, qrot(sl.quat, tip_local));
            const V3 mid_r = qrot(sl.quat, mul(add(root_local, tip_local), 0.5f));
            const V3 v_mid = add(sl.vel_mps, vcross(sl.ang_vel_rps, mid_r));
            if (sl.blade_count == kBladeRingCap) {
                sl.blade_head = (sl.blade_head + 1) % kBladeRingCap;
                sl.blade_count--;
            }
            VrPhysBladeSample& bs = sl.blade_ring[(sl.blade_head + sl.blade_count) % kBladeRingCap];
            v3_store(to_world_units(root, g_ctx), bs.root_units);
            v3_store(to_world_units(tip, g_ctx), bs.tip_units);
            v3_store(qrot(g_ctx.turn_rot, v_mid), bs.mid_vel_mps);
            v3_store(to_world_units(sl.pos_m, g_ctx), bs.grip_units);
            q4_store(qmul(g_ctx.turn_rot, sl.quat), bs.quat);
            bs.time_ns = g_last_time_ns;
            sl.blade_count++;
        }

        // Flight recorder: one sample per step for the weapon slot.
        if (g_log_enabled && (&sl == &g_slots[VRPHYS_SLOT_WEAPON]) && has_segment) {
            LogRec r;
            memset(&r, 0, sizeof(r));
            r.t = g_last_time_ns;
            r.dt = dt_s;
            v3_store(log_hand_target, r.hand);
            v3_store(log_pos_spring, r.pos_spring);
            q4_store(log_quat_spring, r.quat_spring);
            v3_store(sl.pos_m, r.pos_final);
            q4_store(sl.quat, r.quat_final);
            v3_store(sl.vel_mps, r.vel);
            v3_store(sl.ang_vel_rps, r.ang);
            v3_store(to_world_units(add(sl.pos_m, qrot(sl.quat, root_local)), g_ctx), r.root);
            v3_store(to_world_units(add(sl.pos_m, qrot(sl.quat, tip_local)), g_ctx), r.tip);
            r.nc = sl.contact_count;
            r.nprims = n_work;
            r.prim_hash = g_prim_hash;
            for (int ci = 0; ci < sl.contact_count && ci < 4; ci++) {
                v3_store(sl.contact_ns[ci], r.c[ci].n);
                v3_store(to_world_units(sl.contact_pts[ci], g_ctx), r.c[ci].p);
                r.c[ci].pen = sl.contact_pens[ci];
                r.c[ci].id = sl.contact_ids[ci];
            }
            log_push(r);
        }
    }
}

// --------------------------------------------------------------------------
// Game-facing queries
// --------------------------------------------------------------------------

bool vrphys_get_hand_velocity(int hand, float out_lin_mps[3], float out_ang_rps[3]) {
    if (out_lin_mps) {
        out_lin_mps[0] = out_lin_mps[1] = out_lin_mps[2] = 0.0f;
    }
    if (out_ang_rps) {
        out_ang_rps[0] = out_ang_rps[1] = out_ang_rps[2] = 0.0f;
    }
    if (hand < 0 || hand > 1) {
        return false;
    }
    const HandState& h = g_hands[hand];
    if (!h.have_latest || h.steps_since_sample >= kInactiveAfterSteps) {
        return false;
    }
    if (out_lin_mps) {
        v3_store(h.latest_lin_mps, out_lin_mps);
    }
    if (out_ang_rps) {
        v3_store(h.latest_ang_rps, out_ang_rps);
    }
    return true;
}

int vrphys_get_hand_path(int hand, VrPhysHandSample* out, int max_samples) {
    if (hand < 0 || hand > 1 || out == nullptr || max_samples <= 0) {
        return 0;
    }
    HandState& h = g_hands[hand];
    int n = h.ring_count < max_samples ? h.ring_count : max_samples;
    for (int i = 0; i < n; i++) {
        out[i] = h.ring[(h.ring_head + i) % kHistoryCap];
    }
    h.ring_head = (h.ring_head + n) % kHistoryCap;
    h.ring_count -= n;
    return n;
}

// --------------------------------------------------------------------------
// Held-object slots
// --------------------------------------------------------------------------

void vrphys_set_object(int slot, const VrPhysObjectDesc* desc_or_null) {
    if (slot < 0 || slot >= VRPHYS_SLOT_COUNT) {
        return;
    }
    SlotState& sl = g_slots[slot];
    if (desc_or_null == nullptr) {
        sl.active = false;
        sl.state_valid = false;
        sl.in_contact = false;
        sl.two_hand = false;
        sl.blade_head = 0;
        sl.blade_count = 0;
        return;
    }
    // Re-pushing a descriptor every tick is the expected idiom (soh owns the params); only a
    // fresh activation resets the simulated state.
    if (!sl.active) {
        sl.state_valid = false;
        sl.in_contact = false;
        sl.two_hand = false;
        sl.blade_head = 0;
        sl.blade_count = 0;
    }
    // Letting go with the second hand hands the off hand back to its raw controller at once
    // (the next step would clear it anyway; this closes the gap before that step).
    if (desc_or_null->secondary_hand < 0) {
        sl.two_hand = false;
    }
    sl.active = true;
    sl.desc = *desc_or_null;
}

bool vrphys_is_object_active(int slot) {
    return slot >= 0 && slot < VRPHYS_SLOT_COUNT && g_slots[slot].active;
}

bool vrphys_get_object_pose(int slot, float out_pos_units[3], float out_quat_xyzw[4], float out_lin_vel_mps[3],
                            float out_ang_vel_rps[3]) {
    if (slot < 0 || slot >= VRPHYS_SLOT_COUNT || !g_slots[slot].active || !g_slots[slot].state_valid ||
        !g_ctx.valid) {
        if (out_pos_units) {
            out_pos_units[0] = out_pos_units[1] = out_pos_units[2] = 0.0f;
        }
        if (out_quat_xyzw) {
            out_quat_xyzw[0] = out_quat_xyzw[1] = out_quat_xyzw[2] = 0.0f;
            out_quat_xyzw[3] = 1.0f;
        }
        if (out_lin_vel_mps) {
            out_lin_vel_mps[0] = out_lin_vel_mps[1] = out_lin_vel_mps[2] = 0.0f;
        }
        if (out_ang_vel_rps) {
            out_ang_vel_rps[0] = out_ang_vel_rps[1] = out_ang_vel_rps[2] = 0.0f;
        }
        return false;
    }
    const SlotState& sl = g_slots[slot];
    if (out_pos_units) {
        v3_store(to_world_units(sl.pos_m, g_ctx), out_pos_units);
    }
    if (out_quat_xyzw) {
        q4_store(qmul(g_ctx.turn_rot, sl.quat), out_quat_xyzw);
    }
    if (out_lin_vel_mps) {
        v3_store(qrot(g_ctx.turn_rot, sl.vel_mps), out_lin_vel_mps);
    }
    if (out_ang_vel_rps) {
        v3_store(qrot(g_ctx.turn_rot, sl.ang_vel_rps), out_ang_vel_rps);
    }
    return true;
}

void vrphys_log_set_enabled(bool enabled) {
    if (enabled && !g_log_enabled) {
        g_log_head = 0;
        g_log_count = 0;
    }
    g_log_enabled = enabled;
}

int vrphys_log_count() {
    return g_log_count;
}

int vrphys_log_write(const char* path) {
    if (g_log == nullptr || g_log_count == 0 || path == nullptr) {
        return 0;
    }
    FILE* f = fopen(path, "w");
    if (f == nullptr) {
        return -1;
    }
    fprintf(f, "# vr physics capture v1\n");
    fprintf(f, "# world_scale=%.4f records=%d\n", g_ctx.world_scale, g_log_count);
    fprintf(f, "# S rows: step,t_ns,dt,handx,handy,handz,sprx,spry,sprz,sqx,sqy,sqz,sqw,"
               "finx,finy,finz,fqx,fqy,fqz,fqw,vx,vy,vz,wx,wy,wz,"
               "rootx,rooty,rootz,tipx,tipy,tipz,nc,nprims,primhash\n");
    fprintf(f, "# C rows: step,idx,nx,ny,nz,pen,id,px,py,pz\n");
    for (int i = 0; i < g_log_count; i++) {
        const LogRec& r = g_log[(g_log_head + i) % kLogCap];
        fprintf(f,
                "S,%d,%llu,%.6f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,"
                "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%u\n",
                i, (unsigned long long)r.t, r.dt, r.hand[0], r.hand[1], r.hand[2], r.pos_spring[0],
                r.pos_spring[1], r.pos_spring[2], r.quat_spring[0], r.quat_spring[1], r.quat_spring[2],
                r.quat_spring[3], r.pos_final[0], r.pos_final[1], r.pos_final[2], r.quat_final[0],
                r.quat_final[1], r.quat_final[2], r.quat_final[3], r.vel[0], r.vel[1], r.vel[2], r.ang[0],
                r.ang[1], r.ang[2], r.root[0], r.root[1], r.root[2], r.tip[0], r.tip[1], r.tip[2], r.nc,
                r.nprims, r.prim_hash);
        for (int c = 0; c < r.nc && c < 4; c++) {
            fprintf(f, "C,%d,%d,%.5f,%.5f,%.5f,%.6f,%d,%.3f,%.3f,%.3f\n", i, c, r.c[c].n[0], r.c[c].n[1],
                    r.c[c].n[2], r.c[c].pen, r.c[c].id, r.c[c].p[0], r.c[c].p[1], r.c[c].p[2]);
        }
    }
    fclose(f);
    const int n = g_log_count;
    g_log_head = 0;
    g_log_count = 0;
    return n;
}

int vrphys_get_object_contacts(int slot, float* out_pos_units_xyz, float* out_normals_xyz, int max_contacts) {
    if (slot < 0 || slot >= VRPHYS_SLOT_COUNT || !g_ctx.valid || max_contacts <= 0) {
        return 0;
    }
    const SlotState& sl = g_slots[slot];
    const int n = sl.contact_count < max_contacts ? sl.contact_count : max_contacts;
    for (int i = 0; i < n; i++) {
        v3_store(to_world_units(sl.contact_pts[i], g_ctx), &out_pos_units_xyz[i * 3]);
        v3_store(qrot(g_ctx.turn_rot, sl.contact_ns[i]), &out_normals_xyz[i * 3]);
    }
    return n;
}

// The rendered orientation of a slot: sl.quat plus the cosmetic weight lag, which rides ONLY the
// served pose (rotation about the grip, raw frame) — physics stays on sl.quat.
static Q4 served_quat(const SlotState& sl) {
    const float om = len(sl.vis_off);
    if (om > 1e-5f) {
        const float half = om * 0.5f;
        const float s = sinf(half) / om;
        const Q4 offq = { sl.vis_off.x * s, sl.vis_off.y * s, sl.vis_off.z * s, cosf(half) };
        return qnorm(qmul(offq, sl.quat));
    }
    return sl.quat;
}

bool vrphys_get_hand_sim_pose_raw(int hand, float out_pos_m[3], float out_quat_xyzw[4]) {
    for (const SlotState& sl : g_slots) {
        if (sl.active && sl.state_valid && sl.desc.primary_hand == hand) {
            v3_store(sl.pos_m, out_pos_m);
            q4_store(served_quat(sl), out_quat_xyzw);
            return true;
        }
    }
    // A two-handed hold also owns the SECOND hand's position: pinned to its grip point on the
    // simulated handle, so Link's off hand holds the handle and lags with the object instead of
    // floating at the raw controller. The wrist keeps its own tracked orientation.
    for (const SlotState& sl : g_slots) {
        if (sl.active && sl.state_valid && sl.two_hand && sl.desc.secondary_hand == hand) {
            v3_store(add(sl.pos_m, qrot(served_quat(sl), v3(sl.desc.grip_local_secondary_m))), out_pos_m);
            q4_store(qnorm(g_hands[hand].pend_quat), out_quat_xyzw);
            return true;
        }
    }
    return false;
}

// --------------------------------------------------------------------------
// Contact primitives, blade path, events, haptics
// --------------------------------------------------------------------------

void vrphys_mesh_set_region(const float center_units[3], float radius_units, bool enabled) {
    g_mesh_enabled = enabled && center_units != nullptr && radius_units > 0.0f;
    if (!g_mesh_enabled) {
        g_mesh_count[0] = g_mesh_count[1] = 0;
        return;
    }
    g_mesh_center = v3(center_units);
    g_mesh_radius = radius_units;
}

void vrphys_mesh_mask(bool masked) {
    g_mesh_masked = masked;
}

void vrphys_mesh_set_flesh(bool flesh) {
    g_mesh_flesh = flesh;
}

bool vrphys_mesh_collecting() {
    return g_mesh_enabled && !g_mesh_masked;
}

int vrphys_mesh_get_debug_tris(float* out_xyz9_per_tri, int max_tris) {
    const int n = g_mesh_dbg_count < max_tris ? g_mesh_dbg_count : max_tris;
    for (int i = 0; i < n; i++) {
        v3_store(g_mesh_dbg[i].a, out_xyz9_per_tri + i * 9);
        v3_store(g_mesh_dbg[i].b, out_xyz9_per_tri + i * 9 + 3);
        v3_store(g_mesh_dbg[i].c, out_xyz9_per_tri + i * 9 + 6);
    }
    return n;
}

void vrphys_mesh_consider_tri(const float a[3], const float b[3], const float c[3]) {
    int& cnt = g_mesh_count[g_mesh_write];
    if (cnt >= kMeshCap) {
        return;
    }
    const float r = g_mesh_radius;
    const V3 va = v3(a), vb = v3(b), vc = v3(c);
    // Region reject on the tri's AABB (world units).
    const float minx = fminf(va.x, fminf(vb.x, vc.x)), maxx = fmaxf(va.x, fmaxf(vb.x, vc.x));
    if (maxx < g_mesh_center.x - r || minx > g_mesh_center.x + r) {
        return;
    }
    const float miny = fminf(va.y, fminf(vb.y, vc.y)), maxy = fmaxf(va.y, fmaxf(vb.y, vc.y));
    if (maxy < g_mesh_center.y - r || miny > g_mesh_center.y + r) {
        return;
    }
    const float minz = fminf(va.z, fminf(vb.z, vc.z)), maxz = fmaxf(va.z, fmaxf(vb.z, vc.z));
    if (maxz < g_mesh_center.z - r || minz > g_mesh_center.z + r) {
        return;
    }
    // BACK-FACE rejection, the lesson the collision-mesh gather already taught: the harvest
    // sphere reaches through walls, and the far side of a wall (or the world's outer shell)
    // pushes the opposite way from the face the blade is actually touching — feeding the
    // solver both is the flat-surface jitter. Rendered geometry is one-sided (CCW front), so
    // drop faces whose front does not look at the hand region. Double-sided decor is drawn
    // as two windings, and the facing copy survives.
    const V3 n = vcross(sub(vb, va), sub(vc, va));
    const float n_len = len(n);
    if (n_len < 1e-6f) {
        return; // degenerate
    }
    const float back_margin = 4.0f; // world units, matches the collision gather's kBackMargin
    if (vdot(n, sub(g_mesh_center, va)) < -back_margin * n_len) {
        return;
    }
    g_mesh_buf[g_mesh_write][cnt++] = { va, vb, vc, g_mesh_flesh ? (3 << 12) : (1 << 12) };
}

void vrphys_set_contact_prims(const VrPhysContactPrim* prims, int count) {
    if (prims == nullptr || count <= 0) {
        g_prim_count = 0;
        return;
    }
    if (count > VRPHYS_MAX_CONTACT_PRIMS) {
        count = VRPHYS_MAX_CONTACT_PRIMS;
    }
    memcpy(g_prims, prims, sizeof(VrPhysContactPrim) * count);
    g_prim_count = count;

    // Order-independent hash of the pushed set (type + id + vertices, quantized to 1/8 unit).
    // Static level geometry re-pushed each tick must hash IDENTICALLY; a changing hash while the
    // player stands still is direct evidence of discovery churn rather than a solver problem.
    uint32_t h = 0;
    for (int i = 0; i < count; i++) {
        uint32_t e = 2166136261u;
        auto mix = [&e](uint32_t v) {
            e ^= v;
            e *= 16777619u;
        };
        mix((uint32_t)prims[i].type);
        mix((uint32_t)prims[i].id);
        const float* f[3] = { prims[i].a, prims[i].b, prims[i].c };
        for (int k = 0; k < 3; k++) {
            for (int j = 0; j < 3; j++) {
                mix((uint32_t)(int32_t)(f[k][j] * 8.0f));
            }
        }
        h += e; // sum => order independent
    }
    g_prim_hash = h;
}

int vrphys_get_blade_path(int slot, VrPhysBladeSample* out, int max_samples) {
    if (slot < 0 || slot >= VRPHYS_SLOT_COUNT || out == nullptr || max_samples <= 0) {
        return 0;
    }
    SlotState& sl = g_slots[slot];
    const int n = sl.blade_count < max_samples ? sl.blade_count : max_samples;
    for (int i = 0; i < n; i++) {
        out[i] = sl.blade_ring[(sl.blade_head + i) % kBladeRingCap];
    }
    sl.blade_head = (sl.blade_head + n) % kBladeRingCap;
    sl.blade_count -= n;
    return n;
}

int vrphys_drain_events(VrPhysEvent* out, int max_events) {
    if (out == nullptr || max_events <= 0 || g_event_count == 0) {
        return 0;
    }
    const int n = g_event_count < max_events ? g_event_count : max_events;
    memcpy(out, g_events, sizeof(VrPhysEvent) * n);
    if (n < g_event_count) {
        memmove(g_events, g_events + n, sizeof(VrPhysEvent) * (g_event_count - n));
    }
    g_event_count -= n;
    return n;
}

int vrphys_take_haptic_requests(VrPhysHapticReq* out, int max) {
    if (out == nullptr || max <= 0 || g_haptic_count == 0) {
        return 0;
    }
    const int n = g_haptic_count < max ? g_haptic_count : max;
    memcpy(out, g_haptics, sizeof(VrPhysHapticReq) * n);
    if (n < g_haptic_count) {
        memmove(g_haptics, g_haptics + n, sizeof(VrPhysHapticReq) * (g_haptic_count - n));
    }
    g_haptic_count -= n;
    return n;
}
