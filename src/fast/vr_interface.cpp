#include "vr_interface.h"
#include "fast/vr_openxr.h"
#include "fast/vr_physics.h"

#include <chrono>
#include <string.h>

namespace {
std::chrono::steady_clock::time_point g_game_tick_start;
} // namespace

extern "C" {

void VR_GameTickBegin(void) {
    if (vr_is_initialized()) {
        g_game_tick_start = std::chrono::steady_clock::now();
    }
}

void VR_GameTickEnd(void) {
    if (vr_is_initialized()) {
        vr_report_game_tick_ms(
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - g_game_tick_start).count());
    }
}

void VR_SetFlatScreen(bool enabled) {
    vr_set_flat_screen(enabled);
}

bool VR_IsFlatScreen(void) {
    return vr_get_flat_screen();
}

bool VR_IsInitialized() {
    return vr_is_initialized();
}

void VR_ApplyModeRequest(void) {
    vr_apply_mode_request();
}

void VR_SetOverlayDisplayList(void* commands) {
    vr_set_hud_commands(commands);
}

void VR_SetLinkEyeHeight(float units) {
    vr_set_link_eye_height(units);
}

void VR_SetViewFade(float fade) {
    vr_set_view_fade(fade);
}

void VR_SetCameraYaw(int16_t yaw_binang) {
    vr_set_camera_yaw(yaw_binang);
}

void VR_SetFirstPerson(bool enabled) {
    vr_set_first_person(enabled);
}

bool VR_GetFirstPerson(void) {
    return vr_is_first_person();
}

void VR_SetCameraAnchor(float x, float y, float z) {
    vr_set_camera_anchor(x, y, z);
}

int16_t VR_GetHeadYaw(void) {
    return vr_get_head_yaw();
}

int16_t VR_GetHeadingYaw(void) {
    return vr_get_heading_yaw();
}

void VR_RecenterHeading(int16_t linkYaw) {
    vr_recenter_heading(linkYaw);
}

void VR_SetLockOnYaw(int16_t yawBinang, bool active) {
    vr_set_lockon_yaw(yawBinang, active);
}

void VR_GetCameraPose(float eye[3], float fwd[3], float up[3]) {
    vr_get_camera_pose(eye, fwd, up);
}

float VR_GetCullingFovy(void) {
    return vr_get_culling_fovy();
}

void VR_GetRoomscaleDesired(float out[2]) {
    vr_get_roomscale_desired(out);
}

void VR_AddRoomscaleDisplacement(float dx, float dz) {
    vr_add_roomscale_displacement(dx, dz);
}

void VR_GetRoomscaleOrigin(float out[2]) {
    vr_get_roomscale_origin(out);
}

void VR_ResetRoomscale(void) {
    vr_reset_roomscale();
}

void VR_ClampRoomscaleLean(float max_units) {
    vr_clamp_roomscale_lean(max_units);
}

bool VR_GetAimRay(int hand, float pos[3], float dir[3]) {
    return vr_get_aim_ray(hand, pos, dir);
}

bool VR_GetHandPose(int hand, float pos[3], float quat[4]) {
    return vr_get_hand_pose(hand, pos, quat);
}

bool VR_IsHandActive(int hand) {
    return vr_is_hand_active(hand);
}

uint16_t VR_GetControllerButton(int hand) {
    return vr_get_controller_buttons(hand);
}

void VR_GetThumbstick(int hand, float* x, float* y) {
    vr_get_thumbstick(hand, x, y);
}

void VR_SetStickSuppressed(int hand, int32_t suppressed) {
    vr_set_stick_suppressed(hand, suppressed != 0);
}

float VR_GetTrigger(int hand) {
    return vr_get_trigger(hand);
}

float VR_GetGrip(int hand) {
    return vr_get_grip(hand);
}

bool VR_GetHandMatrix(int hand, float out[4][4]) {
    return vr_get_hand_matrix(hand, out);
}

int32_t VR_PhysGetInterfaceVersion(void) {
    // Compiled into the library: the game compares this against ITS copy of the header's
    // VR_PHYS_INTERFACE_VERSION, so a stale submodule build is caught at init.
    return VR_PHYS_INTERFACE_VERSION;
}

bool VR_GetHandVelocity(int hand, float linVelMps[3], float angVelRps[3]) {
    return vrphys_get_hand_velocity(hand, linVelMps, angVelRps);
}

int32_t VR_GetHandPath(int hand, VrHandSample* out, int32_t maxSamples) {
    if (out == nullptr || maxSamples <= 0) {
        return 0;
    }
    // VrHandSample (C ABI) and VrPhysHandSample (internal) are field-for-field identical; copy
    // explicitly rather than aliasing so the two headers can evolve with a compile error, not UB.
    static_assert(sizeof(VrHandSample) == sizeof(VrPhysHandSample), "hand sample structs must match");
    VrPhysHandSample tmp[16];
    int32_t total = 0;
    while (total < maxSamples) {
        int want = maxSamples - total;
        if (want > 16) {
            want = 16;
        }
        const int got = vrphys_get_hand_path(hand, tmp, want);
        for (int i = 0; i < got; i++) {
            memcpy(out[total + i].pos, tmp[i].pos_units, sizeof(float) * 3);
            memcpy(out[total + i].quat, tmp[i].quat, sizeof(float) * 4);
            memcpy(out[total + i].linVelMps, tmp[i].lin_vel_mps, sizeof(float) * 3);
            memcpy(out[total + i].angVelRps, tmp[i].ang_vel_rps, sizeof(float) * 3);
            out[total + i].timeNs = tmp[i].time_ns;
        }
        total += got;
        if (got < want) {
            break;
        }
    }
    return total;
}

void VR_TriggerHaptic(int hand, float amplitude01, float freqHz, float durationMs) {
    vr_trigger_haptic(hand, amplitude01, freqHz, durationMs);
}

float VR_GetWorldScale(void) {
    return vr_get_world_scale();
}

void VR_PhysSetObject(int slot, const VrHeldObjectDesc* descOrNull) {
    if (descOrNull == nullptr) {
        vrphys_set_object(slot, nullptr);
        return;
    }
    const VrHeldObjectDesc& d = *descOrNull;
    VrPhysObjectDesc tmp;
    tmp.primary_hand = d.primaryHand;
    tmp.secondary_hand = d.secondaryHand;
    tmp.lin_freq_hz = d.linFreqHz;
    tmp.lin_zeta = d.linZeta;
    tmp.ang_freq_hz = d.angFreqHz;
    tmp.ang_zeta = d.angZeta;
    tmp.max_accel_mps2 = d.maxAccelMps2;
    memcpy(tmp.grip_local_root_m, d.gripLocalRootM, sizeof(float) * 3);
    memcpy(tmp.grip_local_tip_m, d.gripLocalTipM, sizeof(float) * 3);
    tmp.contact_enabled = d.contactEnabled != 0;
    tmp.friction = d.friction;
    tmp.blade_radius_m = d.bladeRadiusM;
    tmp.touch_tolerance_m = d.touchToleranceM;
    tmp.max_ang_accel = d.maxAngAccel;
    tmp.pivot_only = d.pivotOnly != 0;
    tmp.grip_local_edge_m[0] = d.gripLocalEdgeM[0];
    tmp.grip_local_edge_m[1] = d.gripLocalEdgeM[1];
    tmp.grip_local_edge_m[2] = d.gripLocalEdgeM[2];
    tmp.tip_taper_frac = d.tipTaperFrac;
    tmp.passthrough_speed_mps = d.passthroughSpeedMps;
    tmp.cut_drag_flesh = d.cutDragFlesh;
    tmp.cut_drag_world = d.cutDragWorld;
    tmp.visual_lag_s = d.visualLagS;
    tmp.visual_snap_hz = d.visualSnapHz;
    memcpy(tmp.grip_local_secondary_m, d.gripLocalSecondaryM, sizeof(float) * 3);
    memcpy(tmp.grip_local_com_m, d.gripLocalComM, sizeof(float) * 3);
    tmp.gravity_scale = d.gravityScale;
    vrphys_set_object(slot, &tmp);
}

void VR_PhysSetMeshRegion(const float centerUnits[3], float radiusUnits, int32_t enabled) {
    vrphys_mesh_set_region(centerUnits, radiusUnits, enabled != 0);
}

void VR_PhysMeshMask(int32_t masked) {
    vrphys_mesh_mask(masked != 0);
}

int32_t VR_PhysGetMeshDebugTris(float* outXyz9PerTri, int32_t maxTris) {
    return vrphys_mesh_get_debug_tris(outXyz9PerTri, maxTris);
}

void VR_PhysSetContactPrims(const VrContactPrim* prims, int32_t count) {
    VrPhysContactPrim tmp[VRPHYS_MAX_CONTACT_PRIMS];
    int n = 0;
    if (prims != nullptr && count > 0) {
        n = count < VRPHYS_MAX_CONTACT_PRIMS ? count : VRPHYS_MAX_CONTACT_PRIMS;
        for (int i = 0; i < n; i++) {
            tmp[i].type = prims[i].type;
            memcpy(tmp[i].a, prims[i].a, sizeof(float) * 3);
            memcpy(tmp[i].b, prims[i].b, sizeof(float) * 3);
            memcpy(tmp[i].c, prims[i].c, sizeof(float) * 3);
            tmp[i].radius = prims[i].radius;
            tmp[i].id = prims[i].id;
        }
    }
    vrphys_set_contact_prims(n > 0 ? tmp : nullptr, n);
}

int32_t VR_PhysGetBladePath(int slot, VrBladeSample* out, int32_t maxSamples) {
    if (out == nullptr || maxSamples <= 0) {
        return 0;
    }
    VrPhysBladeSample tmp[16];
    int32_t total = 0;
    while (total < maxSamples) {
        int want = maxSamples - total;
        if (want > 16) {
            want = 16;
        }
        const int got = vrphys_get_blade_path(slot, tmp, want);
        for (int i = 0; i < got; i++) {
            memcpy(out[total + i].root, tmp[i].root_units, sizeof(float) * 3);
            memcpy(out[total + i].tip, tmp[i].tip_units, sizeof(float) * 3);
            memcpy(out[total + i].midVelMps, tmp[i].mid_vel_mps, sizeof(float) * 3);
            memcpy(out[total + i].gripPos, tmp[i].grip_units, sizeof(float) * 3);
            memcpy(out[total + i].quat, tmp[i].quat, sizeof(float) * 4);
            out[total + i].timeNs = tmp[i].time_ns;
        }
        total += got;
        if (got < want) {
            break;
        }
    }
    return total;
}

int32_t VR_PhysGetContacts(int slot, float* outPosXYZ, float* outNormalXYZ, int32_t maxContacts) {
    return vrphys_get_object_contacts(slot, outPosXYZ, outNormalXYZ, maxContacts);
}

void VR_PhysLogSetEnabled(bool enabled) {
    vrphys_log_set_enabled(enabled);
}

int32_t VR_PhysLogCount(void) {
    return vrphys_log_count();
}

int32_t VR_PhysLogWrite(const char* path) {
    return vrphys_log_write(path);
}

int32_t VR_PhysDrainEvents(VrContactEvent* out, int32_t maxEvents) {
    if (out == nullptr || maxEvents <= 0) {
        return 0;
    }
    VrPhysEvent tmp[8];
    int32_t total = 0;
    while (total < maxEvents) {
        int want = maxEvents - total;
        if (want > 8) {
            want = 8;
        }
        const int got = vrphys_drain_events(tmp, want);
        for (int i = 0; i < got; i++) {
            out[total + i].type = tmp[i].type;
            out[total + i].slot = tmp[i].slot;
            out[total + i].primId = tmp[i].prim_id;
            memcpy(out[total + i].pos, tmp[i].pos_units, sizeof(float) * 3);
            memcpy(out[total + i].normal, tmp[i].normal, sizeof(float) * 3);
            out[total + i].impactMps = tmp[i].impact_mps;
            out[total + i].timeNs = tmp[i].time_ns;
        }
        total += got;
        if (got < want) {
            break;
        }
    }
    return total;
}

void VR_SetHandScale(float s) {
    vr_set_hand_scale(s);
}

void VR_SetHandMirror(int hand, bool mirror) {
    vr_set_hand_mirror(hand, mirror);
}

void VR_RegisterHandMatrix(const void* mtx, int hand) {
    vr_register_hand_matrix(mtx, hand);
}

void VR_RegisterHandChildMatrix(const void* mtx, int hand, const float* localMf16) {
    vr_register_hand_child_matrix(mtx, hand, localMf16);
}

void VR_RegisterHeadChildMatrix(const void* mtx, const float* localMf16) {
    vr_register_head_child_matrix(mtx, localMf16);
}

bool VR_GetHeadMatrix(float out[4][4]) {
    return vr_get_head_matrix(out);
}

void VR_ClearHandMatrices(void) {
    vr_clear_hand_matrices();
}

}
