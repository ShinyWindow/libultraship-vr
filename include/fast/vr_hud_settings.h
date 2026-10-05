#pragma once

// Wrist HUD layout settings, shared by the renderer (vr_openxr.cpp reads them) and the game's
// menu (soh SohMenuVRSettings.cpp edits and exports them), so both agree on every CVar name and
// default. Two profiles, Adult and Child: Link's age picks one at runtime.
//
// Units. Element offsets, card sizes and padding are LAYOUT UNITS: 1 unit = one native HUD pixel
// at 100% element size (a heart is about 10 units). Panel position is centimetres in the
// controller's grip frame (+X right, +Y up, -Z forward), rotation degrees, panel size percent of
// 0.8 mm per unit.
//
// CVar names: gVrHud.<Adult|Child>.<Left|Right>Panel.<Field>, gVrHud.<Adult|Child>.El.<Key>.<Field>

#include <cstdio>
#include "vr_interface.h"

struct VrHudPanelDefaults {
    float x, y, z;            // cm, grip frame
    float pitch, yaw, roll;   // degrees
    float scale;              // percent
    float width, height;      // layout units, 0 = fit the elements
    float padding;            // layout units around the elements
    float backing;            // percent opacity of the dark card
};

// Defaults [profile][hand]: profile 0 = Adult, 1 = Child; hand 0 = left controller (vitals),
// 1 = right controller (buttons + minimap). Tuned in the headset by the user (October 4, 2026).
// Child keeps its own panel placement (X..Roll) with the adult card shape (Scale..Backing).
static const VrHudPanelDefaults kVrHudPanelDefaults[2][2] = {
    {
        { 2.44f, 4.71f, -7.14f, -47.52f, 153.26f, -149.48f, 100.0f, 0.0f, 0.0f, 3.0f, 70.0f },
        { 1.87f, 2.85f, -5.75f, -39.96f, 144.63f, -174.85f, 100.0f, 143.0f, 137.0f, 3.0f, 73.0f },
    },
    {
        { 0.53f, -2.04f, -9.54f, -37.17f, 154.57f, -155.33f, 100.0f, 0.0f, 0.0f, 3.0f, 70.0f },
        { -4.12f, 6.51f, -3.19f, -65.64f, -134.79f, 136.67f, 100.0f, 143.0f, 137.0f, 3.0f, 73.0f },
    },
};

struct VrHudElementDesc {
    int id;            // VR_HUD_EL_*
    const char* key;   // CVar key
    const char* label; // menu label
    int hand;          // 0 left, 1 right
    float x[2], y[2];  // default offset from the auto-layout slot, layout units [Adult, Child]
    float scale[2];    // default size, percent
    int show[2];       // default visibility
};

// Element offsets are the same for both ages (the user's adult layout).
#define VR_HUD_EL_DEFAULT { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 100.0f, 100.0f }, { 1, 1 }
#define VR_HUD_EL_AT(x, y) { x, x }, { y, y }, { 100.0f, 100.0f }, { 1, 1 }
static const VrHudElementDesc kVrHudElements[] = {
    { VR_HUD_EL_HEARTS, "Hearts", "Hearts", 0, VR_HUD_EL_AT(3.0f, 0.0f) },
    { VR_HUD_EL_MAGIC, "Magic", "Magic Bar", 0, VR_HUD_EL_AT(-25.0f, -3.0f) },
    { VR_HUD_EL_RUPEES, "Rupees", "Rupees", 0, VR_HUD_EL_DEFAULT },
    { VR_HUD_EL_KEYS, "Keys", "Small Keys", 0, VR_HUD_EL_DEFAULT },
    { VR_HUD_EL_TIMER, "Timer", "Timer", 0, VR_HUD_EL_DEFAULT },
    { VR_HUD_EL_GAME_TIMER, "GameTimer", "Gameplay Timer", 0, VR_HUD_EL_DEFAULT },
    { VR_HUD_EL_BTN_B, "ButtonB", "B Button", 1, VR_HUD_EL_AT(-7.0f, 0.0f) },
    { VR_HUD_EL_BTN_A, "ButtonA", "A Button", 1, VR_HUD_EL_AT(-10.0f, 0.0f) },
    { VR_HUD_EL_BTN_C_UP, "ButtonCUp", "C-Up (Navi)", 1, VR_HUD_EL_AT(-10.0f, 0.0f) },
    { VR_HUD_EL_BTN_C_LEFT, "ButtonCLeft", "C-Left", 1, VR_HUD_EL_AT(-10.0f, 0.0f) },
    { VR_HUD_EL_BTN_C_DOWN, "ButtonCDown", "C-Down", 1, VR_HUD_EL_AT(-10.0f, 0.0f) },
    { VR_HUD_EL_BTN_C_RIGHT, "ButtonCRight", "C-Right", 1, VR_HUD_EL_AT(-10.0f, 0.0f) },
    { VR_HUD_EL_BTN_DPAD, "DPad", "D-Pad", 1, VR_HUD_EL_DEFAULT },
    { VR_HUD_EL_MOUNT, "Mount", "Carrots / Archery Score", 1, VR_HUD_EL_DEFAULT },
    { VR_HUD_EL_MINIMAP, "Minimap", "Minimap", 1, VR_HUD_EL_AT(-1.0f, -11.0f) },
};
#undef VR_HUD_EL_AT
#undef VR_HUD_EL_DEFAULT
static const int kVrHudElementCount = (int)(sizeof(kVrHudElements) / sizeof(kVrHudElements[0]));

// The button cluster keeps vanilla's arrangement (z64interface.h, also the defaults of SoH's HUD
// editor): TV-pixel top-lefts B (160, 17), A (186, 9), C-left (227, 18), C-down (249, 34),
// C-up (254, 16), C-right (271, 18). In VR all of them are right-edge anchored by the same shift,
// so the renderer places the cluster rigidly from this one fixed reference point.
static const float kVrHudClusterRefX = 160.0f; // B_BUTTON_X (leftmost)
static const float kVrHudClusterRefY = 9.0f;   // A_BUTTON_Y (topmost)

static inline const char* VrHudProfileName(int child) {
    return child ? "Child" : "Adult";
}

static inline void VrHudPanelCVar(char* out, size_t n, int child, int hand, const char* field) {
    snprintf(out, n, "gVrHud.%s.%sPanel.%s", VrHudProfileName(child), hand ? "Right" : "Left", field);
}

static inline void VrHudElementCVar(char* out, size_t n, int child, const char* key, const char* field) {
    snprintf(out, n, "gVrHud.%s.El.%s.%s", VrHudProfileName(child), key, field);
}

static inline const VrHudElementDesc* VrHudElementById(int id) {
    for (int i = 0; i < kVrHudElementCount; i++) {
        if (kVrHudElements[i].id == id) {
            return &kVrHudElements[i];
        }
    }
    return nullptr;
}
