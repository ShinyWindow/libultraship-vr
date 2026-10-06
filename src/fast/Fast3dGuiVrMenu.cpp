// SOH [VR] The SoH (ImGui) menu on the headset's menu panel: the panel frame (display size, the
// controller-ray mouse), the panel render, and the on-panel keyboard. The panel quad, the ray
// hit-test, the open/close button and the game-input gate live in vr_openxr.cpp ("SoH menu panel").
//
// One ImGui frame serves both screens: it is laid out for the panel (1600x1000), drawn into the
// panel's image without the desktop's game view, then drawn to the desktop window as well (scaled
// to fit on OpenGL, 1:1 on DirectX 11, whose ImGui renderer ignores FramebufferScale).
// The keyboard is drawn on the foreground draw list and hit-tested here, never as ImGui widgets:
// a click on an ImGui item would take the active id away from the text field being typed into.
#include "fast/Fast3dGui.h"
#include "fast/vr_openxr.h"
#include "ship/Context.h"
#include "ship/window/Window.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cfloat>
#include <chrono>

namespace Fast {

namespace {

// One key. Character keys (key == ImGuiKey_None) type ch, or shiftCh after Shift.
struct VrKeyDef {
    int row;
    const char* label;
    const char* shiftLabel;
    char ch, shiftCh;
    ImGuiKey key;
    float width; // in key units
};

constexpr ImGuiKey kShiftKey = ImGuiKey_LeftShift; // toggles the one-shot shift; never sent to ImGui

// clang-format off
const VrKeyDef kKeys[] = {
    { 0, "`", "~", '`', '~' }, { 0, "1", "!", '1', '!' }, { 0, "2", "@", '2', '@' }, { 0, "3", "#", '3', '#' },
    { 0, "4", "$", '4', '$' }, { 0, "5", "%", '5', '%' }, { 0, "6", "^", '6', '^' }, { 0, "7", "&", '7', '&' },
    { 0, "8", "*", '8', '*' }, { 0, "9", "(", '9', '(' }, { 0, "0", ")", '0', ')' }, { 0, "-", "_", '-', '_' },
    { 0, "=", "+", '=', '+' }, { 0, "Back", "Back", 0, 0, ImGuiKey_Backspace, 2.0f },

    { 1, "q", "Q", 'q', 'Q' }, { 1, "w", "W", 'w', 'W' }, { 1, "e", "E", 'e', 'E' }, { 1, "r", "R", 'r', 'R' },
    { 1, "t", "T", 't', 'T' }, { 1, "y", "Y", 'y', 'Y' }, { 1, "u", "U", 'u', 'U' }, { 1, "i", "I", 'i', 'I' },
    { 1, "o", "O", 'o', 'O' }, { 1, "p", "P", 'p', 'P' }, { 1, "[", "{", '[', '{' }, { 1, "]", "}", ']', '}' },
    { 1, "\\", "|", '\\', '|' },

    { 2, "a", "A", 'a', 'A' }, { 2, "s", "S", 's', 'S' }, { 2, "d", "D", 'd', 'D' }, { 2, "f", "F", 'f', 'F' },
    { 2, "g", "G", 'g', 'G' }, { 2, "h", "H", 'h', 'H' }, { 2, "j", "J", 'j', 'J' }, { 2, "k", "K", 'k', 'K' },
    { 2, "l", "L", 'l', 'L' }, { 2, ";", ":", ';', ':' }, { 2, "'", "\"", '\'', '"' },
    { 2, "Done", "Done", 0, 0, ImGuiKey_Enter, 2.0f },

    { 3, "Shift", "SHIFT", 0, 0, kShiftKey, 2.0f },
    { 3, "z", "Z", 'z', 'Z' }, { 3, "x", "X", 'x', 'X' }, { 3, "c", "C", 'c', 'C' }, { 3, "v", "V", 'v', 'V' },
    { 3, "b", "B", 'b', 'B' }, { 3, "n", "N", 'n', 'N' }, { 3, "m", "M", 'm', 'M' }, { 3, ",", "<", ',', '<' },
    { 3, ".", ">", '.', '>' }, { 3, "/", "?", '/', '?' },

    { 4, "Space", "Space", ' ', ' ', ImGuiKey_None, 8.0f },
    { 4, "<", "<", 0, 0, ImGuiKey_LeftArrow, 1.5f }, { 4, ">", ">", 0, 0, ImGuiKey_RightArrow, 1.5f },
};
// clang-format on
constexpr int kKeyCount = (int)(sizeof(kKeys) / sizeof(kKeys[0]));
constexpr int kRows = 5;
constexpr float kRowUnits = 15.0f; // the widest row (row 0), in key units

float KeyWidth(const VrKeyDef& k) {
    return k.width > 0.0f ? k.width : 1.0f;
}

bool Repeats(const VrKeyDef& k) {
    return k.key == ImGuiKey_Backspace || k.key == ImGuiKey_LeftArrow || k.key == ImGuiKey_RightArrow;
}

double NowSeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

void Fast3dGui::VrMenuNewFrame() {
    ImGuiIO& io = ImGui::GetIO();
    mVrPanelFrame = vr_menu_panel_active();

    if (!mVrPanelFrame) {
        if (mVrPanelPrev) {
            // Back to the desktop: its own layout, mouse and (if they were on) floating windows.
            if (mVrViewportsWereOn) {
                io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
            }
            io.ConfigInputTrickleEventQueue = mVrTrickleWas;
            if (mVrKeyUp != ImGuiKey_None) {
                io.AddKeyEvent(mVrKeyUp, false);
                mVrKeyUp = ImGuiKey_None;
            }
            io.MouseDrawCursor = false;
            io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
            mVrKeyboardShown = false;
            mVrShift = false;
            mVrPrevDown = false;
            mVrKeyHeld = -1;
        }
        mVrPanelPrev = false;
        return;
    }

    if (!mVrPanelPrev) {
        // Floating (OS) windows can't exist on the panel: keep everything in the main viewport.
        mVrViewportsWereOn = (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) != 0;
        io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;
        // The ray moves the mouse every frame, and scrolling adds a wheel event every frame too.
        // A trickling queue applies only one of those per frame and keeps the rest for later, so the
        // backlog (and the pointer's lag) would grow for as long as the stick is held. Every event
        // here is already one per frame: apply them all.
        mVrTrickleWas = io.ConfigInputTrickleEventQueue;
        io.ConfigInputTrickleEventQueue = false;
    }
    mVrPanelPrev = true;

    // Without trickling, a key's down and up in one frame would read as never pressed: the up goes
    // out a frame after the down.
    if (mVrKeyUp != ImGuiKey_None) {
        io.AddKeyEvent(mVrKeyUp, false);
        mVrKeyUp = ImGuiKey_None;
    }

    int w = 0, h = 0;
    vr_menu_panel_size(&w, &h);
    io.DisplaySize = ImVec2((float)w, (float)h);
    io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
    io.MouseDrawCursor = true; // the cursor on the panel shows where the ray lands

    VrMenuPointer p;
    vr_menu_take_pointer(&p);
    mVrPointerValid = p.valid;
    mVrPointerX = p.x;
    mVrPointerY = p.y;
    const bool pressEdge = p.down && !mVrPrevDown;
    mVrPrevDown = p.down;

    const bool onKeyboard = mVrKeyboardShown && p.valid && p.x >= mVrKeyboardMin.x && p.x <= mVrKeyboardMax.x &&
                            p.y >= mVrKeyboardMin.y && p.y <= mVrKeyboardMax.y;
    if (onKeyboard) {
        int hit = -1;
        for (const VrKeyRect& r : mVrKeys) {
            if (p.x >= r.min.x && p.x <= r.max.x && p.y >= r.min.y && p.y <= r.max.y) {
                hit = r.key;
                break;
            }
        }
        const double now = NowSeconds();
        bool press = false;
        if (pressEdge && hit >= 0) {
            press = true;
            mVrKeyHeld = hit;
            mVrKeyRepeatAt = now + 0.45;
        } else if (p.down && hit >= 0 && hit == mVrKeyHeld && Repeats(kKeys[hit]) && now >= mVrKeyRepeatAt) {
            press = true;
            mVrKeyRepeatAt = now + 0.06;
        }
        if (!p.down) {
            mVrKeyHeld = -1;
        }
        if (press) {
            const VrKeyDef& k = kKeys[hit];
            if (k.key == kShiftKey) {
                mVrShift = !mVrShift;
            } else if (k.key == ImGuiKey_None) {
                io.AddInputCharacter((unsigned int)(unsigned char)(mVrShift ? k.shiftCh : k.ch));
                mVrShift = false;
            } else {
                io.AddKeyEvent(k.key, true);
                mVrKeyUp = k.key;
            }
            vr_trigger_haptic(p.hand, 0.2f, 0.0f, 10.0f);
        }
        // ImGui must not see this click (it would deactivate the text field being typed into).
        io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    } else {
        mVrKeyHeld = -1;
        io.AddMouseSourceEvent(ImGuiMouseSource_Mouse);
        io.AddMousePosEvent(p.valid ? p.x : -FLT_MAX, p.valid ? p.y : -FLT_MAX);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, p.down);
    }
    if (p.wheel != 0.0f) {
        io.AddMouseWheelEvent(0.0f, p.wheel);
    }
}

void Fast3dGui::VrMenuDrawKeyboard() {
    mVrKeyboardShown = false;
    mVrKeys.clear();
    ImGuiIO& io = ImGui::GetIO();
    if (!mVrPanelFrame || !io.WantTextInput) {
        return;
    }

    const float W = io.DisplaySize.x;
    const float H = io.DisplaySize.y;
    const float pad = 12.0f;
    const float kbW = W * 0.94f;
    const float unit = kbW / kRowUnits;
    const float keyH = unit * 0.8f;
    const float gap = unit * 0.08f;
    const float kbH = kRows * keyH + 2.0f * pad;
    // Below the text field's caret if there is room, above it otherwise.
    const ImGuiContext& g = *ImGui::GetCurrentContext();
    const float caretY = g.PlatformImeData.WantVisible ? g.PlatformImeData.InputPos.y : 0.0f;
    const float top = caretY > H * 0.55f ? pad : H - kbH - pad;
    const float left = (W - kbW) * 0.5f;
    mVrKeyboardMin = ImVec2(left - pad, top - pad);
    mVrKeyboardMax = ImVec2(left + kbW + pad, top + kbH - pad);

    ImDrawList* dl = ImGui::GetForegroundDrawList(ImGui::GetMainViewport()); // no current window here
    dl->AddRectFilled(mVrKeyboardMin, mVrKeyboardMax, IM_COL32(18, 20, 26, 240), 14.0f);
    ImFont* font = ImGui::GetFont();
    const float fontSize = keyH * 0.42f;

    // Row widths, to centre each row.
    float rowUnits[kRows] = {};
    for (int i = 0; i < kKeyCount; i++) {
        rowUnits[kKeys[i].row] += KeyWidth(kKeys[i]);
    }
    float x[kRows];
    for (int r = 0; r < kRows; r++) {
        x[r] = left + (kbW - rowUnits[r] * unit) * 0.5f;
    }

    for (int i = 0; i < kKeyCount; i++) {
        const VrKeyDef& k = kKeys[i];
        const float kw = KeyWidth(k) * unit;
        const ImVec2 mn(x[k.row] + gap * 0.5f, top + k.row * keyH + gap * 0.5f);
        const ImVec2 mx(x[k.row] + kw - gap * 0.5f, top + (k.row + 1) * keyH - gap * 0.5f);
        x[k.row] += kw;
        mVrKeys.push_back({ mn, mx, i });

        const bool hover = mVrPointerValid && mVrPointerX >= mn.x && mVrPointerX <= mx.x && mVrPointerY >= mn.y &&
                           mVrPointerY <= mx.y;
        const bool held = mVrKeyHeld == i;
        const bool shiftOn = k.key == kShiftKey && mVrShift;
        const ImU32 fill = held || shiftOn ? IM_COL32(70, 130, 200, 255)
                           : hover         ? IM_COL32(78, 84, 98, 255)
                                           : IM_COL32(48, 52, 62, 255);
        dl->AddRectFilled(mn, mx, fill, 8.0f);

        const char* label = mVrShift ? k.shiftLabel : k.label;
        const ImVec2 ts = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, label);
        dl->AddText(font, fontSize, ImVec2((mn.x + mx.x - ts.x) * 0.5f, (mn.y + mx.y - ts.y) * 0.5f),
                    IM_COL32(235, 238, 245, 255), label);
    }
    mVrKeyboardShown = true;
}

void Fast3dGui::VrMenuRenderPanel(ImDrawData* data) {
    // The panel gets everything but the desktop's game view (the "Main Game" window: the headset
    // has the real world behind the panel).
    const ImGuiWindow* game = ImGui::FindWindowByName("Main Game");
    ImDrawData panel = *data;
    panel.CmdLists.resize(0);
    panel.TotalVtxCount = 0;
    panel.TotalIdxCount = 0;
    for (ImDrawList* list : data->CmdLists) {
        if (game != nullptr && list == game->DrawList) {
            continue;
        }
        panel.CmdLists.push_back(list);
        panel.TotalVtxCount += list->VtxBuffer.Size;
        panel.TotalIdxCount += list->IdxBuffer.Size;
    }
    panel.CmdListsCount = panel.CmdLists.Size;
    panel.FramebufferScale = ImVec2(1.0f, 1.0f);
    if (vr_begin_menu()) {
        RenderDrawDataBackend(&panel);
        vr_end_menu(); // also binds the window framebuffer again
    }

    // The desktop shows the same frame, fitted into the window.
    auto wnd = Ship::Context::GetRawInstance()->GetWindow();
    if (wnd != nullptr && data->DisplaySize.x > 0.0f && data->DisplaySize.y > 0.0f) {
        const float s = std::min((float)wnd->GetWidth() / data->DisplaySize.x, (float)wnd->GetHeight() / data->DisplaySize.y);
        if (s > 0.0f) {
            data->FramebufferScale = ImVec2(s, s);
        }
    }
}

} // namespace Fast
