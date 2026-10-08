#include "fast/Fast3dWindow.h"

#include "ship/Context.h"
#include "ship/config/Config.h"
#include "ship/controller/controldeck/ControlDeck.h"
#include "ship/config/ConsoleVariable.h"
#include "fast/interpreter.h"
#include "fast/backends/gfx_sdl.h"
#include "fast/backends/gfx_dxgi.h"
#include "fast/backends/gfx_opengl.h"
#include "fast/backends/gfx_metal.h"
#include "fast/backends/gfx_direct3d_common.h"
#include "fast/backends/gfx_direct3d11.h"
#include "fast/backends/gfx_window_manager_api.h"

#include "fast/Fast3dGui.h"
#include "fast/vr_openxr.h"
#include "libultraship/bridge/consolevariablebridge.h"

#include <algorithm>
#include <chrono>
#include <fstream>

namespace Fast {

extern void GfxSetInstance(std::shared_ptr<Interpreter> gfx);

Fast3dWindow::Fast3dWindow(std::shared_ptr<Ship::Gui> gui, std::shared_ptr<FastMouseStateManager> mouseStateManager)
    : Ship::Window(gui, mouseStateManager) {
    mWindowManagerApi = nullptr;
    mRenderingApi = nullptr;
    mInterpreter = std::make_shared<Interpreter>();
    GfxSetInstance(mInterpreter);

#ifdef _WIN32
    AddAvailableWindowBackend(WindowBackend::FAST3D_DXGI_DX11);
#endif
#ifdef __APPLE__
    if (Metal_IsSupported()) {
        AddAvailableWindowBackend(WindowBackend::FAST3D_SDL_METAL);
    }
#endif
    AddAvailableWindowBackend(WindowBackend::FAST3D_SDL_OPENGL);
}

Fast3dWindow::Fast3dWindow(std::shared_ptr<Ship::Gui> gui)
    : Fast3dWindow(gui, std::make_shared<FastMouseStateManager>()) {
}

Fast3dWindow::Fast3dWindow(std::vector<std::shared_ptr<Ship::GuiWindow>> guiWindows)
    : Fast3dWindow(std::make_shared<Fast3dGui>(guiWindows)) {
}

Fast3dWindow::Fast3dWindow() : Fast3dWindow(std::vector<std::shared_ptr<Ship::GuiWindow>>()) {
}

Fast3dWindow::~Fast3dWindow() {
    SPDLOG_DEBUG("destruct fast3dwindow");
    vr_shutdown();
    mInterpreter->Destroy();
    delete mRenderingApi;
    delete mWindowManagerApi;
}

void Fast3dWindow::Init() {
    bool gameMode = false;

#ifdef __linux__
    std::ifstream osReleaseFile("/etc/os-release");
    if (osReleaseFile.is_open()) {
        std::string line;
        while (std::getline(osReleaseFile, line)) {
            if (line.find("VARIANT_ID") != std::string::npos) {
                if (line.find("steamdeck") != std::string::npos) {
                    gameMode = std::getenv("XDG_CURRENT_DESKTOP") != nullptr &&
                               std::string(std::getenv("XDG_CURRENT_DESKTOP")) == "gamescope";
                }
                break;
            }
        }
    }
#elif defined(__ANDROID__) || defined(__IOS__)
    gameMode = true;
#endif

    bool isFullscreen;
    uint32_t width, height;
    int32_t posX, posY;

    isFullscreen =
        Ship::Context::GetRawInstance()->GetConfig()->GetBool("Window.Fullscreen.Enabled", false) || gameMode;
    posX = Ship::Context::GetRawInstance()->GetConfig()->GetInt("Window.PositionX", 100);
    posY = Ship::Context::GetRawInstance()->GetConfig()->GetInt("Window.PositionY", 100);

    if (isFullscreen) {
        width = Ship::Context::GetRawInstance()->GetConfig()->GetInt("Window.Fullscreen.Width", gameMode ? 1280 : 1920);
        height =
            Ship::Context::GetRawInstance()->GetConfig()->GetInt("Window.Fullscreen.Height", gameMode ? 800 : 1080);
    } else {
        width = Ship::Context::GetRawInstance()->GetConfig()->GetInt("Window.Width", 640);
        height = Ship::Context::GetRawInstance()->GetConfig()->GetInt("Window.Height", 480);
    }
    Ship::Context::GetRawInstance()->GetWindow()->SetFullscreenScancode(
        Ship::Context::GetRawInstance()->GetConfig()->GetInt("Shortcuts.Fullscreen", Ship::KbScancode::LUS_KB_F11));
    Ship::Context::GetRawInstance()->GetWindow()->SetMouseCaptureScancode(
        Ship::Context::GetRawInstance()->GetConfig()->GetInt("Shortcuts.MouseCapture", Ship::KbScancode::LUS_KB_F2));

    InitWindowManager();
#ifdef ENABLE_DX11
    // SOH [VR] OpenXR needs the D3D11 device on the adapter the runtime names. When VR starts at
    // launch, ask the runtime BEFORE the device exists (the probe's instance is reused by vr_init
    // below); if the headset isn't available yet, take the high-performance GPU, which is where a
    // headset hangs on hybrid laptops and iGPU-enabled desktops. VR off at launch: untouched.
    if (GetWindowBackend() == WindowBackend::FAST3D_DXGI_DX11 && CVarGetInteger("gVrEnabled", 1) &&
        vr_backend_supported(GetWindowBackend())) {
        auto* dx = static_cast<GfxRenderingAPIDX11*>(mRenderingApi);
        uint64_t luid = 0;
        if (vr_probe_required_adapter(GetWindowBackend(), &luid)) {
            dx->mPreferredAdapterLuid = luid;
        }
        dx->mPreferHighPerformance = true;
    }
#endif
    mGfxDebugger = std::make_shared<GfxDebugger>();
    mInterpreter->SetGfxDebugger(mGfxDebugger);
    mInterpreter->Init(mWindowManagerApi, mRenderingApi, Ship::Context::GetRawInstance()->GetName().c_str(),
                       isFullscreen, width, height, posX, posY);
    mWindowManagerApi->SetFullscreenChangedCallback(OnFullscreenChanged);
    mWindowManagerApi->SetKeyboardCallbacks(KeyDown, KeyUp, AllKeysUp);
    mWindowManagerApi->SetMouseCallbacks(MouseButtonDown, MouseButtonUp);

    SetTextureFilter((FilteringMode)Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger(
        CVAR_TEXTURE_FILTER, FILTER_THREE_POINT));

    // SOH [VR] Must run after Interpreter::Init — the OpenXR session binds the renderer's device.
    // With VR mode off, skip entirely: no OpenXR session is created (and no SteamVR launch) until
    // the player toggles VR on (vr_apply_mode_request lazily initializes). On a renderer without
    // VR support, skip too: vr_apply_mode_request points the saved config at one that has it.
    if ((CVarGetInteger("gVrEnabled", 1) || !vr_can_disable()) && vr_backend_supported(GetWindowBackend())) {
        vr_init();
    }
}

int32_t Fast3dWindow::GetTargetFps() {
    return mInterpreter->GetTargetFps();
}

void Fast3dWindow::SetTargetFps(int32_t fps) {
    mInterpreter->SetTargetFps(fps);
}

void Fast3dWindow::SetMaximumFrameLatency(int32_t latency) {
    mInterpreter->SetMaxFrameLatency(latency);
}

void Fast3dWindow::GetPixelDepthPrepare(float x, float y) {
    mInterpreter->GetPixelDepthPrepare(x, y);
}

uint16_t Fast3dWindow::GetPixelDepth(float x, float y) {
    return mInterpreter->GetPixelDepth(x, y);
}

void Fast3dWindow::InitWindowManager() {
    SetWindowBackend(GetSavedWindowBackend());

    switch (GetWindowBackend()) {
#ifdef ENABLE_DX11
        case WindowBackend::FAST3D_DXGI_DX11:
            mWindowManagerApi = new GfxWindowBackendDXGI();
            mRenderingApi = new GfxRenderingAPIDX11(static_cast<GfxWindowBackendDXGI*>(mWindowManagerApi));
            break;
#endif
#ifdef ENABLE_OPENGL
        case WindowBackend::FAST3D_SDL_OPENGL:
            mRenderingApi = new GfxRenderingAPIOGL();
            mWindowManagerApi = new GfxWindowBackendSDL2();
            break;
#endif
#ifdef __APPLE__
        case WindowBackend::FAST3D_SDL_METAL:
            mRenderingApi = new GfxRenderingAPIMetal();
            mWindowManagerApi = new GfxWindowBackendSDL2();
            break;
#endif
        default:
            SPDLOG_ERROR("Could not load the correct rendering backend");
            break;
    }
}

void Fast3dWindow::SetTextureFilter(FilteringMode filteringMode) {
    mInterpreter->GetCurrentRenderingAPI()->SetTextureFilter(filteringMode);
}

void Fast3dWindow::EnableSRGBMode() {
    mInterpreter->mRapi->SetSrgbMode();
}

void Fast3dWindow::SetRendererUCode(UcodeHandlers ucode) {
    gfx_set_target_ucode(ucode);
}

void Fast3dWindow::Close() {
    mWindowManagerApi->Close();
}

void Fast3dWindow::RunGuiOnly() {
    mInterpreter->RunGuiOnly();
}

void Fast3dWindow::StartFrame() {
    mInterpreter->StartFrame();
}

void Fast3dWindow::EndFrame() {
    mInterpreter->EndFrame();
}

bool Fast3dWindow::IsFrameReady() {
    return mWindowManagerApi->IsFrameReady();
}

bool Fast3dWindow::DrawAndRunGraphicsCommands(Gfx* commands, const std::unordered_map<Mtx*, MtxF>& mtxReplacements) {
    std::shared_ptr<Window> wnd = Ship::Context::GetRawInstance()->GetWindow();

    const bool vr = vr_is_initialized();

    // Skip dropped frames.
    // SOH [VR] Not in VR: there, xrWaitFrame is the frame pacer, and this limiter is a second one
    // running off the DESKTOP swapchain's statistics. Worse, a "drop" here returns before any XR
    // call, so the whole frame — wait, begin and end — is skipped and the compositor is left to
    // reproject a stale frame. Let OpenXR decide the cadence.
    if (!vr && !wnd->IsFrameReady()) {
        return false;
    }

    const auto frameStart = std::chrono::steady_clock::now();
    auto elapsedMsSince = [](std::chrono::steady_clock::time_point since) {
        return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - since).count();
    };

    // SOH [VR] Decide up front which of this frame's expensive jobs actually run. vr_begin_frame
    // latches its submit poses from this, so it has to be set before the frame opens.
    bool renderEyes = true;
    bool renderHud = true;
    bool presentDesktop = true;
    if (vr) {
        // Standalone headsets are CPU-bound on the render (October 6 Quest log): redraw every other
        // frame there unless the player chose otherwise. Same default in the menu slider.
#ifdef __ANDROID__
        constexpr int kStereoDivisorDefault = 2;
#else
        constexpr int kStereoDivisorDefault = 1;
#endif
        const uint64_t stereoDivisor =
            (uint64_t)std::clamp(CVarGetInteger("gVrStereoDivisor", kStereoDivisorDefault), 1, 4);
        const uint64_t desktopDivisor = (uint64_t)std::clamp(CVarGetInteger("gVrDesktopViewDivisor", 4), 1, 32);

        // Redraw the stereo pair every Nth XR frame; the frames in between resubmit the previous
        // images and let the compositor reproject them onto the live head pose. Head tracking
        // stays at full rate, only world animation drops to refresh/N.
        renderEyes = (mVrFrameCounter % stereoDivisor) == 0u;

        // Entering or leaving a 2D context swaps which target holds the visible image, so force a
        // redraw on the transition rather than showing a stale panel (or a stale world behind it)
        // for up to a divisor's worth of frames.
        const bool flatScreen = vr_get_flat_screen();
        if (flatScreen != mVrFlatScreenPrev) {
            mVrFlatScreenPrev = flatScreen;
            renderEyes = true;
        }

        // The overlay display list is rebuilt once per 20 Hz game tick, so redrawing it on every
        // interpolated sub-frame renders byte-identical content up to six times at 120 Hz.
        // mInterpolationIndex == 0 marks a tick's first sub-frame (set in soh's RunCommands).
        renderHud = !CVarGetInteger("gVrHudPerTick", 1) || mInterpreter->mInterpolationIndex == 0;

        // The companion window is a courtesy view. Presenting it every XR frame costs an ImGui
        // frame, a full-eye-resolution mirror blit and a desktop Present, all on the critical path.
        presentDesktop = (mVrFrameCounter % desktopDivisor) == 0u;
        // The SoH menu panel's ImGui frame runs every XR frame, and on PC that frame is also the
        // desktop's (Fast3dGuiVrMenu.cpp), so the desktop presents every frame while it is open.
        if (vr_menu_panel_active()) {
            presentDesktop = true;
        }
#ifdef __ANDROID__
        // Standalone headset: no desktop window exists, so no mirror copy, ImGui frame or swap.
        presentDesktop = false;
#endif

        mVrFrameCounter++;
        vr_set_frame_plan(renderEyes, renderHud, presentDesktop);
    }

    auto gui = wnd->GetGui();
    // Setup mouse state manager
    wnd->GetMouseStateManager()->StartFrame();
    // SOH [VR] An ImGui frame runs for the desktop, or for the headset's menu panel (standalone
    // headsets: only for the panel). Latched here: the panel may open or close inside this frame.
    const bool runGui = presentDesktop || (vr && vr_menu_panel_active());
    // Setup of the backend frames and draw initial Window and GUI menus
    if (runGui) {
        gui->StartDraw();
    }
    // Setup game framebuffers to match available window space
    mInterpreter->StartFrame();

    float eyesMs = 0.0f;
    float hudMs = 0.0f;
    float desktopMs = 0.0f;

    // SOH [VR] Stereo path: run the interpreter once per eye into the OpenXR swapchains (or once
    // onto the flat-screen panel for 2D contexts), render the HUD quad, submit the XR frame, then
    // let the GUI composite the desktop companion view from the VR mirror texture.
    if (vr) {
        // The port sets mInterpolationT per sub-frame (see soh RunCommands); mirror it into the VR
        // layer so the camera anchor interpolates in lockstep with the interpolated world.
        vr_set_interp_alpha(mInterpreter->mInterpolationT);
        if (vr_begin_frame()) {
            // The divisor covers the flat-screen panel as well as the stereo pair: it is a quad
            // layer at a fixed pose showing a menu, so the compositor resubmits it perfectly and
            // there is even less to lose than with the eyes.
            if (renderEyes) {
                const auto eyesStart = std::chrono::steady_clock::now();
                if (vr_get_flat_screen()) {
                    // 2D context (file select, pause): whole frame onto the world-locked panel; the
                    // eye swapchains keep their last world frame, resubmitted with its original pose.
                    vr_begin_screen();
                    mInterpreter->Run(commands, mtxReplacements);
                    vr_end_screen();
                } else if (vr_multiview_enabled()) {
                    // Single-pass stereo: one run fills both layers of the eye swapchain.
                    vr_begin_stereo();
                    mInterpreter->Run(commands, mtxReplacements);
                    vr_end_stereo();
                } else {
                    for (int eye = 0; eye < 2; eye++) {
                        vr_begin_eye(eye);
                        mInterpreter->Run(commands, mtxReplacements);
                        vr_end_eye(eye);
                    }
                }
                eyesMs = elapsedMsSince(eyesStart);
            }

            // Head-locked HUD overlay (rendered once, not per-eye)
            Gfx* hudCommands = static_cast<Gfx*>(vr_get_hud_commands());
            if (hudCommands != nullptr && renderHud) {
                const auto hudStart = std::chrono::steady_clock::now();
                vr_begin_hud();
                mInterpreter->Run(hudCommands, mtxReplacements);
                vr_end_hud();
                hudMs = elapsedMsSince(hudStart);
            }

            // Text panel (message system on its own soft-follow quad), same once-per-tick cadence.
            Gfx* textCommands = static_cast<Gfx*>(vr_get_text_commands());
            if (textCommands != nullptr && renderHud) {
                const auto textStart = std::chrono::steady_clock::now();
                vr_begin_text();
                mInterpreter->Run(textCommands, mtxReplacements);
                vr_end_text();
                hudMs += elapsedMsSince(textStart);
            }

            vr_end_frame();
        }

        if (presentDesktop) {
            // Return to the WINDOW backbuffer for the GUI composite. In VR no Run pass resolves
            // into framebuffer 0 (the eyes render into XR swapchains and skip the non-VR exit path
            // that binds + clears it), so without this ImGui draws onto whatever XR target was
            // bound last — already released to the runtime — and the desktop window stays blank.
            mRenderingApi->StartDrawToFramebuffer(0, 1);
            mRenderingApi->ClearFramebuffer(true, true);

            // Desktop companion window: the GUI's game-image composite reads mGfxFrameBuffer; point
            // it at the VR mirror (left eye copy) since the game never rendered into mGameFb.
            mInterpreter->mGfxFrameBuffer = (uintptr_t)vr_get_mirror_texture_id();
        }
    } else {
        // Execute the games gfx commands
        mInterpreter->Run(commands, mtxReplacements);
    }

    if (presentDesktop) {
        const auto desktopStart = std::chrono::steady_clock::now();
        // Renders the game frame buffer to the final window and finishes the GUI
        gui->EndDraw();
        // Finalize swap buffers
        mInterpreter->EndFrame();
        desktopMs = elapsedMsSince(desktopStart);
    } else {
        // Companion window skipped this frame. Still kick the queued GPU work — the XR compositor
        // is the consumer now — but leave the desktop swapchain alone.
        if (runGui) {
            gui->EndDraw(); // the menu panel's frame (no desktop to draw it to)
        }
        mRenderingApi->EndFrame();
    }

    if (vr) {
        vr_report_frame_times(eyesMs, hudMs, desktopMs, elapsedMsSince(frameStart), renderEyes);
    }

    return true;
}

void Fast3dWindow::HandleEvents() {
    mWindowManagerApi->HandleEvents();
}

void Fast3dWindow::SetCursorVisibility(bool visible) {
    mWindowManagerApi->SetCursorVisibility(visible);
}

uint32_t Fast3dWindow::GetWidth() {
    uint32_t width, height;
    int32_t posX, posY;
    mWindowManagerApi->GetDimensions(&width, &height, &posX, &posY);
    return width;
}

uint32_t Fast3dWindow::GetHeight() {
    uint32_t width, height;
    int32_t posX, posY;
    mWindowManagerApi->GetDimensions(&width, &height, &posX, &posY);
    return height;
}

float Fast3dWindow::GetAspectRatio() {
    return mInterpreter->mCurDimensions.aspect_ratio;
}

int32_t Fast3dWindow::GetPosX() {
    uint32_t width, height;
    int32_t posX, posY;
    mWindowManagerApi->GetDimensions(&width, &height, &posX, &posY);
    return posX;
}

int32_t Fast3dWindow::GetPosY() {
    uint32_t width, height;
    int32_t posX, posY;
    mWindowManagerApi->GetDimensions(&width, &height, &posX, &posY);
    return posY;
}

void Fast3dWindow::SetMousePos(Ship::Coords pos) {
    mWindowManagerApi->SetMousePos(pos.x, pos.y);
}

Ship::Coords Fast3dWindow::GetMousePos() {
    int32_t x, y;
    mWindowManagerApi->GetMousePos(&x, &y);
    return { x, y };
}

Ship::Coords Fast3dWindow::GetMouseDelta() {
    int32_t x, y;
    mWindowManagerApi->GetMouseDelta(&x, &y);
    return { x, y };
}

Ship::CoordsF Fast3dWindow::GetMouseWheel() {
    float x, y;
    mWindowManagerApi->GetMouseWheel(&x, &y);
    return { x, y };
}

bool Fast3dWindow::GetMouseState(Ship::MouseBtn btn) {
    return mWindowManagerApi->GetMouseState(static_cast<uint32_t>(btn));
}

void Fast3dWindow::SetMouseCapture(bool capture) {
    mWindowManagerApi->SetMouseCapture(capture);
}

bool Fast3dWindow::IsMouseCaptured() {
    return mWindowManagerApi->IsMouseCaptured();
}

uint32_t Fast3dWindow::GetCurrentRefreshRate() {
    uint32_t refreshRate;
    mWindowManagerApi->GetActiveWindowRefreshRate(&refreshRate);
    return refreshRate;
}

bool Fast3dWindow::SupportsWindowedFullscreen() {
#ifdef __APPLE__
    return false;
#endif

    if (GetWindowBackend() == WindowBackend::FAST3D_SDL_OPENGL) {
        return true;
    }

    return false;
}

bool Fast3dWindow::CanDisableVerticalSync() {
    return mWindowManagerApi->CanDisableVsync();
}

void Fast3dWindow::SetResolutionMultiplier(float multiplier) {
    mInterpreter->SetResolutionMultiplier(multiplier);
}

void Fast3dWindow::SetMsaaLevel(uint32_t value) {
    mInterpreter->SetMsaaLevel(value);
}

void Fast3dWindow::SetFullscreen(bool isFullscreen) {
    // Save current window position before fullscreening
    SaveWindowToConfig();
    mWindowManagerApi->SetFullscreen(isFullscreen);
}

bool Fast3dWindow::IsFullscreen() {
    return mWindowManagerApi->IsFullscreen();
}

bool Fast3dWindow::IsRunning() {
    return mWindowManagerApi->IsRunning();
}

uintptr_t Fast3dWindow::GetGfxFrameBuffer() {
    return mInterpreter->mGfxFrameBuffer;
}

const char* Fast3dWindow::GetKeyName(int32_t scancode) {
    return mWindowManagerApi->GetKeyName(scancode);
}

bool Fast3dWindow::KeyUp(int32_t scancode) {
    if (scancode == Ship::Context::GetRawInstance()->GetWindow()->GetFullscreenScancode()) {
        Ship::Context::GetRawInstance()->GetWindow()->ToggleFullscreen();
    }

    if (scancode == Ship::Context::GetRawInstance()->GetWindow()->GetMouseCaptureScancode()) {
        Ship::Context::GetRawInstance()->GetWindow()->GetMouseStateManager()->ToggleMouseCaptureOverride();
    }

    Ship::Context::GetRawInstance()->GetWindow()->SetLastScancode(-1);
    return Ship::Context::GetRawInstance()->GetControlDeck()->ProcessKeyboardEvent(
        Ship::KbEventType::LUS_KB_EVENT_KEY_UP, static_cast<Ship::KbScancode>(scancode));
}

bool Fast3dWindow::KeyDown(int32_t scancode) {
    bool isProcessed = Ship::Context::GetRawInstance()->GetControlDeck()->ProcessKeyboardEvent(
        Ship::KbEventType::LUS_KB_EVENT_KEY_DOWN, static_cast<Ship::KbScancode>(scancode));
    Ship::Context::GetRawInstance()->GetWindow()->SetLastScancode(scancode);

    return isProcessed;
}

void Fast3dWindow::AllKeysUp() {
    Ship::Context::GetRawInstance()->GetControlDeck()->ProcessKeyboardEvent(Ship::KbEventType::LUS_KB_EVENT_ALL_KEYS_UP,
                                                                            Ship::KbScancode::LUS_KB_UNKNOWN);
}

bool Fast3dWindow::MouseButtonUp(int button) {
    return Ship::Context::GetRawInstance()->GetControlDeck()->ProcessMouseButtonEvent(
        false, static_cast<Ship::MouseBtn>(button));
}

bool Fast3dWindow::MouseButtonDown(int button) {
    bool isProcessed = Ship::Context::GetRawInstance()->GetControlDeck()->ProcessMouseButtonEvent(
        true, static_cast<Ship::MouseBtn>(button));
    return isProcessed;
}

void Fast3dWindow::OnFullscreenChanged(bool isNowFullscreen) {
    std::shared_ptr<Window> wnd = Ship::Context::GetRawInstance()->GetWindow();

    // Re-save fullscreen enabled after
    Ship::Context::GetRawInstance()->GetConfig()->SetBool("Window.Fullscreen.Enabled", isNowFullscreen);
}

std::weak_ptr<Interpreter> Fast3dWindow::GetInterpreterWeak() const {
    return mInterpreter;
}

std::string Fast3dWindow::GetWindowBackendName() {
    switch (GetWindowBackend()) {
        case WindowBackend::FAST3D_DXGI_DX11:
            return "DirectX 11";
        case WindowBackend::FAST3D_SDL_OPENGL:
            return "OpenGL";
        case WindowBackend::FAST3D_SDL_METAL:
            return "Metal";
        default:
            return "";
    }
}

void Fast3dWindow::SetCurrentDimensions(uint32_t width, uint32_t height) {
    SetCurrentDimensions(width, height, GetPosX(), GetPosY());
}

void Fast3dWindow::SetCurrentDimensions(uint32_t width, uint32_t height, int32_t posX, int32_t posY) {
    mWindowManagerApi->SetDimensions(width, height, posX, posY);
    SaveWindowToConfig();
}

void Fast3dWindow::SetCurrentDimensions(bool isFullscreen, uint32_t width, uint32_t height) {
    SetCurrentDimensions(isFullscreen, width, height, GetPosX(), GetPosY());
}

void Fast3dWindow::SetCurrentDimensions(bool isFullscreen, uint32_t width, uint32_t height, int32_t posX,
                                        int32_t posY) {
    auto config = Ship::Context::GetRawInstance()->GetConfig();
    if (!isFullscreen) {
        config->SetInt("Window.Width", static_cast<int32_t>(width));
        config->SetInt("Window.Height", static_cast<int32_t>(height));
        config->SetInt("Window.PositionX", posX);
        config->SetInt("Window.PositionY", posY);
    } else {
        config->SetInt("Window.Fullscreen.Width", static_cast<int32_t>(width));
        config->SetInt("Window.Fullscreen.Height", static_cast<int32_t>(height));
    }
    mWindowManagerApi->SetFullscreen(isFullscreen);
    mWindowManagerApi->SetDimensions(width, height, posX, posY);
    SaveWindowToConfig();
}

Ship::WindowRect Fast3dWindow::GetPrimaryMonitorRect() {
    return mWindowManagerApi->GetPrimaryMonitorRect();
}

std::shared_ptr<GfxDebugger> Fast3dWindow::GetGfxDebugger() const {
    return mGfxDebugger;
}

} // namespace Fast
