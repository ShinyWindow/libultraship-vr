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
#include "fast/vr_benchmark.h"
#include "libultraship/bridge/consolevariablebridge.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

namespace {

struct VrBenchmarkSample {
    double elapsedSeconds;
    float waitMs;
    float eyesMs;
    float hudMs;
    float desktopMs;
    float totalMs;
    float gpuEyesMs;
    float gpuHudMs;
    float gpuDesktopMs;
    float gpuTotalMs;
    bool renderedEyes;
    uint32_t runtimePredictedRefreshRate;
};

struct VrBenchmarkState {
    bool enabled = false;
    bool measuring = false;
    bool finished = false;
    double warmupSeconds = 0.0;
    double durationSeconds = 120.0;
    std::string name = "benchmark";
    int preset = -1; // -1 unchanged, 0 baseline, 1 optimized
    bool titleIntroActive = false;
    bool titleIntroEnded = false;
    std::chrono::steady_clock::time_point vrReadyTime;
    std::chrono::steady_clock::time_point measureStart;
    std::chrono::steady_clock::time_point titleIntroDetectedTime;
    std::vector<VrBenchmarkSample> samples;
};

VrBenchmarkState gVrBenchmark;

std::string SanitizeBenchmarkName(std::string value) {
    for (char& c : value) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) {
            c = '_';
        }
    }
    return value.empty() ? "benchmark" : value;
}

double ParsePositiveSeconds(const char* value, double fallback) {
    if (value == nullptr) {
        return fallback;
    }
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    return end != value && *end == '\0' && std::isfinite(parsed) && parsed >= 0.0 ? parsed : fallback;
}

double Percentile(std::vector<float> values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = percentile * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * fraction;
}

double Average(const std::vector<float>& values) {
    return values.empty() ? 0.0
                          : std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

std::string TimestampForFile() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::ostringstream result;
    result << std::put_time(&local, "%Y%m%d-%H%M%S");
    return result.str();
}

void FinishVrBenchmark() {
    auto& benchmark = gVrBenchmark;
    benchmark.finished = true;

    std::filesystem::create_directories("benchmarks");
    const std::string stem = "benchmarks/" + TimestampForFile() + "_" + benchmark.name;
    std::ofstream csv(stem + ".csv");
    csv << "frame,elapsed_seconds,rendered_eyes,runtime_predicted_refresh_rate_hz,xr_wait_ms,active_cpu_ms,"
           "eyes_ms,hud_ms,desktop_ms,total_frame_ms,gpu_eyes_ms,gpu_hud_ms,gpu_desktop_ms,gpu_total_ms\n";

    std::vector<float> totalTimes;
    std::vector<float> activeTimes;
    std::vector<float> waitTimes;
    std::vector<float> eyesTimes;
    std::vector<float> gpuEyesTimes;
    std::vector<float> gpuHudTimes;
    std::vector<float> gpuDesktopTimes;
    std::vector<float> gpuTotalTimes;
    std::vector<float> frameIntervals;
    totalTimes.reserve(benchmark.samples.size());
    activeTimes.reserve(benchmark.samples.size());
    waitTimes.reserve(benchmark.samples.size());
    eyesTimes.reserve(benchmark.samples.size());
    size_t renderedFrames = 0;
    for (size_t i = 0; i < benchmark.samples.size(); ++i) {
        const auto& sample = benchmark.samples[i];
        const float activeMs = std::max(0.0f, sample.totalMs - sample.waitMs);
        csv << i << ',' << std::fixed << std::setprecision(6) << sample.elapsedSeconds << ','
            << (sample.renderedEyes ? 1 : 0) << ',' << sample.runtimePredictedRefreshRate << ',' << sample.waitMs << ','
            << activeMs << ',' << sample.eyesMs << ',' << sample.hudMs << ','
            << sample.desktopMs << ',' << sample.totalMs << ',' << sample.gpuEyesMs << ',' << sample.gpuHudMs << ','
            << sample.gpuDesktopMs << ',' << sample.gpuTotalMs << '\n';
        totalTimes.push_back(sample.totalMs);
        activeTimes.push_back(activeMs);
        waitTimes.push_back(sample.waitMs);
        if (sample.gpuTotalMs > 0.0f) {
            gpuEyesTimes.push_back(sample.gpuEyesMs);
            gpuHudTimes.push_back(sample.gpuHudMs);
            gpuDesktopTimes.push_back(sample.gpuDesktopMs);
            gpuTotalTimes.push_back(sample.gpuTotalMs);
        }
        if (i > 0) {
            frameIntervals.push_back(static_cast<float>((sample.elapsedSeconds - benchmark.samples[i - 1].elapsedSeconds) *
                                                        1000.0));
        }
        if (sample.renderedEyes) {
            renderedFrames++;
            eyesTimes.push_back(sample.eyesMs);
        }
    }

    const auto cvarInt = [](const char* name, int fallback) {
        return Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger(name, fallback);
    };
    const auto cvarFloat = [](const char* name, float fallback) {
        return Ship::Context::GetRawInstance()->GetConsoleVariables()->GetFloat(name, fallback);
    };
    const double measuredDuration = benchmark.samples.empty() ? 0.0 : benchmark.samples.back().elapsedSeconds;
    const double submittedFps = measuredDuration > 0.0 && benchmark.samples.size() > 1
                                    ? static_cast<double>(benchmark.samples.size() - 1) / measuredDuration
                                    : 0.0;
    const double renderedEyeFps = measuredDuration > 0.0 ? static_cast<double>(renderedFrames) / measuredDuration : 0.0;
    const double frameIntervalP99 = Percentile(frameIntervals, 0.99);
    const double onePercentLowFps = frameIntervalP99 > 0.0 ? 1000.0 / frameIntervalP99 : 0.0;
    const int requestedRefreshRate = cvarInt("gVrRefreshRate", 0);
    const uint32_t runtimePredictedRefreshRate = vr_get_refresh_rate();
    const uint32_t cadenceTargetRate = requestedRefreshRate > 0 ? static_cast<uint32_t>(requestedRefreshRate)
                                                                : runtimePredictedRefreshRate;
    size_t targetRateFrames = 0;
    size_t halfRateFrames = 0;
    size_t refreshRateTransitions = 0;
    uint32_t previousRuntimeRate = 0;
    for (const auto& sample : benchmark.samples) {
        if (sample.runtimePredictedRefreshRate == cadenceTargetRate) {
            targetRateFrames++;
        }
        if (cadenceTargetRate > 0 && sample.runtimePredictedRefreshRate * 2 == cadenceTargetRate) {
            halfRateFrames++;
        }
        if (previousRuntimeRate != 0 && sample.runtimePredictedRefreshRate != previousRuntimeRate) {
            refreshRateTransitions++;
        }
        previousRuntimeRate = sample.runtimePredictedRefreshRate;
    }
    const double targetRatePercent = benchmark.samples.empty()
                                         ? 0.0
                                         : targetRateFrames * 100.0 / static_cast<double>(benchmark.samples.size());
    const bool cadenceStable = targetRatePercent >= 95.0;
    const double expectedRefreshIntervals = measuredDuration * static_cast<double>(cadenceTargetRate);
    const uint64_t estimatedMissedIntervals = expectedRefreshIntervals > benchmark.samples.size()
                                                  ? static_cast<uint64_t>(std::llround(expectedRefreshIntervals -
                                                                                       benchmark.samples.size()))
                                                  : 0;
    uint32_t eyeWidth = 0;
    uint32_t eyeHeight = 0;
    vr_get_recommended_resolution(&eyeWidth, &eyeHeight);

    std::ofstream json(stem + "-summary.json");
    json << std::fixed << std::setprecision(3);
    json << "{\n"
         << "  \"benchmark\": \"Hyrule Field title introduction\",\n"
         << "  \"name\": \"" << benchmark.name << "\",\n"
         << "  \"duration_seconds\": " << measuredDuration << ",\n"
         << "  \"submitted_frames\": " << benchmark.samples.size() << ",\n"
         << "  \"rendered_eye_frames\": " << renderedFrames << ",\n"
         << "  \"eye_width\": " << eyeWidth << ",\n"
         << "  \"eye_height\": " << eyeHeight << ",\n"
         << "  \"requested_refresh_rate_hz\": " << requestedRefreshRate << ",\n"
         << "  \"runtime_predicted_refresh_rate_hz\": " << runtimePredictedRefreshRate << ",\n"
         << "  \"target_rate_frame_percent\": " << targetRatePercent << ",\n"
         << "  \"half_rate_reprojection_frames\": " << halfRateFrames << ",\n"
         << "  \"runtime_rate_transitions\": " << refreshRateTransitions << ",\n"
         << "  \"cadence_valid\": " << (cadenceStable ? "true" : "false") << ",\n"
         << "  \"average_submitted_fps\": " << submittedFps << ",\n"
         << "  \"average_rendered_eye_fps\": " << renderedEyeFps << ",\n"
         << "  \"refresh_rate_attainment_percent\": "
         << (cadenceTargetRate > 0 ? submittedFps * 100.0 / cadenceTargetRate : 0.0) << ",\n"
         << "  \"estimated_missed_refresh_intervals\": " << estimatedMissedIntervals << ",\n"
         << "  \"one_percent_low_submitted_fps\": " << onePercentLowFps << ",\n"
         << "  \"frame_interval_average_ms\": " << Average(frameIntervals) << ",\n"
         << "  \"frame_interval_median_ms\": " << Percentile(frameIntervals, 0.50) << ",\n"
         << "  \"frame_interval_p99_ms\": " << frameIntervalP99 << ",\n"
         << "  \"frame_interval_max_ms\": " << Percentile(frameIntervals, 1.0) << ",\n"
         << "  \"resolution_scale\": " << cvarFloat("gVrResolutionScale", 1.0f) << ",\n"
         << "  \"stereo_divisor\": " << cvarInt("gVrStereoDivisor", 1) << ",\n"
         << "  \"single_pass_stereo\": " << cvarInt("gVrMultiview", 0) << ",\n"
         << "  \"hidden_area_mask\": " << cvarInt("gVrHiddenAreaMask", 0) << ",\n"
         << "  \"d3d11_state_cache\": " << cvarInt("gVrD3D11StateCache", 1) << ",\n"
         << "  \"xr_wait_average_ms\": " << Average(waitTimes) << ",\n"
         << "  \"xr_wait_p99_ms\": " << Percentile(waitTimes, 0.99) << ",\n"
         << "  \"cpu_active_average_ms\": " << Average(activeTimes) << ",\n"
         << "  \"cpu_active_p99_ms\": " << Percentile(activeTimes, 0.99) << ",\n"
         << "  \"gpu_eyes_average_ms\": " << Average(gpuEyesTimes) << ",\n"
         << "  \"gpu_eyes_p99_ms\": " << Percentile(gpuEyesTimes, 0.99) << ",\n"
         << "  \"gpu_hud_average_ms\": " << Average(gpuHudTimes) << ",\n"
         << "  \"gpu_desktop_average_ms\": " << Average(gpuDesktopTimes) << ",\n"
         << "  \"gpu_total_average_ms\": " << Average(gpuTotalTimes) << ",\n"
         << "  \"gpu_total_p99_ms\": " << Percentile(gpuTotalTimes, 0.99) << ",\n"
         << "  \"cpu_total_average_ms\": " << Average(totalTimes) << ",\n"
         << "  \"cpu_total_median_ms\": " << Percentile(totalTimes, 0.50) << ",\n"
         << "  \"cpu_total_p99_ms\": " << Percentile(totalTimes, 0.99) << ",\n"
         << "  \"cpu_total_max_ms\": " << Percentile(totalTimes, 1.0) << ",\n"
         << "  \"cpu_eyes_average_ms\": " << Average(eyesTimes) << ",\n"
         << "  \"cpu_eyes_p99_ms\": " << Percentile(eyesTimes, 0.99) << "\n"
         << "}\n";

    // Benchmark mode is a standalone process. Close the result files before
    // terminating so the game cannot advance another actor frame while its
    // renderer/window is already shutting down.
    csv.close();
    json.close();

    std::ostringstream summary;
    summary << std::fixed << std::setprecision(3)
            << "\nVR BENCHMARK COMPLETE\n"
            << "Output: " << stem << "\n"
            << "Frames: " << benchmark.samples.size() << " submitted, " << renderedFrames << " eye renders\n"
            << "Cadence: " << submittedFps << " submitted FPS, " << renderedEyeFps << " rendered-eye FPS, "
            << onePercentLowFps << " FPS 1% low\n"
            << "Runtime target-rate frames: " << targetRatePercent << "% ("
            << (cadenceStable ? "VALID" : "INVALID - reprojection cadence changed") << ")\n"
            << "CPU total: " << Average(totalTimes) << " ms average, " << Percentile(totalTimes, 0.99)
            << " ms P99\n"
            << "CPU active: " << Average(activeTimes) << " ms average, " << Percentile(activeTimes, 0.99)
            << " ms P99 (xrWaitFrame excluded)\n"
            << "XR wait: " << Average(waitTimes) << " ms average\n"
            << "GPU: " << Average(gpuEyesTimes) << " ms eyes, " << Average(gpuTotalTimes) << " ms total average\n"
            << "CPU eyes: " << Average(eyesTimes) << " ms average, " << Percentile(eyesTimes, 0.99)
            << " ms P99\n";
    std::cout << summary.str() << std::flush;
    SPDLOG_INFO("{}", summary.str());
    std::error_code removeError;
    std::filesystem::remove("benchmarks/benchmark-active.txt", removeError);

    std::quick_exit(EXIT_SUCCESS);
}

} // namespace

extern "C" void vr_benchmark_configure(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i] != nullptr ? argv[i] : "";
        if (argument == "--vr-benchmark") {
            gVrBenchmark.enabled = true;
        } else if (argument == "--benchmark-name" && i + 1 < argc) {
            gVrBenchmark.name = SanitizeBenchmarkName(argv[++i]);
        } else if (argument == "--benchmark-warmup" && i + 1 < argc) {
            gVrBenchmark.warmupSeconds = ParsePositiveSeconds(argv[++i], gVrBenchmark.warmupSeconds);
        } else if (argument == "--benchmark-duration" && i + 1 < argc) {
            gVrBenchmark.durationSeconds = ParsePositiveSeconds(argv[++i], gVrBenchmark.durationSeconds);
        }
    }
    if (gVrBenchmark.enabled) {
        if (gVrBenchmark.name == "baseline") {
            gVrBenchmark.preset = 0;
        } else if (gVrBenchmark.name == "optimized") {
            gVrBenchmark.preset = 1;
        }
        std::filesystem::create_directories("benchmarks");
        std::ofstream marker("benchmarks/benchmark-active.txt");
        marker << "VR benchmark command line recognized.\n"
               << "Name: " << gVrBenchmark.name << "\n"
               << "Warm-up: " << gVrBenchmark.warmupSeconds << " seconds\n"
               << "Duration: " << gVrBenchmark.durationSeconds << " seconds\n"
               << "Waiting for OpenXR initialization.\n";
        std::cout << "\nVR BENCHMARK MODE ENABLED\n"
                  << "Name: " << gVrBenchmark.name << "\n"
                  << "Warm-up: " << gVrBenchmark.warmupSeconds << " seconds\n"
                  << "Measurement: " << gVrBenchmark.durationSeconds << " seconds\n"
                  << "Waiting for OpenXR initialization...\n\n"
                  << std::flush;
    }
}

extern "C" void vr_benchmark_apply_preset(void) {
    if (!gVrBenchmark.enabled || gVrBenchmark.preset < 0) {
        return;
    }
    const int enabled = gVrBenchmark.preset;
    CVarSetInteger("gVrMultiview", enabled);
    CVarSetInteger("gVrHiddenAreaMask", enabled);
    CVarSetInteger("gVrD3D11StateCache", enabled);
    CVarSetInteger("gVrStereoDivisor", 1);
    CVarSetInteger("gVrDesktopViewDivisor", 16);
    CVarSetInteger("gVrHudPerTick", 1);
    CVarSetInteger("gVrStayOnDoff", 1);
    std::cout << "Benchmark preset applied: " << (enabled ? "optimized" : "baseline") << '\n'
              << "  Single-Pass Stereo: " << (enabled ? "ON" : "OFF") << '\n'
              << "  Hidden-Area Mask: " << (enabled ? "ON" : "OFF") << '\n'
              << "  D3D11 State Cache: " << (enabled ? "ON" : "OFF") << "\n\n"
              << "  Stereo Render Divisor: 1\n"
              << "  Companion Window Divisor: 16\n"
              << "  Draw HUD Once Per Game Tick: ON\n\n"
              << "  Stay in VR When Headset Is Removed: ON\n\n"
              << std::flush;
}

extern "C" int vr_benchmark_is_enabled(void) {
    return gVrBenchmark.enabled ? 1 : 0;
}

extern "C" void vr_benchmark_remove_arguments(int* argc, char* argv[]) {
    if (argc == nullptr || argv == nullptr || *argc <= 1) {
        return;
    }
    int writeIndex = 1;
    for (int readIndex = 1; readIndex < *argc; ++readIndex) {
        const std::string argument = argv[readIndex] != nullptr ? argv[readIndex] : "";
        const bool optionWithValue = argument == "--benchmark-name" || argument == "--benchmark-warmup" ||
                                     argument == "--benchmark-duration";
        if (argument == "--vr-benchmark") {
            continue;
        }
        if (optionWithValue) {
            if (readIndex + 1 < *argc) {
                ++readIndex;
            }
            continue;
        }
        argv[writeIndex++] = argv[readIndex];
    }
    *argc = writeIndex;
    argv[writeIndex] = nullptr;
}

extern "C" void vr_benchmark_title_intro_begin(void) {
    if (!gVrBenchmark.enabled || gVrBenchmark.finished) {
        return;
    }
    gVrBenchmark.titleIntroActive = true;
    gVrBenchmark.titleIntroEnded = false;
    gVrBenchmark.titleIntroDetectedTime = std::chrono::steady_clock::now();
    gVrBenchmark.measuring = false;
    gVrBenchmark.vrReadyTime = {};
    gVrBenchmark.measureStart = {};
    gVrBenchmark.samples.clear();
    std::cout << "Hyrule Field title introduction detected; waiting for submitted OpenXR frame.\n" << std::flush;
}

extern "C" void vr_benchmark_title_intro_end(void) {
    if (gVrBenchmark.enabled && gVrBenchmark.titleIntroActive && !gVrBenchmark.finished) {
        gVrBenchmark.titleIntroEnded = true;
    }
}

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
    vr_shutdown_final();
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
    bool updateDesktopMirror = true;
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

        // Keep the desktop swapchain advancing every XR frame. Skipping its entire finalisation
        // path caused severe frame/audio pacing on DXGI; opening ImGui appeared to fix it because
        // the interactive menu forced this normal path back on. The divisor now throttles only the
        // expensive full-eye mirror copy. Between updates the desktop presents the previous copy.
        updateDesktopMirror = (mVrFrameCounter % desktopDivisor) == 0u;
        presentDesktop = true;
        // Keep the interactive menu's mirror current while it is being adjusted.
        if (vr_menu_interactive()) {
            updateDesktopMirror = true;
        }
#ifdef __ANDROID__
        // Standalone headset: no desktop window exists, so no mirror copy, ImGui frame or swap.
        presentDesktop = false;
        updateDesktopMirror = false;
#endif

        mVrFrameCounter++;
        vr_set_frame_plan(renderEyes, renderHud, updateDesktopMirror);
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
    bool submittedXrFrame = false;
    bool gpuTimingFrame = false;

    // SOH [VR] Stereo path: run the interpreter once per eye into the OpenXR swapchains (or once
    // onto the flat-screen panel for 2D contexts), render the HUD quad, submit the XR frame, then
    // let the GUI composite the desktop companion view from the VR mirror texture.
    if (vr) {
        // The port sets mInterpolationT per sub-frame (see soh RunCommands); mirror it into the VR
        // layer so the camera anchor interpolates in lockstep with the interpolated world.
        vr_set_interp_alpha(mInterpreter->mInterpolationT);
        if (vr_begin_frame()) {
            if (gVrBenchmark.enabled) {
                vr_gpu_timing_frame_begin();
                gpuTimingFrame = true;
            }
            // The divisor covers the flat-screen panel as well as the stereo pair: it is a quad
            // layer at a fixed pose showing a menu, so the compositor resubmits it perfectly and
            // there is even less to lose than with the eyes.
            if (renderEyes) {
                Gfx* keyHudCommands = static_cast<Gfx*>(vr_get_hud_commands());
                if (gpuTimingFrame) vr_gpu_timing_phase_begin(0);
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
                    if (keyHudCommands != nullptr && vr_begin_key_world_pass()) {
                        mInterpreter->Run(keyHudCommands, mtxReplacements);
                        vr_end_key_world_pass();
                    }
                    vr_end_stereo();
                } else {
                    for (int eye = 0; eye < 2; eye++) {
                        vr_begin_eye(eye);
                        mInterpreter->Run(commands, mtxReplacements);
                        if (keyHudCommands != nullptr && vr_begin_key_world_pass()) {
                            mInterpreter->Run(keyHudCommands, mtxReplacements);
                            vr_end_key_world_pass();
                        }
                        vr_end_eye(eye);
                    }
                }
                eyesMs = elapsedMsSince(eyesStart);
                if (gpuTimingFrame) vr_gpu_timing_phase_end(0);
            }

            // Head-locked HUD overlay (rendered once, not per-eye)
            Gfx* hudCommands = static_cast<Gfx*>(vr_get_hud_commands());
            if (gpuTimingFrame && renderHud) vr_gpu_timing_phase_begin(1);
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
            if (gpuTimingFrame && renderHud) vr_gpu_timing_phase_end(1);

            vr_end_frame();
            submittedXrFrame = true;
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
        if (gpuTimingFrame) vr_gpu_timing_phase_begin(2);
        const auto desktopStart = std::chrono::steady_clock::now();
        // Renders the game frame buffer to the final window and finishes the GUI
        gui->EndDraw();
        // Finalize swap buffers
        mInterpreter->EndFrame();
        desktopMs = elapsedMsSince(desktopStart);
        if (gpuTimingFrame) vr_gpu_timing_phase_end(2);
    } else {
        // Companion window skipped this frame. Still kick the queued GPU work — the XR compositor
        // is the consumer now — but leave the desktop swapchain alone.
        if (runGui) {
            gui->EndDraw(); // the menu panel's frame (no desktop to draw it to)
        }
        mRenderingApi->EndFrame();
    }

    if (vr) {
        if (gpuTimingFrame) vr_gpu_timing_frame_end();
        const float totalMs = elapsedMsSince(frameStart);
        vr_report_frame_times(eyesMs, hudMs, desktopMs, totalMs, renderEyes);

        if (gVrBenchmark.enabled && !gVrBenchmark.finished && gVrBenchmark.titleIntroActive &&
            !submittedXrFrame) {
            const double frameWaitSeconds = std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() -
                                                gVrBenchmark.titleIntroDetectedTime)
                                                .count();
            if (frameWaitSeconds >= 20.0) {
                std::cout << "VR benchmark aborted: OpenXR submitted no frames within 20 seconds.\n"
                          << std::flush;
                gVrBenchmark.finished = true;
                std::error_code removeError;
                std::filesystem::remove("benchmarks/benchmark-active.txt", removeError);
                Close();
                return true;
            }
            if (gVrBenchmark.vrReadyTime.time_since_epoch().count() != 0) {
                std::cout << "VR benchmark paused because OpenXR stopped submitting frames. Restarting warm-up.\n";
            }
            gVrBenchmark.vrReadyTime = {};
            gVrBenchmark.measureStart = {};
            gVrBenchmark.measuring = false;
            gVrBenchmark.samples.clear();
        } else if (gVrBenchmark.enabled && !gVrBenchmark.finished && gVrBenchmark.titleIntroActive &&
                   submittedXrFrame) {
            const auto now = std::chrono::steady_clock::now();
            if (gVrBenchmark.vrReadyTime.time_since_epoch().count() == 0) {
                gVrBenchmark.vrReadyTime = now;
                std::cout << "VR benchmark warm-up started (" << gVrBenchmark.warmupSeconds << " seconds)\n";
            }
            const double readySeconds =
                std::chrono::duration<double>(now - gVrBenchmark.vrReadyTime).count();
            if (!gVrBenchmark.measuring && readySeconds >= gVrBenchmark.warmupSeconds) {
                gVrBenchmark.measuring = true;
                gVrBenchmark.measureStart = now;
                gVrBenchmark.samples.clear();
                gVrBenchmark.samples.reserve(static_cast<size_t>(gVrBenchmark.durationSeconds * 200.0));
                std::cout << "VR benchmark measurement started; recording the complete Hyrule Field introduction.\n";
            }
            if (gVrBenchmark.measuring) {
                const double elapsed = std::chrono::duration<double>(now - gVrBenchmark.measureStart).count();
                float gpuEyesMs = 0.0f, gpuHudMs = 0.0f, gpuDesktopMs = 0.0f, gpuTotalMs = 0.0f;
                vr_gpu_timing_get_latest(&gpuEyesMs, &gpuHudMs, &gpuDesktopMs, &gpuTotalMs);
                gVrBenchmark.samples.push_back({ elapsed, vr_get_last_wait_ms(), eyesMs, hudMs, desktopMs, totalMs,
                                                 gpuEyesMs, gpuHudMs, gpuDesktopMs, gpuTotalMs, renderEyes,
                                                 vr_get_refresh_rate() });
                if (gVrBenchmark.titleIntroEnded || elapsed >= gVrBenchmark.durationSeconds) {
                    FinishVrBenchmark();
                }
            }
        }
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
