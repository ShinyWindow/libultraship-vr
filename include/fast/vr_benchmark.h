#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Parses the optional --vr-benchmark command-line arguments before the renderer is initialized.
// Unknown arguments are ignored so the application's existing command line remains compatible.
void vr_benchmark_configure(int argc, char* argv[]);
int vr_benchmark_is_enabled(void);
void vr_benchmark_remove_arguments(int* argc, char* argv[]);
void vr_benchmark_apply_preset(void);
void vr_benchmark_title_intro_begin(void);
void vr_benchmark_title_intro_end(void);

#ifdef __cplusplus
}
#endif
