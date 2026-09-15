#pragma once
#include <cstdint>

// Shared, fixed-width metadata for the separately linked upstream RAD workers.
struct CompilerEncoding {
    double scale[3] = {1.1, 1.1, 1.1};
    double gamma[3] = {.5, .5, .5};
    double preclip = 255, postclip = 255, minimum = 0, rounding = 0;
    double direct_scale = 2, casting_threshold = 25;
    uint32_t bounces = 1, reserved = 0;
};
struct CompilerBoundary {
    int32_t face = 0;
    float weight[3] = {1.0f / 3, 1.0f / 3, 1.0f / 3};
};

#ifdef LM_NATIVE_WORKER
void lm_native_loaded();
void lm_native_structure();
void lm_native_sample(int face, int slot, int sample, int count, const float *raw, float floor);
void lm_native_capture();
void lm_native_finish();
#endif
