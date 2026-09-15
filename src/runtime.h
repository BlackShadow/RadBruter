#pragma once
#include <stddef.h>
#include <stdint.h>
#include "gpu_types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum lm_phase {
    LM_LOAD,
    LM_GEOMETRY,
    LM_DIRECT,
    LM_TRANSFERS,
    LM_BOUNCE,
    LM_ENCODE,
    LM_WRITE,
    LM_PHASE_COUNT
};
typedef struct lm_stats {
    double seconds[LM_PHASE_COUNT];
    double cpu_seconds[LM_PHASE_COUNT];
    uint64_t geometry_hits, transfer_hits, transfer_misses;
    uint64_t cache_read_bytes, cache_write_bytes, gpu_calls, gpu_bytes;
    uint64_t direct_hits, direct_misses;
    double gpu_seconds;
    uint64_t gpu_direct_calls, gpu_direct_pairs, gpu_direct_uncertain;
    uint64_t direct_overflows, cache_lock_contentions;
    uint64_t gpu_trace_calls, gpu_trace_rays, gpu_trace_uncertain;
    uint64_t indirect_hits, indirect_misses;
    uint64_t gpu_indirect_calls, gpu_indirect_samples, gpu_indirect_uncertain;
    uint64_t gpu_scene_uploads, gpu_transfer_uploads, gpu_plan_uploads;
    uint64_t gpu_transfer_reuses, cache_verified_bytes;
} lm_stats;

void lm_mark(int phase);
void lm_initialize(void);
int lm_original_grid(int face, int *mins, int *size);
extern int lm_reference_mode;
extern int lm_audit_gpu_direct;
typedef struct lm_direct_hit {
    uint32_t light;
    union { float ratio; uint32_t sky; };
} lm_direct_hit;
#define LM_SKY_DIRECT 0x80000000u
#define LM_SKY_DIFFUSE 0x40000000u
#define LM_LIGHT_INDEX 0x3fffffffu
void lm_build_facelights(void);
int lm_direct_lookup(int face, int point, const lm_direct_hit **hits, uint32_t *count);
void lm_direct_record(int face, uint32_t light, float ratio);
void lm_direct_record_sky(int face, uint32_t light, const float *value, int diffuse);
const float *lm_direct_sky(int face, uint32_t index);
int lm_direct_recording(void);
int lm_direct_prepass(void);
void lm_add_direct_ray(int face, unsigned int light, const float *position, const float *origin);
void lm_collect_direct(int face, int point, const float *position, const float *normal);
int lm_direct_visibility(int face, int point, unsigned int light, const float *position, const float *normal);
int lm_enqueue_patch_ray(int node, const float *start, const float *stop, unsigned int bit);
void lm_flush_patch_rays(void);
int lm_sample_membership(int face, int sample, int included);
int lm_emission_class(const float *emission);
int lm_geometry_load(void *data, size_t size);
void lm_geometry_save(const void *data, size_t size);
int lm_transfers_load(void);
void lm_transfers_save(void);
int lm_gpu_gather(void);
void lm_build_indirect(void);
int lm_indirect_ready(void);
int lm_indirect_sample(int face, int sample, float *result);
int lm_face_sample_count(int face);
const float *lm_face_sample_position(int face, int sample);
void lm_prepare_indirect_plan(LMIndirectSample *plans, const uint32_t *offsets);
void lm_evaluate_indirect(const LMIndirectSample *plan, float *result);
void lm_finish(void);
int qrad_main(int argc, char **argv);

#ifdef __cplusplus
}
#include <filesystem>
#include <string>
#include <vector>
namespace lm {
std::string sha256(const void *data, size_t size);
int gpu_service(const std::filesystem::path &socket);
void gpu_self_test();
void cache_self_test(const std::filesystem::path &directory);
} // namespace lm
#endif
