// RadBruter adapter for independently linked upstream RAD engines, September 2026.
#include "runtime.h"
#include "platform.h"
#include "hash.h"
#include "compiler.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include "qrad.h"
#undef min
#undef max

namespace {
using Bytes = std::vector<unsigned char>;
struct RawFace {
    std::array<std::vector<std::array<float, 6>>, 4> samples;
    float floor = 0;
};
std::vector<RawFace> raw_faces;
std::vector<dface_t> final_faces;
Bytes final_light;
std::vector<std::array<int32_t, 2>> patch_records;
std::vector<CompilerBoundary> boundaries;
std::string structure;
lm_stats stats{};
int phase = -1;
using Clock = std::chrono::steady_clock;
Clock::time_point previous;
double previous_cpu = 0;

void write_all(int fd, const void *data, size_t size) {
    auto p = static_cast<const char *>(data);
    while (size) {
        auto n = lm::write(fd, p, size);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("Compiler transport write failed");
        p += n;
        size -= n;
    }
}
template <class T> void write_value(int fd, const T &value) {
    write_all(fd, &value, sizeof value);
}
using Hash = lm::Hash;
} // namespace
extern "C" void lm_mark(int next) {
    auto now = Clock::now();
    double cpu = lm::cpu_seconds();
    if (phase >= 0) {
        stats.seconds[phase] += std::chrono::duration<double>(now - previous).count();
        stats.cpu_seconds[phase] += cpu - previous_cpu;
    }
    phase = next;
    previous = now;
    previous_cpu = cpu;
}
void lm_native_loaded() {
    raw_faces.resize(g_numfaces);
}
void lm_native_structure() {
    Hash hash;
    hash.add(g_num_patches);
    patch_records.resize(g_num_patches);
    std::vector<bool> seen(g_numfaces, false);
    for (unsigned i = 0; i < g_num_patches; ++i) {
        const auto &p = g_patches[i];
        hash.add(p.faceNumber);
        hash.add(p.origin);
        hash.add(p.area);
        hash.add(p.scale);
        hash.add(p.chop);
        hash.add(p.flags);
        hash.add(p.samples);
        hash.add(p.winding->m_NumPoints);
        hash.add(p.winding->m_Points, p.winding->m_NumPoints * sizeof(vec3_t));
#if defined(HLRAD_TEXLIGHTTHRESHOLD_FIX) || defined(LM_SDHLT)
        hash.add(p.emitmode);
#else
        bool casting = VectorAvg(p.baselight) >= g_dlight_threshold;
        hash.add(casting);
#endif
        hash.add(p.emitstyle);
        hash.add(p.totalstyle);
        patch_records[i] = {p.faceNumber, 0};
        if (!seen[p.faceNumber]) {
            seen[p.faceNumber] = true;
            CompilerBoundary boundary;
            boundary.face = p.faceNumber;
#if defined(LM_VHLT) || defined(LM_SDHLT)
            // light_surface overrides RAD emission on this face.
            if (g_face_texlights[p.faceNumber])
                continue;
            for (int c = 0; c < 3; ++c)
                boundary.weight[c] = p.texturereflectivity[c] / 3;
#endif
            boundaries.push_back(boundary);
        }
    }
    structure = hash.finish();
}
void lm_native_sample(int face, int slot, int sample, int count, const float *raw, float floor) {
    auto &data = raw_faces.at(face);
    data.floor = floor;
    auto &values = data.samples.at(slot);
    if (!sample)
        values.resize(count);
    for (int c = 0; c < 3; ++c)
        values.at(sample)[3 + c] = raw[c];
}
void lm_native_capture() {
    final_faces.assign(g_dfaces, g_dfaces + g_numfaces);
    final_light.assign(g_dlightdata, g_dlightdata + g_lightdatasize);
}
void lm_native_finish() {
    lm_mark(-1);
    if (final_faces.empty())
        lm_native_capture();
    CompilerEncoding encoding;
    for (int c = 0; c < 3; ++c) {
        encoding.scale[c] = g_colour_lightscale[c];
        encoding.gamma[c] = g_colour_qgamma[c];
    }
    encoding.direct_scale = g_direct_scale;
    encoding.casting_threshold = g_dlight_threshold;
    encoding.bounces = g_numbounce;
#if defined(LM_ZHLT)
    encoding.preclip = g_maxlight;
    encoding.postclip = -1;
#else
    encoding.preclip = -1;
    encoding.postclip = g_limitthreshold;
    encoding.minimum = g_minlight;
    encoding.rounding = .5;
#endif
    int fd = atoi(getenv("LM_RESULT_FD"));
    write_value(fd, uint32_t(0x4c4d523a));
    write_value(fd, stats);
    write_all(fd, structure.data(), 64);
    uint32_t counts[3] = {uint32_t(final_faces.size()), uint32_t(final_light.size()),
                          uint32_t(patch_records.size())};
    write_all(fd, counts, sizeof counts);
    write_all(fd, final_faces.data(), final_faces.size() * sizeof(dface_t));
    write_all(fd, final_light.data(), final_light.size());
    write_all(fd, patch_records.data(), patch_records.size() * sizeof(patch_records[0]));
    write_value(fd, encoding);
    for (const auto &face : raw_faces)
        write_value(fd, face.floor);
    write_value(fd, uint32_t(boundaries.size()));
    write_all(fd, boundaries.data(), boundaries.size() * sizeof(CompilerBoundary));
    fd = atoi(getenv("LM_FLOAT_FD"));
    for (size_t face = 0; face < raw_faces.size(); ++face)
        for (int slot = 0; slot < 4; ++slot) {
            const auto &samples = raw_faces[face].samples[slot];
            if (samples.empty())
                continue;
            int32_t header[3] = {int32_t(face), final_faces[face].styles[slot], int32_t(samples.size())};
            write_all(fd, header, sizeof header);
            write_all(fd, samples.data(), samples.size() * sizeof(samples[0]));
        }
}
int native_rad_main(int argc, char **argv);
extern "C" int lm_original_grid(int face, int *mins, int *size) {
    try { return lm::original_grid(face, g_numfaces, mins, size); }
    catch (const std::exception &e) { Error("%s", e.what()); }
    return 0;
}
int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--version")) {
        printf("%s %s\n", LM_COMPILER_NAME, LM_COMPILER_COMMIT);
        return 0;
    }
    try {
        lm::initialize_process();
        if (!getenv("LM_RESULT_FD") || !getenv("LM_FLOAT_FD") || !getenv("LM_BSP_INPUT"))
            throw std::runtime_error("Start this compiler through radbruter");
        bool explicit_rad = false;
        for (int i = 1; i + 1 < argc; ++i)
            if (!strcmp(argv[i], "-lights")) {
                auto path = std::filesystem::weakly_canonical(argv[i + 1]);
                auto relative = path.lexically_relative(std::filesystem::current_path());
                if (relative.empty() || *relative.begin() == "..")
                    throw std::runtime_error("RAD files must stay inside this project");
                explicit_rad = true;
            }
        if (!explicit_rad)
            throw std::runtime_error("An explicit generated RAD is required");
        lm_mark(LM_LOAD);
        return native_rad_main(argc, argv);
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
