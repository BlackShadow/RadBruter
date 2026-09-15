#include "runtime.h"
#include "platform.h"
#include "checked_cast.h"
#include "hash.h"
extern "C" {
#include "qrad.h"
extern float maxchop, minchop;
extern qboolean texscale;
extern vec3_t emitlight[MAX_PATCHES], addlight[MAX_PATCHES];
extern void GatherLight(int thread);
extern LMTraceNode *tnodes;
extern byte *vismatrix;
extern int TestLine_r(int, vec3_t, vec3_t);
}
#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <thread>
#include "gpu_api.h"
#include "kernels.h"

namespace fs = std::filesystem;
extern "C" {
int lm_reference_mode = 0;
int lm_audit_gpu_direct = 0;
}
namespace {
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<unsigned char>;
using Hash = lm::Hash;
struct FD {
    int value = -1;
    explicit FD(int value = -1) : value(value) {}
    ~FD() {
        if (value >= 0)
            lm::close_file(value);
    }
    FD(const FD &) = delete;
    FD &operator=(const FD &) = delete;
};
void write_all(int fd, const void *data, size_t size) {
    auto p = static_cast<const char *>(data);
    while (size) {
        int64_t n = lm::write(fd, p, size);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("IPC/write failed");
        p += n;
        size -= n;
    }
}
void read_all(int fd, void *data, size_t size) {
    auto p = static_cast<char *>(data);
    while (size) {
        int64_t n = lm::read(fd, p, size);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("IPC/read failed");
        p += n;
        size -= n;
    }
}
void append(Bytes &out, const void *data, size_t bytes) {
    if (!bytes)
        return;
    const auto *p = static_cast<const unsigned char *>(data);
    out.insert(out.end(), p, p + bytes);
}
template <class T> void append(Bytes &out, const T &value) {
    append(out, &value, sizeof value);
}
struct Cursor {
    std::span<const unsigned char> data;
    size_t pos = 0;
    template <class T> explicit Cursor(const T &value) : data(value.data(), value.size()) {}
    const void *take(size_t size) {
        if (size > data.size() - pos)
            throw std::runtime_error("Truncated cache");
        const void *p = data.data() + pos;
        pos += size;
        return p;
    }
    template <class T> T get() {
        T v;
        memcpy(&v, take(sizeof v), sizeof v);
        return v;
    }
};
struct CachedBytes {
    void *mapping = nullptr;
    size_t length = 0;
    ~CachedBytes() { reset(); }
    void reset() {
        if (mapping != nullptr) lm::unmap(mapping, length);
        mapping = nullptr;
        length = 0;
    }
    const unsigned char *data() const {
        return static_cast<const unsigned char *>(mapping) + 64;
    }
    size_t size() const { return length ? length - 64 : 0; }
    CachedBytes() = default;
    CachedBytes(const CachedBytes &) = delete;
    CachedBytes &operator=(const CachedBytes &) = delete;
};
lm_stats stats{};
std::mutex gpu_stats_mutex;
Clock::time_point previous;
double previous_cpu = 0;
int phase = -1;
std::string geometry_key, world_key, transfer_key, source_structure_key;
fs::path cache_dir;
std::unique_ptr<FD> geometry_lock, transfer_lock;
struct DirectFace {
    std::vector<uint32_t> offsets;
    std::vector<lm_direct_hit> hits;
    std::vector<std::array<float, 3>> sky;
    std::span<const uint32_t> loaded_offsets;
    std::span<const lm_direct_hit> loaded_hits;
    std::span<const std::array<float, 3>> loaded_sky;
};
std::vector<DirectFace> direct_faces;
CachedBytes direct_payload, transfer_payload;
std::span<const uint32_t> transfer_offsets, transfer_table;
std::vector<Bytes> sample_membership;
std::vector<Bytes> frozen_membership;
std::atomic<bool> direct_collecting{false};
std::atomic<size_t> direct_bytes{0};
bool direct_loaded = false;
std::string direct_key;
std::unique_ptr<FD> direct_lock;
constexpr size_t direct_budget = 512ULL << 20;
bool direct_overflow = false;
struct DirectSample {
    float position[3], normal[3];
};
struct DirectVisibilityFace {
    std::vector<DirectSample> samples;
    std::vector<uint32_t> offsets;
    std::vector<LMTraceRay> rays;
    Bytes visibility;
    int lookup_point = -1;
    size_t cursor = 0;
};
std::vector<DirectVisibilityFace> direct_visibility_faces;
std::vector<uint32_t> indirect_offsets;
std::vector<LMIndirectResult> indirect_results;
thread_local bool gpu_prepass = false, gpu_visibility_ready = false;
void build_gpu_visibility(int face);

bool valid_indirect_plan(const LMIndirectSample &p, uint64_t patch_count) {
    if (p.kind > 3 || !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.x2) ||
        !std::isfinite(p.y1))
        return false;
    unsigned used = p.kind == 2 ? 3 : p.kind == 3 ? 2 : p.kind;
    for (unsigned i = 0; i < used; ++i)
        if (p.patch[i] >= patch_count)
            return false;
    return true;
}

std::string nonlighting_key() {
    Hash hash;
    auto lump = [&hash](const auto *data, int count) {
        hash.add(count);
        hash.add(data, size_t(count) * sizeof *data);
    };
    lump(dmodels, nummodels);
    lump(dplanes, numplanes);
    lump(dvertexes, numvertexes);
    lump(dnodes, numnodes);
    lump(texinfo, numtexinfo);
    hash.add(numfaces);
    for (int i = 0; i < numfaces; ++i)
        hash.add(&dfaces[i], offsetof(dface_t, styles));
    lump(dclipnodes, numclipnodes);
    lump(dleafs, numleafs);
    lump(dmarksurfaces, nummarksurfaces);
    lump(dedges, numedges);
    lump(dsurfedges, numsurfedges);
    lump(dvisdata, visdatasize);
    lump(dtexdata, texdatasize);
    lump(dentdata, entdatasize);
    return hash.finish();
}

std::unique_ptr<FD> lock_cache(const fs::path &file) {
    auto fd =
        std::make_unique<FD>(lm::open_file((file.string() + ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600));
    if (fd->value < 0)
        throw std::runtime_error("Cannot lock lighting cache");
    if (!lm::try_lock(fd->value)) {
        ++stats.cache_lock_contentions;
        return nullptr;
    }
    return fd;
}
bool load_cache(const fs::path &file, CachedBytes &payload) {
    payload.reset();
    FD fd(lm::open_file(file.c_str(), O_RDONLY | O_CLOEXEC));
    lm::FileInfo info{};
    if (fd.value < 0 || lm::file_info(fd.value, info) || !lm::regular_file(info) ||
        info.st_size < 64 || uint64_t(info.st_size) > (2ULL << 30)) return false;
    payload.length = info.st_size;
    payload.mapping = lm::map_readonly(fd.value, payload.length);
    if (payload.mapping == nullptr) { payload.length = 0; return false; }
    // Cache entries are immutable and atomically published in a private run directory.
    // Verify their checksum once per inode/version, then share read-only mapped pages.
    bool reusable_stamp = false;
    auto stamp = lm::file_stamp(fd.value, info, &reusable_stamp);
    std::array<uint64_t, 7> checked{};
    // Immutable verification records avoid replacing a file another worker is reading.
    auto marker_path = file.parent_path() / ("verified-" + lm::sha256(stamp.data(), sizeof stamp));
    FD marker(reusable_stamp ? lm::open_file(marker_path, O_RDONLY | O_CLOEXEC) : -1);
    lm::FileInfo marker_info{};
    bool verified = marker.value >= 0 && !lm::file_info(marker.value, marker_info) &&
                    marker_info.st_size == int64_t(sizeof checked) &&
                    lm::read_at(marker.value, checked.data(), sizeof checked, 0) == int64_t(sizeof checked) &&
                    checked == stamp;
    stats.cache_read_bytes += payload.length;
    if (!verified) {
        stats.cache_verified_bytes += payload.size();
        if (lm::sha256(payload.data(), payload.size()) !=
            std::string(static_cast<const char *>(payload.mapping), 64)) {
            payload.reset();
            return false;
        }
        FD record(reusable_stamp ? lm::open_file(marker_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC) : -1);
        // A missing or incomplete hint only causes the next reader to hash again.
        if (record.value >= 0) lm::write(record.value, stamp.data(), sizeof stamp);
    }
    return true;
}
void save_cache(const fs::path &file, const Bytes &payload) {
    fs::path temporary = file.string() + "." + std::to_string(lm::process_id()) + ".tmp";
    try {
        FD fd(lm::open_file(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
        if (fd.value < 0)
            throw std::runtime_error("Cannot create lighting cache");
        auto digest = lm::sha256(payload.data(), payload.size());
        write_all(fd.value, digest.data(), digest.size());
        write_all(fd.value, payload.data(), payload.size());
        lm::atomic_replace(temporary, file);
        stats.cache_write_bytes += payload.size() + 64;
    } catch (...) {
        fs::remove(temporary);
        throw;
    }
}
std::string patch_key(bool structural = false) {
    Hash hash;
    hash.add(world_key.data(), world_key.size());
    const uint32_t version = 5;
    hash.add(version);
    hash.add(num_patches);
    if (structural) {
        hash.add(maxchop);
        hash.add(minchop);
        hash.add(texscale);
    }
    for (unsigned i = 0; i < num_patches; ++i) {
        const auto &p = patches[i];
        hash.add(p.origin);
        hash.add(p.normal);
        hash.add(p.area);
        hash.add(p.sky);
        hash.add(p.faceNumber);
        hash.add(p.mins);
        hash.add(p.maxs);
        hash.add(p.plane->dist);
        hash.add(face_offset[p.faceNumber]);
        hash.add(p.winding->numpoints);
        hash.add(p.winding->p, size_t(p.winding->numpoints) * sizeof(vec3_t));
        if (structural) {
            hash.add(p.chop);
            hash.add(p.scale);
            hash.add(p.samples);
            hash.add(lm_emission_class(p.baselight));
        }
    }
    if (structural)
        for (const auto &face : sample_membership) {
            hash.add(face.size());
            hash.add(face.data(), face.size());
        }
    return hash.finish();
}
std::string sampling_key() {
    Hash hash;
    auto patches = patch_key();
    hash.add(patches.data(), patches.size());
    hash.add(extra);
    hash.add(smoothing_threshold);
    return hash.finish();
}

void prepare_direct_cache() {
    direct_faces.clear();
    direct_payload.reset();
    direct_faces.resize(numfaces);
    direct_loaded = false;
    direct_collecting = false;
    direct_bytes = 0;
    direct_overflow = false;
    if (cache_dir.empty())
        return;
    Hash hash;
    hash.add(world_key.data(), world_key.size());
    hash.add(uint32_t(2));
    hash.add(extra);
    hash.add(smoothing_threshold);
    hash.add(indirect_sun);
    uint32_t count = 0;
    for (int leaf = 1; leaf < numleafs; ++leaf)
        for (auto light = directlights[leaf]; light; light = light->next) {
            if (light->type == emit_skylight)
                hash.add(light->intensity); // Entity sky colors are fixed within this entry.
            hash.add(leaf);
            hash.add(light->type);
            hash.add(light->style);
            hash.add(light->origin);
            hash.add(light->normal);
            hash.add(light->stopdot);
            hash.add(light->stopdot2);
            ++count;
        }
    direct_key = hash.finish();
    auto file = cache_dir / ("direct-" + direct_key);
    if (load_cache(file, direct_payload)) {
        try {
            Cursor cursor{direct_payload};
            if (cursor.get<uint32_t>() != uint32_t(numfaces))
                throw std::runtime_error("Face count changed");
            for (auto &face : direct_faces) {
                auto points = cursor.get<uint32_t>(), hits = cursor.get<uint32_t>(), sky = cursor.get<uint32_t>();
                if (!points || points > 65536 || hits > (1u << 26) || sky > points * 2)
                    throw std::runtime_error("Invalid direct cache size");
                face.loaded_offsets = {static_cast<const uint32_t *>(cursor.take(size_t(points) * 4)), points};
                face.loaded_hits = {static_cast<const lm_direct_hit *>(cursor.take(size_t(hits) * sizeof(lm_direct_hit))), hits};
                face.loaded_sky = {static_cast<const std::array<float, 3> *>(cursor.take(size_t(sky) * 12)), sky};
                if (face.loaded_offsets.front() != 0 || face.loaded_offsets.back() != hits ||
                    !std::is_sorted(face.loaded_offsets.begin(), face.loaded_offsets.end()))
                    throw std::runtime_error("Invalid direct cache offsets");
                for (auto hit : face.loaded_hits)
                    if ((hit.light & LM_LIGHT_INDEX) >= count ||
                        ((hit.light & ~LM_LIGHT_INDEX) ?
                         ((hit.light & ~LM_LIGHT_INDEX) == ~LM_LIGHT_INDEX || hit.sky >= sky) :
                         !std::isfinite(hit.ratio)))
                        throw std::runtime_error("Invalid cached light");
                for (const auto &value : face.loaded_sky)
                    for (float channel : value)
                        if (!std::isfinite(channel)) throw std::runtime_error("Invalid cached sky light");
            }
            if (cursor.pos != direct_payload.size())
                throw std::runtime_error("Trailing direct cache data");
            direct_loaded = true;
            ++stats.direct_hits;
            direct_lock.reset();
            return;
        } catch (const std::runtime_error &) {
            direct_faces.clear();
            direct_faces.resize(numfaces);
        }
    }
    ++stats.direct_misses;
    // Readers use immutable, atomically published cache files without an
    // exclusive lock. Only a producer needs the non-blocking publication lock.
    CachedBytes marker;
    if (load_cache(file.string() + ".oversize", marker) && marker.size() == sizeof(size_t)) {
        size_t previous_budget;
        memcpy(&previous_budget, marker.data(), sizeof previous_budget);
        if (previous_budget >= direct_budget)
            return;
    }
    direct_lock = lock_cache(file);
    direct_collecting = bool(direct_lock);
}

void abandon_direct_cache() {
    if (!direct_collecting.exchange(false))
        return;
    direct_overflow = true;
    ++stats.direct_overflows;
    Bytes marker;
    append(marker, direct_budget);
    save_cache(cache_dir / ("direct-" + direct_key + ".oversize"), marker);
    direct_lock.reset(); // Release immediately, not at the end of the CPU stage.
}

void finish_direct_cache() {
    if (direct_lock && direct_collecting) {
        Bytes payload;
        append(payload, uint32_t(numfaces));
        for (auto &face : direct_faces) {
            face.offsets.push_back(lm::checked_cast<uint32_t>(face.hits.size()));
            append(payload, uint32_t(face.offsets.size()));
            append(payload, uint32_t(face.hits.size()));
            append(payload, uint32_t(face.sky.size()));
            append(payload, face.offsets.data(), face.offsets.size() * 4);
            append(payload, face.hits.data(), face.hits.size() * sizeof(lm_direct_hit));
            append(payload, face.sky.data(), face.sky.size() * 12);
        }
        save_cache(cache_dir / ("direct-" + direct_key), payload);
    }
    direct_collecting = false;
    direct_loaded = false;
    direct_faces.clear();
    direct_payload.reset();
    direct_lock.reset();
}
struct GpuHeader {
    uint32_t magic = 0x4c4d4732, rows = 0;
    uint64_t entries = 0;
};
struct GpuReply {
    uint32_t status = 0, reserved = 0;
    double seconds = 0;
};
static_assert(sizeof(GpuHeader) == 16 && sizeof(lm_stats) == 336);
static_assert(sizeof(transfer_t) == 4 && sizeof(float) == 4 && sizeof(void *) == 8);
static_assert(sizeof(LMTraceNode) == 32 && sizeof(LMTraceRay) == 32);
static_assert(sizeof(LMIndirectSample) == 32 && sizeof(LMIndirectResult) == 16);
static_assert(std::endian::native == std::endian::little);

void cuda_check(CUresult code) {
    if (code != CUDA_SUCCESS) {
        const char *text = nullptr;
        lm::gpu_api().cuGetErrorString(code, &text);
        throw std::runtime_error(std::string("CUDA: ") + (text ? text : "unknown error"));
    }
}
void nvrtc_check(nvrtcResult code) {
    if (code != NVRTC_SUCCESS)
        throw std::runtime_error(std::string("NVRTC: ") + lm::gpu_api().nvrtcGetErrorString(code));
}
struct DeviceBuffer {
    // This host fails standalone CUDA byte-copy roundtrips. Mapped pinned
    // memory lets kernels access fresh working data without those copy calls.
    // Synchronize CPU access; retaining allocation capacity never reuses data.
    CUdeviceptr ptr = 0;
    void *host = nullptr;
    size_t capacity = 0;
    ~DeviceBuffer() {
        if (host)
            lm::gpu_api().cuMemFreeHost(host);
    }
    void reserve(size_t size) {
        if (size <= capacity)
            return;
        if (host) {
            cuda_check(lm::gpu_api().cuMemFreeHost(host));
            host = nullptr;
            ptr = 0;
            capacity = 0;
        }
        cuda_check(lm::gpu_api().cuMemHostAlloc(&host, size, CU_MEMHOSTALLOC_DEVICEMAP));
        cuda_check(lm::gpu_api().cuMemHostGetDevicePointer(&ptr, host, 0));
        capacity = size;
    }
    void upload(const void *data, size_t size, CUstream stream, CUdevice) {
        if (!size)
            return;
        cuda_check(lm::gpu_api().cuStreamSynchronize(stream));
        reserve(size);
        memcpy(host, data, size);
    }
    void download(void *data, size_t size, CUstream stream) {
        if (!size)
            return;
        cuda_check(lm::gpu_api().cuStreamSynchronize(stream));
        memcpy(data, host, size);
    }
    void initialize(size_t size, CUstream stream) {
        cuda_check(lm::gpu_api().cuStreamSynchronize(stream));
        reserve(size);
        memset(host, 255, size);
    }
};
class GPU {
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    std::string module_image;
    CUfunction gather = nullptr;
    CUfunction trace_kernel = nullptr;
    CUfunction interpolate_kernel = nullptr;
    CUstream stream = nullptr;
    DeviceBuffer emission, output;
    // Keep each stage's input/output lifetime separate.
    DeviceBuffer interpolation_light, interpolation_output;
    // One immutable transfer table may survive compatible cached requests.
    // Emission, ray and interpolation inputs remain fresh on every request.
    DeviceBuffer nodes, offsets, transfers, plans;
    DeviceBuffer ray_input, ray_output;
    std::string resident_transfer_key;
    uint32_t resident_rows = 0;
    size_t resident_entries = 0;

  public:
    GPU() {
        cuda_check(lm::gpu_api().cuInit(0));
        cuda_check(lm::gpu_api().cuDeviceGet(&device, 0));
        cuda_check(lm::gpu_api().cuDevicePrimaryCtxSetFlags(device, CU_CTX_SCHED_BLOCKING_SYNC));
        cuda_check(lm::gpu_api().cuDevicePrimaryCtxRetain(&context, device));
        cuda_check(lm::gpu_api().cuCtxSetCurrent(context));
        cuda_check(lm::gpu_api().cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 8192));
        cuda_check(lm::gpu_api().cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        int major, minor;
        cuda_check(lm::gpu_api().cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        cuda_check(lm::gpu_api().cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        std::string arch = "--gpu-architecture=compute_" + std::to_string(major * 10 + minor);
        nvrtcProgram program = nullptr;
        nvrtc_check(lm::gpu_api().nvrtcCreateProgram(&program, lm_cuda_source, "radbruter.cu", 0, nullptr, nullptr));
        const char *options[] = {arch.c_str(), "--std=c++17", "--fmad=false"};
        auto status = lm::gpu_api().nvrtcCompileProgram(program, 3, options);
        if (status != NVRTC_SUCCESS) {
            size_t size;
            lm::gpu_api().nvrtcGetProgramLogSize(program, &size);
            std::string log(size, '\0');
            lm::gpu_api().nvrtcGetProgramLog(program, log.data());
            lm::gpu_api().nvrtcDestroyProgram(&program);
            throw std::runtime_error("CUDA kernel compilation failed: " + log);
        }
        size_t size;
        nvrtc_check(lm::gpu_api().nvrtcGetPTXSize(program, &size));
        module_image.resize(size);
        nvrtc_check(lm::gpu_api().nvrtcGetPTX(program, module_image.data()));
        nvrtc_check(lm::gpu_api().nvrtcDestroyProgram(&program));
        cuda_check(lm::gpu_api().cuModuleLoadData(&module, module_image.data()));
        cuda_check(lm::gpu_api().cuModuleGetFunction(&gather, module, "gather_light"));
        cuda_check(lm::gpu_api().cuModuleGetFunction(&trace_kernel, module, "trace_rays"));
        cuda_check(lm::gpu_api().cuModuleGetFunction(&interpolate_kernel, module, "interpolate_light"));
        // Keep the module image alive and finish loading every kernel before
        // servicing requests, including tracing after gathering/interpolation.
        cuda_check(lm::gpu_api().cuFuncLoad(gather));
        cuda_check(lm::gpu_api().cuFuncLoad(trace_kernel));
        cuda_check(lm::gpu_api().cuFuncLoad(interpolate_kernel));
        cuda_check(lm::gpu_api().cuCtxSynchronize());
    }
    ~GPU() {
        if (stream)
            lm::gpu_api().cuStreamDestroy(stream);
        if (module)
            lm::gpu_api().cuModuleUnload(module);
        // The process owns the primary context until all DeviceBuffers destruct.
    }
    void interpolate(const std::vector<LMIndirectSample> &samples, const std::vector<float> &light,
                     std::vector<LMIndirectResult> &result) {
        unsigned count = lm::checked_cast<unsigned>(samples.size());
        result.resize(count);
        if (!count)
            return;
        plans.upload(samples.data(), samples.size() * sizeof(LMIndirectSample), stream, device);
        interpolation_light.upload(light.data(), light.size() * sizeof(float), stream, device);
        interpolation_output.initialize(result.size() * sizeof(LMIndirectResult), stream);
        void *args[] = {&plans.ptr, &count, &interpolation_light.ptr, &interpolation_output.ptr};
        cuda_check(lm::gpu_api().cuLaunchKernel(interpolate_kernel, (count + 255) / 256, 1, 1, 256, 1, 1, 0, stream,
                                  args, nullptr));
        interpolation_output.download(result.data(), result.size() * sizeof(LMIndirectResult), stream);
    }
    void trace(const std::vector<LMTraceNode> &scene, const std::vector<LMTraceRay> &rays, Bytes &result) {
        unsigned node_count = lm::checked_cast<unsigned>(scene.size()), count = lm::checked_cast<unsigned>(rays.size());
        result.resize(count);
        if (!count)
            return;
        nodes.upload(scene.data(), scene.size() * sizeof(LMTraceNode), stream, device);
        ray_input.upload(rays.data(), rays.size() * sizeof(LMTraceRay), stream, device);
        ray_output.initialize(result.size(), stream);
        void *args[] = {&nodes.ptr, &node_count, &ray_input.ptr, &count, &ray_output.ptr};
        cuda_check(
            lm::gpu_api().cuLaunchKernel(trace_kernel, (count + 127) / 128, 1, 1, 128, 1, 1, 0, stream, args, nullptr));
        ray_output.download(result.data(), result.size(), stream);
        auto invalid = std::find_if(result.begin(), result.end(), [](auto value) { return value > 2; });
        if (invalid != result.end())
            throw std::runtime_error("CUDA produced invalid visibility " + std::to_string(*invalid) +
                                     " at ray " + std::to_string(invalid - result.begin()) + " of " +
                                     std::to_string(count));
    }
    bool has_transfers(const std::string &key, uint32_t rows, size_t entries) const {
        return !key.empty() && key == resident_transfer_key && rows == resident_rows &&
               entries == resident_entries;
    }
    void calculate(std::span<const uint32_t> row_offsets, std::span<const uint32_t> weights,
                   const std::vector<float> &light, std::vector<float> &result,
                   const std::string &key = {}, bool reuse = false) {
        uint32_t rows = lm::checked_cast<uint32_t>(light.size() / 3);
        result.resize(light.size());
        if (!rows)
            return;
        if (reuse) {
            if (key.empty() || key != resident_transfer_key || rows != resident_rows)
                throw std::runtime_error("Invalid resident transfer request");
        } else {
            offsets.upload(row_offsets.data(), row_offsets.size() * 4, stream, device);
            transfers.upload(weights.data(), weights.size() * 4, stream, device);
            resident_transfer_key = key;
            resident_rows = rows;
            resident_entries = weights.size();
        }
        emission.upload(light.data(), light.size() * 4, stream, device);
        output.initialize(light.size() * 4, stream);
        void *arguments[] = {&offsets.ptr, &transfers.ptr, &emission.ptr, &output.ptr, &rows};
        cuda_check(
            lm::gpu_api().cuLaunchKernel(gather, (rows * 3 + 255) / 256, 1, 1, 256, 1, 1, 0, stream, arguments, nullptr));
        output.download(result.data(), result.size() * 4, stream);
        if (std::any_of(result.begin(), result.end(), [](float value) { return !std::isfinite(value); }))
            throw std::runtime_error("CUDA gather produced non-finite output");
    }
};
} // namespace

std::string lm::sha256(const void *data, size_t size) {
    Hash hash;
    hash.add(data, size);
    return hash.finish();
}
extern "C" int lm_original_grid(int face, int *mins, int *size) {
    try { return lm::original_grid(face, numfaces, mins, size); }
    catch (const std::exception &e) { Error("%s", e.what()); }
    return 0;
}
extern "C" void lm_mark(int next) {
    auto now = Clock::now();
    double cpu = lm::cpu_seconds();
    if (phase >= 0) {
        stats.seconds[phase] += std::chrono::duration<double>(now - previous).count();
        stats.cpu_seconds[phase] += cpu - previous_cpu;
    }
    previous = now;
    previous_cpu = cpu;
    phase = next;
}
extern "C" void lm_initialize() {
    try {
        lm_reference_mode = getenv("LM_REFERENCE") != nullptr;
        lm_audit_gpu_direct = getenv("LM_AUDIT_GPU_DIRECT") != nullptr;
        sample_membership.resize(numfaces);
        source_structure_key = nonlighting_key();
        Hash geo;
        geo.add(numnodes);
        geo.add(dnodes, size_t(numnodes) * sizeof *dnodes);
        geo.add(numplanes);
        geo.add(dplanes, size_t(numplanes) * sizeof *dplanes);
        geo.add(numleafs);
        geo.add(dleafs, size_t(numleafs) * sizeof *dleafs);
        geometry_key = geo.finish();
        Hash world;
        world.add(geometry_key.data(), geometry_key.size());
        world.add(dvertexes, size_t(numvertexes) * sizeof *dvertexes);
        world.add(dfaces, size_t(numfaces) * sizeof *dfaces);
        world.add(dmodels, size_t(nummodels) * sizeof *dmodels);
        world.add(dedges, size_t(numedges) * sizeof *dedges);
        world.add(dsurfedges, size_t(numsurfedges) * sizeof *dsurfedges);
        world.add(dmarksurfaces, size_t(nummarksurfaces) * sizeof *dmarksurfaces);
        world.add(texinfo, size_t(numtexinfo) * sizeof *texinfo);
        world.add(dvisdata, visdatasize);
        world.add(dtexdata, texdatasize);
        world.add(dentdata, entdatasize);
        world_key = world.finish();
        if (const char *dir = getenv("LM_CACHE_DIR")) {
            cache_dir = dir;
            fs::create_directories(cache_dir);
        }
    } catch (const std::exception &e) {
        Error("%s", e.what());
    }
}
extern "C" int lm_direct_recording() {
    return direct_collecting && !gpu_prepass;
}
extern "C" int lm_direct_lookup(int face, int point, const lm_direct_hit **hits, uint32_t *count) {
    if (gpu_prepass)
        return 0;
    try {
        if (direct_loaded) {
            const auto &data = direct_faces.at(face);
            if (point < 0 || size_t(point + 1) >= data.loaded_offsets.size())
                throw std::runtime_error("Direct sample layout changed");
            *count = data.loaded_offsets[point + 1] - data.loaded_offsets[point];
            *hits = *count ? data.loaded_hits.data() + data.loaded_offsets[point] : nullptr;
            return 1;
        }
        if (direct_collecting) {
            auto &data = direct_faces.at(face);
            if (size_t(point) != data.offsets.size())
                throw std::runtime_error("Unexpected direct sample order");
            data.offsets.push_back(lm::checked_cast<uint32_t>(data.hits.size()));
            if (direct_bytes.fetch_add(4) + 4 > direct_budget)
                abandon_direct_cache();
        }
        return 0;
    } catch (const std::exception &e) {
        Error("%s", e.what());
        return 0;
    }
}
extern "C" void lm_direct_record(int face, uint32_t light, float ratio) {
    try {
        if (direct_collecting) {
            direct_faces.at(face).hits.push_back({light, ratio});
            if (direct_bytes.fetch_add(sizeof(lm_direct_hit)) + sizeof(lm_direct_hit) > direct_budget)
                abandon_direct_cache();
        }
    } catch (const std::exception &e) {
        Error("%s", e.what());
    }
}
extern "C" int lm_sample_membership(int face, int sample, int included) {
    if (!frozen_membership.empty()) {
        const auto &base = frozen_membership.at(face);
        if (sample < 0 || size_t(sample) >= base.size())
            Error("Frozen derivative sample layout changed");
        included = base[sample];
    }
    auto &mask = sample_membership.at(face);
    mask.resize(sample + 1);
    mask[sample] = included;
    return included;
}
extern "C" void lm_direct_record_sky(int face, uint32_t light, const float *value, int diffuse) {
    try {
        if (!direct_collecting) return;
        auto &data = direct_faces.at(face);
        lm_direct_hit hit{};
        hit.light = light | (diffuse ? LM_SKY_DIFFUSE : LM_SKY_DIRECT);
        hit.sky = lm::checked_cast<uint32_t>(data.sky.size());
        data.sky.push_back({value[0], value[1], value[2]});
        data.hits.push_back(hit);
        constexpr size_t bytes = sizeof(lm_direct_hit) + 12;
        if (direct_bytes.fetch_add(bytes) + bytes > direct_budget) abandon_direct_cache();
    } catch (const std::exception &e) { Error("%s", e.what()); }
}
extern "C" const float *lm_direct_sky(int face, uint32_t index) {
    return direct_faces.at(face).loaded_sky[index].data();
}
extern "C" void IndexDirectLights(void);
extern "C" int lm_direct_prepass() {
    return gpu_prepass;
}
namespace {
thread_local std::vector<LMTraceRay> patch_rays;
}
extern "C" int lm_enqueue_patch_ray(int node, const float *start, const float *stop, unsigned bit) {
    if (lm_reference_mode || !getenv("LM_GPU_SOCKET"))
        return 0;
    LMTraceRay ray{};
    memcpy(ray.start, start, sizeof ray.start);
    memcpy(ray.stop, stop, sizeof ray.stop);
    ray.node = node;
    ray.bit = bit;
    patch_rays.push_back(ray);
    if (patch_rays.size() >= 65536)
        lm_flush_patch_rays();
    return 1;
}
namespace {
struct TraceResult {
    Bytes flags;
    double seconds;
};
TraceResult trace_gpu_rays(const LMTraceRay *rays, size_t count) {
    const char *path = getenv("LM_GPU_SOCKET");
    if (!path) throw std::runtime_error("Missing CUDA service path");
    FD fd(lm::connect_stream(path));
    GpuHeader header;
    header.magic = 0x4c4d5432;
    header.rows = lm::checked_cast<uint32_t>(count);
    header.entries = numnodes;
    write_all(fd.value, &header, sizeof header);
    GpuReply reply;
    read_all(fd.value, &reply, sizeof reply);
    if (reply.status != 1)
        throw std::runtime_error("CUDA trace scene rejected");
    uint64_t bytes = count * (sizeof(LMTraceRay) + 1) + numnodes * sizeof(LMTraceNode);
    write_all(fd.value, tnodes, numnodes * sizeof(LMTraceNode));
    write_all(fd.value, rays, count * sizeof(LMTraceRay));
    read_all(fd.value, &reply, sizeof reply);
    if (reply.status)
        throw std::runtime_error("CUDA visibility tracing failed");
    Bytes results(count);
    read_all(fd.value, results.data(), results.size());
    for (auto value : results)
        if (value > 2)
            throw std::runtime_error("Invalid CUDA trace result");
    {
        std::lock_guard lock(gpu_stats_mutex);
        stats.gpu_bytes += bytes;
        ++stats.gpu_scene_uploads;
    }
    return {std::move(results), reply.seconds};
}
} // namespace
extern "C" void lm_flush_patch_rays() {
    if (patch_rays.empty())
        return;
    try {
        auto traced = trace_gpu_rays(patch_rays.data(), patch_rays.size());
        auto &results = traced.flags;
        uint64_t uncertain = 0;
        bool audit = getenv("LM_AUDIT_GPU_TRACE") != nullptr;
        for (size_t i = 0; i < patch_rays.size(); ++i) {
            auto &ray = patch_rays[i];
            bool visible = results[i] == 1;
            if (audit && results[i] != 2 &&
                visible != (TestLine_r(ray.node, ray.start, ray.stop) == CONTENTS_EMPTY)) {
                Error("CUDA trace mismatch: node %d, (%a %a %a) -> (%a %a %a), result %d", ray.node,
                      ray.start[0], ray.start[1], ray.start[2], ray.stop[0], ray.stop[1], ray.stop[2],
                      int(results[i]));
            }
            if (results[i] == 2) {
                ++uncertain;
                visible = TestLine_r(ray.node, ray.start, ray.stop) == CONTENTS_EMPTY;
            }
            if (visible)
                std::atomic_ref<byte>(vismatrix[ray.bit >> 3]).fetch_or(
                    byte(1u << (ray.bit & 7)), std::memory_order_relaxed);
        }
        {
            std::lock_guard lock(gpu_stats_mutex);
            ++stats.gpu_trace_calls;
            stats.gpu_trace_rays += patch_rays.size();
            stats.gpu_trace_uncertain += uncertain;
            stats.gpu_seconds += traced.seconds;
        }
        patch_rays.clear();
    } catch (const std::exception &e) {
        Error("%s", e.what());
    }
}
extern "C" void lm_collect_direct(int face, int point, const float *position, const float *normal) {
    if (!gpu_prepass)
        return;
    auto &row = direct_visibility_faces.at(face);
    if (size_t(point) != row.offsets.size())
        Error("CUDA direct sample order changed");
    row.offsets.push_back(lm::checked_cast<uint32_t>(row.rays.size()));
    if (lm_audit_gpu_direct) {
        DirectSample sample{};
        memcpy(sample.position, position, sizeof sample.position);
        memcpy(sample.normal, normal, sizeof sample.normal);
        row.samples.push_back(sample);
    }
}
extern "C" void lm_add_direct_ray(int face, unsigned light, const float *position, const float *origin) {
    LMTraceRay ray{};
    memcpy(ray.start, position, sizeof ray.start);
    memcpy(ray.stop, origin, sizeof ray.stop);
    ray.bit = light;
    direct_visibility_faces.at(face).rays.push_back(ray);
}
extern "C" int lm_direct_visibility(int face, int point, unsigned light, const float *position,
                                    const float *normal) {
    if (!gpu_visibility_ready)
        return -1;
    auto &row = direct_visibility_faces.at(face);
    if (point < 0 || size_t(point + 1) >= row.offsets.size())
        Error("CUDA direct lookup out of bounds");
    if (lm_audit_gpu_direct &&
        (memcmp(row.samples[point].position, position, 12) || memcmp(row.samples[point].normal, normal, 12)))
        Error("CUDA prepass changed sample face %d point %d", face, point);
    if (row.lookup_point != point) {
        row.lookup_point = point;
        row.cursor = row.offsets[point];
    }
    size_t end = row.offsets[point + 1];
    while (row.cursor < end && row.rays[row.cursor].bit < light)
        ++row.cursor;
    if (row.cursor == end || row.rays[row.cursor].bit != light)
        return 0;
    return row.visibility[row.cursor];
}
extern "C" void lm_build_facelights() {
    try {
        if (!lm_reference_mode) {
            if (const char *value = getenv("LM_MEMBERSHIP_FD")) {
                int fd = atoi(value);
                lm::FileInfo info{};
                if (lm::file_info(fd, info) || info.st_size < 68 || info.st_size > (64 << 20))
                    throw std::runtime_error("Invalid derivative sampling descriptor");
                Bytes encoded(info.st_size);
                if (lm::seek(fd, 0, SEEK_SET) < 0) throw std::runtime_error("Cannot read derivative samples");
                read_all(fd, encoded.data(), encoded.size());
                Cursor cursor{encoded};
                std::string key(static_cast<const char *>(cursor.take(64)), 64);
                if (cursor.get<uint32_t>() != uint32_t(numfaces))
                    throw std::runtime_error("Derivative face count changed");
                std::vector<Bytes> masks(numfaces);
                for (auto &mask : masks) {
                    auto size = cursor.get<uint32_t>();
                    if (size > 65536) throw std::runtime_error("Invalid derivative sample count");
                    const auto *data = static_cast<const unsigned char *>(cursor.take(size));
                    mask.assign(data, data + size);
                    if (std::any_of(mask.begin(), mask.end(), [](auto v) { return v > 1; }))
                        throw std::runtime_error("Invalid derivative membership");
                }
                if (cursor.pos != encoded.size()) throw std::runtime_error("Trailing derivative samples");
                // Geometry/casting changes still use ordinary samples and must
                // pass the solver's structural and channel-independence checks.
                if (key == sampling_key()) frozen_membership = std::move(masks);
            }
        }
        IndexDirectLights();
        prepare_direct_cache();
        bool use_gpu = !lm_reference_mode && !direct_loaded && getenv("LM_GPU_SOCKET");
        if (use_gpu)
            direct_visibility_faces.resize(numfaces);
        unsigned count = 1;
        if (!lm_reference_mode)
            if (const char *value = getenv("LM_FACE_THREADS"))
                count = std::clamp(atoi(value), 1, 64);
        std::atomic<int> next{0};
        std::exception_ptr failure;
        std::mutex failure_mutex;
        auto work = [&] {
            try {
                for (;;) {
                    int face = next++;
                    if (face >= numfaces)
                        break;
                    if (use_gpu) {
                        gpu_prepass = true;
                        BuildFacelights(face);
                        gpu_prepass = false;
                        build_gpu_visibility(face);
                        gpu_visibility_ready = true;
                    }
                    BuildFacelights(face);
                    gpu_visibility_ready = false;
                    if (use_gpu)
                        direct_visibility_faces[face] = {};
                }
            } catch (...) {
                std::lock_guard lock(failure_mutex);
                if (!failure)
                    failure = std::current_exception();
                next = numfaces;
            }
        };
        std::vector<std::thread> threads;
        for (unsigned i = 1; i < count; ++i)
            threads.emplace_back(work);
        work();
        for (auto &thread : threads)
            thread.join();
        if (failure)
            std::rethrow_exception(failure);
        direct_visibility_faces.clear();
        finish_direct_cache();
    } catch (const std::exception &e) {
        Error("%s", e.what());
    }
}
extern "C" int lm_geometry_load(void *data, size_t size) {
    try {
        if (cache_dir.empty())
            return 0;
        auto file = cache_dir / ("geometry-" + geometry_key);
        CachedBytes payload;
        if (load_cache(file, payload) && payload.size() == size) {
            memcpy(data, payload.data(), size);
            ++stats.geometry_hits;
            geometry_lock.reset();
            return 1;
        }
        geometry_lock = lock_cache(file);
        return 0;
    } catch (const std::exception &e) {
        Error("%s", e.what());
        return 0;
    }
}
extern "C" void lm_geometry_save(const void *data, size_t size) {
    try {
        if (!geometry_lock)
            return;
        Bytes payload;
        append(payload, data, size);
        save_cache(cache_dir / ("geometry-" + geometry_key), payload);
        geometry_lock.reset();
    } catch (const std::exception &e) {
        Error("%s", e.what());
    }
}
extern "C" int lm_transfers_load() {
    try {
        transfer_offsets = {};
        transfer_table = {};
        transfer_payload.reset();
        transfer_key = patch_key();
        if (cache_dir.empty()) {
            ++stats.transfer_misses;
            return 0;
        }
        auto file = cache_dir / ("transfers-" + transfer_key);
        if (load_cache(file, transfer_payload)) {
            try {
                Cursor cursor{transfer_payload};
                if (cursor.get<uint32_t>() != num_patches)
                    throw std::runtime_error("Patch count changed");
                auto count = cursor.get<uint32_t>();
                if (count > (1u << 28)) throw std::runtime_error("Invalid transfer count");
                std::span<const uint32_t> offsets{
                    static_cast<const uint32_t *>(cursor.take(size_t(num_patches + 1) * 4)), num_patches + 1};
                std::span<const uint32_t> table{
                    static_cast<const uint32_t *>(cursor.take(size_t(count) * 4)), count};
                if (offsets.front() != 0 || offsets.back() != count ||
                    !std::is_sorted(offsets.begin(), offsets.end()))
                    throw std::runtime_error("Invalid transfer offsets");
                for (auto t : table)
                    if ((t & 65535u) >= num_patches)
                        throw std::runtime_error("Invalid transfer index");
                if (cursor.pos != transfer_payload.size())
                    throw std::runtime_error("Trailing cache bytes");
                for (unsigned i = 0; i < num_patches; ++i) {
                    patches[i].numtransfers = offsets[i + 1] - offsets[i];
                    // QRAD only reads transfer rows after construction. Keep the
                    // mapping alive through its CPU/GPU gather and process exit.
                    patches[i].transfers = const_cast<transfer_t *>(
                        reinterpret_cast<const transfer_t *>(table.data() + offsets[i]));
                }
                transfer_offsets = offsets;
                transfer_table = table;
                ++stats.transfer_hits;
                transfer_lock.reset();
                return 1;
            } catch (const std::runtime_error &) { /* Rebuild a corrupt/incompatible cache. */
            }
        }
        transfer_lock = lock_cache(file);
        ++stats.transfer_misses;
        return 0;
    } catch (const std::exception &e) {
        Error("%s", e.what());
        return 0;
    }
}
extern "C" void lm_transfers_save() {
    try {
        if (!transfer_lock)
            return;
        Bytes payload;
        append(payload, uint32_t(num_patches));
        std::vector<uint32_t> offsets(num_patches + 1);
        for (unsigned i = 0; i < num_patches; ++i)
            offsets[i + 1] = offsets[i] + patches[i].numtransfers;
        append(payload, offsets.back());
        append(payload, offsets.data(), offsets.size() * 4);
        for (unsigned i = 0; i < num_patches; ++i) {
            append(payload, patches[i].transfers, size_t(patches[i].numtransfers) * sizeof(transfer_t));
        }
        save_cache(cache_dir / ("transfers-" + transfer_key), payload);
        transfer_lock.reset();
    } catch (const std::exception &e) {
        Error("%s", e.what());
    }
}
extern "C" int lm_gpu_gather() {
    const char *path = getenv("LM_GPU_SOCKET");
    if (!path || !*path || !num_patches)
        return 0;
    try {
        std::vector<uint32_t> owned_offsets, owned_table;
        auto offsets = transfer_offsets, table = transfer_table;
        if (offsets.empty()) {
            owned_offsets.resize(num_patches + 1);
            size_t entries = 0;
            for (unsigned i = 0; i < num_patches; ++i) entries += patches[i].numtransfers;
            owned_table.reserve(entries);
            for (unsigned i = 0; i < num_patches; ++i) {
                owned_offsets[i] = lm::checked_cast<uint32_t>(owned_table.size());
                for (int j = 0; j < patches[i].numtransfers; ++j) {
                    auto t = patches[i].transfers[j];
                    owned_table.push_back(uint32_t(t.patch) | uint32_t(t.transfer) << 16);
                }
            }
            owned_offsets.back() = lm::checked_cast<uint32_t>(owned_table.size());
            offsets = owned_offsets;
            table = owned_table;
        }
        if (!path) throw std::runtime_error("Missing CUDA service path");
        FD fd(lm::connect_stream(path));
        GpuHeader header;
        bool reusable = !cache_dir.empty();
        if (reusable) header.magic = 0x4c4d4733;
        header.rows = num_patches;
        header.entries = table.size();
        write_all(fd.value, &header, sizeof header);
        if (reusable) write_all(fd.value, transfer_key.data(), 64);
        GpuReply reply;
        read_all(fd.value, &reply, sizeof reply);
        if (reply.status != 1 && !(reusable && reply.status == 3))
            throw std::runtime_error("CUDA worker rejected transfers");
        if (reply.status == 1) {
            write_all(fd.value, offsets.data(), offsets.size() * 4);
            write_all(fd.value, table.data(), table.size() * 4);
            stats.gpu_bytes += (offsets.size() + table.size()) * 4;
            ++stats.gpu_transfer_uploads;
        } else ++stats.gpu_transfer_reuses;
        write_all(fd.value, emitlight, num_patches * sizeof(vec3_t));
        read_all(fd.value, &reply, sizeof reply);
        if (reply.status)
            throw std::runtime_error("CUDA radiosity calculation failed");
        read_all(fd.value, addlight, num_patches * sizeof(vec3_t));
        if (getenv("LM_AUDIT_GPU_GATHER")) {
            std::vector<float> actual(num_patches * 3);
            memcpy(actual.data(), addlight, actual.size() * sizeof(float));
            RunThreadsOn(num_patches, false, GatherLight);
            if (memcmp(actual.data(), addlight, actual.size() * sizeof(float)))
                throw std::runtime_error("CUDA radiosity differs from QRAD gather");
        }
        ++stats.gpu_calls;
        stats.gpu_bytes += num_patches * sizeof(vec3_t) * 2;
        stats.gpu_seconds += reply.seconds;
        return 1;
    } catch (const std::exception &e) {
        Error("%s", e.what());
        return 0;
    }
}

extern "C" int lm_indirect_ready() {
    return !indirect_results.empty();
}
extern "C" int lm_indirect_sample(int face, int sample, float *result) {
    if (indirect_results.empty())
        return 0;
    if (face < 0 || size_t(face + 1) >= indirect_offsets.size() || sample < 0 ||
        uint32_t(sample) >= indirect_offsets[face + 1] - indirect_offsets[face])
        Error("Indirect sample lookup out of bounds");
    memcpy(result, indirect_results[indirect_offsets[face] + sample].light, sizeof(float) * 3);
    return 1;
}
extern "C" void lm_build_indirect() {
    indirect_results.clear();
    indirect_offsets.clear();
    const char *path = getenv("LM_GPU_SOCKET");
    if (lm_reference_mode || !path || !numbounce || !num_patches)
        return;
    try {
        indirect_offsets.resize(numfaces + 1);
        Hash geometry;
        geometry.add(uint32_t(1));
        geometry.add(world_key.data(), world_key.size());
        auto patches_key = patch_key();
        geometry.add(patches_key.data(), patches_key.size());
        geometry.add(smoothing_threshold);
        for (int face = 0; face < numfaces; ++face) {
            int count = lm_face_sample_count(face);
            if (count < 0 || uint64_t(indirect_offsets[face]) + count > (1u << 20))
                throw std::runtime_error("Invalid indirect sample count");
            indirect_offsets[face + 1] = indirect_offsets[face] + count;
            geometry.add(count);
            for (int sample = 0; sample < count; ++sample)
                geometry.add(lm_face_sample_position(face, sample), sizeof(float) * 3);
        }
        if (!indirect_offsets.back())
            return;
        std::vector<LMIndirectSample> plans(indirect_offsets.back());
        auto cache_key = geometry.finish();
        auto file = cache_dir / ("interpolation-" + cache_key);
        CachedBytes payload;
        bool loaded = !cache_dir.empty() && load_cache(file, payload) &&
                      payload.size() == plans.size() * sizeof(LMIndirectSample);
        if (loaded) {
            memcpy(plans.data(), payload.data(), payload.size());
            loaded = std::all_of(plans.begin(), plans.end(),
                                 [](const auto &p) { return valid_indirect_plan(p, num_patches); });
        }
        if (loaded)
            ++stats.indirect_hits;
        else {
            ++stats.indirect_misses;
            auto lock = cache_dir.empty() ? nullptr : lock_cache(file);
            std::fill(plans.begin(), plans.end(), LMIndirectSample{});
            lm_prepare_indirect_plan(plans.data(), indirect_offsets.data());
            if (lock) {
                Bytes encoded;
                append(encoded, plans.data(), plans.size() * sizeof(LMIndirectSample));
                save_cache(file, encoded);
            }
        }
        if (!path) throw std::runtime_error("Missing CUDA service path");
        FD fd(lm::connect_stream(path));
        GpuHeader header;
        header.magic = 0x4c4d4932;
        header.rows = lm::checked_cast<uint32_t>(plans.size());
        header.entries = num_patches;
        write_all(fd.value, &header, sizeof header);
        GpuReply reply;
        read_all(fd.value, &reply, sizeof reply);
        if (reply.status != 1)
            throw std::runtime_error("CUDA interpolation plan rejected");
        write_all(fd.value, plans.data(), plans.size() * sizeof(LMIndirectSample));
        stats.gpu_bytes += plans.size() * sizeof(LMIndirectSample);
        ++stats.gpu_plan_uploads;
        std::vector<float> light(num_patches * 3);
        for (unsigned i = 0; i < num_patches; ++i)
            memcpy(light.data() + i * 3, patches[i].totallight, sizeof(vec3_t));
        write_all(fd.value, light.data(), light.size() * sizeof(float));
        read_all(fd.value, &reply, sizeof reply);
        if (reply.status)
            throw std::runtime_error("CUDA interpolation failed");
        indirect_results.resize(plans.size());
        read_all(fd.value, indirect_results.data(), indirect_results.size() * sizeof(LMIndirectResult));
        uint64_t uncertain = 0;
        bool audit = getenv("LM_AUDIT_GPU_INDIRECT") != nullptr;
        for (size_t i = 0; i < plans.size(); ++i) {
            auto &result = indirect_results[i];
            if (result.uncertain > 1)
                throw std::runtime_error("Invalid CUDA interpolation result");
            if (result.uncertain) {
                ++uncertain;
                lm_evaluate_indirect(&plans[i], result.light);
            } else if (audit) {
                float expected[3];
                lm_evaluate_indirect(&plans[i], expected);
                if (memcmp(expected, result.light, sizeof expected))
                    throw std::runtime_error("CUDA interpolation differs from QRAD at sample " +
                                             std::to_string(i));
            }
        }
        ++stats.gpu_indirect_calls;
        stats.gpu_indirect_samples += plans.size();
        stats.gpu_indirect_uncertain += uncertain;
        stats.gpu_seconds += reply.seconds;
        stats.gpu_bytes += light.size() * sizeof(float) + indirect_results.size() * sizeof(LMIndirectResult);
    } catch (const std::exception &e) {
        Error("%s", e.what());
    }
}
namespace {
void build_gpu_visibility(int index) {
    auto &face = direct_visibility_faces[index];
    face.offsets.push_back(lm::checked_cast<uint32_t>(face.rays.size()));
    face.visibility.reserve(face.rays.size());
    for (size_t first = 0; first < face.rays.size();) {
        size_t count = std::min(size_t(65536), face.rays.size() - first);
        auto result = trace_gpu_rays(face.rays.data() + first, count);
        if (lm_audit_gpu_direct)
            for (size_t i = 0; i < count; ++i) {
                auto &ray = face.rays[first + i];
                if (result.flags[i] != 2 &&
                    (result.flags[i] == 1) != (TestLine_r(0, ray.start, ray.stop) == CONTENTS_EMPTY))
                    Error("CUDA direct ray mismatch face %d ray %zu", index, first + i);
            }
        uint64_t uncertain = std::count(result.flags.begin(), result.flags.end(), 2);
        face.visibility.insert(face.visibility.end(), result.flags.begin(), result.flags.end());
        {
            std::lock_guard lock(gpu_stats_mutex);
            ++stats.gpu_direct_calls;
            stats.gpu_direct_pairs += count;
            stats.gpu_direct_uncertain += uncertain;
            stats.gpu_seconds += result.seconds;
        }
        first += count;
    }
}
} // namespace
extern "C" void lm_finish() {
    lm_mark(-1);
    if (const char *path = getenv("LM_RESULT_FD")) {
        try {
            if (nonlighting_key() != source_structure_key)
                throw std::runtime_error("Compiler changed non-lighting BSP data");
            int fd = atoi(path);
            const uint32_t magic = 0x4c4d5239;
            write_all(fd, &magic, sizeof magic);
            write_all(fd, &stats, sizeof stats);
            auto structure = patch_key(true);
            write_all(fd, structure.data(), 64);
            uint32_t counts[3] = {uint32_t(numfaces), uint32_t(lightdatasize), num_patches};
            write_all(fd, counts, sizeof counts);
            write_all(fd, dfaces, numfaces * sizeof(dface_t));
            write_all(fd, dlightdata, lightdatasize);
            for (unsigned i = 0; i < num_patches; ++i) {
                int values[2] = {patches[i].faceNumber, patches[i].samples};
                write_all(fd, values, sizeof values);
            }
            auto sample_key = sampling_key();
            write_all(fd, sample_key.data(), 64);
            auto face_count = uint32_t(numfaces);
            write_all(fd, &face_count, 4);
            for (const auto &mask : sample_membership) {
                auto count = uint32_t(mask.size());
                write_all(fd, &count, 4);
                write_all(fd, mask.data(), mask.size());
            }
        } catch (const std::exception &e) {
            Error("%s", e.what());
        }
    }
}

int lm::gpu_service(const fs::path &path) {
    GPU gpu;
    lm::StreamServer server(path);
    for (;;) {
        FD client(server.accept());
        if (client.value < 0) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error("CUDA socket accept failed");
        }
        try {
            GpuHeader header;
            read_all(client.value, &header, sizeof header);
            if (header.magic == 0x4c4d4932) {
                if (!header.rows || header.rows > (1u << 20) || !header.entries ||
                    header.entries > MAX_PATCHES)
                    throw std::runtime_error("Invalid CUDA interpolation request");
                GpuReply reply;
                reply.status = 1;
                write_all(client.value, &reply, sizeof reply);
                auto start = Clock::now();
                std::vector<LMIndirectSample> plans(header.rows);
                read_all(client.value, plans.data(), plans.size() * sizeof(LMIndirectSample));
                for (const auto &plan : plans)
                    if (!valid_indirect_plan(plan, header.entries))
                        throw std::runtime_error("Invalid CUDA interpolation plan");
                std::vector<float> light(header.entries * 3);
                read_all(client.value, light.data(), light.size() * sizeof(float));
                std::vector<LMIndirectResult> result;
                gpu.interpolate(plans, light, result);
                reply.status = 0;
                reply.seconds = std::chrono::duration<double>(Clock::now() - start).count();
                write_all(client.value, &reply, sizeof reply);
                write_all(client.value, result.data(), result.size() * sizeof(LMIndirectResult));
                continue;
            }
            if (header.magic == 0x4c4d5432) {
                if (!header.rows || header.rows > (1u << 20) || !header.entries ||
                    header.entries > MAX_MAP_NODES)
                    throw std::runtime_error("Invalid CUDA ray batch");
                GpuReply reply;
                reply.status = 1;
                write_all(client.value, &reply, sizeof reply);
                auto start = Clock::now();
                std::vector<LMTraceNode> nodes(header.entries);
                read_all(client.value, nodes.data(), nodes.size() * sizeof(LMTraceNode));
                std::vector<LMTraceRay> rays(header.rows);
                read_all(client.value, rays.data(), rays.size() * sizeof(LMTraceRay));
                Bytes result;
                gpu.trace(nodes, rays, result);
                reply.status = 0;
                reply.seconds = std::chrono::duration<double>(Clock::now() - start).count();
                write_all(client.value, &reply, sizeof reply);
                write_all(client.value, result.data(), result.size());
                continue;
            }
            if ((header.magic != 0x4c4d4732 && header.magic != 0x4c4d4733) || !header.rows || header.rows > MAX_PATCHES ||
                header.entries > uint64_t(header.rows) * header.rows || header.entries > (1ULL << 28))
                throw std::runtime_error("Invalid CUDA request");
            std::string key;
            if (header.magic == 0x4c4d4733) {
                key.resize(64);
                read_all(client.value, key.data(), key.size());
                if (key.find_first_not_of("0123456789abcdef") != std::string::npos)
                    throw std::runtime_error("Invalid transfer signature");
            }
            bool reuse = gpu.has_transfers(key, header.rows, header.entries);
            GpuReply reply;
            reply.status = reuse ? 3 : 1;
            write_all(client.value, &reply, sizeof reply);
            auto start = Clock::now();
            std::vector<uint32_t> offsets, transfers;
            if (!reuse) {
                offsets.resize(header.rows + 1);
                transfers.resize(header.entries);
                read_all(client.value, offsets.data(), offsets.size() * 4);
                read_all(client.value, transfers.data(), transfers.size() * 4);
                if (offsets.front() || offsets.back() != transfers.size() ||
                    !std::is_sorted(offsets.begin(), offsets.end()))
                    throw std::runtime_error("Invalid CSR offsets");
                for (auto t : transfers)
                    if ((t & 65535u) >= header.rows)
                        throw std::runtime_error("Invalid CSR source");
            }
            std::vector<float> emission(header.rows * 3), output;
            read_all(client.value, emission.data(), emission.size() * 4);
            gpu.calculate(offsets, transfers, emission, output, key, reuse);
            reply.status = 0;
            reply.seconds = std::chrono::duration<double>(Clock::now() - start).count();
            write_all(client.value, &reply, sizeof reply);
            write_all(client.value, output.data(), output.size() * 4);
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            GpuReply reply;
            reply.status = 2;
            try {
                write_all(client.value, &reply, sizeof reply);
            } catch (...) {
            }
        }
    }
}
void lm::gpu_self_test() {
    GPU gpu;
    std::vector<LMTraceNode> test_nodes(3);
    test_nodes[0] = {3, {.6f, .8f, 0}, 13.f, {1, 2}, 0};
    test_nodes[1] = {1, {0, 1, 0}, 0, {-1, -2}, 0};
    test_nodes[2] = {2, {0, 0, 1}, 8, {-6, -1}, 0};
    std::vector<LMTraceRay> rays(65536);
    uint32_t random = 0x345678;
    auto *old_nodes = tnodes;
    tnodes = test_nodes.data();
    size_t fallback = 0;
    constexpr unsigned batches = 4;
    // Changed multi-megabyte inputs exercise reused device buffers and copy
    // completion before execution on the non-blocking compute stream.
    for (unsigned batch = 0; batch < batches; ++batch) {
        for (auto &ray : rays) {
            ray.node = batch == 3 ? 1 : 0;
            for (int c = 0; c < 3; ++c) {
                random = random * 1664525u + 1013904223u;
                ray.start[c] = float(int(random % 200001) - 100000) / 100.f;
                random = random * 1664525u + 1013904223u;
                ray.stop[c] = float(int(random % 200001) - 100000) / 100.f;
            }
        }
        for (int i = 0; i < 32; ++i) {
            rays[i].start[0] = rays[i].stop[0] = 100;
            rays[i].start[1] = std::nextafter(float(.01 * (i % 3 - 1)), i % 2 ? INFINITY : -INFINITY);
            rays[i].stop[1] = i % 2 ? 1.0f : -1.0f;
        }
        Bytes traced;
        gpu.trace(test_nodes, rays, traced);
        for (size_t i = 0; i < rays.size(); ++i) {
            bool reference = TestLine_r(rays[i].node, rays[i].start, rays[i].stop) == CONTENTS_EMPTY;
            if (traced[i] == 2)
                ++fallback;
            else if ((traced[i] == 1) != reference)
                throw std::runtime_error("CUDA visibility disagrees with QRAD reference at ray " +
                                         std::to_string(i));
        }
    }
    // Exercise shrinking and growing requests through the retained buffers.
    // Real direct-light batches vary by face and interleave across workers.
    for (unsigned count : {1u, 3u, 127u, 128u, 129u, 2047u, 4097u, 8192u, 7u, 65u}) {
        std::vector<LMTraceRay> changed(rays.begin(), rays.begin() + count);
        Bytes traced;
        gpu.trace(test_nodes, changed, traced);
        for (size_t i = 0; i < changed.size(); ++i)
            if (traced[i] != 2 && (traced[i] == 1) !=
                (TestLine_r(changed[i].node, changed[i].start, changed[i].stop) == CONTENTS_EMPTY))
                throw std::runtime_error("CUDA changing-size trace batch differs from QRAD");
    }
    tnodes = old_nodes;
    std::cout << "CUDA tracing: " << rays.size() * batches << " reference comparisons with reused buffers; "
              << fallback << " guarded fallbacks\n";
    constexpr unsigned rows = 513;
    std::vector<uint32_t> offsets(rows + 1), transfers;
    std::vector<float> emission(rows * 3), expected(rows * 3), actual;
    for (unsigned i = 0; i < emission.size(); ++i)
        emission[i] = float((i * 73) % 10007) / 131.0f;
    for (unsigned i = 0; i < rows; ++i) {
        offsets[i] = lm::checked_cast<uint32_t>(transfers.size());
        for (unsigned j = 0; j < i % 97; ++j) {
            auto source = (i * 13 + j * 23) % rows, weight = (i * 3 + j * 83) % 16384;
            transfers.push_back(source | weight << 16);
            for (unsigned c = 0; c < 3; ++c) {
                volatile float product = emission[source * 3 + c] * float(weight);
                expected[i * 3 + c] += product;
            }
        }
    }
    offsets.back() = lm::checked_cast<uint32_t>(transfers.size());
    num_patches = rows;
    for (unsigned i = 0; i < rows; ++i) {
        patches[i].numtransfers = offsets[i + 1] - offsets[i];
        patches[i].transfers = reinterpret_cast<transfer_t *>(transfers.data() + offsets[i]);
    }
    memcpy(emitlight, emission.data(), emission.size() * 4);
    RunThreadsOn(rows, false, GatherLight);
    memcpy(expected.data(), addlight, expected.size() * 4);
    gpu.calculate(offsets, transfers, emission, actual);
    for (size_t i = 0; i < actual.size(); ++i)
        if (std::bit_cast<uint32_t>(actual[i]) != std::bit_cast<uint32_t>(expected[i]))
            throw std::runtime_error("CUDA gather differs at channel " + std::to_string(i) + ": GPU=" +
                                     std::to_string(actual[i]) + ", CPU=" + std::to_string(expected[i]) +
                                     ", bits " + std::to_string(std::bit_cast<uint32_t>(actual[i])) + " / " +
                                     std::to_string(std::bit_cast<uint32_t>(expected[i])));
    const std::string resident_key(64, 'a');
    gpu.calculate(offsets, transfers, emission, actual, resident_key);
    if (!gpu.has_transfers(resident_key, rows, transfers.size()) ||
        gpu.has_transfers(std::string(64, 'b'), rows, transfers.size()) ||
        gpu.has_transfers(resident_key, rows + 1, transfers.size()))
        throw std::runtime_error("Resident transfers accepted an incompatible key or size");
    std::fill(emission.begin(), emission.end(), 0.0f);
    gpu.calculate({}, {}, emission, actual, resident_key, true);
    if (std::any_of(actual.begin(), actual.end(), [](float x) { return x != 0; }))
        throw std::runtime_error("CUDA reused stale emission");

    std::vector<LMIndirectSample> plans(1025);
    for (size_t i = 0; i < plans.size(); ++i) {
        auto &p = plans[i];
        p.kind = i % 4;
        for (unsigned c = 0; c < 3; ++c)
            p.patch[c] = (i * 13 + c * 97) % rows;
        p.x = float(int(i % 101) - 50) / 53.f;
        p.y = float(int(i % 73) - 36) / 37.f;
        p.x2 = i % 7 == 0 ? .009f : i % 7 == 1 ? .01f : .25f;
        p.y1 = i % 11 == 0 ? 0.f : i % 11 == 1 ? -.01f : -.375f;
    }
    size_t certified = 0, guarded = 0;
    constexpr unsigned interpolation_iterations = 64;
    for (unsigned iteration = 0; iteration < interpolation_iterations; ++iteration) {
        // Same dimensions, different geometry/weights: each request must use
        // its new inputs even when the device allocation is already large enough.
        for (auto &plan : plans)
            for (auto &patch : plan.patch)
                patch = (patch + 7) % rows;
        for (auto &transfer : transfers)
            transfer = (transfer & 65535u) | (((transfer >> 16) + 13) % 16384u) << 16;
        for (unsigned i = 0; i < rows; ++i)
            for (unsigned c = 0; c < 3; ++c) {
                float value = float(int((i * 59 + c * 31 + iteration * 7) % 2003) - 1001) / 13.f;
                if (iteration == 1)
                    value *= 1e-30f;
                if (iteration == 2)
                    value *= 1e6f;
                patches[i].totallight[c] = emission[i * 3 + c] = value;
            }
        std::vector<LMIndirectResult> interpolated;
        gpu.interpolate(plans, emission, interpolated);
        for (size_t i = 0; i < plans.size(); ++i) {
            float reference[3];
            lm_evaluate_indirect(&plans[i], reference);
            if (interpolated[i].uncertain == 1)
                ++guarded;
            else if (interpolated[i].uncertain || memcmp(interpolated[i].light, reference, sizeof reference))
                throw std::runtime_error("CUDA interpolation differs at test sample " + std::to_string(i));
            else
                ++certified;
        }
        // A recovery alternates these stages while retaining CUDA buffers.
        // Check the return to gathering after the larger interpolation output.
        memcpy(emitlight, emission.data(), emission.size() * sizeof(float));
        RunThreadsOn(rows, false, GatherLight);
        gpu.calculate(offsets, transfers, emission, actual);
        if (memcmp(actual.data(), addlight, actual.size() * sizeof(float)))
            throw std::runtime_error("CUDA gather changed after interpolation buffer reuse");
        if (iteration % 16 == 0) {
            test_nodes[0].dist += 17;
            test_nodes[1].dist -= 11;
            tnodes = test_nodes.data();
            std::vector<LMTraceRay> changed(rays.begin(), rays.begin() + 1025);
            for (auto &ray : changed)
                ray.node = 0;
            Bytes traced;
            gpu.trace(test_nodes, changed, traced);
            for (size_t i = 0; i < changed.size(); ++i)
                if (traced[i] != 2 && (traced[i] == 1) !=
                    (TestLine_r(0, changed[i].start, changed[i].stop) == CONTENTS_EMPTY))
                    throw std::runtime_error("CUDA retained a previous request's scene");
            tnodes = old_nodes;
        }
    }
    if (certified * 3 < plans.size() * interpolation_iterations * 2)
        throw std::runtime_error("CUDA interpolation did not certify enough test samples");
    std::cout << "CUDA interpolation: " << certified << " exact CPU comparisons, " << guarded
              << " guarded fallbacks; empty/copy/triangle/edge, changed light and interleaved gather passed\n";
    num_patches = 0;
    std::cout << "CUDA radiosity: exact CPU/GPU agreement; fresh scenes, transfers and plans passed\n";
}
void lm::cache_self_test(const fs::path &directory) {
    cache_dir = directory;
    fs::create_directories(cache_dir);
    auto contested = cache_dir / "producer-test";
    auto owner = lock_cache(contested);
    auto start = Clock::now();
    if (!owner || lock_cache(contested) || std::chrono::duration<double>(Clock::now() - start).count() > .1)
        throw std::runtime_error("Cache contention blocked a lighting worker");
    owner.reset();
    direct_key = std::string(64, '7');
    direct_lock = lock_cache(cache_dir / ("direct-" + direct_key));
    direct_faces.resize(1);
    direct_collecting = true;
    direct_bytes = direct_budget;
    lm_direct_record(0, 0, 1.f);
    if (direct_collecting || direct_lock || !direct_overflow ||
        !fs::exists(cache_dir / ("direct-" + direct_key + ".oversize")))
        throw std::runtime_error("Direct cache overflow retained its producer lock");
    if (!lock_cache(cache_dir / ("direct-" + direct_key)))
        throw std::runtime_error("Overflow producer lock remained held");
    direct_faces.clear();
    geometry_key = std::string(64, '0');
    world_key = std::string(64, '1');
    int old_faces = numfaces, old_leafs = numleafs;
    auto *old_light = directlights[1];
    directlight_t light{};
    light.type = emit_point;
    numfaces = 1;
    numleafs = 2;
    directlights[1] = &light;
    prepare_direct_cache();
    const lm_direct_hit *hits;
    uint32_t hit_count;
    lm_direct_lookup(0, 0, &hits, &hit_count);
    const float signed_ratio = -2.5e-8f;
    lm_direct_record(0, 0, signed_ratio);
    finish_direct_cache();
    prepare_direct_cache();
    if (!lm_direct_lookup(0, 0, &hits, &hit_count) || hit_count != 1 || hits[0].ratio != signed_ratio)
        throw std::runtime_error("Valid signed QRAD ratios were rejected from the cache");
    finish_direct_cache();
    light.origin[0] = 1;
    prepare_direct_cache();
    direct_bytes = direct_budget;
    lm_direct_record(0, 0, 1.f);
    finish_direct_cache();
    prepare_direct_cache();
    if (direct_collecting || direct_lock || direct_loaded)
        throw std::runtime_error("Oversized direct cache was retried at the same budget");
    finish_direct_cache();
    numfaces = old_faces;
    numleafs = old_leafs;
    directlights[1] = old_light;
    std::array<int, 8> original{1, 2, 3, 4, 5, 6, 7, 8}, read{};
    if (lm_geometry_load(read.data(), sizeof read))
        throw std::runtime_error("Unexpected geometry cache hit");
    lm_geometry_save(original.data(), sizeof original);
    if (!lm_geometry_load(read.data(), sizeof read) || read != original)
        throw std::runtime_error("Geometry cache roundtrip failed");
    auto verified_bytes = stats.cache_verified_bytes;
    bool reusable_stamp = false;
    {
        FD file(lm::open_file(cache_dir / ("geometry-" + geometry_key), O_RDONLY | O_CLOEXEC));
        lm::FileInfo info{};
        if (file.value >= 0 && !lm::file_info(file.value, info))
            lm::file_stamp(file.value, info, &reusable_stamp);
    }
    if (!lm_geometry_load(read.data(), sizeof read) ||
        (reusable_stamp && stats.cache_verified_bytes != verified_bytes))
        throw std::runtime_error("An unchanged cache entry was hashed again");
    {
        auto file = cache_dir / ("geometry-" + geometry_key);
        std::fstream corrupt(file, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekp(64);
        char changed = 9;
        corrupt.write(&changed, 1);
        corrupt.close();
        if (lm_geometry_load(read.data(), sizeof read))
            throw std::runtime_error("Changed cache data bypassed checksum validation");
        lm_geometry_save(original.data(), sizeof original);
        if (!lm_geometry_load(read.data(), sizeof read) || read != original)
            throw std::runtime_error("Atomically replaced cache entry was not revalidated");
    }
    num_patches = 1;
    patches[0] = {};
    patches[0].plane = &dplanes[0];
    patches[0].winding = AllocWinding(3);
    patches[0].winding->numpoints = 3;
    for (int i = 0; i < 3; ++i)
        for (int c = 0; c < 3; ++c)
            patches[0].winding->p[i][c] = i == c;
    patches[0].area = 1;
    patches[0].baselight[0] = patches[0].baselight[1] = patches[0].baselight[2] = 30;
    if (lm_transfers_load())
        throw std::runtime_error("Unexpected transfer cache hit");
    lm_transfers_save();
    patches[0].baselight[0] = 31;
    if (!lm_transfers_load())
        throw std::runtime_error("Compatible emission did not reuse transfers");
    auto structure = patch_key(true);
    patches[0].samples = 5;
    patches[0].baselight[0] = patches[0].baselight[1] = patches[0].baselight[2] = 24;
    if (!lm_transfers_load() || patch_key(true) == structure)
        throw std::runtime_error("Transfer reuse and fitting structure were not separated");
    patches[0].baselight[0] = patches[0].baselight[1] = patches[0].baselight[2] = 30;
    patches[0].origin[0] = 1;
    if (lm_transfers_load())
        throw std::runtime_error("Geometry change did not invalidate transfers");
    transfer_lock.reset();
    FreeWinding(patches[0].winding);
    num_patches = 0;
    std::cout << "Cache: geometry/transfer reuse and independent sampling/casting invalidation passed\n";
}
