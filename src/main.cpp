#include "runtime.h"
#include "compiler.h"
#include "platform.h"
#include "checked_cast.h"
#include "progress.h"
extern "C" {
#include "cmdlib.h"
#include "mathlib.h"
#include "bspfile.h"
}
#include <Eigen/Core>
#include <Eigen/QR>
#include <LBFGSB.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<unsigned char>;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RGB = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
const fs::path root = fs::current_path();
volatile sig_atomic_t cancelled = 0;
thread_local unsigned batch_thread_share = 0;
void cancel_handler(int) {
    cancelled = 1;
}
void check_cancelled() {
    if (cancelled)
        throw std::runtime_error("Cancelled");
}
double seconds(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
std::string elapsed_time(double elapsed) {
    auto total = std::max(0LL, std::llround(elapsed));
    std::string text;
    auto append = [&](long long value, const char *unit) {
        if (!text.empty()) text += ' ';
        text += std::to_string(value) + ' ' + unit;
        if (value != 1) text += 's';
    };
    if (total >= 3600) append(total / 3600, "hour");
    if (total / 60 % 60) append(total / 60 % 60, "minute");
    if (total % 60 || text.empty()) append(total % 60, "second");
    return text;
}
std::string upper(std::string s) {
    for (auto &c : s)
        c = std::toupper(static_cast<unsigned char>(c));
    return s;
}
std::string lower(std::string s) {
    for (auto &c : s)
        c = std::tolower(static_cast<unsigned char>(c));
    return s;
}
std::string worker_error_summary(const std::string &log) {
    std::istringstream lines(log);
    std::string line, last;
    bool after_error = false;
    while (std::getline(lines, line)) {
        auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        line = line.substr(first, line.find_last_not_of(" \t\r") - first + 1);
        if (line.find("ERROR") != std::string::npos && line.find_first_not_of("* ERROR") == std::string::npos) {
            after_error = true;
            continue;
        }
        if (after_error || line.starts_with("Error:") || line.starts_with("Assume "))
            return line.substr(0, 400);
        last = line;
    }
    return last.empty() ? "Worker exited without an error message" : last.substr(0, 400);
}
bool inside_project(const fs::path &path) {
    auto relative = fs::weakly_canonical(path).lexically_relative(fs::canonical(root));
    return !relative.empty() && *relative.begin() != "..";
}
struct TempDir {
    fs::path path;
    TempDir() : path(lm::temporary_directory(root)) {}
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};
struct FD {
    int fd = -1;
    explicit FD(int fd = -1) : fd(fd) {}
    ~FD() {
        if (fd >= 0)
            lm::close_file(fd);
    }
    FD(const FD &) = delete;
    FD &operator=(const FD &) = delete;
};
void write_all(int fd, const void *data, size_t size) {
    auto *p = static_cast<const char *>(data);
    while (size) {
        int64_t n = lm::write(fd, p, size);
        if (n < 0 && errno == EINTR) {
            check_cancelled();
            continue;
        }
        if (n <= 0)
            throw std::runtime_error("Write failed");
        p += n;
        size -= n;
    }
}
void read_all(int fd, void *data, size_t size) {
    auto *p = static_cast<unsigned char *>(data);
    while (size) {
        auto count = lm::read(fd, p, size);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("Stream read failed");
        p += count;
        size -= count;
    }
}
Bytes read_fd(int fd) {
    lm::FileInfo st{};
    if (lm::file_info(fd, st) || st.st_size < 0 || uint64_t(st.st_size) > (4ULL << 30))
        throw std::runtime_error("Invalid worker response size");
    Bytes data(st.st_size);
    size_t pos = 0;
    while (pos < data.size()) {
        auto n = lm::read_at(fd, data.data() + pos, data.size() - pos, pos);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("Worker response read failed");
        pos += n;
    }
    return data;
}
Bytes read_file(const fs::path &file) {
    FD fd(lm::open_file(file.c_str(), O_RDONLY | O_CLOEXEC));
    if (fd.fd < 0)
        throw std::runtime_error("Cannot read " + file.string());
    return read_fd(fd.fd);
}
std::unique_ptr<FD> memory_file(const fs::path &directory, const char *name, const Bytes &data = {}) {
    auto result = std::make_unique<FD>(lm::memory_file(directory, name));
    if (result->fd < 0)
        throw std::runtime_error("Cannot allocate in-memory probe transport");
    write_all(result->fd, data.data(), data.size());
    return result;
}
void atomic_write(const fs::path &destination, const std::string &text) {
    fs::create_directories(destination.parent_path());
    std::string pattern =
        (destination.parent_path() / ("." + destination.filename().string() + ".XXXXXX")).string();
    FD fd(lm::temporary_file(pattern));
    if (fd.fd < 0)
        throw std::runtime_error("Cannot save result");
    try {
        write_all(fd.fd, text.data(), text.size());
        lm::atomic_replace(pattern, destination);
    } catch (...) {
        fs::remove(pattern);
        throw;
    }
}
struct Cursor {
    const Bytes &data;
    size_t pos = 0;
    const unsigned char *take(size_t n) {
        if (n > data.size() - pos)
            throw std::runtime_error("Truncated BSP/worker data");
        auto p = data.data() + pos;
        pos += n;
        return p;
    }
    template <class T> T get() {
        T v;
        memcpy(&v, take(sizeof v), sizeof v);
        return v;
    }
};
template <class T> T value_at(const Bytes &bytes, size_t offset) {
    Cursor cursor{bytes, offset};
    if (offset > bytes.size())
        throw std::runtime_error("BSP offset out of range");
    return cursor.get<T>();
}
struct Profile {
    mutable std::mutex mutex;
    lm::Progress *progress = nullptr;
    std::map<std::string, double> stages;
    lm_stats compiler{};
    uint64_t probes = 0, probe_cache_hits = 0, disk_bytes_read = 0, disk_bytes_written = 0, ipc_bytes = 0,
             worker_peak_rss = 0;
    uint64_t derivative_probes = 0, jacobian_builds = 0, jacobian_reuses = 0, threshold_probes = 0;
    uint64_t derivative_batched_columns = 0, derivative_fallback_columns = 0, frozen_derivative_builds = 0;
    uint64_t optimizer_stalls = 0;
    Clock::time_point start = Clock::now();
    void status(std::string text, int remaining_phases = -1, double phase_seconds = 0) {
        if (progress) progress->set(std::move(text), remaining_phases, phase_seconds);
    }
    void note(const std::string &text) { if (progress) progress->note(text); }
    void add(const std::string &stage, double value) {
        std::lock_guard lock(mutex);
        stages[stage] += value;
    }
    void sample(const lm_stats &stats, uint64_t memory, uint64_t ipc) {
        std::lock_guard lock(mutex);
        ++probes;
        if (progress) progress->baked();
        ipc_bytes += ipc;
        worker_peak_rss = std::max(worker_peak_rss, memory);
        for (int i = 0; i < LM_PHASE_COUNT; ++i) {
            compiler.seconds[i] += stats.seconds[i];
            compiler.cpu_seconds[i] += stats.cpu_seconds[i];
        }
        compiler.geometry_hits += stats.geometry_hits;
        compiler.transfer_hits += stats.transfer_hits;
        compiler.transfer_misses += stats.transfer_misses;
        compiler.cache_read_bytes += stats.cache_read_bytes;
        compiler.cache_write_bytes += stats.cache_write_bytes;
        compiler.gpu_calls += stats.gpu_calls;
        compiler.gpu_bytes += stats.gpu_bytes;
        compiler.gpu_seconds += stats.gpu_seconds;
        compiler.gpu_direct_calls += stats.gpu_direct_calls;
        compiler.gpu_direct_pairs += stats.gpu_direct_pairs;
        compiler.gpu_direct_uncertain += stats.gpu_direct_uncertain;
        compiler.direct_overflows += stats.direct_overflows;
        compiler.cache_lock_contentions += stats.cache_lock_contentions;
        compiler.gpu_trace_calls += stats.gpu_trace_calls;
        compiler.gpu_trace_rays += stats.gpu_trace_rays;
        compiler.gpu_trace_uncertain += stats.gpu_trace_uncertain;
        compiler.direct_hits += stats.direct_hits;
        compiler.direct_misses += stats.direct_misses;
        compiler.indirect_hits += stats.indirect_hits;
        compiler.indirect_misses += stats.indirect_misses;
        compiler.gpu_indirect_calls += stats.gpu_indirect_calls;
        compiler.gpu_indirect_samples += stats.gpu_indirect_samples;
        compiler.gpu_indirect_uncertain += stats.gpu_indirect_uncertain;
        compiler.gpu_scene_uploads += stats.gpu_scene_uploads;
        compiler.gpu_transfer_uploads += stats.gpu_transfer_uploads;
        compiler.gpu_plan_uploads += stats.gpu_plan_uploads;
        compiler.gpu_transfer_reuses += stats.gpu_transfer_reuses;
        compiler.cache_verified_bytes += stats.cache_verified_bytes;
    }
};
struct Timer {
    Profile &profile;
    std::string name;
    Clock::time_point start = Clock::now();
    Timer(Profile &profile, std::string name) : profile(profile), name(std::move(name)) {}
    ~Timer() { profile.add(name, seconds(start)); }
};
using Entity = std::map<std::string, std::string>;
struct Face {
    dface_t disk{};
    std::string texture;
    int width = 0, height = 0, u = 0, v = 0;
    double minlight = 0;
    std::map<int, std::pair<int, int>> offsets;
};
class BSP {
  public:
    Bytes bytes;
    dheader_t header{};
    std::vector<Entity> entities;
    std::vector<std::string> textures;
    std::vector<Face> faces;
    RGB target;
    Vector floor;
    std::vector<int> face_ids, styles;
    std::string digest;
    CompilerEncoding encoding;
    bool native_encoding = false;
    template <class T> std::vector<T> lump(int index) const {
        const auto &l = header.lumps[index];
        if (l.filelen % sizeof(T))
            throw std::runtime_error("Invalid BSP lump stride");
        std::vector<T> result(l.filelen / sizeof(T));
        if (l.filelen)
            memcpy(result.data(), bytes.data() + l.fileofs, l.filelen);
        return result;
    }
    explicit BSP(Bytes bytes) : bytes(std::move(bytes)) {
        static_assert(sizeof(dheader_t) == 124 && sizeof(dface_t) == 20 && sizeof(dmodel_t) == 64 &&
                      sizeof(texinfo_t) == 40);
        header = value_at<dheader_t>(this->bytes, 0);
        if (header.version != 30)
            throw std::runtime_error("Expected a GoldSrc BSP30 map");
        for (const auto &l : header.lumps)
            if (l.fileofs < 0 || l.filelen < 0 || uint64_t(l.fileofs) + l.filelen > this->bytes.size())
                throw std::runtime_error("BSP lump outside the file");
        const size_t limits[15] = {MAX_MAP_ENTSTRING,
                                   MAX_MAP_PLANES * sizeof(dplane_t),
                                   MAX_MAP_MIPTEX,
                                   MAX_MAP_VERTS * sizeof(dvertex_t),
                                   MAX_MAP_VISIBILITY,
                                   MAX_MAP_NODES * sizeof(dnode_t),
                                   MAX_MAP_TEXINFO * sizeof(texinfo_t),
                                   MAX_MAP_FACES * sizeof(dface_t),
                                   MAX_MAP_LIGHTING,
                                   MAX_MAP_CLIPNODES * sizeof(dclipnode_t),
                                   MAX_MAP_LEAFS * sizeof(dleaf_t),
                                   MAX_MAP_MARKSURFACES * 2,
                                   MAX_MAP_EDGES * sizeof(dedge_t),
                                   MAX_MAP_SURFEDGES * 4,
                                   MAX_MAP_MODELS * sizeof(dmodel_t)};
        for (int i = 0; i < 15; ++i)
            if (size_t(header.lumps[i].filelen) > limits[i])
                throw std::runtime_error("Map exceeds the integrated QRAD's BSP30 limits");
        digest = lm::sha256(this->bytes.data(), this->bytes.size());
        const auto &ent = header.lumps[0];
        std::string text(reinterpret_cast<const char *>(this->bytes.data() + ent.fileofs), ent.filelen);
        std::regex blocks(R"(\{([^}]*)\})"), pairs("\"([^\"\\n]*)\"\\s*\"([^\"\\n]*)\"");
        for (auto it = std::sregex_iterator(text.begin(), text.end(), blocks); it != std::sregex_iterator();
             ++it) {
            std::string body = (*it)[1];
            Entity entity;
            for (auto kv = std::sregex_iterator(body.begin(), body.end(), pairs);
                 kv != std::sregex_iterator(); ++kv)
                entity[(*kv)[1]] = (*kv)[2];
            entities.push_back(std::move(entity));
        }
        auto models = lump<dmodel_t>(14);
        auto vertices = lump<dvertex_t>(3);
        auto edges = lump<dedge_t>(12);
        auto surfedges = lump<int32_t>(13);
        auto info = lump<texinfo_t>(6);
        auto disk_faces = lump<dface_t>(7);
        auto planes = lump<dplane_t>(1);
        if (models.empty() || disk_faces.empty())
            throw std::runtime_error("BSP contains no usable world geometry");
        const auto &tl = header.lumps[2];
        int count = value_at<int32_t>(this->bytes, tl.fileofs);
        if (count < 0 || count > MAX_MAP_TEXTURES || size_t(4 + count * 4) > size_t(tl.filelen))
            throw std::runtime_error("Invalid texture table");
        for (int i = 0; i < count; ++i) {
            int offset = value_at<int32_t>(this->bytes, tl.fileofs + 4 + i * 4);
            if (offset < 0) {
                textures.emplace_back("<missing>");
                continue;
            }
            if (uint64_t(offset) + sizeof(miptex_t) > uint64_t(tl.filelen))
                throw std::runtime_error("Invalid texture offset");
            auto texture = value_at<miptex_t>(this->bytes, tl.fileofs + offset);
            std::string name(texture.name, strnlen(texture.name, 16));
            if (name.empty() ||
                std::any_of(name.begin(), name.end(), [](unsigned char c) { return std::isspace(c); }))
                throw std::runtime_error("Invalid texture name");
            textures.push_back(lower(name));
        }
        std::vector<double> minlight(disk_faces.size());
        for (size_t i = 0; i < models.size(); ++i) {
            const auto &model = models[i];
            if (model.firstface < 0 || model.numfaces < 0 ||
                size_t(model.firstface) + model.numfaces > disk_faces.size())
                throw std::runtime_error("Invalid model face range");
            const Entity *owner = entities.empty() ? nullptr : &entities.front();
            for (const auto &entity : entities)
                if (entity.contains("model") && entity.at("model") == "*" + std::to_string(i)) {
                    owner = &entity;
                    break;
                }
            double value = owner && owner->contains("_minlight")
                               ? double(float(std::stod(owner->at("_minlight"))) * 128.0f)
                               : 0;
            if (!std::isfinite(value) || value < 0)
                throw std::runtime_error("Invalid model minimum light");
            std::fill(minlight.begin() + model.firstface, minlight.begin() + model.firstface + model.numfaces,
                      value);
        }
        std::vector<int> boundaries{header.lumps[8].filelen};
        for (auto f : disk_faces)
            if (f.lightofs >= 0)
                boundaries.push_back(f.lightofs);
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
        std::vector<std::array<double, 3>> samples;
        for (size_t fid = 0; fid < disk_faces.size(); ++fid) {
            Face face;
            face.disk = disk_faces[fid];
            face.minlight = minlight[fid];
            const auto &f = face.disk;
            if (f.texinfo < 0 || size_t(f.texinfo) >= info.size() || f.planenum < 0 ||
                size_t(f.planenum) >= planes.size() || f.numedges < 3 || f.firstedge < 0 ||
                size_t(f.firstedge) + f.numedges > surfedges.size())
                throw std::runtime_error("Invalid face geometry");
            const auto &ti = info[f.texinfo];
            if (ti.miptex < 0 || size_t(ti.miptex) >= textures.size())
                throw std::runtime_error("Invalid face texture");
            face.texture = textures[ti.miptex];
            std::array<float, 2> low{INFINITY, INFINITY}, high{-INFINITY, -INFINITY};
            std::array<float, 2> low_qrad{INFINITY, INFINITY}, high_qrad{-INFINITY, -INFINITY};
            std::array<double, 2> low64{INFINITY, INFINITY}, high64{-INFINITY, -INFINITY};
            for (int k = 0; k < f.numedges; ++k) {
                int64_t index = surfedges[f.firstedge + k];
                size_t edge_index = index < 0 ? -index : index;
                if (edge_index >= edges.size())
                    throw std::runtime_error("Invalid edge reference");
                auto vertex_index = edges[edge_index].v[index < 0 ? 1 : 0];
                if (vertex_index >= vertices.size())
                    throw std::runtime_error("Invalid vertex reference");
                const auto &v = vertices[vertex_index];
                for (int c = 0; c < 2; ++c) {
                    float uv = ((v.point[0] * ti.vecs[c][0] + v.point[1] * ti.vecs[c][1]) +
                                v.point[2] * ti.vecs[c][2]) +
                               ti.vecs[c][3];
                    double uv64 = double(v.point[0]) * ti.vecs[c][0] + double(v.point[1]) * ti.vecs[c][1] +
                                  double(v.point[2]) * ti.vecs[c][2] + ti.vecs[c][3];
                    // Linux QRAD evaluates the dot product with x87 intermediates,
                    // then stores it in vec_t (float) before taking extrema.
                    // This can differ from both float-only and double extents.
                    float uv_qrad =
                        float(static_cast<long double>(v.point[0]) * ti.vecs[c][0] +
                              static_cast<long double>(v.point[1]) * ti.vecs[c][1] +
                              static_cast<long double>(v.point[2]) * ti.vecs[c][2] + ti.vecs[c][3]);
                    if (!std::isfinite(uv64) || std::abs(uv64) > 1e8)
                        throw std::runtime_error("Invalid texture coordinates");
                    low[c] = std::min(low[c], uv);
                    high[c] = std::max(high[c], uv);
                    low_qrad[c] = std::min(low_qrad[c], uv_qrad);
                    high_qrad[c] = std::max(high_qrad[c], uv_qrad);
                    low64[c] = std::min(low64[c], uv64);
                    high64[c] = std::max(high64[c], uv64);
                }
            }
            face.u = lm::checked_cast<int>(std::floor(low[0] / 16));
            face.v = lm::checked_cast<int>(std::floor(low[1] / 16));
            face.width = lm::checked_cast<int>(std::ceil(high[0] / 16)) - face.u + 1;
            face.height = lm::checked_cast<int>(std::ceil(high[1] / 16)) - face.v + 1;
            if (f.lightofs >= 0) {
                int nstyles = 0;
                while (nstyles < 4 && f.styles[nstyles] != 255)
                    ++nstyles;
                auto end = std::upper_bound(boundaries.begin(), boundaries.end(), f.lightofs);
                if (!nstyles || end == boundaries.end())
                    throw std::runtime_error("Invalid lightmap offset/style");
                size_t span = *end - f.lightofs;
                if (uint64_t(face.width) * face.height * nstyles * 3 != span) {
                    int original_u = face.u, original_v = face.v, original_width = face.width,
                        original_height = face.height;
                    face.u = lm::checked_cast<int>(std::floor(low_qrad[0] / 16));
                    face.v = lm::checked_cast<int>(std::floor(low_qrad[1] / 16));
                    face.width = lm::checked_cast<int>(std::ceil(high_qrad[0] / 16)) - face.u + 1;
                    face.height = lm::checked_cast<int>(std::ceil(high_qrad[1] / 16)) - face.v + 1;
                    if (uint64_t(face.width) * face.height * nstyles * 3 != span) {
                        face.u = lm::checked_cast<int>(std::floor(low64[0] / 16));
                        face.v = lm::checked_cast<int>(std::floor(low64[1] / 16));
                        face.width = lm::checked_cast<int>(std::ceil(high64[0] / 16)) - face.u + 1;
                        face.height = lm::checked_cast<int>(std::ceil(high64[1] / 16)) - face.v + 1;
                    }
                    if (uint64_t(face.width) * face.height * nstyles * 3 != span) {
                        face.u = original_u;
                        face.v = original_v;
                        face.width = original_width;
                        face.height = original_height;
                    }
                    if (uint64_t(f.lightofs) + uint64_t(face.width) * face.height * nstyles * 3 >
                        uint64_t(header.lumps[8].filelen))
                        throw std::runtime_error("Cannot establish original lightmap dimensions");
                }
                int size = face.width * face.height;
                for (int style = 0; style < nstyles; ++style) {
                    face.offsets[f.styles[style]] = {lm::checked_cast<int>(samples.size()), size};
                    size_t offset = size_t(header.lumps[8].fileofs) + f.lightofs + style * size * 3;
                    if (offset + size * 3 > this->bytes.size())
                        throw std::runtime_error("Truncated lightmap");
                    for (int i = 0; i < size; ++i) {
                        samples.push_back({double(this->bytes[offset + i * 3]),
                                           double(this->bytes[offset + i * 3 + 1]),
                                           double(this->bytes[offset + i * 3 + 2])});
                        face_ids.push_back(lm::checked_cast<int>(fid));
                        styles.push_back(f.styles[style]);
                    }
                }
            }
            faces.push_back(std::move(face));
        }
        if (samples.empty())
            throw std::runtime_error("BSP contains no baked lightmaps");
        // VHLT descendants can share lightmap offsets and append a safety buffer.
        // Validate each face's range above instead of requiring a gapless lighting lump.
        target = Eigen::Map<const RGB>(samples.front().data(), samples.size(), 3);
        floor.resize(samples.size());
        for (size_t i = 0; i < samples.size(); ++i)
            floor[i] = faces[face_ids[i]].minlight;
    }
    Bytes without_lights() const {
        std::string text;
        for (const auto &entity : entities) {
            if (entity.contains("classname") && entity.at("classname").starts_with("light"))
                continue;
            text += "{\n";
            for (const auto &[key, value] : entity)
                text += "\"" + key + "\" \"" + value + "\"\n";
            text += "}\n";
        }
        text.push_back('\0');
        Bytes output(sizeof(dheader_t));
        auto copy = header;
        for (int i = 0; i < 15; ++i) {
            while (output.size() % 4)
                output.push_back(0);
            copy.lumps[i].fileofs = lm::checked_cast<int32_t>(output.size());
            if (i == 0) {
                copy.lumps[i].filelen = lm::checked_cast<int32_t>(text.size());
                output.insert(output.end(), text.begin(), text.end());
            } else
                output.insert(output.end(), bytes.begin() + header.lumps[i].fileofs,
                              bytes.begin() + header.lumps[i].fileofs + header.lumps[i].filelen);
        }
        memcpy(output.data(), &copy, sizeof copy);
        return output;
    }
};
struct Options {
    fs::path bsp, output, profile;
    std::vector<fs::path> wad_dirs;
    std::string backend = "auto";
    std::string compiler = "qrad";
    std::string quality;
    int bounces = -1;
    int qrad_smooth = 45;
    int workers = 0, passes = 3, decimals = 0;
    bool verbose = false, cache = true, batched = true, self_test = false;
    bool frozen_derivatives = true;
    bool detect_only = false;
    bool consensus = true, snap = true;
    bool final_refinement = true, passes_explicit = false;
};
void apply_quality(Options &options) {
    if (!options.passes_explicit)
        options.passes = options.quality == "fast" ? 1 : options.quality == "balanced" ? 2 : 3;
    options.final_refinement = options.quality != "fast";
    options.consensus = options.consensus && options.quality == "full";
    options.snap = options.snap && options.quality != "fast";
}
std::string choose_quality(const fs::path &map) {
    std::cerr << "Loaded " << map.filename().string() << ".\n\n"
              << "Choose recovery quality:\n"
              << "  1) Fast      - Quicker, lower accuracy\n"
              << "  2) Medium    - Balanced accuracy \n"
              << "  3) Full      - Highest accuracy, may take longer\n\n";
    for (;;) {
        check_cancelled();
        std::cerr << "Enter your choice: " << std::flush;
        std::string choice;
        if (!std::getline(std::cin, choice)) {
            check_cancelled();
            std::cerr << '\n';
            return "balanced";
        }
        const auto first = choice.find_first_not_of(" \t\r");
        choice = first == std::string::npos ? "" :
                 lower(choice.substr(first, choice.find_last_not_of(" \t\r") - first + 1));
        if (choice == "1" || choice == "fast") return "fast";
        if (choice.empty() || choice == "2" || choice == "balanced") return "balanced";
        if (choice == "3" || choice == "full") return "full";
        std::cerr << "Please enter 1, 2 or 3.\n";
    }
}
struct Fingerprint {
    std::vector<std::string> preferred;
    std::vector<std::string> evidence;
};
Fingerprint fingerprint(const BSP &bsp) {
    Fingerprint result;
    for (const auto &entity : bsp.entities) {
        auto field = entity.find("compiler");
        if (field == entity.end())
            continue;
        auto stamp = lower(field->second);
        result.evidence.push_back("CSG compiler stamp: " + field->second + "; the RAD stage can differ");
        if (stamp.find("sdhlt") != std::string::npos || stamp.find("seedee") != std::string::npos)
            result.preferred = {"sdhlt"};
        else if (stamp.find("schlt") != std::string::npos || stamp.find("sven") != std::string::npos)
            result.preferred = {"schlt"};
        else if (stamp.find("vl") != std::string::npos)
            result.preferred = {"vhlt", "sdhlt"};
        else if (stamp.find("zhlt") != std::string::npos)
            result.preferred = {"zhlt"};
        else if (stamp.find("qrad") != std::string::npos || stamp.find("valve") != std::string::npos)
            result.preferred = {"qrad"};
        else if (stamp.find("paranoia") != std::string::npos || stamp.find("p2rad") != std::string::npos)
            result.preferred = {"p2rad"};
        break;
    }
    uint64_t used_end = 0, extended_end = 0;
    for (const auto &face : bsp.faces)
        if (face.disk.lightofs >= 0) {
            uint64_t styles = face.offsets.size();
            used_end = std::max(used_end, uint64_t(face.disk.lightofs) +
                                              uint64_t(face.width) * face.height * styles * 3);
            extended_end = std::max(extended_end, uint64_t(face.disk.lightofs) + 17 * 17 * styles * 3);
        }
    auto &lighting = bsp.header.lumps[8];
    if (extended_end == uint64_t(lighting.filelen) && extended_end > used_end &&
        std::all_of(bsp.bytes.begin() + lighting.fileofs + used_end,
                    bsp.bytes.begin() + lighting.fileofs + lighting.filelen, [](auto v) { return v == 0; })) {
        result.evidence.push_back("VHLT-style extended lightmap safety buffer");
        if (result.preferred.empty())
            result.preferred = {"vhlt", "sdhlt"};
    }
    for (const auto &name : bsp.textures)
        if (name.size() > 5 && name.substr(1, 4) == "_rad") {
            result.evidence.push_back("Embedded lightmap texture from the VHLT family");
            if (result.preferred.empty())
                result.preferred = {"vhlt", "sdhlt"};
            break;
        }
    if (result.evidence.empty())
        result.evidence.push_back("No distinctive compiler marker; BSP30 alone is shared by these compilers");
    return result;
}
std::map<std::string, std::string> environment() {
    auto result = lm::process_environment();
    for (auto it = result.begin(); it != result.end();) {
        auto key = upper(it->first);
        if (key.starts_with("LM_") && key != "LM_AUDIT_GPU_DIRECT" && key != "LM_AUDIT_GPU_TRACE" &&
            key != "LM_AUDIT_GPU_INDIRECT" && key != "LM_AUDIT_GPU_GATHER")
            it = result.erase(it);
        else ++it;
    }
    return result;
}
struct Child : lm::Child {
    Child() : lm::Child(::check_cancelled) {}
};
unsigned worker_count(const Options &options, size_t samples) {
    auto [available, memory] = lm::available_resources();
    // Include the bounded direct cache, its atomic-write payload and the
    // compiler's transfer/sample storage when selecting parallelism.
    uint64_t per_worker = (1536ULL << 20) + samples * 512;
    unsigned memory_limit = unsigned(std::max(uint64_t(1), (memory / 2) / per_worker));
    if (options.workers)
        return std::max(1u, std::min(unsigned(options.workers), memory_limit));
    return std::max(1u, std::min({available, memory_limit, 64u}));
}
struct Probe {
    RGB raw, direct, bytes;
    std::vector<std::array<int, 2>> patches;
    std::vector<CompilerBoundary> boundaries;
    std::string layout;
    Bytes membership;
    bool complete = true;
};
using Emissions = std::map<std::string, std::array<float, 3>>;
class Engine {
    BSP &bsp;
    Options options;
    Profile &profile;
    fs::path work, executable, socket;
    std::unique_ptr<FD> source, entityless, gpu_log, grids;
    Child gpu;
    std::atomic<uint64_t> sequence{0};
    std::mutex probe_mutex;
    bool encoding_ready = false;
    std::unordered_map<std::string, std::shared_ptr<Probe>> recent;

  public:
    unsigned workers;
    // Rolling wall time for completed fitting passes on this compiler/backend.
    double seconds_per_refinement = 0;
    Engine(BSP &bsp, const Options &options, Profile &profile, const fs::path &work)
        : bsp(bsp), options(options), profile(profile), work(work) {
        executable = lm::executable_path();
        workers = worker_count(options, bsp.target.rows());
        source = memory_file(work, "radbruter-input", bsp.bytes);
        entityless = memory_file(work, "radbruter-no-entities", bsp.without_lights());
        std::vector<lm::LightmapGrid> original_grids(bsp.faces.size());
        for (size_t i = 0; i < bsp.faces.size(); ++i) {
            const auto &face = bsp.faces[i];
            if (!face.offsets.empty()) original_grids[i] = {face.u, face.v, face.width, face.height};
        }
        std::array<uint32_t, 2> grid_header{lm::grid_magic, lm::checked_cast<uint32_t>(original_grids.size())};
        Bytes grid_data(sizeof grid_header + original_grids.size() * sizeof(lm::LightmapGrid));
        memcpy(grid_data.data(), grid_header.data(), sizeof grid_header);
        memcpy(grid_data.data() + sizeof grid_header, original_grids.data(),
               original_grids.size() * sizeof(lm::LightmapGrid));
        grids = memory_file(work, "radbruter-original-grids", grid_data);
        if (options.compiler != "qrad" && options.backend == "cuda")
            throw std::runtime_error(
                "CUDA gathering currently supports QRAD only; use --backend auto or cpu for " +
                options.compiler);
        if (options.compiler != "qrad" &&
            !fs::is_regular_file(executable.parent_path() / "compilers" / ("radbruter-" + options.compiler + lm::executable_suffix)))
            throw std::runtime_error("Missing native compiler: " + options.compiler);
        if (options.backend != "cpu" && options.compiler == "qrad") {
            Timer timer(profile, "gpu_startup");
            socket = work / "gpu.sock";
            gpu_log = memory_file(work, "radbruter-gpu-log");
            gpu.start({executable.string(), "--gpu-service", socket.string()}, environment(),
                      {{gpu_log->fd, 1}, {gpu_log->fd, 2}});
            auto start = Clock::now();
            while (!fs::exists(socket)) {
                check_cancelled();
                if (gpu.exited()) {
                    gpu.stop();
                    auto raw = read_fd(gpu_log->fd);
                    std::string error(raw.begin(), raw.end());
                    socket.clear();
                    if (options.backend == "cuda")
                        throw std::runtime_error("CUDA initialization failed: " + error);
                    break;
                }
                if (seconds(start) > 60) {
                    gpu.stop();
                    socket.clear();
                    if (options.backend == "cuda") throw std::runtime_error("CUDA initialization timed out");
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        if (options.verbose)
            profile.note("64-bit native recovery; " + std::to_string(workers) + " threads; " + backend() +
                         " radiosity; " + (options.cache ? "temporary cache for this recovery only"
                                                        : "lighting caches disabled"));
    }
    std::string backend() const { return socket.empty() ? "cpu" : "cuda"; }
    std::string compiler() const { return options.compiler; }
    bool cache_enabled() const { return options.cache; }
    bool freeze_derivatives() const { return options.frozen_derivatives && options.compiler == "qrad"; }
    template <class Function> void parallel(size_t count, Function fn) {
        if (!count)
            return;
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::exception_ptr error;
        std::mutex errors;
        std::vector<std::thread> threads;
        for (unsigned t = 0; t < std::min(size_t(workers), count); ++t)
            threads.emplace_back([&] {
                batch_thread_share = std::max(1u, workers / unsigned(std::min(size_t(workers), count)));
                try {
                    for (;;) {
                        check_cancelled();
                        if (failed)
                            break;
                        size_t index = next++;
                        if (index >= count)
                            break;
                        fn(index);
                    }
                } catch (...) {
                    std::lock_guard lock(errors);
                    if (!error)
                        error = std::current_exception();
                    failed = true;
                }
            });
        for (auto &thread : threads)
            thread.join();
        if (error)
            std::rethrow_exception(error);
    }
    std::shared_ptr<Probe> bake(const Emissions &values, bool entities = true, bool cpu = false,
                                const std::string *formatted = nullptr, const Bytes *membership = nullptr) {
        check_cancelled();
        std::ostringstream rad;
        if (formatted)
            rad << *formatted;
        else
            for (const auto &[name, value] : values)
                rad << name << ' ' << std::setprecision(9) << value[0] << ' ' << value[1] << ' ' << value[2]
                    << '\n';
        std::string text = rad.str(),
                    signature = text + (entities ? "entities" : "none") + (cpu ? "cpu" : backend());
        if (membership) {
            if (cpu || formatted || options.compiler != "qrad")
                throw std::runtime_error("Only derivative probes may freeze sample membership");
            signature += "fixed-samples:" + lm::sha256(membership->data(), membership->size());
        }
        auto key = lm::sha256(signature.data(), signature.size());
        if (options.cache && !formatted && !cpu) {
            std::lock_guard lock(probe_mutex);
            auto it = recent.find(key);
            if (it != recent.end()) {
                ++profile.probe_cache_hits;
                return it->second;
            }
        }
        uint64_t id = sequence++;
        auto rad_path = work / ("probe" + std::to_string(id) + ".rad");
        struct Remove {
            fs::path path;
            ~Remove() {
                std::error_code ec;
                fs::remove(path, ec);
            }
        } cleanup{rad_path};
        {
            Timer timer(profile, "rad_io");
            atomic_write(rad_path, text);
            std::lock_guard lock(profile.mutex);
            profile.disk_bytes_written += text.size();
        }
        auto floats = memory_file(work, "radbruter-floats"), result = memory_file(work, "radbruter-result"),
             log = memory_file(work, "radbruter-log");
        auto env = environment();
        env.erase("LM_CACHE_DIR");
        env.erase("LM_MEMBERSHIP_FD");
        env["LM_BSP_INPUT"] = lm::input_descriptor_path();
        env["LM_NO_BSP_WRITE"] = "1";
        env["LM_FLOAT_FD"] = "121";
        env["LM_RESULT_FD"] = "122";
        env["LM_GRID_FD"] = "124";
        env["LM_FACE_THREADS"] = std::to_string(batch_thread_share ? batch_thread_share : workers);
        if (cpu)
            env["LM_REFERENCE"] = "1";
        if (options.cache && !cpu)
            env["LM_CACHE_DIR"] = (work / "cache").string();
        if (!cpu && !socket.empty())
            env["LM_GPU_SOCKET"] = socket.string();
        std::unique_ptr<FD> sampling;
        if (membership) {
            sampling = memory_file(work, "radbruter-derivative-samples", *membership);
            env["LM_MEMBERSHIP_FD"] = "123";
        }
        Child child;
        std::vector<std::string> command{executable.string(),
                                         "--qrad-worker",
                                         "-extra",
                                         "-bounce",
                                         "1",
                                         "-scale",
                                         "1.1",
                                         "-gamma",
                                         ".5",
                                         "-smooth",
                                         std::to_string(options.qrad_smooth),
                                         "-chop",
                                         "64",
                                         "-maxchop",
                                         "64",
                                         "-lights",
                                         rad_path.string(),
                                         (work / ("probe" + std::to_string(id) + ".bsp")).string()};
        if (options.compiler != "qrad") {
            command = {(executable.parent_path() / "compilers" / ("radbruter-" + options.compiler + lm::executable_suffix)).string(),
                       "-nolog",
                       "-threads",
                       std::to_string(cpu ? 1 : (batch_thread_share ? batch_thread_share : workers)),
                       "-extra",
                       "-lights",
                       rad_path.string()};
            if (options.bounces >= 0) {
                command.push_back("-bounce");
                command.push_back(std::to_string(options.bounces));
            }
            for (const auto &directory : options.wad_dirs) {
                command.push_back("-waddir");
                command.push_back(directory.string());
            }
            command.push_back((work / ("probe" + std::to_string(id) + ".bsp")).string());
        } else if (options.bounces >= 0) {
            command[4] = std::to_string(options.bounces);
        }
        auto start = Clock::now();
        std::vector<std::pair<int, int>> descriptors{{entities ? source->fd : entityless->fd, 120},
            {floats->fd, 121}, {result->fd, 122}, {grids->fd, 124}, {log->fd, 1}, {log->fd, 2}};
        if (sampling) descriptors.emplace_back(sampling->fd, 123);
        child.start(command, env, descriptors);
        int code;
        auto rss = child.wait(code);
        profile.add("compiler_wall_sum", seconds(start));
        if (code) {
            auto data = read_fd(log->fd);
            std::string error(data.begin(), data.end());
            if (!options.verbose)
                error = worker_error_summary(error);
            else if (error.size() > 3000)
                error.erase(0, error.size() - 3000);
            throw std::runtime_error("Lighting compiler failed (" + std::to_string(code) + "): " + error);
        }
        Timer timer(profile, "probe_ipc_and_decode");
        auto binary = read_fd(result->fd), dump = read_fd(floats->fd);
        Cursor cursor{binary};
        auto protocol = cursor.get<uint32_t>();
        if (protocol != 0x4c4d5239 && protocol != 0x4c4d523a)
            throw std::runtime_error("Bad native worker protocol");
        auto measured = cursor.get<lm_stats>();
        auto probe = std::make_shared<Probe>();
        auto layout = cursor.take(64);
        probe->layout.assign(reinterpret_cast<const char *>(layout), 64);
        auto nfaces = cursor.get<uint32_t>(), lighting_size = cursor.get<uint32_t>(),
             npatches = cursor.get<uint32_t>();
        if (nfaces != bsp.faces.size() || lighting_size > MAX_MAP_LIGHTING || npatches > 65536)
            throw std::runtime_error("Invalid native worker result");
        std::vector<dface_t> faces(nfaces);
        memcpy(faces.data(), cursor.take(nfaces * sizeof(dface_t)), nfaces * sizeof(dface_t));
        const auto *lighting = cursor.take(lighting_size);
        probe->patches.resize(npatches);
        memcpy(probe->patches.data(), cursor.take(npatches * 8), npatches * 8);
        if (protocol == 0x4c4d5239) {
            auto start = cursor.pos;
            cursor.take(64);
            if (cursor.get<uint32_t>() != nfaces) throw std::runtime_error("Invalid sampling faces");
            for (uint32_t i = 0; i < nfaces; ++i) {
                auto count = cursor.get<uint32_t>();
                if (count > 65536) throw std::runtime_error("Invalid sampling mask");
                const auto *mask = cursor.take(count);
                if (std::any_of(mask, mask + count, [](auto v) { return v > 1; }))
                    throw std::runtime_error("Invalid sampling membership");
            }
            probe->membership.assign(binary.begin() + start, binary.begin() + cursor.pos);
        }
        if (protocol == 0x4c4d523a) {
            auto encoding = cursor.get<CompilerEncoding>();
            std::vector<float> floors(nfaces);
            memcpy(floors.data(), cursor.take(nfaces * sizeof(float)), nfaces * sizeof(float));
            std::lock_guard lock(probe_mutex);
            if (!encoding_ready) {
                bsp.encoding = encoding;
                bsp.native_encoding = true;
                for (size_t i = 0; i < bsp.face_ids.size(); ++i)
                    bsp.floor[i] = floors[bsp.face_ids[i]];
                encoding_ready = true;
            } else if (memcmp(&bsp.encoding, &encoding, sizeof encoding)) {
                throw std::runtime_error("Compiler encoding settings changed between probes");
            }
            auto count = cursor.get<uint32_t>();
            if (count > nfaces)
                throw std::runtime_error("Invalid compiler casting boundaries");
            probe->boundaries.resize(count);
            memcpy(probe->boundaries.data(), cursor.take(count * sizeof(CompilerBoundary)),
                   count * sizeof(CompilerBoundary));
            for (const auto &boundary : probe->boundaries)
                if (boundary.face < 0 || uint32_t(boundary.face) >= nfaces)
                    throw std::runtime_error("Invalid casting face");
        }
        if (cursor.pos != binary.size())
            throw std::runtime_error("Trailing native worker data");
        probe->raw = RGB::Zero(bsp.target.rows(), 3);
        probe->direct = probe->raw;
        probe->bytes = probe->raw;
        Cursor raw{dump};
        size_t accounted = 0;
        while (raw.pos < dump.size()) {
            auto fid = raw.get<int32_t>(), style = raw.get<int32_t>(), count = raw.get<int32_t>();
            if (fid < 0 || size_t(fid) >= bsp.faces.size() || count < 0 || count > 65536)
                throw std::runtime_error("Invalid float lightmap record");
            auto p = raw.take(size_t(count) * 24);
            auto original = bsp.faces[fid].offsets.find(style);
            if (original == bsp.faces[fid].offsets.end()) {
                probe->complete = false;
                continue;
            }
            auto [offset, expected] = original->second;
            if (expected != count)
                throw std::runtime_error("Native lightmap grid differs from original BSP at face " +
                                         std::to_string(fid) + ": expected " + std::to_string(expected) +
                                         ", received " + std::to_string(count));
            int slot = 0;
            while (slot < 4 && faces[fid].styles[slot] != style)
                ++slot;
            if (slot == 4 || faces[fid].lightofs < 0 ||
                uint64_t(faces[fid].lightofs) + uint64_t(slot + 1) * count * 3 > lighting_size)
                throw std::runtime_error("Invalid rebuilt lighting offset");
            for (int i = 0; i < count; ++i)
                for (int c = 0; c < 3; ++c) {
                    float direct, indirect;
                    memcpy(&direct, p + i * 24 + c * 4, 4);
                    memcpy(&indirect, p + i * 24 + 12 + c * 4, 4);
                    probe->direct(offset + i, c) = direct;
                    probe->raw(offset + i, c) = 2.0 * direct + indirect;
                    probe->bytes(offset + i, c) =
                        lighting[faces[fid].lightofs + slot * count * 3 + i * 3 + c];
                }
            accounted += count;
        }
        if (accounted != size_t(bsp.target.rows()))
            probe->complete = false;
        for (size_t i = 0; i < faces.size(); ++i)
            if (memcmp(&faces[i], &bsp.faces[i].disk, 12))
                throw std::runtime_error("Compiler changed face geometry");
        profile.sample(measured, rss, binary.size() + dump.size());
        if (!formatted && !cpu && options.cache) {
            std::lock_guard lock(probe_mutex);
            // A small bounded cache only avoids exact duplicate evaluations.
            // Derivative arrays belong to the solver, not an accumulating history.
            if (recent.size() >= std::max(2u, workers))
                recent.erase(recent.begin());
            recent[key] = probe;
        }
        return probe;
    }
};
bool training_sample(const BSP &bsp, int i) {
    return i == 0 || bsp.face_ids[i] != bsp.face_ids[i - 1] || bsp.styles[i] != bsp.styles[i - 1] ||
           (i + bsp.face_ids[i]) % 5 != 0;
}
std::vector<int> training_rows(const BSP &bsp, bool all = false) {
    std::vector<int> rows;
    for (int i = 0; i < bsp.target.rows(); ++i)
        if (bsp.styles[i] == 0 && (all || training_sample(bsp, i)))
            rows.push_back(i);
    if (rows.empty())
        throw std::runtime_error("Map has too few independent style-0 samples for fitting");
    return rows;
}
struct Quality {
    double mae = 0, capped = 0, rmse = 0, exact = 0, within2 = 0, train_mae = 0, train_capped = 0,
           heldout_mae = 0, style0_exact = 0, style0_mae = 0;
};
Quality quality(const BSP &bsp, const Probe &probe) {
    Quality result;
    size_t count = 0, train = 0, heldout = 0;
    for (int i = 0; i < bsp.target.rows(); ++i)
        for (int c = 0; c < 3; ++c) {
            double error = std::abs(probe.bytes(i, c) - bsp.target(i, c));
            ++count;
            result.mae += error;
            result.capped += std::min(error, 3.0);
            result.rmse += error * error;
            result.exact += error == 0;
            result.within2 += error <= 2;
            if (bsp.styles[i] == 0) {
                result.style0_exact += error == 0;
                result.style0_mae += error;
                if (training_sample(bsp, i)) {
                    ++train;
                    result.train_mae += error;
                    result.train_capped += std::min(error, 3.0);
                } else {
                    ++heldout;
                    result.heldout_mae += error;
                }
            }
        }
    result.mae /= count;
    result.capped /= count;
    result.rmse = std::sqrt(result.rmse / count);
    result.exact /= count;
    result.within2 /= count;
    result.train_mae /= std::max(size_t(1), train);
    result.train_capped /= std::max(size_t(1), train);
    result.heldout_mae /= std::max(size_t(1), heldout);
    result.style0_exact /= std::max(size_t(1), train + heldout);
    result.style0_mae /= std::max(size_t(1), train + heldout);
    return result;
}
Emissions emissions(const std::vector<std::string> &names, const RGB &values) {
    Emissions result;
    for (size_t i = 0; i < names.size(); ++i)
        result[names[i]] = {float(values(i, 0)), float(values(i, 1)), float(values(i, 2))};
    return result;
}
Vector nnls(const Matrix &a, const Vector &b) {
    int n = lm::checked_cast<int>(a.cols());
    Vector x = Vector::Zero(n);
    if (!n)
        return x;
    Matrix gram = a.transpose() * a;
    Vector rhs = a.transpose() * b;
    std::vector<bool> active(n, false);
    const double tolerance = 1e-11 * std::max(1.0, rhs.cwiseAbs().maxCoeff());
    for (int iteration = 0; iteration < n * 30; ++iteration) {
        Vector gradient = rhs - gram * x;
        int best = -1;
        double largest = tolerance;
        for (int j = 0; j < n; ++j)
            if (!active[j] && gradient[j] > largest) {
                largest = gradient[j];
                best = j;
            }
        if (best < 0)
            break;
        active[best] = true;
        for (int inner = 0; inner < n * 3; ++inner) {
            std::vector<int> indices;
            for (int j = 0; j < n; ++j)
                if (active[j])
                    indices.push_back(j);
            Matrix g(indices.size(), indices.size());
            Vector r(indices.size());
            for (size_t i = 0; i < indices.size(); ++i) {
                r[i] = rhs[indices[i]];
                for (size_t j = 0; j < indices.size(); ++j)
                    g(i, j) = gram(indices[i], indices[j]);
            }
            Vector local = g.completeOrthogonalDecomposition().solve(r), candidate = Vector::Zero(n);
            for (size_t i = 0; i < indices.size(); ++i)
                candidate[indices[i]] = local[i];
            bool positive = true;
            for (auto j : indices)
                if (candidate[j] <= 0)
                    positive = false;
            if (positive) {
                x = candidate;
                break;
            }
            double alpha = 1;
            for (auto j : indices)
                if (candidate[j] <= 0 && x[j] > candidate[j])
                    alpha = std::min(alpha, x[j] / (x[j] - candidate[j]));
            x += alpha * (candidate - x);
            for (auto j : indices)
                if (x[j] <= 1e-12) {
                    x[j] = 0;
                    active[j] = false;
                }
        }
    }
    return x.cwiseMax(0);
}
struct LightingObjective {
    RGB base, target;
    Vector floor;
    std::array<Matrix, 3> matrix;
    std::string objective = "midpoint", robust = "huber";
    double width = .5, margin = .002;
    CompilerEncoding encoding;
    bool native = false;
    double operator()(const Vector &x, Vector &gradient) {
        check_cancelled();
        int rows = lm::checked_cast<int>(base.rows()), parameters = lm::checked_cast<int>(matrix[0].cols());
        RGB raw = base, g(rows, 3);
        for (int c = 0; c < 3; ++c) {
            Vector column(parameters);
            for (int j = 0; j < parameters; ++j)
                column[j] = x[j * 3 + c];
            raw.col(c) += matrix[c] * column;
        }
        double loss = 0;
        for (int i = 0; i < rows; ++i) {
            if (native) {
                double value[3], jac[3][3]{};
                for (int c = 0; c < 3; ++c) {
                    value[c] = std::max(raw(i, c) * encoding.scale[c], std::max(floor[i], 1e-12));
                    jac[c][c] =
                        raw(i, c) * encoding.scale[c] > std::max(floor[i], 1e-12) ? encoding.scale[c] : 0;
                }
                auto normalize = [&](double limit) {
                    int peak = int(std::max_element(value, value + 3) - value);
                    if (limit < 0 || value[peak] <= limit)
                        return;
                    double factor = limit / value[peak], peak_jac[3];
                    std::copy(jac[peak], jac[peak] + 3, peak_jac);
                    for (int c = 0; c < 3; ++c)
                        for (int d = 0; d < 3; ++d)
                            jac[c][d] = factor * (jac[c][d] - value[c] / value[peak] * peak_jac[d]);
                    for (auto &v : value)
                        v *= factor;
                };
                normalize(encoding.preclip);
                for (int c = 0; c < 3; ++c) {
                    double slope = encoding.gamma[c] * std::pow(value[c] / 256, encoding.gamma[c] - 1);
                    value[c] = 256 * std::pow(value[c] / 256, encoding.gamma[c]);
                    for (auto &v : jac[c])
                        v *= slope;
                }
                normalize(encoding.postclip);
                double local[3];
                for (int c = 0; c < 3; ++c) {
                    if (value[c] < encoding.minimum || value[c] > 255) {
                        value[c] = std::clamp(value[c], encoding.minimum, 255.0);
                        std::fill(jac[c], jac[c] + 3, 0);
                    }
                    double target = this->target(i, c) - encoding.rounding;
                    double residual = objective == "interval"
                                          ? std::min(value[c] - target - margin, 0.0) +
                                                std::max(value[c] - target - 1 + margin, 0.0)
                                          : value[c] - target - .5;
                    double z = residual / width;
                    if (robust == "cauchy") {
                        loss += .5 * width * width * std::log1p(z * z);
                        local[c] = residual / (1 + z * z);
                    } else {
                        double r = std::sqrt(1 + z * z);
                        loss += width * width * (r - 1);
                        local[c] = residual / r;
                    }
                }
                for (int d = 0; d < 3; ++d) {
                    g(i, d) = 0;
                    for (int c = 0; c < 3; ++c)
                        g(i, d) += local[c] * jac[c][d] / (rows * 3);
                }
                continue;
            }
            double scaled[3], clipped[3], local[3];
            int peak_channel = 0;
            for (int c = 0; c < 3; ++c) {
                scaled[c] = std::max(raw(i, c) * 1.1, std::max(floor[i], 1e-12));
                if (scaled[c] > scaled[peak_channel])
                    peak_channel = c;
            }
            double peak = scaled[peak_channel], factor = std::min(1.0, 255.0 / peak), weighted = 0;
            for (int c = 0; c < 3; ++c) {
                clipped[c] = scaled[c] * factor;
                double prediction = 16 * std::sqrt(clipped[c]), residual;
                if (objective == "interval")
                    residual = std::min(prediction - target(i, c) - margin, 0.0) +
                               std::max(prediction - target(i, c) - 1 + margin, 0.0);
                else
                    residual = prediction - target(i, c) - .5;
                double z = residual / width, derivative;
                if (robust == "cauchy") {
                    loss += .5 * width * width * std::log1p(z * z);
                    derivative = residual / (1 + z * z);
                } else {
                    double r = std::sqrt(1 + z * z);
                    loss += width * width * (r - 1);
                    derivative = residual / r;
                }
                local[c] = derivative * 8 / std::sqrt(clipped[c]);
                weighted += local[c] * scaled[c];
                g(i, c) = local[c] * factor;
            }
            if (peak > 255)
                g(i, peak_channel) -= factor / peak * weighted;
            for (int c = 0; c < 3; ++c)
                g(i, c) *= raw(i, c) * 1.1 > std::max(floor[i], 1e-12) ? 1.1 / (rows * 3) : 0;
        }
        gradient.resize(parameters * 3);
        for (int c = 0; c < 3; ++c) {
            Vector column = matrix[c].transpose() * g.col(c);
            for (int j = 0; j < parameters; ++j)
                gradient[j * 3 + c] = column[j];
        }
        return loss / (rows * 3);
    }
};
struct SolveInfo {
    int iterations = 0;
    double loss = 0;
    bool converged = false;
};
SolveInfo minimize(LightingObjective &function, Vector &x, const Vector &lo, const Vector &hi, int iterations,
                   Profile &profile) {
    Timer timer(profile, "optimization");
    LBFGSpp::LBFGSBParam<double> parameters;
    parameters.m = 30;
    parameters.epsilon = 2e-9;
    parameters.epsilon_rel = 2e-9;
    parameters.past = 1;
    parameters.delta = 1e-12;
    parameters.max_iterations = iterations;
    parameters.max_linesearch = 40;
    LBFGSpp::LBFGSBSolver<double> solver(parameters);
    SolveInfo info;
    Vector best = x, scratch;
    double best_loss = function(x, scratch);
    if (profile.progress) profile.progress->evaluated();
    auto objective = [&](const Vector &point, Vector &gradient) {
        double loss = function(point, gradient);
        if (profile.progress) profile.progress->evaluated();
        if (!std::isfinite(loss) || !gradient.allFinite())
            throw std::runtime_error("Non-finite optimization objective");
        if (loss < best_loss) {
            best_loss = loss;
            best = point;
        }
        return loss;
    };
    try {
        info.iterations = solver.minimize(objective, x, info.loss, lo, hi);
        info.converged = info.iterations < iterations;
    } catch (const std::exception &e) {
        std::string message = e.what();
        if (message != "the line search step became smaller than the minimum value allowed" &&
            message != "the line search routine is unable to sufficiently decrease the function value" &&
            message != "the moving direction does not decrease the objective function value")
            throw;
        info.iterations = -1;
        std::lock_guard lock(profile.mutex);
        ++profile.optimizer_stalls;
    }
    x = best;
    info.loss = best_loss;
    return info;
}
LightingObjective make_objective(const BSP &bsp, const RGB &base, const std::array<Matrix, 3> &matrices,
                                 bool all = false) {
    auto rows = training_rows(bsp, all);
    LightingObjective function;
    function.encoding = bsp.encoding;
    function.native = bsp.native_encoding;
    function.base.resize(rows.size(), 3);
    function.target.resize(rows.size(), 3);
    function.floor.resize(rows.size());
    for (int c = 0; c < 3; ++c)
        function.matrix[c].resize(rows.size(), matrices[c].cols());
    for (size_t i = 0; i < rows.size(); ++i) {
        function.base.row(i) = base.row(rows[i]);
        function.target.row(i) = bsp.target.row(rows[i]);
        function.floor[i] = bsp.floor[rows[i]];
        for (int c = 0; c < 3; ++c)
            function.matrix[c].row(i) = matrices[c].row(rows[i]);
    }
    return function;
}
double median(std::vector<double> values) {
    if (values.empty())
        return 0;
    auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return values.size() % 2 ? *middle : (*middle + *std::max_element(values.begin(), middle)) * .5;
}
struct Fit {
    std::vector<std::string> names;
    RGB values;
    std::shared_ptr<Probe> probe;
    std::vector<std::string> recipes;
};
using Basis = std::array<Matrix, 3>;
double decoded_target(const BSP &bsp, int row, int channel) {
    if (!bsp.native_encoding)
        return std::pow(bsp.target(row, channel) + .5, 2) / (256 * 1.1);
    double value = bsp.target(row, channel) + .5 - bsp.encoding.rounding;
    return 256 * std::pow(std::max(value, 0.0) / 256, 1 / bsp.encoding.gamma[channel]) /
           bsp.encoding.scale[channel];
}
double sample_ceiling(const BSP &bsp) {
    return bsp.native_encoding && bsp.encoding.postclip >= 0 ? std::min(250.0, bsp.encoding.postclip - 2)
                                                             : 250;
}
RGB coarse_fit(const BSP &bsp, const std::vector<std::string> &names, const Basis &basis, const RGB &baseline,
               bool exclude_floor, Profile &profile) {
    Timer timer(profile, "linear_initialization");
    RGB x = RGB::Zero(names.size(), 3);
    for (int c = 0; c < 3; ++c) {
        std::vector<int> rows;
        for (int i = 0; i < bsp.target.rows(); ++i) {
            double peak = bsp.target.row(i).maxCoeff();
            if (bsp.styles[i] == 0 && training_sample(bsp, i) && peak < sample_ceiling(bsp) &&
                (exclude_floor ? (bsp.native_encoding
                                      ? decoded_target(bsp, i, c) * bsp.encoding.scale[c] > bsp.floor[i] + 1
                                      : bsp.target(i, c) > 16 * std::sqrt(bsp.floor[i]) + 2)
                               : peak > 5))
                rows.push_back(i);
        }
        // A genuinely dark channel can have no samples above the minimum-light
        // floor. Leave its initial emission zero; the full clipped objective
        // still evaluates every channel, including floor and saturated samples.
        if (rows.empty())
            continue;
        Matrix a(rows.size(), names.size());
        Vector y(rows.size()), weights = Vector::Ones(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            a.row(i) = basis[c].row(rows[i]);
            y[i] = decoded_target(bsp, rows[i], c) - baseline(rows[i], c);
        }
        for (int iteration = 0; iteration < 4; ++iteration) {
            Matrix weighted = a.array().colwise() * weights.array();
            Vector target = y.cwiseProduct(weights);
            x.col(c) = nnls(weighted, target);
            Vector residual = a * x.col(c) - y;
            for (int i = 0; i < residual.size(); ++i)
                weights[i] = std::sqrt(1 / std::max(1.0, std::abs(residual[i]) / 3));
        }
    }
    return x;
}
bool residual_emitter(const std::array<std::vector<double>, 3> &samples,
                      const std::array<double, 3> &value) {
    if (samples[0].size() < 10 ||
        !std::all_of(value.begin(), value.end(), [](double v) { return std::isfinite(v); }))
        return false;
    auto strongest = std::max_element(value.begin(), value.end());
    double weakest = *std::min_element(value.begin(), value.end());
    if (weakest > 2) return true;
    // Colored self-emission can have a weak or zero channel. Requiring all
    // three channels above 2 missed c2a4's green water (blue is about 1.76).
    // For these extra candidates require nonnegative channel medians and
    // broad support, so a few bright errors cannot invent an emitter.
    if (weakest < 0 || *strongest <= 2) return false;
    const auto &channel = samples[strongest - value.begin()];
    size_t supported = std::count_if(channel.begin(), channel.end(),
                                    [](double v) { return std::isfinite(v) && v > 2; });
    return supported * 3 >= channel.size() * 2;
}
Fit refine_initial(const BSP &bsp, std::vector<std::string> names, Basis basis, const RGB &base,
                   const RGB &unlit, Engine &engine, Profile &profile) {
    RGB x = coarse_fit(bsp, names, basis, base, false, profile);
    RGB residual(bsp.target.rows(), 3);
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < residual.rows(); ++i)
            residual(i, c) = decoded_target(bsp, i, c);
        residual.col(c) -= base.col(c) + basis[c] * x.col(c);
    }
    residual *= .5;
    std::vector<bool> added(names.size(), false);
    for (const auto &name : bsp.textures) {
        if (std::find(names.begin(), names.end(), name) != names.end())
            continue;
        std::array<std::vector<double>, 3> samples;
        for (int i = 0; i < bsp.target.rows(); ++i)
            if (bsp.styles[i] == 0 && training_sample(bsp, i) && bsp.floor[i] == 0 &&
                bsp.target.row(i).maxCoeff() < sample_ceiling(bsp) &&
                bsp.faces[bsp.face_ids[i]].texture == name)
                for (int c = 0; c < 3; ++c)
                    samples[c].push_back(residual(i, c));
        if (samples[0].size() < 10)
            continue;
        std::array<double, 3> value{median(samples[0]), median(samples[1]), median(samples[2])};
        if (!residual_emitter(samples, value))
            continue;
        int column = lm::checked_cast<int>(names.size());
        names.push_back(name);
        added.push_back(true);
        for (auto &matrix : basis)
            matrix.conservativeResize(Eigen::NoChange, column + 1);
        x.conservativeResize(column + 1, 3);
        if (bsp.native_encoding) {
            auto probe = engine.bake({{name, {1000, 1000, 1000}}}, false);
            for (int c = 0; c < 3; ++c)
                basis[c].col(column) = (probe->raw.col(c) - unlit.col(c)) / 1000;
        } else
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < bsp.target.rows(); ++i)
                    basis[c](i, column) =
                        bsp.styles[i] == 0 && bsp.faces[bsp.face_ids[i]].texture == name ? 2 : 0;
        for (int c = 0; c < 3; ++c)
            x(column, c) = value[c];
    }
    for (size_t j = 0; j < names.size(); ++j) {
        std::vector<double> samples;
        for (int i = 0; i < bsp.target.rows(); ++i)
            if (bsp.styles[i] == 0 && bsp.faces[bsp.face_ids[i]].texture == names[j])
                samples.push_back(bsp.target.row(i).maxCoeff());
        if (!bsp.native_encoding && x.row(j).mean() < 25 && (median(samples) < 250 || added[j]))
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < bsp.target.rows(); ++i)
                    basis[c](i, j) =
                        bsp.styles[i] == 0 && bsp.faces[bsp.face_ids[i]].texture == names[j] ? 2 : 0;
    }
    if (names.empty()) {
        Fit fit;
        fit.values.resize(0, 3);
        fit.probe = engine.bake({});
        return fit;
    }
    x = coarse_fit(bsp, names, basis, base, true, profile);
    Vector scale(names.size());
    for (size_t j = 0; j < names.size(); ++j)
        scale[j] = std::max(30.0, x.row(j).maxCoeff());
    Basis matrix;
    for (int c = 0; c < 3; ++c)
        matrix[c] = basis[c] * scale.asDiagonal();
    auto function = make_objective(bsp, base, matrix);
    function.width = 2;
    Vector initial(names.size() * 3);
    for (size_t j = 0; j < names.size(); ++j)
        for (int c = 0; c < 3; ++c)
            initial[j * 3 + c] = x(j, c) / scale[j];
    minimize(function, initial, Vector::Zero(initial.size()), Vector::Constant(initial.size(), INFINITY), 500,
             profile);
    Fit fit;
    for (size_t j = 0; j < names.size(); ++j) {
        std::array<float, 3> value{float(initial[j * 3] * scale[j]), float(initial[j * 3 + 1] * scale[j]),
                                   float(initial[j * 3 + 2] * scale[j])};
        if (*std::max_element(value.begin(), value.end()) <= 1)
            continue;
        int row = lm::checked_cast<int>(fit.names.size());
        fit.names.push_back(names[j]);
        fit.values.conservativeResize(row + 1, 3);
        for (int c = 0; c < 3; ++c)
            fit.values(row, c) = value[c];
    }
    fit.probe = engine.bake(emissions(fit.names, fit.values));
    return fit;
}
Fit bootstrap(const BSP &bsp, Engine &engine, Profile &profile) {
    Timer timer(profile, "initial_fit");
    const std::vector<std::string> tokens{"~", "scrn", "comp", "crt", "mon", "lite", "lght", "light"};
    std::vector<std::string> names;
    for (const auto &name : bsp.textures) {
        bool used = std::any_of(bsp.faces.begin(), bsp.faces.end(),
                                [&](const Face &f) { return f.texture == name && !f.offsets.empty(); });
        if (used &&
            std::any_of(tokens.begin(), tokens.end(),
                        [&](const std::string &token) { return name.find(token) != std::string::npos; }) &&
            std::find(names.begin(), names.end(), name) == names.end())
            names.push_back(name);
    }
    auto baseline = engine.bake({});
    RGB unlit = bsp.native_encoding ? engine.bake({}, false)->raw : RGB::Zero(bsp.target.rows(), 3);
    Basis basis;
    for (auto &matrix : basis)
        matrix.resize(bsp.target.rows(), names.size());
    engine.parallel(names.size(), [&](size_t j) {
        auto probe = engine.bake({{names[j], {1000, 1000, 1000}}}, false);
        for (int c = 0; c < 3; ++c)
            basis[c].col(j) = (probe->raw.col(c) - unlit.col(c)) / 1000;
    });
    auto fit = refine_initial(bsp, names, basis, baseline->raw, unlit, engine, profile);
    std::vector<std::string> additions;
    for (const auto &name : bsp.textures) {
        if (std::find(names.begin(), names.end(), name) != names.end() ||
            std::find(additions.begin(), additions.end(), name) != additions.end())
            continue;
        std::array<std::vector<double>, 3> difference;
        size_t saturated = 0;
        std::array<size_t, 3> clipped_deficit{};
        for (int i = 0; i < bsp.target.rows(); ++i)
            if (bsp.styles[i] == 0 && bsp.floor[i] == 0 && bsp.faces[bsp.face_ids[i]].texture == name) {
                saturated += bsp.target.row(i).maxCoeff() >= sample_ceiling(bsp);
                for (int c = 0; c < 3; ++c) {
                    difference[c].push_back(bsp.target(i, c) - fit.probe->bytes(i, c));
                    clipped_deficit[c] += bsp.target(i, c) >= sample_ceiling(bsp) &&
                                          difference[c].back() > 20;
                }
            }
        if (difference[0].size() < 9 || double(saturated) / difference[0].size() <= .75)
            continue;
        std::array<double, 3> value{median(difference[0]), median(difference[1]), median(difference[2])};
        // A clipped colored emitter need not have any deficit in its weak
        // channels. Require a large unexplained deficit in the same clipped
        // channel across most of the texture's samples. This only nominates a
        // compiler-probed basis: the full encoder and unsaturated neighboring
        // spill constrain its emission; 255 is never an exact raw-light target.
        bool colored_deficit = std::any_of(clipped_deficit.begin(), clipped_deficit.end(),
            [&](size_t count) { return count * 3 >= difference[0].size() * 2; });
        if ((*std::min_element(value.begin(), value.end()) > 5 &&
             *std::max_element(value.begin(), value.end()) > 20) || colored_deficit)
            additions.push_back(name);
    }
    if (!additions.empty()) {
        int original = lm::checked_cast<int>(names.size());
        names.insert(names.end(), additions.begin(), additions.end());
        for (auto &matrix : basis)
            matrix.conservativeResize(Eigen::NoChange, names.size());
        engine.parallel(additions.size(), [&](size_t j) {
            auto probe = engine.bake({{additions[j], {1000, 1000, 1000}}}, false);
            for (int c = 0; c < 3; ++c)
                basis[c].col(original + j) = (probe->raw.col(c) - unlit.col(c)) / 1000;
        });
        fit = refine_initial(bsp, names, basis, baseline->raw, unlit, engine, profile);
    }
    return fit;
}
std::array<Matrix, 3> derivatives(const BSP &bsp, const Fit &fit, Engine &engine, Profile &profile,
                                  bool batched) {
    Timer timer(profile, "derivatives");
    auto before = profile.probes;
    std::array<Matrix, 3> result;
    for (auto &matrix : result)
        matrix.resize(bsp.target.rows(), fit.names.size());
    using ProbePtr = std::shared_ptr<Probe>;
    std::vector<std::array<double, 3>> steps(fit.names.size());
    std::vector<bool> done(fit.names.size(), false);
    const Bytes *membership = engine.freeze_derivatives() && !fit.probe->membership.empty() ?
                              &fit.probe->membership : nullptr;
    if (membership) ++profile.frozen_derivative_builds;
    auto bake_derivative = [&](const RGB &values) {
        return engine.bake(emissions(fit.names, values), true, false, nullptr, membership);
    };
    for (size_t j = 0; j < fit.names.size(); ++j)
        for (int c = 0; c < 3; ++c)
            steps[j][c] =
                std::max({std::abs(fit.values(j, c)) * .0005, fit.values.row(j).maxCoeff() * 1e-5, .002});
    auto stable = [&](const ProbePtr &probe) { return probe->layout == fit.probe->layout; };
    auto store_column = [&](size_t j, int c, const ProbePtr &plus, const ProbePtr &minus, double hp,
                            double hm, bool good_plus, bool good_minus) {
        if (good_plus && good_minus && hp > 0 && hm > 0)
            result[c].col(j) = (hm * (plus->raw.col(c) - fit.probe->raw.col(c)) / hp +
                                hp * (fit.probe->raw.col(c) - minus->raw.col(c)) / hm) /
                               (hp + hm);
        else if (good_plus && hp > 0)
            result[c].col(j) = (plus->raw.col(c) - fit.probe->raw.col(c)) / hp;
        else
            result[c].col(j) = (fit.probe->raw.col(c) - minus->raw.col(c)) / hm;
    };
    if (batched) {
        std::vector<std::array<ProbePtr, 3>> probes(fit.names.size());
        engine.parallel(fit.names.size() * 3, [&](size_t task) {
            size_t j = task / 3;
            int direction = task % 3;
            RGB values = fit.values;
            Eigen::Index channel;
            fit.values.row(j).maxCoeff(&channel);
            for (int c = 0; c < 3; ++c)
                if (direction != 2 || c == channel)
                    values(j, c) =
                        float(std::max(0.0, fit.values(j, c) + (direction == 1 ? -1 : 1) * steps[j][c]));
            probes[j][direction] = bake_derivative(values);
        });
        for (size_t j = 0; j < fit.names.size(); ++j) {
            auto &plus = probes[j][0], &minus = probes[j][1], &check = probes[j][2];
            bool valid = stable(plus) && stable(minus) && stable(check);
            Eigen::Index channel;
            fit.values.row(j).maxCoeff(&channel);
            for (int c = 0; c < 3; ++c) {
                const auto &expected = c == channel ? plus->raw : fit.probe->raw;
                valid = valid && (check->raw.col(c) - expected.col(c)).cwiseAbs().maxCoeff() == 0;
            }
            if (valid) {
                for (int c = 0; c < 3; ++c) {
                    double hp = float(fit.values(j, c) + steps[j][c]) - fit.values(j, c);
                    double hm = fit.values(j, c) - float(std::max(0.0, fit.values(j, c) - steps[j][c]));
                    store_column(j, c, plus, minus, hp, hm, true, true);
                }
                done[j] = true;
                ++profile.derivative_batched_columns;
            }
        }
    }
    std::vector<std::pair<size_t, int>> pending;
    for (size_t j = 0; j < fit.names.size(); ++j)
        if (!done[j])
        {
            ++profile.derivative_fallback_columns;
            for (int c = 0; c < 3; ++c)
                pending.emplace_back(j, c);
        }
    for (int attempt = 0; attempt < 6 && !pending.empty(); ++attempt) {
        std::vector<std::array<ProbePtr, 2>> probes(pending.size());
        std::vector<std::array<double, 2>> distances(pending.size());
        engine.parallel(pending.size() * 2, [&](size_t task) {
            auto [j, c] = pending[task / 2];
            int direction = task % 2;
            RGB values = fit.values;
            values(j, c) = float(std::max(0.0, fit.values(j, c) + (direction ? -1 : 1) * steps[j][c]));
            double distance = std::abs(values(j, c) - fit.values(j, c));
            distances[task / 2][direction] = distance;
            probes[task / 2][direction] = distance ? bake_derivative(values) : fit.probe;
        });
        std::vector<std::pair<size_t, int>> retry;
        for (size_t k = 0; k < pending.size(); ++k) {
            auto [j, c] = pending[k];
            auto &plus = probes[k][0], &minus = probes[k][1];
            double hp = distances[k][0], hm = distances[k][1];
            bool good_plus = hp > 0 && stable(plus), good_minus = hm > 0 && stable(minus);
            for (int other = 0; other < 3; ++other)
                if (other != c) {
                    good_plus = good_plus &&
                                (plus->raw.col(other) - fit.probe->raw.col(other)).cwiseAbs().maxCoeff() == 0;
                    good_minus =
                        good_minus &&
                        (minus->raw.col(other) - fit.probe->raw.col(other)).cwiseAbs().maxCoeff() == 0;
                }
            // A stable one-sided slope avoids shrinking into float cancellation noise.
            if (good_plus || good_minus)
                store_column(j, c, plus, minus, hp, hm, good_plus, good_minus);
            else {
                steps[j][c] *= .5;
                retry.push_back({j, c});
            }
        }
        pending = std::move(retry);
    }
    if (!pending.empty())
        throw std::runtime_error("No stable derivative for " + fit.names[pending.front().first]);
    profile.derivative_probes += profile.probes - before;
    return result;
}
struct LinearModel {
    RGB values, raw;
    std::vector<std::string> names;
    std::array<Matrix, 3> derivative;
    std::string structure;
    bool matches(const Fit &fit) const {
        if (structure != fit.probe->layout || names != fit.names)
            return false;
        for (int c = 0; c < 3; ++c) {
            Vector predicted = raw.col(c) + derivative[c] * (fit.values.col(c) - values.col(c));
            Vector tolerance = (fit.probe->raw.col(c).cwiseAbs() * 4e-6).array() + 1e-4;
            if (((predicted - fit.probe->raw.col(c)).cwiseAbs().array() > tolerance.array()).any())
                return false;
        }
        return true;
    }
};
Fit refine_precision(const BSP &bsp, const Fit &start, Engine &engine, Profile &profile, double trust,
                     bool batched, LinearModel *reuse = nullptr, bool all = false) {
    auto refinement_start = Clock::now();
    LinearModel local;
    auto &model = reuse ? *reuse : local;
    if (engine.cache_enabled() && !bsp.native_encoding && model.matches(start))
        ++profile.jacobian_reuses;
    else {
        model.derivative = derivatives(bsp, start, engine, profile, batched);
        model.values = start.values;
        model.names = start.names;
        model.raw = start.probe->raw;
        model.structure = start.probe->layout;
        ++profile.jacobian_builds;
    }
    const auto &deriv = model.derivative;
    RGB scale = start.values.cwiseAbs().cwiseMax(1);
    std::array<Matrix, 3> scaled;
    for (int c = 0; c < 3; ++c)
        scaled[c] = deriv[c] * scale.col(c).asDiagonal();
    auto function = make_objective(bsp, start.probe->raw, scaled, all);
    Fit best = start;
    auto base_quality = quality(bsp, *start.probe), best_quality = base_quality;
    const std::array<std::pair<const char *, const char *>, 4> objectives{
        {{"interval", "cauchy"}, {"interval", "huber"}, {"midpoint", "huber"}, {"midpoint", "cauchy"}}};
    std::array<Fit, 4> candidates;
    engine.parallel(candidates.size(), [&](size_t index) {
        auto local_function = function;
        local_function.objective = objectives[index].first;
        local_function.robust = objectives[index].second;
        Vector initial = Vector::Zero(start.values.size()), lo(initial.size()),
               hi = Vector::Constant(initial.size(), trust);
        for (int j = 0; j < start.values.rows(); ++j)
            for (int c = 0; c < 3; ++c)
                lo[j * 3 + c] = std::max(-trust, -start.values(j, c) / scale(j, c));
        minimize(local_function, initial, lo, hi, 1200, profile);
        auto &candidate = candidates[index];
        candidate.names = start.names;
        candidate.values.resize(start.values.rows(), 3);
        for (int j = 0; j < start.values.rows(); ++j)
            for (int c = 0; c < 3; ++c)
                candidate.values(j, c) =
                    float(std::max(0.0, start.values(j, c) + initial[j * 3 + c] * scale(j, c)));
        candidate.probe = engine.bake(emissions(candidate.names, candidate.values));
    });
    for (auto &candidate : candidates) {
        auto measured = quality(bsp, *candidate.probe);
        if (all ? (measured.capped < best_quality.capped ||
                   (measured.capped == best_quality.capped && measured.mae < best_quality.mae))
                : (measured.heldout_mae <= base_quality.heldout_mae * 1.10 + 1e-5 &&
                   (measured.train_capped < best_quality.train_capped ||
                    (measured.train_capped == best_quality.train_capped &&
                     measured.train_mae < best_quality.train_mae)))) {
            best = std::move(candidate);
            best_quality = measured;
        }
    }
    double elapsed = seconds(refinement_start);
    engine.seconds_per_refinement = engine.seconds_per_refinement > 0
                                        ? (engine.seconds_per_refinement + elapsed) / 2 : elapsed;
    return best;
}
Fit explore_thresholds(const BSP &bsp, const Fit &start, Engine &engine, Profile &profile) {
    Timer timer(profile, "threshold_search");
    std::vector<Fit> candidates;
    Fit neutral = start;
    bool changed_neutral = false;
    for (int j = 0; j < start.values.rows(); ++j) {
        double mean = start.values.row(j).mean();
        if (mean <= 0)
            continue;
        std::vector<double> levels;
        if (bsp.native_encoding) {
            for (const auto &boundary : start.probe->boundaries) {
                if (bsp.faces[boundary.face].texture != start.names[j])
                    continue;
                double weighted = 0;
                for (int c = 0; c < 3; ++c)
                    weighted += start.values(j, c) * boundary.weight[c];
                if (weighted > 0) {
                    double level = mean * bsp.encoding.casting_threshold / weighted;
                    if (std::abs(mean - level) <= std::max(1.0, .04 * level) &&
                        std::none_of(levels.begin(), levels.end(),
                                     [&](double other) { return std::abs(other - level) < 1e-4; }))
                        levels.push_back(level);
                }
            }
        } else if (std::abs(mean - 25) <= 1)
            levels.push_back(25);
        for (double level : levels) {
            for (double target : {level - .0001, level + .0001}) {
                Fit candidate = start;
                for (int c = 0; c < 3; ++c)
                    candidate.values(j, c) = float(start.values(j, c) * target / mean);
                candidates.push_back(std::move(candidate));
            }
            if (start.values.row(j).maxCoeff() - start.values.row(j).minCoeff() < .25) {
                Fit candidate = start;
                candidate.values.row(j).setConstant(level);
                candidates.push_back(std::move(candidate));
                neutral.values.row(j).setConstant(level);
                changed_neutral = true;
            }
        }
    }
    if (changed_neutral)
        candidates.push_back(std::move(neutral));
    auto before = profile.probes;
    engine.parallel(candidates.size(), [&](size_t i) {
        candidates[i].probe = engine.bake(emissions(candidates[i].names, candidates[i].values));
    });
    profile.threshold_probes += profile.probes - before;
    Fit best = start;
    auto best_quality = quality(bsp, *start.probe);
    for (auto &candidate : candidates) {
        auto measured = quality(bsp, *candidate.probe);
        if (candidate.probe->complete &&
            (measured.capped < best_quality.capped ||
             (measured.capped == best_quality.capped && measured.mae < best_quality.mae))) {
            best_quality = measured;
            best = std::move(candidate);
        }
    }
    return best;
}
std::string number(double value, int decimals) {
    if (!std::isfinite(value) || value < 0)
        throw std::runtime_error("Invalid recovered light value");
    std::ostringstream out;
    out << std::fixed << std::setprecision(decimals) << value;
    auto text = out.str();
    if (decimals) {
        while (text.back() == '0')
            text.pop_back();
        if (text.back() == '.')
            text.pop_back();
    }
    return text == "-0" ? "0" : text;
}
std::string format_rad(const Fit &fit, int decimals) {
    std::ostringstream text;
    for (size_t j = 0; j < fit.names.size(); ++j) {
        double peak = fit.values.row(j).maxCoeff();
        if (peak <= 1)
            continue;
        text << std::left << std::setw(20) << upper(fit.names[j]);
        if (j < fit.recipes.size() && !fit.recipes[j].empty()) {
            text << fit.recipes[j] << '\n';
            continue;
        }
        if (fit.values(j, 0) == fit.values(j, 1) && fit.values(j, 1) == fit.values(j, 2))
            text << number(peak, decimals);
        else {
            for (int c = 0; c < 3; ++c)
                text << number(fit.values(j, c) * 255 / peak, decimals) << ' ';
            text << number(peak, decimals);
        }
        text << '\n';
    }
    return text.str();
}
// Align the already verified recipe without changing any numeric token.
// Numeric precision is applied before baking and validation in formatted_fit.
std::string format_rad_output(const std::string &rad) {
    std::vector<std::vector<std::string>> entries;
    std::array<size_t, 5> widths{};
    std::istringstream input(rad);
    for (std::string line; std::getline(input, line);) {
        std::istringstream fields(line);
        std::string name;
        if (!(fields >> name))
            continue;
        std::vector<std::string> entry{std::move(name)};
        for (std::string token; fields >> token;) {
            size_t end = 0;
            double value = std::stod(token, &end);
            if (end != token.size() || !std::isfinite(value) || value < 0)
                throw std::runtime_error("Invalid recovered RAD value");
            entry.push_back(std::move(token));
        }
        if (!fields.eof() || (entry.size() != 2 && entry.size() != 4 && entry.size() != 5))
            throw std::runtime_error("Invalid recovered RAD entry");
        for (size_t c = 0; c + 1 < entry.size(); ++c)
            widths[c] = std::max(widths[c], entry[c].size());
        entries.push_back(std::move(entry));
    }
    std::ostringstream output;
    for (const auto &entry : entries) {
        for (size_t c = 0; c < entry.size(); ++c) {
            output << entry[c];
            if (c + 1 < entry.size())
                output << std::string(widths[c] - entry[c].size() + (c ? 1 : 2), ' ');
        }
        output << '\n';
    }
    return output.str();
}
// The uniform-gamma encoders reduce to one RGB-preserving cap in linear
// light. This also includes QRAD/ZHLT, which cap before applying gamma.
struct ByteEncoder {
    CompilerEncoding settings;
    std::array<std::array<double, 257>, 3> cut{};
    double cap = INFINITY;
    int minimum = 0;
    bool supported = true;
    explicit ByteEncoder(const BSP &bsp) : settings(bsp.encoding) {
        if (!bsp.native_encoding)
            settings.postclip = -1;
        for (int c = 0; c < 3; ++c) {
            supported = supported && settings.gamma[c] > 0 && settings.scale[c] > 0 &&
                        settings.gamma[c] == settings.gamma[0];
            for (int q = 1; q < 256; ++q)
                cut[c][q] = 256 * std::pow(std::max(0.0, q - settings.rounding) / 256, 1 / settings.gamma[c]);
            cut[c][256] = INFINITY;
        }
        minimum = std::clamp(int(std::floor(settings.minimum + settings.rounding)), 0, 255);
        if (settings.preclip >= 0)
            cap = settings.preclip;
        if (settings.postclip >= 0)
            cap = std::min(cap, 256 * std::pow(settings.postclip / 256, 1 / settings.gamma[0]));
    }
    std::array<double, 3> linear(const Eigen::RowVector3d &raw, double floor) const {
        std::array<double, 3> result;
        for (int c = 0; c < 3; ++c)
            result[c] = std::max(raw[c] * settings.scale[c], std::max(0.0, floor));
        double peak = *std::max_element(result.begin(), result.end());
        if (peak > cap)
            for (auto &value : result)
                value *= cap / peak;
        return result;
    }
    int byte(double linear, int c) const {
        return std::max(minimum,
                        int(std::upper_bound(cut[c].begin(), cut[c].end(), linear) - cut[c].begin()) - 1);
    }
};
struct Interval {
    double lo = 0, hi = 0;
};
struct Vote {
    double point;
    int change;
};
struct Plateau {
    double lo = 0, hi = 0;
    int support = 0, current = 0, votes = 0;
    bool limited = false;
};
Plateau maximum_overlap(std::vector<Vote> &events, Interval bounds) {
    Plateau result{bounds.lo, bounds.hi};
    result.votes = lm::checked_cast<int>(events.size() / 2);
    if (events.empty()) {
        result.limited = true;
        return result;
    }
    std::sort(events.begin(), events.end(), [](auto a, auto b) { return a.point < b.point; });
    int count = 0, best = -1;
    double previous = bounds.lo;
    auto segment = [&](double end) {
        if (end <= previous)
            return;
        if (previous <= 0 && end > 0)
            result.current = count;
        double distance = previous > 0 ? previous : end < 0 ? -end : 0;
        double old_distance = result.lo > 0 ? result.lo : result.hi < 0 ? -result.hi : 0;
        if (count > best || (count == best && distance < old_distance)) {
            best = count;
            result.lo = previous;
            result.hi = end;
        } else if (count == best && previous == result.hi)
            result.hi = end;
    };
    for (size_t i = 0; i < events.size();) {
        double point = events[i].point;
        segment(point);
        do {
            count += events[i++].change;
        } while (i < events.size() && events[i].point == point);
        previous = point;
    }
    segment(bounds.hi);
    result.support = std::max(0, best);
    result.limited = result.lo == bounds.lo || result.hi == bounds.hi;
    return result;
}
// Allowed un-floored intensity on channel 'moving', while an output byte
// remains in its quantization bin. Other channels may change through clipping.
Interval sample_interval(const ByteEncoder &encoder, const Eigen::RowVector3d &raw, double floor, int moving,
                         int output, int target) {
    if (target < encoder.minimum)
        return {1, 0};
    double lo = target == encoder.minimum ? 0 : encoder.cut[output][target];
    double hi = encoder.cut[output][target + 1];
    std::array<double, 3> fixed;
    double other_peak = 0;
    for (int c = 0; c < 3; ++c) {
        fixed[c] = std::max(raw[c] * encoder.settings.scale[c], std::max(0.0, floor));
        if (c != moving)
            other_peak = std::max(other_peak, fixed[c]);
    }
    double factor = other_peak > encoder.cap ? encoder.cap / other_peak : 1;
    Interval intensity{-INFINITY, INFINITY};
    if (moving == output) {
        if (lo > encoder.cap || hi <= 0 || factor <= 0)
            return {1, 0};
        intensity.lo = lo / factor;
        if (hi <= encoder.cap)
            intensity.hi = hi / factor;
    } else {
        double maximum = fixed[output] * factor;
        if (lo > maximum || hi <= 0)
            return {1, 0};
        if (!std::isfinite(encoder.cap) || fixed[output] == 0)
            return maximum >= lo && maximum < hi ? intensity : Interval{1, 0};
        if (hi <= maximum)
            intensity.lo = fixed[output] * encoder.cap / hi;
        if (lo > 0)
            intensity.hi = fixed[output] * encoder.cap / lo;
    }
    // max(unfloored, floor) extends bins containing the floor to -infinity.
    floor = std::max(0.0, floor);
    if (intensity.hi <= floor)
        return {1, 0};
    if (intensity.lo <= floor)
        intensity.lo = -INFINITY;
    return intensity;
}
struct AccuracyReport {
    Quality before, consensus, after;
    std::string status = "disabled";
    int moves = 0, sweeps = 0, consensus_bakes = 0, snapped = 0;
    size_t snap_bakes = 0;
    struct Range {
        std::string name;
        int channel;
        double lo, hi;
        int support, votes;
        bool limited;
    };
    std::vector<Range> ranges;
    struct Attempt {
        Quality measured;
        bool accepted, structure_changed;
        double largest_error_increase = 0;
    };
    std::vector<Attempt> attempts;
};
struct ByteScore {
    uint64_t exact = 0, absolute = 0;
};
ByteScore style0_score(const BSP &bsp, const Probe &probe) {
    ByteScore result;
    for (int i = 0; i < bsp.target.rows(); ++i)
        if (bsp.styles[i] == 0)
            for (int c = 0; c < 3; ++c) {
                int error = std::abs(int(probe.bytes(i, c)) - int(bsp.target(i, c)));
                result.exact += error == 0;
                result.absolute += error;
            }
    return result;
}
bool accuracy_accept(const BSP &bsp, const Probe &candidate, const Probe &current, bool allow_equal) {
    if (!candidate.complete)
        return false;
    auto a = style0_score(bsp, candidate), b = style0_score(bsp, current);
    if (a.exact < b.exact ||
        (a.exact == b.exact && (a.absolute > b.absolute || (!allow_equal && a.absolute == b.absolute))))
        return false;
    auto qa = quality(bsp, candidate), qb = quality(bsp, current);
    // A handful of 2->3 byte crossings must not veto thousands of corrected
    // bytes. Aggregate errors still cannot increase, and new outliers are bounded.
    double mild_budget = double(std::max<uint64_t>(1, (a.exact - b.exact) / 100)) / bsp.target.size();
    if (qa.mae > qb.mae + 1e-6 || qa.rmse > qb.rmse + 1e-5 || qa.within2 + mild_budget + 1e-12 < qb.within2)
        return false;
    for (int i = 0; i < bsp.target.rows(); ++i)
        for (int c = 0; c < 3; ++c)
            if (std::abs(candidate.bytes(i, c) - bsp.target(i, c)) >
                std::abs(current.bytes(i, c) - bsp.target(i, c)) + 4)
                return false;
    return true;
}
Eigen::RowVector3d recipe_values(const std::string &recipe) {
    std::istringstream input(recipe);
    std::vector<float> values;
    float value;
    while (input >> value)
        values.push_back(value);
    if (values.size() == 1)
        return Eigen::RowVector3d::Constant(values[0]);
    if (values.size() != 3 && values.size() != 4)
        throw std::runtime_error("Invalid generated RAD recipe");
    Eigen::RowVector3d result(values[0], values[1], values[2]);
    if (values.size() == 4)
        for (int c = 0; c < 3; ++c)
            result[c] = float(result[c] * (double(values[3]) / 255));
    return result;
}
Fit formatted_fit(Fit fit, int decimals, Engine &engine) {
    auto text = format_rad(fit, decimals);
    std::istringstream input(text);
    std::string line;
    size_t row = 0;
    fit.recipes.resize(fit.names.size());
    while (std::getline(input, line)) {
        std::istringstream entry(line);
        std::string name, recipe;
        entry >> name;
        std::getline(entry, recipe);
        while (row < fit.names.size() && upper(fit.names[row]) != name)
            ++row;
        if (row == fit.names.size())
            throw std::runtime_error("Generated RAD rows changed order");
        recipe.erase(0, recipe.find_first_not_of(" \t"));
        auto values = recipe_values(recipe);
        if (values[0] == values[1] && values[1] == values[2])
            recipe = number(values[0], decimals);
        fit.recipes[row] = recipe;
        fit.values.row(row++) = recipe_values(recipe);
    }
    text = format_rad(fit, decimals);
    fit.probe = engine.bake({}, true, false, &text);
    return fit;
}
void update_model(const BSP &bsp, const Fit &fit, Engine &engine, Profile &profile, LinearModel &model,
                  bool batched) {
    if (engine.cache_enabled() && !bsp.native_encoding && model.matches(fit)) {
        ++profile.jacobian_reuses;
        return;
    }
    model.derivative = derivatives(bsp, fit, engine, profile, batched);
    model.values = fit.values;
    model.raw = fit.probe->raw;
    model.names = fit.names;
    model.structure = fit.probe->layout;
    ++profile.jacobian_builds;
}
Interval coordinate_bounds(const BSP &bsp, const Fit &fit, const RGB &values, int j, int channel) {
    double origin = values(j, channel), radius = std::max(.25, std::abs(fit.values(j, channel)) * .01);
    Interval result{std::max(0.0, fit.values(j, channel) - radius) - origin,
                    fit.values(j, channel) + radius - origin};
    auto constrain = [&](const float *weight, double threshold) {
        if (weight[channel] <= 0)
            return;
        double value = 0;
        for (int c = 0; c < 3; ++c)
            value += values(j, c) * weight[c];
        double edge = (threshold - value) / weight[channel];
        if (value >= threshold)
            result.lo = std::max(result.lo, edge);
        else
            result.hi = std::min(result.hi, edge);
    };
    if (bsp.native_encoding) {
        for (const auto &boundary : fit.probe->boundaries)
            if (bsp.faces[boundary.face].texture == fit.names[j])
                constrain(boundary.weight, bsp.encoding.casting_threshold);
    } else {
        const float weights[3] = {1.0f / 3, 1.0f / 3, 1.0f / 3};
        constrain(weights, 25);
    }
    return result;
}
Plateau parameter_plateau(const BSP &bsp, const ByteEncoder &encoder, const RGB &raw,
                          Eigen::Ref<const Vector> derivative, int channel, Interval bounds) {
    std::vector<Vote> events;
    events.reserve(bsp.target.rows() * 2);
    if (!(bounds.hi > bounds.lo))
        return {0, 0, 0, 0, 0, true};
    for (int i = 0; i < bsp.target.rows(); ++i) {
        if (bsp.styles[i] != 0 || std::abs(derivative[i]) * (bounds.hi - bounds.lo) < 1e-10)
            continue;
        double slope = derivative[i] * encoder.settings.scale[channel];
        for (int c = 0; c < 3; ++c) {
            auto intensity =
                sample_interval(encoder, raw.row(i), bsp.floor[i], channel, c, int(bsp.target(i, c)));
            if (intensity.hi <= intensity.lo)
                continue;
            double a = (intensity.lo - raw(i, channel) * encoder.settings.scale[channel]) / slope;
            double b = (intensity.hi - raw(i, channel) * encoder.settings.scale[channel]) / slope;
            if (a > b)
                std::swap(a, b);
            a = std::max(a, bounds.lo);
            b = std::min(b, bounds.hi);
            // Bins that cover the entire search range cannot favor any move.
            if (a >= b || (a == bounds.lo && b == bounds.hi))
                continue;
            events.push_back({a, 1});
            events.push_back({b, -1});
        }
    }
    return maximum_overlap(events, bounds);
}
Fit consensus_refine(const BSP &bsp, Fit fit, Engine &engine, const Options &options, Profile &profile,
                     LinearModel &model, AccuracyReport &report) {
    Timer timer(profile, "byte_consensus");
    ByteEncoder encoder(bsp);
    if (!encoder.supported) {
        report.status = "unsupported_nonuniform_gamma";
        return fit;
    }
    report.status = "checked";
    update_model(bsp, fit, engine, profile, model, options.batched);
    for (int pass = 0; pass < 2; ++pass) {
        RGB values = fit.values, raw = fit.probe->raw;
        int moves = 0;
        for (int sweep = 0; sweep < 6; ++sweep) {
            ++report.sweeps;
            bool changed = false;
            for (int j = 0; j < values.rows(); ++j)
                for (int c = 0; c < 3; ++c) {
                    check_cancelled();
                    auto bounds = coordinate_bounds(bsp, fit, values, j, c);
                    auto plateau =
                        parameter_plateau(bsp, encoder, raw, model.derivative[c].col(j), c, bounds);
                    if (plateau.support <= plateau.current)
                        continue;
                    double step = float(values(j, c) + (plateau.lo + plateau.hi) * .5) - values(j, c);
                    if (!step)
                        continue;
                    values(j, c) += step;
                    raw.col(c) += model.derivative[c].col(j) * step;
                    ++moves;
                    changed = true;
                }
            if (!changed)
                break;
        }
        report.moves += moves;
        if (!moves)
            break;
        Fit candidate = fit;
        candidate.values = values;
        candidate.recipes.clear();
        candidate = formatted_fit(std::move(candidate), options.decimals, engine);
        ++report.consensus_bakes;
        bool accepted = accuracy_accept(bsp, *candidate.probe, *fit.probe, false);
        report.attempts.push_back(
            {quality(bsp, *candidate.probe), accepted, candidate.probe->layout != fit.probe->layout});
        report.attempts.back().largest_error_increase =
            ((candidate.probe->bytes - bsp.target).cwiseAbs() - (fit.probe->bytes - bsp.target).cwiseAbs())
                .maxCoeff();
        if (!accepted)
            break;
        fit = std::move(candidate);
        if (!model.matches(fit)) {
            update_model(bsp, fit, engine, profile, model, options.batched);
        }
    }
    return fit;
}
void measure_plateaus(const BSP &bsp, const Fit &fit, const LinearModel &model, AccuracyReport &report) {
    report.ranges.clear();
    ByteEncoder encoder(bsp);
    if (!encoder.supported || !model.matches(fit))
        return;
    for (int j = 0; j < fit.values.rows(); ++j)
        for (int c = 0; c < 3; ++c) {
            auto bounds = coordinate_bounds(bsp, fit, fit.values, j, c);
            auto p = parameter_plateau(bsp, encoder, fit.probe->raw, model.derivative[c].col(j), c, bounds);
            report.ranges.push_back({fit.names[j], c, fit.values(j, c) + p.lo, fit.values(j, c) + p.hi,
                                     p.support, p.votes, p.limited});
        }
}
struct RecipeCandidate {
    std::string recipe;
    Eigen::RowVector3d values;
    double cost = 0, distance = 0;
    ByteScore score{};
};
double recipe_cost(const std::string &recipe) {
    std::istringstream input(recipe);
    std::vector<double> values;
    std::string token;
    double cost = 0;
    while (input >> token) {
        values.push_back(std::stod(token));
        if (token.find('.') == std::string::npos)
            while (token.size() > 1 && token.back() == '0')
                token.pop_back();
        cost += 2 + token.size();
    }
    if (values.size() == 4)
        cost += (255 - *std::max_element(values.begin(), values.begin() + 3)) / 255;
    return cost;
}
std::vector<RecipeCandidate> rounded_recipes(const Eigen::RowVector3d &values,
                                             const std::array<Interval, 3> &plateaus, int decimals) {
    std::map<std::string, RecipeCandidate> unique;
    auto add = [&](std::vector<double> numbers) {
        std::string text;
        for (double n : numbers) {
            if (!std::isfinite(n) || n < 0)
                return;
            auto digits = number(n, 0);
            if (!text.empty())
                text += ' ';
            text += digits;
        }
        if (numbers.size() == 4) {
            double peak = *std::max_element(numbers.begin(), numbers.begin() + 3);
            if (peak < 128 || peak > 255)
                return;
        }
        auto decoded = recipe_values(text);
        if (decoded[0] == decoded[1] && decoded[1] == decoded[2]) {
            text = number(decoded[0], decimals);
            decoded = recipe_values(text);
        }
        if (decoded.maxCoeff() <= 1)
            return;
        double distance = 0;
        for (int c = 0; c < 3; ++c) {
            double radius = std::max(.25, std::abs(values[c]) * .01);
            if (std::abs(decoded[c] - values[c]) > radius)
                return;
            double outside = std::max({plateaus[c].lo - decoded[c], decoded[c] - plateaus[c].hi, 0.0});
            double width = std::max({plateaus[c].hi - plateaus[c].lo, std::abs(values[c]) * 1e-5, .001});
            distance += std::pow(outside / width, 2) + .001 * std::pow((decoded[c] - values[c]) / radius, 2);
        }
        unique.emplace(text, RecipeCandidate{text, decoded, recipe_cost(text), distance});
    };
    const double units[] = {1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000};
    for (double step : units) {
        add({std::round(values.mean() / step) * step});
        add({std::round(values[0] / step) * step, std::round(values[1] / step) * step,
             std::round(values[2] / step) * step});
    }
    // Enumerate a generic authored-color prior; no texture names or reference
    // RAD values participate. Prefer the brighter of equivalent RGB aliases.
    for (int peak_color = 128; peak_color <= 255; ++peak_color) {
        double estimate = values.maxCoeff() * 255 / peak_color;
        for (double unit : units) {
            double brightness = std::round(estimate / unit) * unit;
            if (brightness <= 0)
                continue;
            for (int mask = 0; mask < 8; ++mask) {
                std::vector<double> numbers(4);
                for (int c = 0; c < 3; ++c) {
                    double color = values[c] * 255 / brightness;
                    numbers[c] = mask & (1 << c) ? std::ceil(color) : std::floor(color);
                }
                numbers[3] = brightness;
                add(std::move(numbers));
            }
        }
    }
    std::vector<RecipeCandidate> result;
    for (auto &[key, candidate] : unique)
        result.push_back(std::move(candidate));
    std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
        return a.distance != b.distance ? a.distance < b.distance : a.cost < b.cost;
    });
    // Keep both evidence-near and simple alternatives before the more expensive
    // whole-lightmap prediction. Actual bakes decide which may be accepted.
    if (result.size() > 48) {
        std::partial_sort(result.begin() + 40, result.begin() + 48, result.end(),
                          [](const auto &a, const auto &b) {
                              return a.cost != b.cost ? a.cost < b.cost : a.distance < b.distance;
                          });
        result.resize(48);
    }
    return result;
}
ByteScore recipe_prediction(const BSP &bsp, const ByteEncoder &encoder, const Fit &fit,
                            const LinearModel &model, int j, const Eigen::RowVector3d &values) {
    ByteScore result;
    auto delta = (values - fit.values.row(j)).eval();
    for (int i = 0; i < bsp.target.rows(); ++i)
        if (bsp.styles[i] == 0) {
            Eigen::RowVector3d raw = fit.probe->raw.row(i);
            for (int c = 0; c < 3; ++c)
                raw[c] += model.derivative[c](i, j) * delta[c];
            auto linear = encoder.linear(raw, bsp.floor[i]);
            for (int c = 0; c < 3; ++c) {
                int target = int(bsp.target(i, c));
                bool exact = target >= encoder.minimum &&
                             (target == encoder.minimum || linear[c] >= encoder.cut[c][target]) &&
                             linear[c] < encoder.cut[c][target + 1];
                result.exact += exact;
                if (!exact)
                    result.absolute += std::abs(encoder.byte(linear[c], c) - target);
            }
        }
    return result;
}
Fit snap_refine(const BSP &bsp, Fit fit, Engine &engine, const Options &options, Profile &profile,
                LinearModel &model, AccuracyReport &report) {
    Timer timer(profile, "authored_rounding");
    ByteEncoder encoder(bsp);
    if (!encoder.supported)
        return fit;
    for (int j = 0; j < fit.values.rows(); ++j) {
        check_cancelled();
        if (!model.matches(fit))
            update_model(bsp, fit, engine, profile, model, options.batched);
        std::array<Interval, 3> ranges;
        for (int c = 0; c < 3; ++c) {
            auto bounds = coordinate_bounds(bsp, fit, fit.values, j, c);
            auto p = parameter_plateau(bsp, encoder, fit.probe->raw, model.derivative[c].col(j), c, bounds);
            ranges[c] = {fit.values(j, c) + p.lo, fit.values(j, c) + p.hi};
        }
        auto candidates = rounded_recipes(fit.values.row(j), ranges, options.decimals);
        engine.parallel(candidates.size(), [&](size_t k) {
            candidates[k].score = recipe_prediction(bsp, encoder, fit, model, j, candidates[k].values);
        });
        std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
            if (a.score.exact != b.score.exact)
                return a.score.exact > b.score.exact;
            if (a.score.absolute != b.score.absolute)
                return a.score.absolute < b.score.absolute;
            if (a.cost != b.cost)
                return a.cost < b.cost;
            return a.distance < b.distance;
        });
        std::vector<Fit> probes;
        std::vector<double> costs;
        for (const auto &candidate : candidates) {
            if (candidate.recipe == fit.recipes[j])
                continue;
            Fit next = fit;
            next.recipes[j] = candidate.recipe;
            probes.push_back(std::move(next));
            costs.push_back(candidate.cost);
            if (probes.size() == 3)
                break;
        }
        engine.parallel(probes.size(), [&](size_t k) {
            probes[k] = formatted_fit(std::move(probes[k]), options.decimals, engine);
        });
        report.snap_bakes += probes.size();
        int best = -1;
        const auto unchanged = style0_score(bsp, *fit.probe);
        const double unchanged_cost = recipe_cost(fit.recipes[j]);
        for (size_t k = 0; k < probes.size(); ++k) {
            if (!accuracy_accept(bsp, *probes[k].probe, *fit.probe, true))
                continue;
            auto a = style0_score(bsp, *probes[k].probe);
            if (a.exact == unchanged.exact && a.absolute == unchanged.absolute && costs[k] >= unchanged_cost)
                continue;
            auto b = best < 0 ? ByteScore{} : style0_score(bsp, *probes[best].probe);
            if (best < 0 || a.exact > b.exact ||
                (a.exact == b.exact &&
                 (a.absolute < b.absolute || (a.absolute == b.absolute && costs[k] < costs[best]))))
                best = lm::checked_cast<int>(k);
        }
        if (best >= 0) {
            fit = std::move(probes[best]);
            ++report.snapped;
        }
    }
    return fit;
}
Fit refine_accuracy(const BSP &bsp, Fit fit, Engine &engine, const Options &options, Profile &profile,
                    LinearModel &model, AccuracyReport &report) {
    Timer timer(profile, "accuracy_refinement");
    fit = formatted_fit(std::move(fit), options.decimals, engine);
    report.before = report.consensus = report.after = quality(bsp, *fit.probe);
    if (fit.names.empty() || (!options.consensus && !options.snap))
        return fit;
    if (options.consensus)
        fit = consensus_refine(bsp, std::move(fit), engine, options, profile, model, report);
    else {
        update_model(bsp, fit, engine, profile, model, options.batched);
        report.status = "rounding_only";
    }
    report.consensus = quality(bsp, *fit.probe);
    if (options.snap)
        fit = snap_refine(bsp, std::move(fit), engine, options, profile, model, report);
    report.after = quality(bsp, *fit.probe);
    measure_plateaus(bsp, fit, model, report);
    return fit;
}
std::string quoted(const std::string &text) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else
            out << c;
    }
    out << '"';
    return out.str();
}
struct CompilerTrial {
    std::string name, error;
    Quality quality;
    double elapsed = 0;
    uint64_t probes = 0;
};
struct Selection {
    std::unique_ptr<Engine> engine;
    Fit fit;
    Fingerprint fingerprint;
    std::vector<CompilerTrial> trials;
    bool automatic = false, skipped_alternatives = false;
};
Selection select_compiler(BSP &bsp, const Options &options, Profile &profile, const fs::path &work) {
    Timer timer(profile, "compiler_selection");
    Selection result;
    result.fingerprint = fingerprint(bsp);
    result.automatic = options.compiler == "auto";
    const std::vector<std::string> supported{"qrad", "zhlt", "vhlt", "sdhlt"};
    std::vector<std::string> candidates;
    if (result.automatic) {
        for (const auto &name : result.fingerprint.preferred) {
            if (std::find(supported.begin(), supported.end(), name) == supported.end())
                throw std::runtime_error("BSP suggests " + name +
                                         ", which has no integrated RAD backend. "
                                         "Use --compiler only if you know a compatible supported engine.");
            candidates.push_back(name);
        }
        for (const auto &name : supported)
            if (std::find(candidates.begin(), candidates.end(), name) == candidates.end())
                candidates.push_back(name);
    } else
        candidates = {options.compiler};
    const auto original_floor = bsp.floor;
    Vector best_floor;
    CompilerEncoding best_encoding;
    bool best_native = false;
    Quality best;
    for (const auto &name : candidates) {
        check_cancelled();
        auto start = Clock::now();
        auto before = profile.probes;
        CompilerTrial trial;
        trial.name = name;
        try {
            bsp.native_encoding = false;
            bsp.encoding = CompilerEncoding{};
            bsp.floor = original_floor;
            auto local = options;
            local.compiler = name;
            if (result.automatic && name != "qrad" && local.backend == "cuda")
                local.backend = "cpu";
            auto directory = work / name;
            fs::create_directory(directory);
            auto engine = std::make_unique<Engine>(bsp, local, profile, directory);
            profile.status("Bruting " + options.bsp.stem().string() + " with " + upper(name) + " - " +
                           upper(engine->backend()) + ", " + std::to_string(engine->workers) + " threads");
            auto fit = bootstrap(bsp, *engine, profile);
            if (result.automatic && options.passes && !fit.names.empty())
                fit = refine_precision(bsp, fit, *engine, profile, 1, options.batched);
            if (!fit.probe->complete)
                throw std::runtime_error("Face/style layout does not match this compiler");
            trial.quality = quality(bsp, *fit.probe);
            if (options.verbose)
                profile.note(name + ": MAE " + std::to_string(trial.quality.mae) + "; " +
                             std::to_string(100 * trial.quality.exact) + "% exact channels");
            if (!result.engine || trial.quality.capped < best.capped ||
                (trial.quality.capped == best.capped && trial.quality.mae < best.mae)) {
                result.engine = std::move(engine);
                result.fit = std::move(fit);
                best = trial.quality;
                best_floor = bsp.floor;
                best_encoding = bsp.encoding;
                best_native = bsp.native_encoding;
            }
        } catch (const std::exception &e) {
            check_cancelled();
            if (!result.automatic)
                throw;
            trial.error = e.what();
            if (options.verbose)
                profile.note(name + " candidate failed: " + trial.error);
            else
                profile.status("Skipped " + upper(name));
        }
        trial.elapsed = seconds(start);
        trial.probes = profile.probes - before;
        result.trials.push_back(std::move(trial));
        // A close reproduction is enough to select a working model. It is not
        // proof of authorship: compatible engines/settings can produce the same bytes.
        if (result.engine && best.mae <= .1 && best.exact >= .98) {
            result.skipped_alternatives = result.trials.size() < candidates.size();
            break;
        }
    }
    if (!result.engine) {
        std::string errors = "No supported compiler could reproduce this BSP layout.";
        for (const auto &trial : result.trials)
            errors += "\n" + trial.name + ": " + trial.error;
        throw std::runtime_error(errors);
    }
    bsp.floor = std::move(best_floor);
    bsp.encoding = best_encoding;
    bsp.native_encoding = best_native;
    if (options.backend == "cuda" && result.engine->compiler() != "qrad")
        throw std::runtime_error("Selected " + result.engine->compiler() +
                                 " requires the CPU backend; use --backend auto or cpu");
    if (options.verbose)
        profile.note("Selected " + result.engine->compiler() +
                     (result.automatic ? " as the fitting model; compiler identity is not proven" : " by request"));
    profile.status("Selected " + upper(result.engine->compiler()) + "; " +
                   std::to_string(result.fit.names.size()) + " texture lights found");
    return result;
}
std::string profile_json(const Profile &profile, const Options &options, const Engine &engine, const BSP &bsp,
                         const Quality &result, const Probe &validation, const Selection &selection,
                         const AccuracyReport &accuracy) {
    std::ostringstream out;
    out << std::setprecision(10);
    out << "{\n  \"schema\": 2,\n  \"source\": " << quoted(options.bsp.string())
        << ",\n  \"source_sha256\": " << quoted(bsp.digest)
        << ",\n  \"compiler\": " << quoted(engine.compiler())
        << ",\n  \"quality\": " << quoted(options.quality)
        << ",\n  \"refinement_passes\": " << options.passes
        << ",\n  \"compiler_selection\": {\"automatic\": " << (selection.automatic ? "true" : "false")
        << ", \"identity_proven\": false, \"skipped_alternatives\": "
        << (selection.skipped_alternatives ? "true" : "false") << ", \"evidence\": [";
    for (size_t i = 0; i < selection.fingerprint.evidence.size(); ++i)
        out << (i ? ", " : "") << quoted(selection.fingerprint.evidence[i]);
    out << "], \"trials\": [";
    for (size_t i = 0; i < selection.trials.size(); ++i) {
        const auto &trial = selection.trials[i];
        out << (i ? ", " : "") << "{\"compiler\": " << quoted(trial.name)
            << ", \"seconds\": " << trial.elapsed << ", \"probes\": " << trial.probes;
        if (!trial.error.empty())
            out << ", \"error\": " << quoted(trial.error);
        else
            out << ", \"mae\": " << trial.quality.mae << ", \"exact_fraction\": " << trial.quality.exact;
        out << '}';
    }
    out << "]},\n  \"encoding\": {\"scale\": [";
    for (int c = 0; c < 3; ++c)
        out << (c ? ", " : "") << bsp.encoding.scale[c];
    out << "], \"gamma\": [";
    for (int c = 0; c < 3; ++c)
        out << (c ? ", " : "") << bsp.encoding.gamma[c];
    out << "], \"preclip\": " << bsp.encoding.preclip << ", \"postclip\": " << bsp.encoding.postclip
        << ", \"rounding\": " << bsp.encoding.rounding
        << ", \"bounces\": " << (options.bounces >= 0 ? options.bounces : int(bsp.encoding.bounces)) << '}'
        << ",\n  \"backend\": " << quoted(engine.backend())
        << ",\n  \"validation_backend\": \"cpu_uncached\",\n  \"workers\": " << engine.workers
        << ",\n  \"wall_seconds\": " << seconds(profile.start) << ",\n  \"stages_seconds\": {";
    bool first = true;
    for (const auto &[name, time] : profile.stages) {
        if (!first)
            out << ',';
        first = false;
        out << "\n    " << quoted(name) << ": " << time;
    }
    out << "\n  },\n  \"compiler_seconds_sum\": {";
    const char *phases[] = {"bsp_read_parse", "geometry",          "direct_lighting", "transfers",
                            "radiosity",      "lightmap_encoding", "bsp_write"};
    for (int i = 0; i < LM_PHASE_COUNT; ++i)
        out << (i ? "," : "") << "\n    " << quoted(phases[i]) << ": " << profile.compiler.seconds[i];
    out << "\n  },\n  \"compiler_cpu_seconds_sum\": {";
    for (int i = 0; i < LM_PHASE_COUNT; ++i)
        out << (i ? "," : "") << "\n    " << quoted(phases[i]) << ": " << profile.compiler.cpu_seconds[i];
    out << "\n  },\n  \"probes\": " << profile.probes
        << ",\n  \"lighting_cache\": " << (options.cache ? "true" : "false")
        << ",\n  \"cache_scope\": " << quoted(options.cache ? "current_recovery_only" : "disabled")
        << ",\n  \"duplicate_probes_reused\": " << profile.probe_cache_hits
        << ",\n  \"derivative_probes\": " << profile.derivative_probes
        << ",\n  \"frozen_derivatives\": " << (options.frozen_derivatives ? "true" : "false")
        << ",\n  \"qrad_smooth\": " << options.qrad_smooth
        << ",\n  \"frozen_derivative_builds\": " << profile.frozen_derivative_builds
        << ",\n  \"derivative_batched_columns\": " << profile.derivative_batched_columns
        << ",\n  \"derivative_fallback_columns\": " << profile.derivative_fallback_columns
        << ",\n  \"jacobian_builds\": " << profile.jacobian_builds
        << ",\n  \"jacobian_reuses\": " << profile.jacobian_reuses
        << ",\n  \"threshold_probes\": " << profile.threshold_probes
        << ",\n  \"optimizer_stalls\": " << profile.optimizer_stalls
        << ",\n  \"geometry_cache_hits\": " << profile.compiler.geometry_hits
        << ",\n  \"transfer_cache_hits\": " << profile.compiler.transfer_hits
        << ",\n  \"transfer_cache_misses\": " << profile.compiler.transfer_misses
        << ",\n  \"direct_cache_hits\": " << profile.compiler.direct_hits
        << ",\n  \"direct_cache_misses\": " << profile.compiler.direct_misses
        << ",\n  \"cache_bytes_read\": " << profile.compiler.cache_read_bytes
        << ",\n  \"cache_bytes_written\": " << profile.compiler.cache_write_bytes
        << ",\n  \"disk_bytes_read\": " << profile.disk_bytes_read
        << ",\n  \"disk_bytes_written\": " << profile.disk_bytes_written
        << ",\n  \"memory_transport_bytes\": " << profile.ipc_bytes
        << ",\n  \"worker_peak_rss_bytes\": " << profile.worker_peak_rss
        << ",\n  \"gpu_scene_uploads\": " << profile.compiler.gpu_scene_uploads
        << ",\n  \"gpu_transfer_uploads\": " << profile.compiler.gpu_transfer_uploads
        << ",\n  \"gpu_transfer_reuses\": " << profile.compiler.gpu_transfer_reuses
        << ",\n  \"cache_verified_bytes\": " << profile.compiler.cache_verified_bytes
        << ",\n  \"gpu_plan_uploads\": " << profile.compiler.gpu_plan_uploads
        << ",\n  \"gpu_calls\": " << profile.compiler.gpu_calls
        << ",\n  \"gpu_ipc_bytes\": " << profile.compiler.gpu_bytes
        << ",\n  \"gpu_service_seconds\": " << profile.compiler.gpu_seconds
        << ",\n  \"gpu_direct_batches\": " << profile.compiler.gpu_direct_calls
        << ",\n  \"gpu_direct_pairs\": " << profile.compiler.gpu_direct_pairs
        << ",\n  \"gpu_direct_cpu_fallbacks\": " << profile.compiler.gpu_direct_uncertain
        << ",\n  \"direct_cache_overflows\": " << profile.compiler.direct_overflows
        << ",\n  \"cache_producer_contentions\": " << profile.compiler.cache_lock_contentions
        << ",\n  \"gpu_trace_batches\": " << profile.compiler.gpu_trace_calls
        << ",\n  \"gpu_trace_rays\": " << profile.compiler.gpu_trace_rays
        << ",\n  \"gpu_trace_cpu_fallbacks\": " << profile.compiler.gpu_trace_uncertain
        << ",\n  \"interpolation_cache_hits\": " << profile.compiler.indirect_hits
        << ",\n  \"interpolation_cache_misses\": " << profile.compiler.indirect_misses
        << ",\n  \"gpu_interpolation_batches\": " << profile.compiler.gpu_indirect_calls
        << ",\n  \"gpu_interpolation_samples\": " << profile.compiler.gpu_indirect_samples
        << ",\n  \"gpu_interpolation_cpu_fallbacks\": " << profile.compiler.gpu_indirect_uncertain
        << ",\n  \"validation\": {\"mae\": " << result.mae << ", \"rmse\": " << result.rmse
        << ", \"exact_fraction\": " << result.exact << ", \"within_two_fraction\": " << result.within2
        << ", \"style0_exact_fraction\": " << result.style0_exact << ", \"style0_mae\": " << result.style0_mae
        << "},\n  \"accuracy_refinement\": {\"status\": " << quoted(accuracy.status)
        << ", \"consensus_enabled\": " << (options.consensus ? "true" : "false")
        << ", \"rounding_enabled\": " << (options.snap ? "true" : "false")
        << ", \"moves\": " << accuracy.moves << ", \"sweeps\": " << accuracy.sweeps
        << ", \"consensus_bakes\": " << accuracy.consensus_bakes
        << ", \"rounding_bakes\": " << accuracy.snap_bakes << ", \"rounded_textures\": " << accuracy.snapped;
    const std::pair<const char *, const Quality *> steps[] = {{"before", &accuracy.before},
                                                              {"after_consensus", &accuracy.consensus},
                                                              {"after_rounding", &accuracy.after}};
    for (auto [name, measured] : steps)
        out << ", " << quoted(name) << ": {\"exact_fraction\": " << measured->exact
            << ", \"mae\": " << measured->mae << ", \"style0_exact_fraction\": " << measured->style0_exact
            << ", \"style0_mae\": " << measured->style0_mae << '}';
    out << ", \"consensus_attempts\": [";
    for (size_t i = 0; i < accuracy.attempts.size(); ++i) {
        const auto &attempt = accuracy.attempts[i];
        out << (i ? ", " : "") << "{\"accepted\": " << (attempt.accepted ? "true" : "false")
            << ", \"structure_changed\": " << (attempt.structure_changed ? "true" : "false")
            << ", \"style0_exact_fraction\": " << attempt.measured.style0_exact
            << ", \"mae\": " << attempt.measured.mae << ", \"rmse\": " << attempt.measured.rmse
            << ", \"largest_error_increase\": " << attempt.largest_error_increase << '}';
    }
    out << ']';
    out << ", \"range_interpretation\": \"Local maximum-consensus intervals in effective RGB; other "
           "parameters held fixed. "
           "Search-bounded, approximate and conditional on this compiler model; not confidence intervals.\", "
           "\"ranges\": [";
    for (size_t i = 0; i < accuracy.ranges.size(); ++i) {
        const auto &range = accuracy.ranges[i];
        out << (i ? "," : "") << "\n    {\"texture\": " << quoted(upper(range.name))
            << ", \"channel\": " << range.channel << ", \"low\": " << range.lo << ", \"high\": " << range.hi
            << ", \"support\": " << range.support << ", \"varying_votes\": " << range.votes
            << ", \"search_limited\": " << (range.limited ? "true" : "false") << '}';
    }
    out << "]},\n  \"largest_face_residuals\": [";
    struct Residual {
        size_t count = 0;
        double absolute = 0, squared = 0, maximum = 0;
    };
    std::vector<Residual> residuals(bsp.faces.size());
    for (int i = 0; i < bsp.target.rows(); ++i)
        for (int c = 0; c < 3; ++c) {
            double error = std::abs(validation.bytes(i, c) - bsp.target(i, c));
            auto &face = residuals[bsp.face_ids[i]];
            ++face.count;
            face.absolute += error;
            face.squared += error * error;
            face.maximum = std::max(face.maximum, error);
        }
    std::vector<size_t> order(residuals.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return residuals[a].absolute > residuals[b].absolute; });
    for (size_t i = 0; i < std::min(size_t(20), order.size()); ++i) {
        size_t fid = order[i];
        const auto &face = residuals[fid];
        if (!face.count || !face.absolute)
            break;
        out << (i ? "," : "") << "\n    {\"face\": " << fid
            << ", \"texture\": " << quoted(upper(bsp.faces[fid].texture)) << ", \"channels\": " << face.count
            << ", \"mae\": " << face.absolute / face.count
            << ", \"rmse\": " << std::sqrt(face.squared / face.count)
            << ", \"maximum_error\": " << face.maximum << '}';
    }
    out << "\n  ]\n}\n";
    return out.str();
}
void recover(Options options) {
    Profile profile;
    lm::Progress progress(options.verbose, !options.detect_only);
    profile.progress = &progress;
    profile.status("Reading " + options.bsp.filename().string());
    Bytes bytes;
    {
        Timer timer(profile, "input_io");
        bytes = read_file(options.bsp);
        profile.disk_bytes_read += bytes.size();
    }
    auto start = Clock::now();
    BSP bsp(std::move(bytes));
    profile.add("bsp_parse", seconds(start));
    auto detected = fingerprint(bsp);
    if (options.detect_only) {
        for (const auto &evidence : detected.evidence)
            std::cout << evidence << '\n';
        std::cout << "RAD candidates:";
        for (const auto &name : detected.preferred)
            std::cout << ' ' << name;
        if (detected.preferred.empty())
            std::cout << " qrad zhlt vhlt sdhlt";
        std::cout << "\nCompiler identity is not guaranteed. No probes run; no files written.\n";
        return;
    }
    if (options.quality.empty()) {
        if (lm::terminal_input() && lm::terminal_width(2)) {
            progress.pause();
            options.quality = choose_quality(options.bsp);
            profile.start += progress.resume();
        } else {
            options.quality = "balanced";
        }
    }
    apply_quality(options);
    if (options.verbose) profile.note("Recovery quality: " + options.quality);
    if (options.verbose)
        for (const auto &evidence : detected.evidence)
            profile.note(evidence);
    TempDir temporary;
    if (options.compiler == "auto")
        profile.status("Detecting compiler and hardware");
    else
        profile.status("Starting " + upper(options.compiler));
    auto selection = select_compiler(bsp, options, profile, temporary.path);
    auto &engine = *selection.engine;
    auto fit = std::move(selection.fit);
    const auto brute_label = "Bruting " + options.bsp.stem().string() + " with " + upper(engine.compiler()) + " - ";
    auto brute_status = [&](const std::string &stage, int remaining_phases) {
        profile.status(brute_label + stage, remaining_phases, engine.seconds_per_refinement);
    };
    if (options.final_refinement) fit = explore_thresholds(bsp, fit, engine, profile);
    LinearModel linear;
    for (int pass = 0; pass < options.passes && !fit.names.empty(); ++pass) {
        // Remaining passes, final refinement, RAD rounding, and CPU validation.
        brute_status("pass " + std::to_string(pass + 1) + '/' + std::to_string(options.passes),
                     options.passes - pass + (options.final_refinement ? 1 : 0) + 2);
        Timer timer(profile, "precision_passes");
        auto next =
            refine_precision(bsp, fit, engine, profile, pass < 2 ? 1.0 : .35, options.batched, &linear);
        bool unchanged = (next.values - fit.values).cwiseAbs().maxCoeff() == 0;
        fit = std::move(next);
        if (unchanged)
            break;
    }
    if (options.final_refinement && options.passes && !fit.names.empty()) {
        if (options.verbose)
            profile.note("Refitting all samples and checking casting thresholds");
        brute_status("final refinement", 3);
        fit = refine_precision(bsp, fit, engine, profile, .1, options.batched, &linear, true);
        fit = explore_thresholds(bsp, fit, engine, profile);
    }
    AccuracyReport accuracy;
    if (options.verbose)
        profile.note("Refining byte agreement and verified RAD rounding");
    brute_status("rounding RAD values", 2);
    fit = refine_accuracy(bsp, std::move(fit), engine, options, profile, linear, accuracy);
    brute_status("CPU validation", 1);
    auto text = format_rad_output(format_rad(fit, options.decimals));
    auto accelerated = engine.bake({}, true, false, &text);
    if (!accelerated->complete ||
        (accelerated->raw - fit.probe->raw).cwiseAbs().maxCoeff() != 0 ||
        (accelerated->bytes - fit.probe->bytes).cwiseAbs().maxCoeff() != 0)
        throw std::runtime_error("Final RAD formatting changed the verified lighting");
    std::shared_ptr<Probe> validation;
    {
        Timer timer(profile, "final_validation");
        validation = engine.bake({}, true, true, &text);
    }
    if (!validation->complete)
        throw std::runtime_error("Final bake does not reproduce the original face/style layout");
    if ((accelerated->raw - validation->raw).cwiseAbs().maxCoeff() != 0 ||
        (accelerated->bytes - validation->bytes).cwiseAbs().maxCoeff() != 0)
        throw std::runtime_error("Accelerated lighting differs from the uncached CPU reference");
    auto measured = quality(bsp, *validation);
    {
        Timer timer(profile, "source_integrity_io");
        auto current = read_file(options.bsp);
        profile.disk_bytes_read += current.size();
        if (lm::sha256(current.data(), current.size()) != bsp.digest)
            throw std::runtime_error("Input BSP changed during recovery");
    }
    {
        Timer timer(profile, "result_io");
        atomic_write(options.output, text);
        profile.disk_bytes_written += text.size();
    }
    if (!options.profile.empty())
        atomic_write(options.profile,
                     profile_json(profile, options, engine, bsp, measured, *validation, selection, accuracy));
    progress.finish();
    std::cout << "Bruteforce is done. Took " << elapsed_time(seconds(profile.start))
              << "\nSaved file: " << options.output.string() << '\n' << std::flush;
}
void numerical_self_test() {
    if (lm::checked_cast<int32_t>(int64_t(std::numeric_limits<int32_t>::min())) !=
            std::numeric_limits<int32_t>::min() ||
        lm::checked_cast<uint32_t>(uint64_t(std::numeric_limits<uint32_t>::max())) !=
            std::numeric_limits<uint32_t>::max() ||
        lm::checked_cast<int>(std::floor(-1.25)) != -2 ||
        lm::checked_cast<unsigned char>(255.0) != 255)
        throw std::runtime_error("Valid numeric boundary conversion failed");
    auto reject_overflow = [](auto convert) {
        try { convert(); }
        catch (const std::overflow_error &) { return; }
        throw std::runtime_error("Out-of-range numeric conversion was accepted");
    };
    reject_overflow([] { lm::checked_cast<uint32_t>(-1); });
    reject_overflow([] { lm::checked_cast<int32_t>(std::numeric_limits<uint64_t>::max()); });
    reject_overflow([] { lm::checked_cast<uint32_t>(uint64_t(std::numeric_limits<uint32_t>::max()) + 1); });
    reject_overflow([] { lm::checked_cast<int32_t>(double(std::numeric_limits<int32_t>::max()) + 1.0); });
    reject_overflow([] { lm::checked_cast<unsigned char>(256.0); });
    reject_overflow([] { lm::checked_cast<int>(std::numeric_limits<double>::infinity()); });
    reject_overflow([] { lm::checked_cast<int>(std::numeric_limits<double>::quiet_NaN()); });
    Matrix a(6, 3);
    a << 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 3, 1;
    Vector known(3);
    known << 0, 4, 7;
    if ((nnls(a, a * known) - known).norm() > 1e-8)
        throw std::runtime_error("Nonnegative initializer failed");
    LightingObjective function;
    function.base.resize(6, 3);
    function.target.resize(6, 3);
    function.floor.resize(6);
    for (int i = 0; i < 6; ++i) {
        function.floor[i] = i == 1 ? 50 : 0;
        for (int c = 0; c < 3; ++c) {
            function.base(i, c) = i == 2 ? 400 + c * 20 : 5 + i * 3 + c;
            function.target(i, c) = 20 + i * 23 + c * 7;
        }
    }
    for (int c = 0; c < 3; ++c) {
        function.matrix[c].resize(6, 2);
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 2; ++j)
                function.matrix[c](i, j) = .5 + (i + j + c) % 5;
    }
    Vector point = Vector::Constant(6, .7), gradient;
    for (int mode = 0; mode < 4; ++mode) {
        function.native = mode != 0;
        function.encoding.preclip = mode == 1 ? 180 : -1;
        function.encoding.postclip = mode == 2 ? 188 : 255;
        function.encoding.minimum = mode == 3 ? 70 : 0;
        function.encoding.rounding = mode >= 2 ? .5 : 0;
        for (int c = 0; c < 3; ++c) {
            function.encoding.scale[c] = 1.5 + .1 * c;
            function.encoding.gamma[c] = .5 + .02 * c;
        }
        for (auto [objective, robust] : {std::pair{"interval", "cauchy"}, std::pair{"interval", "huber"},
                                         std::pair{"midpoint", "huber"}}) {
            function.objective = objective;
            function.robust = robust;
            function(point, gradient);
            for (int j = 0; j < point.size(); ++j) {
                auto plus = point, minus = point;
                plus[j] += 1e-5;
                minus[j] -= 1e-5;
                Vector ignore;
                double numerical = (function(plus, ignore) - function(minus, ignore)) / 2e-5;
                if (std::abs(numerical - gradient[j]) > 1e-6)
                    throw std::runtime_error("Lighting gradient differs from finite differences");
            }
        }
    }
    Fit format;
    format.names = {"~color", "gray"};
    format.values.resize(2, 3);
    format.values << 100, 50, 25, 20, 20, 20;
    if (format_rad(format, 3) != "~COLOR              255 127.5 63.75 100\nGRAY                20\n")
        throw std::runtime_error("Clean RAD formatting failed");
    if (format_rad_output(format_rad(format, 3)) !=
        "~COLOR  255 127.5 63.75 100\nGRAY    20\n")
        throw std::runtime_error("Final RAD alignment changed fractional emission values");
    format.values.row(1).setConstant(100.95);
    if (format_rad_output(format_rad(format, Options{}.decimals)) !=
        "~COLOR  255 128 64 100\nGRAY    101\n")
        throw std::runtime_error("Default RAD output did not round colors and brightness to whole numbers");
    try {
        BSP invalid(Bytes(124));
        throw std::runtime_error("Invalid BSP accepted");
    } catch (const std::runtime_error &e) {
        if (std::string(e.what()) != "Expected a GoldSrc BSP30 map")
            throw;
    }
    std::cout << "Native fitting: initializer, clipping/minlight gradients, RAD formatting and invalid BSP "
                 "handling passed\n";
}
void process_self_test(const fs::path &directory) {
    if (lm::sha256("abc", 3) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        throw std::runtime_error("SHA-256 known-answer check failed");
    auto transport = memory_file(directory, "radbruter-self-test", Bytes{0, 1, 10, 13, 26, 255});
    auto log = memory_file(directory, "radbruter-self-test-log");
    Child child;
    child.start({lm::executable_path().string(), "--transport-self-test", "space \"quoted\" trailing\\"}, environment(),
                {{transport->fd, 120}, {log->fd, 1}, {log->fd, 2}});
    int code;
    child.wait(code);
    if (code || read_fd(log->fd) != Bytes({0, 1, 10, 13, 26, 255}))
        throw std::runtime_error("Memory probe transport failed");
    {
        auto path = directory / "stream test.sock";
        lm::StreamServer server(path);
        Child peer;
        peer.start({lm::executable_path().string(), "--stream-self-test", path.string()}, environment(), {});
        FD stream(server.accept());
        Bytes sent(256 * 1024), received(sent.size());
        for (size_t i = 0; i < sent.size(); ++i) sent[i] = static_cast<unsigned char>(i * 31);
        write_all(stream.fd, sent.data(), sent.size());
        read_all(stream.fd, received.data(), received.size());
        peer.wait(code);
        if (code || sent != received) throw std::runtime_error("Binary worker stream failed");
    }
    auto destination = directory / "result.rad";
    atomic_write(destination, "OLD 20\n");
    atomic_write(destination, "NEW 30\n");
    if (read_file(destination) != Bytes({'N', 'E', 'W', ' ', '3', '0', '\n'}))
        throw std::runtime_error("Atomic RAD save failed");
    fs::path bad = directory / "directory";
    fs::create_directory(bad);
    try {
        atomic_write(bad, "test");
        throw std::runtime_error("Invalid save accepted");
    } catch (const fs::filesystem_error &) {
    }
    for (auto &path : fs::directory_iterator(directory))
        if (path.path().filename().string().starts_with(".directory."))
            throw std::runtime_error("Failed save leaked a temporary file");
    Child sleeper;
    sleeper.start({lm::executable_path().string(), "--sleep-self-test"}, environment(), {});
    auto pid = sleeper.id();
    sleeper.stop();
    if (lm::process_exists(pid))
        throw std::runtime_error("Cancelled worker survived cleanup");
    std::cout << "Runtime: memory transport, atomic writes and worker cancellation passed\n";
}
Bytes room_fixture(std::string entities = "{\n\"classname\" \"worldspawn\"\n}\n", bool padded = false,
                   bool sky = false, const char *panel_name = "~panel") {
    // A small generated BSP exercises the integrated compiler without game assets.
    std::array<Bytes, 15> lumps;
    auto store = [&](int index, const auto &values) {
        if (values.empty())
            return;
        const auto *data = reinterpret_cast<const unsigned char *>(values.data());
        lumps[index].assign(data, data + values.size() * sizeof(values[0]));
    };
    entities.push_back('\0');
    store(0, entities);
    std::vector<dvertex_t> vertices(8);
    for (int i = 0; i < 8; ++i)
        for (int c = 0; c < 3; ++c)
            vertices[i].point[c] = i & (4 >> c) ? 64.0f : -64.0f;
    store(3, vertices);
    const int polygons[6][4] = {{4, 5, 7, 6}, {0, 2, 3, 1}, {2, 6, 7, 3},
                                {0, 1, 5, 4}, {1, 3, 7, 5}, {0, 4, 6, 2}};
    std::vector<dedge_t> edges(1);
    std::vector<int32_t> surfedges;
    std::vector<dface_t> faces(6);
    std::vector<dplane_t> planes(6);
    std::vector<dnode_t> nodes(6);
    std::vector<texinfo_t> info(6);
    std::map<std::pair<int, int>, int> edge_ids;
    for (int i = 0; i < 6; ++i) {
        planes[i].normal[i / 2] = 1;
        planes[i].dist = i % 2 ? -64.0f : 64.0f;
        planes[i].type = i / 2;
        nodes[i].planenum = i;
        int next = i == 5 ? -2 : i + 1;
        nodes[i].children[0] = i % 2 ? next : -1;
        nodes[i].children[1] = i % 2 ? -1 : next;
        for (int c = 0; c < 3; ++c) {
            nodes[i].mins[c] = -64;
            nodes[i].maxs[c] = 64;
        }
        nodes[i].firstface = i;
        nodes[i].numfaces = 1;
        faces[i].planenum = i;
        faces[i].side = i % 2 ? 0 : 1;
        faces[i].firstedge = lm::checked_cast<int32_t>(surfedges.size());
        faces[i].numedges = 4;
        faces[i].texinfo = i;
        faces[i].styles[0] = 0;
        faces[i].styles[1] = faces[i].styles[2] = faces[i].styles[3] = 255;
        faces[i].lightofs = i * 9 * 9 * 3;
        info[i].vecs[0][i / 2 == 0 ? 1 : 0] = 1;
        info[i].vecs[1][i / 2 == 2 ? 1 : 2] = 1;
        info[i].miptex = i == 4 ? 1 : 0;
        if (sky) {
            info[i].miptex = i == 0 ? 1 : 0;
            if (i == 4) {
                info[i].miptex = 2;
                info[i].flags = TEX_SPECIAL;
                faces[i].lightofs = -1;
                nodes[i].children[0] = -3;
            }
        }
        for (int k = 0; k < 4; ++k) {
            int a = polygons[i][k], b = polygons[i][(k + 1) % 4];
            auto key = std::minmax(a, b);
            auto found = edge_ids.find(key);
            if (found == edge_ids.end()) {
                int id = lm::checked_cast<int>(edges.size());
                dedge_t edge{};
                edge.v[0] = a;
                edge.v[1] = b;
                edges.push_back(edge);
                edge_ids[key] = id;
                surfedges.push_back(id);
            } else {
                int id = found->second;
                surfedges.push_back(edges[id].v[0] == a ? id : -id);
            }
        }
    }
    store(1, planes);
    store(5, nodes);
    store(6, info);
    store(7, faces);
    store(12, edges);
    store(13, surfedges);
    std::vector<dleaf_t> leaves(sky ? 3 : 2);
    leaves[0].contents = CONTENTS_SOLID;
    leaves[0].visofs = -1;
    leaves[1].contents = CONTENTS_EMPTY;
    leaves[1].visofs = 0;
    if (sky) {
        leaves[2].contents = CONTENTS_SKY;
        leaves[2].visofs = -1;
    }
    for (auto &leaf : leaves)
        for (int c = 0; c < 3; ++c) {
            leaf.mins[c] = -64;
            leaf.maxs[c] = 64;
        }
    leaves[1].nummarksurfaces = 6;
    store(10, leaves);
    store(11, std::vector<uint16_t>{0, 1, 2, 3, 4, 5});
    lumps[4] = {1};
    lumps[8] = Bytes(6 * 9 * 9 * 3, 20);
    if (padded)
        lumps[8].resize(5 * 9 * 9 * 3 + 17 * 17 * 3, 0);
    std::vector<dmodel_t> models(1);
    models[0].numfaces = 6;
    models[0].visleafs = sky ? 2 : 1;
    for (int c = 0; c < 3; ++c) {
        models[0].mins[c] = -64;
        models[0].maxs[c] = 64;
    }
    store(14, models);
    int texture_count = sky ? 3 : 2;
    lumps[2].resize(4 + texture_count * 4);
    memcpy(lumps[2].data(), &texture_count, 4);
    for (int i = 0; i < texture_count; ++i) {
        int offset = lm::checked_cast<int>(lumps[2].size());
        memcpy(lumps[2].data() + 4 + i * 4, &offset, 4);
        miptex_t texture{};
        if (strlen(panel_name) >= sizeof texture.name)
            throw std::runtime_error("Fixture texture name is too long");
        strcpy(texture.name, i == 2 ? "sky" : i ? panel_name : "wall");
        texture.width = texture.height = 16;
        int position = sizeof texture;
        for (int mip = 0; mip < 4; ++mip) {
            texture.offsets[mip] = position;
            position += 256 >> (mip * 2);
        }
        Bytes data(position + 2 + 768, 0);
        memcpy(data.data(), &texture, sizeof texture);
        uint16_t palette = 256;
        memcpy(data.data() + position, &palette, 2);
        std::fill(data.begin() + position + 2, data.end(), 128);
        lumps[2].insert(lumps[2].end(), data.begin(), data.end());
    }
    dheader_t header{};
    header.version = 30;
    Bytes bytes(sizeof header);
    for (int i = 0; i < 15; ++i) {
        while (bytes.size() % 4)
            bytes.push_back(0);
        header.lumps[i].fileofs = lm::checked_cast<int32_t>(bytes.size());
        header.lumps[i].filelen = lm::checked_cast<int32_t>(lumps[i].size());
        bytes.insert(bytes.end(), lumps[i].begin(), lumps[i].end());
    }
    memcpy(bytes.data(), &header, sizeof header);
    return bytes;
}
Bytes fractional_grid_fixture(bool extended) {
    auto bytes = room_fixture();
    auto header = value_at<dheader_t>(bytes, 0);
    // A closed 126-unit room. Its first face straddles an old x87/SSE grid boundary.
    for (int i = 0; i < 8; ++i) {
        size_t offset = size_t(header.lumps[3].fileofs) + i * sizeof(dvertex_t);
        auto vertex = value_at<dvertex_t>(bytes, offset);
        for (auto &coordinate : vertex.point) coordinate *= 63.0f / 64.0f;
        memcpy(bytes.data() + offset, &vertex, sizeof vertex);
    }
    for (int i = 0; i < 6; ++i) {
        size_t offset = size_t(header.lumps[1].fileofs) + i * sizeof(dplane_t);
        auto plane = value_at<dplane_t>(bytes, offset);
        plane.dist *= 63.0f / 64.0f;
        memcpy(bytes.data() + offset, &plane, sizeof plane);
    }
    auto texture = value_at<texinfo_t>(bytes, header.lumps[6].fileofs);
    texture.vecs[0][1] = 4.0f / 3.0f;
    texture.vecs[0][3] = 52.0f;
    memcpy(bytes.data() + header.lumps[6].fileofs, &texture, sizeof texture);
    int added = ((extended ? 13 : 12) - 9) * 9 * 3;
    for (int i = 1; i < 6; ++i) {
        size_t offset = size_t(header.lumps[7].fileofs) + i * sizeof(dface_t);
        auto face = value_at<dface_t>(bytes, offset);
        face.lightofs += added;
        memcpy(bytes.data() + offset, &face, sizeof face);
    }
    int end = header.lumps[8].fileofs + header.lumps[8].filelen;
    bytes.insert(bytes.begin() + end, added, 0);
    header.lumps[8].filelen += added;
    for (int i = 0; i < 15; ++i)
        if (i != 8 && header.lumps[i].fileofs >= end) header.lumps[i].fileofs += added;
    memcpy(bytes.data(), &header, sizeof header);
    return bytes;
}
void original_grid_self_test(const fs::path &directory, const Options &options) {
    for (bool extended : {false, true}) {
        BSP bsp(fractional_grid_fixture(extended));
        const auto &face = bsp.faces[0];
        if (face.u != (extended ? -3 : -2) || face.width != (extended ? 13 : 12) || face.height != 9)
            throw std::runtime_error("Original fractional grid was parsed incorrectly");
        auto work = directory / (extended ? "grid-extended" : "grid-float");
        fs::create_directories(work);
        Profile profile;
        Engine engine(bsp, options, profile, work);
        auto probe = engine.bake({{"~panel", {80, 50, 20}}});
        auto reference = engine.bake({{"~panel", {80, 50, 20}}}, true, true);
        if (!probe->complete || !reference->complete ||
            (probe->raw - reference->raw).cwiseAbs().maxCoeff() != 0 ||
            (probe->bytes - reference->bytes).cwiseAbs().maxCoeff() != 0)
            throw std::runtime_error("Original fractional grids differ from CPU validation");
    }
    std::cout << "Original grids: float and extended-precision layouts preserved\n";
}
void bsp_grid_self_test() {
    auto bytes = room_fixture();
    auto header = value_at<dheader_t>(bytes, 0);
    // A fractional texture scale near a luxel boundary: float operations give
    // 3x7, doubles give 4x8, and QRAD's x87 product stored as float gives 3x8.
    for (int i = 0; i < 8; ++i) {
        size_t offset = header.lumps[3].fileofs + i * sizeof(dvertex_t);
        auto vertex = value_at<dvertex_t>(bytes, offset);
        vertex.point[1] = vertex.point[1] > 0 ? 960.0f : 912.0f;
        vertex.point[2] = vertex.point[2] > 0 ? -1536.0f : -1680.0f;
        memcpy(bytes.data() + offset, &vertex, sizeof vertex);
    }
    auto texture = value_at<texinfo_t>(bytes, header.lumps[6].fileofs);
    texture.vecs[0][1] = 2.0f / 3;
    texture.vecs[1][2] = -2.0f / 3;
    texture.vecs[1][3] = -128;
    memcpy(bytes.data() + header.lumps[6].fileofs, &texture, sizeof texture);
    for (int i = 0; i < 6; ++i) {
        size_t offset = header.lumps[7].fileofs + i * sizeof(dface_t);
        auto face = value_at<dface_t>(bytes, offset);
        face.lightofs = i == 0 ? 0 : -1;
        memcpy(bytes.data() + offset, &face, sizeof face);
    }
    header.lumps[8].filelen = 3 * 8 * 3;
    for (int i = 0; i < header.lumps[8].filelen; ++i)
        bytes[header.lumps[8].fileofs + i] = i;
    memcpy(bytes.data(), &header, sizeof header);
    BSP bsp(std::move(bytes));
    const auto &face = bsp.faces[0];
    if (face.u != 38 || face.v != 56 || face.width != 3 || face.height != 8 || bsp.target.rows() != 24 ||
        bsp.target(23, 2) != 71)
        throw std::runtime_error("QRAD fractional-scale lightmap grid parsing failed");
    std::cout << "BSP reader: QRAD fractional-scale lightmap grid passed\n";
}
void accuracy_self_test() {
    {
        std::array<std::vector<double>, 3> samples;
        auto detected = [&](std::array<double, 3> value) {
            for (int c = 0; c < 3; ++c) samples[c].assign(12, value[c]);
            return residual_emitter(samples, value);
        };
        if (!detected({18, 23, 1.76}) || !detected({18, 0, 0}) ||
            detected({1.9, 1.9, 1.9}) || detected({18, -1, 0}) ||
            detected({18, 0, std::numeric_limits<double>::quiet_NaN()}))
            throw std::runtime_error("Colored emitter discovery lost signal or accepted invalid residuals");
        samples[0] = {0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3};
        samples[1].assign(12, 0);
        samples[2].assign(12, 0);
        if (residual_emitter(samples, {3, 0, 0}))
            throw std::runtime_error("Weak-channel discovery accepted poorly supported residuals");
    }
    std::vector<Vote> events{{-1, 1}, {1, -1}, {-.25, 1}, {.25, -1}, {.1, 1}, {.5, -1}, {3, 1}, {4, -1}};
    auto p = maximum_overlap(events, {-5, 5});
    if (p.support != 3 || p.current != 2 || p.lo != .1 || p.hi != .25)
        throw std::runtime_error("Maximum byte-interval consensus failed");
    BSP bsp(room_fixture());
    uint32_t random = 7;
    auto sample = [&]() {
        random = random * 1664525u + 1013904223u;
        return random / double(UINT32_MAX);
    };
    for (int mode = 0; mode < 4; ++mode) {
        bsp.native_encoding = mode != 0;
        bsp.encoding = CompilerEncoding{};
        bsp.encoding.postclip = -1;
        if (mode >= 2) {
            bsp.encoding.preclip = -1;
            bsp.encoding.postclip = mode == 2 ? 188 : 255;
            bsp.encoding.minimum = mode == 3 ? 20 : 0;
            bsp.encoding.rounding = .5;
            for (int c = 0; c < 3; ++c) {
                bsp.encoding.scale[c] = 2;
                bsp.encoding.gamma[c] = .55;
            }
        }
        ByteEncoder encoder(bsp);
        for (int test = 0; test < 300; ++test) {
            Eigen::RowVector3d raw(sample() * 600, sample() * 600, sample() * 600);
            double floor = test % 3 ? 0 : 30;
            for (int moving = 0; moving < 3; ++moving)
                for (int output = 0; output < 3; ++output) {
                    int target = encoder.byte(encoder.linear(raw, floor)[output], output);
                    auto interval = sample_interval(encoder, raw, floor, moving, output, target);
                    for (double value : {-100.0, 0.0, sample() * 100, sample() * 500, sample() * 1000}) {
                        auto point = raw;
                        point[moving] = value;
                        bool actual = encoder.byte(encoder.linear(point, floor)[output], output) == target;
                        double scaled = value * encoder.settings.scale[moving];
                        bool predicted = scaled > interval.lo && scaled < interval.hi;
                        if (actual != predicted)
                            throw std::runtime_error("Clipped byte interval disagrees with encoder");
                    }
                }
        }
    }
    Probe base, bad;
    base.bytes = bsp.target.array() + 1;
    bad.bytes = bsp.target;
    bad.bytes(0, 0) += 100;
    if (accuracy_accept(bsp, bad, base, false))
        throw std::runtime_error("Byte-count improvement bypassed the residual guard");
    auto a = recipe_values("26 34 46 1000"), b = recipe_values("13 17 23 2000");
    if ((a - b).cwiseAbs().maxCoeff() != 0)
        throw std::runtime_error("Equivalent RAD representations changed emission");
    std::cout << "Accuracy: interval sweep, clipped/minimum-light bins and residual guards passed\n";
}
void compiler_self_test(const fs::path &directory, const std::string &backend) {
    BSP bsp(room_fixture());
    if (!fingerprint(bsp).preferred.empty())
        throw std::runtime_error("Unstamped BSP was falsely identified");
    BSP stamped(room_fixture("{\n\"classname\" \"worldspawn\"\n\"compiler\" \"ZHLT v3.4 VL34 (test)\"\n}\n"));
    if (fingerprint(stamped).preferred != std::vector<std::string>{"vhlt", "sdhlt"})
        throw std::runtime_error("Shared VHLT/SDHLT stamp was not kept ambiguous");
    BSP padded(room_fixture("{\n\"classname\" \"worldspawn\"\n}\n", true));
    if (fingerprint(padded).preferred != std::vector<std::string>{"vhlt", "sdhlt"})
        throw std::runtime_error("Lightmap padding fingerprint was missed");
    Options options;
    options.backend = backend;
    options.workers = 2;
    original_grid_self_test(directory, options);
    Profile profile;
    Engine engine(bsp, options, profile, directory);
    bool cuda = engine.backend() == "cuda";
    if (backend == "auto") std::cout << "Automatic backend: " << engine.backend() << '\n';
    if (cuda) lm::gpu_self_test();
    Emissions lights{{"~panel", {80, 50, 20}}};
    auto original = engine.bake(lights);
    std::string rad = "~panel 80 50 20\n";
    auto cached = engine.bake({}, true, false, &rad);
    if (!original->complete || !cached->complete ||
        (original->raw - cached->raw).cwiseAbs().maxCoeff() != 0 ||
        (original->bytes - cached->bytes).cwiseAbs().maxCoeff() != 0)
        throw std::runtime_error("Cached compiler changed lighting");
    if (!profile.compiler.geometry_hits || !profile.compiler.transfer_hits || !profile.compiler.direct_hits)
        throw std::runtime_error("Integrated compiler did not reuse geometry, direct lighting and transfers");
    auto reference = engine.bake(lights, true, true);
    if ((reference->raw - original->raw).cwiseAbs().maxCoeff() != 0 ||
        (reference->bytes - original->bytes).cwiseAbs().maxCoeff() != 0)
        throw std::runtime_error(
            "Accelerated compiler differs from uncached CPU reference: raw=" +
            std::to_string((reference->raw - original->raw).cwiseAbs().maxCoeff()) +
            ", bytes=" + std::to_string((reference->bytes - original->bytes).cwiseAbs().maxCoeff()));
    if (cuda && (profile.compiler.gpu_calls != 2 || profile.compiler.gpu_transfer_uploads != 1 ||
                 profile.compiler.gpu_transfer_reuses != 1))
        throw std::runtime_error("Integrated compiler did not reuse compatible CUDA transfers");
    if (cuda && (!profile.compiler.gpu_direct_calls || !profile.compiler.gpu_trace_calls))
        throw std::runtime_error("Integrated compiler did not use CUDA direct/transfer tracing");
    if (cuda && (!profile.compiler.gpu_indirect_calls || !profile.compiler.indirect_hits))
        throw std::runtime_error("Integrated compiler did not reuse CUDA interpolation geometry");
    {
        // An exact repeated request may reuse its result only with caching enabled.
        auto count = profile.probes;
        auto duplicate = engine.bake(lights);
        if (profile.probes != count || !profile.probe_cache_hits || duplicate != original)
            throw std::runtime_error("Identical probe did not reuse its current-run result");
        Options uncached_options = options;
        uncached_options.cache = false;
        Profile uncached_profile;
        fs::create_directories(directory / "uncached");
        Engine uncached(bsp, uncached_options, uncached_profile, directory / "uncached");
        auto first = uncached.bake(lights), second = uncached.bake(lights);
        if (uncached_profile.probes != 2 || uncached_profile.probe_cache_hits ||
            uncached_profile.compiler.geometry_hits || uncached_profile.compiler.direct_hits ||
            uncached_profile.compiler.transfer_hits || uncached_profile.compiler.indirect_hits ||
            fs::exists(directory / "uncached" / "cache") ||
            (first->raw - reference->raw).cwiseAbs().maxCoeff() != 0 ||
            (second->raw - reference->raw).cwiseAbs().maxCoeff() != 0 ||
            (second->bytes - reference->bytes).cwiseAbs().maxCoeff() != 0)
            throw std::runtime_error("Uncached mode reused lighting or changed its result");
    }
    for (float level : {24.9999f, 25.0f, 25.0001f, 80.0f}) {
        Emissions changed{{"~panel", {level, level, level}}};
        auto fast = engine.bake(changed);
        auto exact = engine.bake(changed, true, true);
        if ((fast->raw - exact->raw).cwiseAbs().maxCoeff() != 0 ||
            (fast->bytes - exact->bytes).cwiseAbs().maxCoeff() != 0)
            throw std::runtime_error("Cache or face threads changed threshold lighting at " +
                                     std::to_string(level) + ": raw=" +
                                     std::to_string((fast->raw - exact->raw).cwiseAbs().maxCoeff()) +
                                     ", direct=" +
                                     std::to_string((fast->direct - exact->direct).cwiseAbs().maxCoeff()) +
                                     ", layout=" + (fast->layout == exact->layout ? "same" : "different") +
                                     ", bytes=" +
                                     std::to_string((fast->bytes - exact->bytes).cwiseAbs().maxCoeff()));
    }
    {
        const std::string world = "{\n\"classname\" \"worldspawn\"\n\"_minlight\" \".25\"\n}\n";
        BSP floor_bsp(room_fixture(world));
        if ((floor_bsp.floor.array() != 32).any())
            throw std::runtime_error("Worldspawn minimum light mismatch");
        BSP owner_bsp(room_fixture(world + "{\n\"model\" \"*0\"\n\"_minlight\" \".5\"\n}\n"
                                           "{\n\"model\" \"*0\"\n\"_minlight\" \".75\"\n}\n"));
        if ((owner_bsp.floor.array() != 64).any())
            throw std::runtime_error("Model owner minimum light mismatch");
        BSP point_bsp(room_fixture(world +
                                   "{\n\"classname\" \"light\"\n\"origin\" \"0 0 16\"\n\"_light\" \"40\"\n}\n"
                                   "{\n\"classname\" \"light_spot\"\n\"origin\" \"0 0 0\"\n\"angle\" "
                                   "\"-1\"\n\"_light\" \"30\"\n}\n"));
        Profile point_profile;
        Engine points(point_bsp, options, point_profile, directory);
        auto point_cold = points.bake(lights);
        auto point_fast = points.bake({}, true, false, &rad), point_exact = points.bake(lights, true, true);
        if (!point_profile.compiler.direct_hits ||
            (point_cold->raw - point_exact->raw).cwiseAbs().maxCoeff() != 0 ||
            (point_fast->raw - point_exact->raw).cwiseAbs().maxCoeff() != 0 ||
            (point_fast->bytes - point_exact->bytes).cwiseAbs().maxCoeff() != 0 ||
            (cuda && !point_profile.compiler.gpu_direct_calls))
            throw std::runtime_error("Point/spot direct cache changed lighting");
    }
    if (!cuda) {
        // Place a real QRAD sample on its RGB-average membership threshold.
        // Derivative bakes may freeze that mask; ordinary candidates must not.
        auto bright = engine.bake({{"~panel", {100, 100, 100}}});
        bool crossed = false;
        for (int row = 0; row < bright->direct.rows() && !crossed; ++row) {
            double level = 100 / bright->direct.row(row).mean();
            if (!std::isfinite(level) || level < 30 || level > 300) continue;
            float base_level = float(level * (1 - .00005));
            Emissions base_lights{{"~panel", {base_level, base_level, base_level}}};
            auto base = engine.bake(base_lights);
            auto changed = base_lights;
            changed["~panel"][0] += base_level * .0005f;
            auto normal = engine.bake(changed);
            if (normal->layout == base->layout) continue;
            auto frozen = engine.bake(changed, true, false, nullptr, &base->membership);
            if (frozen->layout != base->layout ||
                (frozen->raw.rightCols(2) - base->raw.rightCols(2)).cwiseAbs().maxCoeff() != 0)
                throw std::runtime_error("Frozen derivative changed another channel or its sampling structure");
            auto exact = engine.bake(changed, true, true);
            if ((normal->raw - exact->raw).cwiseAbs().maxCoeff() != 0 ||
                (normal->raw - frozen->raw).cwiseAbs().maxCoeff() == 0)
                throw std::runtime_error("Derivative-only sampling leaked into ordinary candidate lighting");
            Fit threshold;
            threshold.names = {"~panel"};
            threshold.values = RGB::Constant(1, 3, base_level);
            threshold.probe = base;
            auto before = profile.derivative_probes;
            derivatives(bsp, threshold, engine, profile, true);
            if (profile.derivative_probes - before > 3)
                throw std::runtime_error("Frozen sample threshold still required derivative fallback");
            crossed = true;
        }
        if (!crossed) throw std::runtime_error("Fixture did not exercise a sampling membership crossing");
    }
    for (int variant = 0; variant < 3; ++variant) {
        std::string world = "{\n\"classname\" \"worldspawn\"\n}\n";
        std::string sun = "{\n\"classname\" \"light_environment\"\n\"origin\" \"0 0 16\"\n"
                          "\"_light\" \"255 160 80 100\"\n\"pitch\" \"" +
                          std::string(variant == 2 ? "90" : "-90") + "\"\n}\n";
        sun += "{\n\"classname\" \"light\"\n\"origin\" \"0 0 0\"\n\"_light\" \"30\"\n}\n";
        auto sky_work = directory / ("sky" + std::to_string(variant));
        fs::create_directories(sky_work);
        BSP sky_bsp(room_fixture(world + sun, false, variant != 0));
        Profile sky_profile;
        Engine sky_engine(sky_bsp, options, sky_profile, sky_work);
        auto cold = sky_engine.bake(lights);
        auto warm = sky_engine.bake({}, true, false, &rad);
        auto exact = sky_engine.bake(lights, true, true);
        Emissions changed{{"~panel", {120, 75, 30}}};
        auto modified = sky_engine.bake(changed);
        auto changed_exact = sky_engine.bake(changed, true, true);
        if (!sky_profile.compiler.direct_hits ||
            (cold->raw - exact->raw).cwiseAbs().maxCoeff() != 0 ||
            (warm->raw - exact->raw).cwiseAbs().maxCoeff() != 0 ||
            (warm->bytes - exact->bytes).cwiseAbs().maxCoeff() != 0 ||
            (modified->raw - changed_exact->raw).cwiseAbs().maxCoeff() != 0 ||
            (modified->bytes - changed_exact->bytes).cwiseAbs().maxCoeff() != 0)
            throw std::runtime_error("Skylight cache differs from CPU, variant " + std::to_string(variant));
        if (variant == 1) {
            auto without = sky_engine.bake(lights, false, true);
            if ((exact->raw - without->raw).cwiseAbs().maxCoeff() < 10)
                throw std::runtime_error("Sky fixture did not produce meaningful sunlight");
        }
    }
    if (!cuda) {
        // "wall" has no light-like name. Its weak blue channel must be found
        // from measured residuals, independently of the named panel emitter.
        Emissions colored{{"wall", {18, 23, 1.76f}}, {"~panel", {80, 50, 20}}};
        auto colored_reference = engine.bake(colored, true, true);
        Bytes colored_bytes = bsp.bytes;
        for (int i = 0; i < bsp.target.rows(); ++i)
            for (int c = 0; c < 3; ++c)
                colored_bytes[bsp.header.lumps[8].fileofs + i * 3 + c] = lm::checked_cast<unsigned char>(colored_reference->bytes(i, c));
        BSP colored_target(std::move(colored_bytes));
        auto colored_work = directory / "colored-discovery";
        fs::create_directories(colored_work);
        Profile colored_profile;
        Engine colored_recovery(colored_target, options, colored_profile, colored_work);
        auto colored_fit = bootstrap(colored_target, colored_recovery, colored_profile);
        auto wall = std::find(colored_fit.names.begin(), colored_fit.names.end(), "wall");
        if (wall == colored_fit.names.end())
            throw std::runtime_error("Recovery missed the unnamed weak-channel colored emitter");
        colored_fit = refine_precision(colored_target, colored_fit, colored_recovery,
                                       colored_profile, .35, true);
        if (quality(colored_target, *colored_fit.probe).mae > 1)
            throw std::runtime_error("Recovered colored-emitter recipe does not reproduce its fixture");
        colored_fit = formatted_fit(std::move(colored_fit), options.decimals, colored_recovery);
        auto exported = format_rad_output(format_rad(colored_fit, options.decimals));
        auto exported_probe = colored_recovery.bake({}, true, true, &exported);
        if ((exported_probe->raw - colored_fit.probe->raw).cwiseAbs().maxCoeff() != 0 ||
            (exported_probe->bytes - colored_fit.probe->bytes).cwiseAbs().maxCoeff() != 0)
            throw std::runtime_error("Final colored RAD export changed the verified lightmap");
    }
    if (!cuda) {
        // The emitting panel has no naming hint and every self-lit red luxel
        // clips. Only unsaturated spill onto the non-emitting walls constrains
        // its strength; clipped bytes must not become exact raw-light targets.
        BSP saturated_bsp(room_fixture("{\n\"classname\" \"worldspawn\"\n}\n",
                                       false, false, "paint"));
        auto saturated_work = directory / "saturated-discovery";
        fs::create_directories(saturated_work);
        Profile saturated_profile;
        Engine saturated_engine(saturated_bsp, options, saturated_profile, saturated_work);
        auto saturated_reference = saturated_engine.bake({{"paint", {200, 0, 0}}}, true, true);
        size_t emitter_samples = 0, clipped_samples = 0, spill_samples = 0;
        Bytes measured = saturated_bsp.bytes;
        for (int i = 0; i < saturated_bsp.target.rows(); ++i) {
            if (saturated_bsp.faces[saturated_bsp.face_ids[i]].texture == "paint") {
                ++emitter_samples;
                clipped_samples += saturated_reference->bytes(i, 0) == 255;
            } else
                spill_samples += saturated_reference->bytes(i, 0) > 5 &&
                                 saturated_reference->bytes(i, 0) < 250;
            for (int c = 0; c < 3; ++c)
                measured[saturated_bsp.header.lumps[8].fileofs + i * 3 + c] =
                    lm::checked_cast<unsigned char>(saturated_reference->bytes(i, c));
        }
        if (!emitter_samples || clipped_samples != emitter_samples || spill_samples < 10)
            throw std::runtime_error("Saturated fixture lacks clipped self-light or unsaturated spill");
        BSP saturated_target(std::move(measured));
        auto recovery_work = saturated_work / "recovery";
        fs::create_directories(recovery_work);
        Engine saturated_recovery(saturated_target, options, saturated_profile, recovery_work);
        auto fit = bootstrap(saturated_target, saturated_recovery, saturated_profile);
        if (fit.names != std::vector<std::string>{"paint"})
            throw std::runtime_error("Saturated discovery missed paint or invented a wall emitter");
        fit = refine_precision(saturated_target, fit, saturated_recovery, saturated_profile, .35, true);
        fit = formatted_fit(std::move(fit), options.decimals, saturated_recovery);
        if (quality(saturated_target, *fit.probe).mae > 1 ||
            std::abs(fit.values(0, 0) - 200) > 2 || fit.values(0, 1) > 1 || fit.values(0, 2) > 1)
            throw std::runtime_error("Saturated recovery did not recover strength from neighboring spill");
        auto exported = format_rad_output(format_rad(fit, options.decimals));
        auto checked = saturated_recovery.bake({}, true, true, &exported);
        if ((checked->raw - fit.probe->raw).cwiseAbs().maxCoeff() != 0 ||
            (checked->bytes - fit.probe->bytes).cwiseAbs().maxCoeff() != 0)
            throw std::runtime_error("Saturated RAD export changed the validated lighting");

        // The same clipped red appearance caused by a known point light must
        // not invent texture emission. Baseline compiler lighting explains it.
        BSP point_bsp(room_fixture("{\n\"classname\" \"worldspawn\"\n}\n"
            "{\n\"classname\" \"light\"\n\"origin\" \"0 0 0\"\n"
            "\"_light\" \"255 0 0 10000\"\n}\n", false, false, "paint"));
        auto point_work = saturated_work / "point-control";
        fs::create_directories(point_work);
        Profile point_profile;
        Engine point_engine(point_bsp, options, point_profile, point_work);
        auto point_reference = point_engine.bake({}, true, true);
        point_bsp.target = point_reference->bytes;
        if (point_bsp.target.col(0).maxCoeff() != 255)
            throw std::runtime_error("Point-light control does not contain clipped red samples");
        auto point_fit = bootstrap(point_bsp, point_engine, point_profile);
        if (!point_fit.names.empty() || quality(point_bsp, *point_fit.probe).mae != 0)
            throw std::runtime_error("Known saturated point lighting invented texture emission");
    }
    if (!cuda) {
        Bytes measured = bsp.bytes;
        for (int i = 0; i < bsp.target.rows(); ++i)
            for (int c = 0; c < 3; ++c)
                measured[bsp.header.lumps[8].fileofs + i * 3 + c] = lm::checked_cast<unsigned char>(reference->bytes(i, c));
        BSP target(std::move(measured));
        Profile fitting;
        Engine recovery(target, options, fitting, directory);
        auto fit = bootstrap(target, recovery, fitting);
        if (fit.names.empty())
            throw std::runtime_error("Native recovery found no fixture emitter");
        fit = refine_precision(target, fit, recovery, fitting, .35, true);
        if (quality(target, *fit.probe).mae > 2)
            throw std::runtime_error("Native recovery could not fit the generated fixture");
        fit.values.row(0) = Eigen::RowVector3d(80.25, 50.2, 20.12);
        fit.probe = recovery.bake(emissions(fit.names, fit.values));
        AccuracyReport accuracy;
        LinearModel model;
        auto refined = refine_accuracy(target, fit, recovery, options, fitting, model, accuracy);
        if (accuracy.after.style0_exact < accuracy.before.style0_exact ||
            accuracy.after.mae > accuracy.before.mae ||
            (refined.values.row(0) - Eigen::RowVector3d(80, 50, 20)).cwiseAbs().maxCoeff() > .15)
            throw std::runtime_error("Byte consensus did not recover perturbed fixture emission");
        Fit gray;
        gray.names = {"~panel"};
        gray.values = RGB::Constant(1, 3, 25);
        gray.probe = recovery.bake(emissions(gray.names, gray.values));
        target.target = gray.probe->bytes;
        AccuracyReport gray_accuracy;
        LinearModel gray_model;
        gray =
            refine_accuracy(target, std::move(gray), recovery, options, fitting, gray_model, gray_accuracy);
        if (gray.recipes != std::vector<std::string>{"25"})
            throw std::runtime_error(
                "Rounding replaced an exact simple recipe with an equal-quality alternative");
    }
    std::cout << "Integrated compiler: memory-only probes, cache reuse and CPU reference agreement passed ("
              << options.backend << ")\n";
}
void native_compiler_self_test(const fs::path &directory, const std::string &family) {
    BSP bsp(room_fixture());
    Options options;
    options.compiler = family;
    options.backend = "cpu";
    options.workers = 2;
    original_grid_self_test(directory, options);
    Profile profile;
    Engine engine(bsp, options, profile, directory);
    Emissions lights{{"~panel", {80, 50, 20}}};
    auto original = engine.bake(lights);
    auto reference = engine.bake(lights, true, true);
    if (!original->complete || !reference->complete ||
        (original->raw - reference->raw).cwiseAbs().maxCoeff() != 0 ||
        (original->bytes - reference->bytes).cwiseAbs().maxCoeff() != 0)
        throw std::runtime_error(family + " parallel and serial lighting differ");
    // Test the encoder against observed compiler bytes, not another copy of its formula.
    bsp.target = original->bytes;
    std::array<Matrix, 3> matrix;
    for (auto &channel : matrix)
        channel = Matrix::Zero(bsp.target.rows(), 1);
    auto objective = make_objective(bsp, original->raw, matrix, true);
    objective.objective = "interval";
    objective.margin = 0;
    Vector gradient;
    if (objective(Vector::Zero(3), gradient) > 1e-9)
        throw std::runtime_error(family + " raw samples do not match its byte encoder");
    auto saturated = engine.bake({{"~panel", {800, 500, 200}}});
    bsp.target = saturated->bytes;
    auto clipped_objective = make_objective(bsp, saturated->raw, matrix, true);
    clipped_objective.objective = "interval";
    clipped_objective.margin = 0;
    if (clipped_objective(Vector::Zero(3), gradient) > 1e-9)
        throw std::runtime_error(family + " saturated samples do not match its byte encoder");
    bsp.target = original->bytes;
    auto fit = bootstrap(bsp, engine, profile);
    if (fit.names.empty())
        throw std::runtime_error(family + " found no fixture emitter");
    fit = refine_precision(bsp, fit, engine, profile, .35, true);
    auto measured = quality(bsp, *fit.probe);
    if (measured.mae > .5 || (fit.values.row(0) - Eigen::RowVector3d(80, 50, 20)).cwiseAbs().maxCoeff() > 2)
        throw std::runtime_error(family + " could not recover known fixture emissions; MAE " +
                                 std::to_string(measured.mae));
    std::string text = format_rad(fit, options.decimals);
    auto formatted = engine.bake({}, true, true, &text);
    if (!formatted->complete || quality(bsp, *formatted).mae > .5)
        throw std::runtime_error(family + " formatted RAD did not reproduce the fixture");
    AccuracyReport accuracy;
    LinearModel model;
    fit = refine_accuracy(bsp, std::move(fit), engine, options, profile, model, accuracy);
    if (accuracy.after.style0_exact < accuracy.before.style0_exact ||
        accuracy.after.mae > accuracy.before.mae + 1e-6)
        throw std::runtime_error(family + " byte refinement regressed the fixture");
    std::cout << family << ": transport, threading, encoder and known-emission recovery passed\n";
}

void help() {
    std::cout << "RadBruter - recover texture lights from a GoldSrc BSP\n\n"
              << "Usage: radbruter map[.bsp] [-o lights.rad]\n\n"
              << "Compiler, CUDA/CPU and thread count are detected automatically.\n"
              << "Default output: output/<map>.rad. Live progress updates in place.\n\n"
              << "Choose Fast, Balanced or Full after the map loads.\n\n"
              << "  -o, --output FILE   Save the recovered RAD here\n"
              << "      --waddir PATH  Find external WAD textures here\n"
              << "      --quality MODE Use fast, balanced or full without prompting\n"
              << "  -qrad              Force QRAD\n"
              << "  -zhlt              Force ZHLT\n"
              << "  -vhlt              Force VHLT\n"
              << "  -sdhlt             Force SDHLT\n"
              << "  -v, --verbose       Show detailed diagnostics\n"
              << "  -h, --help          Show this help\n";
}
Options parse(int argc, char **argv) {
    Options options;
    options.compiler = "auto";
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto value = [&]() {
            if (++i == argc)
                throw std::runtime_error("Missing value after " + flag);
            return std::string(argv[i]);
        };
        auto integer = [&]() {
            const auto text = value();
            int result;
            auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
            if (error != std::errc{} || end != text.data() + text.size())
                throw std::runtime_error("Expected an integer after " + flag);
            return result;
        };
        if (flag == "--output" || flag == "-o")
            options.output = fs::weakly_canonical(fs::absolute(value()));
        else if (flag == "--profile")
            options.profile = fs::weakly_canonical(fs::absolute(value()));
        else if (flag == "--backend")
            options.backend = value();
        else if (flag == "--quality")
            options.quality = lower(value());
        else if (flag == "--compiler")
            options.compiler = lower(value());
        else if (flag == "-qrad" || flag == "-zhlt" || flag == "-vhlt" || flag == "-sdhlt")
            options.compiler = flag.substr(1);
        else if (flag == "--waddir") {
            auto directory = fs::weakly_canonical(fs::absolute(value()));
            if (!fs::is_directory(directory))
                throw std::runtime_error("WAD search directory does not exist");
            options.wad_dirs.push_back(directory);
        } else if (flag == "--detect")
            options.detect_only = true;
        else if (flag == "--bounces")
            options.bounces = integer();
        else if (flag == "--qrad-smooth")
            options.qrad_smooth = integer();
        else if (flag == "--threads" || flag == "--workers")
            options.workers = integer();
        else if (flag == "--passes") {
            options.passes = integer();
            options.passes_explicit = true;
        } else if (flag == "--decimals")
            options.decimals = integer();
        else if (flag == "--no-cache")
            options.cache = false;
        else if (flag == "--no-consensus")
            options.consensus = false;
        else if (flag == "--no-snap")
            options.snap = false;
        else if (flag == "--independent-probes")
            options.batched = false;
        else if (flag == "--no-frozen-derivatives")
            options.frozen_derivatives = false;
        else if (flag == "--verbose" || flag == "-v")
            options.verbose = true;
        else if (flag == "--self-test")
            options.self_test = true;
        else if (!flag.starts_with('-') && options.bsp.empty())
            options.bsp = fs::weakly_canonical(fs::absolute(flag));
        else
            throw std::runtime_error("Unknown argument: " + flag);
    }
    if (options.backend != "auto" && options.backend != "cpu" && options.backend != "cuda")
        throw std::runtime_error("Backend must be auto, cpu or cuda");
    if (!options.quality.empty() && options.quality != "fast" && options.quality != "balanced" &&
        options.quality != "full")
        throw std::runtime_error("Quality must be fast, balanced or full");
    if (options.compiler == "schlt" || options.compiler == "p2rad")
        throw std::runtime_error(options.compiler + " has no integrated backend; P2RAD is not SCHLT");
    if (options.compiler != "auto" && options.compiler != "qrad" && options.compiler != "zhlt" &&
        options.compiler != "vhlt" && options.compiler != "sdhlt")
        throw std::runtime_error("Compiler must be auto, qrad, zhlt, vhlt or sdhlt");
    if (options.bounces < -1 || options.bounces > 100)
        throw std::runtime_error("Invalid bounce count");
    if (options.qrad_smooth < 0 || options.qrad_smooth > 180)
        throw std::runtime_error("Invalid QRAD smoothing angle");
    if (options.qrad_smooth != 45 && options.compiler != "qrad")
        throw std::runtime_error("A QRAD smoothing override requires --compiler qrad");
    if (options.workers < 0 || options.workers > 64 || options.passes < 0 || options.passes > 10 ||
        options.decimals < 0 || options.decimals > 9)
        throw std::runtime_error("Invalid thread/pass/decimal limit");
    if (!options.self_test) {
        if (!options.bsp.empty() && options.bsp.extension().empty()) options.bsp += ".bsp";
        if (options.bsp.empty() || lower(options.bsp.extension().string()) != ".bsp" ||
            !fs::is_regular_file(options.bsp))
            throw std::runtime_error("Supply an existing .bsp file");
        if (options.output.empty())
            options.output = fs::weakly_canonical(root / "output" / (options.bsp.stem().string() + ".rad"));
        if (lower(options.output.extension().string()) != ".rad")
            throw std::runtime_error("Output must be a .rad file");
        if (!options.profile.empty() && (options.profile == options.bsp || options.profile == options.output))
            throw std::runtime_error("Profile must have a separate destination");
    }
    return options;
}
} // namespace

int main(int argc, char **argv) {
    try {
        lm::initialize_process();
        if (argc == 3 && std::string(argv[1]) == "--transport-self-test") {
            if (std::string(argv[2]) != "space \"quoted\" trailing\\")
                throw std::runtime_error("Worker argument quoting failed");
            auto bytes = read_fd(120);
            write_all(1, bytes.data(), bytes.size());
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--sleep-self-test") {
            std::this_thread::sleep_for(std::chrono::seconds(60));
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--stream-self-test") {
            FD stream(lm::connect_stream(argv[2]));
            Bytes bytes(256 * 1024);
            read_all(stream.fd, bytes.data(), bytes.size());
            write_all(stream.fd, bytes.data(), bytes.size());
            return 0;
        }
        std::signal(SIGINT, cancel_handler);
        std::signal(SIGTERM, cancel_handler);
#ifndef _WIN32
        std::signal(SIGPIPE, SIG_IGN);
#endif
        if (argc > 1 && std::string(argv[1]) == "--qrad-worker") {
            if (!getenv("LM_NO_BSP_WRITE") || !getenv("LM_BSP_INPUT") || !getenv("LM_RESULT_FD"))
                throw std::runtime_error("Compiler workers must be started through the native runner");
            for (int i = 2; i + 1 < argc; ++i)
                if (std::string(argv[i]) == "-lights" && !inside_project(argv[i + 1]))
                    throw std::runtime_error("RAD files must stay inside this project");
            // Worker signals terminate immediately; the parent owns cleanup.
            std::signal(SIGINT, SIG_DFL);
            std::signal(SIGTERM, SIG_DFL);
            return qrad_main(argc - 1, argv + 1);
        }
        if (argc == 3 && std::string(argv[1]) == "--gpu-service") {
            if (!inside_project(argv[2]))
                throw std::runtime_error("GPU socket must stay inside the project");
            std::signal(SIGINT, SIG_DFL);
            std::signal(SIGTERM, SIG_DFL);
            return lm::gpu_service(argv[2]);
        }
        if (argc == 1 || (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) {
            help();
            return 0;
        }
        auto options = parse(argc, argv);
        if (options.self_test) {
            TempDir work;
            numerical_self_test();
            bsp_grid_self_test();
            accuracy_self_test();
            process_self_test(work.path);
            lm::cache_self_test(work.path / "cache");
            fs::create_directory(work.path / "compiler");
            if (options.compiler == "auto" || options.compiler == "qrad")
                compiler_self_test(work.path / "compiler", options.backend);
            else
                native_compiler_self_test(work.path / "compiler", options.compiler);
            return 0;
        }
        recover(options);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << (cancelled ? "Cancelled; temporary files removed.\n"
                                : "Error: " + std::string(e.what()) + "\n");
        return cancelled ? 130 : 1;
    }
}
