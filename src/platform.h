#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#ifndef O_CLOEXEC
#define O_CLOEXEC _O_NOINHERIT
#endif
#endif

namespace lm {
#ifdef _WIN32
using FileInfo = struct _stat64;
inline constexpr const char *executable_suffix = ".exe";
#else
using FileInfo = struct stat;
inline constexpr const char *executable_suffix = "";
#endif
int open_file(const std::filesystem::path &path, int flags, int mode = 0600);
int close_file(int fd);
int64_t read(int fd, void *data, size_t size);
int64_t write(int fd, const void *data, size_t size);
int64_t read_at(int fd, void *data, size_t size, int64_t offset);
int64_t seek(int fd, int64_t offset, int origin);
int file_info(int fd, FileInfo &info);
bool regular_file(const FileInfo &info);
std::array<uint64_t, 7> file_stamp(int fd, const FileInfo &info, bool *reusable = nullptr);
bool try_lock(int fd);
void *map_readonly(int fd, size_t size);
void unmap(void *data, size_t size);
int memory_file(const std::filesystem::path &directory, const char *name);
int temporary_file(std::string &pattern);
std::filesystem::path temporary_directory(const std::filesystem::path &root);
void atomic_replace(const std::filesystem::path &from, const std::filesystem::path &to);
std::filesystem::path executable_path();
uint64_t process_id();
bool process_exists(uint64_t id);
double cpu_seconds();
std::pair<unsigned, uint64_t> available_resources();
std::map<std::string, std::string> process_environment();
void initialize_process();
unsigned terminal_width(int fd);
bool terminal_input();
const char *input_descriptor_path();

struct LightmapGrid {
    int32_t u = 0, v = 0, width = 0, height = 0;
};
static_assert(sizeof(LightmapGrid) == 16);
inline constexpr uint32_t grid_magic = 0x52424731;
int original_grid(int face, int face_count, int *mins, int *size);

class SharedLibrary {
    void *handle = nullptr;

  public:
    explicit SharedLibrary(const std::filesystem::path &path);
    ~SharedLibrary();
    SharedLibrary(const SharedLibrary &) = delete;
    SharedLibrary &operator=(const SharedLibrary &) = delete;
    void *symbol(const char *name) const;
};

class Child {
    struct State;
    std::unique_ptr<State> state;
    std::function<void()> check_cancelled;

  public:
    explicit Child(std::function<void()> check = {});
    ~Child();
    Child(const Child &) = delete;
    Child &operator=(const Child &) = delete;
    void start(const std::vector<std::string> &arguments,
               const std::map<std::string, std::string> &environment,
               const std::vector<std::pair<int, int>> &descriptors);
    void stop();
    bool exited();
    uint64_t id() const;
    uint64_t wait(int &code);
};

// Stream descriptors use the same binary read/write operations as probe files.
int connect_stream(const std::filesystem::path &path);
class StreamServer {
    struct State;
    std::unique_ptr<State> state;

  public:
    explicit StreamServer(const std::filesystem::path &path);
    ~StreamServer();
    int accept();
};
} // namespace lm
