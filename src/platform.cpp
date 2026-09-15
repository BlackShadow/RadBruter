#include "platform.h"
#include "hash.h"
#include "checked_cast.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <process.h>
#include <winioctl.h>
#else
#include <csignal>
#include <dlfcn.h>
#include <sched.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace lm {
namespace fs = std::filesystem;
namespace {
std::string unique_name() {
    std::random_device random;
    std::ostringstream text;
    text << std::hex << random() << random() << random() << random();
    return text.str();
}
#ifdef _WIN32
[[noreturn]] void windows_error(const char *message) {
    throw std::system_error(int(GetLastError()), std::system_category(), message);
}
struct Handle {
    HANDLE value = INVALID_HANDLE_VALUE;
    explicit Handle(HANDLE value = INVALID_HANDLE_VALUE) : value(value) {}
    ~Handle() { if (value && value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    Handle(const Handle &) = delete;
    Handle &operator=(const Handle &) = delete;
    HANDLE release() { return std::exchange(value, INVALID_HANDLE_VALUE); }
};
HANDLE handle(int fd) { return reinterpret_cast<HANDLE>(_get_osfhandle(fd)); }
int descriptor(HANDLE value) {
    if (value == INVALID_HANDLE_VALUE) return -1;
    int fd = _open_osfhandle(reinterpret_cast<intptr_t>(value), _O_RDWR | _O_BINARY | _O_NOINHERIT);
    if (fd < 0) CloseHandle(value);
    return fd;
}
std::wstring widen(const std::string &text) {
    // Match the narrow arguments and paths supplied by the Windows CRT.
    int size = MultiByteToWideChar(CP_ACP, 0, text.data(), int(text.size()), nullptr, 0);
    if (!size && !text.empty()) windows_error("Cannot encode worker argument");
    std::wstring result(size, L'\0');
    MultiByteToWideChar(CP_ACP, 0, text.data(), int(text.size()), result.data(), size);
    return result;
}
std::wstring quote(const std::string &argument) {
    std::wstring result = L"\"";
    size_t slashes = 0;
    for (wchar_t c : widen(argument)) {
        if (c == L'\\') { ++slashes; continue; }
        result.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
        slashes = 0;
        result += c;
    }
    result.append(slashes * 2, L'\\');
    result += L'"';
    return result;
}
std::wstring pipe_name(const fs::path &path) {
    auto text = path.string();
    Hash hash;
    hash.add(text.data(), text.size());
    return L"\\\\.\\pipe\\radbruter-" + widen(hash.finish());
}
HANDLE create_pipe(const std::wstring &name, bool first) {
    HANDLE pipe = CreateNamedPipeW(name.c_str(), PIPE_ACCESS_DUPLEX |
        (first ? FILE_FLAG_FIRST_PIPE_INSTANCE : 0),
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
        PIPE_UNLIMITED_INSTANCES, 65536, 65536, 0, nullptr);
    if (pipe == INVALID_HANDLE_VALUE) windows_error("Cannot create CUDA pipe");
    return pipe;
}
#endif
} // namespace

int open_file(const fs::path &path, int flags, int mode) {
#ifdef _WIN32
    DWORD access = (flags & _O_RDWR) ? GENERIC_READ | GENERIC_WRITE :
                   (flags & _O_WRONLY) ? GENERIC_WRITE : GENERIC_READ;
    DWORD creation = (flags & _O_CREAT) ? ((flags & _O_EXCL) ? CREATE_NEW :
                     (flags & _O_TRUNC) ? CREATE_ALWAYS : OPEN_ALWAYS) :
                     (flags & _O_TRUNC) ? TRUNCATE_EXISTING : OPEN_EXISTING;
    HANDLE file = CreateFileW(path.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                             nullptr, creation, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        DWORD error = GetLastError();
        errno = (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS) ? EEXIST :
                error == ERROR_FILE_NOT_FOUND ? ENOENT : EACCES;
        return -1;
    }
    int fd = _open_osfhandle(reinterpret_cast<intptr_t>(file),
                            (flags & (_O_RDWR | _O_WRONLY)) | _O_BINARY | _O_NOINHERIT);
    if (fd < 0) CloseHandle(file);
    return fd;
#else
    return ::open(path.c_str(), flags, mode);
#endif
}
int close_file(int fd) {
#ifdef _WIN32
    return _close(fd);
#else
    return ::close(fd);
#endif
}
int64_t read(int fd, void *data, size_t size) {
#ifdef _WIN32
    return _read(fd, data, unsigned(std::min(size, size_t(INT_MAX))));
#else
    return ::read(fd, data, size);
#endif
}
int64_t write(int fd, const void *data, size_t size) {
#ifdef _WIN32
    return _write(fd, data, unsigned(std::min(size, size_t(INT_MAX))));
#else
    return ::write(fd, data, size);
#endif
}
int64_t read_at(int fd, void *data, size_t size, int64_t offset) {
#ifdef _WIN32
    OVERLAPPED position{};
    position.Offset = DWORD(offset);
    position.OffsetHigh = DWORD(uint64_t(offset) >> 32);
    DWORD count;
    if (!ReadFile(handle(fd), data, DWORD(std::min(size, size_t(INT_MAX))), &count, &position)) {
        errno = EIO;
        return -1;
    }
    return count;
#else
    return ::pread(fd, data, size, offset);
#endif
}
int64_t seek(int fd, int64_t offset, int origin) {
#ifdef _WIN32
    return _lseeki64(fd, offset, origin);
#else
    return ::lseek(fd, offset, origin);
#endif
}
int file_info(int fd, FileInfo &info) {
#ifdef _WIN32
    return _fstat64(fd, &info);
#else
    return ::fstat(fd, &info);
#endif
}
bool regular_file(const FileInfo &info) {
#ifdef _WIN32
    return (info.st_mode & _S_IFMT) == _S_IFREG;
#else
    return S_ISREG(info.st_mode);
#endif
}
std::array<uint64_t, 7> file_stamp(int fd, const FileInfo &info, bool *reusable) {
#ifdef _WIN32
    BY_HANDLE_FILE_INFORMATION identity{};
    FILE_BASIC_INFO times{};
    if (!GetFileInformationByHandle(handle(fd), &identity) ||
        !GetFileInformationByHandleEx(handle(fd), FileBasicInfo, &times, sizeof times))
        windows_error("Cannot identify cache file");
    // Windows timestamps can coincide for rapid writes. The journal sequence detects those changes.
    std::array<unsigned char, 4096> buffer{};
    DWORD received = 0;
    USN_RECORD_V2 record{};
    bool versioned = DeviceIoControl(handle(fd), FSCTL_READ_FILE_USN_DATA, nullptr, 0,
        buffer.data(), DWORD(buffer.size()), &received, nullptr) && received >= sizeof record;
    if (versioned) memcpy(&record, buffer.data(), sizeof record);
    versioned = versioned && record.MajorVersion == 2 && record.Usn > 0;
    if (reusable) *reusable = versioned;
    return {identity.dwVolumeSerialNumber, (uint64_t(identity.nFileIndexHigh) << 32) | identity.nFileIndexLow,
            uint64_t(info.st_size), uint64_t(times.LastWriteTime.QuadPart), versioned ? uint64_t(record.Usn) : 0,
            uint64_t(times.ChangeTime.QuadPart), 0};
#else
    if (reusable) *reusable = true;
    return {uint64_t(info.st_dev), uint64_t(info.st_ino), uint64_t(info.st_size),
            uint64_t(info.st_mtim.tv_sec), uint64_t(info.st_mtim.tv_nsec),
            uint64_t(info.st_ctim.tv_sec), uint64_t(info.st_ctim.tv_nsec)};
#endif
}
bool try_lock(int fd) {
#ifdef _WIN32
    OVERLAPPED position{};
    if (LockFileEx(handle(fd), LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &position))
        return true;
    if (GetLastError() == ERROR_LOCK_VIOLATION) return false;
    windows_error("Cannot lock lighting cache");
#else
    if (!flock(fd, LOCK_EX | LOCK_NB)) return true;
    if (errno == EWOULDBLOCK) return false;
    throw std::system_error(errno, std::generic_category(), "Cannot lock lighting cache");
#endif
}
void *map_readonly(int fd, size_t size) {
#ifdef _WIN32
    Handle mapping(CreateFileMappingW(handle(fd), nullptr, PAGE_READONLY, 0, 0, nullptr));
    return mapping.value ? MapViewOfFile(mapping.value, FILE_MAP_READ, 0, 0, size) : nullptr;
#else
    void *data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    return data == MAP_FAILED ? nullptr : data;
#endif
}
void unmap(void *data, size_t size) {
#ifdef _WIN32
    UnmapViewOfFile(data);
#else
    munmap(data, size);
#endif
}
int memory_file(const fs::path &directory, const char *name) {
#ifdef _WIN32
    auto path = directory / (std::string(name) + "-" + unique_name());
    return descriptor(CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, CREATE_NEW,
        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr));
#else
    return memfd_create(name, MFD_CLOEXEC);
#endif
}
int temporary_file(std::string &pattern) {
#ifdef _WIN32
    auto prefix = pattern.substr(0, pattern.size() - 6);
    for (int attempt = 0; attempt < 32; ++attempt) {
        pattern = prefix + unique_name();
        int fd = open_file(pattern, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC);
        if (fd >= 0 || errno != EEXIST) return fd;
    }
    return -1;
#else
    int fd = mkstemp(pattern.data());
    if (fd >= 0) fcntl(fd, F_SETFD, FD_CLOEXEC);
    return fd;
#endif
}
fs::path temporary_directory(const fs::path &root) {
#ifdef _WIN32
    for (int attempt = 0; attempt < 32; ++attempt) {
        auto path = root / (".radbruter-run-" + unique_name());
        if (fs::create_directory(path)) return path;
    }
#else
    std::string pattern = (root / ".radbruter-run-XXXXXX").string();
    if (mkdtemp(pattern.data())) return pattern;
#endif
    throw std::runtime_error("Cannot create temporary workspace");
}
void atomic_replace(const fs::path &from, const fs::path &to) {
#ifdef _WIN32
    if (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw fs::filesystem_error("Cannot publish file", from, to,
                                   std::error_code(int(GetLastError()), std::system_category()));
#else
    fs::rename(from, to);
#endif
}
fs::path executable_path() {
#ifdef _WIN32
    std::wstring path(32768, L'\0');
    DWORD length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    if (!length || length == path.size()) windows_error("Cannot locate executable");
    path.resize(length);
    return path;
#else
    return fs::read_symlink("/proc/self/exe");
#endif
}
uint64_t process_id() {
#ifdef _WIN32
    return GetCurrentProcessId();
#else
    return getpid();
#endif
}
bool process_exists(uint64_t id) {
#ifdef _WIN32
    Handle process(OpenProcess(SYNCHRONIZE, FALSE, DWORD(id)));
    return process.value && WaitForSingleObject(process.value, 0) == WAIT_TIMEOUT;
#else
    return kill(pid_t(id), 0) == 0;
#endif
}
double cpu_seconds() {
#ifdef _WIN32
    FILETIME created, exited, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user))
        windows_error("Cannot read process timing");
    return (((uint64_t(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime) +
            ((uint64_t(user.dwHighDateTime) << 32) | user.dwLowDateTime)) * 1e-7;
#else
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6 +
           usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6;
#endif
}
std::pair<unsigned, uint64_t> available_resources() {
    unsigned available = std::max(1u, std::thread::hardware_concurrency());
    uint64_t memory = 512ULL << 20;
#ifdef _WIN32
    DWORD_PTR process_mask, system_mask;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        available = 0;
        for (; process_mask; process_mask >>= 1) available += unsigned(process_mask & 1);
    }
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof status;
    if (GlobalMemoryStatusEx(&status)) memory = status.ullAvailPhys;
#else
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    if (!sched_getaffinity(0, sizeof cpus, &cpus)) available = unsigned(CPU_COUNT(&cpus));
    std::ifstream quota_file("/sys/fs/cgroup/cpu.max");
    std::string quota;
    uint64_t period;
    if (quota_file >> quota >> period && quota != "max" && period)
        available = std::min(available, unsigned(std::max<uint64_t>(1, std::stoull(quota) / period)));
    std::ifstream mem("/proc/meminfo");
    std::string line;
    while (std::getline(mem, line))
        if (line.rfind("MemAvailable:", 0) == 0) {
            std::istringstream row(line.substr(13));
            row >> memory;
            memory *= 1024;
        }
    std::ifstream max_file("/sys/fs/cgroup/memory.max"), current_file("/sys/fs/cgroup/memory.current");
    std::string maximum;
    uint64_t current;
    if (max_file >> maximum && maximum != "max" && current_file >> current) {
        uint64_t cap = std::stoull(maximum);
        memory = std::min(memory, cap > current ? cap - current : 0);
    }
#endif
    return {std::max(1u, available), memory};
}
std::map<std::string, std::string> process_environment() {
    std::map<std::string, std::string> result;
#ifdef _WIN32
    char **entries = _environ;
#else
    char **entries = environ;
#endif
    for (char **p = entries; p && *p; ++p) {
        std::string entry = *p;
        auto equal = entry.find('=');
        if (equal != std::string::npos && equal) result[entry.substr(0, equal)] = entry.substr(equal + 1);
    }
    return result;
}
void initialize_process() {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    for (int target = 120; target <= 124; ++target) {
        auto name = "LM_INHERITED_FD_" + std::to_string(target);
        const char *value = getenv(name.c_str());
        if (!value) continue;
        int source = descriptor(reinterpret_cast<HANDLE>(std::stoull(value)));
        if (source < 0 || _dup2(source, target)) {
            if (source >= 0) _close(source);
            throw std::runtime_error("Cannot inherit lighting transport");
        }
        _close(source);
        _putenv_s(name.c_str(), "");
    }
#endif
}
bool terminal_input() {
#ifdef _WIN32
    DWORD mode = 0;
    return GetConsoleMode(handle(0), &mode) != 0;
#else
    return isatty(STDIN_FILENO) != 0;
#endif
}
unsigned terminal_width(int fd) {
#ifdef _WIN32
    DWORD mode = 0;
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (!GetConsoleMode(handle(fd), &mode) || !GetConsoleScreenBufferInfo(handle(fd), &info)) return 0;
    return unsigned(info.srWindow.Right - info.srWindow.Left + 1);
#else
    const char *term = std::getenv("TERM");
    if (!isatty(fd) || (term && std::string(term) == "dumb")) return 0;
    winsize size{};
    return !ioctl(fd, TIOCGWINSZ, &size) && size.ws_col ? size.ws_col : 80;
#endif
}
const char *input_descriptor_path() {
#ifdef _WIN32
    return "@radbruter-input";
#else
    return "/proc/self/fd/120";
#endif
}

int original_grid(int face, int face_count, int *mins, int *size) {
    static const auto grids = [] {
        std::vector<LightmapGrid> result;
        const char *value = getenv("LM_GRID_FD");
        if (!value) return result;
        int fd = std::stoi(value);
        FileInfo info{};
        if (fd < 0 || file_info(fd, info) || info.st_size < 8 ||
            info.st_size > 8 + 65536 * int64_t(sizeof(LightmapGrid)))
            throw std::runtime_error("Invalid original lightmap grid data");
        auto read_exact = [fd](void *data, size_t bytes, int64_t offset) {
            auto p = static_cast<unsigned char *>(data);
            while (bytes) {
                auto count = read_at(fd, p, bytes, offset);
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0) throw std::runtime_error("Cannot read original lightmap grids");
                p += count;
                bytes -= size_t(count);
                offset += count;
            }
        };
        std::array<uint32_t, 2> header{};
        read_exact(header.data(), sizeof header, 0);
        if (header[0] != grid_magic || header[1] > 65536 ||
            info.st_size != int64_t(sizeof header + size_t(header[1]) * sizeof(LightmapGrid)))
            throw std::runtime_error("Invalid original lightmap grid header");
        result.resize(header[1]);
        read_exact(result.data(), result.size() * sizeof(LightmapGrid), sizeof header);
        for (const auto &grid : result) {
            if (!grid.width && !grid.height) continue;
            if (grid.width < 1 || grid.height < 1 || grid.width > 256 || grid.height > 256 ||
                grid.u < -(1 << 26) || grid.u > (1 << 26) || grid.v < -(1 << 26) || grid.v > (1 << 26))
                throw std::runtime_error("Invalid original lightmap grid dimensions");
        }
        return result;
    }();
    if (grids.empty()) return 0;
    if (face < 0 || face >= face_count || grids.size() != size_t(face_count))
        throw std::runtime_error("Original lightmap grids do not match the BSP");
    const auto &grid = grids[size_t(face)];
    if (!grid.width) return 0;
    mins[0] = grid.u;
    mins[1] = grid.v;
    size[0] = grid.width - 1;
    size[1] = grid.height - 1;
    return 1;
}

SharedLibrary::SharedLibrary(const fs::path &path) {
#ifdef _WIN32
    DWORD flags = path.is_absolute() ? LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
                                    : LOAD_LIBRARY_SEARCH_SYSTEM32;
    DWORD previous_mode = 0;
    BOOL changed_mode = SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &previous_mode);
    handle = LoadLibraryExW(path.c_str(), nullptr, flags);
    if (changed_mode) SetThreadErrorMode(previous_mode, nullptr);
#else
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (!handle) throw std::runtime_error("Unavailable GPU library: " + path.filename().string());
}
SharedLibrary::~SharedLibrary() {
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}
void *SharedLibrary::symbol(const char *name) const {
#ifdef _WIN32
    auto address = reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    auto address = dlsym(handle, name);
#endif
    if (!address) throw std::runtime_error(std::string("Unavailable GPU function: ") + name);
    return address;
}

struct Child::State {
#ifdef _WIN32
    HANDLE process = nullptr, job = nullptr;
    DWORD pid = 0;
#else
    pid_t pid = -1;
#endif
};
Child::Child(std::function<void()> check) : state(std::make_unique<State>()), check_cancelled(std::move(check)) {}
Child::~Child() { stop(); }
uint64_t Child::id() const { return uint64_t(state->pid); }
void Child::stop() {
#ifdef _WIN32
    if (!state->process) return;
    TerminateJobObject(state->job, 1);
    WaitForSingleObject(state->process, INFINITE);
    CloseHandle(state->process);
    CloseHandle(state->job);
    state->process = state->job = nullptr;
    state->pid = 0;
#else
    if (state->pid < 0) return;
    kill(-state->pid, SIGTERM);
    for (int i = 0; i < 50; ++i) {
        int status;
        pid_t ended = waitpid(state->pid, &status, WNOHANG);
        if (ended == state->pid || (ended < 0 && errno == ECHILD)) { state->pid = -1; return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    kill(-state->pid, SIGKILL);
    while (waitpid(state->pid, nullptr, 0) < 0 && errno == EINTR) {}
    state->pid = -1;
#endif
}
bool Child::exited() {
#ifdef _WIN32
    if (!state->process) return true;
    DWORD status = WaitForSingleObject(state->process, 0);
    if (status == WAIT_FAILED) windows_error("Cannot poll worker");
    return status == WAIT_OBJECT_0;
#else
    if (state->pid < 0) return true;
    int status;
    if (waitpid(state->pid, &status, WNOHANG) == state->pid) { state->pid = -1; return true; }
    return false;
#endif
}
void Child::start(const std::vector<std::string> &arguments,
                  const std::map<std::string, std::string> &environment,
                  const std::vector<std::pair<int, int>> &descriptors) {
    stop();
    if (arguments.empty()) throw std::runtime_error("Missing worker executable");
#ifdef _WIN32
    auto env = environment;
    std::vector<std::unique_ptr<Handle>> owned;
    std::vector<HANDLE> inherited;
    std::map<int, HANDLE> reopened;
    SECURITY_ATTRIBUTES security{sizeof security, nullptr, TRUE};
    Handle null(CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            &security, OPEN_EXISTING, 0, nullptr));
    if (null.value == INVALID_HANDLE_VALUE) windows_error("Cannot open worker streams");
    inherited.push_back(null.value);
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof startup;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = startup.StartupInfo.hStdOutput = startup.StartupInfo.hStdError = null.value;
    for (auto [from, to] : descriptors) {
        HANDLE file;
        auto found = reopened.find(from);
        if (found != reopened.end()) file = found->second;
        else {
            auto copy = std::make_unique<Handle>(ReOpenFile(handle(from), GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0));
            if (copy->value == INVALID_HANDLE_VALUE ||
                !SetHandleInformation(copy->value, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT))
                windows_error("Cannot duplicate worker transport");
            file = copy->value;
            reopened[from] = file;
            inherited.push_back(file);
            owned.push_back(std::move(copy));
        }
        if (to == 0) startup.StartupInfo.hStdInput = file;
        else if (to == 1) startup.StartupInfo.hStdOutput = file;
        else if (to == 2) startup.StartupInfo.hStdError = file;
        else env["LM_INHERITED_FD_" + std::to_string(to)] = std::to_string(reinterpret_cast<uintptr_t>(file));
    }
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    std::vector<unsigned char> attributes(bytes);
    startup.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
    if (!InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &bytes))
        windows_error("Cannot prepare worker handles");
    struct Attributes {
        LPPROC_THREAD_ATTRIBUTE_LIST list;
        ~Attributes() { DeleteProcThreadAttributeList(list); }
    } cleanup{startup.lpAttributeList};
    if (!UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                   inherited.data(), inherited.size() * sizeof(HANDLE), nullptr, nullptr))
        windows_error("Cannot restrict worker handles");
    std::wstring command;
    for (const auto &arg : arguments) { if (!command.empty()) command += L' '; command += quote(arg); }
    std::vector<std::wstring> entries;
    for (const auto &[key, value] : env) entries.push_back(widen(key + "=" + value));
    std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) { return _wcsicmp(a.c_str(), b.c_str()) < 0; });
    std::wstring block;
    for (const auto &entry : entries) { block += entry; block += L'\0'; }
    block += L'\0';
    Handle job(CreateJobObjectW(nullptr, nullptr));
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!job.value || !SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof limits))
        windows_error("Cannot create worker job");
    PROCESS_INFORMATION info{};
    if (!CreateProcessW(widen(arguments[0]).c_str(), command.data(), nullptr, nullptr, TRUE,
        CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
        block.data(), nullptr, &startup.StartupInfo, &info)) windows_error("Cannot launch native lighting worker");
    Handle process(info.hProcess), thread(info.hThread);
    if (!AssignProcessToJobObject(job.value, process.value) || ResumeThread(thread.value) == DWORD(-1)) {
        DWORD error = GetLastError();
        TerminateProcess(process.value, 1);
        WaitForSingleObject(process.value, INFINITE);
        SetLastError(error);
        windows_error("Cannot supervise native lighting worker");
    }
    state->process = process.release();
    state->job = job.release();
    state->pid = info.dwProcessId;
#else
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    for (auto [from, to] : descriptors) posix_spawn_file_actions_adddup2(&actions, from, to);
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attributes, 0);
    std::vector<std::string> entries;
    std::vector<char *> envp, argv;
    for (const auto &[key, value] : environment) entries.push_back(key + "=" + value);
    for (auto &entry : entries) envp.push_back(entry.data());
    envp.push_back(nullptr);
    for (const auto &arg : arguments) argv.push_back(const_cast<char *>(arg.c_str()));
    argv.push_back(nullptr);
    int status = posix_spawn(&state->pid, arguments[0].c_str(), &actions, &attributes, argv.data(), envp.data());
    posix_spawnattr_destroy(&attributes);
    posix_spawn_file_actions_destroy(&actions);
    if (status) { state->pid = -1; throw std::system_error(status, std::generic_category(), "Cannot launch native lighting worker"); }
#endif
}
uint64_t Child::wait(int &code) {
#ifdef _WIN32
    if (!state->process) throw std::runtime_error("No running worker");
    while (!exited()) {
        if (check_cancelled) check_cancelled();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    DWORD result;
    if (!GetExitCodeProcess(state->process, &result)) windows_error("Cannot read worker exit code");
    PROCESS_MEMORY_COUNTERS memory{};
    GetProcessMemoryInfo(state->process, &memory, sizeof memory);
    code = int(result);
    auto peak = uint64_t(memory.PeakWorkingSetSize);
    CloseHandle(state->process);
    CloseHandle(state->job);
    state->process = state->job = nullptr;
    state->pid = 0;
    return peak;
#else
    rusage usage{};
    int status;
    for (;;) {
        pid_t ended = wait4(state->pid, &status, WNOHANG, &usage);
        if (ended == state->pid) break;
        if (ended < 0 && errno != EINTR) throw std::runtime_error("Cannot wait for compiler worker");
        if (check_cancelled) check_cancelled();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    state->pid = -1;
    code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    return uint64_t(usage.ru_maxrss) * 1024;
#endif
}

int connect_stream(const fs::path &path) {
#ifdef _WIN32
    auto name = pipe_name(path);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    for (;;) {
        HANDLE pipe = CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (pipe != INVALID_HANDLE_VALUE) return descriptor(pipe);
        if (GetLastError() != ERROR_PIPE_BUSY || std::chrono::steady_clock::now() >= deadline)
            windows_error("Cannot connect to CUDA worker");
        WaitNamedPipeW(name.c_str(), 1000);
    }
#else
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    auto text = path.string();
    if (fd < 0 || text.size() >= sizeof address.sun_path) {
        if (fd >= 0) ::close(fd);
        throw std::runtime_error("Invalid CUDA socket path");
    }
    strcpy(address.sun_path, text.c_str());
    if (connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof address)) {
        ::close(fd);
        throw std::runtime_error("Cannot connect to CUDA worker");
    }
    return fd;
#endif
}
struct StreamServer::State {
    fs::path path;
#ifdef _WIN32
    std::wstring name;
    HANDLE pending = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
    ~State() {
#ifdef _WIN32
        if (pending != INVALID_HANDLE_VALUE) CloseHandle(pending);
#else
        if (fd >= 0) ::close(fd);
#endif
        std::error_code error;
        fs::remove(path, error);
    }
};
StreamServer::StreamServer(const fs::path &path) : state(std::make_unique<State>()) {
    state->path = path;
#ifdef _WIN32
    state->name = pipe_name(path);
    state->pending = create_pipe(state->name, true);
    std::ofstream ready(path, std::ios::binary);
    if (!ready) throw std::runtime_error("Cannot publish CUDA service readiness");
#else
    state->fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    auto text = path.string();
    if (text.size() >= sizeof address.sun_path) throw std::runtime_error("GPU socket path too long");
    strcpy(address.sun_path, text.c_str());
    if (state->fd < 0 || bind(state->fd, reinterpret_cast<sockaddr *>(&address), sizeof address) || listen(state->fd, 64))
        throw std::runtime_error("Cannot open CUDA worker socket");
    chmod(path.c_str(), 0600);
#endif
}
StreamServer::~StreamServer() = default;
int StreamServer::accept() {
#ifdef _WIN32
    if (!ConnectNamedPipe(state->pending, nullptr) && GetLastError() != ERROR_PIPE_CONNECTED)
        windows_error("CUDA pipe accept failed");
    HANDLE next = create_pipe(state->name, false);
    return descriptor(std::exchange(state->pending, next));
#else
    return accept4(state->fd, nullptr, nullptr, SOCK_CLOEXEC);
#endif
}
} // namespace lm
