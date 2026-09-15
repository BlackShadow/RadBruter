#pragma once
#include "platform.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace lm {
// Refresh one console line. Redirected output and verbose mode retain plain-text logs.
class Progress {
    using Clock = std::chrono::steady_clock;
    std::mutex mutex;
    std::atomic<uint64_t> bakes{0}, evaluations{0};
    Clock::time_point start = Clock::now(), last_update = start, phase_start = start, pause_start = start;
    std::string message;
    int remaining_phases = -1;
    double seconds_per_phase = 0;
    bool enabled, interactive, paused = false;
    size_t displayed = 0, frame = 0;
    std::jthread ticker;

    static std::string duration_text(long long seconds) {
        if (seconds >= 3600)
            return std::to_string(seconds / 3600) + "h" + std::to_string(seconds / 60 % 60) + "m";
        if (seconds >= 60) return std::to_string(seconds / 60) + "m" + std::to_string(seconds % 60) + "s";
        return std::to_string(seconds) + "s";
    }
    std::string eta_text(Clock::time_point now) const {
        if (remaining_phases < 1 || !std::isfinite(seconds_per_phase) || seconds_per_phase <= 0)
            return " | ETA estimating";
        double elapsed = std::chrono::duration<double>(now - phase_start).count();
        double remaining = seconds_per_phase * remaining_phases - elapsed;
        // An overrun makes the estimate stale; never promise zero seconds while work remains.
        if (remaining <= 0) return " | ETA recalculating";
        return " | ETA ~" + duration_text(static_cast<long long>(std::ceil(remaining)));
    }

    void clear_line() {
        if (!displayed) return;
        auto width = terminal_width(2);
        auto count = width ? std::min(displayed, size_t(width - 1)) : displayed;
        std::cerr << '\r' << std::string(count, ' ') << '\r' << std::flush;
        displayed = 0;
    }
    void render(bool changed) {
        if (!enabled || paused || message.empty()) return;
        auto now = Clock::now();
        if (!changed && !interactive && now - last_update < std::chrono::seconds(5)) return;
        last_update = now;
        std::string bake_text = " | " + std::to_string(bakes.load(std::memory_order_relaxed)) + " bakes";
        auto count = evaluations.load(std::memory_order_relaxed);
        std::string evaluation_text = count ? " | " + std::to_string(count) + " evals" : "";
        std::string elapsed_text = " | " + duration_text(std::chrono::duration_cast<std::chrono::seconds>(now - start).count());
        std::string eta = eta_text(now);
        std::string suffix = bake_text + evaluation_text + elapsed_text + eta;
        if (!interactive) {
            std::cerr << message << suffix << '\n';
            return;
        }
        auto width = terminal_width(2);
        if (!width) return;
        size_t limit = width - 1;
        // Keep the map/compiler and ETA visible on narrow consoles before optional counters.
        if (message.size() + suffix.size() + 2 > limit) suffix = bake_text + elapsed_text + eta;
        if (message.size() + suffix.size() + 2 > limit) suffix = bake_text + eta;
        if (message.size() + suffix.size() + 2 > limit) suffix = eta;
        std::string stage = message;
        size_t room = limit > suffix.size() + 2 ? limit - suffix.size() - 2 : 0;
        if (stage.size() > room) stage = room > 3 ? stage.substr(0, room - 3) + "..." : stage.substr(0, room);
        std::string text(1, "|/-\\"[frame++ % 4]);
        text += ' ' + stage + suffix;
        if (text.size() > limit) text.resize(limit);
        size_t previous = std::min(displayed, limit);
        std::cerr << '\r' << text << std::string(previous > text.size() ? previous - text.size() : 0, ' ')
                  << std::flush;
        displayed = text.size();
    }

  public:
    explicit Progress(bool verbose = false, bool enabled = true)
        : enabled(enabled), interactive(enabled && !verbose && terminal_width(2) > 0) {
        if (enabled) ticker = std::jthread([this](std::stop_token stop) {
            while (!stop.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                if (stop.stop_requested()) break;
                try {
                    std::lock_guard lock(mutex);
                    render(false);
                } catch (...) { return; } // Display failures must not terminate the recovery.
            }
        });
    }
    ~Progress() { finish(); }
    Progress(const Progress &) = delete;
    Progress &operator=(const Progress &) = delete;
    void set(std::string text, int phases = -1, double phase_seconds = 0) {
        std::lock_guard lock(mutex);
        message = std::move(text);
        remaining_phases = phases;
        seconds_per_phase = phase_seconds;
        phase_start = Clock::now();
        render(true);
    }
    void note(const std::string &text) {
        std::lock_guard lock(mutex);
        clear_line();
        std::cerr << text << '\n';
    }
    void pause() {
        std::lock_guard lock(mutex);
        if (paused) return;
        paused = true;
        pause_start = Clock::now();
        clear_line();
    }
    Clock::duration resume() {
        std::lock_guard lock(mutex);
        if (!paused) return Clock::duration::zero();
        auto waited = Clock::now() - pause_start;
        start += waited;
        phase_start += waited;
        last_update = Clock::now();
        paused = false;
        return waited;
    }
    void baked() { bakes.fetch_add(1, std::memory_order_relaxed); }
    void evaluated() { evaluations.fetch_add(1, std::memory_order_relaxed); }
    void finish() {
        if (ticker.joinable()) {
            ticker.request_stop();
            ticker.join();
        }
        std::lock_guard lock(mutex);
        enabled = false;
        clear_line();
    }
};
} // namespace lm
