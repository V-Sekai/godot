//  SPDX-FileCopyrightText: 2023 Martin Heistermann <martin.heistermann@unibe.ch>
//  SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <tuple>
#include <cassert>
#include <ostream>
#include <algorithm>

#include <libTimekeeper/Duration.hh>
#include <libTimekeeper/detail/ClockPosix.hh>
#include <libTimekeeper/detail/ClockWindows.hh>

namespace Timekeeper {

class Clock
{
public:
  using WallClock = std::chrono::steady_clock;
  void start() {
    wall_start_ = WallClock::now();
    std::tie(user_start_, system_start_) = detail::user_system_now();
  }

  [[nodiscard]]
  Duration elapsed() {
    auto [user_end, system_end] = detail::user_system_now();
    auto wall_dur = std::chrono::duration_cast<Duration::Dur>(WallClock::now() - wall_start_);
    auto user_dur = user_end - user_start_;
    auto system_dur =  system_end - system_start_;
    return {.wallclock = wall_dur, .user = user_dur, .system = system_dur};
  }
private:
  WallClock::time_point wall_start_;
  Duration::Dur user_start_;
  Duration::Dur system_start_;
};

/**
 * Accumulate time durations and number of intervals between resume/stop calls.
 */
class StopWatch
{
public:
    void reset() {
      elapsed_ = Duration::zero();
      count_ = 0;
    }

    void resume() {
        assert(!running_); running_ = true;
        clock_.start();
        ++count_;
    }

    void stop()
    {
        assert(running_); running_ = false;
        elapsed_ += clock_.elapsed();
    }

    size_t count() const { return count_; }
    bool running() const {return running_;}
    Duration elapsed() const { return elapsed_;}

private:
    Clock clock_;
    Duration elapsed_ = Duration::zero();

    bool running_ = false;
    size_t count_ = 0; // how often have we resumed this stopwatch?
};

class HierarchicalStopWatch
{
public:
    HierarchicalStopWatch(const std::string &name)
        : name_(name)
    {}

    HierarchicalStopWatch(const std::string &name,
                          HierarchicalStopWatch &parent)
        : name_(name)
    {
        parent.add_child(this);
        // TODO: should remove this from parent again in destructor!
        // Or always keep this in a shared_ptr and hand that to the parent
    }

    void reset()
    {
        watch_.reset();
        for (auto ch: children_) {
            ch->reset();
        }
    }
    void resume() { watch_.resume(); }
    void stop() { watch_.stop();}

    constexpr const std::string &name() const { return name_; }
    constexpr StopWatch       &watch()       { return watch_; }
    constexpr StopWatch const &watch() const { return watch_; }
    size_t count() const { return watch_.count(); }
    /// Own clock is not used, just a total of children
    bool is_group() const { return count() == 0; }
    Duration elapsed() const {
      if (is_group()) {
        Duration total = Duration::zero();
        for (auto child: children_) {
            total += child->watch().elapsed();
        }
        return total;
      } else {
        return watch_.elapsed();
      }
    }
    [[deprecated]]
    Duration unaccounted() const
    {
      if (is_group()) {
        return Duration::zero();
      } else {
        Duration dur = watch_.elapsed();
        for (auto child: children_) {
            dur -= child->watch().elapsed();
        }
        return dur;
      }
    }

    std::vector<HierarchicalStopWatch*>::const_iterator children_begin() const {return children_.cbegin();}
    std::vector<HierarchicalStopWatch*>::const_iterator children_end()   const {return children_.cend();}

private:
    friend std::ostream& operator<<(std::ostream& os, const HierarchicalStopWatch &sw);

    void add_child(HierarchicalStopWatch *sw)
    {
        children_.push_back(sw);
    }

    std::string name_;
    StopWatch watch_;
    std::vector<HierarchicalStopWatch*> children_;
};

std::ostream& operator<<(std::ostream& os, const HierarchicalStopWatch &sw);




/**
 * RAII helper to conveniently time the execution of parts of the code.
 * If the given watch is stopped at construction time, it will resume it and stop when it goes out of scope.
 * If it is already running at construction time, do nothing (not upon destruction either).
 */
class ScopedStopWatch
{
public:
    [[nodiscard]] ScopedStopWatch(StopWatch &watch)
        : watch_(watch)
    {
        maybe_start();
    }
    [[nodiscard]] ScopedStopWatch(HierarchicalStopWatch &hsw)
        : ScopedStopWatch(hsw.watch())
    {}

    ~ScopedStopWatch()
    {
        if (should_stop_) {
            watch_.stop();
        }
    }
private:
    void maybe_start()
    {
        if (watch_.running()) {
            should_stop_ = false;
        } else {
            should_stop_ = true;
            watch_.resume();
        }
    }

    StopWatch &watch_;
    bool should_stop_ = true;
};

struct HierarchicalStopWatchResult
{
    HierarchicalStopWatchResult(const HierarchicalStopWatch &hsw)
        : name(hsw.name())
        , duration(hsw.elapsed())
        , count(hsw.count())
    {
        auto cbegin = hsw.children_begin();
        auto cend = hsw.children_end();
        children.reserve(std::distance(cbegin, cend));
        std::for_each(cbegin, cend,
                  [this](const HierarchicalStopWatch *ch)
        {
            children.emplace_back(*ch);
        });
    }
    bool is_group() const { return count == 0;}
    void add_child(HierarchicalStopWatchResult hsw) {
        children.push_back(std::move(hsw));
    }
    Duration unaccounted() const {
      if (is_group()) {
        return Duration::zero();
      } else {
        Duration dur = duration;
        for (const auto &child: children) {
            dur -= child.duration;
        }
        return dur;
      }
    }

    std::string name;
    Duration duration;
    size_t count;
    std::vector<HierarchicalStopWatchResult> children;
};

} //namespace Timekeeper


