//  SPDX-FileCopyrightText: 2023 Martin Heistermann <martin.heistermann@unibe.ch>
//  SPDX-License-Identifier: MIT
#pragma once
#include <chrono>

namespace Timekeeper {
struct Duration {
  using Dur = std::chrono::microseconds;
  Dur wallclock;
  Dur user;
  Dur system;
  static Duration zero() {
    return {Dur::zero(), Dur::zero(), Dur::zero()};
  }
  Duration &operator+=(Duration const& other) {
    wallclock += other.wallclock;
    user      += other.user;
    system    += other.system;
    return *this;
  }
  Duration &operator-=(Duration const& other) {
    wallclock -= other.wallclock;
    user      -= other.user;
    system    -= other.system;
    return *this;
  }
};
}
