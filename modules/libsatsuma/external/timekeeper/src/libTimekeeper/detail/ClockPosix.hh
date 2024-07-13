//  SPDX-FileCopyrightText: 2023 Martin Heistermann <martin.heistermann@unibe.ch>
//  SPDX-License-Identifier: MIT
#pragma once

#include <utility>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))

extern "C" {
#include <sys/time.h>
#include <sys/resource.h>
}

#include <libTimekeeper/Duration.hh>
#include <chrono>

namespace Timekeeper::detail {

constexpr inline Duration::Dur duration_from_timeval(timeval const &tv)
{
    return std::chrono::duration_cast<Duration::Dur>(
        std::chrono::seconds{tv.tv_sec}
    + std::chrono::microseconds{tv.tv_usec});
}

inline std::pair<Duration::Dur, Duration::Dur> user_system_now()
{
  struct rusage ru;
  ::getrusage(RUSAGE_SELF, &ru);
  return {detail::duration_from_timeval(ru.ru_utime),
          detail::duration_from_timeval(ru.ru_stime)};
}

} // namespace Timekeeper::detail

#endif
