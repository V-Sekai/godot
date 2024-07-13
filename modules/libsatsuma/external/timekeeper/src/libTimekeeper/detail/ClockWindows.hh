//  SPDX-FileCopyrightText: 2023 Martin Heistermann <martin.heistermann@unibe.ch>
//  SPDX-License-Identifier: MIT
#pragma once

#if (defined _WIN32) || (defined __MINGW32__)

#include <libTimekeeper/Duration.hh>
#include <chrono>

#ifdef NOMINMAX
#  include <Windows.h>
#else
#  define NOMINMAX
#  include <Windows.h>
#  undef NOMINMAX
#endif


#include <processthreadsapi.h>

namespace Timekeeper::detail {

inline std::pair<Duration::Dur, Duration::Dur> user_system_now()
{
  FILETIME creation, exit;
  uint64_t user = 0, system = 0; // number of 100ns ticks
  ::GetProcessTimes(::GetCurrentProcess(), &creation, &exit,
      reinterpret_cast<LPFILETIME>(&system), reinterpret_cast<LPFILETIME>(&user));
  auto user_dur   = duration_cast<Duration::Dur>(std::chrono::microseconds(user/10));
  auto system_dur = duration_cast<Duration::Dur>(std::chrono::microseconds(system/10));
  return {user_dur, system_dur};
}

} // namespace Timekeeper::detail

#endif // windows
