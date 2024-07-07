/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "platform_api_vmcore.h"
#include <time.h>

#if defined(__MINGW32__)
#include <_mingw.h>
// llvm-mingw 10 adds timespec_get. We cannot redefine it in newer compilers.
#if __MINGW64_VERSION_MAJOR < 10
static int timespec_get(struct timespec *spec, int)
{
    __int64 wintime;
    GetSystemTimeAsFileTime((FILETIME *)&wintime);
    wintime -= 116444736000000000ll;            //1jan1601 to 1jan1970
    spec->tv_sec = wintime / 10000000ll;        //seconds
    spec->tv_nsec = wintime % 10000000ll * 100; //nano-seconds
    return 0;
}
#endif
#endif
uint64
os_time_get_boot_microsecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);

    return ((uint64)ts.tv_sec) * 1000 * 1000 + ((uint64)ts.tv_nsec) / 1000;
}

