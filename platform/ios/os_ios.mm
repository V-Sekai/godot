/**************************************************************************/
/*  os_ios.mm                                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#import "os_ios.h"

#import "display_server_ios.h"
#include "servers/display_server_embedded.h"

#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/machine.h>

#ifdef IOS_ENABLED

OS_IOS *OS_IOS::get_singleton() {
	return (OS_IOS *)OS_AppleEmbedded::get_singleton();
}

OS_IOS::OS_IOS() :
		OS_AppleEmbedded() {
#ifndef LIBGODOT_ENABLED
	DisplayServerIOS::register_ios_driver();
#endif
	DisplayServerEmbedded::register_embedded_driver();
}

OS_IOS::~OS_IOS() {
	AudioDriverManager::reset();
}

String OS_IOS::get_name() const {
	return "iOS";
}

String OS_IOS::get_processor_name() const {
	NSMutableString *ns_cpu = [[NSMutableString alloc] init];
    size_t size;
	cpu_type_t type;
	cpu_subtype_t subtype;
	size = sizeof(type);
	sysctlbyname("hw.cputype", &type, &size, NULL, 0);

	size = sizeof(subtype);
	sysctlbyname("hw.cpusubtype", &subtype, &size, NULL, 0);

	// values for cputype and cpusubtype defined in mach/machine.h
	if (type == CPU_TYPE_X86_64) {
		[ns_cpu appendString:@"x86_64"];
	} else if (type == CPU_TYPE_X86) {
		[ns_cpu appendString:@"x86"];
	} else if (type == CPU_TYPE_ARM) {
		[ns_cpu appendString:@"ARM"];
		switch(subtype)
		{
			case CPU_SUBTYPE_ARM_V6:
				[ns_cpu appendString:@"V6"];
				break;
			case CPU_SUBTYPE_ARM_V7:
				[ns_cpu appendString:@"V7"];
				break;
			case CPU_SUBTYPE_ARM_V8:
				[ns_cpu appendString:@"V8"];
				break;
		}
	} else if (type == CPU_TYPE_ARM64) {
		[ns_cpu appendString:@"ARM64"];
	}
	if ([ns_cpu length] != 0) {
		return String::utf8([ns_cpu UTF8String]);
	}
	ERR_FAIL_V_MSG("", String("Couldn't get the CPU model name. Returning an empty string."));
}

#endif // IOS_ENABLED
