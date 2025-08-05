/**************************************************************************/
/*  clap_effect_instance.cpp                                              */
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

#include "clap_effect_instance.h"
#include "clap_plugin_host.h"

ClapAudioEffectInstance::ClapAudioEffectInstance() {
	host = nullptr;
}

ClapAudioEffectInstance::~ClapAudioEffectInstance() {
	if (host) {
		delete host;
		host = nullptr;
	}
}

void ClapAudioEffectInstance::process(const AudioFrame *p_src_buffer, AudioFrame *p_dst_buffer, int32_t p_frame_count) {
	if (host && host->_plugin.get() && host->isPluginActive()) {
		host->process(p_src_buffer, p_dst_buffer, p_frame_count);
	} else {
		// Passthrough if no host or plugin
		for (int32_t i = 0; i < p_frame_count; i++) {
			p_dst_buffer[i] = p_src_buffer[i];
		}
	}
}

bool ClapAudioEffectInstance::process_silence() const {
	if (host) {
		return host->process_silence();
	}
	return false; // No silence processing if no host
}
