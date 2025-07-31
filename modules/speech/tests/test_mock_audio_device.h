/**************************************************************************/
/*  test_mock_audio_device.h                                             */
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

#pragma once

#include "../mock_audio_device.h"
#include "tests/test_macros.h"

namespace TestMockAudioDevice {

TEST_CASE("[MockAudioDevice] Basic initialization") {
	MockAudioDevice device;
	
	// Test that device initializes properly
	CHECK_MESSAGE(device.get_sample_rate() > 0, "MockAudioDevice: Sample rate is positive");
	CHECK_MESSAGE(device.get_buffer_size() > 0, "MockAudioDevice: Buffer size is positive");
}

TEST_CASE("[MockAudioDevice] Audio generation") {
	MockAudioDevice device;
	
	// Test basic audio generation
	device.start_playback();
	CHECK_MESSAGE(device.is_playing(), "MockAudioDevice: Device reports playing state correctly");
	
	device.stop_playback();
	CHECK_MESSAGE(!device.is_playing(), "MockAudioDevice: Device reports stopped state correctly");
}

TEST_CASE("[MockAudioDevice] Network simulation") {
	MockAudioDevice device;
	
	// Test network delay simulation
	device.set_network_delay(100); // 100ms delay
	CHECK_MESSAGE(device.get_network_delay() == 100, "MockAudioDevice: Network delay is set correctly");
	
	// Test jitter simulation
	device.set_network_jitter(50); // 50ms jitter
	CHECK_MESSAGE(device.get_network_jitter() == 50, "MockAudioDevice: Network jitter is set correctly");
}

} // namespace TestMockAudioDevice
