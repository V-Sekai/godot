/**************************************************************************/
/*  vr_util.cpp                                                           */
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

#include "vr_util.h"

bool enable_passthrough(Ref<XRInterface> p_xr_interface, Viewport *p_viewport) {
	ERR_FAIL_COND_V(!p_xr_interface.is_valid() || !p_xr_interface->is_initialized(), false);
	ERR_FAIL_NULL_V(p_viewport, false);

	if (!p_xr_interface->is_passthrough_supported()) {
		print_verbose("Passthrough is not supported");
		return false;
	}

	p_viewport->set_transparent_background(true);
	if (!p_xr_interface->start_passthrough()) {
		print_verbose("Unable to start passthrough");
		p_viewport->set_transparent_background(false);
		return false;
	}

	print_verbose("Passthrough enabled");
	return true;
}

void disable_passthrough(Ref<XRInterface> p_xr_interface, Viewport *p_viewport) {
	ERR_FAIL_COND(!p_xr_interface.is_valid() || !p_xr_interface->is_initialized());
	ERR_FAIL_NULL(p_viewport);

	if (!p_xr_interface->is_passthrough_enabled()) {
		return;
	}

	p_xr_interface->stop_passthrough();
	p_viewport->set_transparent_background(false);
	print_verbose("Passthrough disabled");
}

void update_passthrough_mode(Ref<XRInterface> p_xr_interface, Viewport *p_viewport) {
	bool should_enable_passthrough = EDITOR_GET("interface/xr_editor/enable_passthrough");
	if (should_enable_passthrough) {
		enable_passthrough(p_xr_interface, p_viewport);
	} else {
		disable_passthrough(p_xr_interface, p_viewport);
	}
}

void configure_xr_interface(Ref<XRInterface> p_xr_interface, Viewport *p_viewport) {
	ERR_FAIL_COND(!p_xr_interface.is_valid() || !p_xr_interface->is_initialized());
	ERR_FAIL_NULL(p_viewport);

	p_xr_interface->set_play_area_mode(XRInterface::XR_PLAY_AREA_ROOMSCALE);

	// Make sure V-Sync is OFF or our monitor frequency will limit our headset
	// TODO improve this to only override v-sync when the player is wearing the headset
	DisplayServer::get_singleton()->window_set_vsync_mode(DisplayServer::VSYNC_DISABLED);

	// Enable our viewport for VR use
	p_viewport->set_vrs_mode(Viewport::VRS_XR);
	p_viewport->set_use_xr(true);
}
