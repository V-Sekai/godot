/**************************************************************************/
/*  clap_audio_effect.cpp                                                 */
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

#include "clap_audio_effect.h"

#include "clap/clap.h"

#include "clap_effect_instance.h"
#include "clap_plugin_host.h"

#include "core/object/class_db.h"
#include "core/string/print_string.h"
#include "servers/audio_server.h"

void ClapAudioEffect::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_plugin_path", "path"), &ClapAudioEffect::set_plugin_path);
	ClassDB::bind_method(D_METHOD("get_plugin_path"), &ClapAudioEffect::get_plugin_path);
	ClassDB::bind_method(D_METHOD("print_type", "variant"), &ClapAudioEffect::print_type);

	// GUI management methods
	ClassDB::bind_method(D_METHOD("has_gui"), &ClapAudioEffect::has_gui);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "plugin_path", PROPERTY_HINT_GLOBAL_FILE, "*"), "set_plugin_path", "get_plugin_path");
}

void ClapAudioEffect::set_plugin_path(const String &p_path) {
	plugin_path = p_path;
}

String ClapAudioEffect::get_plugin_path() const {
	return plugin_path;
}

Ref<AudioEffectInstance> ClapAudioEffect::instantiate() {
	ClapPluginHost *host = new ClapPluginHost();

	if (plugin_path.is_empty()) {
		print_line("CLAP: Error - plugin_path is not set");
		delete host;
		// Return a valid but inactive effect instance to prevent crashes
		Ref<ClapAudioEffectInstance> effect_instance = { memnew(ClapAudioEffectInstance) };
		effect_instance->host = nullptr; // Will cause passthrough behavior
		return effect_instance;
	}

	print_line("CLAP: Attempting to load plugin from: " + plugin_path);

	if (!host->load(plugin_path.utf8().get_data(), 0)) {
		print_line("CLAP: Error loading audio effect from: " + plugin_path);
		delete host;
		// Return a valid but inactive effect instance to prevent crashes
		Ref<ClapAudioEffectInstance> effect_instance = { memnew(ClapAudioEffectInstance) };
		effect_instance->host = nullptr; // Will cause passthrough behavior
		return effect_instance;
	}

	print_line("CLAP: Plugin loaded successfully, activating plugin");

	// Activate the plugin immediately on the main thread
	// Get audio settings from AudioServer
	double sample_rate = AudioServer::get_singleton()->get_mix_rate();
	host->activate(sample_rate, 256);

	// Auto-show GUI if the plugin has one
	if (host->hasGui()) {
		print_line("CLAP: Plugin has GUI, automatically showing it");
		if (host->createGui()) {
			host->showGui();
		}
	}

	// Create the effect instance
	Ref<ClapAudioEffectInstance> effect_instance = { memnew(ClapAudioEffectInstance) };
	effect_instance->host = host;

	return effect_instance;
}

void ClapAudioEffect::print_type(const Variant &p_variant) const {
	clap_plugin_entry entry;
	entry.clap_version = CLAP_VERSION;
	print_line(vformat("CLAP version: %d.%d.%d", entry.clap_version.major, entry.clap_version.minor, entry.clap_version.revision));
}

// GUI management methods
bool ClapAudioEffect::has_gui() const {
	// We need to get the host from an active instance
	// For now, we'll need to create a temporary host to check GUI support
	if (plugin_path.is_empty()) {
		return false;
	}

	ClapPluginHost *temp_host = new ClapPluginHost();
	bool has_gui = false;

	if (temp_host->load(plugin_path.utf8().get_data(), 0)) {
		has_gui = temp_host->hasGui();
	}

	delete temp_host;
	return has_gui;
}
