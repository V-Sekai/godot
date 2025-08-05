#include "example_class.h"

#include "clap/clap.h"

#include <clap_effect_instance.h>
#include <clap_plugin_host.h>

#include <dlfcn.h>
#include <godot_cpp/classes/os.hpp>

void ClapAudioEffect::_bind_methods() {
	godot::ClassDB::bind_method(D_METHOD("print_type", "variant"), &ClapAudioEffect::print_type);
}

Ref<AudioEffectInstance> ClapAudioEffect::_instantiate() {
	ClapPluginHost *host = new ClapPluginHost();

	static constexpr char path[] = "/Users/lukas/Library/Audio/Plug-Ins/CLAP/Apricot.clap/Contents/MacOS/Apricot";
	if (!host->load(path, 0)) {
		print_line("Error loading audio effect");
		return {};
	}

	host->activate(48000, 512);

	// FIXME Crashes the program, probably because it's not set up.
	// if (!host->_plugin->guiShow()) {
	// 	print_line("Error showing GUI");
	// 	return {};
	// }

	Ref<ClapAudioEffectInstance> effect_instance = { memnew(ClapAudioEffectInstance) };
	effect_instance->host = host;

	return effect_instance;
}

void ClapAudioEffect::print_type(const Variant &p_variant) const {
	clap_plugin_entry entry;
	entry.clap_version = CLAP_VERSION;
	print_line(vformat("CLAP version: %d.%d.%d", entry.clap_version.major, entry.clap_version.minor, entry.clap_version.revision));
}
