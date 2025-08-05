#pragma once

#include "godot_cpp/classes/audio_effect.hpp"
#include "godot_cpp/classes/ref_counted.hpp"
#include "godot_cpp/classes/wrapped.hpp"
#include "godot_cpp/variant/variant.hpp"

#include <clap/clap.h>
#include <clap/helpers/event-list.hh>
#include <clap/helpers/reducing-param-queue.hh>

using namespace godot;

class ClapAudioEffect : public AudioEffect {
	GDCLASS(ClapAudioEffect, AudioEffect)

protected:
	static void _bind_methods();

public:
	ClapAudioEffect() = default;
	~ClapAudioEffect() override = default;

	Ref<AudioEffectInstance> _instantiate() override;

	void print_type(const Variant &p_variant) const;
};
