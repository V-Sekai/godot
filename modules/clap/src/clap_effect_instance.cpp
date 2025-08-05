#include "clap_effect_instance.h"

#include <clap_plugin_host.h>

void ClapAudioEffectInstance::_bind_methods() {
}

ClapAudioEffectInstance::ClapAudioEffectInstance() {
}

void ClapAudioEffectInstance::_process(const void *p_src_buffer, AudioFrame *p_dst_buffer, int32_t p_frame_count) {
	host->process(p_src_buffer, p_dst_buffer, p_frame_count);
}

bool ClapAudioEffectInstance::_process_silence() const {
	return host->process_silence();
}
