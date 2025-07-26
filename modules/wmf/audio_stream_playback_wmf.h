/**************************************************************************/
/*  audio_stream_playback_wmf.h                                           */
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

#include "servers/audio/audio_stream.h"
#include "wmf_audio_decoder.h"
#include "core/os/mutex.h"
#include "scene/resources/audio_video_synchronizer.h"

class AudioStreamPlaybackWMF : public AudioStreamPlayback {
	GDCLASS(AudioStreamPlaybackWMF, AudioStreamPlayback);

private:
	WMFAudioDecoder *audio_decoder = nullptr;
	Ref<AudioVideoSynchronizer> synchronizer;
	
	bool playing_state = false;
	bool paused_state = false;
	double playback_position = 0.0;
	
	// Audio mixing
	Vector<float> mix_buffer;
	int mix_rate = 44100;
	int channels = 2;
	
	Mutex audio_mutex;

public:
	AudioStreamPlaybackWMF();
	~AudioStreamPlaybackWMF();
	
	// AudioStreamPlayback interface
	virtual void start(double p_from_pos = 0.0) override;
	virtual void stop() override;
	virtual bool is_playing() const override;
	
	virtual int get_loop_count() const override;
	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;
	
	virtual int mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) override;
	
	// WMF specific
	void set_audio_decoder(WMFAudioDecoder *p_decoder);
	void set_paused(bool p_paused);
	bool is_paused() const { return paused_state; }
	
	// Synchronizer integration
	void set_synchronizer(const Ref<AudioVideoSynchronizer> &p_synchronizer);
	Ref<AudioVideoSynchronizer> get_synchronizer() const;
};

class AudioStreamWMF : public AudioStream {
	GDCLASS(AudioStreamWMF, AudioStream);

private:
	WMFAudioDecoder *audio_decoder = nullptr;
	double length = 0.0;
	Ref<AudioVideoSynchronizer> shared_synchronizer;

protected:
	static void _bind_methods();

public:
	// AudioStream interface
	virtual Ref<AudioStreamPlayback> instantiate_playback() override;
	virtual String get_stream_name() const override;
	virtual double get_length() const override;
	virtual bool is_monophonic() const override { return false; }
	
	// WMF specific
	void set_audio_decoder(WMFAudioDecoder *p_decoder);
	void set_length(double p_length) { length = p_length; }
	void set_shared_synchronizer(const Ref<AudioVideoSynchronizer> &p_synchronizer) { shared_synchronizer = p_synchronizer; }

	AudioStreamWMF();
	~AudioStreamWMF();
};
