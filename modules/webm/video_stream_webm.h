/*************************************************************************/
/*  video_stream_webm.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef VIDEO_STREAM_WEBM_H
#define VIDEO_STREAM_WEBM_H

#include "core/io/resource_loader.h"
#include "scene/resources/texture.h"
#include "scene/resources/video_stream.h"

class WebMFrame;
class WebMDemuxer;
class VPXDecoder;
class OpusVorbisDecoder;

class VideoStreamPlaybackWebm : public VideoStreamPlayback {
	GDCLASS(VideoStreamPlaybackWebm, VideoStreamPlayback);

	String file_name;
	int audio_track = 0;

	WebMDemuxer *webm = nullptr;
	VPXDecoder *video = nullptr;
	OpusVorbisDecoder *audio = nullptr;

	WebMFrame **video_frames = nullptr, *audio_frame = nullptr;
	int64_t video_frames_pos = 0, video_frames_capacity = 0;

	int num_decoded_samples = 0, samples_offset = -1;
	AudioMixCallback mix_callback = nullptr;
	void *mix_udata = nullptr;

	bool playing = false, paused = false;
	double delay_compensation = 0.0;
	double time = 0.0, video_frame_delay = 0.0, video_pos = 0.0;

	Vector<uint8_t> frame_data;
	Ref<ImageTexture> texture = memnew(ImageTexture);

	float *pcm = nullptr;

public:
	VideoStreamPlaybackWebm();
	virtual ~VideoStreamPlaybackWebm() override;

	bool open_file(const String &p_file);

	virtual void stop() override;
	virtual void play() override;

	virtual bool is_playing() const override;

	virtual void set_paused(bool p_paused) override;
	virtual bool is_paused() const override;

	virtual double get_length() const override;

	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;

	virtual void set_audio_track(int p_idx) override;

	virtual Ref<Texture2D> get_texture() const override;
	virtual void update(double p_delta) override;

	virtual void set_mix_callback(AudioMixCallback p_callback, void *p_userdata) override;
	virtual int get_channels() const override;
	virtual int get_mix_rate() const override;

private:
	inline bool has_enough_video_frames() const;
	bool should_process(WebMFrame &video_frame);

	void delete_pointers();
};

/**/

class VideoStreamWebm : public VideoStream {
	GDCLASS(VideoStreamWebm, VideoStream);

	int audio_track = 0;

protected:
	static void _bind_methods();

public:
	VideoStreamWebm();

	virtual Ref<VideoStreamPlayback> instantiate_playback() override;

	virtual void set_audio_track(int p_track) override;
};

class ResourceFormatLoaderWebm : public ResourceFormatLoader {
public:
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#endif // VIDEO_STREAM_WEBM_H
