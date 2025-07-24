/**************************************************************************/
/*  video_stream_mkv.cpp                                                  */
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

#include "video_stream_mkv.h"

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "servers/audio_server.h"

// libsimplewebm
#include <OpusVorbisDecoder.hpp>
#include <WebMDemuxer.hpp>

// libwebm
#include <mkvparser/mkvparser.h>

class MkvReader : public mkvparser::IMkvReader {
public:
	MkvReader(const String &p_file) {
		file = FileAccess::open(p_file, FileAccess::READ);

		ERR_FAIL_COND_MSG(file.is_null(), "Failed loading resource: '" + p_file + "'.");
	}
	~MkvReader() {
	}

	virtual int Read(long long pos, long len, unsigned char *buf) {
		if (file.is_valid()) {
			if (file->get_position() != (uint64_t)pos) {
				file->seek(pos);
			}
			if (file->get_buffer(buf, len) == (uint64_t)len) {
				return 0;
			}
		}
		return -1;
	}

	virtual int Length(long long *total, long long *available) {
		if (file.is_valid()) {
			const uint64_t len = file->get_length();
			if (total) {
				*total = len;
			}
			if (available) {
				*available = len;
			}
			return 0;
		}
		return -1;
	}

private:
	Ref<FileAccess> file;
};

VideoStreamPlaybackMKV::VideoStreamPlaybackMKV() {}

VideoStreamPlaybackMKV::~VideoStreamPlaybackMKV() {
	delete_pointers();
}

bool VideoStreamPlaybackMKV::open_file(const String &p_file) {
	file_name = p_file;
	webm = memnew(WebMDemuxer(new MkvReader(file_name), 0, audio_track));
	if (webm->isOpen()) {
		// Store video metadata for external handling - with null checks
		if (webm->getVideoCodec() != WebMDemuxer::NO_VIDEO) {
			video_width = webm->getWidth();
			video_height = webm->getHeight();
		} else {
			// No video track or unsupported video codec - use default dimensions
			video_width = 640;
			video_height = 480;
			print_verbose("VideoStreamMKV: No supported video track found, using default dimensions");
		}
		video_duration = webm->getLength();

		// Only handle audio decoding
		if (webm->getAudioCodec() != WebMDemuxer::NO_AUDIO) {
			audio = memnew(OpusVorbisDecoder(*webm));
			if (audio->isOpen()) {
				audio_frame = memnew(WebMFrame);
				pcm = (float *)memalloc(sizeof(float) * audio->getBufferSamples() * webm->getChannels());
			} else {
				memdelete(audio);
				audio = nullptr;
				print_verbose("VideoStreamMKV: Failed to open audio decoder");
			}
		} else {
			print_verbose("VideoStreamMKV: No supported audio track found");
		}

		// Create a placeholder texture for video metadata
		Vector<uint8_t> placeholder_data;
		placeholder_data.resize(video_width * video_height * 4);
		// Fill with black pixels
		for (int i = 0; i < placeholder_data.size(); i += 4) {
			placeholder_data.write[i] = 0; // R
			placeholder_data.write[i + 1] = 0; // G
			placeholder_data.write[i + 2] = 0; // B
			placeholder_data.write[i + 3] = 255; // A
		}

		Ref<Image> img = Image::create_from_data(video_width, video_height, false, Image::FORMAT_RGBA8, placeholder_data);
		placeholder_texture = memnew(ImageTexture);
		placeholder_texture->set_image(img);

		return true;
	}
	memdelete(webm);
	webm = nullptr;
	return false;
}

void VideoStreamPlaybackMKV::stop() {
	if (playing) {
		delete_pointers();

		pcm = nullptr;
		audio_frame = nullptr;
		audio = nullptr;

		open_file(file_name); // This should not fail here.

		num_decoded_samples = 0;
		samples_offset = -1;
		video_pos = 0.0;
	}
	playing = false;
	time = 0;
}

void VideoStreamPlaybackMKV::play() {
	if (!playing) {
		time = 0;
	} else {
		stop();
	}
	playing = true;
	delay_compensation = GLOBAL_GET("audio/video/video_delay_compensation_ms");
	delay_compensation /= 1000.0;
}

bool VideoStreamPlaybackMKV::is_playing() const {
	return playing;
}

void VideoStreamPlaybackMKV::set_paused(bool p_paused) {
	paused = p_paused;
}

bool VideoStreamPlaybackMKV::is_paused() const {
	return paused;
}

double VideoStreamPlaybackMKV::get_length() const {
	if (webm) {
		return webm->getLength();
	}
	return 0.0f;
}

double VideoStreamPlaybackMKV::get_playback_position() const {
	return video_pos;
}

void VideoStreamPlaybackMKV::seek(double p_time) {
	if (!webm) {
		return;
	}
	time = webm->seek(p_time);
	video_pos = time;
}

void VideoStreamPlaybackMKV::set_audio_track(int p_idx) {
	audio_track = p_idx;
}

Ref<Texture2D> VideoStreamPlaybackMKV::get_texture() const {
	return placeholder_texture;
}

void VideoStreamPlaybackMKV::update(double p_delta) {
	if ((!playing || paused) || !webm) {
		return;
	}

	time += p_delta;

	if (time < video_pos) {
		return;
	}

	bool audio_buffer_full = false;

	if (samples_offset > -1) {
		//Mix remaining samples
		const int to_read = num_decoded_samples - samples_offset;
		const int mixed = mix_callback(mix_udata, pcm + samples_offset * webm->getChannels(), to_read);
		if (mixed != to_read) {
			samples_offset += mixed;
			audio_buffer_full = true;
		} else {
			samples_offset = -1;
		}
	}

	const bool hasAudio = (audio && mix_callback);

	// Only process audio frames - video frames are handled externally
	if (hasAudio && !audio_buffer_full) {
		WebMFrame video_frame; // Dummy frame for demuxer

		if (!webm->readFrame(&video_frame, audio_frame)) {
			// Can't demux, EOS?
			if (webm->isEOS()) {
				stop();
			}
			return;
		}

		if (audio_frame->isValid() && audio->getPCMF(*audio_frame, pcm, num_decoded_samples) && num_decoded_samples > 0) {
			const int mixed = mix_callback(mix_udata, pcm, num_decoded_samples);

			if (mixed != num_decoded_samples) {
				samples_offset = mixed;
				audio_buffer_full = true;
			}
		}

		// Update video position based on audio frame timing
		if (audio_frame->isValid()) {
			video_pos = audio_frame->time;
		}
	}

	if (webm && webm->isEOS()) {
		stop();
	}
}

void VideoStreamPlaybackMKV::set_mix_callback(VideoStreamPlayback::AudioMixCallback p_callback, void *p_userdata) {
	mix_callback = p_callback;
	mix_udata = p_userdata;
}

int VideoStreamPlaybackMKV::get_channels() const {
	if (audio) {
		return webm->getChannels();
	}
	return 0;
}

int VideoStreamPlaybackMKV::get_mix_rate() const {
	if (audio) {
		return webm->getSampleRate();
	}
	return 0;
}

void VideoStreamPlaybackMKV::delete_pointers() {
	if (pcm) {
		memfree(pcm);
		pcm = nullptr;
	}

	if (audio_frame) {
		memdelete(audio_frame);
		audio_frame = nullptr;
	}

	if (audio) {
		memdelete(audio);
		audio = nullptr;
	}

	if (webm) {
		memdelete(webm);
		webm = nullptr;
	}
}

VideoStreamMKV::VideoStreamMKV() {}

Ref<VideoStreamPlayback> VideoStreamMKV::instantiate_playback() {
	Ref<VideoStreamPlaybackMKV> pb = memnew(VideoStreamPlaybackMKV);
	pb->set_audio_track(audio_track);
	if (pb->open_file(file)) {
		return pb;
	}
	return nullptr;
}

void VideoStreamMKV::_bind_methods() {
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "file", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "set_file", "get_file");
}

void VideoStreamMKV::set_audio_track(int p_track) {
	audio_track = p_track;
}

Ref<Resource> ResourceFormatLoaderMKV::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		if (r_error) {
			*r_error = ERR_CANT_OPEN;
		}
		return Ref<Resource>();
	}

	VideoStreamMKV *stream = memnew(VideoStreamMKV);
	stream->set_file(p_path);

	Ref<VideoStreamMKV> mkv_stream = Ref<VideoStreamMKV>(stream);

	if (r_error) {
		*r_error = OK;
	}

	f->flush();
	return mkv_stream;
}

void ResourceFormatLoaderMKV::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("mkv");
	p_extensions->push_back("webm"); // WebM is a subset of MKV
}

bool ResourceFormatLoaderMKV::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "VideoStream");
}

String ResourceFormatLoaderMKV::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "mkv" || el == "webm") {
		return "VideoStreamMKV";
	}
	return "";
}
