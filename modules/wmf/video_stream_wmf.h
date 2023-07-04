#ifndef WINDOWS_MEDIA_FOUNDATION_VIDEO_H
#define WINDOWS_MEDIA_FOUNDATION_VIDEO_H

#include "core/io/resource_loader.h"
#include "core/os/mutex.h"
#include "scene/resources/video_stream.h"

#include <deque>

class MediaGrabberCallback;
struct IMFMediaSession;
struct IMFMediaSource;
struct IMFTopology;
struct IMFPresentationClock;

struct FrameData {
	int64_t sample_time = 0;
	Vector<uint8_t> data;
};

class VideoStreamPlaybackWMF : public VideoStreamPlayback {
	GDCLASS(VideoStreamPlaybackWMF, VideoStreamPlayback);

	IMFMediaSession *media_session;
	IMFMediaSource *media_source;
	IMFTopology *topology;
	IMFPresentationClock *presentation_clock;
	MediaGrabberCallback *sample_grabber_callback;

	Vector<FrameData> cache_frames;
	int read_frame_idx = 0;
	int write_frame_idx = 0;

	Vector<uint8_t> frame_data;
	Ref<ImageTexture> texture;
	Mutex mtx;

	bool is_video_playing = false;
	bool is_video_paused = false;
	bool is_video_seekable = false;

	int id = 0;

	AudioMixCallback mix_callback = nullptr;
	void *mix_udata = nullptr;

	double time = 0;
	
	void shutdown_stream();

public:
	struct StreamInfo {
		Point2i size;
		float fps = 0.0f;
		float duration = 0.0f;
	};
	StreamInfo stream_info;

	virtual void play() override;
	virtual void stop() override;
	virtual bool is_playing() const override;

	virtual void set_paused(bool p_paused) override;
	virtual bool is_paused() const override;

	virtual double get_length() const override;
	virtual String get_stream_name() const;

	virtual int get_loop_count() const;

	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;

	void set_file(const String &p_file);

	virtual Ref<Texture2D> get_texture() const override;
	virtual void update(double p_delta) override;

	virtual void set_mix_callback(AudioMixCallback p_callback, void *p_userdata) override;
	virtual int get_channels() const override;
	virtual int get_mix_rate() const override;

	virtual void set_audio_track(int p_idx) override;

	FrameData *get_next_writable_frame();
	void write_frame_done();
	void present();

	int64_t next_sample_time();

	VideoStreamPlaybackWMF();
	~VideoStreamPlaybackWMF();
};

class VideoStreamWMF : public VideoStream {
	GDCLASS(VideoStreamWMF, VideoStream);
	RES_BASE_EXTENSION("wmfvstr");

	String file;
	int audio_track = 0;

protected:
	static void _bind_methods();

public:
	Ref<VideoStreamPlayback> instantiate_playback() override;

	void set_file(const String &p_file);

	String get_file() {
		return file;
	}

	void set_audio_track(int p_track) override {
		audio_track = p_track;
	}

	VideoStreamWMF();
	~VideoStreamWMF();
};

#endif
