#ifndef MediaGrabberCallback_H
#define MediaGrabberCallback_H

#include "core/io/resource_loader.h"
#include "core/os/mutex.h"
#include "scene/resources/video_stream.h"
#include <mfidl.h>

class VideoStreamPlaybackWMF;

class MediaGrabberCallback : public IMFSampleGrabberSinkCallback {
	long m_cRef = 0;
	VideoStreamPlaybackWMF *playback;
	int width = 0;
	int height = 0;

	IMFTransform *m_pColorTransform = nullptr;
	IMFSample *m_pSample = nullptr;
	IMFSample *m_pOutSample = nullptr;

	MediaGrabberCallback(VideoStreamPlaybackWMF *playback);

public:
	static HRESULT CreateInstance(MediaGrabberCallback **ppCB, VideoStreamPlaybackWMF *playback);
	virtual ~MediaGrabberCallback();

	// IUnknown methods
	STDMETHODIMP QueryInterface(REFIID iid, void **ppv);
	STDMETHODIMP_(ULONG)
	AddRef();
	STDMETHODIMP_(ULONG)
	Release();

	STDMETHODIMP OnClockStart(MFTIME hnsSystemTime, LONGLONG llClockStartOffset);
	STDMETHODIMP OnClockStop(MFTIME hnsSystemTime);
	STDMETHODIMP OnClockPause(MFTIME hnsSystemTime);
	STDMETHODIMP OnClockRestart(MFTIME hnsSystemTime);
	STDMETHODIMP OnClockSetRate(MFTIME hnsSystemTime, float flRate);

	STDMETHODIMP OnSetPresentationClock(IMFPresentationClock *pClock);
	STDMETHODIMP OnProcessSample(REFGUID guidMajorMediaType, DWORD dwSampleFlags,
			LONGLONG llSampleTime, LONGLONG llSampleDuration, const BYTE *pSampleBuffer,
			DWORD dwSampleSize);
	STDMETHODIMP OnShutdown();

	HRESULT CreateMediaSample(DWORD cbData, IMFSample **ppSample);

	void set_frame_size(int w, int h);
	void set_color_transform(IMFTransform *mft) { m_pColorTransform = mft; }
};

#endif
