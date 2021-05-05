#ifndef SAMPLEGRABBERCALLBACK_H
#define SAMPLEGRABBERCALLBACK_H

#include "core/io/resource_loader.h"
#include "core/os/mutex.h"
#include "scene/resources/video_stream.h"
#include <mfidl.h>

class VideoStreamPlaybackWMF;

class SampleGrabberCallback : public IMFSampleGrabberSinkCallback {
	long m_cRef;
	VideoStreamPlaybackWMF *playback;
	Mutex &mtx;
	int width;
	int height;

	IMFTransform *m_pColorTransform = nullptr;
	IMFSample *m_pSample = nullptr;
	IMFSample *m_pOutSample = nullptr;

	SampleGrabberCallback(VideoStreamPlaybackWMF *playback, Mutex &mtx);

public:
	static HRESULT CreateInstance(SampleGrabberCallback **ppCB, VideoStreamPlaybackWMF *playback, Mutex &mtx);
	~SampleGrabberCallback();

	// IUnknown methods
	STDMETHODIMP QueryInterface(REFIID iid, void **ppv);
	STDMETHODIMP_(ULONG)
	AddRef();
	STDMETHODIMP_(ULONG)
	Release();

	// IMFClockStateSink methods
	STDMETHODIMP OnClockStart(MFTIME hnsSystemTime, LONGLONG llClockStartOffset);
	STDMETHODIMP OnClockStop(MFTIME hnsSystemTime);
	STDMETHODIMP OnClockPause(MFTIME hnsSystemTime);
	STDMETHODIMP OnClockRestart(MFTIME hnsSystemTime);
	STDMETHODIMP OnClockSetRate(MFTIME hnsSystemTime, float flRate);

	// IMFSampleGrabberSinkCallback methods
	STDMETHODIMP OnSetPresentationClock(IMFPresentationClock *pClock);
	STDMETHODIMP OnProcessSample(REFGUID guidMajorMediaType, DWORD dwSampleFlags,
			LONGLONG llSampleTime, LONGLONG llSampleDuration, const BYTE *pSampleBuffer,
			DWORD dwSampleSize);
	STDMETHODIMP OnShutdown();

	HRESULT CreateMediaSample(DWORD cbData, IMFSample **ppSample);

	// custom methods
	void set_frame_size(int w, int h);
	void set_color_transform(IMFTransform *mft) { m_pColorTransform = mft; }
};

#endif
