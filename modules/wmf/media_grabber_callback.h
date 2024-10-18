/**************************************************************************/
/*  media_grabber_callback.h                                              */
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

#ifndef MEDIA_GRABBER_CALLBACK_H
#define MEDIA_GRABBER_CALLBACK_H

#include "core/io/resource_loader.h"
#include "core/os/mutex.h"
#include "scene/resources/video_stream.h"
#include <mfidl.h>

#define CHECK_HR(func)                                                            \
	if (SUCCEEDED(hr)) {                                                          \
		hr = (func);                                                              \
		if (FAILED(hr)) {                                                         \
			print_line(vformat("%s failed, return: %s", __FUNCTION__, itos(hr))); \
		}                                                                         \
	}

class VideoStreamPlaybackWMF;

class MediaGrabberCallback : public IMFSampleGrabberSinkCallback {
	long m_cRef = 0;
	VideoStreamPlaybackWMF *playback;
	Mutex &mtx;
	int width = 0;
	int height = 0;

	IMFTransform *m_pColorTransform = nullptr;
	IMFSample *m_pSample = nullptr;
	IMFSample *m_pOutSample = nullptr;

	MediaGrabberCallback(VideoStreamPlaybackWMF *playback, Mutex &p_mtx);

public:
	virtual ~MediaGrabberCallback() {}
	static HRESULT CreateInstance(MediaGrabberCallback **ppCB, VideoStreamPlaybackWMF *playback, Mutex &p_mtx);

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

#endif // MEDIA_GRABBER_CALLBACK_H
