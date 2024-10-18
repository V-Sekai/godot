/**************************************************************************/
/*  media_grabber_callback.cpp                                            */
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

#include "media_grabber_callback.h"
#include "core/string/print_string.h"
#include "video_stream_wmf.h"
#include <Shlwapi.h>
#include <mfapi.h>
#include <minwindef.h>
#include <cassert>
#include <cstdio>
#include <new>

MediaGrabberCallback::MediaGrabberCallback(VideoStreamPlaybackWMF *p_playback, Mutex &p_mtx) :
		m_cRef(1), playback(p_playback), mtx(p_mtx) {
}

HRESULT MediaGrabberCallback::CreateInstance(MediaGrabberCallback **ppCB, VideoStreamPlaybackWMF *p_playback, Mutex &p_mtx) {
	*ppCB = new (std::nothrow) MediaGrabberCallback(p_playback, p_mtx);

	if (ppCB == nullptr) {
		return E_OUTOFMEMORY;
	}
	return S_OK;
}

STDMETHODIMP MediaGrabberCallback::QueryInterface(REFIID riid, void **ppv) {
	static const QITAB qit[] = {
		// #pragma clang diagnostic ignored "-Wc++11-narrowing"
		QITABENT(MediaGrabberCallback, IMFSampleGrabberSinkCallback),
		// #pragma clang diagnostic ignored "-Wc++11-narrowing"
		QITABENT(MediaGrabberCallback, IMFClockStateSink),
		{ 0, 0 }
	};
	return QISearch(this, qit, riid, ppv);
}

STDMETHODIMP_(ULONG)
MediaGrabberCallback::AddRef() {
	return InterlockedIncrement(&m_cRef);
}

STDMETHODIMP_(ULONG)
MediaGrabberCallback::Release() {
	ULONG cRef = InterlockedDecrement(&m_cRef);
	if (cRef == 0) {
		delete this;
	}
	return cRef;
}

// IMFClockStateSink methods.

// In these example, the IMFClockStateSink methods do not perform any actions.
// You can use these methods to track the state of the sample grabber sink.

STDMETHODIMP MediaGrabberCallback::OnClockStart(MFTIME hnsSystemTime, LONGLONG llClockStartOffset) {
	return S_OK;
}

STDMETHODIMP MediaGrabberCallback::OnClockStop(MFTIME hnsSystemTime) {
	return S_OK;
}

STDMETHODIMP MediaGrabberCallback::OnClockPause(MFTIME hnsSystemTime) {
	return S_OK;
}

STDMETHODIMP MediaGrabberCallback::OnClockRestart(MFTIME hnsSystemTime) {
	return S_OK;
}

STDMETHODIMP MediaGrabberCallback::OnClockSetRate(MFTIME hnsSystemTime, float flRate) {
	return S_OK;
}

// IMFSampleGrabberSink methods.

STDMETHODIMP MediaGrabberCallback::OnSetPresentationClock(IMFPresentationClock *pClock) {
	return S_OK;
}

HRESULT MediaGrabberCallback::CreateMediaSample(DWORD cbData, IMFSample **ppSample) {
	assert(ppSample);

	HRESULT hr = S_OK;

	IMFSample *pSample = nullptr;
	IMFMediaBuffer *pBuffer = nullptr;

	hr = MFCreateSample(&pSample);
	CHECK_HR(hr);

	hr = MFCreateMemoryBuffer(cbData, &pBuffer);
	CHECK_HR(hr);

	hr = pSample->AddBuffer(pBuffer);
	CHECK_HR(hr);

	*ppSample = pSample;
	(*ppSample)->AddRef();

	return hr;
}

STDMETHODIMP MediaGrabberCallback::OnProcessSample(REFGUID guidMajorMediaType,
		DWORD dwSampleFlags,
		LONGLONG llSampleTime,
		LONGLONG llSampleDuration,
		const BYTE *pSampleBuffer,
		DWORD dwSampleSize) {
	HRESULT hr = S_OK;
	//assert(frame_data->size() == width * height * 3);

	const int rgb24FrameSize = width * height * 3;
	if (m_pSample == nullptr) {
		CreateMediaSample(dwSampleSize, &m_pSample);
	}
	if (m_pOutSample == nullptr) {
		CreateMediaSample(rgb24FrameSize, &m_pOutSample);
	}

	IMFMediaBuffer *pMediaBuffer = nullptr;
	m_pSample->SetSampleTime(llSampleTime);
	m_pSample->SetSampleDuration(llSampleDuration);
	m_pSample->GetBufferByIndex(0, &pMediaBuffer);

	BYTE *pData = nullptr;
	pMediaBuffer->Lock(&pData, NULL, NULL);
	{
		memcpy(pData, pSampleBuffer, dwSampleSize);
		hr = pMediaBuffer->SetCurrentLength(dwSampleSize);
	}
	pMediaBuffer->Unlock();

	DWORD ProcessStatus;
	CHECK_HR(m_pColorTransform->ProcessInput(0, m_pSample, 0));
	if (FAILED(hr)) {
		print_line("Failed to process video frames");
	}

	MFT_OUTPUT_DATA_BUFFER RGBOutputDataBuffer;
	RGBOutputDataBuffer.dwStreamID = 0;
	RGBOutputDataBuffer.dwStatus = 0;
	RGBOutputDataBuffer.pEvents = NULL;
	RGBOutputDataBuffer.pSample = m_pOutSample;
	CHECK_HR(m_pColorTransform->ProcessOutput(0, 1, &RGBOutputDataBuffer, &ProcessStatus));
	if (FAILED(hr)) {
		print_line("Failed to process video frames");
	}

	IMFMediaBuffer *pOutputBuffer;
	RGBOutputDataBuffer.pSample->GetBufferByIndex(0, &pOutputBuffer);

	BYTE *outData;
	DWORD outDataLen;
	pOutputBuffer->Lock(&outData, NULL, &outDataLen);

	mtx.lock();
	{
		FrameData *frame = playback->get_next_writable_frame();
		frame->sample_time = llSampleTime / 10000;
		//print_line(itos(llSampleTime));

		uint8_t *dst = frame->data.ptrw();

		char *rgb_buffer = (char *)dst;
		// convert 4 pixels at once
		for (size_t i = 0; i < outDataLen; i += 12) {
			rgb_buffer[i + 0] = outData[i + 2];
			rgb_buffer[i + 1] = outData[i + 1];
			rgb_buffer[i + 2] = outData[i + 0];

			rgb_buffer[i + 3] = outData[i + 5];
			rgb_buffer[i + 4] = outData[i + 4];
			rgb_buffer[i + 5] = outData[i + 3];

			rgb_buffer[i + 6] = outData[i + 8];
			rgb_buffer[i + 7] = outData[i + 7];
			rgb_buffer[i + 8] = outData[i + 6];

			rgb_buffer[i + 9] = outData[i + 11];
			rgb_buffer[i + 10] = outData[i + 10];
			rgb_buffer[i + 11] = outData[i + 9];
		}
		// memcpy(rgb_buffer, outData, outDataLen);
	}
	mtx.unlock();

	pOutputBuffer->Unlock();

	playback->write_frame_done();

	return S_OK;
}

STDMETHODIMP MediaGrabberCallback::OnShutdown() {
	print_line(__FUNCTION__);
	return S_OK;
}

void MediaGrabberCallback::set_frame_size(int w, int h) {
	width = w;
	height = h;
}
