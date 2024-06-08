#include "sample_grabber_callback.h"
#include <new>
#include <cstdio>
#include <cassert>
#include <Shlwapi.h>
#include <mfapi.h>
#include "core/print_string.h"
#include "video_stream_wmf.h"


#define CHECK_HR(func) if (SUCCEEDED(hr)) { hr = (func); if (FAILED(hr)) { print_line(__FUNCTION__ " failed, return:" + itos(hr)); } }

SampleGrabberCallback::SampleGrabberCallback(VideoStreamPlaybackWMF* playback, Mutex& mtx)
: playback(playback)
, mtx(mtx)
, m_cRef(1)
{
}

HRESULT SampleGrabberCallback::CreateInstance(SampleGrabberCallback **ppCB, VideoStreamPlaybackWMF* playback, Mutex& mtx)
{
    //print_line(__FUNCTION__);

    *ppCB = new (std::nothrow) SampleGrabberCallback(playback, mtx);

    if (ppCB == nullptr)
    {
        return E_OUTOFMEMORY;
    }
    return S_OK;
}

SampleGrabberCallback::~SampleGrabberCallback() {
    //print_line(__FUNCTION__);
}

STDMETHODIMP SampleGrabberCallback::QueryInterface(REFIID riid, void** ppv)
{
    //print_line(__FUNCTION__);
    static const QITAB qit[] =
    {
        QITABENT(SampleGrabberCallback, IMFSampleGrabberSinkCallback),
        QITABENT(SampleGrabberCallback, IMFClockStateSink),
    { 0 }
    };
    return QISearch(this, qit, riid, ppv);
}

STDMETHODIMP_(ULONG) SampleGrabberCallback::AddRef()
{
    return InterlockedIncrement(&m_cRef);
}

STDMETHODIMP_(ULONG) SampleGrabberCallback::Release()
{
    ULONG cRef = InterlockedDecrement(&m_cRef);
    if (cRef == 0)
    {
        delete this;
    }
    return cRef;
}

// IMFClockStateSink methods.

// In these example, the IMFClockStateSink methods do not perform any actions. 
// You can use these methods to track the state of the sample grabber sink.

STDMETHODIMP SampleGrabberCallback::OnClockStart(MFTIME hnsSystemTime, LONGLONG llClockStartOffset)
{
    //print_line(__FUNCTION__);
    return S_OK;
}

STDMETHODIMP SampleGrabberCallback::OnClockStop(MFTIME hnsSystemTime)
{
    //print_line(__FUNCTION__);
    return S_OK;
}

STDMETHODIMP SampleGrabberCallback::OnClockPause(MFTIME hnsSystemTime)
{
    //print_line(__FUNCTION__);
    return S_OK;
}

STDMETHODIMP SampleGrabberCallback::OnClockRestart(MFTIME hnsSystemTime)
{
    //print_line(__FUNCTION__);
    return S_OK;
}

STDMETHODIMP SampleGrabberCallback::OnClockSetRate(MFTIME hnsSystemTime, float flRate)
{
    //print_line(__FUNCTION__);
    return S_OK;
}

// IMFSampleGrabberSink methods.

STDMETHODIMP SampleGrabberCallback::OnSetPresentationClock(IMFPresentationClock* pClock)
{
    //print_line(__FUNCTION__);
    return S_OK;
}

HRESULT SampleGrabberCallback::CreateMediaSample(DWORD cbData, IMFSample **ppSample) {
	assert(ppSample);

	HRESULT hr = S_OK;

	IMFSample *pSample = nullptr;
	IMFMediaBuffer *pBuffer = nullptr;

	CHECK_HR(hr = MFCreateSample(&pSample));
	CHECK_HR(hr = MFCreateMemoryBuffer(cbData, &pBuffer));
	CHECK_HR(hr = pSample->AddBuffer(pBuffer));

	*ppSample = pSample;
	(*ppSample)->AddRef();

	return hr;
}

STDMETHODIMP SampleGrabberCallback::OnProcessSample(REFGUID guidMajorMediaType,
                                                    DWORD dwSampleFlags,
                                                    LONGLONG llSampleTime,
                                                    LONGLONG llSampleDuration,
                                                    const BYTE* pSampleBuffer,
                                                    DWORD dwSampleSize)
{
	HRESULT hr = S_OK;
    //assert(frame_data->size() == width * height * 3);

	const int rgb24FrameSize = width * height * 3;
	if (m_pSample == nullptr) CreateMediaSample(dwSampleSize, &m_pSample);
	if (m_pOutSample == nullptr) CreateMediaSample(rgb24FrameSize, &m_pOutSample);

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
	if (FAILED(hr))
		print_line("Failed to process video frames");

	MFT_OUTPUT_DATA_BUFFER RGBOutputDataBuffer;
	RGBOutputDataBuffer.dwStreamID = 0;
	RGBOutputDataBuffer.dwStatus = 0;
	RGBOutputDataBuffer.pEvents = NULL;
	RGBOutputDataBuffer.pSample = m_pOutSample;
	CHECK_HR(m_pColorTransform->ProcessOutput(0, 1, &RGBOutputDataBuffer, &ProcessStatus));
	if (FAILED(hr))
		print_line("Failed to process video frames");

	IMFMediaBuffer *pOutputBuffer;
	RGBOutputDataBuffer.pSample->GetBufferByIndex(0, &pOutputBuffer);

	BYTE *outData;
	DWORD outDataLen;
	pOutputBuffer->Lock(&outData, NULL, &outDataLen);

	//mtx.lock();
	{
		FrameData* frame = playback->get_next_writable_frame();
		frame->sample_time = llSampleTime / 10000;
		//print_line(itos(llSampleTime));

		uint8_t* dst = frame->data.write().ptr();

		char *rgb_buffer = (char *)dst;
		// convert 4 pixels at once
		for (int i = 0; i < outDataLen; i += 12) {

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
		//memcpy(rgb_buffer, outData, outDataLen);
	}
    //mtx.unlock();

	pOutputBuffer->Unlock();

	playback->write_frame_done();

    return S_OK;
}

STDMETHODIMP SampleGrabberCallback::OnShutdown()
{
    print_line(__FUNCTION__);
    return S_OK;
}

void SampleGrabberCallback::set_frame_size(int w, int h)
{
	width = w;
	height = h;
}
