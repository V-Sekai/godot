/**
 * Copyright (c) 2019-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_PEER_CONNECTION_H
#define RTC_IMPL_PEER_CONNECTION_H

#include "common.hpp"
#include "datachannel.hpp"
#include "dtlstransport.hpp"
#include "icetransport.hpp"
#include "init.hpp"
#include "processor.hpp"
#include "sctptransport.hpp"
#include "track.hpp"

#include "rtc/peerconnection.hpp"

#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace rtc::impl {

struct PeerConnection : std::enable_shared_from_this<PeerConnection> {
	using State = rtc::PeerConnection::State;
	using IceState = rtc::PeerConnection::IceState;
	using GatheringState = rtc::PeerConnection::GatheringState;
	using SignalingState = rtc::PeerConnection::SignalingState;

	PeerConnection(Configuration config_);
	~PeerConnection();

	void close();
	void remoteClose();

	optional<Description> localDescription() const;
	optional<Description> remoteDescription() const;
	size_t remoteMaxMessageSize() const;

	RTC_WRAPPED(shared_ptr<IceTransport>) initIceTransport();
	RTC_WRAPPED(shared_ptr<DtlsTransport>) initDtlsTransport();
	RTC_WRAPPED(shared_ptr<SctpTransport>) initSctpTransport();
	shared_ptr<IceTransport> getIceTransport() const;
	shared_ptr<DtlsTransport> getDtlsTransport() const;
	shared_ptr<SctpTransport> getSctpTransport() const;
	void closeTransports();

	void endLocalCandidates();
	void rollbackLocalDescription();
	bool checkFingerprint(const std::string &fingerprint) const;
	RTC_WRAPPED(void) forwardMessage(message_ptr message);
	void forwardMedia(message_ptr message);
	void forwardBufferedAmount(uint16_t stream, size_t amount);

	RTC_WRAPPED(shared_ptr<DataChannel>) emplaceDataChannel(string label, DataChannelInit init);
	std::pair<shared_ptr<DataChannel>, bool> findDataChannel(uint16_t stream);
	bool removeDataChannel(uint16_t stream);
	uint16_t maxDataChannelStream() const;
	RTC_WRAPPED(void) assignDataChannels();
	void iterateDataChannels(std::function<RTC_WRAPPED(void)(shared_ptr<DataChannel> channel)> func);
	void openDataChannels();
	void closeDataChannels();
	void remoteCloseDataChannels();

#if RTC_ENABLE_MEDIA
	shared_ptr<Track> emplaceTrack(Description::Media description);
	void iterateTracks(std::function<void(shared_ptr<Track> track)> func);
	void openTracks();
	void closeTracks();
#endif

	RTC_WRAPPED(void) validateRemoteDescription(const Description &description);
	RTC_WRAPPED(void) processLocalDescription(Description description);
	RTC_WRAPPED(void) processLocalCandidate(Candidate candidate);
	RTC_WRAPPED(void) processRemoteDescription(Description description);
	RTC_WRAPPED(void) processRemoteCandidate(Candidate candidate);
	string localBundleMid() const;

#if RTC_ENABLE_MEDIA
	void setMediaHandler(shared_ptr<MediaHandler> handler);
	shared_ptr<MediaHandler> getMediaHandler();
#endif

	void triggerDataChannel(weak_ptr<DataChannel> weakDataChannel);
#if RTC_ENABLE_MEDIA
	void triggerTrack(weak_ptr<Track> weakTrack);
#endif

	void triggerPendingDataChannels();
	void triggerPendingTracks();

	void flushPendingDataChannels();
	void flushPendingTracks();

	bool changeState(State newState);
	bool changeIceState(IceState newState);
	bool changeGatheringState(GatheringState newState);
	bool changeSignalingState(SignalingState newState);

	void resetCallbacks();

	// Helper method for asynchronous callback invocation
	template <typename... Args> void trigger(synchronized_callback<Args...> *cb, Args... args) {
		(*cb)(std::move(args...));
	}

	const Configuration config;
	std::atomic<State> state = State::New;
	std::atomic<IceState> iceState = IceState::New;
	std::atomic<GatheringState> gatheringState = GatheringState::New;
	std::atomic<SignalingState> signalingState = SignalingState::Stable;
	std::atomic<bool> negotiationNeeded = false;
	std::atomic<bool> closing = false;
	std::mutex signalingMutex;

	synchronized_callback<shared_ptr<rtc::DataChannel>> dataChannelCallback;
	synchronized_callback<Description> localDescriptionCallback;
	synchronized_callback<Candidate> localCandidateCallback;
	synchronized_callback<State> stateChangeCallback;
	synchronized_callback<IceState> iceStateChangeCallback;
	synchronized_callback<GatheringState> gatheringStateChangeCallback;
	synchronized_callback<SignalingState> signalingStateChangeCallback;
#if RTC_ENABLE_MEDIA
	synchronized_callback<shared_ptr<rtc::Track>> trackCallback;
#endif

private:
	void updateTrackSsrcCache(const Description &description);

	const init_token mInitToken = Init::Instance().token();
	const future_certificate_ptr mCertificate;

	Processor mProcessor;
	optional<Description> mLocalDescription, mRemoteDescription;
	optional<Description> mCurrentLocalDescription;
	mutable std::mutex mLocalDescriptionMutex, mRemoteDescriptionMutex;

#if RTC_ENABLE_MEDIA
	shared_ptr<MediaHandler> mMediaHandler;

	mutable std::shared_mutex mMediaHandlerMutex;
#endif
	shared_ptr<IceTransport> mIceTransport;
	shared_ptr<DtlsTransport> mDtlsTransport;
	shared_ptr<SctpTransport> mSctpTransport;

	std::unordered_map<uint16_t, weak_ptr<DataChannel>> mDataChannels; // by stream ID
	std::vector<weak_ptr<DataChannel>> mUnassignedDataChannels;
	std::shared_mutex mDataChannelsMutex;

#if RTC_ENABLE_MEDIA
	std::unordered_map<string, weak_ptr<Track>> mTracks;         // by mid
	std::unordered_map<uint32_t, weak_ptr<Track>> mTracksBySsrc; // by SSRC
	std::vector<weak_ptr<Track>> mTrackLines;                    // by SDP order
	std::shared_mutex mTracksMutex;
#endif

	Queue<shared_ptr<DataChannel>> mPendingDataChannels;

#if RTC_ENABLE_MEDIA
	Queue<shared_ptr<Track>> mPendingTracks;
#endif
};

} // namespace rtc::impl

#endif
