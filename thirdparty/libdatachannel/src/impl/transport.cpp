/**
 * Copyright (c) 2019-2022 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transport.hpp"

namespace rtc::impl {

Transport::Transport(shared_ptr<Transport> lower, state_callback callback)
    : mLower(std::move(lower)), mStateChangeCallback(std::move(callback)) {}

Transport::~Transport() {
	unregisterIncoming();

	if (mLower) {
		mLower->stop();
		mLower.reset();
	}
}

void Transport::registerIncoming() {
	if (mLower) {
		PLOG_VERBOSE << "Registering incoming callback";
		mLower->onRecv([this](message_ptr message) -> void {
			RTC_TRY {
				Transport::incoming(message);
			} RTC_CATCH(RTC_EXCEPTION e) {
				PLOG_WARNING << e.RTC_WHAT(); // FIXME
			}
		});
	}
}

void Transport::unregisterIncoming() {
	if (mLower) {
		PLOG_VERBOSE << "Unregistering incoming callback";
		mLower->onRecv(nullptr);
	}
}

Transport::State Transport::state() const { return mState; }

void Transport::onRecv(message_callback callback) { mRecvCallback = std::move(callback); }

void Transport::onStateChange(state_callback callback) {
	mStateChangeCallback = std::move(callback);
}

RTC_WRAPPED(void) Transport::start() { registerIncoming(); RTC_RET; }

void Transport::stop() { unregisterIncoming(); }

RTC_WRAPPED(bool) Transport::send(message_ptr message) { return outgoing(message); }

void Transport::recv(message_ptr message) {
	RTC_TRY {
		mRecvCallback(message);
	} RTC_CATCH (const RTC_EXCEPTION &e) {
		PLOG_WARNING << e.RTC_WHAT();
	}
}

void Transport::changeState(State state) {
	RTC_TRY {
		if (mState.exchange(state) != state)
			mStateChangeCallback(state);
	} RTC_CATCH (const RTC_EXCEPTION &e) {
		PLOG_WARNING << e.RTC_WHAT();
	}
}

void Transport::incoming(message_ptr message) { recv(message); }

RTC_WRAPPED(bool) Transport::outgoing(message_ptr message) {
	if (mLower)
		return mLower->send(message);
	else
		return false;
}

} // namespace rtc::impl
