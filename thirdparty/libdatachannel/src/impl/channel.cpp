/**
 * Copyright (c) 2019-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "channel.hpp"
#include "internals.hpp"

namespace rtc::impl {

void Channel::triggerOpen() {
	mOpenTriggered = true;
	RTC_TRY {
		openCallback();
	} RTC_CATCH (const RTC_EXCEPTION &e) {
		PLOG_WARNING << "Uncaught exception in callback: " << e.RTC_WHAT();
	}
	flushPendingMessages();
}

void Channel::triggerClosed() {
	RTC_TRY {
		closedCallback();
	} RTC_CATCH (const RTC_EXCEPTION &e) {
		PLOG_WARNING << "Uncaught exception in callback: " << e.RTC_WHAT();
	}
}

void Channel::triggerError(string error) {
	RTC_TRY {
		errorCallback(std::move(error));
	} RTC_CATCH (const RTC_EXCEPTION &e) {
		PLOG_WARNING << "Uncaught exception in callback: " << e.RTC_WHAT();
	}
}

void Channel::triggerAvailable(size_t count) {
	if (count == 1) {
		RTC_TRY {
			availableCallback();
		} RTC_CATCH (const RTC_EXCEPTION &e) {
			PLOG_WARNING << "Uncaught exception in callback: " << e.RTC_WHAT();
		}
	}

	flushPendingMessages();
}

void Channel::triggerBufferedAmount(size_t amount) {
	size_t previous = bufferedAmount.exchange(amount);
	size_t threshold = bufferedAmountLowThreshold.load();
	if (previous > threshold && amount <= threshold) {
		RTC_TRY {
			bufferedAmountLowCallback();
		} RTC_CATCH (const RTC_EXCEPTION &e) {
			PLOG_WARNING << "Uncaught exception in callback: " << e.RTC_WHAT();
		}
	}
}

void Channel::flushPendingMessages() {
	if (!mOpenTriggered)
		return;

	while (messageCallback) {
		auto next = receive();
		if (!next)
			break;

		RTC_TRY {
			messageCallback(*next);
		} RTC_CATCH (const RTC_EXCEPTION &e) {
			PLOG_WARNING << "Uncaught exception in callback: " << e.RTC_WHAT();
		}
	}
}

void Channel::resetOpenCallback() {
	mOpenTriggered = false;
	openCallback = nullptr;
}

void Channel::resetCallbacks() {
	mOpenTriggered = false;
	openCallback = nullptr;
	closedCallback = nullptr;
	errorCallback = nullptr;
	availableCallback = nullptr;
	bufferedAmountLowCallback = nullptr;
	messageCallback = nullptr;
}

} // namespace rtc::impl
