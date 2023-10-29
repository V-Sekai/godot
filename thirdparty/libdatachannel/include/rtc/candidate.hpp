/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_CANDIDATE_H
#define RTC_CANDIDATE_H

#include "common.hpp"

#include <string>

namespace rtc {

class RTC_CPP_EXPORT Candidate {
public:
	enum class Family { Unresolved, Ipv4, Ipv6 };
	enum class Type { Unknown, Host, ServerReflexive, PeerReflexive, Relayed };
	enum class TransportType { Unknown, Udp, TcpActive, TcpPassive, TcpSo, TcpUnknown };

	Candidate();
	static RTC_WRAPPED(Candidate) create(string candidate);
	static RTC_WRAPPED(Candidate) create(string candidate, string mid);

	void hintMid(string mid);
	RTC_WRAPPED(void) changeAddress(string addr);
	RTC_WRAPPED(void) changeAddress(string addr, uint16_t port);
	RTC_WRAPPED(void) changeAddress(string addr, string service);

	enum class ResolveMode { Simple, Lookup };
	bool resolve(ResolveMode mode = ResolveMode::Simple);

	Type type() const;
	TransportType transportType() const;
	uint32_t priority() const;
	string candidate() const;
	string mid() const;
	operator string() const;

	bool operator==(const Candidate &other) const;
	bool operator!=(const Candidate &other) const;

	bool isResolved() const;
	Family family() const;
	optional<string> address() const;
	optional<uint16_t> port() const;

private:
	RTC_WRAPPED(void) parse(string candidate);

	string mFoundation;
	uint32_t mComponent, mPriority;
	string mTypeString, mTransportString;
	Type mType;
	TransportType mTransportType;
	string mNode, mService;
	string mTail;

	optional<string> mMid;

	// Extracted on resolution
	Family mFamily;
	string mAddress;
	uint16_t mPort;
};

} // namespace rtc

RTC_CPP_EXPORT std::ostream &operator<<(std::ostream &out, const rtc::Candidate &candidate);
RTC_CPP_EXPORT std::ostream &operator<<(std::ostream &out, const rtc::Candidate::Type &type);
RTC_CPP_EXPORT std::ostream &operator<<(std::ostream &out,
                                        const rtc::Candidate::TransportType &transportType);

#endif
