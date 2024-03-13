#include "bonjour.h"

#ifdef _WIN32
#include "WinSock2.h"
#elif __APPLE__
#include <sys/select.h>
#include <sys/errno.h>
#include <unistd.h>
#elif __linux__
#include <arpa/inet.h>
#include <sys/select.h>
#include <sys/errno.h>
#include <unistd.h>
#endif

#include <stdio.h>

void Bonjour::addClientRef(DNSServiceRef ref) {
    MutexLock lock(mutex);
    if (ref)
        clientRefSet.insert(ref);
}

void Bonjour::removeClientRef(DNSServiceRef ref) {
    MutexLock lock(mutex);
    for (std::set<DNSServiceRef>::iterator it = clientRefSet.begin(); it != clientRefSet.end(); ) {
        if (*it == ref)
            it = clientRefSet.erase(it);
        else
            ++it;
    }
}

void Bonjour::handleEvents(void *ud) {
    Bonjour *bonjour = static_cast<Bonjour*>(ud);
	int max_fd = 0;
	fd_set readfds;
	struct timeval tv;
	int result;
    std::map<DNSServiceRef, int> mapRefFd;

    print_line("Bonjour: starting event handler...");
	while (!bonjour->thread_exit) {
        if (bonjour->clientRefSet.size() == 0) {
#ifdef _WIN32
            Sleep(10000);
#else
            usleep(10);
#endif
        }

        mapRefFd.clear();
		FD_ZERO(&readfds);

        bonjour->mutex.lock();
        for (std::set<DNSServiceRef>::iterator it = bonjour->clientRefSet.begin(); it != bonjour->clientRefSet.end(); ++it) {
            int fd = DNSServiceRefSockFD(*it);
            mapRefFd.insert(std::pair<DNSServiceRef, int>(*it, fd));
            FD_SET(fd, &readfds);
            if (fd > max_fd)
                max_fd = fd;
        }
        bonjour->mutex.unlock();

		tv.tv_sec = 0;
		tv.tv_usec = 1000;
		result = select(max_fd + 1, &readfds, (fd_set*)NULL, (fd_set*)NULL, &tv);

		if (result > 0) {
			DNSServiceErrorType err = kDNSServiceErr_NoError;

            for (std::map<DNSServiceRef, int>::iterator it = mapRefFd.begin(); it != mapRefFd.end(); ++it) {
                if (FD_ISSET(it->second, &readfds)) {
                    err = DNSServiceProcessResult(it->first);
                    if (err)
                        print_line("Bonjour: DNSServiceProcessResult returned:" + itos(err));
                }
            }
		} else if (result < 0){
			if (errno == EINTR || errno == EBADF)
				continue;
			print_line("select() returned:" + itos(result) + " errono:" + itos(errno));
		}
	}
    print_line("Bonjour: event handler exited");
}

void DNSSD_API Bonjour::BonjourResolveIP(DNSServiceRef sdref, const DNSServiceFlags flags, uint32_t ifIndex, DNSServiceErrorType errorCode, const char* fullname, uint16_t rrtype, uint16_t rrclass, uint16_t rdlen, const void* rdata, uint32_t ttl, void* context) {
    BonjourServiceData *data = static_cast<BonjourServiceData*>(context);
    if (errorCode != kDNSServiceErr_NoError) {
        print_line("Bonjour: resolving IP returns:" + itos(errorCode));
        return;
    }
    if (!(flags & kDNSServiceFlagsAdd)) {
        print_line("Bonjour: ignore DNS delete event");
        return;
    }

    if (data->resolved) {
        print_line("Bonjour: ignore DNS query since we already resolved before");
        return;
    }

    const unsigned char* rd  = static_cast<const unsigned char*>(rdata);
    char rdb[1000] = "";
    switch ( rrtype ) {
        case kDNSServiceType_A:
            snprintf(rdb, sizeof(rdb), "%d.%d.%d.%d", rd[0], rd[1], rd[2], rd[3]);
            data->ip = String(rdb);
            print_line("Bonjour: resolved IP of " + String(fullname) + ":" + data->ip);
            data->bonjour->emit_signal("service_added", data->name, data->ip, data->port);
            data->resolved = true;
            break;
        default :
            print_line("Bonjour: unknown type: " + itos(rrtype) + ":" + String(rdb));
            break;
    }
}

void DNSSD_API Bonjour::ResolveBonjourLocalName(DNSServiceRef sdref, const DNSServiceFlags flags, uint32_t ifIndex, DNSServiceErrorType errorCode, const char* fullname, const char* hosttarget, uint16_t opaqueport, uint16_t txtLen, const unsigned char* txtRecord, void* context) {
    BonjourServiceData *data = static_cast<BonjourServiceData*>(context);
    if (errorCode != kDNSServiceErr_NoError)
    {
        print_line("Bonjour: resolving service returns:" + itos(errorCode));
        return;
    }
    data->port = ntohs(opaqueport);

    DNSServiceErrorType error;
    DNSServiceFlags nflags = 0;
    uint16_t rrtype = kDNSServiceType_A, rrclass = kDNSServiceClass_IN;
    nflags |= kDNSServiceFlagsReturnIntermediates;
    nflags |= kDNSServiceFlagsSuppressUnusable;
    DNSServiceRef ref;
    error = DNSServiceQueryRecord(
                &ref,
                nflags,
                kDNSServiceInterfaceIndexAny,
                hosttarget,
                rrtype,
                rrclass,
                BonjourResolveIP,
                data);
    if ( error ) {
        print_line("Bonjour: DNSServiceQueryRecord failed:" + itos(error));
        return;
    }
    data->bonjour->addClientRef(ref);
}

void DNSSD_API Bonjour::serviceCallback(DNSServiceRef service, DNSServiceFlags flags, uint32_t interfaceIndex, DNSServiceErrorType errorCode, const char* name, const char* regtype, const char *domain, void *context) {
    Bonjour *bonjour = static_cast<Bonjour*>(context);

	if (errorCode != kDNSServiceErr_NoError) {
        print_line("Bonjour: browsing service returns:" + itos(errorCode));
        return;
    }

    String sname = String(name);
    if (flags & kDNSServiceFlagsAdd) {
        if (bonjour->clientMap.count(sname) > 0) {
            print_line("Bonjour: already has a client named " + sname);
            return;
        }
        BonjourServiceData *data = memnew(BonjourServiceData);
        data->bonjour = bonjour;
        data->name = sname;
        data->resolved = false;
        print_line("Bonjour: adding " + data->name + "@" + String(domain));
        bonjour->clientMap.insert(std::pair<String, BonjourServiceData*>(sname, data));

        DNSServiceRef ref;
        DNSServiceErrorType err = DNSServiceResolve(&ref, 0, interfaceIndex, name, regtype, domain, ResolveBonjourLocalName, data);
        if (err != kDNSServiceErr_NoError) {
            print_line("Bonjour: resolve service failed with error:" + itos(err));
            return;
        }
        bonjour->addClientRef(ref);
    } else {
        for (std::map<String, BonjourServiceData*>::iterator it = bonjour->clientMap.begin(); it != bonjour->clientMap.end(); ) {
            if (it->first == sname) {
                memdelete(it->second);
                it = bonjour->clientMap.erase(it);
                print_line("Bonjour: removing " + sname + "@" + String(domain));
                bonjour->emit_signal("service_removed", String(name));
            } else
                ++it;
        }
    }
}

bool Bonjour::browseService(const String &type) {
    std::wstring ws = type.c_str();
    std::string stype(ws.begin(), ws.end());
    const char* c_type = stype.c_str();

    DNSServiceRef ref;

    if (is_browsing) {
		print_line("Bonjour: already browsing a service");
		return false;
    }

    DNSServiceErrorType err = DNSServiceBrowse(&ref, 0, kDNSServiceInterfaceIndexAny, c_type, NULL, serviceCallback, this);
	if (err != kDNSServiceErr_NoError) {
		print_line("Bonjour: browse service failed with error:" + itos(err));
		return false;
	}
    print_line("Bonjour: browsing service type:" + type);
    addClientRef(ref);
    is_browsing = true;

    if (!eventThread) {
        eventThread = memnew(Thread);
        eventThread->start(handleEvents, this);
    }
    return true;
}

bool Bonjour::registerBonjour(const String &type, const String &name, int port) {
    std::wstring ws;

    ws = name.c_str();
    std::string sname(ws.begin(), ws.end());
    const char* c_name = sname.c_str();

    ws = type.c_str();
    std::string stype(ws.begin(), ws.end());
    const char* c_type = stype.c_str();

    if (registerClientRef) {
		print_line("Bonjour: already registered, skipping");
		return false;
    }

	// registering
	DNSServiceErrorType err = DNSServiceRegister(&registerClientRef, 0, kDNSServiceInterfaceIndexAny,
		c_name, c_type, "local", NULL, htons(port), 0, NULL, reg_reply, NULL);
	if (err != kDNSServiceErr_NoError) {
		print_line("Bonjour: registration failed with error:" + itos(err));
		return false;
	}
	print_line("Bonjour: registration done!");
    addClientRef(registerClientRef);

    if (!eventThread) {
        eventThread = memnew(Thread);
        eventThread->start(handleEvents, this);
    }
	return true;
}

void DNSSD_API Bonjour::reg_reply(DNSServiceRef sdref, const DNSServiceFlags flags, DNSServiceErrorType errorCode,
		const char* name, const char* regtype, const char* domain, void* context) {
	String rName = name, rRegType = regtype, rDomain = domain;

	if (errorCode == kDNSServiceErr_NoError) {
		if (flags & kDNSServiceFlagsAdd)
			print_line("Bonjour: Registered name: \"" + rName + "\" type: \"" + rRegType + "\" on domain: \"" + rDomain + "\"");
		else
			print_line("Bonjour: Name registration removed");
	}
	else if (errorCode == kDNSServiceErr_NameConflict) {
		print_line("Bonjour: Name in use, please choose another\n");
	}
	else {
		print_line("Bonjour: Error:" + itos(errorCode));
	}

	if (!(flags & kDNSServiceFlagsMoreComing))
		fflush(stdout);
}

void Bonjour::unregisterBonjour() {
    if (registerClientRef) {
        removeClientRef(registerClientRef);
        DNSServiceRefDeallocate(registerClientRef);
        registerClientRef = NULL;
    }
}

void Bonjour::_bind_methods() {
    ClassDB::bind_method(D_METHOD("register", "type", "name", "port"), &Bonjour::registerBonjour);
    ClassDB::bind_method(D_METHOD("browse", "type"), &Bonjour::browseService);
    ClassDB::bind_method(D_METHOD("unregister"), &Bonjour::unregisterBonjour);

    ADD_SIGNAL(MethodInfo("service_added", PropertyInfo(Variant::STRING, "name"),
                PropertyInfo(Variant::STRING, "ip"),
                PropertyInfo(Variant::INT, "port")));
    ADD_SIGNAL(MethodInfo("service_removed", PropertyInfo(Variant::STRING, "name")));
}

Bonjour::Bonjour() :
    is_browsing(false),
    registerClientRef(NULL),
    eventThread(NULL),
    thread_exit(false) {
}

Bonjour::~Bonjour() {
    if (eventThread) {
        thread_exit = true;
        eventThread->wait_to_finish();
        eventThread = NULL;
    }
    // clean up data
    for (std::map<String, BonjourServiceData*>::iterator it = clientMap.begin(); it != clientMap.end(); ++it) {
        memdelete(it->second);
    }
}

