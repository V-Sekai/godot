#ifndef BONJOUR_H
#define BONJOUR_H

#include "core/reference.h"
#include "core/os/thread.h"
#include "core/os/mutex.h"
#include <string>
#include <set>
#include <map>
#include "dns_sd.h"

class Bonjour;

struct BonjourServiceData {
    String name;
    String ip;
    int port;
    bool resolved;
    Bonjour *bonjour;
};

class Bonjour : public Reference {
    GDCLASS(Bonjour, Reference);

    bool is_browsing;
    DNSServiceRef registerClientRef;
    std::set<DNSServiceRef> clientRefSet;
    std::map<String, BonjourServiceData*> clientMap;
    Mutex mutex;

    Thread *eventThread;
    bool thread_exit;

    void addClientRef(DNSServiceRef ref);
    void removeClientRef(DNSServiceRef ref);

    // static functions
	static void DNSSD_API reg_reply(DNSServiceRef sdref, const DNSServiceFlags flags, DNSServiceErrorType errorCode, const char* name, const char* regtype, const char* domain, void* context);

    static void DNSSD_API serviceCallback(DNSServiceRef service, DNSServiceFlags flags, uint32_t interfaceIndex, DNSServiceErrorType errorCode, const char* name, const char* regtype, const char *domain, void *context);

    static void DNSSD_API ResolveBonjourLocalName(DNSServiceRef sdref, const DNSServiceFlags flags, uint32_t ifIndex, DNSServiceErrorType errorCode, const char* fullname, const char* hosttarget, uint16_t opaqueport, uint16_t txtLen, const unsigned char* txtRecord, void* context);

    static void DNSSD_API BonjourResolveIP(DNSServiceRef sdref, const DNSServiceFlags flags, uint32_t ifIndex, DNSServiceErrorType errorCode, const char* fullname, uint16_t rrtype, uint16_t rrclass, uint16_t rdlen, const void* rdata, uint32_t ttl, void* context);

	static void handleEvents(void *ud);

protected:
    static void _bind_methods();

public:
    bool registerBonjour(const String &type, const String &name, int port);
    bool browseService(const String &type);

    void unregisterBonjour();

    Bonjour();
    ~Bonjour();
};

#endif
