#ifndef SENTRY_OBJECT_H
#define SENTRY_OBJECT_H

#include "core/object/object.h"

class Sentry : public Object {
	GDCLASS(Sentry, Object);

	static Sentry *singleton;

protected:
	static void _bind_methods();
	void _init();

	bool send_ip_address = false;

public:
	static Sentry *get_singleton();

	void _set_username(String p_username);
	void _send_event(String p_text);
	void _send_breadcrumb(String p_type, String p_message);

	Sentry();
	~Sentry();
};

#endif // SENTRY_OBJECT_H
