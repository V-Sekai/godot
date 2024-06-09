/**************************************************************************/
/*  open_telemetry.h                                                      */
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

#ifndef OPEN_TELEMETRY_H
#define OPEN_TELEMETRY_H

#include "core/object/ref_counted.h"
#include "core/variant/dictionary.h"

#include "libdesync_c_interface.h"

class OpenTelemetry : public RefCounted {
	GDCLASS(OpenTelemetry, RefCounted);

protected:
	static void _bind_methods();

public:
	String init_tracer_provider(String p_name, String p_host, Dictionary p_attributes);
	String start_span(String p_name);
	String start_span_with_parent(String p_name, String p_parent_span_uuid);
	void add_event(String p_span_uuid, String p_event_name);
	void set_attributes(String p_span_uuid, Dictionary p_attributes);
	void record_error(String p_span_uuid, String p_error);
	void end_span(String p_span_uuid);
	String shutdown();
};

#endif // OPEN_TELEMETRY_H
