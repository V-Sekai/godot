/**************************************************************************/
/*  fabric_snapshot.cpp                                                   */
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

#include "fabric_snapshot.h"

#include "core/object/class_db.h"

void FabricSnapshot::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_version", "version"), &FabricSnapshot::set_version);
	ClassDB::bind_method(D_METHOD("get_version"), &FabricSnapshot::get_version);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "version"), "set_version", "get_version");

	ClassDB::bind_method(D_METHOD("set_global_ids", "ids"), &FabricSnapshot::set_global_ids);
	ClassDB::bind_method(D_METHOD("get_global_ids"), &FabricSnapshot::get_global_ids);
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "global_ids"), "set_global_ids", "get_global_ids");

	ClassDB::bind_method(D_METHOD("set_positions", "positions"), &FabricSnapshot::set_positions);
	ClassDB::bind_method(D_METHOD("get_positions"), &FabricSnapshot::get_positions);
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT64_ARRAY, "positions"), "set_positions", "get_positions");

	ClassDB::bind_method(D_METHOD("set_velocities", "velocities"), &FabricSnapshot::set_velocities);
	ClassDB::bind_method(D_METHOD("get_velocities"), &FabricSnapshot::get_velocities);
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT64_ARRAY, "velocities"), "set_velocities", "get_velocities");

	ClassDB::bind_method(D_METHOD("set_accelerations", "accelerations"), &FabricSnapshot::set_accelerations);
	ClassDB::bind_method(D_METHOD("get_accelerations"), &FabricSnapshot::get_accelerations);
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT64_ARRAY, "accelerations"), "set_accelerations", "get_accelerations");

	ClassDB::bind_method(D_METHOD("set_payloads", "payloads"), &FabricSnapshot::set_payloads);
	ClassDB::bind_method(D_METHOD("get_payloads"), &FabricSnapshot::get_payloads);
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "payloads"), "set_payloads", "get_payloads");

	ClassDB::bind_method(D_METHOD("get_entity_count"), &FabricSnapshot::get_entity_count);
}
