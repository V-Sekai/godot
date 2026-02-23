/**************************************************************************/
/*  owner_picker.h                                                        */
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

#pragma once

/**
 * owner_picker.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef OWNER_PICKER_H
#define OWNER_PICKER_H

#ifdef LIMBOAI_MODULE
#include "scene/gui/dialogs.h"
#include "scene/gui/item_list.h"
#endif

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/accept_dialog.hpp>
#include <godot_cpp/classes/item_list.hpp>
using namespace godot;
#endif

class OwnerPicker : public AcceptDialog {
	GDCLASS(OwnerPicker, AcceptDialog);

private:
	ItemList *owners_item_list;

	void _item_activated(int p_item);
	void _selection_confirmed();

protected:
	static void _bind_methods();

	void _notification(int p_what);

	Vector<String> _find_owners(const String &p_path) const;

public:
	void pick_and_open_owner_of_resource(const String &p_path);

	OwnerPicker();
};

#endif // OWNER_PICKER_H
