/**************************************************************************/
/*  usd_state.h                                                           */
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

#include "core/io/resource.h"

// TODO: Update export state to use TinyUSDZ
// TinyUSDZ headers
// Workaround for Texture name conflict: rename TinyUSDZ's Texture before including
#define Texture TinyUSDZTexture
#include "tinyusdz.hh"
#undef Texture

namespace tinyusdz {
	class Stage;
}

class UsdState : public Resource {
	GDCLASS(UsdState, Resource);

private:
	// Export state
	String _copyright;
	float _bake_fps;

	// USD-specific state
	// TODO: Update to use TinyUSDZ API (tinyusdz::Stage)
	tinyusdz::Stage *_stage;

protected:
	static void _bind_methods();

public:
	UsdState();

	// Getters and setters
	void set_copyright(const String &p_copyright);
	String get_copyright() const;

	void set_bake_fps(float p_fps);
	float get_bake_fps() const;

	// Stage management
	// TODO: Update to use TinyUSDZ API
	void set_stage(tinyusdz::Stage *p_stage);
	tinyusdz::Stage *get_stage() const;
};
