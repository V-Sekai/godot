/**************************************************************************/
/*  animation_tree_handler_plugin.h                                       */
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

#include "editor/plugins/editor_plugin.h"
#include "scene/animation/animation_tree.h"

// The only reason this class exists is to provide a way for MMAnimationNode
// to access the AnimationTree in editor.
// If a new and better way of doing this comes along, please remove this class!
class AnimationTreeHandlerPlugin : public EditorPlugin {
	GDCLASS(AnimationTreeHandlerPlugin, EditorPlugin)

public:
	AnimationTreeHandlerPlugin();
	virtual ~AnimationTreeHandlerPlugin() = default;

	virtual bool handles(Object *p_object) const override;
	virtual void edit(Object *p_object) override;

	static AnimationTreeHandlerPlugin *get_singleton();
	AnimationTree *get_animation_tree() const {
		return _animation_tree;
	}

protected:
	static void _bind_methods();

	static AnimationTreeHandlerPlugin *_singleton;

private:
	AnimationTree *_animation_tree;
};
