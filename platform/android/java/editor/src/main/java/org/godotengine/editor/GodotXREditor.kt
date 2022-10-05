/*************************************************************************/
/*  GodotXREditor.kt                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

package org.godotengine.editor

import org.godotengine.godot.xr.XRMode

/**
 * Provide support for running the editor window in an immersive environment.
 */
open class GodotXREditor: GodotEditor() {

	override fun getCommandLine(): MutableList<String> {
		val updatedArgs = super.getCommandLine()
		if (!updatedArgs.contains(XRMode.OPENXR.cmdLineArg)) {
			updatedArgs.add(XRMode.OPENXR.cmdLineArg)
		}
		if (!updatedArgs.contains(XR_MODE_ARG)) {
			updatedArgs.add(XR_MODE_ARG)
			updatedArgs.add("on")
		}
		return updatedArgs
	}

	override fun getEditorWindowInfo() = XR_EDITOR_MAIN_INFO

	override fun getGodotAppLayout() = R.layout.godot_xr_layout
}
