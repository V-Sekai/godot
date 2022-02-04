/*************************************************************************/
/*  gltf_document_extension.h                                            */
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

#ifndef GLTF_DOCUMENT_EXTENSION_H
#define GLTF_DOCUMENT_EXTENSION_H

#include "core/io/resource.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"

#include "core/object/gdvirtual.gen.inc"
#include "core/object/script_language.h"
#include "core/variant/native_ptr.h"

class GLTFState;
class GLTFNode;
class GLTFDocumentExtension : public Resource {
	GDCLASS(GLTFDocumentExtension, Resource);

	Dictionary import_settings;
	Dictionary export_settings;

protected:
	static void _bind_methods();

public:
	virtual bool is_changing_original_tree() const;
	virtual Array get_import_setting_keys() const;
	virtual Variant get_import_setting(const StringName p_key) const;
	virtual void set_import_setting(const StringName p_key, Variant p_var);
	virtual Array get_export_setting_keys() const;
	virtual Variant get_export_setting(const StringName p_key) const;
	virtual void set_export_setting(const StringName p_key, Variant p_var);
	virtual Error import_preflight(Ref<GLTFState> p_state);
	virtual Error import_post_parse(Ref<GLTFState> p_state);
	virtual Error export_post(Ref<GLTFState> p_state);
	// These stages should not be able to modify the original node when checking conditions.
	virtual Error import_post(Ref<GLTFState> p_state, Node *p_node);
	virtual Error export_preflight(Node *p_state);
	// End condition checking stages.
	Error import_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Dictionary &r_dict, Node *p_node);
	Error export_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Dictionary &r_dict, Node *p_node);
	GDVIRTUAL1R(int, _import_preflight, RES);
	GDVIRTUAL1R(int, _import_post_parse, RES);
	GDVIRTUAL4R(int, _import_node, RES, RES, Dictionary, Object *);
	GDVIRTUAL2R(int, _import_post, RES, Object *);
	GDVIRTUAL1R(int, _export_preflight, Object *);
	GDVIRTUAL4R(int, _export_node, RES, RES, Dictionary, Object *);
	GDVIRTUAL1R(int, _export_post, RES);
};

#endif // GLTF_DOCUMENT_EXTENSION_H
