/**************************************************************************/
/*  fbx_document_extension.h                                              */
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

#ifndef FBX_DOCUMENT_EXTENSION_H
#define FBX_DOCUMENT_EXTENSION_H

#include "../fbx_state.h"

class FBXDocumentExtension : public Resource {
	GDCLASS(FBXDocumentExtension, Resource);

protected:
	static void _bind_methods();

public:
	// Import process.
	virtual Error import_preflight(Ref<FBXState> p_state, Vector<String> p_extensions);
	virtual Vector<String> get_supported_extensions();
	virtual Error parse_node_extensions(Ref<FBXState> p_state, Ref<FBXNode> p_gltf_node, Dictionary &p_extensions);
	virtual Error parse_image_data(Ref<FBXState> p_state, const PackedByteArray &p_image_data, const String &p_mime_type, Ref<Image> r_image);
	virtual String get_image_file_extension();
	virtual Error parse_texture_json(Ref<FBXState> p_state, const Dictionary &p_texture_json, Ref<FBXTexture> r_gltf_texture);
	virtual Node3D *generate_scene_node(Ref<FBXState> p_state, Ref<FBXNode> p_gltf_node, Node *p_scene_parent);
	virtual Error import_post_parse(Ref<FBXState> p_state);
	virtual Error import_node(Ref<FBXState> p_state, Ref<FBXNode> p_gltf_node, Dictionary &r_json, Node *p_node);
	virtual Error import_post(Ref<FBXState> p_state, Node *p_node);

	// Import process.
	GDVIRTUAL2R(Error, _import_preflight, Ref<FBXState>, Vector<String>);
	GDVIRTUAL0R(Vector<String>, _get_supported_extensions);
	GDVIRTUAL3R(Error, _parse_node_extensions, Ref<FBXState>, Ref<FBXNode>, Dictionary);
	GDVIRTUAL4R(Error, _parse_image_data, Ref<FBXState>, PackedByteArray, String, Ref<Image>);
	GDVIRTUAL0R(String, _get_image_file_extension);
	GDVIRTUAL3R(Error, _parse_texture_json, Ref<FBXState>, Dictionary, Ref<FBXTexture>);
	GDVIRTUAL3R(Node3D *, _generate_scene_node, Ref<FBXState>, Ref<FBXNode>, Node *);
	GDVIRTUAL1R(Error, _import_post_parse, Ref<FBXState>);
	GDVIRTUAL4R(Error, _import_node, Ref<FBXState>, Ref<FBXNode>, Dictionary, Node *);
	GDVIRTUAL2R(Error, _import_post, Ref<FBXState>, Node *);
};

#endif // FBX_DOCUMENT_EXTENSION_H
