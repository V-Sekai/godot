/*************************************************************************/
/*  rich_text_label_commonmark.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "cmark-gfm_export.h"
#include "config.h"
#include "thirdparty/cmark-gfm/src/cmark-gfm.h"

#include "rich_text_label_commonmark.h"

void RichTextLabelCommonmark::_bind_methods() {
	ClassDB::bind_method(D_METHOD("parse_commonmark", "commonmark"), &RichTextLabelCommonmark::parse_commonmark);
	ClassDB::bind_method(D_METHOD("append_commonmark", "commonmark"), &RichTextLabelCommonmark::append_commonmark);

	ClassDB::bind_method(D_METHOD("set_commonmark", "text"), &RichTextLabelCommonmark::set_commonmark);
	ClassDB::bind_method(D_METHOD("get_commonmark"), &RichTextLabelCommonmark::get_commonmark);

	ClassDB::bind_method(D_METHOD("set_use_commonmark", "enable"), &RichTextLabelCommonmark::set_use_commonmark);
	ClassDB::bind_method(D_METHOD("is_using_commonmark"), &RichTextLabelCommonmark::is_using_commonmark);

	ADD_GROUP("Commonmark", "commonmark_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "commonmark_enabled"), "set_use_commonmark", "is_using_commonmark");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "commonmark_text", PROPERTY_HINT_MULTILINE_TEXT), "set_commonmark", "get_commonmark");
}

String RichTextLabelCommonmark::commonmark_to_html() {
	Vector<uint8_t> string_bytes = commonmark.to_utf8_buffer();
	char *cmark_bytes = cmark_markdown_to_html((const char *)string_bytes.ptr(), string_bytes.size() - 1, CMARK_OPT_DEFAULT);
	String new_string;
	new_string.parse_utf8(cmark_bytes);
	if (cmark_bytes) {
		free(cmark_bytes);
	}
	return new_string;
}

Error RichTextLabelCommonmark::parse_commonmark(const String &p_commonmark) {
	clear();
	return append_commonmark(p_commonmark);
}

Error RichTextLabelCommonmark::append_commonmark(const String &p_commonmark) {
	if (p_commonmark.is_empty()) {
		return OK;
	}
	Vector<uint8_t> string_bytes = p_commonmark.to_utf8_buffer();
	set_process_internal(false);
	cmark_node *root = cmark_parse_document((const char *)string_bytes.ptr(), string_bytes.size() - 1,
			CMARK_OPT_DEFAULT);
	char *literal = cmark_render_plaintext(root, CMARK_OPT_DEFAULT, get_fixed_width());
	String plaintext;
	plaintext.parse_utf8(literal);
	free(literal);
	add_text(plaintext);
	set_process_internal(true);
	return OK;
}

void RichTextLabelCommonmark::set_use_commonmark(bool p_enable) {
	use_commonmark = p_enable;
	set_commonmark(commonmark);
	notify_property_list_changed();
}
bool RichTextLabelCommonmark::is_using_commonmark() const {
	return use_commonmark;
}

void RichTextLabelCommonmark::set_commonmark(const String &p_commonmark) {
	commonmark = p_commonmark;

	if (is_inside_tree() && use_commonmark) {
		parse_commonmark(p_commonmark);
	} else { // raw text
		clear();
		add_text(p_commonmark);
	}
}

String RichTextLabelCommonmark::get_commonmark() const {
	return commonmark;
}

void RichTextLabelCommonmark::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (!commonmark.is_empty()) {
				set_commonmark(commonmark);
			}
		} break;
	}
}
