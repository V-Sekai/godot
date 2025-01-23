/**************************************************************************/
/*  vector_graphics_path.cpp                                              */
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

#include "vector_graphics_path.h"
#include "core/io/file_access.h"
#include "scene/2d/sprite_2d.h"
#include "vector_graphics_adaptive_renderer.h"
#include "vector_graphics_color.h"
#include "vector_graphics_linear_gradient.h"
#include "vector_graphics_radial_gradient.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#endif

static tove::PaintRef to_tove_paint(Ref<VGPaint> p_paint) {
	Ref<VGColor> color = p_paint;
	if (color.is_null()) {
		Ref<VGGradient> gradient = p_paint;
		if (gradient.is_null()) {
			return tove::PaintRef();
		} else {
			Ref<VGLinearGradient> linear_gradient = p_paint;
			Ref<VGRadialGradient> radial_gradient = p_paint;

			tove::PaintRef tove_gradient;

			if (!linear_gradient.is_null()) {
				Vector2 p1 = linear_gradient->get_p1();
				Vector2 p2 = linear_gradient->get_p2();
				tove_gradient = tove::tove_make_shared<tove::LinearGradient>(
						p1.x, p1.y, p2.x, p2.y);
			} else if (!radial_gradient.is_null()) {
				Vector2 center = radial_gradient->get_center();
				Vector2 focal = radial_gradient->get_focal();
				float radius = radial_gradient->get_radius();
				tove_gradient = tove::tove_make_shared<tove::RadialGradient>(
						center.x, center.y, focal.x, focal.y, radius);
			} else {
				return tove::PaintRef();
			}
			Ref<Gradient> color_ramp = gradient->get_color_ramp();
			if (!color_ramp.is_null()) {
				Vector<float> offsets = color_ramp->get_offsets();
				Vector<Color> colors = color_ramp->get_colors();
				for (int i = 0; i < offsets.size(); i++) {
					tove_gradient->addColorStop(
							offsets[i],
							colors[i].r,
							colors[i].g,
							colors[i].b,
							colors[i].a);
				}
			}

			return tove_gradient;
		}
	} else {
		const Color c = color->get_color();
		return tove::tove_make_shared<tove::Color>(c.r, c.g, c.b, c.a);
	}
}

static Ref<VGPaint> from_tove_paint(const tove::PaintRef &p_paint) {
	if (p_paint) {
		if (p_paint->isGradient()) {
			Ref<Gradient> color_ramp;
			color_ramp.instantiate();

			auto grad = std::dynamic_pointer_cast<tove::AbstractGradient>(p_paint);

			if (!color_ramp.is_null()) {
				Vector<float> offsets = color_ramp->get_offsets();
				Vector<Color> colors = color_ramp->get_colors();
				for (int i = 0; i < offsets.size(); i++) {
					grad->addColorStop(
							offsets[i],
							colors[i].r,
							colors[i].g,
							colors[i].b,
							colors[i].a);
				}
			}

			ToveGradientParameters params;
			grad->getGradientParameters(params);

			auto lgrad = std::dynamic_pointer_cast<tove::LinearGradient>(p_paint);
			auto rgrad = std::dynamic_pointer_cast<tove::RadialGradient>(p_paint);
			if (lgrad.get()) {
				Ref<VGLinearGradient> linear_gradient;
				linear_gradient.instantiate();
				linear_gradient->set_p1(Vector2(params.values[0], params.values[1]));
				linear_gradient->set_p2(Vector2(params.values[2], params.values[3]));
				linear_gradient->set_color_ramp(color_ramp);
				return linear_gradient;
			} else if (rgrad.get()) {
				Ref<VGRadialGradient> radial_gradient;
				radial_gradient.instantiate();
				radial_gradient->set_center(Vector2(params.values[0], params.values[1]));
				radial_gradient->set_focal(Vector2(params.values[2], params.values[3]));
				radial_gradient->set_radius(params.values[4]);
				radial_gradient->set_color_ramp(color_ramp);
				return radial_gradient;
			}

			return Ref<VGPaint>();
		} else {
			ToveRGBA rgba;
			p_paint->getRGBA(rgba, 1.0f);
			Ref<VGColor> color;
			color.instantiate();
			color->set_color(Color(rgba.r, rgba.g, rgba.b, rgba.a));
			return color;
		}
	} else {
		return Ref<VGPaint>();
	}
}

static Point2 compute_center(const tove::PathRef &p_path) {
	const float *bounds = p_path->getBounds();
	return Point2((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2);
}

void VGPath::import_svg(const String &p_path) {

	String units = "px";
	float dpi = 96.0;

	Vector<uint8_t> buf = FileAccess::get_file_as_bytes(p_path);

	String str;
	str.parse_utf8((const char *)buf.ptr(), buf.size());

	tove::GraphicsRef tove_graphics = tove::Graphics::createFromSVG(
			str.utf8().ptr(), units.utf8().ptr(), dpi);

	const float *bounds = tove_graphics->getBounds();

	float s = 256.0f / MAX(bounds[2] - bounds[0], bounds[3] - bounds[1]);
	if (s > 1.0f) {
		tove::nsvg::Transform nsvg_transform(s, 0, 0, 0, s, 0);
		nsvg_transform.setWantsScaleLineWidth(true);
		tove_graphics->set(tove_graphics, nsvg_transform);
	}

	const int n = tove_graphics->getNumPaths();
	for (int i = 0; i < n; i++) {
		tove::PathRef current_tove_path = tove_graphics->getPath(i);
		Point2 center = compute_center(current_tove_path);
		current_tove_path->set(current_tove_path, tove::nsvg::Transform(1, 0, -center.x, 0, 1, -center.y));

		VGPath *path = memnew(VGPath(current_tove_path));
		path->set_position(center);

		std::string name = current_tove_path->getName();
		if (name.empty()) {
			name = "Path";
		}

		path->set_name(String(name.c_str()));

		add_child(path, true);
		path->set_owner(get_owner());
	}
}

void VGPath::recenter() {
	Point2 center = compute_center(tove_path);
	tove_path->set(tove_path, tove::nsvg::Transform(1, 0, -center.x, 0, 1, -center.y));
	vg_transform = vg_transform.translated(center);

	set_position(get_position() + get_transform().untranslated().xform(vg_transform.columns[2]));
	vg_transform = vg_transform.untranslated();
	set_dirty();
}

tove::GraphicsRef VGPath::create_tove_graphics() const {
	tove::GraphicsRef tove_graphics = tove::tove_make_shared<tove::Graphics>();
	tove_graphics->addPath(tove_path);
	return tove_graphics;
}

void VGPath::add_tove_path(const tove::GraphicsRef &p_tove_graphics) const {
	const Transform2D t = get_transform() * vg_transform;

	const Vector2 &tx = t.columns[0];
	const Vector2 &ty = t.columns[1];
	const Vector2 &to = t.columns[2];
	tove::nsvg::Transform tove_transform(tx.x, ty.x, to.x, tx.y, ty.y, to.y);

	const tove::PathRef transformed_path = tove::tove_make_shared<tove::Path>();
	transformed_path->set(tove_path, tove_transform);
	p_tove_graphics->addPath(transformed_path);
}

void VGPath::set_inherited_dirty(Node *p_node) {
	const int n = p_node->get_child_count();
	for (int i = 0; i < n; i++) {
		Node *child = p_node->get_child(i);
		if (child->is_class_ptr(get_class_ptr_static())) {
			VGPath *path = Object::cast_to<VGPath>(child);
			if (path->inherits_renderer()) {
				path->set_dirty();
				set_inherited_dirty(path);
			}
		} else {
			set_inherited_dirty(child);
		}
	}
}

void VGPath::compose_graphics(const tove::GraphicsRef &p_tove_graphics,
		const Transform2D &p_transform, const Node *p_node) {

	const int n = p_node->get_child_count();
	for (int i = 0; i < n; i++) {
		Node *child = p_node->get_child(i);

		Transform2D t;
		if (child->is_class_ptr(CanvasItem::get_class_ptr_static())) {
			t = p_transform * Object::cast_to<CanvasItem>(child)->get_transform();
		} else {
			t = p_transform;
		}

		compose_graphics(p_tove_graphics, t, child);
	}

	if (p_node->is_class_ptr(get_class_ptr_static())) {
		const VGPath *path = Object::cast_to<VGPath>(p_node);
		p_tove_graphics->addPath(new_transformed_path(
				path->get_tove_path(), p_transform));
	}
}

bool VGPath::inherits_renderer() const {
	return renderer.is_null();
}

VGPath *VGPath::get_root_path() {
	VGPath *root = this;
	Node *node = get_parent();
	while (node) {
		if (node->is_class_ptr(get_class_ptr_static())) {
			root = Object::cast_to<VGPath>(node);
		}
		node = node->get_parent();
	}
	return root;
}

Ref<VGRenderer> VGPath::get_inherited_renderer() const {
	if (!inherits_renderer()) {
		return renderer;
	}

	Node *node = get_parent();
	while (node) {
		if (node->is_class_ptr(get_class_ptr_static())) {
			VGPath *path = Object::cast_to<VGPath>(node);
			if (path->renderer.is_valid()) {
				return path->renderer;
			}
		}
		node = node->get_parent();
	}
	return Ref<VGRenderer>();
}

void VGPath::update_mesh_representation() {

	if (!dirty) {
		return;
	}
	dirty = false;

	if (!is_empty()) {
		Ref<VGRenderer> current_renderer = get_inherited_renderer();
		if (current_renderer.is_valid()) {
			Ref<Material> ignored_material; // ignored
			Ref<Texture> ignored_texture; // ignored
			_ALLOW_DISCARD_ current_renderer->render_mesh(mesh, ignored_material, ignored_texture, this, false, false);
			texture = current_renderer->render_texture(this, false);
		}
	}
}

void VGPath::update_tove_fill_color() {
	tove_path->setFillColor(to_tove_paint(fill_color));
}

void VGPath::update_tove_line_color() {
	tove_path->setLineColor(to_tove_paint(line_color));
}

void VGPath::create_fill_color() {
	set_fill_color(from_tove_paint(tove_path->getFillColor()));
}

void VGPath::create_line_color() {
	set_line_color(from_tove_paint(tove_path->getLineColor()));
}

void VGPath::_renderer_changed() {
	set_dirty();
	set_inherited_dirty(this);
}

void VGPath::_bubble_change() {
	Node *node = get_parent();
	while (node) {
		if (node->is_class_ptr(get_class_ptr_static())) {
			Object::cast_to<VGPath>(node)->subtree_graphics = tove::GraphicsRef();
		}
		node = node->get_parent();
	}
}

void VGPath::_transform_changed(Node *p_node) {
	if (p_node->is_class_ptr(get_class_ptr_static())) {
		VGPath *path = Object::cast_to<VGPath>(p_node);
		Ref<VGRenderer> renderer = path->get_inherited_renderer();
		if (renderer.is_valid() && renderer->is_dirty_on_transform_change()) {
			path->set_dirty();
		}
	}

	/*const int n = p_node->get_child_count();
	for (int i = 0; i < n; i++) {
		Node *child = p_node->get_child(i);
		_transform_changed(child);
	}*/
}

void VGPath::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_DRAW: {
			update_mesh_representation();
			if (!is_empty()) {
				draw_mesh(mesh, texture, Transform2D());
			}
		} break;
		case NOTIFICATION_PARENTED: {
			_bubble_change();
			if (inherits_renderer()) {
				set_dirty();
				set_inherited_dirty(this);
			}
		} break;
		case NOTIFICATION_UNPARENTED: {
			set_dirty();
			_bubble_change();
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (is_inside_tree()) {
				_bubble_change();
				_transform_changed(this);
			}
		} break;
	}
}

void VGPath::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_renderer", "renderer"), &VGPath::set_renderer);
	ClassDB::bind_method(D_METHOD("get_renderer"), &VGPath::get_renderer);

	ClassDB::bind_method(D_METHOD("set_fill_color", "color"), &VGPath::set_fill_color);
	ClassDB::bind_method(D_METHOD("get_fill_color"), &VGPath::get_fill_color);

	ClassDB::bind_method(D_METHOD("set_line_color", "color"), &VGPath::set_line_color);
	ClassDB::bind_method(D_METHOD("get_line_color"), &VGPath::get_line_color);

	ClassDB::bind_method(D_METHOD("set_line_width", "width"), &VGPath::set_line_width);
	ClassDB::bind_method(D_METHOD("get_line_width"), &VGPath::get_line_width);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "renderer", PROPERTY_HINT_RESOURCE_TYPE, "VGMeshRenderer"), "set_renderer", "get_renderer");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "fill_color", PROPERTY_HINT_RESOURCE_TYPE, "VGColor", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT), "set_fill_color", "get_fill_color");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "line_color", PROPERTY_HINT_RESOURCE_TYPE, "VGColor", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT), "set_line_color", "get_line_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "line_width", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_line_width", "get_line_width");

	ClassDB::bind_method(D_METHOD("insert_curve", "subpath", "t"), &VGPath::insert_curve);
	ClassDB::bind_method(D_METHOD("remove_curve", "subpath", "curve"), &VGPath::remove_curve);
	ClassDB::bind_method(D_METHOD("set_points", "subpath", "points"), &VGPath::set_points);

	ClassDB::bind_method(D_METHOD("_renderer_changed"), &VGPath::_renderer_changed);
	ClassDB::bind_method(D_METHOD("recenter"), &VGPath::recenter);

	ClassDB::bind_method(D_METHOD("import_svg", "path"), &VGPath::import_svg);
}

bool VGPath::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	set_dirty();

	if (name == "name") {
		String s = p_value;
		CharString t = s.utf8();
		tove_path->setName(t.get_data());
	} else if (name == "line_width") {
		tove_path->setLineWidth(p_value);
	} else if (name == "fill_rule") {
		if (p_value == "nonzero") {
			tove_path->setFillRule(TOVE_FILLRULE_NON_ZERO);
		} else if (p_value == "evenodd") {
			tove_path->setFillRule(TOVE_FILLRULE_EVEN_ODD);
		} else {
			return false;
		}
	} else if (name.begins_with("subpaths/")) {
		int subpath = name.get_slicec('/', 1).to_int();
		String subwhat = name.get_slicec('/', 2);

		if (tove_path->getNumSubpaths() == subpath) {
			tove::SubpathRef tove_subpath = tove::tove_make_shared<tove::Subpath>();
			tove_path->addSubpath(tove_subpath);
		}

		ERR_FAIL_INDEX_V(subpath, tove_path->getNumSubpaths(), false);
		tove::SubpathRef tove_subpath = tove_path->getSubpath(subpath);

		if (subwhat == "closed") {
			tove_subpath->setIsClosed(p_value);
		}
		if (subwhat == "points") {
			PackedVector2Array pts = p_value;
			const int n = pts.size();
			if (n > 0) {
				float *buf = new float[n * 2];
				for (int i = 0; i < n; i++) {
					const Vector2 &p = pts[i];
					buf[2 * i + 0] = p.x;
					buf[2 * i + 1] = p.y;
				}
				tove_subpath->setPoints(buf, n, false);
				delete[] buf;
			}
		} else {
			return false;
		}
	} else {
		return false;
	}

	return true;
}

bool VGPath::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;

	if (name == "name") {
		r_ret = String::utf8(tove_path->getName());
	} else if (name == "line_width") {
		r_ret = tove_path->getLineWidth();
	} else if (name == "fill_rule") {
		switch (tove_path->getFillRule()) {
			case TOVE_FILLRULE_NON_ZERO: {
				r_ret = "nonzero";
			} break;
			case TOVE_FILLRULE_EVEN_ODD: {
				r_ret = "evenodd";
			} break;
			default: {
				return false;
			} break;
		}
	} else if (name.begins_with("subpaths/")) {
		int subpath = name.get_slicec('/', 1).to_int();
		String subwhat = name.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(subpath, tove_path->getNumSubpaths(), false);
		tove::SubpathRef tove_subpath = tove_path->getSubpath(subpath);
		if (subwhat == "closed") {
			r_ret = tove_subpath->isClosed();
		} else if (subwhat == "points") {
			const int n = tove_subpath->getNumPoints();
			PackedVector2Array out;
			out.resize(n);
			{
				const float *pts = tove_subpath->getPoints();
				for (int i = 0; i < n; i++) {
					out.write[i] = Vector2(pts[2 * i + 0], pts[2 * i + 1]);
				}
			}
			r_ret = out;
		} else {
			return false;
		}
	} else {
		return false;
	}

	return true;
}

void VGPath::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back(PropertyInfo(Variant::STRING, "name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "line_width", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
	p_list->push_back(PropertyInfo(Variant::INT, "fill_rule", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));

	for (int j = 0; j < tove_path->getNumSubpaths(); j++) {
		const String subpath_prefix = "subpaths/" + itos(j);

		p_list->push_back(PropertyInfo(Variant::BOOL, subpath_prefix + "/closed", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, subpath_prefix + "/points", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
	}
}

Ref<VGRenderer> VGPath::get_renderer() {
	return renderer;
}

void VGPath::set_renderer(const Ref<VGRenderer> &p_renderer) {

	if (renderer.is_valid()) {
		renderer->disconnect("changed", callable_mp(this, &VGPath::_renderer_changed));
	}
	renderer = p_renderer;
	if (renderer.is_valid()) {
		renderer->connect("changed", callable_mp(this, &VGPath::_renderer_changed));
	}

	set_inherited_dirty(this);
	set_dirty();
}

Ref<VGPaint> VGPath::get_fill_color() const {
	return fill_color;
}

void VGPath::set_fill_color(const Ref<VGPaint> &p_paint) {

	if (fill_color == p_paint) {
		return;
	}

	fill_color = p_paint;

	update_tove_fill_color();
	set_dirty();
}

Ref<VGPaint> VGPath::get_line_color() const {
	return line_color;
}

void VGPath::set_line_color(const Ref<VGPaint> &p_paint) {
	if (line_color == p_paint) {
		return;
	}
	line_color = p_paint;
	update_tove_line_color();
	set_dirty();
}

float VGPath::get_line_width() const {
	return tove_path->getLineWidth();
}

void VGPath::set_line_width(const float p_line_width) {
	tove_path->setLineWidth(p_line_width);
	set_dirty();
}

bool VGPath::is_inside(const Point2 &p_point) const {
	return tove_path->isInside(p_point.x, p_point.y);
}

VGPath *VGPath::find_clicked_child(const Point2 &p_point) {
	const int n = get_child_count();
	for (int i = 0; i < n; i++) {
		Node *child = get_child(i);
		if (!child->is_class_ptr(get_class_ptr_static())) {
			continue;
		}
		VGPath *path = Object::cast_to<VGPath>(child);
		if (!path->is_visible()) {
			continue;
		}
		Point2 p = (path->get_transform() * vg_transform).affine_inverse().xform(p_point);
		if (path->is_inside(p)) {
			return path;
		}
	}

	return nullptr;
}
#ifdef TOOLS_ENABLED
Rect2 VGPath::_edit_get_rect() const {
	return tove_bounds_to_rect2(get_subtree_graphics()->getBounds());
}

bool VGPath::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	return get_subtree_graphics()->hit(p_point.x, p_point.y) != tove::PathRef();
}

void VGPath::_edit_set_position(const Point2 &p_position) {
	set_position(p_position);
	notify_property_list_changed();
}

void VGPath::_changed_callback(Object *p_changed, const char *p_prop) {
	if (fill_color.ptr() == p_changed) {
		update_tove_fill_color();
		set_dirty();
	}

	if (line_color.ptr() == p_changed) {
		update_tove_line_color();
		set_dirty();
	}
}
#endif

void VGPath::set_points(int p_subpath, Array p_points) {
	ERR_FAIL_INDEX(p_subpath, tove_path->getNumSubpaths());
	tove::SubpathRef tove_subpath = tove_path->getSubpath(p_subpath);
	const int n = p_points.size();
	float *p = new float[2 * n];
	for (int i = 0; i < n; i++) {
		const Vector2 q = p_points[i];
		p[2 * i + 0] = q.x;
		p[2 * i + 1] = q.y;
	}
	tove_subpath->setPoints(p, n, false);
	set_dirty();
	delete[] p;
}

void VGPath::insert_curve(int p_subpath, float p_t) {
	ERR_FAIL_INDEX(p_subpath, tove_path->getNumSubpaths());
	tove::SubpathRef tove_subpath = tove_path->getSubpath(p_subpath);
	tove_subpath->insertCurveAt(p_t);
	set_dirty();
}

void VGPath::remove_curve(int p_subpath, int p_curve) {
	ERR_FAIL_INDEX(p_subpath, tove_path->getNumSubpaths());
	tove::SubpathRef tove_subpath = tove_path->getSubpath(p_subpath);
	tove_subpath->removeCurve(p_curve);
	set_dirty();
}

void VGPath::set_dirty(bool p_children) {
	if (p_children) {
		set_inherited_dirty(this);
		return;
	}

	Node *node = this;
	while (node) {
		if (node->is_class_ptr(get_class_ptr_static())) {
			Object::cast_to<VGPath>(node)->subtree_graphics = tove::GraphicsRef();
		}
		node = node->get_parent();
	}

	dirty = true;
	notify_property_list_changed();
}

bool VGPath::is_empty() const {
	const int n = tove_path->getNumSubpaths();
	for (int i = 0; i < n; i++) {
		if (tove_path->getSubpath(i)->getNumPoints() > 0) {
			return false;
		}
	}
	return true;
}

int VGPath::get_num_subpaths() const {
	return tove_path->getNumSubpaths();
}

tove::SubpathRef VGPath::get_subpath(int p_subpath) const {
	return tove_path->getSubpath(p_subpath);
}

tove::PathRef VGPath::get_tove_path() const {
	return tove_path;
}

tove::GraphicsRef VGPath::get_subtree_graphics() const {
	if (!subtree_graphics) {
		subtree_graphics = tove::tove_make_shared<tove::Graphics>();
		compose_graphics(subtree_graphics, Transform2D(), this);
	}
	return subtree_graphics;
}

Node2D *VGPath::create_mesh_node() {

	Ref<VGRenderer> current_renderer = get_inherited_renderer();
	if (current_renderer.is_valid()) {
		if (current_renderer->prefer_sprite()) {
			Sprite2D *sprite = memnew(Sprite2D);
			sprite->set_texture(current_renderer->render_texture(this, true));

			// Size2 s = get_global_transform().get_scale();
			Size2 s = get_transform().get_scale();
			float current_scale = MAX(s.width, s.height);

			Transform2D t;
			t.scale(Size2(current_scale, current_scale));

			sprite->set_transform(get_transform() * t.affine_inverse());
			sprite->set_name(get_name());
			sprite->set_centered(false);

			return sprite;
		} else {
			MeshInstance2D *mesh_inst = memnew(MeshInstance2D);
			Ref<ArrayMesh> current_mesh;
			current_mesh.instantiate();
			Ref<Material> current_material;
			Ref<Texture> current_texture;
			_ALLOW_DISCARD_ current_renderer->render_mesh(current_mesh, current_material, current_texture, this, true, false);
			mesh_inst->set_mesh(current_mesh);
			if (current_material.is_valid()) {
				mesh_inst->set_material(current_material);
			}
			if (current_texture.is_valid()) {
				mesh_inst->set_texture(current_texture);
			}

			mesh_inst->set_transform(get_transform());
			mesh_inst->set_name(get_name());
			return mesh_inst;
		}
	}

	return NULL;
}

void VGPath::set_tove_path(tove::PathRef p_path) {
	tove_path = p_path;
	create_fill_color();
	create_line_color();
	set_dirty();
}

VGPath::VGPath() {
	tove_path = tove::tove_make_shared<tove::Path>();
	set_notify_transform(true);

	// when created as a unique item from the UI, populate with default content.

	tove::SubpathRef tove_subpath = tove::tove_make_shared<tove::Subpath>();
	tove_subpath->drawEllipse(0, 0, 100, 100);
	tove_path->addSubpath(tove_subpath);

	tove_path->setFillColor(tove::tove_make_shared<tove::Color>(0.8, 0.1, 0.1));
	tove_path->setLineColor(tove::tove_make_shared<tove::Color>(0, 0, 0));

	set_dirty();
}

VGPath::VGPath(tove::PathRef p_path) {
	set_notify_transform(true);
	set_tove_path(p_path);
}

VGPath::~VGPath() {
}

VGPath *VGPath::create_from_svg(Ref<Resource> p_resource) {
	if (!p_resource->get_path().ends_with(".svg")) {
		return NULL;
	}

	VGPath *root = memnew(VGPath(tove::tove_make_shared<tove::Path>()));

	Ref<VGMeshRenderer> renderer;
	renderer.instantiate();
	root->set_renderer(renderer);

	return root;
}

Node *createVectorSprite(Ref<Resource> p_resource) {
	VGPath *path = VGPath::create_from_svg(p_resource);
	if (path) {
		return path;
	} else {
		return memnew(Sprite2D); // not a vector file
	}
}

void configureVectorSprite(Node *p_child, Ref<Resource> p_resource) {
#ifdef TOOLS_ENABLED
	if (p_child->is_class_ptr(VGPath::get_class_ptr_static())) {
		EditorUndoRedoManager::get_singleton()->add_do_method(p_child, "import_svg", p_resource->get_path());
		EditorUndoRedoManager::get_singleton()->add_do_reference(p_child);
	}
#endif
}
