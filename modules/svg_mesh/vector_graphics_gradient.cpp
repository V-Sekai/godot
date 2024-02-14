/*************************************************************************/
/*  vg_gradient.cp          	                                             */
/*************************************************************************/

#include "vector_graphics_gradient.h"
#include "core/core_string_names.h"

void VGGradient::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_color_ramp", "color_ramp"), &VGGradient::set_color_ramp);
	ClassDB::bind_method(D_METHOD("get_color_ramp"), &VGGradient::get_color_ramp);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "color_ramp", PROPERTY_HINT_RESOURCE_TYPE, "Gradient", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT), "set_color_ramp", "get_color_ramp");
}

VGGradient::VGGradient() {
}

void VGGradient::set_color_ramp(const Ref<Gradient> &p_color_ramp) {
	color_ramp = p_color_ramp;
}

Ref<Gradient> VGGradient::get_color_ramp() const {
	return color_ramp;
}
