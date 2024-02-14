/*************************************************************************/
/*  vg_adaptive_renderer.cpp                                             */
/*************************************************************************/

#include "vector_graphics_adaptive_renderer.h"
#include "modules/svg_mesh/thirdparty/tove2d/src/cpp/mesh/meshifier.h"

VGMeshRenderer::VGMeshRenderer() :
		quality(1) {
	create_tesselator();
}

void VGMeshRenderer::create_tesselator() {
	tesselator = tove::tove_make_shared<tove::AdaptiveTesselator>(
			new tove::AdaptiveFlattener<tove::DefaultCurveFlattener>(
					tove::DefaultCurveFlattener(2 * quality, 6)));
}

float VGMeshRenderer::get_quality() {
	return quality;
}

void VGMeshRenderer::set_quality(float p_quality) {
	quality = p_quality;
	create_tesselator();
	emit_changed();
}

void VGMeshRenderer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_quality", "quality"), &VGMeshRenderer::set_quality);
	ClassDB::bind_method(D_METHOD("get_quality"), &VGMeshRenderer::get_quality);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "quality", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_quality", "get_quality");
}
