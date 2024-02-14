/*************************************************************************/
/*  vg_renderer.h                                                        */
/*************************************************************************/

#include "vector_graphics_renderer.h"

void VGRenderer::clear_mesh(Ref<ArrayMesh> &p_mesh) {
	if (p_mesh.is_valid()) {
		p_mesh->clear_surfaces();
	} else {
		p_mesh.instantiate();
	}
}
