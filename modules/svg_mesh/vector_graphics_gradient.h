/*************************************************************************/
/*  vg_gradient.h 	                                                     */
/*************************************************************************/

#ifndef VG_GRADIENT_H
#define VG_GRADIENT_H

#include "scene/resources/gradient.h"
#include "vector_graphics_paint.h"

class VGGradient : public VGPaint {
	GDCLASS(VGGradient, VGPaint);

	Ref<Gradient> color_ramp;

protected:
	static void _bind_methods();

public:
	VGGradient();

	void set_color_ramp(const Ref<Gradient> &p_color_ramp);
	Ref<Gradient> get_color_ramp() const;
};

#endif // VG_GRADIENT_H
