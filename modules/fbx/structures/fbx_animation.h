/**************************************************************************/
/*  fbx_animation.h                                                       */
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

#ifndef FBX_ANIMATION_H
#define FBX_ANIMATION_H

#include "scene/animation/animation_player.h"

class FBXAnimation : public Resource {
	GDCLASS(FBXAnimation, Resource);

protected:
	static void _bind_methods();

public:
	enum Interpolation {
		INTERP_LINEAR,
		INTERP_STEP,
		INTERP_CATMULLROMSPLINE,
		INTERP_CUBIC_SPLINE,
	};

	template <class T>
	struct Channel {
		Interpolation interpolation = INTERP_LINEAR;
		Vector<real_t> times;
		Vector<T> values;
	};

	struct Track {
		Channel<Vector3> position_track;
		Channel<Quaternion> rotation_track;
		Channel<Vector3> scale_track;
	};

	struct BlendShapeTrack {
		Channel<real_t> weight_track;
	};

public:
	bool get_loop() const;
	void set_loop(bool p_val);
	double get_time_begin() const;
	void set_time_begin(double p_val);
	double get_time_end() const;
	void set_time_end(double p_val);
	HashMap<int, FBXAnimation::Track> &get_tracks();
	HashMap<int, BlendShapeTrack> &get_blend_tracks();
	FBXAnimation();

private:
	bool loop = false;
	double time_begin = 0.0;
	double time_end = 0.0;
	HashMap<int, Track> tracks;
	HashMap<int, BlendShapeTrack> blend_tracks;
};

#endif // FBX_ANIMATION_H
