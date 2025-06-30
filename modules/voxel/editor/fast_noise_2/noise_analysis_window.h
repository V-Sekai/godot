/**************************************************************************/
/*  noise_analysis_window.h                                               */
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

#pragma once

#include "../../util/noise/fast_noise_2.h"
#include <scene/gui/dialogs.h>

class SpinBox;
class LineEdit;
class ProgressBar;
class OptionButton;

namespace zylann {

class ChartView;

// This is an experimental tool to check noise properties empirically,
// by sampling it a lot of times and seeing what the minimum and maximum values are.
class NoiseAnalysisWindow : public AcceptDialog {
	GDCLASS(NoiseAnalysisWindow, AcceptDialog)
public:
	NoiseAnalysisWindow();

	void set_noise(Ref<FastNoise2> noise);

private:
	enum Dimension { //
		DIMENSION_2D = 0,
		DIMENSION_3D,
		_DIMENSION_COUNT
	};

	void _on_calculate_button_pressed();
	void _notification(int p_what);
	void _process();

	static void _bind_methods();

	Ref<FastNoise2> _noise;

	OptionButton *_dimension_option_button = nullptr;
	SpinBox *_step_count_spinbox = nullptr;
	SpinBox *_step_minimum_length_spinbox = nullptr;
	SpinBox *_step_maximum_length_spinbox = nullptr;
	SpinBox *_area_size_spinbox = nullptr;
	SpinBox *_samples_count_spinbox = nullptr;

	ChartView *_chart_view = nullptr;

	ProgressBar *_progress_bar = nullptr;

	LineEdit *_minimum_value_line_edit = nullptr;
	LineEdit *_maximum_value_line_edit = nullptr;
	LineEdit *_maximum_derivative_line_edit = nullptr;

	Button *_calculate_button = nullptr;

	struct AnalysisParams {
		Dimension dimension;
		int step_count;
		float step_minimum_length;
		float step_maximum_length;
		float area_size;
		int samples_count;
	};

	AnalysisParams _analysis_params;
	int _current_step = -1;

	struct AnalysisResults {
		float minimum_value;
		float maximum_value;
		float maximum_derivative;
		PackedVector2Array maximum_derivative_per_step_length;
	};

	AnalysisResults _results;
};

} // namespace zylann
