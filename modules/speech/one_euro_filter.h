#pragma once

#include "core/object/ref_counted.h"

class LowPassFilter {
public:
	double last_value = 0.0f;

	inline float filter(float p_value, float p_alpha) {
		double alpha_clamped = CLAMP(p_alpha, 0.0f, 1.0f);
		last_value = alpha_clamped * p_value + (1.0f - alpha_clamped) * last_value;
		last_value = p_alpha * p_value + (1.0f - p_alpha) * last_value;
		return last_value;
	}

	inline void reset() {
		last_value = 0.0f;
	}
};

class OneEuroFilter : public RefCounted {
	GDCLASS(OneEuroFilter, RefCounted);

	double min_cutoff_freq = 1.0;
	double beta_val = 0.0;
	double d_cutoff_freq = 1.0;

	LowPassFilter *x_lpf = nullptr;
	LowPassFilter *dx_lpf = nullptr;

	bool initialized = false;

protected:
	static void _bind_methods();

private:
	void _clear_filters();
	void _initialize_filters();
	double _compute_alpha(double p_rate, double p_cutoff_f) const;

public:
	OneEuroFilter() {
		_initialize_filters();
	}

	OneEuroFilter(const OneEuroFilter &) = delete;
	OneEuroFilter(OneEuroFilter &&) = delete;
	OneEuroFilter &operator=(OneEuroFilter &&) = delete;
	OneEuroFilter(float p_initial_min_cutoff, float p_initial_beta) :
			min_cutoff_freq(p_initial_min_cutoff),
			beta_val(p_initial_beta),
			d_cutoff_freq(p_initial_min_cutoff) {
		if (min_cutoff_freq < CMP_EPSILON) {
			min_cutoff_freq = CMP_EPSILON;
		}
		if (d_cutoff_freq < CMP_EPSILON) {
			d_cutoff_freq = CMP_EPSILON;
		}
		_initialize_filters();
	}

	~OneEuroFilter() {
		_clear_filters();
	}

	void configure(double p_new_min_cutoff, double p_new_beta);

	void reset();

	double apply(double p_value, double p_delta_time);
};