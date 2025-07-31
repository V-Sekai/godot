/**************************************************************************/
/*  one_euro_filter.cpp                                                   */
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

#include "one_euro_filter.h"

#include <cmath>

OneEuroFilter::OneEuroFilter() {
    // Use default configuration
}

OneEuroFilter::OneEuroFilter(const FilterConfig &p_config) : config(p_config) {
}

OneEuroFilter::OneEuroFilter(float p_cutoff, float p_beta) {
    config.cutoff = p_cutoff;
    config.beta = p_beta;
}

float OneEuroFilter::low_pass_filter(float value, float alpha, float &prev_filtered_value) {
    float filtered_value = alpha * value + (1.0f - alpha) * prev_filtered_value;
    prev_filtered_value = filtered_value;
    return filtered_value;
}

float OneEuroFilter::smoothing_factor(float cutoff, float delta_time) {
    float r = 2.0f * 3.14159265359f * cutoff * delta_time;
    return r / (r + 1.0f);
}

float OneEuroFilter::filter(float value, float delta_time) {
    // Handle first call
    if (first_time) {
        first_time = false;
        prev_value = value;
        prev_derivative = 0.0f;
        prev_time = 0.0f;
        return value;
    }
    
    // Ensure delta_time is positive and reasonable
    if (delta_time <= 0.0f) {
        delta_time = 0.001f; // Default to 1ms if invalid
    }
    
    // Calculate derivative
    float derivative = (value - prev_value) / delta_time;
    
    // Filter the derivative
    float alpha_d = smoothing_factor(config.derivate_cutoff, delta_time);
    float filtered_derivative = low_pass_filter(derivative, alpha_d, prev_derivative);
    
    // Calculate adaptive cutoff frequency
    float cutoff_freq = config.min_cutoff + config.beta * std::abs(filtered_derivative);
    
    // Filter the value
    float alpha = smoothing_factor(cutoff_freq, delta_time);
    float filtered_value = low_pass_filter(value, alpha, prev_value);
    
    return filtered_value;
}

void OneEuroFilter::set_config(const FilterConfig &p_config) {
    config = p_config;
}

OneEuroFilter::FilterConfig OneEuroFilter::get_config() const {
    return config;
}

void OneEuroFilter::reset() {
    first_time = true;
    prev_value = 0.0f;
    prev_derivative = 0.0f;
    prev_time = 0.0f;
}

void OneEuroFilter::set_cutoff(float p_cutoff) {
    config.cutoff = p_cutoff;
}

void OneEuroFilter::set_beta(float p_beta) {
    config.beta = p_beta;
}

void OneEuroFilter::set_min_cutoff(float p_min_cutoff) {
    config.min_cutoff = p_min_cutoff;
}

void OneEuroFilter::set_derivate_cutoff(float p_derivate_cutoff) {
    config.derivate_cutoff = p_derivate_cutoff;
}

float OneEuroFilter::get_cutoff() const {
    return config.cutoff;
}

float OneEuroFilter::get_beta() const {
    return config.beta;
}

float OneEuroFilter::get_min_cutoff() const {
    return config.min_cutoff;
}

float OneEuroFilter::get_derivate_cutoff() const {
    return config.derivate_cutoff;
}
