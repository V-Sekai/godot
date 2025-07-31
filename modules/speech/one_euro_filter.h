/**************************************************************************/
/*  one_euro_filter.h                                                     */
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

/**
 * OneEuroFilter - A low-pass filter for smoothing noisy signals
 * 
 * This filter is designed to reduce jitter in real-time data streams
 * while maintaining responsiveness. It's particularly effective for
 * audio timing synchronization where we need to smooth network jitter
 * without introducing significant latency.
 * 
 * Based on the 1€ Filter by Géry Casiez, Nicolas Roussel, and Daniel Vogel
 * https://cristal.univ-lille.fr/~casiez/1euro/
 */
class OneEuroFilter {
public:
    struct FilterConfig {
        float cutoff = 0.1f;           // Lower = less jitter, more lag
        float beta = 5.0f;             // Higher = less lag, more jitter  
        float min_cutoff = 1.0f;       // Minimum cutoff frequency
        float derivate_cutoff = 1.0f;  // Derivative filter cutoff
    };

private:
    FilterConfig config;
    
    // Filter state
    bool first_time = true;
    float prev_value = 0.0f;
    float prev_derivative = 0.0f;
    float prev_time = 0.0f;
    
    // Low-pass filter implementation
    float low_pass_filter(float value, float alpha, float &prev_filtered_value);
    
    // Calculate smoothing factor
    float smoothing_factor(float cutoff, float delta_time);

public:
    OneEuroFilter();
    OneEuroFilter(const FilterConfig &p_config);
    OneEuroFilter(float p_cutoff, float p_beta);
    
    // Main filtering function
    float filter(float value, float delta_time);
    
    // Configuration
    void set_config(const FilterConfig &p_config);
    FilterConfig get_config() const;
    
    // Reset filter state
    void reset();
    
    // Individual parameter setters
    void set_cutoff(float p_cutoff);
    void set_beta(float p_beta);
    void set_min_cutoff(float p_min_cutoff);
    void set_derivate_cutoff(float p_derivate_cutoff);
    
    // Individual parameter getters
    float get_cutoff() const;
    float get_beta() const;
    float get_min_cutoff() const;
    float get_derivate_cutoff() const;
};
