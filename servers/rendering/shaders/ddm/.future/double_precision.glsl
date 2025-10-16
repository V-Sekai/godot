/* double_precision.glsl */
/*
 * Emulated Double Precision Library for Direct Delta Mush
 *
 * Implements Dekker multiplication and Shewchuk summation algorithms
 * to achieve double precision accuracy using float pairs.
 *
 * Based on:
 * - Dekker, T.J. (1971): "A Floating-Point Technique for Extending the Available Precision"
 * - Shewchuk, J.R. (1997): "Adaptive Precision Floating-Point Arithmetic"
 */

#ifndef DOUBLE_PRECISION_GLSL
#define DOUBLE_PRECISION_GLSL

// Double precision type: float pair (hi, lo)
struct double_t {
	float hi;
	float lo;
};

// Constants
const float SPLIT_CONSTANT = 4097.0;
const double_t DOUBLE_ZERO = double_t(0.0, 0.0);
const double_t DOUBLE_ONE = double_t(1.0, 0.0);

// Convert float to double_t
double_t float_to_double(float x) {
	return double_t(x, 0.0);
}

// Convert double_t to float
float double_to_float(double_t x) {
	return x.hi + x.lo;
}

// Quick two-sum
double_t quick_two_sum(float a, float b) {
	float s = a + b;
	float err = b - (s - a);
	return double_t(s, err);
}

// Two-sum
double_t two_sum(float a, float b) {
	float s = a + b;
	float v = s - a;
	float err = (a - (s - v)) + (b - v);
	return double_t(s, err);
}

// Split float for Dekker
void split(float a, out float a_hi, out float a_lo) {
	float t = SPLIT_CONSTANT * a;
	a_hi = t - (t - a);
	a_lo = a - a_hi;
}

// Two-product
double_t two_product(float a, float b) {
	float p = a * b;
	float a_hi, a_lo, b_hi, b_lo;
	split(a, a_hi, a_lo);
	split(b, b_hi, b_lo);
	float err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
	return double_t(p, err);
}

// Addition
double_t double_add(double_t a, double_t b) {
	double_t s = two_sum(a.hi, b.hi);
	double_t t = two_sum(a.lo, b.lo);
	s.lo += t.hi;
	s = quick_two_sum(s.hi, s.lo);
	s.lo += t.lo;
	s = quick_two_sum(s.hi, s.lo);
	return s;
}

// Multiplication
double_t double_mul(double_t a, double_t b) {
	double_t p = two_product(a.hi, b.hi);
	p.lo += a.hi * b.lo + a.lo * b.hi;
	p = quick_two_sum(p.hi, p.lo);
	return p;
}

// Division
double_t double_div(double_t a, double_t b) {
	float q = a.hi / b.hi;
	double_t r = double_add(a, double_t(-b.hi * q, -b.lo * q));
	float q2 = r.hi / b.hi;
	return two_sum(q, q2);
}

// Square root
double_t double_sqrt(double_t x) {
	if (x.hi <= 0.0) {
		return DOUBLE_ZERO;
	}
	float s = sqrt(x.hi);
	if (s == 0.0) {
		return DOUBLE_ZERO;
	}
	double_t s_dd = float_to_double(s);
	double_t quot = double_div(x, s_dd);
	double_t sum = double_add(s_dd, quot);
	return double_t(sum.hi * 0.5, sum.lo * 0.5);
}

#endif
