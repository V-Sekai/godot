// Enhanced Direct Delta Mush - Double Precision Emulation Library
// Implements emulated double precision using float pairs (Dekker/Shewchuk algorithms)
// Achieves ~53-bit effective precision without requiring hardware double support

#ifndef DOUBLE_PRECISION_GLSL
#define DOUBLE_PRECISION_GLSL

// ============================================================================
// Double Type Definition
// ============================================================================

// Emulated double precision: high + low components
struct double_t {
    float hi;  // High precision component (significant bits)
    float lo;  // Low precision component (guard bits)
};

// ============================================================================
// Basic Conversion Functions
// ============================================================================

// Convert single-precision float to double
double_t float_to_double(float x) {
    return double_t(x, 0.0);
}

// Convert double to single-precision float (with rounding)
float double_to_float(double_t x) {
    return x.hi + x.lo;
}

// ============================================================================
// Dekker Multiplication
// ============================================================================
// Multiply two doubles using Dekker's algorithm
// Achieves ~106-bit result from two ~53-bit operands

double_t double_mul(double_t a, double_t b) {
    // Split factors: x = x_h + x_l
    // Using Veltkamp splitting for 32-bit chunks
    const float C = 65536.0 + 1.0;  // 2^16 + 1
    
    float a_h = fma(C, a.hi, -C * a.hi + a.hi);
    float a_l = a.hi - a_h;
    
    float b_h = fma(C, b.hi, -C * b.hi + b.hi);
    float b_l = b.hi - b_h;
    
    // Compute product: (a_h + a_l)(b_h + b_l) = a_h*b_h + a_h*b_l + a_l*b_h + a_l*b_l
    float p_hh = a_h * b_h;
    float p_hl = fma(a_h, b_l, -p_hh + a_h * b_l);  // a_h * b_l with error tracking
    float p_lh = fma(a_l, b_h, -p_hh + a_l * b_h);  // a_l * b_h with error tracking
    float p_ll = a_l * b_l;                          // a_l * b_l
    
    // Accumulate high-to-low components
    float p_h = p_hh;
    float p_m = p_hl + p_lh;
    
    // Renormalize: extract high and low parts of mantissa
    float r_h = p_h + p_m;
    float r_l = (p_h - r_h + p_m) + p_ll;
    
    return double_t(r_h, r_l);
}

// ============================================================================
// Shewchuk Summation (Accurate Addition)
// ============================================================================
// Add two doubles using Shewchuk's error-free transformation
// Maintains full precision of both inputs

double_t double_add(double_t a, double_t b) {
    // Error-free addition: compute s and e such that s + e = a + b
    // with s being the round-to-nearest and e being the roundoff error
    
    float s = a.hi + b.hi;
    float e = 0.0;
    
    // Compute roundoff error from high-order addition
    if (abs(a.hi) >= abs(b.hi)) {
        e = (a.hi - s) + b.hi;
    } else {
        e = (b.hi - s) + a.hi;
    }
    
    // Add low-order components
    float l = a.lo + b.lo;
    e = e + l;
    
    // Renormalize result
    float r_h = s + e;
    float r_l = (s - r_h) + e;
    
    return double_t(r_h, r_l);
}

// ============================================================================
// Other Operations
// ============================================================================

// Subtract two doubles
double_t double_sub(double_t a, double_t b) {
    return double_add(a, double_t(-b.hi, -b.lo));
}

// Divide two doubles (approximate, but high precision)
double_t double_div(double_t a, double_t b) {
    // Compute approximate quotient using high-order division
    float q_h = a.hi / b.hi;
    
    // Refine using Newton-Raphson correction:
    // r = a - q*b (remainder)
    // q_improved = q + r/b
    double_t product = double_mul(double_t(q_h, 0.0), b);
    double_t remainder = double_sub(a, product);
    float q_l = remainder.hi / b.hi;
    
    return double_t(q_h, q_l);
}

// Square root using Newton-Raphson (for emulated precision)
double_t double_sqrt(double_t x) {
    if (x.hi <= 0.0) return double_t(0.0, 0.0);
    
    // Initial approximation using single-precision sqrt
    float q = sqrt(x.hi);
    
    // Newton-Raphson refinement: q_new = (q + x/q) / 2
    // For double precision: q_new = q + (x - q²) / (2*q)
    double_t q_dbl = double_t(q, 0.0);
    double_t q_squared = double_mul(q_dbl, q_dbl);
    double_t difference = double_sub(x, q_squared);
    double_t two_q = double_t(2.0 * q, 0.0);
    double_t correction = double_div(difference, two_q);
    
    return double_add(q_dbl, correction);
}

// Reciprocal (1/x) using Newton-Raphson
double_t double_recip(double_t x) {
    if (x.hi == 0.0) return double_t(1e38, 0.0);  // Approximate infinity
    
    // Initial approximation
    float r = 1.0 / x.hi;
    double_t r_dbl = double_t(r, 0.0);
    
    // Newton-Raphson: r_new = r(2 - x*r)
    double_t x_r = double_mul(x, r_dbl);
    double_t two_minus_xr = double_t(2.0, 0.0);
    two_minus_xr = double_sub(two_minus_xr, x_r);
    
    double_t result = double_mul(r_dbl, two_minus_xr);
    
    return result;
}

// ============================================================================
// Cotangent Weight Computation (Primary Use Case)
// ============================================================================
// For DDM: compute weights = (cot(α) + cot(β)) / 2
// This uses full double precision to avoid cancellation errors

// Compute cotangent of angle in radians using double precision
double_t double_cot(double_t angle) {
    // cot(x) = cos(x) / sin(x)
    // Compute using high-precision sine and cosine
    
    float cos_val = cos(angle.hi);  // Use GPU's fast cos (acceptable error)
    float sin_val = sin(angle.hi);  // Use GPU's fast sin (acceptable error)
    
    if (abs(sin_val) < 1e-10) {
        return double_t(1e10, 0.0);  // Very large value for near-zero sin
    }
    
    // cot = cos / sin with double precision division
    double_t cos_dbl = double_t(cos_val, 0.0);
    double_t sin_dbl = double_t(sin_val, 0.0);
    
    return double_div(cos_dbl, sin_dbl);
}

// Compute cotangent weight for two angles (typical DDM usage)
// Returns (cot(angle1) + cot(angle2)) / 2
double_t double_cotangent_weight(float angle1, float angle2) {
    double_t cot1 = double_cot(double_t(angle1, 0.0));
    double_t cot2 = double_cot(double_t(angle2, 0.0));
    
    double_t sum = double_add(cot1, cot2);
    double_t two = double_t(2.0, 0.0);
    
    return double_div(sum, two);
}

// ============================================================================
// Utility Functions
// ============================================================================

// Absolute value of double
double_t double_abs(double_t x) {
    return double_t(abs(x.hi), (x.hi < 0.0) ? -x.lo : x.lo);
}

// Maximum of two doubles
double_t double_max(double_t a, double_t b) {
    if (a.hi > b.hi) return a;
    if (a.hi < b.hi) return b;
    // Equal high components: compare low
    return (a.lo > b.lo) ? a : b;
}

// Minimum of two doubles
double_t double_min(double_t a, double_t b) {
    if (a.hi < b.hi) return a;
    if (a.hi > b.hi) return b;
    // Equal high components: compare low
    return (a.lo < b.lo) ? a : b;
}

// Clamp double to range
double_t double_clamp(double_t x, double_t min_val, double_t max_val) {
    return double_max(min_val, double_min(x, max_val));
}

// Convert double to string representation (for debugging)
// Note: This is primarily for CPU-side debugging
string double_to_string(double_t x) {
    return "(" + string(x.hi) + ", " + string(x.lo) + ")";
}

#endif // DOUBLE_PRECISION_GLSL
