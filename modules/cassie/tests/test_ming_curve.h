#ifndef TEST_MING_CURVE_H
#define TEST_MING_CURVE_H

#include "tests/test_macros.h"
#include "../thirdparty/multipolygon_triangulator/MingCurve.h"

namespace TestMingCurve {

TEST_CASE("[Modules][Cassie][MingCurve] Monkey's Saddle") {
    // Define the range and step size for x and y.
    double x_start = -2.0, x_end = 2.0, x_step = 0.1;
    double y_start = -2.0, y_end = 2.0, y_step = 0.1;

    // Calculate the number of points.
    int num_points = ((x_end - x_start) / x_step + 1) * ((y_end - y_start) / y_step + 1);

    // Initialize an array of points representing the Monkey's Saddle.
    double* saddle_points = new double[3 * num_points];

    // Generate the points.
    int index = 0;
    for (double x = x_start; x <= x_end; x += x_step) {
        for (double y = y_start; y <= y_end; y += y_step) {
            double z = pow(x, 3) - 3 * x * pow(y, 2);
            saddle_points[3 * index] = x;
            saddle_points[3 * index + 1] = y;
            saddle_points[3 * index + 2] = z;
            ++index;
        }
    }

    // Create a MingCurve object.
    MingCurve my_curve(saddle_points, num_points, num_points, false);

    // Perform edge protection on the curve.
    bool is_dmwt = true; // Whether to use the DMWT method for edge protection.
    my_curve.edgeProtect(is_dmwt);

    // Clean up.
    delete[] saddle_points;
}

} // namespace TestMingCurve

#endif