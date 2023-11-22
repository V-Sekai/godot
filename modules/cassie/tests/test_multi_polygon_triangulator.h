
#ifndef TEST_MULTI_POLYGON_TRIANGULATOR_H
#define TEST_MULTI_POLYGON_TRIANGULATOR_H

#include "tests/test_macros.h"

#include "../thirdparty/multipolygon_triangulator/DMWT.h"

namespace TestPolygonTriangulation {

TEST_CASE("[Modules][Cassie][PolygonTriangulation] monkey's saddle") {
    // Define a series of points from a monkey's saddle
    int ptn = 100; // Number of points
    double pts[ptn * 3]; // Array to hold x, y, z coordinates of each point
    for (int i = 0; i < ptn; ++i) {
        double x = (double)i / ptn;
        double y = (double)i / ptn;
        double z = pow(x, 3) - 3 * x * pow(y, 2); // Monkey's saddle equation
        pts[i * 3] = x;
        pts[i * 3 + 1] = y;
        pts[i * 3 + 2] = z;
    }

    // Create PolygonTriangulation object with these points
    Ref<PolygonTriangulation> polyTri = PolygonTriangulation::_create_with_degenerates(ptn, pts, nullptr, false);
    CHECK(polyTri.is_valid());
}
} //namespace TestPolygonTriangulation

#endif