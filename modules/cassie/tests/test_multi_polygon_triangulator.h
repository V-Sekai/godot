
#ifndef TEST_MULTI_POLYGON_TRIANGULATOR_H
#define TEST_MULTI_POLYGON_TRIANGULATOR_H

#include "tests/test_macros.h"

#include "../src/DMWT.h"

namespace TestPolygonTriangulation {
TEST_CASE("[Modules][PolygonTriangulation] create") {
	Ref<PolygonTriangulation> triangulator = PolygonTriangulation::_create();

	triangulator->set_weights(1.0f, 2.0f, 3.0f, 4.0f, 5.0f);

	bool result = triangulator->start();

	CHECK(result == true);
}
} //namespace TestPolygonTriangulation

#endif