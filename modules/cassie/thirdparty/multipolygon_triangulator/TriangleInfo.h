#ifndef _TRIANGLE_INFO_H_
#define _TRIANGLE_INFO_H_
#include <float.h>

class TriangleInfo {
public:
	TriangleInfo() { optCost[0] = optCost[1] = optCost[2] = FLT_MIN; }
	~TriangleInfo() {}
	int getOptSize() {
		return sizeof(optCost) + sizeof(optTile);
	}
	int getSize() {
		return sizeof(optCost) + sizeof(optTile) + sizeof(edgeIndex) +
				sizeof(triIndex);
	}
	int edgeIndex[3]; // index of {1st,2nd,3rd} edge in edgeInfoList;
	int triIndex[3]; // index of this triangle in the $side list of edge {1,2,3}
	float optCost[3]; // optimal cost, init as BIGNUM
	int optTile[3]; // optimal tiling, init as 0
};

#endif