#ifndef _EDGE_INFO_H_
#define _EDGE_INFO_H_
#include <float.h>

class EdgeInfo {
public:
	EdgeInfo();
	~EdgeInfo();

	int v1;
	int v2;
	int leftsize;
	int rightsize;
	int *leftTris; // list of triangle index on the LEFT
	char *leftEdgeInd; // ith edge in corresponding triangle
	int *rightTris; // list of triangle index on the RIGHT
	char *rightEdgeInd; // ith edge in corresponding triangle
	int get_size();
};

#endif