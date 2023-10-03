#include "EdgeInfo.h"

EdgeInfo::EdgeInfo() {
	leftsize = -1;
	rightsize = -1;
}

EdgeInfo::~EdgeInfo() {
	delete[] leftTris;
	delete[] leftEdgeInd;
	delete[] rightTris;
	delete[] rightEdgeInd;
}

int EdgeInfo::get_size() {
	int totalSize = 0;
	totalSize += leftsize * (sizeof(int) + sizeof(char));
	totalSize += rightsize * (sizeof(int) + sizeof(char));
	return totalSize;
}
