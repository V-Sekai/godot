#ifndef _MINGCURVE_H_
#define _MINGCURVE_H_

#ifndef _CONF_H_
#define _CONF_H_

#define DO_EXP false
#define GO_CMD 1

#define SAVE_FACE 0
#define SAVE_TILE 0
#define SAVE_NEWCURVE 0

#define PI 3.141593
#define hfPI 1.570796
#define BADEDGE_LIMIT 30
#define plainEPS 0.001
#define plainPTB 0.00000001

// measurements
#define USEONLYWORSTBITRI 0

#define SAVEOBJ 0

#endif

#include "core/io/file_access.h"
#include "core/math/delaunay_3d.h"
#include "core/variant/variant.h"
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "thirdparty/eigen/Eigen/Dense"

using namespace std;

int delaunayRestrictedTriangulation(const double *inCurve, const int inNum,
		double **outCurve, int *outPn,
		double **outFaces, int *outNum,
		float *weights, bool dosmooth, int subd,
		int laps, Eigen::MatrixXd &V,
		Eigen::MatrixXi &F);

/**
 * The MingCurve class is used to represent a curve in 3D space. It provides methods for reading and manipulating the curve data.
 *
 * <p> This class includes methods for edge protection, statistics calculation, and handling degenerated cases. It also allows saving the curve data to a file.
 *
 */
class MingCurve {
public:
	MingCurve(const double *inCurve, const int inNum, int limit, bool hasNorm);
	MingCurve(const double *inCurve, const float *inNrom, const int inNum,
			int limit, bool hasNorm);
	~MingCurve();
	int getNumOfPoints();
	double *getPoints();
	double *getDeGenPoints();
	char *getFilename();
	float *getNormal();

	//----------------Edge protection----------------//
	bool edgeProtect(bool isdmwt);
	void saveCurve(const String &curvefilein, PackedFloat64Array pts, int num);

	//-------------evaluations--------------//
	float timeReadIn;
	float timeEdgeProtect;
	void statistics();

	// ------------------- for cycle project -----------------//
	bool isDeGen; // degenerated cases: plane
	bool badInput;

	//----------------for edge protection------------//
	bool loadOrgCurve(const double *inCurve, const int inNum);
	void loadOrgNorm(const float *inNorm);
private:
	char *filename;
	int numofpoints;
	double *points;
	double *DeGenPoints;
	float *normals;
	int PT_LIMIT;
	bool EXPSTOP;
	bool withNorm;
	int org_n;
	int n_before;
	int n_after;
	float n_ratio;
	std::vector<Vector3> tempC;
	std::vector<Vector3> tempOrgC;
	std::vector<std::vector<int>> tempAdj;
	std::vector<Vector3> tempNorm;
	std::vector<std::vector<int>> tempAdjNorm;
	void protectCorner();
	double getAngle(int p1, int p2, int p3);
	double getPt2LineDist(int p1, int p2, int p3);
	void splitEdge(int p1, int p2ind, const Vector3 &newP);
	void insertMidPointsTetgen();
	std::vector<std::vector<int>> badEdge;
	std::vector<int> newEdge;
	std::vector<int> newAdj;
	std::vector<int> newNorm;
	std::vector<char> newClip;
	bool isProtected();
	void callTetgen();

	// ------------------- for cycle project -----------------//
	bool isDeGenCase();
	// bool isDeGen; // degenerated cases: plane
	void splitEdge(int p1, int p2ind, const Vector3 &newP, const Vector3 &newOrgP);

	bool getCurveAfterEP();
	bool sameOrientation(const vector<int> &newCurve);
	bool passTetGen();

	std::vector<double> radius;
	std::vector<double> orgradius;
	std::vector<std::vector<char>> cliped;
	int perturbNum;

	// not used for now
	int numofcurves;
	int numofnormals;
	bool isOpen;
	int capacity;
};

#endif