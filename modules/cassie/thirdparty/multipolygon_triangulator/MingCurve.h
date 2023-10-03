#ifndef _MINGCURVE_H_
#define _MINGCURVE_H_

#ifndef _CONF_H_
#define _CONF_H_

#include "Errors.h"

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

using namespace std;

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

private:
	char *filename;
	int numofpoints;
	double *points;
	double *DeGenPoints;
	float *normals;
	int PT_LIMIT;
	bool EXPSTOP;
	bool withNorm;
	//----------------for edge protection------------//
	bool loadOrgCurve(const double *inCurve, const int inNum);
	void loadOrgNorm(const float *inNorm);
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
	void perturbPts(double ptb);
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

MingCurve::MingCurve(const double *inCurve, const int inNum, int limit,
		bool hasNorm) {
	numofpoints = 0;
	PT_LIMIT = limit;
	EXPSTOP = false;
	withNorm = hasNorm;
	isDeGen = false;
	badInput = false;
	perturbNum = 0;
	newEdge.push_back(0);
	newEdge.push_back(0);
	newAdj.push_back(0);
	newAdj.push_back(0);
	newNorm.push_back(0);
	newNorm.push_back(0);
	newClip.push_back(0);
	newClip.push_back(0);
	loadOrgCurve(inCurve, inNum);
}

MingCurve::MingCurve(const double *inCurve, const float *inNorm,
		const int inNum, int limit, bool hasNorm) {
	numofpoints = 0;
	PT_LIMIT = limit;
	EXPSTOP = false;
	withNorm = hasNorm;
	isDeGen = false;
	badInput = false;
	perturbNum = 0;
	newEdge.push_back(0);
	newEdge.push_back(0);
	newAdj.push_back(0);
	newAdj.push_back(0);
	newNorm.push_back(0);
	newNorm.push_back(0);
	newClip.push_back(0);
	newClip.push_back(0);
	loadOrgCurve(inCurve, inNum);
	if (withNorm) {
		loadOrgNorm(inNorm);
	}
}

MingCurve::~MingCurve() {
	if (!badInput && !isDeGen) {
		delete[] points;
		if (withNorm) {
			delete[] normals;
		}
	}
}

int MingCurve::getNumOfPoints() { return numofpoints; }
double *MingCurve::getPoints() { return points; }
double *MingCurve::getDeGenPoints() { return DeGenPoints; }
float *MingCurve::getNormal() { return normals; }

void MingCurve::loadOrgNorm(const float *inNorm) {
	numofnormals = org_n;
	// read normal set data
	for (int i = 0; i < numofnormals; i++) {
		tempNorm.push_back(
				Vector3(inNorm[i * 3], inNorm[i * 3 + 1], inNorm[i * 3 + 2]));
		newNorm[0] = i;
		newNorm[1] = i - 1;
		tempAdjNorm.push_back(newNorm);
	}
	tempAdjNorm[0][1] = org_n - 1;

	withNorm = true;
}

void MingCurve::saveCurve(const String &curvefilein, PackedFloat64Array pts, int num) {
	String newcurvefile = curvefilein + ".EP.curve";
	Ref<FileAccess> writer = FileAccess::open(newcurvefile, FileAccess::WRITE);
	if (writer.is_valid()) {
		writer->store_string("1\n");
		writer->store_string(vformat("%d 0 1\n", num));
		for (int i = 0; i < num; i++) {
			writer->store_string(vformat("%f %f %f\n", pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]));
		}
		writer->close();
	}
}

//=====================================Get
//Measures==================================//
double MingCurve::getAngle(int p1, int p2, int p3) {
	double res = 0.0;
	Vector3 v1 = tempC[p1];
	Vector3 v2 = tempC[p2];
	Vector3 v3 = tempC[p3];
	Vector3 vec1 = v1 - v2;
	Vector3 vec2 = v1 - v3;
	vec1.normalize();
	vec2.normalize();
	res = vec1.dot(vec2);
	res = res < -1.0 ? -1.0 : res;
	res = res > 1.0 ? 1.0 : res;
	return acos(res);
}

double MingCurve::getPt2LineDist(int p1, int p2, int p3) {
	Vector3 v1 = tempC[p1];
	Vector3 v2 = tempC[p2];
	Vector3 v3 = tempC[p3];
	Vector3 vec12 = v2 - v1;
	Vector3 vec23 = v3 - v2;
	Vector3 vec13 = v3 - v1;

	// Check for zero vectors
	if (vec12.is_zero_approx() || vec23.is_zero_approx()) {
		return vec13.length();
	}

	if (vec12.dot(vec23) > 0) {
		return vec12.length();
	}
	if (vec13.dot(vec23) < 0) {
		return vec13.length();
	}

	return sqrt(vec13.cross(vec23).length_squared() / vec23.length_squared());
}

//=====================================Get
//Measures==================================//
void MingCurve::splitEdge(int p1, int p2ind, const Vector3 &newP) {
	// add new point
	int p2 = tempAdj[p1][p2ind];
	tempC.push_back(newP);
	org_n++;
	newAdj[0] = p1;
	newAdj[1] = p2;
	tempAdj.push_back(newAdj);
	if (withNorm) {
		// and its adj norms
		int newNind = tempAdjNorm[p1][p2ind];
		Vector3 newN = tempNorm[newNind];
		tempNorm.push_back(newN);
		newNorm[0] = newNind;
		newNorm[1] = newNind;
		tempAdjNorm.push_back(newNorm);
	}
	tempAdj[p1][p2ind] = org_n - 1;
	int pos = 0;
	if (tempAdj[p2][0] != p1)
		pos = 1;
	tempAdj[p2][pos] = org_n - 1;
}

void MingCurve::splitEdge(int p1, int p2ind, const Vector3 &newP,
		const Vector3 &newOrgP) {
	// add new point
	int p2 = tempAdj[p1][p2ind];
	tempC.push_back(newP);
	org_n++;
	tempOrgC.push_back(newOrgP);
	newAdj[0] = p1;
	newAdj[1] = p2;
	tempAdj.push_back(newAdj);
	if (withNorm) {
		// and its adj norms
		int newNind = tempAdjNorm[p1][p2ind];
		Vector3 newN = tempNorm[newNind];
		tempNorm.push_back(newN);
		newNorm[0] = newNind;
		newNorm[1] = newNind;
		tempAdjNorm.push_back(newNorm);
	}
	tempAdj[p1][p2ind] = org_n - 1;
	int pos = 0;
	if (tempAdj[p2][0] != p1)
		pos = 1;
	tempAdj[p2][pos] = org_n - 1;
}
// Protect acute corners
void MingCurve::protectCorner() {
	radius.clear();
	cliped.clear();
	vector<int> acuteList;

	int sizeCliped = org_n;
	for (int i = 0; i < org_n; i++) {
		newClip[0] = 0;
		newClip[1] = 0;
		cliped.push_back(newClip);
		radius.push_back(FLT_MAX);
	}
	if (isDeGen) {
		orgradius.clear();
		for (int i = 0; i < org_n; i++) {
			orgradius.push_back(FLT_MAX);
		}
	}
	// gather acute corners
	for (int i = 0; i < org_n; i++) {
		if (getAngle(i, tempAdj[i][0], tempAdj[i][1]) < hfPI) {
			acuteList.push_back(i);
			cliped[i][0] = 1;
			cliped[i][1] = 1;
		}
	}
	// computer distance to other edges (circular protection)
	int acsize = acuteList.size();
	int p1, p2, p3;
	int end = org_n - 1;
	double dist;
	for (int i = 0; i < acsize; i++) {
		p1 = acuteList[i];
		for (int j = 0; j < org_n; j++) {
			p2 = j;
			p3 = j + 1;
			if (j == end) {
				p2 = 0;
				p3 = end;
			}
			if (p1 == p2 || p1 == p3)
				continue;
			dist = getPt2LineDist(p1, p2, p3);
			radius[p1] = std::min(dist, radius[p1]);
		}
	}
	// computer cut positons std::min(dist,S,L/3,2D/3)
	for (int i = 0; i < acsize; i++) {
		double M = FLT_MAX;
		double L = FLT_MAX;
		double D = FLT_MAX;
		double S = FLT_MAX;

		double orgM = FLT_MAX;
		double orgL = FLT_MAX;
		double orgD = FLT_MAX;
		double orgS = FLT_MAX;
		int pos;
		p1 = acuteList[i];
		for (int j = 0; j < 2; j++) {
			p2 = tempAdj[p1][j];

			M = (tempC[p1] - tempC[p2]).length();
			S = std::min(S, M);
			if (cliped[p1][j])
				L = std::min(L, M);
			// int pos = 0;
			if (tempAdj[p2][0] != p1)
				pos = 1;
			if (cliped[p2][pos])
				D = std::min(D, M);

			if (isDeGen) {
				orgM = (tempOrgC[p1] - tempOrgC[p2]).length();
				orgS = std::min(orgS, orgM);
				if (cliped[p1][j])
					orgL = std::min(orgL, orgM);
				if (cliped[p2][pos])
					orgD = std::min(orgD, orgM);
			}
		}
		// TODO: consider reduce it a little if it makes program more robust
		radius[p1] =
				std::min(std::min(radius[p1], S), std::min(L / 3.0, D * 2.0 / 3.0));
		if (isDeGen) {
			orgradius[p1] = std::min(std::min(orgradius[p1], orgS),
					std::min(orgL / 3.0, orgD * 2.0 / 3.0));
		}
	}
	// split cliped edges
	for (int i = 0; i < acsize; i++) {
		p1 = acuteList[i];
		for (int j = 0; j < 2; j++) {
			p2 = tempAdj[p1][j];

			Vector3 newP(tempC[p1]);
			Vector3 dir(tempC[p2] - tempC[p1]);
			dir.normalize();
			newP += dir * (radius[p1] * 0.9);

			if (isDeGen) {
				Vector3 newOrgP(tempOrgC[p1]);
				Vector3 Orgdir(tempOrgC[p2] - tempOrgC[p1]);
				Orgdir.normalize();
				newOrgP += Orgdir * (orgradius[p1] * 0.9);
				splitEdge(p1, j, newP, newOrgP);
			} else {
				splitEdge(p1, j, newP);
			}
		}
	}
}

bool MingCurve::passTetGen() {
	LocalVector<Vector3> mingw_points;
	mingw_points.resize(org_n);
	for (int32_t point_i = 0; point_i < org_n; point_i++) {
		mingw_points[point_i] = tempC[point_i];
	}
	LocalVector<OutputSimply> soup = Delaunay3d::tetrahedralize(points);
	if (soup.size() == 0) {
		return false;
	}
	return true;
}

// Check whether the curve is already edge-protected
bool MingCurve::isProtected() {
	badEdge.clear();
	char **used = new char *[org_n];
	for (int i = 0; i < org_n; i++) {
		used[i] = new char[2];
		used[i][0] = 0;
		used[i][1] = 0;
	}

	Vector<Vector3> points;
	Vector3 pt;
	for (int i = 0; i < org_n; i++) {
		pt = tempC[i];
		points.push_back(Vector3(pt[0], pt[1], pt[2]));
	}

	try {
		Delaunay3d::TetrahedronSoup soup = Delaunay3d::tetrahedralize(points);
	} catch (int e) {
		cout << "MWT: tetrahedralization problem " << e << endl;
		badInput = true;
		delete[] used;
		return true;
	}

	// save badedges for latter splitting
	for (int i = 0; i < org_n; i++) {
		for (int j = 0; j < 2; j++) {
			if ((!used[i][j]) && i < tempAdj[i][j]) {
				// int * newEdge = new int[2];
				newEdge[0] = i;
				newEdge[1] = j;
				badEdge.push_back(newEdge);
			}
		}
	}

	int besize = badEdge.size();
	if (besize > BADEDGE_LIMIT) {
		// errors(ERROR_TETGEN,filename);
		badInput = true;
		delete[] used;
		return true;
	}

	delete[] used;
	if (besize > 0)
		return false;

	// save points and tris

	return true;
}

// Insert mid points on edges that are not edge-protected
void MingCurve::insertMidPointsTetgen() {
	int besize, p, q, j;
	Vector3 p1, p2, newP, orgp1, orgp2, newOrgP;
	while (!isProtected()) {
		if (badInput)
			break;
		besize = badEdge.size();
		for (int i = 0; i < besize; i++) {
			p = badEdge[i][0];
			j = badEdge[i][1];
			p1 = tempC[p];
			q = tempAdj[p][j];
			p2 = tempC[q];
			Vector3 vec = p2 - Vector3();
			newP = p1 + vec;
			newP *= 0.5;

			if (isDeGen) {
				orgp1 = tempOrgC[p];
				orgp2 = tempOrgC[q];
				Vector3 orgvec = orgp2 - Vector3();
				newOrgP = orgp1 + orgvec;
				newOrgP *= 0.5;
				splitEdge(p, j, newP, newOrgP);
			} else {
				splitEdge(p, j, newP);
			}
		}
	}
}
// For getCurveAfterEP(), to check whether the orientation changes
// if so, normals should be fliped
bool MingCurve::sameOrientation(const vector<int> &newCurve) {
	int ncsize = newCurve.size();
	int pos1 = -1, pos2 = -1, pos = 0;
	while (pos1 == -1 || pos2 == -1) {
		if (newCurve[pos] == 1)
			pos1 = pos;
		if (newCurve[pos] == 2)
			pos2 = pos;
		pos++;
	}
	return pos2 > pos1;
}
// Recover ordered curve points
bool MingCurve::getCurveAfterEP() {
	int start = 0, cur, pre, next;
	vector<int> newCurve;
	newCurve.push_back(0);
	pre = 0;
	cur = tempAdj[0][0];
	while (cur != start) {
		newCurve.push_back(cur);
		next = tempAdj[cur][0] + tempAdj[cur][1] - pre;
		pre = cur;
		cur = next;
	}
	numofpoints = newCurve.size();
	n_after = numofpoints;
	n_ratio = (float)n_after / n_before;
	points = new double[3 * numofpoints];
	Vector3 pt;
	for (int i = 0; i < numofpoints; i++) {
		pt = tempC[newCurve[i]];
		points[i * 3 + 0] = pt[0];
		points[i * 3 + 1] = pt[1];
		points[i * 3 + 2] = pt[2];
	}
	if (isDeGen) {
		DeGenPoints = new double[3 * numofpoints];
		for (int i = 0; i < numofpoints; i++) {
			pt = tempOrgC[newCurve[i]];
			DeGenPoints[i * 3 + 0] = pt[0];
			DeGenPoints[i * 3 + 1] = pt[1];
			DeGenPoints[i * 3 + 2] = pt[2];
		}
	}
	if (withNorm) {
		normals = new float[3 * numofpoints];
		Vector3 norm;
		bool sameOri = sameOrientation(newCurve);
		if (sameOri) {
			for (int i = 0; i < numofpoints; i++) {
				norm = tempNorm[newCurve[i]];
				normals[i * 3 + 0] = (float)norm[0];
				normals[i * 3 + 1] = (float)norm[1];
				normals[i * 3 + 2] = (float)norm[2];
			}
		} else {
			for (int i = 0; i < numofpoints; i++) {
				norm = tempNorm[newCurve[i]];
				normals[i * 3 + 0] = (float)-norm[0];
				normals[i * 3 + 1] = (float)-norm[1];
				normals[i * 3 + 2] = (float)-norm[2];
			}
		}
	}
	return true;
}
bool MingCurve::isDeGenCase() {
	if (org_n <= 3) {
		isDeGen = true;
		return true;
	}

	Vector3 startPt = tempC[0];
	Vector3 startNorm = tempC[1] - tempC[0];
	Vector3 stdNorm = (tempC[1] - tempC[0]) ^ (tempC[2] - tempC[0]);
	stdNorm.normalize();

	Vector3 curNorm;
	Vector3 curPt;
	bool breakout = false;
	for (int i = 3; i < org_n - 1; i++) {
		curNorm = (tempC[i] - startPt) ^ startNorm;
		curNorm.normalize();
		if (curNorm * stdNorm - 1 < plainEPS) {
			breakout = true;
			break;
		}
	}
	if (breakout) {
		isDeGen = false;
		return false;
	}
	isDeGen = true;
	return true;
}
// give some perturbation for tempC
void MingCurve::perturbPts(double ptb) {
	for (int i = 0; i < org_n; i++) {
		tempC[i].pertube(ptb * (i % 5));
	}
}
// main procedure
// If the curve is not edge-protected, protect it by 2 steps:
// 1.protectCorner; 2. insert mid points.
bool MingCurve::edgeProtect(bool isdmwt) {
	while (!passTetGen()) {
		isDeGen = true;
		perturbPts(plainPTB);
		perturbNum++;
		if (perturbNum > 500) {
			badInput = true;
			cout << "MWT: plain input, >500 perturbation" << endl;
			break;
		}
	}
	if (!isProtected()) {
		if (!badInput) {
			protectCorner();
			insertMidPointsTetgen();
		}
	}

	if (badInput) {
		return false;
	}
	getCurveAfterEP();

#if SAVE_NEWCURVE
	saveCurve(filename, points, numofpoints);
#endif

	return true;
}

void MingCurve::statistics() {
	if (!DO_EXP) {
		print_line("=================================");
		print_line(" File: \t\t" + String(filename));
		print_line(" N_ratio:\t" + String(n_ratio));
		print_line("");
		print_line(" (T) Read files:\t" + String(timeReadIn));
		print_line(" (T) Edge protect:\t" + String(timeEdgeProtect));
	} else {
		print_line(String(filename) + "\t" + String(n_ratio) + "\t" + String(timeReadIn) + "\t" + String(timeEdgeProtect) + "\t");
	}
}

#include "thirdparty/eigen/Eigen/Core"
#include <iostream>

#include "DWMT.h"
#include "MingCurve.h"
#include "refine.h"

bool Triangulate(double *boundary, int nB, float targetEdgeLength,
		double **vertices, int **faces, int *nV, int *nF) {
	int point_num = nB;

	double *newPoints;
	int newPointNum;
	double *tile_list;
	int tileNum;

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	float m_weightTri = 0;
	float m_weightEdge = 0;
	float m_weightBiTri = 1;
	float m_weightTriBd = 1;
	float m_weightWorsDih = 0;

	float weights[] = { float(m_weightTri), float(m_weightEdge),
		float(m_weightBiTri), float(m_weightTriBd),
		float(m_weightWorsDih) };

	int res = delaunayRestrictedTriangulation(boundary, point_num, &newPoints,
			&newPointNum, &tile_list, &tileNum,
			weights, false, 0, 0, V, F);

	if (res == 0) {
		// Error case
		// Initialize arrays to empties to avoid crash
		*nF = 0;
		*nV = 0;
		*vertices = new double[0];
		*faces = new int[0];
		return false;
	} else {
		// Delete unmanaged arrays (they are useless now that we got the matrices
		// anyway)
		delete[] newPoints;
		delete[] tile_list;

		Eigen::MatrixXd V_fine;
		Eigen::MatrixXi F_fine;

		std::cout << "triangulated" << std::endl;

		refine_patch(V, F, targetEdgeLength, V_fine, F_fine);
		std::cout << "refined patch" << std::endl;

		double *newVertices = new double[V_fine.size()];
		int *newFaces = new int[F_fine.size()];

		*nF = F_fine.rows();
		*nV = V_fine.rows();

		for (int i = 0; i < *nV; i++) {
			newVertices[3 * i + 0] = V_fine(i, 0);
			newVertices[3 * i + 1] = V_fine(i, 1);
			newVertices[3 * i + 2] = V_fine(i, 2);
		}

		for (int i = 0; i < *nF; i++) {
			newFaces[3 * i + 0] = F_fine(i, 0);
			newFaces[3 * i + 1] = F_fine(i, 1);
			newFaces[3 * i + 2] = F_fine(i, 2);
		}

		*vertices = newVertices;
		*faces = newFaces;

		return true;
	}
}

void CleanUp(double **vertices, int **faces) {
	delete[] *vertices;
	delete[] *faces;

	return;
}

int delaunayRestrictedTriangulation(const double *inCurve, const int inNum,
		double **outCurve, int *outPn,
		double **outFaces, int *outNum,
		float *weights, bool dosmooth, int subd,
		int laps, Eigen::MatrixXd &V,
		Eigen::MatrixXi &F) {
	const bool WITH_NORM = false;
	const bool IS_DMWT = true;
	const bool IS_MWT = false;
	const bool IS_LIEPA = false;
	const bool IS_DOT1 = false;
	const int LIMIT = 1000000;
	const int BAD_INPUT = 0;
	const int NO_SOLUTION = 0;
	const int UNKNOWN_ERROR = 0;
	const int SUCCESS = 1;

	float weightTri = weights[0];
	float weightEdge = weights[1];
	float weightBiTri = weights[2];
	float weightTriBd = weights[3];
	float weightWorst = weights[4];

	try {
		MingCurve *myCurve = new MingCurve(inCurve, inNum, LIMIT, WITH_NORM);
		if (!myCurve->edgeProtect(IS_DMWT)) {
			delete myCurve;
			print_line("MWT: (0) bad input, not able to protect curve");
			return BAD_INPUT;
		}

		if (!myCurve->smoothCurve(dosmooth)) {
			delete myCurve;
			print_line("MWT: (0) bad input, not able to smooth curve");
			return BAD_INPUT;
		}

		if (!myCurve->subdivideCurve(subd)) {
			delete myCurve;
			print_line("MWT: (0) bad input, not able to subdivide curve");
			return BAD_INPUT;
		}

		if (!myCurve->laplacianSmooth(laps)) {
			delete myCurve;
			print_line("MWT: (0) bad input, not able to apply Laplacian smoothing");
			return BAD_INPUT;
		}

		// Assuming V and F are output parameters
		myCurve->getVertices(V);
		myCurve->getFaces(F);

		delete myCurve;
	} catch (int e) {
		print_line("MWT: Unknown Error!! Exception Nr. " + String::num(e));
		return UNKNOWN_ERROR;
	}
	return SUCCESS;
}

#endif