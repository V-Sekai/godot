#include "DMWT.h"

#define RIGHT 1 //[a,b]
#define LEFT 0 //[-infinity,a)&(b,+infinity]
#define triangle(t, i) tris[t * 3 + i]
#define point(v) Vector3(points[v * 3], points[v * 3 + 1], points[v * 3 + 2])
#define leftEdgeInd(ei, i) edgeInfoList[ei]->leftEdgeInd[i]
#define leftTris(ei, i) edgeInfoList[ei]->leftTris[i]
#define rightEdgeInd(ei, i) edgeInfoList[ei]->rightEdgeInd[i]
#define rightTris(ei, i) edgeInfoList[ei]->rightTris[i]

using namespace std;

void PolygonTriangulation::init(int ptn, double *pts, double *deGenPts, float *norms, bool isdegen) {
	init_basics();

	isDeGen = isdegen;
	filename = (char *)"DMWT_NULL.curve";
	numofpoints = ptn;
	numoftris = 0;
	points = new double[3 * numofpoints];
	tiling = new int[numofpoints - 2];

	// read point set data
	for (int i = 0; i < numofpoints; i++) {
		points[i * 3] = pts[i * 3];
		points[i * 3 + 1] = pts[i * 3 + 1];
		points[i * 3 + 2] = pts[i * 3 + 2];
	}

	if (isDeGen) {
		deGenPoints = new double[3 * numofpoints];
		for (int i = 0; i < numofpoints; i++) {
			deGenPoints[i * 3] = deGenPts[i * 3];
			deGenPoints[i * 3 + 1] = deGenPts[i * 3 + 1];
			deGenPoints[i * 3 + 2] = deGenPts[i * 3 + 2];
		}
	}

	if (norms != nullptr) {
		withNormal = true;
		normals = new float[3 * numofpoints];
		for (int i = 0; i < numofpoints; i++) {
			normals[i * 3] = norms[i * 3];
			normals[i * 3 + 1] = norms[i * 3 + 1];
			normals[i * 3 + 2] = norms[i * 3 + 2];
		}
	}
}

PolygonTriangulation::PolygonTriangulation() {
	EXPSTOP = false;
	hasIntersect = false;
	hasIntersect2 = false;
	useWorstDihedral = false;
	withNormal = false;
	useBiTri = true;
	dot = 2;
	timePreprocess = 0.0;
	timeMWT = 0.0;
	round = 0;
	numoftilingtris = 0;
}

PolygonTriangulation::PolygonTriangulation(int ptn, double *pts, double *deGenPts, bool isdegen) {
	init(ptn, pts, deGenPts, nullptr, isdegen);
}

PolygonTriangulation::PolygonTriangulation(int ptn, double *pts, double *deGenPts, float *norms, bool isdegen) {
	init(ptn, pts, deGenPts, norms, isdegen);
}

PolygonTriangulation::~PolygonTriangulation() {
	// tris and points will be destroyed by tetgen
	// delete [] filename;
	if (!EXPSTOP) {
		delete[] tiling; // delete [] tris; //delete [] points;
		if (withNormal)
			delete[] normals;
		if (!isDeGen) {
			for (int i = 0; i < numofedges; i++)
				delete edgeInfoList[i];
			for (int i = 0; i < numoftris; i++)
				delete triangleInfoList[i];
		}
	}
}

void PolygonTriangulation::init_basics() {
	EXPSTOP = false;
	hasIntersect = false;
	hasIntersect2 = false;
	DMWT_LIMIT = 100000;
	withNormal = false;
	useBiTri = true;
	dot = 2;
	useWorstDihedral = false;
	timePreprocess = 0.0;
	timeMWT = 0.0;
	round = 0;
	numoftilingtris = 0;
	isDeGen = false;
}

//==================================Weight Functions============================//

void PolygonTriangulation::set_weights(float wtri, float wedge, float wbitri, float wtribd, float wwst) {
	weightTri = wtri;
	weightEdge = wedge;
	weightBiTri = wbitri;
	weightTriBd = wtribd;
	if ((weightBiTri == 0.0f) && (weightTriBd == 0.0f) && (wwst == 0.0f))
		useBiTri = false;
	if (wwst != 0.0f)
		useWorstDihedral = true;
}

float PolygonTriangulation::cost_triangle(float measure) {
	return weightTri * measure;
}
float PolygonTriangulation::cost_edge(float measure) {
	return weightEdge * measure;
}
float PolygonTriangulation::cost_bi_triangle(float measure) {
	return weightBiTri * measure;
}
float PolygonTriangulation::cost_triangle_bd(float measure) {
	return weightTriBd * measure;
}

float PolygonTriangulation::measure_edge(int v1, int v2) {
	Vector3 p1 = point(v1);
	Vector3 p2 = point(v2);
	return (float)(p2 - p1).length();
}

float PolygonTriangulation::measure_triangle(int v1, int v2, int v3) {
	Vector3 p1 = point(v1);
	Vector3 p2 = point(v2);
	Vector3 p3 = point(v3);
	return (p2 - p1).cross(p3 - p2).length() / 2.0f;
}

float PolygonTriangulation::measure_bi_triangle(int v1, int v2, int p, int q) {
	Vector3 p1 = point(v1);
	Vector3 p2 = point(v2);
	Vector3 pp = point(p);
	Vector3 pq = point(q);
	Vector3 n1 = (p2 - pp).cross(p1 - p2);
	n1.normalize();
	Vector3 n2 = (p2 - p1).cross(pq - p2);
	n2.normalize();
	float cosvalue = n1.dot(n2);
	cosvalue = cosvalue < 1.0 ? cosvalue : 1.0f - FLT_EPSILON;
	cosvalue = cosvalue > -1.0 ? cosvalue : -1.0f + FLT_EPSILON;
	return acos(cosvalue);
}

float PolygonTriangulation::measure_triangle_bd(int v1, int v2, int v3, int ni) {
	if (!withNormal)
		return 0.0;
	Vector3 p1 = point(v1);
	Vector3 p2 = point(v2);
	Vector3 p3 = point(v3);
	Vector3 n = Vector3(normals[ni * 3], normals[ni * 3 + 1], normals[ni * 3 + 2]);
	Vector3 nt = (p2 - p1).cross(p3 - p2);
	nt.normalize();
	float cosvalue = (float)nt.dot(n);
	cosvalue = cosvalue < 1.0 ? cosvalue : 1.0f - FLT_EPSILON;
	cosvalue = cosvalue > -1.0 ? cosvalue : -1.0f + FLT_EPSILON;
	return acos(cosvalue);
}

//==================================Tiling Functions============================//

void PolygonTriangulation::preprocess() {
	gen_triangle_candidates();
	build_list();
}

void PolygonTriangulation::clear_tiling() {
	for (int i = 0; i < numoftris; i++) {
		triangleInfoList[i]->optCost[0] = FLT_MIN;
		triangleInfoList[i]->optCost[1] = FLT_MIN;
		triangleInfoList[i]->optCost[2] = FLT_MIN;
	}
	hasIntersect = false;
	hasIntersect2 = false; // intersect
	useBiTri = true;
	dot = 1;
	timeMWT = 0.0;
	timeTotal = 0.0;
	numoftilingtris = 0;
}

bool PolygonTriangulation::start() {
	round++;
	numoftilingtris = 0;
	// clearTiling();
	int optTile;
	tile_segment(startEdge, 0, -1, optimalCost, optTile);
	build_tiling(startEdge, 0, optTile);

	if (numoftilingtris != numofpoints - 2) {
		print_line("NOTICE: No solution!");
		return false;
	}
	return true;
}

void PolygonTriangulation::build_tiling(int eind, char side, int ti) {
	EdgeInfo *einfo;
	TriangleInfo *tinfo;
	int tind, ei;
	int *tlist;
	char newside;
	einfo = edgeInfoList[eind];
	// 1. hit boudary, return assert(ti>=0 <=?)
	if (ti == -1)
		return;
	if (side == RIGHT) {
		tlist = einfo->rightTris;
		ei = einfo->rightEdgeInd[ti];
	} else {
		tlist = einfo->leftTris;
		ei = einfo->leftEdgeInd[ti];
	}
	tind = tlist[ti];
	tinfo = triangleInfoList[tind];
	// tiling.push_back(tind);
	tiling[numoftilingtris] = tind;
	numoftilingtris++;
	for (int ej = 0; ej < 3; ej++) {
		if (ej == ei)
			continue;
		if (tinfo->optCost[ej] == FLT_MIN) {
			cout << "Error in building Tiling! Tri:" << tind << " Edgei:" << ej << endl;
			return;
		}
		newside = ej == 2 ? 1 : 0;
		build_tiling(tinfo->edgeIndex[ej], 1 - newside, tinfo->optTile[ej]);
	}
}

//==================================List Related Functions============================//

char PolygonTriangulation::get_side(int v1, int v2, int v3) {
	return (v3 < v2 && v3 > v1);
}
char PolygonTriangulation::getSide(int i) {
	if (i == 2)
		return 1;
	return 0;
}

/// return # of edges
int PolygonTriangulation::scan_triangles_once() {
	int edgenum = 0;
	char side;
	int min, max, sum = 0, mid, v, v1, v2;
	for (int t = 0; t < numoftris; t++) {
		// sort vertices
		min = INT_MAX;
		max = -1;
		sum = 0;
		for (int i = 0; i < 3; i++) {
			v = triangle(t, i);
			min = min < v ? min : v;
			max = max > v ? max : v;
			sum += v;
		}
		mid = sum - min - max;
		triangle(t, 0) = min;
		triangle(t, 1) = mid;
		triangle(t, 2) = max;

		for (int i = 0; i < 3; i++) {
			if (i == 2) {
				v1 = triangle(t, 0);
				v2 = triangle(t, 2);
			} else {
				v1 = triangle(t, i);
				v2 = triangle(t, i + 1);
			}
			side = get_side(v1, v2, sum - v1 - v2);
			if (ehash[v1][v2] == -1) {
				ehash[v1][v2] = edgenum;
				edgenum++;
			}
			if (side == 0)
				ehashLeft[v1][v2] += 1;
			else
				ehashRight[v1][v2] += 1;
		}
	}
	return edgenum;
}

void PolygonTriangulation::build_list() {
	// initialize ehash
	ehash = new int *[numofpoints];
	ehashLeft = new int *[numofpoints];
	ehashRight = new int *[numofpoints];
	for (int i = 0; i < numofpoints; i++) {
		ehash[i] = new int[numofpoints];
		ehashLeft[i] = new int[numofpoints];
		ehashRight[i] = new int[numofpoints];
		for (int j = 0; j < numofpoints; j++) {
			ehash[i][j] = -1;
			ehashLeft[i][j] = 0;
			ehashRight[i][j] = 0;
		}
	}
	triangleInfoList = new TriangleInfo *[numoftris];
	for (int i = 0; i < numoftris; i++) {
		triangleInfoList[i] = new TriangleInfo();
	}
	// scan triangle list once to assign index of edges
	numofedges = scan_triangles_once();
	// create edgeInfoList & triangleInfoList
	startEdge = ehash[0][1];
	edgeInfoList = new EdgeInfo *[numofedges];
	for (int i = 0; i < numofedges; i++) {
		edgeInfoList[i] = new EdgeInfo();
	}

	int v1;
	int v2;
	int ei = -1;
	int left;
	int right;
	char newside;
	// scan all triangles, setup triangleInfoList and most of edgeInfoList except BiTri information
	// initialize all edgeInfoList and set its left/rightsize, left/rightEdgeInd and left/rightTris
	for (int t = 0; t < numoftris; t++) {
		for (int i = 0; i < 3; i++) {
			if (i == 2) {
				v1 = triangle(t, 0);
				v2 = triangle(t, 2);
			} else {
				v1 = triangle(t, i);
				v2 = triangle(t, i + 1);
			}
			ei = ehash[v1][v2];
			// after process one edge, set ehashLeft[v1][v2]&ehashRight[v1][v2] to
			// the next slot for inserting a triangle
			if (edgeInfoList[ei]->leftsize == -1) {
				left = ehashLeft[v1][v2];
				right = ehashRight[v1][v2];
				edgeInfoList[ei]->v1 = v1;
				edgeInfoList[ei]->v2 = v2;
				edgeInfoList[ei]->leftsize = left;
				edgeInfoList[ei]->rightsize = right;
				edgeInfoList[ei]->leftEdgeInd = new char[left];
				edgeInfoList[ei]->leftTris = new int[left];
				edgeInfoList[ei]->rightEdgeInd = new char[right];
				edgeInfoList[ei]->rightTris = new int[right];
				ehashLeft[v1][v2] = 0;
				ehashRight[v1][v2] = 0;
			}
			left = ehashLeft[v1][v2];
			right = ehashRight[v1][v2];
			newside = i == 2 ? 1 : 0;
			if (newside == 0) {
				leftTris(ei, left) = t;
				leftEdgeInd(ei, left) = i;
				triangleInfoList[t]->triIndex[i] = left;
				ehashLeft[v1][v2]++;
			} else {
				rightTris(ei, right) = t;
				rightEdgeInd(ei, right) = i;
				triangleInfoList[t]->triIndex[i] = right;
				ehashRight[v1][v2]++;
			}
			triangleInfoList[t]->edgeIndex[i] = ei;
		}
	}

	for (int i = 0; i < numofpoints; i++) {
		delete[] ehash[i];
		delete[] ehashLeft[i];
		delete[] ehashRight[i];
	}
	delete[] ehashLeft;
	delete[] ehashRight;
	delete[] ehash;
}

//==================================TetGen Functions============================//

void PolygonTriangulation::gen_triangle_candidates() {
	in.resize(numofpoints);
	int new_num_of_points = in.size() / 3;
	memcpy(in.ptrw(), points, new_num_of_points * sizeof(Vector3));
	Vector<Vector3> points_array;
	points_array.resize(new_num_of_points);

	for (int i = 0; i < new_num_of_points; ++i) {
		points_array.write[i] = Vector3(points[i * 3 + 0], points[i * 3 + 1], points[i * 3 + 2]);
	}

	out = Delaunay3D::tetrahedralize(points_array);
	PackedInt32Array triangles;

	for (int i = 0; i < out.size(); i++) {
		static const int face_order[4][3] = {
			{ 0, 1, 2 },
			{ 0, 2, 3 },
			{ 0, 1, 3 },
			{ 1, 2, 3 }
		};

		for (int j = 0; j < 4; j++) {
			// Add each triangle to the array
			triangles.push_back(out[i].points[face_order[j][0]]);
			triangles.push_back(out[i].points[face_order[j][1]]);
			triangles.push_back(out[i].points[face_order[j][2]]);
		}
	}

	numoftris = triangles.size() / 3;

	// Allocate memory for tris and copy the elements from triangles
	tris = new int[numoftris * 3];
	for (int i = 0; i < numoftris * 3; i++) {
		tris[i] = triangles[i];
	}

	// TODO: Memory leak

	// output .node & .face file
	// out.save_nodes(filename);
#if (SAVE_FACE == 1)
	out.save_faces(filename);
#endif
}
//==================================IO Read Functions============================//

//==================================IO Write Functions============================//
void PolygonTriangulation::set_point_limit(int limit) {
	DMWT_LIMIT = limit;
	if (numofpoints > DMWT_LIMIT) {
		EXPSTOP = true;
	}
}
void PolygonTriangulation::set_round(int r) {
	round = r;
}

//==================================Evaluation Functions============================//
// intersect
bool PolygonTriangulation::triangle_share_edge(int trii, int trij) {
	int num = 0;
	int ipind[3];
	int jpind[3];
	for (int k = 0; k < 3; k++) {
		ipind[k] = tris[trii * 3 + k];
		jpind[k] = tris[trij * 3 + k];
	}
	for (int ii = 0; ii < 3; ii++) {
		for (int jj = 0; jj < 3; jj++) {
			if (ipind[ii] == jpind[jj])
				num++;
		}
	}
	if (num == 2)
		return true;
	return false;
}

void PolygonTriangulation::save_tiling_object(char *tilefile, const double *finalPoints) {
	int n = numoftilingtris;
	std::ofstream writer(tilefile, std::ofstream::out);
	if (!writer.good())
		exit(1);
	writer << "# OBJ File Generated by MMWT\n";
	writer << "# Vertices: " << numofpoints << "\n";
	writer << "# Faces: " << n << "\n";
	// write vertices
	for (int i = 0; i < numofpoints; i++) {
		writer << "v " << finalPoints[i * 3] << " " << finalPoints[i * 3 + 1] << " " << finalPoints[i * 3 + 2] << "\n";
	}
	// write faces
	int t = 0;
	for (int i = 0; i < n; i++) {
		t = tiling[i];
		writer << "f " << triangle(t, 0) + 1 << " " << triangle(t, 1) + 1 << " " << triangle(t, 2) + 1 << "\n";
	}
	writer.close();
}
void PolygonTriangulation::save_mesh_obj(char *tilefile, int nT, const double *mesh) {
	std::ofstream writer(tilefile, std::ofstream::out);
	if (!writer.good())
		exit(1);
	writer << "# OBJ File Generated by MMWT\n";
	writer << "# Vertices: " << nT * 3 << "\n";
	writer << "# Faces: " << nT << "\n";
	// write vertices
	for (int i = 0; i < nT; i++) {
		for (int j = 0; j < 3; j++) {
			writer << "v " << mesh[i * 9 + j * 3] << " " << mesh[i * 9 + j * 3 + 1] << " " << mesh[i * 9 + j * 3 + 2] << "\n";
		}
	}
	// write faces
	for (int i = 0; i < nT; i++) {
		writer << "f " << i * 3 + 1 << " " << i * 3 + 2 << " " << i * 3 + 3 << "\n";
	}
	writer.close();
}

//================ for cycle breaking project=====================
void PolygonTriangulation::get_result(double **outFaces, int *outNum, double **outPoints, float **outNorms, int *outPn, bool dosmooth,
		int subd, int laps) {
	double *finalPoints;
	if (isDeGen) {
		finalPoints = deGenPoints;
	} else {
		finalPoints = points;
	}

	double *outTris;
	{
		*outNum = numoftilingtris;
		outTris = new double[numoftilingtris * 9];
		for (int i = 0; i < numoftilingtris; i++) {
			int triId = tiling[i];
			for (int j = 0; j < 3; j++) {
				int pointID = tris[triId * 3 + j];
				for (int k = 0; k < 3; k++) {
					outTris[i * 9 + j * 3 + k] = finalPoints[pointID * 3 + k];
				}
			}
		}
		*outFaces = outTris;

#if (SAVEOBJ == 1)
		saveMeshObj("nosmth.obj", numoftilingtris, outTris);
#endif
	}

	*outPn = numofpoints;
	double *outPs = new double[numofpoints * 3];
	for (int i = 0; i < numofpoints * 3; i++) {
		outPs[i] = points[i];
	}
	*outPoints = outPs;

	if (withNormal) {
		float *outNs = new float[numofpoints * 3];
		for (int i = 0; i < numofpoints * 3; i++) {
			outNs[i] = normals[i];
		}
		*outNorms = outNs;
	}
}

void PolygonTriangulation::set_dot(bool isdot1) { dot = isdot1 ? 1 : 2; }

float PolygonTriangulation::get_size() {
	if (dot == 1) {
		// DOT1 & PolygonTriangulation: X-TIL+OPT=EIL,TIL_opt,Vlist,Tlist,Nlist,TiList
		int trivalLists = numoftris * 3 * sizeof(int) // tris
				+ numofpoints * 3 * sizeof(double) // points
				+ numofpoints * 3 * sizeof(float) // normals
				+ (numofpoints - 2) * sizeof(int); // tiling
		int EIL = 0;
		for (int i = 0; i < numofedges; i++) { // edgeInfoList
			EIL += edgeInfoList[i]->get_size();
		}
		int OPT = 2 * numofedges * triangleInfoList[0]->getOptSize() / 3; // OptSize
		return (float)(trivalLists + EIL + OPT) / 1048576;

	} else {
		// DOT2 & PolygonTriangulation: X=EIL,TIL,Vlist,Tlist,Nlist,TiList
		int trivalLists = numoftris * 3 * sizeof(int) // tris
				+ numofpoints * 3 * sizeof(double) // points
				+ numofpoints * 3 * sizeof(float) // normals
				+ (numofpoints - 2) * sizeof(int); // tiling
		int EIL = 0, TIL = 0;
		for (int i = 0; i < numofedges; i++) { // edgeInfoList
			EIL += edgeInfoList[i]->get_size();
		}
		TIL = numoftris * triangleInfoList[0]->getSize(); // triangleInfoList
		return (float)(trivalLists + EIL + TIL) / 1048576;
	}
}

void PolygonTriangulation::statistics() {
	timeTotal = timePreprocess + timeMWT + timeTetgen;
	const bool DO_EXP = true;
	if (!DO_EXP) {
		if (dot == 1)
			cout << " [DOT1 PolygonTriangulation]" << endl;
		else
			cout << " [DOT2 PolygonTriangulation]" << endl;
		cout << "---------------------------------" << endl;
		cout << " File: \t\t" << filename << endl;
		cout << " Vertex:\t" << numofpoints << endl;
		cout << " Edge:\t\t" << numofedges << endl;
		cout << " Triangle:\t" << numoftris << endl;
		cout << " Weights: \t" << weightTri << "," << weightEdge << ","
			 << weightBiTri << "," << weightTriBd << endl;
		cout << "" << endl;
		// cout<<" Read files:\t"<<timeReadIn<<endl;
		cout << " (T) Call TetGen:\t" << timeTetgen << endl;
		cout << " (T) Preprocess:\t" << timePreprocess << endl;
		cout << " (T) MWT & Tiling:\t" << timeMWT << endl;
		cout << "" << endl;
		cout << " Total time:\t" << timeTotal << "sec" << endl;
		cout << " Total space:\t" << get_size() << "MB" << endl;
		cout << " Optimal Cost:\t" << optimalCost << endl;
		cout << endl;
		cout << " Intersection:\t" << intsTriInd[0] << "," << intsTriInd[1] << endl;
	} else {
		if (dot == 1)
			cout << "DOT1+PolygonTriangulation==\t";
		else
			cout << "DOT2+PolygonTriangulation==\t";
		cout << numofpoints << "\t" << numofedges << "\t" << numoftris << "\t"
			 << "w" << weightTri << weightEdge << weightBiTri << weightTriBd << "\t"
			 << timeTetgen << "\t" << timePreprocess << "\t" << timeMWT << "\t"
			 << timeTotal << "\t" << optimalCost << "\t" << get_size() << "\t"
			 << hasIntersect << "\t" << hasIntersect2 << "\t" << intsTriInd[0]
			 << "\t" << intsTriInd[1] << "\t" << filename << endl;
	}
}

/*
This main function tiles a segment of the input curve on the side of the edge
whose index is eind, with a "proceeding" triangle whose index in the edge's
triangle list is ti. If ti is -1, that means eind is a boundary edge (i.e., at
the beginning of the algorithm). The function returns the optimal tiling cost,
and the index of the first triangle in the tiling (adjacent to eind) in the
triangle list of eind (-1 if eind is on the boundary).
*/
bool PolygonTriangulation::tile_segment(int eind, char side, int ti, float &thisCost,
		int &thisTile) {
	float optCost = FLT_MAX, subCost = 0.0, subCostSum;
	int tnum, tind, tindofti = -1, ei, ejind, ejtnum;
	int optTile = -1, subTile = -1;
	int ev1, ev2, tv3, v3;
	int *tlist;
	int *ejtlist;
	char *elist;
	char newside;
	char *ejelist;
	thisCost = optCost;
	thisTile = optTile;
	EdgeInfo *einfo;
	TriangleInfo *tinfo;
	EdgeInfo *ejinfo;

	bool isboard = false, hasSolution = true;
	float costtribd, costbitri;

	// for worstbitri
	float worstbitri = -1; //,bitriAngle;
	// int ejv1,ejv2,ejv3,ejtv3,ejtind;

	// ev1, ev2 are two end vertices of the edge
	einfo = edgeInfoList[eind];
	ev1 = einfo->v1;
	ev2 = einfo->v2;
	// get the list of abutting tris on the "side" of current edge
	if (side == RIGHT) {
		tlist = einfo->rightTris;
		tnum = einfo->rightsize;
		elist = einfo->rightEdgeInd;
		if (ti >= 0) { // not the beginning of algorithm
			tindofti = einfo->leftTris[ti];
			tv3 = tris[tindofti * 3] + tris[tindofti * 3 + 1] +
					tris[tindofti * 3 + 2] - ev1 - ev2;
		}
		isboard = einfo->v2 - einfo->v1 == 1; // eg. (v1,v2)=(3,4), null in (3,4)
		if ((tnum == 0) && !isboard)
			hasSolution = false; // no candidate triangle found
	} else {
		tlist = einfo->leftTris;
		tnum = einfo->leftsize;
		elist = einfo->leftEdgeInd;
		if (ti >= 0) {
			tindofti = einfo->rightTris[ti];
			tv3 = tris[tindofti * 3] + tris[tindofti * 3 + 1] +
					tris[tindofti * 3 + 2] - ev1 - ev2;
		}
		isboard = einfo->v2 - einfo->v1 ==
				numofpoints - 1; // eg. (v1,v2)=(n-1,0), null in (0,0)&(n-1,n-1)
		if ((tnum == 0) && !isboard)
			hasSolution = false;
	}
	if (!hasSolution) {
		return false;
	}
	if (isboard) {
		costtribd = 0.0f;
		if (tindofti != -1) {
			if (side == RIGHT)
				costtribd = cost_triangle_bd(measure_triangle_bd(ev1, ev2, tv3, ev1));
			else
				costtribd = cost_triangle_bd(measure_triangle_bd(ev2, ev1, tv3, ev2));
		} else {
			// cout<<"Error: Left and Right are both empty for edge
			// <v1,v2>=<"<<ev1<<","<<ev2<<">."<<endl;
		}
		if (useWorstDihedral) {
			optCost = 0.0;
			worstbitri = 0.0; // worst angle = 0 for a board edge
		} else {
			optCost = cost_edge(measure_edge(ev1, ev2)) + costtribd;
		}
	} else {
		for (int t = 0; t < tnum; t++) {
			tind = tlist[t];
			ei = elist[t];
			tinfo = triangleInfoList[tind];
			v3 = tris[tind * 3] + tris[tind * 3 + 1] + tris[tind * 3 + 2] - ev1 - ev2;
			if (useWorstDihedral) {
				worstbitri = -1;
			} else {
				subCostSum =
						cost_edge(measure_edge(ev1, ev2)) + cost_triangle(measure_triangle(ev1, ev2, v3));
			}
			if (ti == -1) {
				if (useWorstDihedral) {
					worstbitri = 0.0;
				} else {
					if (weightTriBd != 0.0f) {
						if (side == LEFT)
							costtribd = cost_triangle_bd(measure_triangle_bd(ev1, ev2, v3, ev1));
						else
							costtribd = cost_triangle_bd(measure_triangle_bd(ev2, ev1, v3, ev2));
						subCostSum += costtribd;
					}
				}
			} else {
				// dot1 doesn't consider the bi-tri property
				if (weightBiTri != 0.0f && dot == 2) {
					costbitri = cost_bi_triangle(measure_bi_triangle(ev1, ev2, v3, tv3));
					if (useWorstDihedral) {
						worstbitri = costbitri > worstbitri ? costbitri : worstbitri;
					} else {
						subCostSum += costbitri;
					}
				}
			}
			for (int ej = 0; ej < 3; ej++) {
				if (ej == ei)
					continue;
				// need to compute tiling
				if (tinfo->optCost[ej] == FLT_MIN) {
					// vertices are stored in order, e.g. (0,3,7), and
					// edge0=(0,3),edge1=(3,7),edge2=(0,7), the third vertex is on the
					// (0,0,1) side for each edge
					newside = ej == 2 ? 1 : 0;
					tile_segment(tinfo->edgeIndex[ej], 1 - newside, tinfo->triIndex[ej],
							subCost, subTile);
					tinfo->optCost[ej] = subCost;
					tinfo->optTile[ej] = subTile;
					// push optCost&Tile to all triangles that use this edge if useBiTri
					// is true
					if (dot == 1) {
						// if(!useBiTri){
						ejind = tinfo->edgeIndex[ej];
						ejinfo = edgeInfoList[ejind];
						if (newside == RIGHT) {
							ejtnum = ejinfo->rightsize;
							ejtlist = ejinfo->rightTris;
							ejelist = ejinfo->rightEdgeInd;
						} else {
							ejtnum = ejinfo->leftsize;
							ejtlist = ejinfo->leftTris;
							ejelist = ejinfo->leftEdgeInd;
						}
						for (int i = 0; i < ejtnum; i++) {
							TriangleInfo *ejtinfo = triangleInfoList[ejtlist[i]];
							int index = static_cast<int>(ejelist[i]);
							ejtinfo->optCost[index] = subCost;
							ejtinfo->optTile[index] = subTile;
						}
					}
				} // end for computing new subproblem
				// if use worst-bitri, and algo is dot1, need to consider 2 more bi-tri
				// angles
				if (useWorstDihedral) {
					worstbitri =
							tinfo->optCost[ej] > worstbitri ? tinfo->optCost[ej] : worstbitri;
				} else {
					subCostSum += tinfo->optCost[ej];
				}
			}
			if (useWorstDihedral) {
				if (worstbitri < optCost) {
					optCost = worstbitri;
					optTile = t;
				}
			} else {
				if (subCostSum < optCost) {
					optCost = subCostSum;
					optTile = t;
				}
			}
		}
	}
	thisCost = optCost;
	thisTile = optTile;
	return true;
}

Ref<PolygonTriangulation> PolygonTriangulation::_create() {
	return Ref<PolygonTriangulation>(new PolygonTriangulation());
}

Ref<PolygonTriangulation> PolygonTriangulation::_create_with_degenerates(int ptn, double *pts, double *deGenPts, bool isdegen) {
	return Ref<PolygonTriangulation>(new PolygonTriangulation(ptn, pts, deGenPts, isdegen));
}

Ref<PolygonTriangulation> PolygonTriangulation::_create_with_normals(int ptn, double *pts, double *deGenPts, float *norms, bool isdegen) {
	return Ref<PolygonTriangulation>(new PolygonTriangulation(ptn, pts, deGenPts, norms, isdegen));
}

void PolygonTriangulation::_bind_methods() {
	// ClassDB::bind_static_method("PolygonTriangulation", D_METHOD("_create"), &DirAccess::_create);
	// ClassDB::bind_static_method("PolygonTriangulation", D_METHOD("_create_with_degenerates", "points", "degenerate_points", "is_degenerate"), &DirAccess::_create_with_degenerates);
	// ClassDB::bind_static_method("PolygonTriangulation", D_METHOD("_create_with_normals", "points", "degenerate_points", "normals", "is_degenerate"), &DirAccess::_create_with_normals);

	ClassDB::bind_method(D_METHOD("preprocess"), &PolygonTriangulation::preprocess);
	ClassDB::bind_method(D_METHOD("start"), &PolygonTriangulation::start);
	ClassDB::bind_method(D_METHOD("set_weights", "wtri", "wedge", "wbitri", "wtribd", "wwst"), &PolygonTriangulation::set_weights);
	ClassDB::bind_method(D_METHOD("statistics"), &PolygonTriangulation::statistics);
	ClassDB::bind_method(D_METHOD("set_round", "r"), &PolygonTriangulation::set_round);
	ClassDB::bind_method(D_METHOD("set_dot", "isdot1"), &PolygonTriangulation::set_dot);
	ClassDB::bind_method(D_METHOD("clear_tiling"), &PolygonTriangulation::clear_tiling);
	ClassDB::bind_method(D_METHOD("set_point_limit", "limit"), &PolygonTriangulation::set_point_limit);
	// ClassDB::bind_method(D_METHOD("get_result", "outFaces", "outNum", "outPoints", "outNorms", "outPn", "dosmooth", "subd", "laps"), &PolygonTriangulation::get_result);
	// ClassDB::bind_method(D_METHOD("get_result_as_matrices", "matrix1", "matrix2", "matrix3"), &PolygonTriangulation::get_result_as_matrices);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "EXPSTOP"), "", "get_expstop");
}

bool PolygonTriangulation::get_expstop() const {
	return EXPSTOP;
}
