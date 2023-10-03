#ifndef _DMWT_H_
#define _DMWT_H_

#include "EdgeInfo.h"
#include "TriangleInfo.h"

#include "core/math/delaunay_3d.h"
#include "core/object/ref_counted.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

/**
 * The PolygonTriangulation class is used for triangulating multiple polygons.
 */
class PolygonTriangulation : public RefCounted {
	GDCLASS(PolygonTriangulation, RefCounted);

protected:
	static void _bind_methods();

public:
	static Ref<PolygonTriangulation> _create();
	static Ref<PolygonTriangulation> _create_with_degenerates(int ptn, double *pts, double *deGenPts, bool isdegen);
	static Ref<PolygonTriangulation> _create_with_normals(int ptn, double *pts, double *deGenPts, float *norms, bool isdegen);

	PolygonTriangulation();
	PolygonTriangulation(int ptn, double *pts, double *deGenPts, bool isdegen);
	PolygonTriangulation(int ptn, double *pts, double *deGenPts, float *norms, bool isdegen);
	~PolygonTriangulation();
	void preprocess();
	bool start();
	void set_weights(float wtri, float wedge, float wbitri, float wtribd,
			float wwst);
	void statistics();
	void set_round(int r);
	float optimalCost;
	void set_dot(bool isdot1);
	void clear_tiling();
	void set_point_limit(int limit);

	void get_result(double **outFaces, int *outNum, double **outPoints,
			float **outNorms, int *outPn, bool dosmooth, int subd,
			int laps);

	bool EXPSTOP = false;
	bool get_expstop();

protected:
	char *filename = nullptr;
	Vector<Vector3> in;
	Vector<Delaunay3D::OutputSimplex> out;
	int round = 0;
	int startEdge = 0;
	bool withNormal = false;
	bool useBiTri = false;
	bool hasIntersect = false; // Intersect
	bool hasIntersect2 = false;
	char dot = '\0';
	int DMWT_LIMIT = 0;

	int numofpoints = 0;
	int numoftris = 0;
	int numofedges = 0;
	int numofnormals = 0;
	int numoftilingtris = 0;
	int *tris = nullptr;
	double *points = nullptr;
	double *deGenPoints = nullptr;
	float *normals = nullptr;
	EdgeInfo **edgeInfoList = nullptr;
	TriangleInfo **triangleInfoList = nullptr;
	int *tiling = nullptr;

	float weightTri = 0.0f;
	float weightEdge = 0.0f;
	float weightBiTri = 0.0f;
	float weightTriBd = 0.0f;
	bool useWorstDihedral = false;

	int **ehash = nullptr;
	int **ehashLeft = nullptr;
	int **ehashRight = nullptr;

	int intsTriInd[2] = { 0, 0 };

	bool get_expstop() const;
	void init(int ptn, double *pts, double *deGenPts, float *norms, bool isdegen);
	int scan_triangles_once();
	char get_side(int v1, int v2, int v3);

	float measure_edge(int v1, int v2);
	float measure_triangle(int v1, int v2, int v3);
	float measure_bi_triangle(int v1, int v2, int p, int q);
	float measure_triangle_bd(int v1, int v2, int v3, int ni);
	float cost_triangle(float measure);
	float cost_edge(float measure);
	float cost_bi_triangle(float measure);
	float cost_triangle_bd(float measure);

	bool tile_segment(int eind, char side, int ti, float &thiscost, int &thistile);
	void build_tiling(int eind, char side, int ti);
	void build_list();
	void gen_triangle_candidates();
	char getSide(int i);

	void init_basics();
	bool triangle_share_edge(int trii, int trij);

	// ------------------- for cycle project -----------------//
	bool isDeGen = false; // degenerated cases: plane
	void save_tiling_object(char *tilefile, const double *finalPoints);
	void save_mesh_obj(char *tilefile, int nT, const double *mesh);

	//-------------evaluations--------------//
	float timeReadIn;
	float timePreprocess;
	float timeMWT;
	float timeTotal;
	float timeTetgen;
	float get_size();
};

#endif