///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////

#ifndef DEM_BONES_EXT
#define DEM_BONES_EXT

///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////

#include "core/math/quaternion.h"
#ifndef DEM_BONES_DEM_BONES
#define DEM_BONES_DEM_BONES

#include "ConvexLS.h"
#include "thirdparty/eigen/Eigen/Eigen/Dense"
#include "thirdparty/eigen/Eigen/Eigen/Sparse"
#include "thirdparty/eigen/Eigen/Eigen/StdVector"
#include <algorithm>
#include <queue>
#include <set>

#ifndef DEM_BONES_MAT_BLOCKS
#include "MatBlocks.h"
#define DEM_BONES_DEM_BONES_MAT_BLOCKS_UNDEFINED
#endif

namespace Dem {
/**
 * @mainpage Overview
 * Main elements:
 *   - @ref DemBones : base class with the core solver using relative bone
 *     transformations DemBones::bone_transform_mat
 *   - @ref DemBonesExt : extended class to handle hierarchical skeleton with
 *     local rotations/translations and bind matrices
 *   - DemBones/MatBlocks.h: macros to access sub-blocks of packing
 *     transformation/position matrices for convenience
 *
 * Include DemBones/DemBonesExt.h (or DemBones/DemBones.h) with optional
 * DemBones/MatBlocks.h then follow these steps to use the library:
 *   1. Load required data in the base class:
 *       - Rest shapes: DemBones::rest_pose_geometry, DemBones::fv, DemBones::num_vertices
 *       - Sequence: DemBones::vertex, DemBones::num_total_frames, DemBones::frame_start_index,
 *         DemBones::frame_subject_id, DemBones::num_subjects
 *       - Number of bones DemBones::num_bones
 *   2. Load optional data in the base class:
 *       - Skinning weights DemBones::skinning_weights and weights soft-lock
 *         DemBones::lock_weight
 *       - Bone transformations DemBones::bone_transform_mat and bones hard-lock
 *         DemBones::lock_mat
 *   3. [@c optional] Set parameters in the base class:
 *       - DemBones::nIters
 *       - DemBones::nInitIters
 *       - DemBones::nTransIters, DemBones::transAffine,
 *         DemBones::transAffineNorm
 *       - DemBones::nWeightsIters, DemBones::nnz,
 *         DemBones::weightsSmooth, DemBones::weightsSmoothStep, DemBones::weightEps
 *   4. [@c optional] Setup extended class:
 *       - Load data: DemBonesExt::parent, DemBonesExt::preMulInv,
 *         DemBonesExt::rotOrder, DemBonesExt::orient, DemBonesExt::bind
 *       - Set parameter DemBonesExt::bindUpdate
 *   5. [@c optional] Override callback functions (cb...) in the base class
 *      @ref DemBones
 *   6. Call decomposition function DemBones::compute(),
 *      DemBones::computeWeights(), DemBones::computeTranformations(), or
 *      DemBones::init()
 *   7. [@c optional] Get local transformations/bind poses with
 *      DemBonesExt::computeRTB()
 */

/**
 * @class DemBones DemBones.h "DemBones/DemBones.h"
 * @brief Smooth skinning decomposition with rigid bones and sparse, convex weights
 *
 * @details Setup the required data, parameters, and call either compute(),
 *   computeWeights(), computeTranformations(), or init().
 *
 * Callback functions and read-only values can be used to report progress
 *   and stop on convergence: cbInitSplitBegin(), cbInitSplitEnd(), cbIterBegin(),
 *   cbIterEnd(), cbWeightsBegin(), cbWeightsEnd(), cbTranformationsBegin(),
 *   cbTransformationsEnd(), cbTransformationsIterBegin(),
 *   cbTransformationsIterEnd(), cbWeightsIterBegin(), cbWeightsIterEnd(),
 *   rmse(), #iter, #iterTransformations, #iterWeights.
 *
 * @b _Scalar is the floating-point data type. @b _AniMeshScalar is the
 *   floating-point data type of mesh sequence #vertex.
 */

template <class _Scalar, class _AniMeshScalar>
class DemBones {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	using MatrixX = Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using Matrix4 = Eigen::Matrix<_Scalar, 4, 4>;
	using Matrix3 = Eigen::Matrix<_Scalar, 3, 3>;
	using VectorX = Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>;
	using Vector4 = Eigen::Matrix<_Scalar, 4, 1>;
	using Vector3 = Eigen::Matrix<_Scalar, 3, 1>;
	using SparseMatrix = Eigen::SparseMatrix<_Scalar>;
	using Triplet = Eigen::Triplet<_Scalar>;

	//! [@c parameter] Number of global iterations
	int nIters = 30;

	//! [@c parameter] Number of clustering update iterations in the initalization
	int nInitIters = 10;

	//! [@c parameter] Number of bone transformations update iterations per global
	//! iteration
	int nTransIters = 5;
	//! [@c parameter] Translations affinity soft constraint
	_Scalar transAffine = _Scalar(10);
	//! [@c parameter] p-norm for bone translations affinity soft constraint
	_Scalar transAffineNorm = _Scalar(4);

	//! [@c parameter] Number of weights update iterations per global iteratio
	int nWeightsIters = 3;
	//! [@c parameter] Number of non-zero weights per vertex, @c default = 8
	int nnz = 8;
	//! [@c parameter] Weights sparseness soft constraint, @c default = 1e-5
	_Scalar weightsSparseness = _Scalar(1e-5);
	//! [@c parameter] Weights smoothness soft constraint, @c default = 1e-4
	_Scalar weightsSmooth = _Scalar(1e-4);
	//! [@c parameter] Step size for the weights smoothness soft constraint, @c
	//! default = 1.0
	_Scalar weightsSmoothStep = _Scalar(1);
	//! [@c parameter] Epsilon for weights solver, @c default = 1e-15
	_Scalar weightEps = _Scalar(1e-15);

	/** @brief Constructor and setting default parameters
	 */
	DemBones() { clear(); }

	//! Number of vertices, typically indexed by @p i
	int num_vertices;
	//! Number of bones, typically indexed by @p j
	int num_bones;
	//! Number of subjects, typically indexed by @p s
	int num_subjects;
	//! Number of total frames, typically indexed by @p k, #num_total_frames = #frame_start_index(#num_subjects)
	int num_total_frames;

	//! Start frame indices, @c size = #num_subjects+1, #frame_start_index(@p s), #frame_start_index(@p s+1) are
	//! data frames for subject @p s
	Eigen::VectorXi frame_start_index;
	//! Subject index of the frame, @c size = #num_total_frames, #frame_subject_id(@p k)=@p s, where
	//! #frame_start_index(@p s) <= @p k < #frame_start_index(<tt>s</tt>+1)
	Eigen::VectorXi frame_subject_id;

	//! Geometry at the rest poses, @c size = [3*#num_subjects, #num_vertices], #rest_pose_geometry.@a col(@p i).@a
	//! segment(3*@p s, 3) is the rest pose of vertex @p i of subject @p s
	MatrixX rest_pose_geometry;

	//! Skinning weights, @c size = [#num_bones, #num_vertices], #skinning_weights.@a col(@p i) are the skinning
	//! weights of vertex @p i, #skinning_weights(@p j, @p i) is the influence of bone @p j to
	//! vertex @p i
	SparseMatrix skinning_weights;

	//! Skinning weights lock control, @c size = #num_vertices, #lock_weight(@p i) is the amount
	//! of input skinning weights will be kept for vertex @p i, where 0 (no lock)
	//! <= #lock_weight(@p i) <= 1 (full lock)
	VectorX lock_weight;

	/** @brief Bone transformations, @c size = [4*#num_total_frames*4, 4*#num_bones], #bone_transform_mat.@a blk4(@p k,
	 @p j) is the 4*4 relative transformation matrix of bone @p j at frame @p k
		  @details Note that the transformations are relative, that is #bone_transform_mat.@a
	 blk4(@p k, @p j) brings the global transformation of bone @p j from the
	 rest pose to the pose at frame @p k.
  */
	MatrixX bone_transform_mat;

	//! Bone transformation lock control, @c size = #num_bones, #lock_mat(@p j) is the
	//! amount of input transformations will be kept for bone @p j, where
	//! #lock_mat(@p j) = 0 (no lock) or 1 (lock)
	Eigen::VectorXi lock_mat;

	//! Animated mesh sequence, @c size = [3*#num_total_frames, #num_vertices], #vertex.@a col(@p i).@a
	//! segment(3*@p k, 3) is the position of vertex @p i at frame @p k
	Eigen::Matrix<_AniMeshScalar, Eigen::Dynamic, Eigen::Dynamic> vertex;

	//! Mesh topology, @c size=[<tt>number of polygons</tt>], #fv[@p p] is the
	//! vector of vertex indices of polygon @p p
	std::vector<std::vector<int>> fv;

	/** @brief Clear all data
	 */
	void clear() {
		num_vertices = num_bones = num_subjects = num_total_frames = 0;
		frame_start_index.resize(0);
		frame_subject_id.resize(0);
		rest_pose_geometry.resize(0, 0);
		skinning_weights.resize(0, 0);
		lock_weight.resize(0);
		bone_transform_mat.resize(0, 0);
		lock_mat.resize(0);
		vertex.resize(0, 0);
		fv.resize(0);
		modelSize = -1;
		laplacian.resize(0, 0);
	}

	/**
	 * @brief Initialize missing skinning weights and/or bone transformations
	 * @details Depending on the status of #skinning_weights and #bone_transform_mat, this function will:
	 *   - Both #skinning_weights and #bone_transform_mat are already set: do nothing
	 *   - Only one in #skinning_weights or #bone_transform_mat is missing (zero size): initialize
	 *     missing matrix, i.e. #skinning_weights (or #bone_transform_mat)
	 *   - Both #skinning_weights and #bone_transform_mat are missing (zero size): initialize both with
	 *     rigid skinning using approximately #num_bones bones, i.e. values of #skinning_weights are 0 or 1.
	 *     LBG-VQ clustering is peformed using mesh sequence #vertex, rest
	 *     pose geometries #rest_pose_geometry and topology #fv.
	 *   @b Note: as the initialization does not use exactly #num_bones bones,
	 *     the value of #num_bones could be changed when both #skinning_weights and #bone_transform_mat are missing.
	 *
	 * This function is called at the beginning of every compute update
	 * functions as a safeguard.
	 */
	void init() {
		if (modelSize < 0) {
			modelSize =
					sqrt((rest_pose_geometry - (rest_pose_geometry.rowwise().sum() / num_vertices).replicate(1, num_vertices)).squaredNorm() /
							num_vertices / num_subjects);
		}
		if (laplacian.cols() != num_vertices) {
			computeSmoothSolver();
		}

		if (((int)skinning_weights.rows() != num_bones) || ((int)skinning_weights.cols() != num_vertices)) { // No skinning weight
			if (((int)bone_transform_mat.rows() != num_total_frames * 4) ||
					((int)bone_transform_mat.cols() != num_bones * 4)) { // No transformation
				int targetNB = num_bones;
				// LBG-VQ
				num_bones = 1;
				label = Eigen::VectorXi::Zero(num_vertices);
				computeTransFromLabel();

				bool cont = true;
				while (cont) {
					cbInitSplitBegin();
					int prev = num_bones;
					split(targetNB, 3);
					for (int rep = 0; rep < nInitIters; rep++) {
						computeTransFromLabel();
						computeLabel();
						pruneBones(3);
					}
					cont = (num_bones < targetNB) && (num_bones > prev);
					cbInitSplitEnd();
				}
				lock_mat = Eigen::VectorXi::Zero(num_bones);
				labelToWeights();
			} else {
				initWeights(); // Has transformations
			}
		} else { // Has skinning weights
			if (((int)bone_transform_mat.rows() != num_total_frames * 4) ||
					((int)bone_transform_mat.cols() != num_bones * 4)) { // No transformation
				bone_transform_mat = Matrix4::Identity().replicate(num_total_frames, num_bones);
				lock_mat = Eigen::VectorXi::Zero(num_bones);
			}
		}

		if (lock_weight.size() != num_vertices)
			lock_weight = VectorX::Zero(num_vertices);
		if (lock_mat.size() != num_bones)
			lock_mat = Eigen::VectorXi::Zero(num_bones);
	}

	/** @brief Update bone transformations by running #nTransIters iterations with
	 #transAffine and #transAffineNorm regularizers
		  @details Required input data:
				  - Rest shapes: #rest_pose_geometry, #fv, #num_vertices
				  - Sequence: #vertex, #num_total_frames, #frame_start_index, #frame_subject_id, #num_subjects
				  - Number of bones: #num_bones

		  Optional input data:
				  - Skinning weights: #skinning_weights, #lock_weight
				  - Bone transformations: #bone_transform_mat, #lock_mat

		  Output: #bone_transform_mat. Missing #skinning_weights and/or #bone_transform_mat (with zero size) will be initialized
	 by init().
  */
	void computeTranformations() {
		if (nTransIters == 0)
			return;

		init();
		cbTranformationsBegin();

		compute_vuT();
		compute_uuT();

		for (_iterTransformations = 0; _iterTransformations < nTransIters;
				_iterTransformations++) {
			cbTransformationsIterBegin();
			// #pragma omp parallel for
			for (int k = 0; k < num_total_frames; k++)
				for (int j = 0; j < num_bones; j++)
					if (lock_mat(j) == 0) {
						Matrix4 qpT = vuT.blk4(k, j);
						for (int it = uuT.outerIdx(j); it < uuT.outerIdx(j + 1); it++)
							if (uuT.innerIdx(it) != j)
								qpT -= bone_transform_mat.blk4(k, uuT.innerIdx(it)) *
									   uuT.val.blk4(frame_subject_id(k), it);
						qpT2m(qpT, k, j);
					}
			if (cbTransformationsIterEnd())
				return;
		}

		cbTransformationsEnd();
	}

	/** @brief Update skinning weights by running #nWeightsIters iterations with
	 #weightsSmooth and #weightsSmoothStep regularizers
		  @details Required input data:
				  - Rest shapes: #rest_pose_geometry, #fv, #num_vertices
				  - Sequence: #vertex, #num_total_frames, #frame_start_index, #frame_subject_id, #num_subjects
				  - Number of bones: #num_bones

		  Optional input data:
				  - Skinning weights: #skinning_weights, #lock_weight
				  - Bone transformations: #bone_transform_mat, #lock_mat

		  Output: #skinning_weights. Missing #skinning_weights and/or #bone_transform_mat (with zero size) will be initialized
	 by init().

  */
	void computeWeights() {
		if (nWeightsIters == 0)
			return;

		init();
		cbWeightsBegin();

		compute_mTm();

		aTb = MatrixX::Zero(num_bones, num_vertices);
		wSolver.init(nnz);
		std::vector<Triplet, Eigen::aligned_allocator<Triplet>> trip;
		trip.reserve(num_vertices * nnz);

		for (_iterWeights = 0; _iterWeights < nWeightsIters; _iterWeights++) {
			cbWeightsIterBegin();

			compute_ws();
			compute_aTb();

			double reg_scale = pow(modelSize, 2) * num_total_frames;

			trip.clear();
			// #pragma omp parallel for
			for (int i = 0; i < num_vertices; i++) {
				MatrixX aTai;
				compute_aTa(i, aTai);
				aTai = (1 - lock_weight(i)) *
							   (aTai / reg_scale + (weightsSmooth - weightsSparseness) *
														   MatrixX::Identity(num_bones, num_bones)) +
					   lock_weight(i) * MatrixX::Identity(num_bones, num_bones);
				VectorX aTbi = (1 - lock_weight(i)) * (aTb.col(i) / reg_scale +
															  weightsSmooth * ws.col(i)) +
							   lock_weight(i) * skinning_weights.col(i);

				VectorX x = (1 - lock_weight(i)) * ws.col(i) + lock_weight(i) * skinning_weights.col(i);
				Eigen::ArrayXi idx = Eigen::ArrayXi::LinSpaced(num_bones, 0, num_bones - 1);
				std::sort(idx.data(), idx.data() + num_bones,
						[&x](int i1, int i2) { return x(i1) > x(i2); });
				int nnzi = std::min(nnz, num_bones);
				while (x(idx(nnzi - 1)) < weightEps)
					nnzi--;

				VectorX x0 = skinning_weights.col(i).toDense().cwiseMax(0.0);
				x = indexing_vector(x0, idx.head(nnzi));
				_Scalar s = x.sum();
				if (s > _Scalar(0.1))
					x /= s;
				else
					x = VectorX::Constant(nnzi, _Scalar(1) / nnzi);

				wSolver.solve(indexing_row_col(aTai, idx.head(nnzi), idx.head(nnzi)),
						indexing_vector(aTbi, idx.head(nnzi)), x, true, true);

				// #pragma omp critical
				for (int j = 0; j < nnzi; j++)
					if (x(j) != 0)
						trip.push_back(Triplet(idx[j], i, x(j)));
			}

			skinning_weights.resize(num_bones, num_vertices);
			skinning_weights.setFromTriplets(trip.begin(), trip.end());

			if (cbWeightsIterEnd())
				return;
		}

		cbWeightsEnd();
	}

	/** @brief Skinning decomposition by #nIters iterations of alternative
	 updating weights and bone transformations
		  @details Required input data:
				  - Rest shapes: #rest_pose_geometry, #fv, #num_vertices
				  - Sequence: #vertex, #num_total_frames, #frame_start_index, #frame_subject_id, #num_subjects
				  - Number of bones: #num_bones

		  Optional input data:
				  - Skinning weights: #skinning_weights
				  - Bone transformations: #bone_transform_mat

		  Output: #skinning_weights, #bone_transform_mat. Missing #skinning_weights and/or #bone_transform_mat (with zero size) will be
	 initialized by init().
  */
	void compute() {
		init();

		for (_iter = 0; _iter < nIters; _iter++) {
			cbIterBegin();
			computeTranformations();
			computeWeights();
			if (cbIterEnd()) {
				break;
			}
		}
	}

	//! @return Root mean squared reconstruction error
	_Scalar rmse() {
		_Scalar e = 0;
		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++) {
			_Scalar ei = 0;
			Matrix4 mki;
			for (int k = 0; k < num_total_frames; k++) {
				mki.setZero();
				for (typename SparseMatrix::InnerIterator it(skinning_weights, i); it; ++it)
					mki += it.value() * bone_transform_mat.blk4(k, it.row());
				ei += (mki.template topLeftCorner<3, 3>() * rest_pose_geometry.vec3(frame_subject_id(k), i) +
						mki.template topRightCorner<3, 1>() -
						vertex.vec3(k, i).template cast<_Scalar>())
							  .squaredNorm();
			}
			// #pragma omp atomic
			e += ei;
		}
		return std::sqrt(e / num_total_frames / num_vertices);
	}

	//! Callback function invoked before each spliting of bone clusters in
	//! initialization
	virtual void cbInitSplitBegin() {}
	//! Callback function invoked after each spliting of bone clusters in
	//! initialization
	virtual void cbInitSplitEnd() {}

	//! Callback function invoked before each global iteration update
	virtual void cbIterBegin() {}
	//! Callback function invoked after each global iteration update, stop
	//! iteration if return true
	virtual bool cbIterEnd() { return false; }

	//! Callback function invoked before each skinning weights update
	virtual void cbWeightsBegin() {}
	//! Callback function invoked after each skinning weights update
	virtual void cbWeightsEnd() {}

	//! Callback function invoked before each bone transformations update
	virtual void cbTranformationsBegin() {}
	//! Callback function invoked after each bone transformations update
	virtual void cbTransformationsEnd() {}

	//! Callback function invoked before each local bone transformations update
	//! iteration
	virtual void cbTransformationsIterBegin() {}
	//! Callback function invoked after each local bone transformations update
	//! iteration, stop iteration if return true
	virtual bool cbTransformationsIterEnd() { return false; }

	//! Callback function invoked before each local weights update iteration
	virtual void cbWeightsIterBegin() {}
	//! Callback function invoked after each local weights update iteration, stop
	//! iteration if return true
	virtual bool cbWeightsIterEnd() { return false; }

private:
	int _iter, _iterTransformations, _iterWeights;

	/** Best rigid transformation from covariance matrix
		  @param _qpT is the 4*4 covariance matrix
		  @param k is the frame number
		  @param j is the bone index
  */
	void qpT2m(const Matrix4 &_qpT, int k, int j) {
		if (_qpT(3, 3) != 0) {
			Matrix4 qpT = _qpT / _qpT(3, 3);
			Eigen::JacobiSVD<Matrix3> svd(
					qpT.template topLeftCorner<3, 3>() -
							qpT.template topRightCorner<3, 1>() *
									qpT.template bottomLeftCorner<1, 3>(),
					Eigen::ComputeFullU | Eigen::ComputeFullV);
			Matrix3 d = Matrix3::Identity();
			d(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();
			bone_transform_mat.rotMat(k, j) = svd.matrixU() * d * svd.matrixV().transpose();
			bone_transform_mat.transVec(k, j) =
					qpT.template topRightCorner<3, 1>() -
					bone_transform_mat.rotMat(k, j) * qpT.template bottomLeftCorner<1, 3>().transpose();
		}
	}

	/** Fitting error
		  @param i is the vertex index
		  @param j is the bone index
  */
	_Scalar errorVtxBone(int i, int j, bool par = true) {
		_Scalar e = 0;
		// #pragma omp parallel for if (par)
		for (int k = 0; k < num_total_frames; k++)
			// #pragma omp atomic
			e += (bone_transform_mat.rotMat(k, j) * rest_pose_geometry.vec3(frame_subject_id(k), i) + bone_transform_mat.transVec(k, j) -
					vertex.vec3(k, i).template cast<_Scalar>())
						 .squaredNorm();
		return e;
	}

	//! label(i) is the index of the bone associated with vertex i
	Eigen::VectorXi label;

	//! Comparator for heap with smallest values on top
	struct TripletLess {
		bool operator()(const Triplet &t1, const Triplet &t2) {
			return t1.value() > t2.value();
		}
	};

	/** Update labels of vertices
	 */
	void computeLabel() {
		VectorX ei(num_vertices);
		Eigen::VectorXi seed = Eigen::VectorXi::Constant(num_bones, -1);
		VectorX gLabelMin(num_bones);
		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++) {
			int j = label(i);
			if (j != -1) {
				ei(i) = errorVtxBone(i, j, false);
				if ((seed(j) == -1) || (ei(i) < gLabelMin(j))) {
					// #pragma omp critical
					if ((seed(j) == -1) || (ei(i) < gLabelMin(j))) {
						gLabelMin(j) = ei(i);
						seed(j) = i;
					}
				}
			}
		}

		std::priority_queue<Triplet,
				std::vector<Triplet, Eigen::aligned_allocator<Triplet>>,
				TripletLess>
				heap;
		for (int j = 0; j < num_bones; j++)
			if (seed(j) != -1)
				heap.push(Triplet(j, seed(j), ei(seed(j))));

		if (laplacian.cols() != num_vertices) {
			computeSmoothSolver();
		}

		std::vector<bool> dirty(num_vertices, true);
		while (!heap.empty()) {
			Triplet top = heap.top();
			heap.pop();
			int i = (int)top.col();
			int j = (int)top.row();
			if (dirty[i]) {
				label(i) = j;
				ei(i) = top.value();
				dirty[i] = false;
				for (typename SparseMatrix::InnerIterator it(laplacian, i); it; ++it) {
					int i2 = (int)it.row();
					if (dirty[i2]) {
						double tmp = (label(i2) == j) ? ei(i2) : errorVtxBone(i2, j);
						heap.push(Triplet(j, i2, tmp));
					}
				}
			}
		}

		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++)
			if (label(i) == -1) {
				_Scalar gMin;
				for (int j = 0; j < num_bones; j++) {
					_Scalar ej = errorVtxBone(i, j, false);
					if ((label(i) == -1) || (gMin > ej)) {
						gMin = ej;
						label(i) = j;
					}
				}
			}
	}

	/** Update bone transformation from label
	 */
	void computeTransFromLabel() {
		bone_transform_mat = Matrix4::Identity().replicate(num_total_frames, num_bones);
		// #pragma omp parallel for
		for (int frame_i = 0; frame_i < num_total_frames; frame_i++) {
			MatrixX qpT = MatrixX::Zero(4, 4 * num_bones);
			for (int vertex_i = 0; vertex_i < num_vertices; vertex_i++) {
				if (label(vertex_i) != -1) {
					int32_t subject = frame_subject_id(frame_i);
					qpT.blk4(0, label(vertex_i)) += Vector4(vertex.vec3(frame_i, vertex_i).template cast<_Scalar>().homogeneous()) *
													rest_pose_geometry.vec3(subject, vertex_i).homogeneous().transpose();
				}
			}
			for (int bone_i = 0; bone_i < num_bones; bone_i++) {
				qpT2m(qpT.blk4(0, bone_i), frame_i, bone_i);
			}
		}
	}

	/** Set matrix skinning_weights from label
	 */
	void labelToWeights() {
		std::vector<Triplet, Eigen::aligned_allocator<Triplet>> trip(num_vertices);
		for (int i = 0; i < num_vertices; i++)
			trip[i] = Triplet(label(i), i, _Scalar(1));
		skinning_weights.resize(num_bones, num_vertices);
		skinning_weights.setFromTriplets(trip.begin(), trip.end());
		lock_weight = VectorX::Zero(num_vertices);
	}

	/** Split bone clusters
		  @param maxB is the maximum number of bones
		  @param threshold*2 is the minimum size of the bone cluster to be
	 splited
  */
	void split(int maxB, int threshold) {
		// Centroids
		MatrixX cu = MatrixX::Zero(3 * num_subjects, num_bones);
		Eigen::VectorXi s = Eigen::VectorXi::Zero(num_bones);
		for (int i = 0; i < num_vertices; i++) {
			cu.col(label(i)) += rest_pose_geometry.col(i);
			s(label(i))++;
		}
		for (int j = 0; j < num_bones; j++)
			if (s(j) != 0)
				cu.col(j) /= _Scalar(s(j));

		// Distance to centroid & error
		VectorX d(num_vertices), e(num_vertices);
		VectorX minD = VectorX::Constant(num_bones, std::numeric_limits<_Scalar>::max());
		VectorX minE = VectorX::Constant(num_bones, std::numeric_limits<_Scalar>::max());
		VectorX ce = VectorX::Zero(num_bones);

		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++) {
			int j = label(i);
			d(i) = (rest_pose_geometry.col(i) - cu.col(j)).norm();
			e(i) = sqrt(errorVtxBone(i, j, false));
			if (d(i) < minD(j)) {
				// #pragma omp critical
				minD(j) = std::min(minD(j), d(i));
			}
			if (e(i) < minE(j)) {
				// #pragma omp critical
				minE(j) = std::min(minE(j), e(i));
			}
			// #pragma omp atomic
			ce(j) += e(i);
		}

		// Seed
		Eigen::VectorXi seed = Eigen::VectorXi::Constant(num_bones, -1);
		VectorX gMax(num_bones);

		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++) {
			int j = label(i);
			double tmp = abs((e(i) - minE(j)) * (d(i) - minD(j)));

			if ((seed(j) == -1) || (tmp > gMax(j))) {
				// #pragma omp critical
				if ((seed(j) == -1) || (tmp > gMax(j))) {
					gMax(j) = tmp;
					seed(j) = i;
				}
			}
		}

		int countID = num_bones;
		_Scalar avgErr = ce.sum() / num_bones;
		for (int j = 0; j < num_bones; j++)
			if ((countID < maxB) && (s(j) > threshold * 2) &&
					(ce(j) > avgErr / 100)) {
				int newLabel = countID++;
				int i = seed(j);
				for (typename SparseMatrix::InnerIterator it(laplacian, i); it; ++it)
					label(it.row()) = newLabel;
			}
		num_bones = countID;
	}

	/** Remove bones with small number of associated vertices
		  @param threshold is the minimum number of vertices assigned to a bone
  */
	void pruneBones(int threshold) {
		Eigen::VectorXi s = Eigen::VectorXi::Zero(num_bones);
		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++) {
			// #pragma omp atomic
			s(label(i))++;
		}

		Eigen::VectorXi newID(num_bones);
		int countID = 0;
		for (int j = 0; j < num_bones; j++)
			if (s(j) < threshold)
				newID(j) = -1;
			else
				newID(j) = countID++;

		if (countID == num_bones)
			return;

		for (int j = 0; j < num_bones; j++)
			if (newID(j) != -1)
				bone_transform_mat.template middleCols<4>(newID(j) * 4) =
						bone_transform_mat.template middleCols<4>(j * 4);

		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++)
			label(i) = newID(label(i));

		num_bones = countID;
		bone_transform_mat.conservativeResize(num_total_frames * 4, num_bones * 4);
		computeLabel();
	}

	/** Initialize skinning weights with rigid bind to the best bone
	 */
	void initWeights() {
		label = Eigen::VectorXi::Constant(num_vertices, -1);
		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++) {
			_Scalar gMin;
			for (int j = 0; j < num_bones; j++) {
				_Scalar ej = errorVtxBone(i, j, false);
				if ((label(i) == -1) || (gMin > ej)) {
					gMin = ej;
					label(i) = j;
				}
			}
		}
		computeLabel();
		labelToWeights();
	}

	//! vuT.blk4(k, j) = \sum_{i=0}^{num_vertices-1}  skinning_weights(j, i)*vertex.vec3(k,
	//! i).homogeneous()*rest_pose_geometry.vec3(frame_subject_id(k), i).homogeneous()^T
	MatrixX vuT;

	/** Pre-compute vuT with bone translations affinity soft constraint
	 */
	void compute_vuT() {
		vuT = MatrixX::Zero(num_total_frames * 4, num_bones * 4);
		// #pragma omp parallel for
		for (int k = 0; k < num_total_frames; k++) {
			MatrixX vuTp = MatrixX::Zero(4, num_bones * 4);
			for (int i = 0; i < num_vertices; i++)
				for (typename SparseMatrix::InnerIterator it(skinning_weights, i); it; ++it) {
					Matrix4 tmp =
							Vector4(vertex.vec3(k, i).template cast<_Scalar>().homogeneous()) *
							rest_pose_geometry.vec3(frame_subject_id(k), i).homogeneous().transpose();
					vuT.blk4(k, it.row()) += it.value() * tmp;
					vuTp.blk4(0, it.row()) += pow(it.value(), transAffineNorm) * tmp;
				}
			for (int j = 0; j < num_bones; j++)
				if (vuTp(3, j * 4 + 3) != 0)
					vuT.blk4(k, j) +=
							(transAffine * vuT(k * 4 + 3, j * 4 + 3) / vuTp(3, j * 4 + 3)) *
							vuTp.blk4(0, j);
		}
	}

	//! uuT is a sparse block matrix, uuT(j, k).block<4, 4>(s*4, 0) =
	//! \sum{i=0}{num_vertices-1} skinning_weights(j, i)*skinning_weights(k,
	//! i)*rest_pose_geometry.col(i).segment<3>(s*3).homogeneous().transpose()*rest_pose_geometry.col(i).segment<3>(s*3).homogeneous()
	struct SparseMatrixBlock {
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		MatrixX val;
		Eigen::VectorXi innerIdx, outerIdx;
	} uuT;

	/** Pre-compute uuT for bone transformations update
	 */
	void compute_uuT() {
		Eigen::MatrixXi pos = Eigen::MatrixXi::Constant(num_bones, num_bones, -1);
		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++)
			for (typename SparseMatrix::InnerIterator it(skinning_weights, i); it; ++it)
				for (typename SparseMatrix::InnerIterator jt(skinning_weights, i); jt; ++jt)
					pos(it.row(), jt.row()) = 1;

		uuT.outerIdx.resize(num_bones + 1);
		uuT.innerIdx.resize(num_bones * num_bones);
		int nnz = 0;
		for (int j = 0; j < num_bones; j++) {
			uuT.outerIdx(j) = nnz;
			for (int i = 0; i < num_bones; i++)
				if (pos(i, j) != -1) {
					uuT.innerIdx(nnz) = i;
					pos(i, j) = nnz++;
				}
		}
		uuT.outerIdx(num_bones) = nnz;
		uuT.innerIdx.conservativeResize(nnz);
		uuT.val = MatrixX::Zero(num_subjects * 4, nnz * 4);

		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++)
			for (typename SparseMatrix::InnerIterator it(skinning_weights, i); it; ++it)
				for (typename SparseMatrix::InnerIterator jt(skinning_weights, i); jt; ++jt)
					if (it.row() >= jt.row()) {
						double _w = it.value() * jt.value();
						MatrixX _uuT(4 * num_subjects, 4);
						Vector4 _u;
						for (int s = 0; s < num_subjects; s++) {
							_u = rest_pose_geometry.vec3(s, i).homogeneous();
							_uuT.blk4(s, 0) = _w * _u * _u.transpose();
						}
						int p = pos(it.row(), jt.row()) * 4;
						for (int c = 0; c < 4; c++)
							for (int r = 0; r < 4 * num_subjects; r++)
								// #pragma omp atomic
								uuT.val(r, p + c) += _uuT(r, c);
					}

		for (int i = 0; i < num_bones; i++)
			for (int j = i + 1; j < num_bones; j++)
				if (pos(i, j) != -1)
					uuT.val.middleCols(pos(i, j) * 4, 4) =
							uuT.val.middleCols(pos(j, i) * 4, 4);
	}

	//! mTm.size = (4*num_subjects*num_bones, 4*num_bones), where mTm.block<4, 4>(s*num_bones+i, j) =
	//! \sum_{k=frame_start_index(s)}^{frame_start_index(s+1)-1} bone_transform_mat.block<3, 4>(k*4, i*4)^T*bone_transform_mat.block<3,
	//! 4>(k*4, j*4)
	MatrixX mTm;

	/** Pre-compute mTm for weights update
	 */
	void compute_mTm() {
		Eigen::MatrixXi idx(2, num_bones * (num_bones + 1) / 2);
		int nPairs = 0;
		for (int i = 0; i < num_bones; i++)
			for (int j = i; j < num_bones; j++) {
				idx(0, nPairs) = i;
				idx(1, nPairs) = j;
				nPairs++;
			}

		mTm = MatrixX::Zero(num_subjects * num_bones * 4, num_bones * 4);
		// #pragma omp parallel for
		for (int p = 0; p < nPairs; p++) {
			int i = idx(0, p);
			int j = idx(1, p);
			for (int k = 0; k < num_total_frames; k++)
				mTm.blk4(frame_subject_id(k) * num_bones + i, j) +=
						bone_transform_mat.blk4(k, i).template topRows<3>().transpose() *
						bone_transform_mat.blk4(k, j).template topRows<3>();
			if (i != j)
				for (int s = 0; s < num_subjects; s++)
					mTm.blk4(s * num_bones + j, i) = mTm.blk4(s * num_bones + i, j);
		}
	}

	//! aTb.col(i) is the A^Tb for vertex i, where A.size = (3*num_total_frames, num_bones),
	//! A.col(j).segment<3>(f*3) is the transformed position of vertex i by bone j
	//! at frame f, b = vertex.col(i).
	MatrixX aTb;

	/** Pre-compute aTb for weights update
	 */
	void compute_aTb() {
		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++)
			for (int j = 0; j < num_bones; j++)
				if ((aTb(j, i) == 0) && (ws(j, i) > weightEps))
					for (int k = 0; k < num_total_frames; k++)
						aTb(j, i) += vertex.vec3(k, i).template cast<_Scalar>().dot(
								bone_transform_mat.blk4(k, j).template topRows<3>() *
								rest_pose_geometry.vec3(frame_subject_id(k), i).homogeneous());
	}

	//! Size of the model=RMS distance to centroid
	_Scalar modelSize;

	//! Laplacian matrix
	SparseMatrix laplacian;

	//! LU factorization of Laplacian
	Eigen::SparseLU<SparseMatrix> smoothSolver;

	/** Pre-compute Laplacian and LU factorization
	 */
	void computeSmoothSolver() {
		int nFV = (int)fv.size();

		_Scalar epsDis = 0;
		for (int f = 0; f < nFV; f++) {
			int nf = (int)fv[f].size();
			for (int g = 0; g < nf; g++) {
				int i = fv[f][g];
				int j = fv[f][(g + 1) % nf];
				epsDis += (rest_pose_geometry.col(i) - rest_pose_geometry.col(j)).norm();
			}
		}
		epsDis = epsDis * weightEps / (_Scalar)num_subjects;

		std::vector<Triplet, Eigen::aligned_allocator<Triplet>> triplet;
		VectorX d = VectorX::Zero(num_vertices);
		std::vector<std::set<int>> isComputed(num_vertices);

		// #pragma omp parallel for
		for (int f = 0; f < nFV; f++) {
			int nf = (int)fv[f].size();
			for (int g = 0; g < nf; g++) {
				int i = fv[f][g];
				int j = fv[f][(g + 1) % nf];

				bool needCompute = false;
				// #pragma omp critical
				if (isComputed[i].find(j) == isComputed[i].end()) {
					needCompute = true;
					isComputed[i].insert(j);
					isComputed[j].insert(i);
				}

				if (needCompute) {
					double val = 0;
					for (int s = 0; s < num_subjects; s++) {
						double du = (rest_pose_geometry.vec3(s, i) - rest_pose_geometry.vec3(s, j)).norm();
						for (int k = frame_start_index(s); k < frame_start_index(s + 1); k++)
							val += pow((vertex.vec3(k, i).template cast<_Scalar>() -
											   vertex.vec3(k, j).template cast<_Scalar>())
													   .norm() -
											   du,
									2);
					}
					val = 1 / (sqrt(val / num_total_frames) + epsDis);

					// #pragma omp critical
					triplet.push_back(Triplet(i, j, -val));
					// #pragma omp atomic
					d(i) += val;

					// #pragma omp critical
					triplet.push_back(Triplet(j, i, -val));
					// #pragma omp atomic
					d(j) += val;
				}
			}
		}

		for (int i = 0; i < num_vertices; i++)
			triplet.push_back(Triplet(i, i, d(i)));

		laplacian.resize(num_vertices, num_vertices);
		laplacian.setFromTriplets(triplet.begin(), triplet.end());

		for (int i = 0; i < num_vertices; i++)
			if (d(i) != 0)
				laplacian.row(i) /= d(i);

		laplacian = weightsSmoothStep * laplacian +
					SparseMatrix((VectorX::Ones(num_vertices)).asDiagonal());
		smoothSolver.compute(laplacian);
	}

	//! Smoothed skinning weights
	MatrixX ws;

	/** Implicit skinning weights Laplacian smoothing
	 */
	void compute_ws() {
		ws = skinning_weights.transpose();
		// #pragma omp parallel for
		for (int j = 0; j < num_bones; j++)
			ws.col(j) = smoothSolver.solve(ws.col(j));
		ws.transposeInPlace();

		// #pragma omp parallel for
		for (int i = 0; i < num_vertices; i++) {
			ws.col(i) = ws.col(i).cwiseMax(0.0);
			_Scalar si = ws.col(i).sum();
			if (si < _Scalar(0.1))
				ws.col(i) = VectorX::Constant(num_bones, _Scalar(1) / num_bones);
			else
				ws.col(i) /= si;
		}
	}

	//! Per-vertex weights solver
	ConvexLS<_Scalar> wSolver;

	/** Pre-compute aTa for weights update on one vertex
		  @param i is the vertex index.
		  @param aTa is the by-reference output of A^TA for vertex i, where
	 A.size = (3*num_total_frames, num_bones), A.col(j).segment<3>(f*3) is the transformed position
	 of vertex i by bone j at frame f.
  */
	void compute_aTa(int i, MatrixX &aTa) {
		aTa = MatrixX::Zero(num_bones, num_bones);
		for (int j1 = 0; j1 < num_bones; j1++)
			for (int j2 = j1; j2 < num_bones; j2++) {
				for (int s = 0; s < num_subjects; s++)
					aTa(j1, j2) += rest_pose_geometry.vec3(s, i).homogeneous().dot(
							mTm.blk4(s * num_bones + j1, j2) * rest_pose_geometry.vec3(s, i).homogeneous());
				if (j1 != j2)
					aTa(j2, j1) = aTa(j1, j2);
			}
	}
};
} // namespace Dem

#ifdef DEM_BONES_DEM_BONES_MAT_BLOCKS_UNDEFINED
#undef blk4
#undef rotMat
#undef transVec
#undef vec3
#undef DEM_BONES_MAT_BLOCKS
#endif

#endif

#include "thirdparty/eigen/Eigen/Eigen/Geometry"
#include <stdint.h>

#ifndef DEM_BONES_MAT_BLOCKS
#include "MatBlocks.h"
#define DEM_BONES_DEM_BONES_EXT_MAT_BLOCKS_UNDEFINED
#endif

#include "core/config/engine.h"
#include "dem_bones.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/importer_mesh.h"
#include "scene/resources/mesh.h"

namespace Dem {
/**  @class DemBonesExt DemBonesExt.h "DemBones/DemBonesExt.h"
		@brief Extended class to handle hierarchical skeleton with local
   rotations/translations and bind matrices

		@details Call computeRTB() to get local rotations/translations and bind
   matrices after skinning decomposition is done and other data is set.

		@b _Scalar is the floating-point data type. @b _AniMeshScalar is the
   floating-point data type of mesh sequence #vertex.
*/
template <class _Scalar, class _AniMeshScalar>
class DemBonesExt : public DemBones<_Scalar, _AniMeshScalar> {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	using MatrixX = Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using Matrix4 = Eigen::Matrix<_Scalar, 4, 4>;
	using Matrix3 = Eigen::Matrix<_Scalar, 3, 3>;
	using VectorX = Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>;
	using Vector4 = Eigen::Matrix<_Scalar, 4, 1>;
	using Vector3 = Eigen::Matrix<_Scalar, 3, 1>;
	using SparseMatrix = Eigen::SparseMatrix<_Scalar>;
	using Triplet = Eigen::Triplet<_Scalar>;

	using DemBones<_Scalar, _AniMeshScalar>::nIters;
	using DemBones<_Scalar, _AniMeshScalar>::nInitIters;
	using DemBones<_Scalar, _AniMeshScalar>::nTransIters;
	using DemBones<_Scalar, _AniMeshScalar>::transAffine;
	using DemBones<_Scalar, _AniMeshScalar>::transAffineNorm;
	using DemBones<_Scalar, _AniMeshScalar>::nWeightsIters;
	using DemBones<_Scalar, _AniMeshScalar>::nnz;
	using DemBones<_Scalar, _AniMeshScalar>::weightsSmooth;
	using DemBones<_Scalar, _AniMeshScalar>::weightsSmoothStep;
	using DemBones<_Scalar, _AniMeshScalar>::weightEps;

	using DemBones<_Scalar, _AniMeshScalar>::num_vertices;
	using DemBones<_Scalar, _AniMeshScalar>::num_bones;
	using DemBones<_Scalar, _AniMeshScalar>::num_subjects;
	using DemBones<_Scalar, _AniMeshScalar>::num_total_frames;
	using DemBones<_Scalar, _AniMeshScalar>::frame_start_index;
	using DemBones<_Scalar, _AniMeshScalar>::frame_subject_id;
	using DemBones<_Scalar, _AniMeshScalar>::rest_pose_geometry;
	using DemBones<_Scalar, _AniMeshScalar>::skinning_weights;
	using DemBones<_Scalar, _AniMeshScalar>::lock_weight;
	using DemBones<_Scalar, _AniMeshScalar>::bone_transform_mat;
	using DemBones<_Scalar, _AniMeshScalar>::lock_mat;
	using DemBones<_Scalar, _AniMeshScalar>::vertex;
	using DemBones<_Scalar, _AniMeshScalar>::fv;

	//! Timestamps for bone transformations #bone_transform_mat, [@c size] = #num_subjects, #fTime(@p k) is
	//! the timestamp of frame @p k
	Eigen::VectorXd fTime;

	//! Name of bones, [@c size] = #num_bones, #boneName(@p j) is the name bone of @p j
	std::vector<std::string> bone_name;

	//! Parent bone index, [@c size] = #num_bones, #parent(@p j) is the index of parent
	//! bone of @p j, #parent(@p j) = -1 if @p j has no parent.
	Eigen::VectorXi parent;

	//! Original bind pre-matrix, [@c size] = [4*#num_subjects, 4*#num_bones], #bind.@a block(4*@p
	//! s, 4*@p j, 4, 4) is the global bind matrix of bone @p j on subject @p s at
	//! the rest pose
	MatrixX bind;

	//! Inverse pre-multiplication matrices, [@c size] = [4*#num_subjects, 4*#num_bones],
	//! #preMulInv.@a block(4*@p s, 4*@p j, 4, 4) is the inverse of pre-local
	//! transformation of bone @p j on subject @p s
	MatrixX pre_mult_inv;

	//! Rotation order, [@c size] = [3*#num_subjects, #num_bones], #rotOrder.@a col(@p j).@a
	//! segment<3>(3*@p s) is the rotation order of bone @p j on subject @p s,
	//! 0=@c X, 1=@c Y, 2=@c Z, e.g. {0, 1, 2} is @c XYZ order
	Eigen::MatrixXi rot_order;

	//! Orientations of bones,  [@c size] = [3*#num_subjects, #num_bones], @p orient.@a col(@p
	//! j).@a segment<3>(3*@p s) is the(@c rx, @c ry, @c rz) orientation of bone
	//! @p j in degree
	MatrixX orient;

	//! Bind transformation update, 0=keep original, 1=set translations to p-norm
	//! centroids (using #transAffineNorm) and rotations to identity, 2=do 1 and
	//! group joints
	int bind_update = 0;

	int patience = 3;

	double tolerance = 1e-3;

	/** @brief Constructor and setting default parameters
	 */
	DemBonesExt();

	/** @brief Clear all data
	 */
	void clear();

	/** @brief Local rotations, translations and global bind matrices of a subject
		  @details Required all data in the base class: #rest_pose_geometry, #fv, #num_vertices, #vertex, #num_total_frames,
	 #frame_start_index, #frame_subject_id, #num_subjects, #bone_transform_mat, #skinning_weights, #num_bones

		  This function will initialize missing attributes:
		  - #parent: -1 vector (if no joint grouping) or parent to a root, [@c
	 size] = #num_bones
		  - #preMulInv: 4*4 identity matrix blocks, [@c size] = [4*#num_subjects, 4*#num_bones]
		  - #rotOrder: {0, 1, 2} vector blocks, [@c size] = [3*#num_subjects, #num_bones]
		  - #orient: 0 matrix, [@c size] = [3*#num_subjects, #num_bones]

		  @param[in] s is the subject index
		  @param[out] lr is the [3*@p nFr, #num_bones] by-reference output local
	 rotations, @p lr.@a col(@p j).segment<3>(3*@p k) is the (@c rx, @c ry, @c
	 rz) of bone @p j at frame @p k
		  @param[out] lt is the [3*@p nFr, #num_bones] by-reference output local
	 translations, @p lt.@a col(@p j).segment<3>(3*@p k) is the (@c tx, @c ty,
	 @c tz) of bone @p j at frame @p k
		  @param[out] gb is the [4, 4*#num_bones] by-reference output global bind
	 matrices, @p gb.@a block(0, 4*@p j, 4, 4) is the bind matrix of bone j
		  @param[out] lbr is the [3, #num_bones] by-reference output local rotations at
	 bind pose @p lbr.@a col(@p j).segment<3>(3*@p k) is the (@c rx, @c ry, @c
	 rz) of bone @p j
		  @param[out] lbt is the [3, #num_bones] by-reference output local translations
	 at bind pose, @p lbt.@a col(@p j).segment<3>(3*@p k) is the (@c tx, @c ty,
	 @c tz) of bone @p j
		  @param[in] degreeRot=true will output rotations in degree, otherwise
	 output in radian
  */
	void compute_rotations_translations_bind_matrices(int s, MatrixX &r_local_rotations, MatrixX &r_local_translations, MatrixX &gb, MatrixX &local_bind_pose_rotation,
			MatrixX &r_local_bind_pose_translation, bool degree_rotation = true);

private:
	void compute();

	/** p-norm centroids (using #transAffineNorm) and rotations to identity
		  @param s is the subject index
		  @param b is the [4, 4*#num_bones] by-reference output global bind matrices,
	 #b.#a block(0, 4*@p j, 4, 4) is the bind matrix of bone @p j
  */
	void compute_centroids(int s, MatrixX &b);

	/** Global bind pose
		  @param s is the subject index
		  @param bindUpdate is the type of bind pose update, 0=keep original, 1
	 or 2=set translations to p-norm centroids (using #transAffineNorm) and
	 rotations to identity
		  @param b is the the [4, 4*#num_bones] by-reference output global bind
	 matrices, #b.#a block(0, 4*@p j, 4, 4) is the bind matrix of bone @p j
  */
	void compute_bind(int p_subject, MatrixX &r_output_global_bind_matrix);

	/** Root joint
	 */
	int compute_root();

	/** Euler angles from rotation matrix
		  @param rMat is the 3*3 rotation matrix
		  @param curRot is the input current Euler angles, it is also the
	 by-reference output closet Euler angles correspond to @p rMat
		  @param ro is the rotation order, 0=@c X, 1=@c Y, 2=@c Z, e.g. {0, 1,
	 2} is @c XYZ order
		  @param eps is the epsilon
  */
	void euler_angles_from_rotation_matrix_to_rot(const Matrix3 &p_basis, Vector3 &r_input_euler, const Eigen::Vector3i &p_rotation_order,
			_Scalar p_epsilon = _Scalar(1e-10));
	struct Key {
		double time = 0.0; // time in secs
	};

	// transform key holds either Vector3 or Quaternion
	template <class T>
	struct TransformKey {
		::Vector3 loc;
		::Quaternion rot;
		::Vector3 scale;
	};

public:
	Dictionary convert_blend_shapes_without_bones(Array p_mesh, ::PackedVector3Array p_vertex_array, HashMap<String, Vector<::Vector3>> p_blends, Vector<Ref<Animation>> p_anims);
};

template <class _Scalar, class _AniMeshScalar>
Dem::DemBonesExt<_Scalar, _AniMeshScalar>::DemBonesExt() {
	clear();
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::clear() {
	fTime.resize(0);
	bone_name.resize(0);
	parent.resize(0);
	bind.resize(0, 0);
	pre_mult_inv.resize(0, 0);
	rot_order.resize(0, 0);
	orient.resize(0, 0);
	DemBones<_Scalar, _AniMeshScalar>::clear();
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::compute_rotations_translations_bind_matrices(int s, MatrixX &r_local_rotations, MatrixX &r_local_translations, MatrixX &gb, MatrixX &r_local_bind_pose_rotation, MatrixX &r_local_bind_pose_translation, bool degree_rot /*= true*/) {
	compute_bind(s, gb);

	if (parent.size() == 0) {
		if (bind_update == 2) {
			int root = compute_root();
			parent = Eigen::VectorXi::Constant(num_bones, root);
			parent(root) = -1;
		} else {
			parent = Eigen::VectorXi::Constant(num_bones, -1);
		}
	}
	if (pre_mult_inv.size() == 0) {
		pre_mult_inv = MatrixX::Identity(4, 4).replicate(num_subjects, num_bones);
	}
	if (rot_order.size() == 0) {
		rot_order = Eigen::Vector3i(0, 1, 2).replicate(num_subjects, num_bones);
	}
	if (orient.size() == 0) {
		orient = MatrixX::Zero(3 * num_subjects, num_bones);
	}

	int nFs = frame_start_index(s + 1) - frame_start_index(s);
	r_local_rotations.resize(nFs * 3, num_bones);
	r_local_translations.resize(nFs * 3, num_bones);
	r_local_bind_pose_rotation.resize(3, num_bones);
	r_local_bind_pose_translation.resize(3, num_bones);

	// #pragma omp parallel for
	for (int bone_i = 0; bone_i < num_bones; bone_i++) {
		Eigen::Vector3i ro = rot_order.col(bone_i).template segment<3>(s * 3);

		Vector3 ov = orient.vec3(s, bone_i) * EIGEN_PI / 180;
		Matrix3 invOM =
				Matrix3(Eigen::AngleAxis<_Scalar>(ov(ro(2)), Vector3::Unit(ro(2)))) *
				Eigen::AngleAxis<_Scalar>(ov(ro(1)), Vector3::Unit(ro(1))) *
				Eigen::AngleAxis<_Scalar>(ov(ro(0)), Vector3::Unit(ro(0)));
		invOM.transposeInPlace();

		Matrix4 lb;
		if (parent(bone_i) == -1) {
			lb = pre_mult_inv.blk4(s, bone_i) * gb.blk4(0, bone_i);
		} else {
			lb = pre_mult_inv.blk4(s, bone_i) * gb.blk4(0, parent(bone_i)).inverse() *
				 gb.blk4(0, bone_i);
		}

		Vector3 curRot = Vector3::Zero();
		euler_angles_from_rotation_matrix_to_rot(invOM * lb.template topLeftCorner<3, 3>(), curRot, ro);
		r_local_bind_pose_rotation.col(bone_i) = curRot;
		r_local_bind_pose_translation.col(bone_i) = lb.template topRightCorner<3, 1>();

		Matrix4 _lm;
		for (int k = 0; k < nFs; k++) {
			if (parent(bone_i) == -1) {
				_lm = pre_mult_inv.blk4(s, bone_i) * bone_transform_mat.blk4(k + frame_start_index(s), bone_i) * gb.blk4(0, bone_i);
			} else {
				_lm = pre_mult_inv.blk4(s, bone_i) *
					  (bone_transform_mat.blk4(k + frame_start_index(s), parent(bone_i)) * gb.blk4(0, parent(bone_i)))
							  .inverse() *
					  bone_transform_mat.blk4(k + frame_start_index(s), bone_i) * gb.blk4(0, bone_i);
			}
			euler_angles_from_rotation_matrix_to_rot(invOM * _lm.template topLeftCorner<3, 3>(), curRot, ro);
			r_local_rotations.vec3(k, bone_i) = curRot;
			r_local_translations.vec3(k, bone_i) = _lm.template topRightCorner<3, 1>();
		}
	}

	if (degree_rot) {
		r_local_rotations *= 180 / EIGEN_PI;
		r_local_bind_pose_rotation *= 180 / EIGEN_PI;
	}
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::compute_centroids(int s, MatrixX &b) {
	MatrixX c = MatrixX::Zero(4, num_bones);
	for (int vert_i = 0; vert_i < num_vertices; vert_i++) {
		for (typename SparseMatrix::InnerIterator it(skinning_weights, vert_i); it; ++it) {
			c.col(it.row()) +=
					pow(it.value(), transAffineNorm) * rest_pose_geometry.vec3(s, vert_i).homogeneous();
		}
	}
	for (int bone_i = 0; bone_i < num_bones; bone_i++) {
		if ((c(3, bone_i) != 0) && (lock_mat(bone_i) == 0)) {
			b.transVec(0, bone_i) = c.col(bone_i).template head<3>() / c(3, bone_i);
		}
	}
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::compute_bind(int p_subject, MatrixX &r_output_global_bind_matrix) {
	if (bind.size() == 0) {
		lock_mat = Eigen::VectorXi::Zero(num_bones);
		bind.resize(num_subjects * 4, num_bones * 4);
		for (int subject_i = 0; subject_i < num_subjects; subject_i++) {
			r_output_global_bind_matrix = MatrixX::Identity(4, 4).replicate(1, num_bones);
			compute_centroids(subject_i, r_output_global_bind_matrix);
			bind.block(4 * subject_i, 0, 4, 4 * num_bones) = r_output_global_bind_matrix;
		}
	}
	r_output_global_bind_matrix = bind.block(4 * p_subject, 0, 4, 4 * num_bones);
	if (bind_update >= 1) {
		compute_centroids(p_subject, r_output_global_bind_matrix);
	}
}

template <class _Scalar, class _AniMeshScalar>
int Dem::DemBonesExt<_Scalar, _AniMeshScalar>::compute_root() {
	VectorX err(num_bones);
	// #pragma omp parallel for
	for (int j = 0; j < num_bones; j++) {
		double ej = 0;
		for (int i = 0; i < num_vertices; i++) {
			for (int frame_i = 0; frame_i < num_total_frames; frame_i++) {
				ej += (bone_transform_mat.rotMat(frame_i, j) * rest_pose_geometry.vec3(frame_subject_id(frame_i), i) + bone_transform_mat.transVec(frame_i, j) -
						vertex.vec3(frame_i, i).template cast<_Scalar>())
							  .squaredNorm();
			}
		}
		err(j) = ej;
	}
	int rj;
	err.minCoeff(&rj);
	return rj;
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::compute() {
	int np = patience;
	double prevErr = -1;
	DemBones<_Scalar, _AniMeshScalar>::init();
	for (int32_t iteration_i = 0; iteration_i < DemBonesExt<_Scalar, _AniMeshScalar>::nIters; iteration_i++) {
		DemBonesExt<_Scalar, _AniMeshScalar>::computeTranformations();
		DemBonesExt<_Scalar, _AniMeshScalar>::computeWeights();
		double err = DemBonesExt<_Scalar, _AniMeshScalar>::rmse();
		if ((err < prevErr * (1 + weightEps)) &&
				((prevErr - err) < DemBonesExt<_Scalar, _AniMeshScalar>::tolerance * prevErr)) {
			np--;
			if (np == 0) {
				print_line("Convergence has been reached.");
				break;
			}
		} else {
			np = patience;
		}
		prevErr = err;
		print_line("Root mean square error is " + rtos(prevErr));
	}
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::euler_angles_from_rotation_matrix_to_rot(const Matrix3 &p_basis, Vector3 &r_input_euler, const Eigen::Vector3i &p_rotation_order, _Scalar p_epsilon /*= _Scalar(1e-10)*/) {
	Vector3 r0 = p_basis.eulerAngles(p_rotation_order(2), p_rotation_order(1), p_rotation_order(0)).reverse();
	_Scalar gMin = (r0 - r_input_euler).squaredNorm();
	Vector3 rMin = r0;
	Vector3 r;
	Matrix3 tmpMat;
	for (int fx = -1; fx <= 1; fx += 2) {
		for (_Scalar sx = -2 * EIGEN_PI; sx < 2.1 * EIGEN_PI; sx += EIGEN_PI) {
			r(0) = fx * r0(0) + sx;
			for (int fy = -1; fy <= 1; fy += 2) {
				for (_Scalar sy = -2 * EIGEN_PI; sy < 2.1 * EIGEN_PI;
						sy += EIGEN_PI) {
					r(1) = fy * r0(1) + sy;
					for (int fz = -1; fz <= 1; fz += 2) {
						for (_Scalar sz = -2 * EIGEN_PI; sz < 2.1 * EIGEN_PI;
								sz += EIGEN_PI) {
							r(2) = fz * r0(2) + sz;
							tmpMat =
									Matrix3(Eigen::AngleAxis<_Scalar>(r(p_rotation_order(2)),
											Vector3::Unit(p_rotation_order(2)))) *
									Eigen::AngleAxis<_Scalar>(r(p_rotation_order(1)), Vector3::Unit(p_rotation_order(1))) *
									Eigen::AngleAxis<_Scalar>(r(p_rotation_order(0)), Vector3::Unit(p_rotation_order(0)));
							if ((tmpMat - p_basis).squaredNorm() < p_epsilon) {
								_Scalar tmp = (r - r_input_euler).squaredNorm();
								if (tmp < gMin) {
									gMin = tmp;
									rMin = r;
								}
							}
						}
					}
				}
			}
		}
	}
	r_input_euler = rMin;
}

struct BoneData {
	String name;
	int parent;
	float radius;
	Transform3D local_transform;
};

Vector<BoneData> bones;
Eigen::MatrixXf wd;
float lockW;
bool hasKeyFrame = false;
Eigen::MatrixXf m;

template <class _Scalar, class _AniMeshScalar>
Dictionary Dem::DemBonesExt<_Scalar, _AniMeshScalar>::convert_blend_shapes_without_bones(Array p_mesh, ::PackedVector3Array p_vertex_array, HashMap<String, Vector<::Vector3>> p_blends, Vector<Ref<Animation>> p_anims) {
	Dictionary output;
	output["mesh_array"] = p_mesh;
	output["animation_library"] = Ref<AnimationLibrary>();
	output["skeleton"] = memnew(Skeleton3D);
	if (p_vertex_array.size() == 0) {
		return output;
	}
	if (!p_anims.size()) {
		return output;
	}
	// TODO: 2021-10-05 Support multiple tracks by putting into one long track and then splitting back.
	Ref<Animation> anim;

	for (Ref<Animation> current_anim : p_anims) {
		if (current_anim->get_length() > 0.0) {
			anim = current_anim;
		}
	}
	if (anim.is_null()) {
		return Dictionary();
	}
	if (!p_blends.size()) {
		return output;
	}
	constexpr float FPS = 30.0f;
	int32_t new_frames = anim->get_length() * FPS;
	vertex.resize(3 * new_frames, p_vertex_array.size());

	//! Animated mesh sequence, @c size = [3*#num_total_frames, #num_vertices], #vertex.@a col(@p i).@a
	//! segment(3*@p k, 3) is the position of vertex @p i at frame @p k
	// Eigen::Matrix<_AniMeshScalar, Eigen::Dynamic, Eigen::Dynamic> vertex;

	for (int32_t frame_i = 0; frame_i < new_frames; frame_i++) {
		for (int32_t j = 0; j < p_vertex_array.size(); j++) {
			vertex(3 * frame_i + 0, j) = p_vertex_array[j].x;
			vertex(3 * frame_i + 1, j) = p_vertex_array[j].y;
			vertex(3 * frame_i + 2, j) = p_vertex_array[j].z;
		}
	}

	for (int32_t track_i = 0; track_i < anim->get_track_count(); track_i++) {
		String track_path = anim->track_get_path(track_i);
		Animation::TrackType track_type = anim->track_get_type(track_i);
		if (track_type != Animation::TYPE_BLEND_SHAPE) {
			continue;
		}
		const double increment = 1.0 / FPS;
		double time = 0.0;
		double length = anim->get_length();
		bool last = false;
		for (int32_t frame_i = 0; frame_i < new_frames; frame_i++) {
			float weight = 0.0f;
			if (anim->blend_shape_track_interpolate(track_i, time, &weight) != OK) {
				continue;
			}
			Vector<::Vector3> blend_vertices = p_blends[track_path];
			for (int32_t j = 0; j < p_vertex_array.size(); j++) {
				::Vector3 current_vertex = ::Vector3(vertex(3 * frame_i + 0, j), vertex(3 * frame_i + 1, j), vertex(3 * frame_i + 2, j));
				::Vector3 lerped_vertex = current_vertex.lerp(blend_vertices[j], weight);

				vertex(3 * frame_i + 0, j) = p_vertex_array[j].x;
				vertex(3 * frame_i + 1, j) = p_vertex_array[j].y;
				vertex(3 * frame_i + 2, j) = p_vertex_array[j].z;
			}
			time += increment;
			if (time >= length && !last) {
				last = true;
				time = length;
			} else if (last) {
				break;
			}
		}
	}
	num_total_frames = new_frames;

	num_vertices = p_vertex_array.size();

	num_subjects = 1;
	num_total_frames = FPS * anim->get_length();
	fTime.resize(num_total_frames);
	frame_start_index.resize(num_subjects + 1);
	frame_start_index(0) = 0;
	for (int s = 0; s < num_subjects; s++) {
		for (int32_t subjects_i = 0; subjects_i < num_subjects; subjects_i++) {
			frame_start_index[s + 1] = frame_start_index[s] + num_total_frames; // TODO: fire 2022-05-22 Support different start frame
		}
	}

	frame_subject_id.resize(num_total_frames);
	for (int s = 0; s < num_subjects; s++) {
		for (int k = frame_start_index(s); k < frame_start_index(s + 1); k++) {
			frame_subject_id[k] = s;
		}
	}

	PackedInt32Array indices = p_mesh[Mesh::ARRAY_INDEX];

	// Assume the mesh is a triangle mesh.
	const int indices_in_tri = 3;
	fv.resize(indices.size() / 3);
	for (int32_t index_i = 0; index_i < indices.size(); index_i += 3) {
		std::vector<int> polygon_indices;
		polygon_indices.resize(indices_in_tri);
		polygon_indices[0] = indices[index_i / 3 + 0];
		polygon_indices[1] = indices[index_i / 3 + 1];
		polygon_indices[2] = indices[index_i / 3 + 2];
		fv[index_i / 3] = polygon_indices;
	}
	rest_pose_geometry.resize(3, num_vertices);
	for (int32_t vertex_i = 0; vertex_i < p_vertex_array.size();
			vertex_i++) {
		float pos_x = p_vertex_array[vertex_i].x;
		float pos_y = p_vertex_array[vertex_i].y;
		float pos_z = p_vertex_array[vertex_i].z;
		rest_pose_geometry.col(vertex_i) << pos_x, pos_y, pos_z;
	}
	nnz = 8;
	bind_update = 2;
	nIters = 100;
	num_bones = 20;
	nInitIters = 20;
	DemBonesExt<_Scalar, _AniMeshScalar>::init();
	DemBonesExt<_Scalar, _AniMeshScalar>::compute();
	bool needCreateJoints = (bone_name.size() == 0);
	double radius;
	Skeleton3D *skeleton = memnew(Skeleton3D);
	if (needCreateJoints) {
		bone_name.resize(num_bones);
		for (int j = 0; j < num_bones; j++) {
			std::ostringstream s;
			s << "joint" << j;
			bone_name[j] = s.str();
			String joint_name = s.str().c_str();
			skeleton->add_bone(joint_name);
		}
		radius = sqrt((rest_pose_geometry - (rest_pose_geometry.rowwise().sum() / num_vertices).replicate(1, num_vertices)).cwiseAbs().rowwise().maxCoeff().squaredNorm() / num_subjects);
	}

	print_line(vformat("The number of bones %d", bone_name.size()));
	Ref<AnimationLibrary> animation_library;
	animation_library.instantiate();
	for (int s = 0; s < num_subjects; s++) {
		MatrixX local_rotations;
		MatrixX local_translations;
		MatrixX gb;
		MatrixX local_bind_pose_rotation;
		MatrixX local_bind_pose_translation;
		bool degree_rot = false;
		DemBonesExt<_Scalar, _AniMeshScalar>::compute_rotations_translations_bind_matrices(s, local_rotations, local_translations, gb, local_bind_pose_rotation, local_bind_pose_translation, degree_rot);
		BoneId bone_id = skeleton->find_bone(String(bone_name[s].c_str()));
		Transform3D rest = Transform3D(
				Basis::from_euler(::Vector3(local_bind_pose_rotation(0), local_bind_pose_rotation(1), local_bind_pose_rotation(2))),
				::Vector3(local_bind_pose_translation(0), local_bind_pose_translation(1), local_bind_pose_translation(2)));
		skeleton->set_bone_rest(bone_id, rest);
		Ref<Animation> animation;
		animation.instantiate();
		animation->set_name("Animation_" + itos(s));
		animation->set_length(double(frame_start_index(s + 1)) / FPS - double(frame_start_index(s)) / FPS);
		if (needCreateJoints) {
			for (int32_t bone_i = 0; bone_i < num_bones; bone_i++) {
				BoneId parent_bone = parent[bone_i];
				int32_t track_i = animation->get_track_count();
				animation->add_track(Animation::TYPE_POSITION_3D);
				animation->track_set_path(track_i, String(bone_name[bone_i].c_str()));
				for (int frame_i = frame_start_index(s); frame_i < frame_start_index(s + 1); frame_i++) {
					animation->position_track_insert_key(track_i, double(frame_i) / FPS, ::Vector3());
				}
				track_i = animation->get_track_count();
				animation->add_track(Animation::TYPE_ROTATION_3D);
				animation->track_set_path(track_i, String(bone_name[bone_i].c_str()));
				for (int frame_j = frame_start_index(s); frame_j < frame_start_index(s + 1); frame_j++) {
					animation->rotation_track_insert_key(track_i, double(frame_j) / FPS, ::Quaternion());
				}
			}
		}
		animation_library->add_animation("Animation " + itos(s), animation);
	}
	SparseMatrix w = (wd.cast<_Scalar>() / num_subjects).sparseView(1, 1e-20);

	lockW /= static_cast<_Scalar>(num_subjects);

	if (!hasKeyFrame) {
		m.resize(0, 0);
	}

	print_line(vformat("Number of vertices %d", num_vertices));
	print_line(vformat("Number of joints %d", num_bones));
	print_line(vformat("Number of frames %d", num_total_frames));
	print_line(vformat("Number of skinning weights %d", skinning_weights.size()));

	Array array_mesh;
	array_mesh.resize(Mesh::ARRAY_MAX);
	PackedVector3Array vertex_array;
	vertex_array.resize(num_vertices);
	for (int32_t vertex_i = 0; vertex_i < num_vertices;
			vertex_i++) {
		vertex_array[vertex_i] = ::Vector3(rest_pose_geometry.col(vertex_i)(0), rest_pose_geometry.col(vertex_i)(1), rest_pose_geometry.col(vertex_i)(2));
	}
	array_mesh[Mesh::ARRAY_VERTEX] = vertex_array;

	output["mesh_array"] = array_mesh;
	output["animation_library"] = animation_library;
	output["skeleton"] = skeleton;
	return output;
}

} // namespace Dem
#ifdef DEM_BONES_DEM_BONES_EXT_MAT_BLOCKS_UNDEFINED
#undef blk4
#undef rotMat
#undef transVec
#undef vec3
#undef DEM_BONES_MAT_BLOCKS
#endif

#undef rotMatFromEuler

#endif
