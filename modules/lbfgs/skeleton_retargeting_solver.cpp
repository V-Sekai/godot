#include "skeleton_retargeting_solver.h"

#include "core/object/class_db.h"

void SkeletonRetargetingSolver::set_source_skeleton(Skeleton3D *p_skel) {
	source_skeleton = p_skel;
}

Skeleton3D *SkeletonRetargetingSolver::get_source_skeleton() const {
	return source_skeleton;
}

void SkeletonRetargetingSolver::set_target_skeleton(Skeleton3D *p_skel) {
	target_skeleton = p_skel;
}

Skeleton3D *SkeletonRetargetingSolver::get_target_skeleton() const {
	return target_skeleton;
}

Dictionary SkeletonRetargetingSolver::optimize_global_transform() {
	// Implementation needed
	return Dictionary();
}

Dictionary SkeletonRetargetingSolver::optimize_bone_local_scales() {
	// Implementation needed
	return Dictionary();
}

double SkeletonRetargetingSolver::call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad) {
	// Implementation needed
	return 0.0;
}

void SkeletonRetargetingSolver::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source_skeleton", "skeleton"), &SkeletonRetargetingSolver::set_source_skeleton);
	ClassDB::bind_method(D_METHOD("get_source_skeleton"), &SkeletonRetargetingSolver::get_source_skeleton);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "source_skeleton", PROPERTY_HINT_RESOURCE_TYPE, "Skeleton3D"), "set_source_skeleton", "get_source_skeleton");

	ClassDB::bind_method(D_METHOD("set_target_skeleton", "skeleton"), &SkeletonRetargetingSolver::set_target_skeleton);
	ClassDB::bind_method(D_METHOD("get_target_skeleton"), &SkeletonRetargetingSolver::get_target_skeleton);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "target_skeleton", PROPERTY_HINT_RESOURCE_TYPE, "Skeleton3D"), "set_target_skeleton", "get_target_skeleton");

	ClassDB::bind_method(D_METHOD("optimize_global_transform"), &SkeletonRetargetingSolver::optimize_global_transform);
	ClassDB::bind_method(D_METHOD("optimize_bone_local_scales"), &SkeletonRetargetingSolver::optimize_bone_local_scales);
}

SkeletonRetargetingSolver::SkeletonRetargetingSolver() {}

SkeletonRetargetingSolver::~SkeletonRetargetingSolver() {}
