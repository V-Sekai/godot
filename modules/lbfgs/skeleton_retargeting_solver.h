#pragma once

#include "lbfgsbpp.h"
#include "core/object/class_db.h" // For GDCLASS
#include "core/variant/dictionary.h"
#include "scene/3d/skeleton_3d.h"

class SkeletonRetargetingSolver : public LBFGSBSolver {
	GDCLASS(SkeletonRetargetingSolver, LBFGSBSolver);

public:
	Skeleton3D* source_skeleton;
	Skeleton3D* target_skeleton;
	// Add properties for joint mapping if needed: Dictionary joint_map;

	void set_source_skeleton(Skeleton3D *p_skel);
	Skeleton3D* get_source_skeleton() const;

	void set_target_skeleton(Skeleton3D *p_skel);
	Skeleton3D * get_target_skeleton() const;
	// void set_joint_map(const Dictionary& p_map) { joint_map = p_map; }

	// Optimizes target_skeleton's root transform (translation, rotation, scale) to match source_skeleton.
	Dictionary optimize_global_transform();

	// Optimizes local scales of target_skeleton bones to match source bone lengths/proportions.
	Dictionary optimize_bone_local_scales();

protected:
	static void _bind_methods();

public:
	virtual double call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad) override;
	SkeletonRetargetingSolver();
	~SkeletonRetargetingSolver();
};
