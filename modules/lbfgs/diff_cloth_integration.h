#pragma once

#include "core/object/object.h"
#include "core/variant/dictionary.h"
#include "scene/resources/mesh.h"
#include "lbfgsbpp.h"

class DiffClothIntegration : public LBFGSBSolver {
	GDCLASS(DiffClothIntegration, LBFGSBSolver);

public:
	Ref<ArrayMesh> character_collision_mesh;
	Ref<ArrayMesh> input_cloth_mesh;

	void set_character_collision_mesh(Ref<ArrayMesh> p_mesh);
	Ref<ArrayMesh> get_character_collision_mesh() const;

	void set_input_cloth_mesh(Ref<ArrayMesh> p_mesh);
	Ref<ArrayMesh> get_input_cloth_mesh() const;

	Dictionary optimize_cloth_drape();

protected:
	static void _bind_methods();

public:
	virtual double call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad) override; // Added override
	DiffClothIntegration();
	~DiffClothIntegration();
};
