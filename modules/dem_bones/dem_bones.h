#pragma once

#include "core/io/resource.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"

class BlendShapeBake : public Resource {
	GDCLASS(BlendShapeBake, Resource);

public:
	Error convert_scene(Node *p_scene);
};
