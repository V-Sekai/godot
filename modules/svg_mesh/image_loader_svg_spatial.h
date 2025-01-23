
#ifndef LOADER_SVG_SPATIAL_H
#define LOADER_SVG_SPATIAL_H

#ifdef TOOLS_ENABLED
#include "core/io/file_access.h"
#include "editor/editor_file_system.h"
#include "editor/import/3d/resource_importer_scene.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/mesh_data_tool.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/surface_tool.h"
#include "scene/resources/texture.h"
#include "vector_graphics_adaptive_renderer.h"
#include "vector_graphics_path.h"

class EditorSceneImporterSVG : public EditorSceneFormatImporter {
	GDCLASS(EditorSceneImporterSVG, EditorSceneFormatImporter);

	static Point2 compute_center(const tove::PathRef &p_path) {
		const float *bounds = p_path->getBounds();
		return Point2((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2);
	}

public:
	EditorSceneImporterSVG();
	~EditorSceneImporterSVG();
	virtual void get_extensions(List<String> *r_extensions) const override;
 	virtual Node *import_scene(const String &p_path, uint32_t p_flags, const HashMap<StringName, Variant> &p_options, List<String> *r_missing_deps, Error *r_err = nullptr) override;
};
#endif
#endif
