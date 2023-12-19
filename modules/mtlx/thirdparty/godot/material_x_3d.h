#ifndef MATERIAL_X_H
#define MATERIAL_X_H

#include "MaterialXCore/Generated.h"

#include "core/io/resource_importer.h"
#include "scene/resources/material.h"

// #include <MaterialXRenderGlsl/GLTextureHandler.h>
// #include <MaterialXRenderGlsl/GLUtil.h>
// #include <MaterialXRenderGlsl/TextureBaker.h>

#include <MaterialXRender/Harmonics.h>
#include <MaterialXRender/OiioImageLoader.h>
#include <MaterialXRender/StbImageLoader.h>
#include <MaterialXRender/Util.h>

#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/ShaderTranslator.h>

#include <MaterialXFormat/Environ.h>
#include <MaterialXFormat/Util.h>

#include <MaterialXCore/Util.h>

// #include <MaterialXGenGlsl/GlslShaderGenerator.h>
// #include <MaterialXGenShader/UnitSystem.h>
// #include <MaterialXRenderGlsl/GlslProgram.h>

#include <MaterialXGenGlsl/EsslShaderGenerator.h>

#include <iostream>

using namespace godot;
namespace mx = MaterialX;
class MTLXLoader : public Resource {
	GDCLASS(MTLXLoader, Resource);
    mx::DocumentPtr _stdLib;
	// mx::GlslRendererPtr _renderer;
protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("_load", "path", "original_path", "use_sub_threads", "cache_mode"), &MTLXLoader::_load);
	}

public:
	virtual Variant _load(const String &p_save_path, const String &p_original_path, bool p_use_sub_threads, int64_t p_cache_mode) const;
	MTLXLoader();
};

using MaterialPtr = std::shared_ptr<class Material>;

class DocumentModifiers {
public:
	mx::StringMap remapElements;
	mx::StringSet skipElements;
	std::string filePrefixTerminator;
};
#endif