#include "oit_resolve_tiles.glsl.gen.h"

class OitResolveTilesShaderRD : public RendererRD::ShaderRD {
public:
	static const char *_vertex_code;
	static const char *_fragment_code;
	static const char *_compute_code;

	OitResolveTilesShaderRD() {
		_setup(_vertex_code, _fragment_code, _compute_code, "oit_resolve_tiles");
	}

	virtual String _get_custom_defines() const {
		return "";
	}

	virtual void _bind_params(const RDShaderInstance *p_shaders) const {

	}
};

const char *OitResolveTilesShaderRD::_vertex_code = nullptr;
const char *OitResolveTilesShaderRD::_fragment_code = nullptr;
const char *OitResolveTilesShaderRD::_compute_code = nullptr;
