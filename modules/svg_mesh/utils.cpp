/*************************************************************************/
/*  utils.cpp   	                                                     */
/*************************************************************************/

#include "utils.h"
#include "core/string/string_builder.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/surface_tool.h"

#include "modules/svg_mesh/thirdparty/tove2d/src/cpp/mesh/mesh.h"
#include "modules/svg_mesh/thirdparty/tove2d/src/cpp/mesh/meshifier.h"
#include "modules/svg_mesh/thirdparty/tove2d/src/cpp/shader/feed/color_feed.h"

tove::PathRef new_transformed_path(const tove::PathRef &p_tove_path, const Transform2D &p_transform) {
	const Vector2 &tx = p_transform.columns[0];
	const Vector2 &ty = p_transform.columns[1];
	const Vector2 &to = p_transform.columns[2];
	tove::nsvg::Transform tove_transform(tx.x, ty.x, to.x, tx.y, ty.y, to.y);
	tove::PathRef tove_path = tove::tove_make_shared<tove::Path>();
	tove_path->set(p_tove_path, tove_transform);
	tove_path->setIndex(p_tove_path->getIndex());
	return tove_path;
}

Ref<ShaderMaterial> copy_mesh(
		Ref<ArrayMesh> &p_mesh,
		tove::MeshRef &p_tove_mesh,
		const tove::GraphicsRef &p_graphics,
		Ref<Texture> &r_texture,
		bool p_spatial) {

	const int n = p_tove_mesh->getVertexCount();
	if (n < 1) {
		return Ref<ShaderMaterial>();
	}

	const bool isPaintMesh = std::dynamic_pointer_cast<tove::PaintMesh>(p_tove_mesh).get() != nullptr;

	const int stride = isPaintMesh ? sizeof(float) * 3 : sizeof(float) * 2 + 4;
	const int size = n * stride;

	Vector<uint8_t> vvertices;
	vvertices.resize(size);
	p_tove_mesh->copyVertexData(vvertices.ptrw(), size);

	const int index_count = p_tove_mesh->getIndexCount();
	Vector<ToveVertexIndex> vindices;
	vindices.resize(index_count);
	p_tove_mesh->copyIndexData(vindices.ptrw(), index_count);

	Vector<int> iarr;
	ERR_FAIL_COND_V(iarr.resize(index_count) != OK, Ref<ShaderMaterial>());
	{
		for (int i = 0; i < index_count; i++) {
			iarr.write[i] = vindices[i];
		}
	}

	Vector<Vector3> varr;
	ERR_FAIL_COND_V(varr.resize(n) != OK, Ref<ShaderMaterial>());

	{
		for (int i = 0; i < n; i++) {
			float *p = (float *)(vvertices.ptrw() + i * stride);
			varr.write[i] = Vector3(p[0], p[1], 0) * Vector3(0.001, -0.001, 0.001);
		}
	}

	Vector<Color> carr;
	if (!isPaintMesh) {
		ERR_FAIL_COND_V(carr.resize(n) != OK, Ref<ShaderMaterial>());
		for (int i = 0; i < n; i++) {
			uint8_t *p = vvertices.ptrw() + i * stride + 2 * sizeof(float);
			carr.write[i] = Color(p[0] / 255.0, p[1] / 255.0, p[2] / 255.0, p[3] / 255.0).srgb_to_linear();
		}
	}

	Vector<Vector2> uvs;
	Ref<Material> material;

	if (isPaintMesh) {
		auto feed = tove::tove_make_shared<tove::ColorFeed>(p_graphics, 1.0f);
		TovePaintColorAllocation alloc = feed->getColorAllocation();
		const int matrix_rows = 3;
		const int npaints = alloc.numPaints;

		Vector<float> matrix_data;
		ERR_FAIL_COND_V(matrix_data.resize(alloc.numPaints * 3 * matrix_rows) != OK, Ref<ShaderMaterial>());
		for (int i = 0; i < alloc.numPaints * 3 * matrix_rows; i++) {
			matrix_data.write[i] = 0.0f;
		}

		Vector<float> arguments_data;
		ERR_FAIL_COND_V(arguments_data.resize(alloc.numPaints) != OK, Ref<ShaderMaterial>());
		for (int i = 0; i < alloc.numPaints; i++) {
			arguments_data.write[i] = 0.0f;
		}

		Vector<uint8_t> pixels;
		ERR_FAIL_COND_V(pixels.resize(npaints * 4 * alloc.numColors) != OK, Ref<ShaderMaterial>());

		ToveGradientData gradientData;

		memset(pixels.ptrw(), 0, npaints * 4 * alloc.numColors);

		gradientData.matrix = matrix_data.ptrw();
		gradientData.matrixRows = matrix_rows;
		gradientData.arguments = arguments_data.ptrw();
		gradientData.colorsTexture = pixels.ptrw();
		gradientData.colorsTextureRowBytes = npaints * 4;
		gradientData.colorsTextureHeight = alloc.numColors;

		feed->bind(gradientData);
		feed->beginUpdate();
		feed->endUpdate();

		Vector<uint8_t> paint_seen;
		ERR_FAIL_COND_V(paint_seen.resize(npaints) != OK, Ref<ShaderMaterial>());
		memset(paint_seen.ptrw(), 0, npaints);

		ERR_FAIL_COND_V(uvs.resize(n) != OK, Ref<ShaderMaterial>());
		{
			for (int i = 0; i < n; i++) {
				const float *p = (float *)(vvertices.ptrw() + i * stride);
				int paint_index = p[2];
				uvs.write[i] = Vector2((paint_index + 0.5f) / npaints, 0.0f);
				paint_seen.write[paint_index] = 1;
			}
		}

		Ref<Image> image = memnew(Image(
				alloc.numPaints,
				alloc.numColors,
				false,
				Image::FORMAT_RGBA8,
				pixels));

		Ref<ImageTexture> texture = Ref<ImageTexture>(memnew(ImageTexture));
		texture->create_from_image(image);
		r_texture = texture;

		Ref<Shader> shader;
		shader.instantiate();

		StringBuilder code;
		String s;
		for (int32_t paint_i = 0; paint_i < alloc.numPaints; paint_i++) {
			bool has_seen = paint_seen[paint_i];
			if (!has_seen) {
				continue;
			}

			code += "if(i==";
			s = String::num_int64(paint_i);
			code += s;
			code += "){";

			const float &v = arguments_data.write[paint_i];
			code += "a=";
			s = String::num_real(v);
			if (!s.is_numeric()) {
				s = "0.0";
			}
			code += s;
			code += ";";

			const int j0 = paint_i * 3 * matrix_rows;
			code += "m=mat3(";

			for (int j = 0; j < 3; j++) {
				if (j > 0) {
					code += "),";
				}
				code += "vec3(";
				for (int k = 0; k < 3; k++) {
					if (k > 0) {
						code += ",";
					}
					float elem = matrix_data[j0 + j + k * 3];
					if (!Math::is_nan(elem)) {
						s = String::num_real(elem);
					} else {
						s = "0.0f";
					}
					if (!s.is_numeric()) {
						s = "0.0";
					}
					code += s;
				}
			}
			code += "));}\n";
		}
		if (!p_spatial) {
			// clang-format off
        String shader_code = String(R"GLSL(
shader_type canvas_item;

varying smooth mediump vec2 gradient_pos;
varying flat mediump vec3 gradient_scale;
varying flat mediump float paint;

void vertex()
{
    int i = int(floor(UV.x * NPAINTS));
    float a;
    mat3 m;

)GLSL") +  code.as_string() + String(R"GLSL(

	gradient_pos = (m * vec3(VERTEX.xy, 1)).xy;
	gradient_scale = vec3(CSTEP, 1.0f - 2.0f * CSTEP, a);

	paint = UV.x;
}

void fragment()
{
	float y = mix(gradient_pos.y, length(gradient_pos), gradient_scale.z);
	y = gradient_scale.x + gradient_scale.y * y;

	vec2 texture_pos_exact = vec2(paint, y);
	COLOR = texture(TEXTURE, texture_pos_exact);
}

)GLSL");
					// clang-format on
					String npaints_str = String::num_real(npaints);
			String cstep_str = String::num_real(0.5f / alloc.numColors);

			shader_code = shader_code.replace("NPAINTS", npaints_str.utf8().get_data());
			shader_code = shader_code.replace("CSTEP", cstep_str.utf8().get_data());
			shader->set_code(shader_code);

			Ref<ShaderMaterial> shader_material;
			shader_material.instantiate();
			shader_material->set_shader(shader);
			material = shader_material;
		} else {
			// clang-format off
        String shader_code = String(R"GLSL(shader_type spatial;

varying smooth highp vec3 gradient_pos;
varying flat highp vec3 gradient_scale;
varying flat highp float paint;
uniform sampler2D tex : hint_albedo;
render_mode cull_disabled;

void vertex()
{
    int i = int(floor(UV.x * NPAINTS));
    float a;
    mat3 m;

)GLSL") +  code.as_string() + String(R"GLSL(

	gradient_pos = m * VERTEX.xyz;
	gradient_scale = vec3(CSTEP, 1.0f - 2.0f * CSTEP, a);

	paint = UV.x;
}

void fragment()
{
	float y = mix(gradient_pos.y, length(gradient_pos), gradient_scale.z);
	y = gradient_scale.x + gradient_scale.y * y;

	vec2 texture_pos_exact = vec2(paint, -y);
	ALBEDO = textureLod(tex, texture_pos_exact, 0).rgb;
	ALPHA = textureLod(tex, texture_pos_exact, 0).a;
}

)GLSL");
					// clang-format on
					String npaints_str = String::num_real(npaints);
			String cstep_str = String::num_real(0.5f / alloc.numColors);

			shader_code = shader_code.replace("NPAINTS", npaints_str.utf8().get_data());
			shader_code = shader_code.replace("CSTEP", cstep_str.utf8().get_data());
			shader->set_code(shader_code);

			Ref<ShaderMaterial> mat;
			mat.instantiate();
			mat->set_shader_parameter("tex", texture);
			mat->set_shader(shader);
			material = mat;
		}
	}

	Array arr;
	ERR_FAIL_COND_V(arr.resize(Mesh::ARRAY_MAX) != OK, Ref<ShaderMaterial>());
	arr[Mesh::ARRAY_VERTEX] = varr;
	arr[Mesh::ARRAY_INDEX] = iarr;
	if (carr.size() > 0) {
		arr[Mesh::ARRAY_COLOR] = carr;
	}
	if (uvs.size() > 0) {
		arr[RS::ARRAY_TEX_UV] = uvs;
	}

	Ref<SurfaceTool> surface_tool;
	surface_tool.instantiate();
	surface_tool->create_from_triangle_arrays(arr);
	surface_tool->index();
	surface_tool->generate_normals();
	surface_tool->generate_tangents();

	p_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, surface_tool->commit_to_arrays());
	return material;
}
