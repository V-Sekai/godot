shader_type spatial;

render_mode unshaded;
render_mode world_vertex_coords;

uniform vec3 _FocusPoint = vec3(0,0,0);
uniform float _NearFade = 1.0;
uniform float _FarFade = 3.0;
uniform float _FadeZone = .1;

vec3 xform(mat4 m, vec3 p, const int w){return (m * vec4(p,float(w))).xyz;}

float computeOpacity(mat4 worldToView, vec3 worldSpacePos, float baseOpacity)
{
    // Compare world space input position with focus point world space position
    float focus_point_distance = distance(worldSpacePos, _FocusPoint);
    float opacity = baseOpacity;
    if (focus_point_distance > _FarFade)
    {
        opacity = clamp(1.0 + (_FarFade - focus_point_distance) / _FadeZone, 0.0, 1.0) * baseOpacity;
    }

    float depth = xform(worldToView, worldSpacePos, 1).z * -1.0;
    if (depth < _NearFade)
    {
        opacity = clamp(-_FadeZone + depth / _NearFade, 0.0, 1.0) * baseOpacity;
    }

    return opacity;
}

void vertex(){
	COLOR.a = computeOpacity(VIEW_MATRIX, VERTEX, 1.0);
}

void fragment() {
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
}
