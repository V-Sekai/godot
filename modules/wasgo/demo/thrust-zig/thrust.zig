const WasGo = @cImport({
    @cInclude("errno.h");
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
    @cInclude("wasgo/wasgo.h");
    @cInclude("Spatial.h");
    @cInclude("Transform.h");
    @cInclude("InputEventKey.h");
});

const print = @import("std").debug.print;

var velocity: f32 = 0;
var frame_counter: i32 = 0;
var thrust: f32 = 0;
var thrust_limit: f32 = 50;
var thrust_delta: f32 = 0;
var gravity_influence: f32 = 100;
var smooth: f32 = 0.9;

export fn _ready() void {
    // thrust_limit = WasGo.get_property_float("thrust_limit");
    // gravity_influence = WasGo.get_property_float("gravity_influence");
    // smooth = WasGo.get_property_float("smoothness");
}

// export fn _unhandled_key_input(p_key_event: WasGo.InputEventKey) void {
//     p_key_event.get_scancode();
//     switch (p_key_event.get_scancode()) {
//         KEY_W => {
//             if (p_key_event.is_pressed()) {
//                 thrust_delta = 10;
//             } else {
//                 thrust_delta = 0;
//             }
//             break;
//         },
//         KEY_S => {
//             if (p_key_event.is_pressed()) {
//                 thrust_delta = -10;
//             } else {
//                 thrust_delta = 0;
//             }
//             break;
//         },

//         else => break,
//     }
// }

export fn _physics_process(delta: f32) void {
    // var y_ang: f32 = 0;
    // var plane: WasGo.Spatial = {};
    // var direction: WasGo.Vector3 = {};
    // var target_vector: WasGo.Vector3 = {};
    // var t: WasGo.Transform = {};
    // var target_speed: f32 = 0;

    // var this_node: Node = WasGo.this_node();
    // var path: Variant = WasGo.get_property_nodepath("other_node");
    // plane = @ptrCast(Spatial, this_node.get_node(path));

    // var b: Basis = plane.get_transform().get_basis();
    // direction = b.get_axis(2);
    // y_ang = direction.angle_to(Vector3(0, 1, 0));
    // if ((thrust <= thrust_limit and thrust_delta > 0) || (thrust >= 0 and thrust_delta < 0)) {
    //     thrust += thrust_delta;
    // }
    // target_speed = thrust + (y_ang / Math_PI - 0.5) * gravity_influence; //we subtract 0.5 from the angle over pi because gravity's influence should be zero at 90 degrees or when the angle over pi is 0.5
    // velocity = (velocity * smooth + target_speed * (1 - smooth)) / 2;
    // target_vector = Vector3(0, 0, velocity);
    // t = plane.get_transform();
    // t.translate(target_vector * delta);
    // plane.set_transform(t);
    // frame_counter = @rem(frame_counter + 1, 120);
    // if (frame_counter == 0) {
    // WasGo.printf("thrust: %d\n", int(thrust * 1000));
    // WasGo.printf("y_ang: %d\n", int(y_ang * 1000));
    // WasGo.printf("velocity: %d\n", int(velocity * 1000));
    // }
}
