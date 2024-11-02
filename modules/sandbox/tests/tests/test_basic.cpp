#include "api.hpp"

// This works: it's being created during initialization
static Dictionary d = Dictionary::Create();

extern "C" Variant test_static_storage(Variant key, Variant val) {
	d[key] = val;
	return d;
}
extern "C" Variant test_failing_static_storage(Variant key, Variant val) {
	// This works only once: it's being created after initialization
	static Dictionary fd = Dictionary::Create();
	fd[key] = val;
	return fd;
}
static Dictionary fd = Dictionary::Create();
extern "C" Variant test_permanent_storage(Variant key, Variant val) {
	fd[key] = val;
	fd = Variant(fd).make_permanent();
	return fd;
}

static String ps = "Hello this is a permanent string";
extern "C" Variant test_permanent_string(String input) {
	ps = input;
	return ps;
}

static Array pa = Array::Create();
extern "C" Variant test_permanent_array(Array input) {
	pa = input;
	return pa;
}

static Dictionary pd = Dictionary::Create();
extern "C" Variant test_permanent_dict(Dictionary input) {
	pd = input;
	return pd;
}

extern "C" Variant test_check_if_permanent(String test) {
	if (test == "string") {
		printf("Checking if string %d is permanent\n", ps.get_variant_index());
		return ps.is_permanent();
	} else if (test == "array") {
		printf("Checking if array %d is permanent\n", pa.get_variant_index());
		return pa.is_permanent();
	} else if (test == "dict") {
		printf("Checking if dictionary %d is permanent\n", pd.get_variant_index());
		return pd.is_permanent();
	}
	return false;
}

extern "C" Variant test_infinite_loop() {
	while (true)
		;
}

extern "C" Variant test_recursive_calls(Node sandbox) {
	sandbox("vmcall", "test_recursive_calls", sandbox);
	return {};
}

extern "C" Variant public_function() {
	return "Hello from the other side";
}

extern "C" Variant test_ping_pong(Variant arg) {
	return arg;
}

extern "C" Variant test_ping_move_pong(Variant arg) {
	Variant v = std::move(arg);
	return v;
}

extern "C" Variant test_variant_eq(Variant arg1, Variant arg2) {
	return arg1 == arg2;
}

extern "C" Variant test_variant_neq(Variant arg1, Variant arg2) {
	return (arg1 != arg2) == false;
}

extern "C" Variant test_variant_lt(Variant arg1, Variant arg2) {
	return arg1 < arg2;
}

extern "C" Variant test_bool(bool arg) {
	return arg;
}

extern "C" Variant test_int(long arg) {
	return arg;
}

extern "C" Variant test_float(double arg) {
	return arg;
}

extern "C" Variant test_string(String arg) {
	return arg;
}

extern "C" Variant test_nodepath(NodePath arg) {
	return arg;
}

extern "C" Variant test_vec2(Vector2 arg) {
	Vector2 result = arg;
	return result;
}
extern "C" Variant test_vec2i(Vector2i arg) {
	Vector2i result = arg;
	return result;
}

extern "C" Variant test_vec3(Vector3 arg) {
	Vector3 result = arg;
	return result;
}
extern "C" Variant test_vec3i(Vector3i arg) {
	Vector3i result = arg;
	return result;
}

extern "C" Variant test_vec4(Vector4 arg) {
	Vector4 result = arg;
	return result;
}
extern "C" Variant test_vec4i(Vector4i arg) {
	Vector4i result = arg;
	return result;
}

extern "C" Variant test_color(Color arg) {
	Color result = arg;
	return result;
}

extern "C" Variant test_plane(Plane arg) {
	Plane result = arg;
	return result;
}

extern "C" Variant test_array(Array array) {
	array.push_back(2);
	array.push_back("4");
	array.push_back(6.0);
	if (array[0] != 2 || array[1] != "4" || array[2] != 6.0) {
		return "Fail";
	}
	if (!(array[0] == 2 && array[1] == "4" && array[2] == 6.0)) {
		return "Fail";
	}
	array[0] = 1;
	array[1] = "2";
	array[2] = 3.0;
	if (int(array[0]) != 1 || String(array[1]) != "2" || double(array[2]) != 3.0) {
		return "Fail";
	}
	if (int(array[0]) == 1 && String(array[1]) == "2" || double(array[2]) == 3.0) {
		return array;
	}
	return "Fail";
}

extern "C" Variant test_dict(Dictionary arg) {
	return arg;
}

extern "C" Variant test_sub_dictionary(Dictionary dict) {
	return Dictionary(dict)["1"];
}

extern "C" Variant test_rid(RID rid) {
	return rid;
}

extern "C" Variant test_object(Object arg) {
	Object result = arg;
	return result;
}

extern "C" Variant test_basis(Basis basis) {
	Basis b = basis;
	return b;
}

extern "C" Variant test_transform2d(Transform2D transform2d) {
	Transform2D t2d = transform2d;
	return t2d;
}

extern "C" Variant test_transform3d(Transform3D transform3d) {
	Transform3D t3d = transform3d;
	return t3d;
}

extern "C" Variant test_quaternion(Quaternion quaternion) {
	Quaternion q2 = quaternion;
	return q2;
}

extern "C" Variant test_callable(Callable callable) {
	return callable.call(1, 2, "3");
}

// clang-format off
extern "C" Variant test_create_callable() {
	Array array = Array::Create();
	array.push_back(1);
	array.push_back(2);
	array.push_back("3");
	return Callable::Create<Variant(Array, int, int, String)>([](Array array, int a, int b, String c) -> Variant {
		return a + b + std::stoi(c.utf8()) + int(array.at(0)) + int(array.at(1)) + std::stoi(array.at(2).as_string().utf8());
	}, array);
}
// clang-format on

extern "C" Variant test_pa_u8(PackedArray<uint8_t> arr) {
	return PackedArray<uint8_t> (arr.fetch());
}
extern "C" Variant test_pa_f32(PackedArray<float> arr) {
	return PackedArray<float> (arr.fetch());
}
extern "C" Variant test_pa_f64(PackedArray<double> arr) {
	return PackedArray<double> (arr.fetch());
}
extern "C" Variant test_pa_i32(PackedArray<int32_t> arr) {
	return PackedArray<int32_t> (arr.fetch());
}
extern "C" Variant test_pa_i64(PackedArray<int64_t> arr) {
	return PackedArray<int64_t> (arr.fetch());
}
extern "C" Variant test_pa_vec2(PackedArray<Vector2> arr) {
	return PackedArray<Vector2> (arr.fetch());
}
extern "C" Variant test_pa_vec3(PackedArray<Vector3> arr) {
	return PackedArray<Vector3> (arr.fetch());
}
extern "C" Variant test_pa_color(PackedArray<Color> arr) {
	return PackedArray<Color> (arr.fetch());
}
extern "C" Variant test_pa_string(PackedArray<std::string> arr) {
	return PackedArray<std::string> (arr.fetch());
}

extern "C" Variant test_create_pa_u8() {
	PackedArray<uint8_t> arr({ 1, 2, 3, 4 });
	return arr;
}
extern "C" Variant test_create_pa_f32() {
	PackedArray<float> arr({ 1, 2, 3, 4 });
	return arr;
}
extern "C" Variant test_create_pa_f64() {
	PackedArray<double> arr({ 1, 2, 3, 4 });
	return arr;
}
extern "C" Variant test_create_pa_i32() {
	PackedArray<int32_t> arr({ 1, 2, 3, 4 });
	return arr;
}
extern "C" Variant test_create_pa_i64() {
	PackedArray<int64_t> arr({ 1, 2, 3, 4 });
	return arr;
}
extern "C" Variant test_create_pa_vec2() {
	PackedArray<Vector2> arr({ { 1, 1 }, { 2, 2 }, { 3, 3 } });
	return arr;
}
extern "C" Variant test_create_pa_vec3() {
	PackedArray<Vector3> arr({ { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } });
	return arr;
}
extern "C" Variant test_create_pa_color() {
	PackedArray<Color> arr({ { 0, 0, 0, 0 }, { 1, 1, 1, 1 } });
	return arr;
}
extern "C" Variant test_create_pa_string() {
	PackedArray<std::string> arr({ "Hello", "from", "the", "other", "side" });
	return arr;
}

extern "C" Variant test_exception() {
	asm("unimp");
	__builtin_unreachable();
}

static bool timer_got_called = false;
extern "C" Variant test_timers() {
	long val1 = 11;
	float val2 = 22.0f;
	return CallbackTimer::native_periodic(0.01, [=](Node timer) -> Variant {
		print("Timer with values: ", val1, val2);
		timer.queue_free();
		timer_got_called = true;
		return {};
	});
}
extern "C" Variant verify_timers() {
	return timer_got_called;
}

extern "C" Variant call_method(Variant v, Variant vmethod, Variant vargs) {
	std::string method = vmethod.as_std_string();
	Array args_array = vargs.as_array();
	std::vector<Variant> args = args_array.to_vector();
	Variant ret;
	v.callp(method, args.data(), args.size(), ret);
	return ret;
}

extern "C" Variant voidcall_method(Variant v, Variant vmethod, Variant vargs) {
	std::string method = vmethod.as_std_string();
	Array args_array = vargs.as_array();
	std::vector<Variant> args = args_array.to_vector();
	v.voidcallp(method, args.data(), args.size());
	return Nil;
}

extern "C" Variant access_a_parent(Node n) {
	Node p = n.get_parent();
	return Nil;
}

extern "C" Variant creates_a_node() {
	return Node::Create("test");
}

extern "C" Variant free_self() {
	get_node()("free");
	return Nil;
}

extern "C" Variant access_an_invalid_child_node() {
	Node n = Node::Create("test");
	Node c = Node::Create("child");
	n.add_child(c);
	c("free");
	c.set_name("child2");
	return c;
}

extern "C" Variant access_an_invalid_child_resource(String path) {
	Variant resource = loadv(path.utf8());
	return resource.method_call("instantiate");
}

extern "C" Variant disable_restrictions() {
	get_node().call("disable_restrictions");
	return Nil;
}

extern "C" Variant test_property_proxy() {
	Node node = Node::Create("Fail 1");
	node.name() = "Fail 1.5";
	node.set_name("Fail 2");
	if (node.get_name() == "Fail 2") {
		node.set("name", "Fail 3");
		if (node.get("name") == "Fail 3") {
			node.name() = "TestOK";
			if (node.name() != "TestOK") {
				return "Fail 4";
			}
		}
	}
	return node.get_name();
}
