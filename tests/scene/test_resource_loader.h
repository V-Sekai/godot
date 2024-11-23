/**************************************************************************/
/*  test_resource_loader.h                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef TEST_RESOURCE_LOADER_H
#define TEST_RESOURCE_LOADER_H

#include "core/io/resource_loader.h"
#include "scene/resources/gradient_texture.h"

#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestResourceLoader {

void init(const String &p_test, const String &p_copy_target = String()) {
	Error err;
	// Setup project settings since it's needed for the import process.
	String project_folder = TestUtils::get_temp_path(p_test.get_file().get_basename());
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	da->make_dir_recursive(project_folder.path_join(".godot").path_join("imported"));
	// Initialize res:// to `project_folder`.
	TestProjectSettingsInternalsAccessor::resource_path() = project_folder;
	err = ProjectSettings::get_singleton()->setup(project_folder, String(), true);

	if (p_copy_target.is_empty()) {
		return;
	}

	// Copy all the necessary test data files to the res:// directory.
	da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	String test_data = String("tests/data").path_join(p_test);
	da = DirAccess::open(test_data);
	CHECK_MESSAGE(da.is_valid(), "Unable to open folder.");
	da->list_dir_begin();
	for (String item = da->get_next(); !item.is_empty(); item = da->get_next()) {
		if (!FileAccess::exists(test_data.path_join(item))) {
			continue;
		}
		Ref<FileAccess> output = FileAccess::open(p_copy_target.path_join(item), FileAccess::WRITE, &err);
		CHECK_MESSAGE(err == OK, "Unable to open output file.");
		output->store_buffer(FileAccess::get_file_as_bytes(test_data.path_join(item)));
		output->close();
	}
	da->list_dir_end();
}

TEST_CASE("[SceneTree][ResourceLoader] Load Resource - Load invalid path") {
	init("null_path");
	Ref<Resource> resource = ResourceLoader::load("");
	CHECK_FALSE(resource.is_valid());
}

TEST_CASE("[SceneTree][ResourceLoader] Load Text Resource - Load valid path") {
	init("load_resource_golden_path", "res://");
	Ref<Texture2D> resource = ResourceLoader::load("authorized_resource.tres", "Texture2D");
	CHECK(resource.is_valid());
}

TEST_CASE("[SceneTree][ResourceLoader] Load Text Resource - Malicious Path") {
	init("load_resource_malicious_path", "res://");
	Ref<Texture2D> resource = ResourceLoader::load("trojan_resource.tres", "Texture2D");
	CHECK(resource.is_valid());
}

TEST_CASE("[SceneTree][ResourceLoader] Load Binary Resource - Load valid path") {
	init("load_resource_golden_path", "res://");
	Ref<Texture2D> resource = ResourceLoader::load("authorized_resource.res", "Texture2D");
	CHECK(resource.is_valid());
}

TEST_CASE("[SceneTree][ResourceLoader] Load Binary Resource - Malicious Path") {
	init("load_resource_malicious_path", "res://");
	Ref<Texture2D> resource = ResourceLoader::load("trojan_resource.res", "Texture2D");
	CHECK(resource.is_valid());
}

TEST_CASE("[SceneTree][ResourceLoader] Load Text Resource Whitelisted - Load invalid path") {
	init("null_path");
	Dictionary ext_whitelist;
	Dictionary type_whitelist;
	Ref<Resource> resource = ResourceLoader::load_whitelisted("", ext_whitelist, type_whitelist);
	CHECK_FALSE(resource.is_valid());
}

TEST_CASE("[SceneTree][ResourceLoader] Load Text Resource Whitelisted - Load valid path") {
	init("load_resource_whitelisted_golden_path", "res://");
	Dictionary ext_whitelist;
	ext_whitelist["res://authorized_resource.tres"] = true;
	ext_whitelist["res://authorized_resource.tres::Gradient_m6m553"] = true;
	Dictionary type_whitelist;
	type_whitelist["GradientTexture2D"] = true;
	type_whitelist["Gradient"] = true;
	Ref<Texture2D> resource = ResourceLoader::load_whitelisted(
			"res://authorized_resource.tres", ext_whitelist, type_whitelist, "Texture2D");
	CHECK(resource.is_valid());
	REQUIRE_EQ(resource->get_class_static(), "Texture2D");
}

TEST_CASE("[SceneTree][ResourceLoader] Load Text Resource Whitelisted - No allowed paths in the whitelist") {
	init("load_resource_whitelisted_malicious_path", "res://");
	Dictionary ext_whitelist;
	Dictionary type_whitelist;
	type_whitelist["GradientTexture2D"] = true;
	type_whitelist["Gradient"] = true;
	Ref<Texture2D> resource = ResourceLoader::load_whitelisted(
			"res://trojan_resource.tres", ext_whitelist, type_whitelist, "Texture2D");
	CHECK_FALSE(resource.is_valid());
}

TEST_CASE("[SceneTree][ResourceLoader] Load Binary Resource Whitelisted - Load valid path") {
	init("load_resource_whitelisted_golden_path", "res://");
	Dictionary ext_whitelist;
	ext_whitelist["res://authorized_resource.res"] = true;
	ext_whitelist["res://authorized_resource.res::Gradient_m6m553"] = true;
	Dictionary type_whitelist;
	type_whitelist["GradientTexture2D"] = true;
	type_whitelist["Gradient"] = true;
	Ref<Texture2D> resource = ResourceLoader::load_whitelisted(
			"res://authorized_resource.res", ext_whitelist, type_whitelist, "Texture2D");
	CHECK(resource.is_valid());
	REQUIRE_EQ(resource->get_class_static(), "Texture2D");
}

TEST_CASE("[SceneTree][ResourceLoader] Load Binary Resource Whitelisted - No allowed paths in the whitelist") {
	init("load_resource_whitelisted_malicious_path", "res://");
	Dictionary ext_whitelist;
	Dictionary type_whitelist;
	type_whitelist["GradientTexture2D"] = true;
	type_whitelist["Gradient"] = true;
	Ref<Texture2D> resource = ResourceLoader::load_whitelisted(
			"res://trojan_resource.res", ext_whitelist, type_whitelist, "Texture2D");
	CHECK_FALSE(resource.is_valid());
}

} // namespace TestResourceLoader

#endif // TEST_RESOURCE_LOADER_H
