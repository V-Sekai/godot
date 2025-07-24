/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "core/object/class_db.h"
#include "video_stream_mkv.h"
#include "video_stream_av1.h"
#include "rendering_device_video_extensions.h"

#ifdef VULKAN_ENABLED
#include "vulkan_video_context.h"
#endif

static Ref<ResourceFormatLoaderMKV> resource_loader_mkv;
static Ref<ResourceFormatLoaderAV1> resource_loader_av1;

void initialize_vk_video_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	// Register MKV support
	resource_loader_mkv.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_mkv, true);

	// Register AV1 support
	resource_loader_av1.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_av1, true);

	GDREGISTER_CLASS(VideoStreamMKV);
	GDREGISTER_CLASS(VideoStreamAV1);
	GDREGISTER_CLASS(VideoStreamPlaybackAV1);
	GDREGISTER_CLASS(RenderingDeviceVideoExtensions);
#ifdef VULKAN_ENABLED
	GDREGISTER_CLASS(VulkanVideoContext);
#endif

}

void uninitialize_vk_video_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	ResourceLoader::remove_resource_format_loader(resource_loader_mkv);
	resource_loader_mkv.unref();
	
	ResourceLoader::remove_resource_format_loader(resource_loader_av1);
	resource_loader_av1.unref();
}
