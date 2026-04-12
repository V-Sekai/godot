/**************************************************************************/
/*  sandbox_globals.cpp                                                   */
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

#include "sandbox.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/debugger/engine_debugger.h"
#include "core/extension/gdextension_manager.h"
#include "core/input/input.h"
#include "core/input/input_map.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/io/resource_uid.h"
#include "core/math/geometry_2d.h"
#include "core/math/geometry_3d.h"
#include "core/object/worker_thread_pool.h"
#include "core/os/time.h"
#include "core/string/translation.h"
#include "core/string/translation_server.h"
#include "main/performance.h"
#include "scene/theme/theme_db.h"
#include "servers/audio/audio_server.h"
#include "servers/display/display_server.h"
#include "servers/display/native_menu.h"
#include "servers/navigation_2d/navigation_server_2d.h"
#include "servers/navigation_3d/navigation_server_3d.h"
#include "servers/physics_2d/physics_server_2d.h"
#include "servers/physics_3d/physics_server_3d.h"
#include "servers/rendering/rendering_server.h"
#include "servers/text/text_server.h"
#include "servers/xr/xr_server.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_interface.h"
#endif

#ifdef MODULE_NAVIGATION_ENABLED
#include "modules/navigation/3d/navigation_mesh_generator.h"
#endif
#include <functional>
#include <unordered_map>

namespace riscv {
std::unordered_map<std::string, std::function<uint64_t()>> global_singleton_list;

static uint64_t get_audio_server() { return uint64_t(uintptr_t(AudioServer::get_singleton())); }
#ifdef TOOLS_ENABLED
static uint64_t get_editor_interface() { return uint64_t(uintptr_t(EditorInterface::get_singleton())); }
#endif
static uint64_t get_display_server() { return uint64_t(uintptr_t(DisplayServer::get_singleton())); }
static uint64_t get_gdextension_manager() { return uint64_t(uintptr_t(GDExtensionManager::get_singleton())); }
static uint64_t get_engine_debugger() { return uint64_t(uintptr_t(EngineDebugger::get_singleton())); }
static uint64_t get_engine() { return uint64_t(uintptr_t(Engine::get_singleton())); }
static uint64_t get_input() { return uint64_t(uintptr_t(Input::get_singleton())); }
static uint64_t get_input_map() { return uint64_t(uintptr_t(InputMap::get_singleton())); }
static uint64_t get_native_menu() { return uint64_t(uintptr_t(NativeMenu::get_singleton())); }
static uint64_t get_navigation_server_2d() { return uint64_t(uintptr_t(NavigationServer2D::get_singleton())); }
static uint64_t get_navigation_server_3d() { return uint64_t(uintptr_t(NavigationServer3D::get_singleton())); }
static uint64_t get_performance() { return uint64_t(uintptr_t(Performance::get_singleton())); }
static uint64_t get_physics_server_2d() { return uint64_t(uintptr_t(PhysicsServer2D::get_singleton())); }
static uint64_t get_physics_server_3d() { return uint64_t(uintptr_t(PhysicsServer3D::get_singleton())); }
static uint64_t get_physics_server_2d_manager() { return uint64_t(uintptr_t(PhysicsServer2DManager::get_singleton())); }
static uint64_t get_physics_server_3d_manager() { return uint64_t(uintptr_t(PhysicsServer3DManager::get_singleton())); }
static uint64_t get_project_settings() { return uint64_t(uintptr_t(ProjectSettings::get_singleton())); }
static uint64_t get_rendering_server() { return uint64_t(uintptr_t(RenderingServer::get_singleton())); }
static uint64_t get_resource_uid() { return uint64_t(uintptr_t(ResourceUID::get_singleton())); }
static uint64_t get_text_server_manager() { return uint64_t(uintptr_t(TextServerManager::get_singleton())); }
static uint64_t get_theme_db() { return uint64_t(uintptr_t(ThemeDB::get_singleton())); }
static uint64_t get_time() { return uint64_t(uintptr_t(Time::get_singleton())); }
static uint64_t get_translation_server() { return uint64_t(uintptr_t(TranslationServer::get_singleton())); }
static uint64_t get_worker_thread_pool() { return uint64_t(uintptr_t(WorkerThreadPool::get_singleton())); }
static uint64_t get_xr_server() { return uint64_t(uintptr_t(XRServer::get_singleton())); }

void init_global_singleton_list() {
	global_singleton_list.emplace("AudioServer", get_audio_server);
#ifdef TOOLS_ENABLED
	global_singleton_list.emplace("EditorInterface", get_editor_interface);
#endif
	global_singleton_list.emplace("DisplayServer", get_display_server);
	global_singleton_list.emplace("GDExtensionManager", get_gdextension_manager);
	global_singleton_list.emplace("EngineDebugger", get_engine_debugger);
	global_singleton_list.emplace("Engine", get_engine);
	global_singleton_list.emplace("Input", get_input);
	global_singleton_list.emplace("InputMap", get_input_map);
	global_singleton_list.emplace("NativeMenu", get_native_menu);
	global_singleton_list.emplace("NavigationServer2D", get_navigation_server_2d);
	global_singleton_list.emplace("NavigationServer3D", get_navigation_server_3d);
	global_singleton_list.emplace("Performance", get_performance);
	global_singleton_list.emplace("PhysicsServer2D", get_physics_server_2d);
	global_singleton_list.emplace("PhysicsServer3D", get_physics_server_3d);
	global_singleton_list.emplace("PhysicsServer2DManager", get_physics_server_2d_manager);
	global_singleton_list.emplace("PhysicsServer3DManager", get_physics_server_3d_manager);
	global_singleton_list.emplace("ProjectSettings", get_project_settings);
	global_singleton_list.emplace("RenderingServer", get_rendering_server);
	global_singleton_list.emplace("ResourceUID", get_resource_uid);
	global_singleton_list.emplace("TextServerManager", get_text_server_manager);
	global_singleton_list.emplace("ThemeDB", get_theme_db);
	global_singleton_list.emplace("Time", get_time);
	global_singleton_list.emplace("TranslationServer", get_translation_server);
	global_singleton_list.emplace("WorkerThreadPool", get_worker_thread_pool);
	global_singleton_list.emplace("XRServer", get_xr_server);
}
} // namespace riscv
