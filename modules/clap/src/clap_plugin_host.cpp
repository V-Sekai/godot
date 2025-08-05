/**************************************************************************/
/*  clap_plugin_host.cpp                                                  */
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

#include "clap_plugin_host.h"
#include "core/input/input.h"
#include "core/math/audio_frame.h"
#include "core/os/os.h"
#include "core/string/print_string.h"

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/vm_map.h>
#endif

ClapPluginHost::ClapPluginHost() :
		BaseHost("Godot Clap Host", // name
				"V-Sekai", // vendor
				"0.1.0", // version
				"https://github.com/V-Sekai/godot" // url
				),
		_plugin_entry(nullptr),
		_plugin_factory(nullptr),
		_lib_handle(nullptr) {
}

ClapPluginHost::~ClapPluginHost() {
	// Clean up plugin first
	_plugin.reset();

	// Deinitialize plugin entry if we have one
	if (_plugin_entry) {
		_plugin_entry->deinit();
		_plugin_entry = nullptr;
	}

	// Close the library handle using Godot's API
	if (_lib_handle) {
		Error err = OS::get_singleton()->close_dynamic_library(_lib_handle);
		if (err != OK) {
			print_line("CLAP: Warning - failed to close dynamic library: " + String::num(err));
		}
		_lib_handle = nullptr;
	}
}

// clap_host callbacks
void ClapPluginHost::requestRestart() noexcept {
	print_line("CLAP: Requesting restart");
}

void ClapPluginHost::requestProcess() noexcept {
	print_line("CLAP: Requesting process");
}

void ClapPluginHost::requestCallback() noexcept {
	print_line("CLAP: Requesting callback");
}

// clap_host_gui
bool ClapPluginHost::implementsGui() const noexcept {
	return true; // Enable GUI support
}

void ClapPluginHost::guiResizeHintsChanged() noexcept {
	print_line("CLAP: GUI resize hints changed");
}

bool ClapPluginHost::guiRequestResize(uint32_t width, uint32_t height) noexcept {
	print_line("CLAP: GUI request resize to " + String::num(width) + "x" + String::num(height));
	return true; // Accept resize requests
}

bool ClapPluginHost::guiRequestShow() noexcept {
	print_line("CLAP: GUI request show");
	return true; // Accept show requests
}

bool ClapPluginHost::guiRequestHide() noexcept {
	print_line("CLAP: GUI request hide");
	return true; // Accept hide requests
}

void ClapPluginHost::guiClosed(bool wasDestroyed) noexcept {
	print_line("CLAP: GUI closed (was destroyed: " + String(wasDestroyed ? "true" : "false") + ")");
}

// clap_host_log
void ClapPluginHost::logLog(clap_log_severity severity, const char *message) const noexcept {
	print_line("CLAP Log: " + String(message));
}

// clap_host_params
void ClapPluginHost::paramsRescan(clap_param_rescan_flags flags) noexcept {
	print_line("CLAP: Params rescan");
}

void ClapPluginHost::paramsClear(clap_id paramId, clap_param_clear_flags flags) noexcept {
	print_line("CLAP: Params clear");
}

void ClapPluginHost::paramsRequestFlush() noexcept {
	print_line("CLAP: Params request flush");
}

// clap_host_posix_fd_support
bool ClapPluginHost::posixFdSupportRegisterFd(int fd, clap_posix_fd_flags_t flags) noexcept {
	print_line("CLAP: POSIX FD register");
	return false;
}

bool ClapPluginHost::posixFdSupportModifyFd(int fd, clap_posix_fd_flags_t flags) noexcept {
	print_line("CLAP: POSIX FD modify");
	return false;
}

bool ClapPluginHost::posixFdSupportUnregisterFd(int fd) noexcept {
	print_line("CLAP: POSIX FD unregister");
	return false;
}

// clap_host_remote_controls
bool ClapPluginHost::implementsRemoteControls() const noexcept {
	return false;
}

void ClapPluginHost::remoteControlsChanged() noexcept {
	print_line("CLAP: Remote controls changed");
}

void ClapPluginHost::remoteControlsSuggestPage(clap_id pageId) noexcept {
	print_line("CLAP: Remote controls suggest page");
}

// clap_host_state
bool ClapPluginHost::implementsState() const noexcept {
	return false;
}

void ClapPluginHost::stateMarkDirty() noexcept {
	print_line("CLAP: State mark dirty");
}

// clap_host_timer_support
bool ClapPluginHost::implementsTimerSupport() const noexcept {
	return false;
}

bool ClapPluginHost::timerSupportRegisterTimer(uint32_t periodMs, clap_id *timerId) noexcept {
	print_line("CLAP: Timer register");
	return false;
}

bool ClapPluginHost::timerSupportUnregisterTimer(clap_id timerId) noexcept {
	print_line("CLAP: Timer unregister");
	return false;
}

// clap_host_thread_check
bool ClapPluginHost::threadCheckIsMainThread() const noexcept {
	// In Godot, we need to properly check which thread we're on
	// For now, assume main thread unless in audio processing
	return !_is_processing;
}

bool ClapPluginHost::threadCheckIsAudioThread() const noexcept {
	// Audio thread is only when we're actively processing
	return _is_processing;
}

// clap_host_thread_pool
bool ClapPluginHost::implementsThreadPool() const noexcept {
	return false;
}

bool ClapPluginHost::threadPoolRequestExec(uint32_t numTasks) noexcept {
	print_line("CLAP: Thread pool request exec");
	return false;
}

// Plugin loading and management
bool ClapPluginHost::load(const char *path, int plugin_index) {
	print_line("CLAP: Loading plugin from " + String(path));

	// Load the dynamic library using Godot's API
	Error err = OS::get_singleton()->open_dynamic_library(String(path), _lib_handle);
	if (err != OK) {
		print_line("CLAP: Failed to load library: " + String::num(err));
		return false;
	}

	// Get the CLAP entry point using Godot's symbol loader
	// Note: clap_entry is a symbol pointing to a clap_plugin_entry structure, NOT a function!
	void *clap_entry_symbol = nullptr;
	err = OS::get_singleton()->get_dynamic_library_symbol_handle(_lib_handle, "clap_entry", clap_entry_symbol);
	if (err != OK) {
		print_line("CLAP: Failed to find clap_entry symbol: " + String::num(err));
		OS::get_singleton()->close_dynamic_library(_lib_handle);
		_lib_handle = nullptr;
		return false;
	}

	// Cast the symbol to the correct structure pointer type
	_plugin_entry = reinterpret_cast<const clap_plugin_entry_t *>(clap_entry_symbol);

	// Validate that we got a valid pointer
	if (!_plugin_entry) {
		print_line("CLAP: clap_entry symbol is null");
		OS::get_singleton()->close_dynamic_library(_lib_handle);
		_lib_handle = nullptr;
		return false;
	}

	print_line("CLAP: Found clap_entry structure at address: " + String::num_uint64((uint64_t)_plugin_entry, 16));

	if (!_plugin_entry) {
		print_line("CLAP: clap_entry() returned null");
		OS::get_singleton()->close_dynamic_library(_lib_handle);
		_lib_handle = nullptr;
		return false;
	}

	print_line("CLAP: Got plugin entry, checking version compatibility...");

	// Check CLAP version compatibility
	if (!clap_version_is_compatible(_plugin_entry->clap_version)) {
		print_line("CLAP: Incompatible CLAP version");
		OS::get_singleton()->close_dynamic_library(_lib_handle);
		_lib_handle = nullptr;
		return false;
	}

	print_line("CLAP: Version compatible, initializing plugin entry...");

	// Initialize the plugin entry
	if (!_plugin_entry->init(path)) {
		print_line("CLAP: Plugin entry initialization failed");
		OS::get_singleton()->close_dynamic_library(_lib_handle);
		_lib_handle = nullptr;
		return false;
	}

	print_line("CLAP: Plugin entry initialized, getting factory...");

	// Get the plugin factory
	_plugin_factory = (const clap_plugin_factory_t *)_plugin_entry->get_factory(CLAP_PLUGIN_FACTORY_ID);
	if (!_plugin_factory) {
		print_line("CLAP: Failed to get plugin factory");
		_plugin_entry->deinit();
		OS::get_singleton()->close_dynamic_library(_lib_handle);
		_lib_handle = nullptr;
		return false;
	}

	print_line("CLAP: Got plugin factory, checking plugin count...");

	// Check if we have any plugins
	uint32_t plugin_count = _plugin_factory->get_plugin_count(_plugin_factory);
	if (plugin_count == 0) {
		print_line("CLAP: No plugins found in factory");
		_plugin_entry->deinit();
		OS::get_singleton()->close_dynamic_library(_lib_handle);
		_lib_handle = nullptr;
		return false;
	}

	print_line("CLAP: Found " + String::num(plugin_count) + " plugins, validating index...");

	// Validate plugin index
	if (plugin_index < 0 || plugin_index >= (int)plugin_count) {
		print_line("CLAP: Invalid plugin index " + String::num(plugin_index) + ", available: " + String::num(plugin_count));
		_plugin_entry->deinit();
		OS::get_singleton()->close_dynamic_library(_lib_handle);
		_lib_handle = nullptr;
		return false;
	}

	// Get plugin descriptor
	const clap_plugin_descriptor_t *descriptor = _plugin_factory->get_plugin_descriptor(_plugin_factory, plugin_index);
	if (!descriptor) {
		print_line("CLAP: Failed to get plugin descriptor");
		_plugin_entry->deinit();
		OS::get_singleton()->close_dynamic_library(_lib_handle);
		_lib_handle = nullptr;
		return false;
	}

	print_line("CLAP: Got plugin descriptor for: " + String(descriptor->name));

	// Create the plugin instance
	const clap_plugin_t *plugin_instance = _plugin_factory->create_plugin(_plugin_factory, clapHost(), descriptor->id);
	if (!plugin_instance) {
		print_line("CLAP: Failed to create plugin instance");
		_plugin_entry->deinit();
		OS::get_singleton()->close_dynamic_library(_lib_handle);
		_lib_handle = nullptr;
		return false;
	}

	print_line("CLAP: Created plugin instance, wrapping in proxy...");

	// Wrap the plugin in our proxy
	_plugin = std::make_unique<PluginProxy>(*plugin_instance, *this);

	// Initialize the plugin
	if (!_plugin->init()) {
		print_line("CLAP: Plugin initialization failed");
		_plugin.reset();
		_plugin_entry->deinit();
		OS::get_singleton()->close_dynamic_library(_lib_handle);
		_lib_handle = nullptr;
		return false;
	}

	print_line("CLAP: Successfully loaded plugin: " + String(descriptor->name));
	return true;
}

void ClapPluginHost::activate(int32_t sample_rate, int32_t blockSize) {
	print_line("CLAP: Activating plugin");

	if (!_plugin.get()) {
		print_line("CLAP: No plugin loaded");
		return;
	}

	// Activate the plugin
	if (!_plugin->activate(sample_rate, 1, blockSize)) {
		print_line("CLAP: Plugin activation failed");
		setPluginState(InactiveWithError);
		return;
	}

	// Start processing after activation
	if (!_plugin->startProcessing()) {
		print_line("CLAP: Plugin start processing failed");
		setPluginState(InactiveWithError);
		return;
	}

	// Setup audio buffers
	_audioIn.data32 = nullptr;
	_audioIn.data64 = nullptr;
	_audioIn.channel_count = 2;
	_audioIn.latency = 0;
	_audioIn.constant_mask = 0;

	_audioOut.data32 = nullptr;
	_audioOut.data64 = nullptr;
	_audioOut.channel_count = 2;
	_audioOut.latency = 0;
	_audioOut.constant_mask = 0;

	// Setup process structure
	_process.steady_time = 0;
	_process.frames_count = blockSize;
	_process.transport = nullptr;
	_process.audio_inputs = &_audioIn;
	_process.audio_inputs_count = 1;
	_process.audio_outputs = &_audioOut;
	_process.audio_outputs_count = 1;
	_process.in_events = _evIn.clapInputEvents();
	_process.out_events = _evOut.clapOutputEvents();

	setPluginState(ActiveAndSleeping);
	print_line("CLAP: Plugin activated successfully");
}

void ClapPluginHost::deactivate() {
	print_line("CLAP: Deactivating plugin");

	if (!isPluginActive()) {
		return;
	}

	// Stop processing before deactivating
	if (_plugin.get()) {
		_plugin->stopProcessing();
		_plugin->deactivate();
	}

	setPluginState(Inactive);
}

void ClapPluginHost::setPluginState(PluginState state) {
	_state = state;
}

bool ClapPluginHost::isPluginActive() const {
	switch (_state) {
		case Inactive:
		case InactiveWithError:
			return false;
		default:
			return true;
	}
}

bool ClapPluginHost::isPluginProcessing() const {
	return _state == ActiveAndProcessing;
}

bool ClapPluginHost::isPluginSleeping() const {
	return _state == ActiveAndSleeping;
}

void ClapPluginHost::handlePluginOutputEvents() {
	// TODO: Implement output event handling
}

void ClapPluginHost::process(const void *p_src_buffer, AudioFrame *p_dst_buffer, int32_t p_frame_count) {
	const AudioFrame *src = static_cast<const AudioFrame *>(p_src_buffer);

	// If no plugin is loaded or not active, just pass through
	if (!_plugin.get() || !isPluginActive()) {
		for (int32_t i = 0; i < p_frame_count; i++) {
			p_dst_buffer[i] = src[i];
		}
		return;
	}

	// Prepare audio buffers for CLAP processing
	float *input_l = new float[p_frame_count];
	float *input_r = new float[p_frame_count];
	float *output_l = new float[p_frame_count];
	float *output_r = new float[p_frame_count];

	// Convert AudioFrame to float arrays
	for (int32_t i = 0; i < p_frame_count; i++) {
		input_l[i] = src[i].left;
		input_r[i] = src[i].right;
	}

	// Setup CLAP audio buffers
	float *input_channels[] = { input_l, input_r };
	float *output_channels[] = { output_l, output_r };

	_audioIn.data32 = input_channels;
	_audioOut.data32 = output_channels;
	_process.frames_count = p_frame_count;

	// Clear event lists
	_evIn.clear();
	_evOut.clear();

	// Process with the plugin
	_is_processing = true;
	setPluginState(ActiveAndProcessing);
	clap_process_status status = _plugin->process(&_process);
	setPluginState(ActiveAndSleeping);
	_is_processing = false;

	// Handle processing result
	if (status == CLAP_PROCESS_ERROR) {
		print_line("CLAP: Plugin processing error");
		setPluginState(ActiveWithError);
		// Fallback to passthrough
		for (int32_t i = 0; i < p_frame_count; i++) {
			p_dst_buffer[i] = src[i];
		}
	} else {
		// Convert float arrays back to AudioFrame
		for (int32_t i = 0; i < p_frame_count; i++) {
			p_dst_buffer[i].left = output_l[i];
			p_dst_buffer[i].right = output_r[i];
		}
	}

	// Handle output events
	handlePluginOutputEvents();

	// Cleanup
	delete[] input_l;
	delete[] input_r;
	delete[] output_l;
	delete[] output_r;
}

bool ClapPluginHost::process_silence() const {
	// Return true if plugin can process silence efficiently
	// For now, return false to indicate we need actual audio processing
	return false;
}

void ClapPluginHost::scanParams() {
	// Placeholder for parameter scanning
}

void ClapPluginHost::scanParam(int32_t index) {
	// Placeholder for individual parameter scanning
}

void ClapPluginHost::scanQuickControls() {
	// Placeholder for quick controls scanning
}

// GUI management methods
bool ClapPluginHost::hasGui() const {
	if (!_plugin.get()) {
		return false;
	}

	const clap_plugin_gui_t *gui_ext = nullptr;
	_plugin->getExtension(gui_ext, CLAP_EXT_GUI);

	return gui_ext != nullptr;
}

bool ClapPluginHost::createGui() {
	if (_gui_created) {
		print_line("CLAP: GUI already created");
		return true;
	}

	if (!_plugin.get()) {
		print_line("CLAP: No plugin loaded");
		return false;
	}

	const clap_plugin_gui_t *gui_ext = nullptr;
	_plugin->getExtension(gui_ext, CLAP_EXT_GUI);

	if (!gui_ext) {
		print_line("CLAP: Plugin does not support GUI");
		return false;
	}

	// Check if the plugin supports a native GUI API we can use
	bool api_supported = false;

	// Get the underlying plugin pointer
	const clap_plugin_t *plugin_ptr = _plugin->clapPlugin();

#ifdef __APPLE__
	// Try Cocoa on macOS
	if (gui_ext->is_api_supported(plugin_ptr, CLAP_WINDOW_API_COCOA, false)) {
		print_line("CLAP: Using Cocoa API for GUI");
		api_supported = true;
	}
#elif defined(_WIN32)
	// Try Win32 on Windows
	if (gui_ext->is_api_supported(plugin_ptr, CLAP_WINDOW_API_WIN32, false)) {
		print_line("CLAP: Using Win32 API for GUI");
		api_supported = true;
	}
#else
	// Try X11 on Linux
	if (gui_ext->is_api_supported(plugin_ptr, CLAP_WINDOW_API_X11, false)) {
		print_line("CLAP: Using X11 API for GUI");
		api_supported = true;
	}
#endif

	if (!api_supported) {
		print_line("CLAP: No supported GUI API found");
		return false;
	}

	// Create the GUI
	const char *api_name;
#ifdef __APPLE__
	api_name = CLAP_WINDOW_API_COCOA;
#elif defined(_WIN32)
	api_name = CLAP_WINDOW_API_WIN32;
#else
	api_name = CLAP_WINDOW_API_X11;
#endif

	if (!gui_ext->create(plugin_ptr, api_name, false)) {
		print_line("CLAP: Failed to create GUI");
		return false;
	}

	_gui_created = true;
	print_line("CLAP: GUI created successfully");
	return true;
}

void ClapPluginHost::destroyGui() {
	if (!_gui_created) {
		return;
	}

	if (!_plugin.get()) {
		return;
	}

	const clap_plugin_gui_t *gui_ext = nullptr;
	_plugin->getExtension(gui_ext, CLAP_EXT_GUI);

	if (gui_ext) {
		gui_ext->destroy(_plugin->clapPlugin());
		_gui_created = false;
		_gui_visible = false;
		print_line("CLAP: GUI destroyed");
	}
}

bool ClapPluginHost::showGui() {
	if (!_gui_created) {
		if (!createGui()) {
			return false;
		}
	}

	if (!_plugin.get()) {
		return false;
	}

	const clap_plugin_gui_t *gui_ext = nullptr;
	_plugin->getExtension(gui_ext, CLAP_EXT_GUI);

	if (!gui_ext) {
		return false;
	}

	// Get the plugin pointer for the GUI operations
	const clap_plugin_t *plugin_ptr = _plugin->clapPlugin();

	// Set parent window if we have one
	if (_gui_parent) {
		clap_window window = {};
#ifdef __APPLE__
		window.api = CLAP_WINDOW_API_COCOA;
		window.cocoa = _gui_parent;
#elif defined(_WIN32)
		window.api = CLAP_WINDOW_API_WIN32;
		window.win32 = _gui_parent;
#else
		window.api = CLAP_WINDOW_API_X11;
		window.x11 = reinterpret_cast<unsigned long>(_gui_parent);
#endif

		if (!gui_ext->set_parent(plugin_ptr, &window)) {
			print_line("CLAP: Failed to set GUI parent");
			return false;
		}
	}

	if (!gui_ext->show(plugin_ptr)) {
		print_line("CLAP: Failed to show GUI");
		return false;
	}

	_gui_visible = true;
	print_line("CLAP: GUI shown");
	return true;
}

bool ClapPluginHost::hideGui() {
	if (!_gui_created || !_gui_visible) {
		return true;
	}

	if (!_plugin.get()) {
		return false;
	}

	const clap_plugin_gui_t *gui_ext = nullptr;
	_plugin->getExtension(gui_ext, CLAP_EXT_GUI);

	if (!gui_ext) {
		return false;
	}

	const clap_plugin_t *plugin_ptr = _plugin->clapPlugin();
	if (!gui_ext->hide(plugin_ptr)) {
		print_line("CLAP: Failed to hide GUI");
		return false;
	}

	_gui_visible = false;
	print_line("CLAP: GUI hidden");
	return true;
}

bool ClapPluginHost::isGuiVisible() const {
	return _gui_visible;
}

void ClapPluginHost::setGuiParent(void *parent_window) {
	_gui_parent = parent_window;
	print_line("CLAP: GUI parent set to " + String::num_uint64((uint64_t)parent_window, 16));
}
