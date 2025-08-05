#include "clap_plugin_host.h"

#include "godot_cpp/core/print_string.hpp"
#include <dlfcn.h>

#include <clap/helpers/host.hxx>
#include <clap/helpers/plugin-proxy.hxx>
#include <clap/helpers/reducing-param-queue.hxx>
#include <godot_cpp/classes/audio_frame.hpp>
#include <godot_cpp/classes/input.hpp>

// Instantiate
template class clap::helpers::Host<PluginHost_MH, PluginHost_CL>;

template class clap::helpers::PluginProxy<PluginHost_MH, PluginHost_CL>;

void ClapPluginHost::requestRestart() noexcept {
	godot::print_line("Requesting restart");
}
void ClapPluginHost::requestProcess() noexcept {
	godot::print_line("Requesting process");
}
void ClapPluginHost::requestCallback() noexcept {
	godot::print_line("Requesting callback");
}
bool ClapPluginHost::implementsGui() const noexcept { return false; }
void ClapPluginHost::guiResizeHintsChanged() noexcept {
	godot::print_line("Requesting resize hints");
}
bool ClapPluginHost::guiRequestResize(uint32_t width, uint32_t height) noexcept {
	godot::print_line("Requesting resize");
	return false;
}
bool ClapPluginHost::guiRequestShow() noexcept {
	godot::print_line("Requesting show");
	return false;
}
bool ClapPluginHost::guiRequestHide() noexcept {
	godot::print_line("Requesting hide");
	return false;
}
void ClapPluginHost::guiClosed(bool wasDestroyed) noexcept {
	godot::print_line("Requesting closed");
}
void ClapPluginHost::logLog(clap_log_severity severity, const char *message) const noexcept {
	godot::print_line("Requesting log");
}
void ClapPluginHost::paramsClear(clap_id paramId, clap_param_clear_flags flags) noexcept {
	godot::print_line("Requesting clear");
}
void ClapPluginHost::paramsRequestFlush() noexcept {
	godot::print_line("Requesting flush");
}
bool ClapPluginHost::posixFdSupportRegisterFd(int fd, clap_posix_fd_flags_t flags) noexcept {
	godot::print_line("Requesting register fd");
	return false;
}
bool ClapPluginHost::posixFdSupportModifyFd(int fd, clap_posix_fd_flags_t flags) noexcept {
	godot::print_line("Requesting modify fd");
	return false;
}
bool ClapPluginHost::posixFdSupportUnregisterFd(int fd) noexcept {
	godot::print_line("Requesting unregister fd");
	return false;
}
bool ClapPluginHost::implementsRemoteControls() const noexcept { return false; }
void ClapPluginHost::remoteControlsChanged() noexcept {
	godot::print_line("Requesting remote controls");
}
void ClapPluginHost::remoteControlsSuggestPage(clap_id pageId) noexcept {
	godot::print_line("Requesting suggest page");
}
bool ClapPluginHost::implementsState() const noexcept { return false; }
void ClapPluginHost::stateMarkDirty() noexcept {
	godot::print_line("state marke dirty");
}
bool ClapPluginHost::implementsTimerSupport() const noexcept { return false; }
bool ClapPluginHost::timerSupportRegisterTimer(uint32_t periodMs, clap_id *timerId) noexcept {
	godot::print_line("Requesting timer register");
	return false;
}
bool ClapPluginHost::timerSupportUnregisterTimer(clap_id timerId) noexcept {
	return false;
}
bool ClapPluginHost::threadCheckIsMainThread() const noexcept {
	godot::print_line("Requesting thread check");
	return true;
}
bool ClapPluginHost::threadCheckIsAudioThread() const noexcept {
	godot::print_line("Requesting thread check");
	return true;
}
bool ClapPluginHost::implementsThreadPool() const noexcept { return false; }
bool ClapPluginHost::threadPoolRequestExec(uint32_t numTasks) noexcept {
	godot::print_line("Requesting thread pool");
	return false;
}

ClapPluginHost::ClapPluginHost() :
		BaseHost("Godot Clap Host", // name
				"ivorius", // vendor
				"0.1.0", // version
				"https://github.com/ivorforce" // url
		) {
}

bool ClapPluginHost::load(const char *path, int plugin_index) {
	void *handle = dlopen(path, RTLD_NOW);

	// Clear any existing errors
	dlerror();

	// Get the function from the library
	_plugin_entry = reinterpret_cast<const struct clap_plugin_entry *>(dlsym(handle, "clap_entry"));
	const char *dlsym_error = dlerror();
	if (dlsym_error || !_plugin_entry) {
		fprintf(stderr, "Error finding symbol 'clap_entry': %s\n", dlsym_error);
		dlclose(handle);
		return false;
	}

	_plugin_entry->init(path);

	_plugin_factory = static_cast<const clap_plugin_factory *>(_plugin_entry->get_factory(CLAP_PLUGIN_FACTORY_ID));
	auto count = _plugin_factory->get_plugin_count(_plugin_factory);
	if (plugin_index >= count) {
		godot::print_line("Not found");
		return false;
	}

	auto desc = _plugin_factory->get_plugin_descriptor(_plugin_factory, plugin_index);
	if (!desc) {
		godot::print_line("No plugin descriptor");
		return false;
	}

	if (!clap_version_is_compatible(desc->clap_version)) {
		// qWarning() << "Incompatible clap version: Plugin is: " << desc->clap_version.major << "."
		// 		   << desc->clap_version.minor << "." << desc->clap_version.revision << " Host is "
		// 		   << CLAP_VERSION.major << "." << CLAP_VERSION.minor << "." << CLAP_VERSION.revision;
		godot::print_line("Incompatible clap version");
		return false;
	}

	godot::print_line("Loaded plugin with id: %s", godot::String::utf8(desc->id));

	const auto plugin = _plugin_factory->create_plugin(_plugin_factory, clapHost(), desc->id);
	if (!plugin) {
		godot::print_line("could not create the plugin");
		return false;
	}

	_plugin = std::make_unique<PluginProxy>(*plugin, *this);

	if (!_plugin->init()) {
		godot::print_line("could not init the plugin");
		return false;
	}

	scanParams();
	scanQuickControls();

	return true;
}

void ClapPluginHost::process(const void *p_src_buffer, godot::AudioFrame *p_dst_buffer, int32_t p_frame_count) {
	// checkForAudioThread();
	if (!_plugin.get())
		return;

	// Fake note in
	static bool noteIn = false;
	bool isPressed = godot::Input::get_singleton()->is_action_pressed("play");
	if (noteIn != isPressed) {
		noteIn = isPressed;

		clap_event_note ev;
		ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
		ev.header.type = isPressed ? CLAP_EVENT_NOTE_ON : CLAP_EVENT_NOTE_OFF;
		// ev.header.time = sampleOffset;
		ev.header.time = 0;
		ev.header.flags = 0;
		ev.header.size = sizeof(ev);
		ev.port_index = 0;
		// ev.key = key;
		ev.key = 60;
		// ev.channel = channel;
		ev.channel = 0;
		ev.note_id = -1;
		// ev.velocity = velocity / 127.0;
		ev.velocity = 1.0;

		_evIn.push(&ev.header);
	}

	// Can't process a plugin that is not active
	if (!isPluginActive())
		return;

	// Do we want to deactivate the plugin?
	// if (_scheduleDeactivate) {
	// 	_scheduleDeactivate = false;
	// 	if (_state == ActiveAndProcessing)
	// 		_plugin->stopProcessing();
	// 	setPluginState(ActiveAndReadyToDeactivate);
	// 	return;
	// }

	// We can't process a plugin which failed to start processing
	if (_state == ActiveWithError)
		return;

	_process.transport = nullptr;

	_process.in_events = _evIn.clapInputEvents();
	_process.out_events = _evOut.clapOutputEvents();

	_audioIn.data32 = (float**)&p_src_buffer;
	_audioIn.channel_count = 1;
	_audioIn.constant_mask = 0;
	_audioIn.latency = 0;

	_audioOut.data32 = (float**)&p_dst_buffer;
	_audioOut.channel_count = 1;
	_audioOut.constant_mask = 0;
	_audioIn.latency = 0;

	// TODO * 2 because we pretend we only have only channel, should interleave later
	_process.frames_count = p_frame_count * 2;
	_process.audio_inputs = &_audioIn;
	_process.audio_inputs_count = 1;
	_process.audio_outputs = &_audioOut;
	_process.audio_outputs_count = 1;

	_evOut.clear();
	// generatePluginInputEvents();

	if (isPluginSleeping()) {
		// if (!_scheduleProcess && _evIn.empty())
		// 	// The plugin is sleeping, there is no request to wake it up and there are no events to
		// 		// process
		// 			return;
		//
		// _scheduleProcess = false;
		if (!_plugin->startProcessing()) {
			// the plugin failed to start processing
			setPluginState(ActiveWithError);
			return;
		}

		setPluginState(ActiveAndProcessing);
	}

	int32_t status = CLAP_PROCESS_SLEEP;
	if (isPluginProcessing()) {
		status = _plugin->process(&_process);
	}
	if (status != CLAP_PROCESS_CONTINUE) {
		godot::print_line("Processing change");
	}

	// De-interleave
	// for (int i = 0; i < p_frame_count / 2; ++i) {
	// 	p_dst_buffer[i].left = p_dst_buffer[i].left;
	// }

	handlePluginOutputEvents();

	_evOut.clear();
	_evIn.clear();
}

void ClapPluginHost::scanParams() { paramsRescan(CLAP_PARAM_RESCAN_ALL); }

void ClapPluginHost::paramsRescan(uint32_t flags) noexcept {
   // checkForMainThread();

   if (!_plugin->canUseParams())
      return;

	godot::print_line("Can use params");

   // // 1. it is forbidden to use CLAP_PARAM_RESCAN_ALL if the plugin is active
   // if (isPluginActive() && (flags & CLAP_PARAM_RESCAN_ALL)) {
   //    throw std::logic_error(
   //       "clap_host_params.recan(CLAP_PARAM_RESCAN_ALL) was called while the plugin is active!");
   //    return;
   // }
   //
   // // 2. scan the params.
   // auto count = _plugin->paramsCount();
   // std::unordered_set<clap_id> paramIds(count * 2);
   //
   // for (int32_t i = 0; i < count; ++i) {
   //    clap_param_info info;
   //    if (!_plugin->paramsGetInfo(i, &info))
   //       throw std::logic_error("clap_plugin_params.get_info did return false!");
   //
   //    if (info.id == CLAP_INVALID_ID) {
   //       std::ostringstream msg;
   //       msg << "clap_plugin_params.get_info() reported a parameter with id = CLAP_INVALID_ID"
   //           << std::endl
   //           << " 2. name: " << info.name << ", module: " << info.module << std::endl;
   //       throw std::logic_error(msg.str());
   //    }
   //
   //    auto it = _params.find(info.id);
   //
   //    // check that the parameter is not declared twice
   //    if (paramIds.count(info.id) > 0) {
   //       Q_ASSERT(it != _params.end());
   //
   //       std::ostringstream msg;
   //       msg << "the parameter with id: " << info.id << " was declared twice." << std::endl
   //           << " 1. name: " << it->second->info().name << ", module: " << it->second->info().module
   //           << std::endl
   //           << " 2. name: " << info.name << ", module: " << info.module << std::endl;
   //       throw std::logic_error(msg.str());
   //    }
   //    paramIds.insert(info.id);
   //
   //    if (it == _params.end()) {
   //       if (!(flags & CLAP_PARAM_RESCAN_ALL)) {
   //          std::ostringstream msg;
   //          msg << "a new parameter was declared, but the flag CLAP_PARAM_RESCAN_ALL was not "
   //                 "specified; id: "
   //              << info.id << ", name: " << info.name << ", module: " << info.module << std::endl;
   //          throw std::logic_error(msg.str());
   //       }
   //
   //       double value = getParamValue(info);
   //       auto param = std::make_unique<PluginParam>(*this, info, value);
   //       checkValidParamValue(*param, value);
   //       _params.insert_or_assign(info.id, std::move(param));
   //    } else {
   //       // update param info
   //       if (!it->second->isInfoEqualTo(info)) {
   //          if (!clapParamsRescanMayInfoChange(flags)) {
   //             std::ostringstream msg;
   //             msg << "a parameter's info did change, but the flag CLAP_PARAM_RESCAN_INFO "
   //                    "was not specified; id: "
   //                 << info.id << ", name: " << info.name << ", module: " << info.module
   //                 << std::endl;
   //             throw std::logic_error(msg.str());
   //          }
   //
   //          if (!(flags & CLAP_PARAM_RESCAN_ALL) &&
   //              !it->second->isInfoCriticallyDifferentTo(info)) {
   //             std::ostringstream msg;
   //             msg << "a parameter's info has critical changes, but the flag CLAP_PARAM_RESCAN_ALL "
   //                    "was not specified; id: "
   //                 << info.id << ", name: " << info.name << ", module: " << info.module
   //                 << std::endl;
   //             throw std::logic_error(msg.str());
   //          }
   //
   //          it->second->setInfo(info);
   //       }
   //
   //       double value = getParamValue(info);
   //       if (it->second->value() != value) {
   //          if (!clapParamsRescanMayValueChange(flags)) {
   //             std::ostringstream msg;
   //             msg << "a parameter's value did change but, but the flag CLAP_PARAM_RESCAN_VALUES "
   //                    "was not specified; id: "
   //                 << info.id << ", name: " << info.name << ", module: " << info.module
   //                 << std::endl;
   //             throw std::logic_error(msg.str());
   //          }
   //
   //          // update param value
   //          checkValidParamValue(*it->second, value);
   //          it->second->setValue(value);
   //          it->second->setModulation(value);
   //       }
   //    }
   // }
   //
   // // remove parameters which are gone
   // for (auto it = _params.begin(); it != _params.end();) {
   //    if (paramIds.find(it->first) != paramIds.end())
   //       ++it;
   //    else {
   //       if (!(flags & CLAP_PARAM_RESCAN_ALL)) {
   //          std::ostringstream msg;
   //          auto &info = it->second->info();
   //          msg << "a parameter was removed, but the flag CLAP_PARAM_RESCAN_ALL was not "
   //                 "specified; id: "
   //              << info.id << ", name: " << info.name << ", module: " << info.module << std::endl;
   //          throw std::logic_error(msg.str());
   //       }
   //       it = _params.erase(it);
   //    }
   // }
   //
   // if (flags & CLAP_PARAM_RESCAN_ALL)
   //    paramsChanged();
}

void ClapPluginHost::scanQuickControls() {
	// checkForMainThread();

	if (!_plugin->canUseRemoteControls())
		return;

	godot::print_line("Can use remote controls");

	// quickControlsSetSelectedPage(CLAP_INVALID_ID);
	// _remoteControlsPages.clear();
	// _remoteControlsPagesIndex.clear();
	//
	// const auto N = _plugin->remoteControlsCount();
	// if (N == 0)
	// 	return;
	//
	// _remoteControlsPages.reserve(N);
	// _remoteControlsPagesIndex.reserve(N);
	//
	// clap_id firstPageId = CLAP_INVALID_ID;
	// for (int i = 0; i < N; ++i) {
	// 	auto page = std::make_unique<clap_remote_controls_page>();
	// 	if (!_plugin->remoteControlsGet(i, page.get())) {
	// 		std::ostringstream msg;
	// 		msg << "clap_plugin_remote_controls.get_page(" << i << ") failed, while the page count is "
	// 			<< N;
	// 		throw std::logic_error(msg.str());
	// 	}
	//
	// 	if (page->page_id == CLAP_INVALID_ID) {
	// 		std::ostringstream msg;
	// 		msg << "clap_plugin_remote_controls.get_page(" << i << ") gave an invalid page_id";
	// 		throw std::invalid_argument(msg.str());
	// 	}
	//
	// 	if (i == 0)
	// 		firstPageId = page->page_id;
	//
	// 	auto it = _remoteControlsPagesIndex.find(page->page_id);
	// 	if (it != _remoteControlsPagesIndex.end()) {
	// 		std::ostringstream msg;
	// 		msg << "clap_plugin_remote_controls.get_page(" << i
	// 			<< ") gave twice the same page_id:" << page->page_id << std::endl
	// 			<< " 1. name: " << it->second->page_name << std::endl
	// 			<< " 2. name: " << page->page_name;
	// 		throw std::invalid_argument(msg.str());
	// 	}
	//
	// 	_remoteControlsPagesIndex.insert_or_assign(page->page_id, page.get());
	// 	_remoteControlsPages.emplace_back(std::move(page));
	// }
	//
	// quickControlsPagesChanged();
	// quickControlsSetSelectedPage(firstPageId);
}

void ClapPluginHost::activate(int32_t sample_rate, int32_t blockSize) {
	// checkForMainThread();

	if (!_plugin.get())
		return;

	assert(!isPluginActive());
	if (!_plugin->activate(sample_rate, blockSize, blockSize)) {
		setPluginState(InactiveWithError);
		return;
	}

	// _scheduleProcess = true;
	setPluginState(ActiveAndSleeping);
}

void ClapPluginHost::deactivate() {
	// checkForMainThread();

	if (!isPluginActive())
		return;

	while (isPluginProcessing() || isPluginSleeping()) {
		// _scheduleDeactivate = true;
		// QThread::msleep(10);
	}

	// _scheduleDeactivate = false;

	_plugin->deactivate();
	setPluginState(Inactive);
}

void ClapPluginHost::setPluginState(PluginState state) {
	switch (state) {
		case Inactive:
			CRASH_COND(_state != ActiveAndReadyToDeactivate);
		break;

		case InactiveWithError:
			CRASH_COND(_state != Inactive);
		break;

		case ActiveAndSleeping:
			CRASH_COND(_state != Inactive && _state != ActiveAndProcessing);
		break;

		case ActiveAndProcessing:
			CRASH_COND(_state != ActiveAndSleeping);
		break;

		case ActiveWithError:
			CRASH_COND(_state != ActiveAndProcessing);
		break;

		case ActiveAndReadyToDeactivate:
			CRASH_COND(_state != ActiveAndProcessing && _state != ActiveAndSleeping &&
					 _state != ActiveWithError);
		break;

		default:
			godot::print_error("Invalid state");
			std::terminate();
	}

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

bool ClapPluginHost::isPluginProcessing() const { return _state == ActiveAndProcessing; }

bool ClapPluginHost::isPluginSleeping() const { return _state == ActiveAndSleeping; }

void ClapPluginHost::handlePluginOutputEvents() {
	for (uint32_t i = 0; i < _evOut.size(); ++i) {
		auto h = _evOut.get(i);
		godot::print_line("got output %d", h->type);
	}
}
