#ifndef CLAP_PLUGIN_HOST_H
#define CLAP_PLUGIN_HOST_H

#include <clap/clap.h>
#include <clap/helpers/event-list.hh>
#include <clap/helpers/host.hh>
#include <clap/helpers/plugin-proxy.hh>
#include <clap/helpers/reducing-param-queue.hh>

namespace godot {
struct AudioFrame;
}

constexpr auto PluginHost_MH = clap::helpers::MisbehaviourHandler::Terminate;
constexpr auto PluginHost_CL = clap::helpers::CheckingLevel::Maximal;

using BaseHost = clap::helpers::Host<PluginHost_MH, PluginHost_CL>;
extern template class clap::helpers::Host<PluginHost_MH, PluginHost_CL>;

using PluginProxy = clap::helpers::PluginProxy<PluginHost_MH, PluginHost_CL>;
extern template class clap::helpers::PluginProxy<PluginHost_MH, PluginHost_CL>;

class ClapPluginHost final : public BaseHost {
private:
	/* clap host callbacks */
	void scanParams();
	void scanParam(int32_t index);

	void scanQuickControls();

protected:
	/////////////////////////
	// clap::helpers::Host //
	/////////////////////////

	// clap_host
	void requestRestart() noexcept override;
	void requestProcess() noexcept override;
	void requestCallback() noexcept override;

	// clap_host_gui
	bool implementsGui() const noexcept override;
	void guiResizeHintsChanged() noexcept override;
	bool guiRequestResize(uint32_t width, uint32_t height) noexcept override;
	bool guiRequestShow() noexcept override;
	bool guiRequestHide() noexcept override;
	void guiClosed(bool wasDestroyed) noexcept override;

	// clap_host_log
	bool implementsLog() const noexcept override { return true; }
	void logLog(clap_log_severity severity, const char *message) const noexcept override;

	// clap_host_params
	bool implementsParams() const noexcept override { return true; }
	void paramsRescan(clap_param_rescan_flags flags) noexcept override;
	void paramsClear(clap_id paramId, clap_param_clear_flags flags) noexcept override;
	void paramsRequestFlush() noexcept override;

	// clap_host_posix_fd_support
	bool implementsPosixFdSupport() const noexcept override { return true; }
	bool posixFdSupportRegisterFd(int fd, clap_posix_fd_flags_t flags) noexcept override;
	bool posixFdSupportModifyFd(int fd, clap_posix_fd_flags_t flags) noexcept override;
	bool posixFdSupportUnregisterFd(int fd) noexcept override;

	// clap_host_remote_controls
	bool implementsRemoteControls() const noexcept override;
	void remoteControlsChanged() noexcept override;
	void remoteControlsSuggestPage(clap_id pageId) noexcept override;

	// clap_host_state
	bool implementsState() const noexcept override;
	void stateMarkDirty() noexcept override;

	// clap_host_timer_support
	bool implementsTimerSupport() const noexcept override;
	bool timerSupportRegisterTimer(uint32_t periodMs, clap_id *timerId) noexcept override;
	bool timerSupportUnregisterTimer(clap_id timerId) noexcept override;

	// clap_host_thread_check
	bool threadCheckIsMainThread() const noexcept override;
	bool threadCheckIsAudioThread() const noexcept override;

	// clap_host_thread_pool
	bool implementsThreadPool() const noexcept override;
	bool threadPoolRequestExec(uint32_t numTasks) noexcept override;

	// clap_host_tail
	bool implementsTail() const noexcept override { return false; }
	void tailChanged() noexcept override {}

public:
	enum PluginState {
		// The plugin is inactive, only the main thread uses it
		Inactive,

		// Activation failed
		InactiveWithError,

		// The plugin is active and sleeping, the audio engine can call set_processing()
		ActiveAndSleeping,

		// The plugin is processing
		ActiveAndProcessing,

		// The plugin did process but is in error
		ActiveWithError,

		// The plugin is not used anymore by the audio engine and can be deactivated on the main
		// thread
		ActiveAndReadyToDeactivate,
	};

	const clap_plugin_entry *_plugin_entry;
	const clap_plugin_factory *_plugin_factory;
	std::unique_ptr<PluginProxy> _plugin;

	clap_audio_buffer _audioIn = {};
	clap_audio_buffer _audioOut = {};
	clap::helpers::EventList _evIn;
	clap::helpers::EventList _evOut;
	clap_process _process;

	PluginState _state = Inactive;
	bool _stateIsDirty = false;

	ClapPluginHost();
	~ClapPluginHost() override = default;

	bool load(const char *path, int plugin_index);

	void activate(int32_t sample_rate, int32_t blockSize);
	void deactivate();

	void setPluginState(::ClapPluginHost::PluginState state);
	bool isPluginProcessing() const;
	bool isPluginActive() const;
	bool isPluginSleeping() const;
	void handlePluginOutputEvents();

	void process(const void *p_src_buffer, godot::AudioFrame *p_dst_buffer, int32_t p_frame_count);
	bool process_silence() const { return true; }
};

#endif //CLAP_PLUGIN_HOST_H
