#include "register_types.h"
#include "resource_importer_wmf_video.h"
#include "video_stream_wmf.h"

#include "mfapi.h"
#include <stdio.h>

void initialize_wmf_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	MFStartup(MF_VERSION, MFSTARTUP_FULL);

#ifdef TOOLS_ENABLED
	Ref<ResourceImporterWMFVideo> wmfv_import;
	wmfv_import.instantiate();
	ResourceFormatImporter::get_singleton()->add_importer(wmfv_import);
#endif
	ClassDB::register_class<VideoStreamWMF>();
}

void uninitialize_wmf_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	MFShutdown();
}
