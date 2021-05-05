#include "register_types.h"
#include "video_stream_wmf.h"
#include "resource_importer_wmf_video.h"

#include <mfapi.h>
#include <stdio.h>

void register_wmf_types() {
    MFStartup(MF_VERSION, MFSTARTUP_FULL);

#ifdef TOOLS_ENABLED
	Ref<ResourceImporterWMFVideo> wmfv_import;
	wmfv_import.instance();
	ResourceFormatImporter::get_singleton()->add_importer(wmfv_import);
#endif
    ClassDB::register_class<VideoStreamWMF>();
}

void unregister_wmf_types() {
    MFShutdown();
}
