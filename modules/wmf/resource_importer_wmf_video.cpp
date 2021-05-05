#include "resource_importer_wmf_video.h"
#include "video_stream_wmf.h"

#include "core/io/file_access.h"
#include "core/io/resource_saver.h"

String ResourceImporterWMFVideo::get_importer_name() const {
	return "WindowsMediaFoundation";
}

String ResourceImporterWMFVideo::get_visible_name() const {
	return "WindowsMediaFoundation";
}

void ResourceImporterWMFVideo::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("mp4");
	p_extensions->push_back("avi");
	p_extensions->push_back("mkv");
	p_extensions->push_back("webm");
}

String ResourceImporterWMFVideo::get_save_extension() const {
	return "wmfvstr";
}

String ResourceImporterWMFVideo::get_resource_type() const {
	return "VideoStreamWMF";
}

bool ResourceImporterWMFVideo::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

int ResourceImporterWMFVideo::get_preset_count() const {
	return 0;
}

String ResourceImporterWMFVideo::get_preset_name(int p_idx) const {
	return String();
}

void ResourceImporterWMFVideo::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "loop"), false));
}

Error ResourceImporterWMFVideo::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	VideoStreamWMF *stream = memnew(VideoStreamWMF);
	stream->set_file(p_source_file);
	Ref<VideoStreamWMF> wmfv_stream = Ref<VideoStreamWMF>(stream);
	return ResourceSaver::save(wmfv_stream, p_save_path + ".wmfvstr");
}

ResourceImporterWMFVideo::ResourceImporterWMFVideo() {
}
