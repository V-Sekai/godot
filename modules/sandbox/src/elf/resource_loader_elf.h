#pragma once

#include "../config.h"
#include <godot_cpp/classes/resource_format_loader.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
GODOT_NAMESPACE

class ResourceFormatLoaderELF : public ResourceFormatLoader {
	GDCLASS(ResourceFormatLoaderELF, ResourceFormatLoader);

protected:
	static void GODOT_CPP_FUNC (bind_methods) () {}

public:
#ifdef GODOT_MODULE
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override;
	virtual bool recognize_path(const String &p_path, const String &p_for_type = String()) const override;
	virtual bool handles_type(const String &p_type) const override;
	virtual String get_resource_type(const String &p_path) const override;
#else
	virtual Variant GODOT_CPP_FUNC( load) (const String &path, const String &original_path, bool use_sub_threads, int32_t cache_mode) const override;
	virtual bool GODOT_CPP_FUNC (recognize_path) (const String &path, const StringName &type) const override;
	virtual bool GODOT_CPP_FUNC (handles_type) (const StringName &type) const override;
	virtual String GODOT_CPP_FUNC (get_resource_type) (const String &p_path) const override;
#endif	
};
