#pragma once

#include "../config.h"
#include <godot_cpp/classes/resource_format_saver.hpp>
#include <godot_cpp/classes/resource_saver.hpp>
GODOT_NAMESPACE

class ResourceFormatSaverELF : public ResourceFormatSaver {
	GDCLASS(ResourceFormatSaverELF, ResourceFormatSaver);

protected:
	static void GODOT_CPP_FUNC (bind_methods)() {}

public:
// ERROR: In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:16:26: error: expected class name
//    16 | class ELFScript : public ScriptExtension {
//       |                          ^
// modules\sandbox/src/elf/script_elf.h:17:21: error: use of undeclared identifier 'ScriptExtension'
//    17 |         GDCLASS(ELFScript, ScriptExtension);
//       |                            ^
// modules\sandbox/src/elf/script_elf.h:17:2: error: constexpr variable '_class_is_enabled' must be initialized by a constant expression
//    17 |         GDCLASS(ELFScript, ScriptExtension);
//       |         ^
// ./core/object/object.h:397:24: note: expanded from macro 'GDCLASS'
//   397 |         static constexpr bool _class_is_enabled = !bool(GD_IS_DEFINED(ClassDB_Disable_##m_class)) && m_inherits::_class_is_enabled;                  \
//       |                               ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: 'get_class' marked 'override' but does not override any member functions
// ./core/object/object.h:398:17: note: expanded from macro 'GDCLASS'
//   398 |         virtual String get_class() const override {                                                                                                  \
//       |                        ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: '_get_class_namev' marked 'override' but does not override any member functions
// ./core/object/object.h:404:28: note: expanded from macro 'GDCLASS'
//   404 |         virtual const StringName *_get_class_namev() const override {                                                                                \
//       |                                   ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: 'is_class' marked 'override' but does not override any member functions
// ./core/object/object.h:425:15: note: expanded from macro 'GDCLASS'
//   425 |         virtual bool is_class(const String &p_class) const override {                                                                                \
//       |                      ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: 'is_class_ptr' marked 'override' but does not override any member functions
// ./core/object/object.h:431:15: note: expanded from macro 'GDCLASS'
//   431 |         virtual bool is_class_ptr(void *p_ptr) const override { return (p_ptr == get_class_ptr_static()) ? true : m_inherits::is_class_ptr(p_ptr); } \
//       |                      ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: '_initialize_classv' marked 'override' but does not override any member functions
// ./core/object/object.h:467:15: note: expanded from macro 'GDCLASS'
//   467 |         virtual void _initialize_classv() override {                                                                                                 \
//       |                      ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: '_getv' marked 'override' but does not override any member functions
// ./core/object/object.h:473:15: note: expanded from macro 'GDCLASS'
//   473 |         virtual bool _getv(const StringName &p_name, Variant &r_ret) const override {                                                                \
//       |                      ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: '_setv' marked 'override' but does not override any member functions
// ./core/object/object.h:484:15: note: expanded from macro 'GDCLASS'
//   484 |         virtual bool _setv(const StringName &p_name, const Variant &p_property) override {                                                           \
//       |                      ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: '_get_property_listv' marked 'override' but does not override any member functions
// ./core/object/object.h:496:15: note: expanded from macro 'GDCLASS'
//   496 |         virtual void _get_property_listv(List<PropertyInfo> *p_list, bool p_reversed) const override {                                               \
//       |                      ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: '_validate_propertyv' marked 'override' but does not override any member functions
// ./core/object/object.h:512:15: note: expanded from macro 'GDCLASS'
//   512 |         virtual void _validate_propertyv(PropertyInfo &p_property) const override {                                                                  \
//       |                      ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: '_property_can_revertv' marked 'override' but does not override any member functions
// ./core/object/object.h:521:15: note: expanded from macro 'GDCLASS'
//   521 |         virtual bool _property_can_revertv(const StringName &p_name) const override {                                                                \
//       |                      ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: '_property_get_revertv' marked 'override' but does not override any member functions
// ./core/object/object.h:532:15: note: expanded from macro 'GDCLASS'
//   532 |         virtual bool _property_get_revertv(const StringName &p_name, Variant &r_ret) const override {                                                \
//       |                      ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:17:2: error: '_notificationv' marked 'override' but does not override any member functions
// ./core/object/object.h:543:15: note: expanded from macro 'GDCLASS'
//   543 |         virtual void _notificationv(int p_notification, bool p_reversed) override {                                                                  \
//       |                      ^
// In file included from modules\sandbox\register_types.cpp:15:
// modules\sandbox/src/elf/script_elf.h:57:31: error: 'editor_can_reload_from_file' marked 'override' but does not override any member functions
//    57 |         virtual bool GODOT_CPP_FUNC (editor_can_reload_from_file)() override;
//       |                                      ^
// modules\sandbox/src/elf/script_elf.h:58:31: error: 'placeholder_erased' marked 'override' but does not override any member functions
//    58 |         virtual void GODOT_CPP_FUNC (placeholder_erased)(void *p_placeholder) override;
//       |                                      ^
// modules\sandbox/src/elf/script_elf.h:59:31: error: 'can_instantiate' marked 'override' but does not override any member functions
//    59 |         virtual bool GODOT_CPP_FUNC (can_instantiate)() const override;
//       |                                      ^
// modules\sandbox/src/elf/script_elf.h:60:38: error: 'get_base_script' marked 'override' but does not override any member functions
//    60 |         virtual Ref<Script> GODOT_CPP_FUNC (get_base_script)() const override;
//       |                                             ^
// fatal error: too many errors emitted, stopping now [-ferror-limit=]
// 20 errors generated.

#ifdef GODOT_MODULE
	virtual Error GODOT_CPP_FUNC (save)(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) override;
	virtual Error GODOT_CPP_FUNC (set_uid)(const String &p_path, int64_t p_uid) override;
	virtual bool GODOT_CPP_FUNC (recognize)(const Ref<Resource> &p_resource) const override;
	virtual void GODOT_CPP_FUNC (get_recognized_extensions)(const Ref<Resource> &p_resource, List<String> *p_extensions) const override;
	virtual bool GODOT_CPP_FUNC (recognize_path)(const Ref<Resource> &p_resource, const String &p_path) const override;
#else
	virtual Error GODOT_CPP_FUNC (save)(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) override;
	virtual Error GODOT_CPP_FUNC (set_uid)(const String &p_path, int64_t p_uid) override;
	virtual bool GODOT_CPP_FUNC (recognize)(const Ref<Resource> &p_resource) const override;
	virtual PackedStringArray GODOT_CPP_FUNC (get_recognized_extensions)(const Ref<Resource> &p_resource) const override;
	virtual bool GODOT_CPP_FUNC (recognize_path)(const Ref<Resource> &p_resource, const String &p_path) const override;
#endif
};
