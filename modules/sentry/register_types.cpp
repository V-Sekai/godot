#include "register_types.h"

#include "core/config/engine.h"
#include "core/object/class_db.h"
#include "sentry_object.h"

static Sentry *_sentry = NULL;

void initialize_sentry_module(ModuleInitializationLevel p_level) {
  if (p_level != MODULE_INITIALIZATION_LEVEL_SERVERS) {
    return;
  }
  ClassDB::register_class<Sentry>();

  // Automatically create singleton instance, this is so that sentry can be
  // initialized as soon as possible to catch any startup crashes
  _sentry = memnew(Sentry);
  Engine::get_singleton()->add_singleton(
      Engine::Singleton("Sentry", Sentry::get_singleton()));
}

void uninitialize_sentry_module(ModuleInitializationLevel p_level) {
  if (p_level != MODULE_INITIALIZATION_LEVEL_SERVERS) {
    return;
  }
  memdelete(_sentry);
}
