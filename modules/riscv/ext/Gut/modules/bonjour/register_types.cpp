/* register_types.cpp */

#include "register_types.h"
#include "bonjour.h"

void register_bonjour_types() {
	ClassDB::register_class<Bonjour>();
}

void unregister_bonjour_types() {
   //nothing to do here
}
