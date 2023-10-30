/* summator.cpp */

#include "summator.h"

void Desync::_bind_methods() {
	ClassDB::bind_method(D_METHOD("untar", "store_url", "index_url", "output_dir_url", "cache_dir_url"), &Desync::untar);
}

Desync::Desync() {
}