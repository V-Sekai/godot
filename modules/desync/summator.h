/* summator.h */

#ifndef SUMMATOR_H
#define SUMMATOR_H

#include "core/error/error_list.h"
#include "core/object/ref_counted.h"
#include "libdesync_c_interface.h"
#include <stdio.h>

class Desync : public RefCounted {
	GDCLASS(Desync, RefCounted);

protected:
	static void _bind_methods();

public:
	Error untar(String p_store_url, String p_index_url, String p_output_dir_url, String p_cache_dir_url) {
		int result = DesyncUntar(p_store_url.utf8().ptrw(),
				p_index_url.utf8().ptrw(),
				p_output_dir_url.utf8().ptrw(),
				p_cache_dir_url.utf8().ptrw());
		if (result != 0) {
			printf("Error: storeUrl, indexUrl, and outputDir are required\n");
            return ERR_INVALID_PARAMETER;
		}
		return OK;
	}
	Desync();
};

#endif // SUMMATOR_H
