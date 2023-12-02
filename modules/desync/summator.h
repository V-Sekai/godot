/**************************************************************************/
/*  summator.h                                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

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
