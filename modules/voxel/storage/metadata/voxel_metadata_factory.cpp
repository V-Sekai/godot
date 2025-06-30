/**************************************************************************/
/*  voxel_metadata_factory.cpp                                            */
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

#include "voxel_metadata_factory.h"
#include "../../util/errors.h"
#include "../../util/string/format.h"

namespace zylann::voxel {

namespace {
VoxelMetadataFactory g_voxel_metadata_factory;
}

VoxelMetadataFactory &VoxelMetadataFactory::get_singleton() {
	return g_voxel_metadata_factory;
}

VoxelMetadataFactory::VoxelMetadataFactory() {
	fill(_constructors, (ConstructorFunc) nullptr);
}

void VoxelMetadataFactory::add_constructor(uint8_t type, ConstructorFunc ctor) {
	ZN_ASSERT(type >= VoxelMetadata::TYPE_CUSTOM_BEGIN);
	const unsigned int i = type - VoxelMetadata::TYPE_CUSTOM_BEGIN;
	ZN_ASSERT_MSG(_constructors[i] == nullptr, "Type already registered");
	_constructors[i] = ctor;
}

void VoxelMetadataFactory::remove_constructor(uint8_t type) {
	ZN_ASSERT(type >= VoxelMetadata::TYPE_CUSTOM_BEGIN);
	const unsigned int i = type - VoxelMetadata::TYPE_CUSTOM_BEGIN;
	ZN_ASSERT_MSG(_constructors[i] != nullptr, "Type not registered");
	_constructors[i] = nullptr;
}

ICustomVoxelMetadata *VoxelMetadataFactory::try_construct(uint8_t type) const {
	ZN_ASSERT_RETURN_V_MSG(
			type >= VoxelMetadata::TYPE_CUSTOM_BEGIN, nullptr, format("Invalid custom metadata type {}", type)
	);
	const unsigned int i = type - VoxelMetadata::TYPE_CUSTOM_BEGIN;

	const ConstructorFunc ctor = _constructors[i];
	ZN_ASSERT_RETURN_V_MSG(ctor != nullptr, nullptr, format("Custom metadata constructor not found for type {}", type));

	ICustomVoxelMetadata *m = ctor();
	ZN_ASSERT_RETURN_V_MSG(
			m != nullptr, nullptr, format("Custom metadata constructor for type {} returned nullptr", type)
	);

	return ctor();
}

} // namespace zylann::voxel
