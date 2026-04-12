/**************************************************************************/
/*  test_fabric_mmog_chunk_url.h                                          */
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

#pragma once

#include "../fabric_mmog_asset.h"

#include "tests/test_macros.h"

namespace TestFabricMMOGChunkUrl {

// Real chunk observed in https://github.com/V-Sekai/casync-v-sekai-game store
// at shard 0006: its filename is <hex>.cacnk, so build_chunk_url() should
// reproduce the full URL exactly.
static void _vsekai_sample_id(uint8_t r_id[FabricMMOGAsset::CHUNK_ID_BYTES]) {
	static const uint8_t kId[FabricMMOGAsset::CHUNK_ID_BYTES] = {
		0x00,
		0x06,
		0xfc,
		0xc5,
		0x5e,
		0xba,
		0xbc,
		0x25,
		0xfd,
		0xdb,
		0x54,
		0xa6,
		0xd2,
		0xf5,
		0x95,
		0xee,
		0xde,
		0xe3,
		0xfb,
		0x87,
		0x6e,
		0x73,
		0x7d,
		0x21,
		0x08,
		0x28,
		0xb6,
		0xb8,
		0x5d,
		0x81,
		0x36,
		0x75,
	};
	memcpy(r_id, kId, sizeof(kId));
}

TEST_CASE("[FabricMMOG] build_chunk_url produces the V-Sekai store layout") {
	uint8_t id[FabricMMOGAsset::CHUNK_ID_BYTES];
	_vsekai_sample_id(id);

	const String url = FabricMMOGAsset::build_chunk_url(
			"https://raw.githubusercontent.com/V-Sekai/casync-v-sekai-game/main/store", id);

	CHECK(url == "https://raw.githubusercontent.com/V-Sekai/casync-v-sekai-game/main/store/0006/"
				 "0006fcc55ebabc25fddb54a6d2f595eedee3fb876e737d210828b6b85d813675.cacnk");
}

TEST_CASE("[FabricMMOG] build_chunk_url strips a trailing slash on the store URL") {
	uint8_t id[FabricMMOGAsset::CHUNK_ID_BYTES];
	_vsekai_sample_id(id);

	const String with_slash = FabricMMOGAsset::build_chunk_url(
			"https://raw.githubusercontent.com/V-Sekai/casync-v-sekai-game/main/store/", id);
	const String without_slash = FabricMMOGAsset::build_chunk_url(
			"https://raw.githubusercontent.com/V-Sekai/casync-v-sekai-game/main/store", id);

	CHECK(with_slash == without_slash);
}

TEST_CASE("[FabricMMOG] build_chunk_url lowercases a zero chunk ID") {
	uint8_t id[FabricMMOGAsset::CHUNK_ID_BYTES] = {};

	const String url = FabricMMOGAsset::build_chunk_url("http://store", id);

	String expected = "http://store/0000/";
	for (int i = 0; i < FabricMMOGAsset::CHUNK_ID_BYTES; i++) {
		expected += "00";
	}
	expected += ".cacnk";
	CHECK(url == expected);
}

} // namespace TestFabricMMOGChunkUrl
