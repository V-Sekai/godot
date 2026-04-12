/**************************************************************************/
/*  test_fabric_mmog_fetch.h                                              */
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

#include "core/crypto/crypto.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "tests/test_macros.h"

namespace TestFabricMMOGFetch {

// Opt-in network test. Set FABRIC_MMOG_NETWORK_TESTS=1 to run.
//
// End-to-end check against the real V-Sekai desync store: fetch the macOS
// build index (the smallest published index — a single-chunk `.caidx`
// pointing at the casync archive chunk), reassemble the referent, and
// verify it was written to the requested output directory. The
// single-chunk shape keeps the test cheap (~1 HTTPS GET for the index +
// 1 for the chunk), but the code path exercises every step of
// `fetch_asset`: parse_caibx → build_chunk_url → http_get_blocking →
// decompress_and_verify_chunk → assemble_from_caibx → file write.
TEST_CASE("[FabricMMOG][Network] fetch_asset reassembles a real V-Sekai index end-to-end") {
	if (OS::get_singleton()->get_environment("FABRIC_MMOG_NETWORK_TESTS") != "1") {
		MESSAGE("Skipping: set FABRIC_MMOG_NETWORK_TESTS=1 to enable network tests.");
		return;
	}

	// Idempotent: the mbedtls implementation ERR_FAILs if called twice, so
	// only load once per test run. The HTTP test may already have done it.
	ERR_PRINT_OFF;
	Crypto::load_default_certificates(String());
	ERR_PRINT_ON;

	const String store_url = FabricMMOGAsset::DEFAULT_STORE_URL;
	const String index_url = "https://raw.githubusercontent.com/V-Sekai/casync-v-sekai-game/main/vsekai_game_macos_x86_64.caidx";

	const String tmp_dir = OS::get_singleton()->get_cache_path().path_join("fabric_mmog_test");
	const String output_dir = tmp_dir.path_join("out");
	const String cache_dir = tmp_dir.path_join("cache");

	// Clean any leftover state from a previous run so the assertions below
	// are reliable regardless of how the host was used.
	if (DirAccess::dir_exists_absolute(output_dir)) {
		DirAccess::remove_absolute(output_dir.path_join("vsekai_game_macos_x86_64"));
	}

	Ref<FabricMMOGAsset> asset;
	asset.instantiate();
	const String result_path = asset->fetch_asset(
			store_url, index_url, output_dir, cache_dir);

	REQUIRE_FALSE(result_path.is_empty());
	CHECK(FileAccess::exists(result_path));

	Ref<FileAccess> f = FileAccess::open(result_path, FileAccess::READ);
	REQUIRE(f.is_valid());
	CHECK(f->get_length() > 0);
}

} // namespace TestFabricMMOGFetch
