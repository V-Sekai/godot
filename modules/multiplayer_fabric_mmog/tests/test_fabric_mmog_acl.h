/**************************************************************************/
/*  test_fabric_mmog_acl.h                                                */
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
#include "core/os/os.h"
#include "tests/test_macros.h"

namespace TestFabricMMOGAcl {

// Opt-in network test. Set FABRIC_MMOG_NETWORK_TESTS=1 and stand up uro on
// 127.0.0.1:4000 with `Uro.Acl.put({"asset:1", "viewer", "user:1"})` seeded
// to mirror the ExUnit fixture. Verifies that `acl_check` POSTs the ReBAC
// tuple and decodes the `{allowed: bool}` response.
TEST_CASE("[FabricMMOG][Network] acl_check returns allowed for a seeded tuple") {
	if (OS::get_singleton()->get_environment("FABRIC_MMOG_NETWORK_TESTS") != "1") {
		MESSAGE("Skipping: set FABRIC_MMOG_NETWORK_TESTS=1 to enable network tests.");
		return;
	}
	ERR_PRINT_OFF;
	Crypto::load_default_certificates(String());
	ERR_PRINT_ON;

	bool allowed = false;
	String error;
	const Error result = FabricMMOGAsset::acl_check(
			"http://127.0.0.1:4000", "asset:1", "viewer", "user:1",
			allowed, error);
	CHECK_MESSAGE(result == OK, error);
	CHECK(allowed);
}

TEST_CASE("[FabricMMOG][Network] acl_check returns not-allowed for an unseeded tuple") {
	if (OS::get_singleton()->get_environment("FABRIC_MMOG_NETWORK_TESTS") != "1") {
		MESSAGE("Skipping: set FABRIC_MMOG_NETWORK_TESTS=1 to enable network tests.");
		return;
	}
	ERR_PRINT_OFF;
	Crypto::load_default_certificates(String());
	ERR_PRINT_ON;

	bool allowed = true;
	String error;
	const Error result = FabricMMOGAsset::acl_check(
			"http://127.0.0.1:4000", "asset:999", "viewer", "user:1",
			allowed, error);
	CHECK_MESSAGE(result == OK, error);
	CHECK_FALSE(allowed);
}

} // namespace TestFabricMMOGAcl
