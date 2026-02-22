/**************************************************************************/
/*  variant.h                                                             */
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

/**
 * variant.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
#ifndef COMPAT_VARIANT_H
#define COMPAT_VARIANT_H

#ifdef LIMBOAI_MODULE
#include "core/variant/variant.h"
#define VARIANT_EVALUATE(m_op, m_lvalue, m_rvalue, r_ret) r_ret = Variant::evaluate(m_op, m_lvalue, m_rvalue)
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/variant/variant.hpp>
using namespace godot;

#define VARIANT_EVALUATE(m_op, m_lvalue, m_rvalue, r_ret)            \
	{                                                                \
		bool r_valid;                                                \
		Variant::evaluate(m_op, m_lvalue, m_rvalue, r_ret, r_valid); \
	}

#endif // LIMBOAI_GDEXTENSION

// *** SHARED ***

#define VARIANT_IS_ARRAY(m_variant) (m_variant.get_type() >= Variant::ARRAY)
#define VARIANT_IS_NUM(m_variant) (m_variant.get_type() == Variant::INT || m_variant.get_type() == Variant::FLOAT)

void VARIANT_DELETE_IF_OBJECT(const Variant &p_variant);

Variant VARIANT_DEFAULT(Variant::Type p_type);

#endif // COMPAT_VARIANT_H
