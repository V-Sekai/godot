/**************************************************************************/
/*  instr_helpers.hpp                                                     */
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

#ifndef INSTR_HELPERS_HPP
#define INSTR_HELPERS_HPP

#define ATOMIC_INSTR(x, ...) INSTRUCTION(x, __VA_ARGS__)
#define FLOAT_INSTR(x, ...) INSTRUCTION(x, __VA_ARGS__)
#define VECTOR_INSTR(x, ...) INSTRUCTION(x, __VA_ARGS__)
#define COMPRESSED_INSTR(x, ...) INSTRUCTION(x, __VA_ARGS__)
#define RVINSTR_ATTR RISCV_HOT_PATH()
#define RVINSTR_COLDATTR RISCV_COLD_PATH()
#define RVPRINTR_ATTR RISCV_COLD_PATH()

#define DECODED_ATOMIC(x) DECODED_INSTR(x)
#define DECODED_FLOAT(x) DECODED_INSTR(x)
#define DECODED_VECTOR(x) DECODED_INSTR(x)
#define DECODED_COMPR(x) DECODED_INSTR(x)

#define CI_CODE(x, y) ((x << 13) | (y))
#define CIC2_CODE(x, y) ((x << 12) | (y))

#define RVREGTYPE(x) typename std::remove_reference_t<decltype(x)>::address_t
#define RVTOSIGNED(x) (static_cast<typename std::make_signed<decltype(x)>::type>(x))
#define RVSIGNTYPE(x) typename std::make_signed<RVREGTYPE(x)>::type
#define RVIMM(x, y) y.signed_imm()
#define RVXLEN(x) (8u * sizeof(RVREGTYPE(x)))
#define RVIS32BIT(x) (sizeof(RVREGTYPE(x)) == 4)
#define RVIS64BIT(x) (sizeof(RVREGTYPE(x)) == 8)
#define RVIS128BIT(x) (sizeof(RVREGTYPE(x)) == 16)
#define RVISGE64BIT(x) (sizeof(RVREGTYPE(x)) >= 8)

#endif // INSTR_HELPERS_HPP
