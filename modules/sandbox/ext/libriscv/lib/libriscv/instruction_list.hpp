/**************************************************************************/
/*  instruction_list.hpp                                                  */
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

#ifndef INSTRUCTION_LIST_HPP
#define INSTRUCTION_LIST_HPP

#define RV32I_LOAD 0b0000011
#define RV32I_STORE 0b0100011
#define RV32I_BRANCH 0b1100011
#define RV32I_JALR 0b1100111
#define RV32I_JAL 0b1101111
#define RV32I_OP_IMM 0b0010011
#define RV32I_OP 0b0110011
#define RV32I_SYSTEM 0b1110011
#define RV32I_LUI 0b0110111
#define RV32I_AUIPC 0b0010111
#define RV32I_FENCE 0b0001111
#define RV64I_OP_IMM32 0b0011011
#define RV64I_OP32 0b0111011
#define RV128I_OP_IMM64 0b1011011
#define RV128I_OP64 0b1111011

#define RV32F_LOAD 0b0000111
#define RV32F_STORE 0b0100111
#define RV32F_FMADD 0b1000011
#define RV32F_FMSUB 0b1000111
#define RV32F_FNMSUB 0b1001011
#define RV32F_FNMADD 0b1001111
#define RV32F_FPFUNC 0b1010011
#define RV32A_ATOMIC 0b0101111

#define RV32F__FADD 0b00000
#define RV32F__FSUB 0b00001
#define RV32F__FMUL 0b00010
#define RV32F__FDIV 0b00011
#define RV32F__FSGNJ_NX 0b00100
#define RV32F__FMIN_MAX 0b00101
#define RV32F__FSQRT 0b01011
#define RV32F__FEQ_LT_LE 0b10100
#define RV32F__FCVT_SD_DS 0b01000
#define RV32F__FCVT_W_SD 0b11000
#define RV32F__FCVT_SD_W 0b11010
#define RV32F__FMV_X_W 0b11100
#define RV32F__FMV_W_X 0b11110

#define RV32V_OP 0b1010111

#define RV32_INSTR_STOP 0x7ff00073

#endif // INSTRUCTION_LIST_HPP
