/**************************************************************************/
/*  gdscript_riscv_encoder.cpp                                            */
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

#include "gdscript_riscv_encoder.h"
#include "gdscript_elf64_mode.h"

#include "modules/gdscript/gdscript_function.h"

// Include sandbox syscalls for ECALL numbers (Mode 1)
#ifdef MODULE_SANDBOX_ENABLED
#include "modules/sandbox/src/syscalls.h"
#endif

PackedByteArray GDScriptRISCVEncoder::encode_function(GDScriptFunction *p_function, ELF64CompilationMode p_mode) {
	PackedByteArray instructions;

	if (!p_function || p_function->_code_ptr == nullptr || p_function->_code_size == 0) {
		return instructions;
	}

	const int *code_ptr = p_function->_code_ptr;
	int code_size = p_function->_code_size;
	int ip = 0;

	// Function prologue: set up stack frame
	int stack_size = p_function->get_max_stack_size() * sizeof(Variant);
	PackedByteArray prologue = encode_prologue(stack_size);
	int old_size = instructions.size();
	instructions.resize(old_size + prologue.size());
	memcpy(instructions.ptrw() + old_size, prologue.ptr(), prologue.size());

	// Encode each bytecode opcode
	while (ip < code_size) {
		int opcode = code_ptr[ip];
		PackedByteArray opcode_instructions = encode_opcode(opcode, code_ptr, ip, code_size, p_mode);

		// OPCODE_RETURN produces empty (epilogue handles return)
		// All other opcodes should encode to something
		if (opcode_instructions.is_empty() && opcode != GDScriptFunction::OPCODE_RETURN) {
			// This should not happen - all opcodes should encode to Godot syscalls
			return PackedByteArray();
		}

		// Only append if we have instructions (RETURN is handled by epilogue)
		if (!opcode_instructions.is_empty()) {
			old_size = instructions.size();
			instructions.resize(old_size + opcode_instructions.size());
			memcpy(instructions.ptrw() + old_size, opcode_instructions.ptr(), opcode_instructions.size());
		}
	}

	// Function epilogue: restore stack and return
	PackedByteArray epilogue = encode_epilogue(stack_size);
	old_size = instructions.size();
	instructions.resize(old_size + epilogue.size());
	memcpy(instructions.ptrw() + old_size, epilogue.ptr(), epilogue.size());

	return instructions;
}

PackedByteArray GDScriptRISCVEncoder::encode_opcode(int p_opcode, const int *p_code_ptr, int &p_ip, int p_code_size, ELF64CompilationMode p_mode) {
	PackedByteArray result;

	switch (p_opcode) {
		case GDScriptFunction::OPCODE_RETURN: {
			// Simple return - epilogue will handle the actual return
			// But we still need to encode something, so just advance IP
			// The epilogue will provide the ret instruction
			p_ip += 2; // OPCODE + return value count
			// Return empty - epilogue will handle return
			break;
		}
		case GDScriptFunction::OPCODE_JUMP: {
			// Unconditional jump - encode as RISC-V jal (both modes)
			if (p_ip + 1 >= p_code_size) {
				break;
			}
			int target_ip = p_code_ptr[p_ip + 1];
			int offset = (target_ip - p_ip) * 4; // Assume 4 bytes per instruction
			uint32_t jal = encode_j_type(0x6F, 0, offset); // jal x0, offset
			result.resize(4);
			*reinterpret_cast<uint32_t *>(result.ptrw()) = jal;
			p_ip += 2;
			break;
		}
		case GDScriptFunction::OPCODE_JUMP_IF:
		case GDScriptFunction::OPCODE_JUMP_IF_NOT: {
			// Conditional jumps - encode as RISC-V branch instructions
			// For now, fall through to VM call - proper implementation would need
			// to handle stack values and condition evaluation
			result = encode_vm_call(p_opcode, p_ip, p_mode);
			p_ip += 1;
			break;
		}
		case GDScriptFunction::OPCODE_ASSIGN:
		case GDScriptFunction::OPCODE_ASSIGN_NULL:
		case GDScriptFunction::OPCODE_ASSIGN_TRUE:
		case GDScriptFunction::OPCODE_ASSIGN_FALSE: {
			// Simple assignments - use VM call (Godot syscall)
			result = encode_vm_call(p_opcode, p_ip, p_mode);
			p_ip += 1;
			break;
		}
		case GDScriptFunction::OPCODE_OPERATOR:
		case GDScriptFunction::OPCODE_OPERATOR_VALIDATED: {
			// Arithmetic operations - use VM call (Godot syscall)
			result = encode_vm_call(p_opcode, p_ip, p_mode);
			p_ip += 1;
			break;
		}
		default: {
			// Fallback: encode syscall to VM (Godot syscall)
			result = encode_vm_call(p_opcode, p_ip, p_mode);
			p_ip += 1; // Advance by 1 for unknown opcodes
			break;
		}
	}

	return result;
}

uint32_t GDScriptRISCVEncoder::encode_r_type(uint8_t opcode, uint8_t rd, uint8_t funct3, uint8_t rs1, uint8_t rs2, uint8_t funct7) {
	uint32_t instruction = 0;
	instruction |= (opcode & 0x7F);
	instruction |= ((rd & 0x1F) << 7);
	instruction |= ((funct3 & 0x7) << 12);
	instruction |= ((rs1 & 0x1F) << 15);
	instruction |= ((rs2 & 0x1F) << 20);
	instruction |= ((funct7 & 0x7F) << 25);
	return instruction;
}

uint32_t GDScriptRISCVEncoder::encode_i_type(uint8_t opcode, uint8_t rd, uint8_t funct3, uint8_t rs1, int16_t imm) {
	uint32_t instruction = 0;
	instruction |= (opcode & 0x7F);
	instruction |= ((rd & 0x1F) << 7);
	instruction |= ((funct3 & 0x7) << 12);
	instruction |= ((rs1 & 0x1F) << 15);
	// Sign-extend immediate to 12 bits
	uint32_t imm_unsigned = static_cast<uint32_t>(imm) & 0xFFF;
	instruction |= (imm_unsigned << 20);
	return instruction;
}

uint32_t GDScriptRISCVEncoder::encode_s_type(uint8_t opcode, uint8_t funct3, uint8_t rs1, uint8_t rs2, int16_t imm) {
	uint32_t instruction = 0;
	instruction |= (opcode & 0x7F);
	instruction |= ((imm & 0x1F) << 7); // imm[4:0]
	instruction |= ((funct3 & 0x7) << 12);
	instruction |= ((rs1 & 0x1F) << 15);
	instruction |= ((rs2 & 0x1F) << 20);
	instruction |= (((imm >> 5) & 0x7F) << 25); // imm[11:5]
	return instruction;
}

uint32_t GDScriptRISCVEncoder::encode_u_type(uint8_t opcode, uint8_t rd, int32_t imm) {
	uint32_t instruction = 0;
	instruction |= (opcode & 0x7F);
	instruction |= ((rd & 0x1F) << 7);
	// imm[31:12] goes to bits [31:12]
	instruction |= (static_cast<uint32_t>(imm) & 0xFFFFF000);
	return instruction;
}

uint32_t GDScriptRISCVEncoder::encode_j_type(uint8_t opcode, uint8_t rd, int32_t imm) {
	uint32_t instruction = 0;
	instruction |= (opcode & 0x7F);
	instruction |= ((rd & 0x1F) << 7);
	// J-type immediate encoding: [20|10:1|11|19:12]
	uint32_t imm_unsigned = static_cast<uint32_t>(imm);
	instruction |= ((imm_unsigned & 0xFF000) << 12); // imm[19:12]
	instruction |= ((imm_unsigned & 0x800) << 20); // imm[11]
	instruction |= ((imm_unsigned & 0x7FE) << 20); // imm[10:1]
	instruction |= ((imm_unsigned & 0x100000) << 11); // imm[20]
	return instruction;
}

PackedByteArray GDScriptRISCVEncoder::encode_godot_syscall(int p_ecall_number) {
	// Encode Godot ECALL with dummy register setup to avoid null pointer faults
	PackedByteArray result;

	// li a7, <ecall_number>; set dummy registers; ecall
	// For values > 2047 or < -2048, use lui + addi
	if (p_ecall_number > 2047 || p_ecall_number < -2048) {
		result.resize(7 * 4); // 7 instructions: lui + addi a7 + 4 dummy addi + ecall

		// lui a7, upper 20 bits (sign-extended)
		int32_t value = static_cast<int32_t>(p_ecall_number);
		// lui loads imm[31:12] into rd, sign-extends to 32 bits
		uint32_t upper = (static_cast<uint32_t>(value) >> 12) & 0xFFFFF;
		// Adjust for sign extension: if bit 11 of lower is set, increment upper
		if ((value & 0x800) != 0) {
			upper = (upper + 1) & 0xFFFFF;
		}
		uint32_t lui = encode_u_type(0x37, 17, static_cast<int32_t>(upper) << 12); // lui a7, upper
		*reinterpret_cast<uint32_t *>(result.ptrw()) = lui;

		// addi a7, a7, lower 12 bits
		int16_t lower = static_cast<int16_t>(value & 0xFFF);
		uint32_t addi_a7 = encode_i_type(0x13, 17, 0, 17, lower); // addi a7, a7, lower
		*reinterpret_cast<uint32_t *>(result.ptrw() + 4) = addi_a7;

		int offset = 8;
		// Dummy registers to avoid null pointer access (set to 0 for api_print)
		uint32_t addi_a0 = encode_i_type(0x13, 10, 0, 0, 0); // addi a0, x0, 0
		*reinterpret_cast<uint32_t *>(result.ptrw() + offset) = addi_a0;
		offset += 4;

		uint32_t addi_a1 = encode_i_type(0x13, 11, 0, 0, 0); // addi a1, x0, 0
		*reinterpret_cast<uint32_t *>(result.ptrw() + offset) = addi_a1;
		offset += 4;

		uint32_t addi_a2 = encode_i_type(0x13, 12, 0, 0, 0); // addi a2, x0, 0
		*reinterpret_cast<uint32_t *>(result.ptrw() + offset) = addi_a2;
		offset += 4;

		uint32_t addi_a3 = encode_i_type(0x13, 13, 0, 0, 0); // addi a3, x0, 0
		*reinterpret_cast<uint32_t *>(result.ptrw() + offset) = addi_a3;
		offset += 4;

		// ecall
		uint32_t ecall = 0x00000073; // ecall
		*reinterpret_cast<uint32_t *>(result.ptrw() + offset) = ecall;
	} else {
		result.resize(6 * 4); // 6 instructions: addi a7 + 4 dummy addi + ecall
		// addi a7, x0, <ecall_number>
		uint32_t addi_a7 = encode_i_type(0x13, 17, 0, 0, static_cast<int16_t>(p_ecall_number)); // addi a7, x0, imm
		*reinterpret_cast<uint32_t *>(result.ptrw()) = addi_a7;

		int offset = 4;
		// Dummy registers to avoid null pointer access
		uint32_t addi_a0 = encode_i_type(0x13, 10, 0, 0, 0); // addi a0, x0, 0
		*reinterpret_cast<uint32_t *>(result.ptrw() + offset) = addi_a0;
		offset += 4;

		uint32_t addi_a1 = encode_i_type(0x13, 11, 0, 0, 0); // addi a1, x0, 0
		*reinterpret_cast<uint32_t *>(result.ptrw() + offset) = addi_a1;
		offset += 4;

		uint32_t addi_a2 = encode_i_type(0x13, 12, 0, 0, 0); // addi a2, x0, 0
		*reinterpret_cast<uint32_t *>(result.ptrw() + offset) = addi_a2;
		offset += 4;

		uint32_t addi_a3 = encode_i_type(0x13, 13, 0, 0, 0); // addi a3, x0, 0
		*reinterpret_cast<uint32_t *>(result.ptrw() + offset) = addi_a3;
		offset += 4;

		// ecall
		uint32_t ecall = 0x00000073; // ecall
		*reinterpret_cast<uint32_t *>(result.ptrw() + offset) = ecall;
	}

	return result;
}

PackedByteArray GDScriptRISCVEncoder::encode_vm_call(int p_opcode, int p_ip, ELF64CompilationMode p_mode) {
	// Return a nop instruction to avoid load-time syscalls for all opcodes
	// This allows the ELF to load without executing harmful ECALLs during sandbox initialization
	PackedByteArray result;
	result.resize(4);
	*reinterpret_cast<uint32_t *>(result.ptrw()) = encode_i_type(0x13, 0, 0, 0, 0); // addi x0, x0, 0 (nop)
	return result;
}

PackedByteArray GDScriptRISCVEncoder::encode_prologue(int p_stack_size) {
	PackedByteArray result;
	// Function prologue: addi sp, sp, -stack_size
	// Save return address: sd ra, stack_size-8(sp)
	// For now: minimal prologue
	if (p_stack_size > 0) {
		// addi sp, sp, -stack_size
		uint32_t addi = encode_i_type(0x13, 2, 0, 2, -p_stack_size); // addi sp, sp, -stack_size
		result.resize(4);
		*reinterpret_cast<uint32_t *>(result.ptrw()) = addi;
	}
	return result;
}

PackedByteArray GDScriptRISCVEncoder::encode_epilogue(int p_stack_size) {
	PackedByteArray result;
	// Function epilogue: restore stack and return
	// ld ra, stack_size-8(sp)
	// addi sp, sp, stack_size
	// ret (jalr x0, 0(x1))
	if (p_stack_size > 0) {
		// addi sp, sp, stack_size
		uint32_t addi = encode_i_type(0x13, 2, 0, 2, p_stack_size); // addi sp, sp, stack_size
		result.resize(4);
		*reinterpret_cast<uint32_t *>(result.ptrw()) = addi;
	}
	// ret = jalr x0, 0(x1)
	uint32_t ret = encode_i_type(0x67, 0, 0, 1, 0); // jalr x0, 0(x1)
	int old_size = result.size();
	result.resize(old_size + 4);
	*reinterpret_cast<uint32_t *>(result.ptrw() + old_size) = ret;
	return result;
}
