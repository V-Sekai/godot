#include "types.hpp"
#include "rv32i_instr.hpp"
#include <unordered_set>
#include <vector>

namespace riscv
{
	template <int W>
	struct TransInstr;

	template <int W>
	struct TransInfo
	{
		const std::vector<rv32i_instruction> instr;
		address_type<W> basepc;
		address_type<W> endpc;
		address_type<W> segment_basepc;
		address_type<W> segment_endpc;
		address_type<W> gp;
		bool trace_instructions;
		bool ignore_instruction_limit;
		bool use_shared_execute_segments;
		std::unordered_set<address_type<W>> jump_locations;
		// Pointer to all the other blocks (including current)
		std::vector<TransInfo<W>>* blocks = nullptr;

		std::unordered_set<address_type<W>>& global_jump_locations;

		uintptr_t arena_ptr;
	};
}
