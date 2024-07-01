#if __has_include(<bit>)
#include <bit>
#define RISCV_HAS_BITOPS
#endif
#include <cmath>
#include <chrono>
#include <fstream>
#include <mutex>
#if defined(__MINGW32__) || defined(__MINGW64__) || defined(_MSC_VER)
# define YEP_IS_WINDOWS 1
# include "windows/dlfcn.h"
# ifdef _MSC_VER
#  define access _access
#  define unlink _unlink
extern "C" int access(const char* path, int mode);
extern "C" int unlink(const char* path);
#  define R_OK   4       /* Test for read permission.  */
# else // _MSC_VER
#  include <unistd.h>
# endif
#else
# include <dlfcn.h>
# include <unistd.h>
#endif
#include "machine.hpp"
#include "decoder_cache.hpp"
#include "instruction_list.hpp"
#include "safe_instr_loader.hpp"
#include "tr_api.hpp"
#include "tr_types.hpp"
#include "util/crc32.hpp"
#ifdef RISCV_EXT_C
#include "rvc.hpp"
#endif

namespace riscv
{
	static constexpr bool VERBOSE_BLOCKS = false;
	static constexpr bool SCAN_FOR_GP = true;

	inline timespec time_now();
	inline long nanodiff(timespec, timespec);
	#define TIME_POINT(x) \
		[[maybe_unused]] timespec x;  \
		if (options.translate_timing) { \
			asm("" : : : "memory");   \
			x = time_now();           \
			asm("" : : : "memory");   \
		}
	extern void  dylib_close(void* dylib);
	extern void* dylib_lookup(void* dylib, const char*);

	template <int W>
	using binary_translation_init_func = void (*)(const CallbackTable<W>&, void*);
	template <int W>
	static CallbackTable<W> create_bintr_callback_table(const CPU<W>& cpu);

	// Translations that are embeddable in the binary will be added as a source
	// file directly in the project, which allows it to run global constructors.
	// The constructor will register the translation with the binary translator,
	// and we can check against this list when loading translations.
	template <int W>
	struct Mapping {
		address_type<W> addr;
		bintr_block_func<W> handler;
	};

	// This implementation is designed to make sure it's not a global constructor
	// instead it will get zeroed from BSS
	static constexpr size_t MAX_EMBEDDED = 12;
	template <int W>
	struct EmbeddedTranslation {
		uint32_t    hash = 0;
		uint32_t    nmappings = 0;
		const Mapping<W>* mappings = nullptr;
		// NOTE: Pointer to the callback table (which we host here)
		riscv::CallbackTable<W>* api_table = nullptr;
	};
	template <int W>
	struct EmbeddedTranslations {
		std::array<EmbeddedTranslation<W>, MAX_EMBEDDED> translations;
		size_t count = 0;
	};
	template <int W>
	static EmbeddedTranslations<W> registered_embedded_translations;

	template <int W>
	static void register_translation(uint32_t hash, const Mapping<W>* mappings, uint32_t nmappings, riscv::CallbackTable<W>* table_ptr)
	{
		auto& translations = registered_embedded_translations<W>;
		if (translations.count >= MAX_EMBEDDED) {
			throw MachineException(INVALID_PROGRAM, "Too many embedded translations");
		}
		auto& translation = translations.translations[translations.count++];
		translation.hash = hash;
		translation.nmappings = nmappings;
		translation.mappings  = mappings;
		translation.api_table = table_ptr;
	}

	static std::string defines_to_string(const std::unordered_map<std::string, std::string>& cflags)
	{
		std::string defstr;
		for (auto pair : cflags) {
			defstr += " -D" + pair.first + "=" + pair.second;
		}
		return defstr;
	}

template <int W>
inline uint32_t opcode(const TransInstr<W>& ti) {
	return rv32i_instruction{ti.instr}.opcode();
}

template <int W>
inline DecoderData<W>& decoder_entry_at(const DecodedExecuteSegment<W>& exec, address_type<W> addr) {
	return exec.decoder_cache()[addr / DecoderCache<W>::DIVISOR];
}

template <int W>
static std::unordered_map<std::string, std::string> create_defines_for(const Machine<W>& machine, const MachineOptions<W>& options)
{
	// Calculate offset from Machine to each counter
	auto counters = const_cast<Machine<W>&> (machine).get_counters();
	const auto ins_counter_offset = uintptr_t(&counters.first) - uintptr_t(&machine);
	const auto max_counter_offset = uintptr_t(&counters.second) - uintptr_t(&machine);
	const auto arena_offset = uintptr_t(&machine.memory.memory_arena_ptr_ref()) - uintptr_t(&machine);

	// Some executables are loaded at high-memory addresses, which is outside of the memory arena.
	auto arena_end          = machine.memory.memory_arena_size();
	auto initial_rodata_end = machine.memory.initial_rodata_end();
	if (!options.translation_use_arena) {
		initial_rodata_end = 0;
		arena_end = 0x1000;
	}

	std::unordered_map<std::string, std::string> defines;
#if defined(__linux__)
	defines.emplace("RISCV_PLATFORM_LINUX", "1");
#elif defined(__APPLE__)
	defines.emplace("RISCV_PLATFORM_DARWIN", "1");
#elif defined(_WIN32)
	defines.emplace("RISCV_PLATFORM_WINDOWS", "1");
#elif defined(__FreeBSD__)
	defines.emplace("RISCV_PLATFORM_FREEBSD", "1");
#elif defined(__OpenBSD__)
	defines.emplace("RISCV_PLATFORM_OPENBSD", "1");
#endif
	defines.emplace("RISCV_TRANSLATION_DYLIB", std::to_string(W));
	defines.emplace("RISCV_MAX_SYSCALLS", std::to_string(RISCV_SYSCALLS_MAX));
	defines.emplace("RISCV_ARENA_END", std::to_string(arena_end));
	defines.emplace("RISCV_ARENA_ROEND", std::to_string(initial_rodata_end));
	defines.emplace("RISCV_INS_COUNTER_OFF", std::to_string(ins_counter_offset));
	defines.emplace("RISCV_MAX_COUNTER_OFF", std::to_string(max_counter_offset));
	defines.emplace("RISCV_ARENA_OFF", std::to_string(arena_offset));
	if constexpr (atomics_enabled) {
		defines.emplace("RISCV_EXT_A", "1");
	}
	if constexpr (compressed_enabled) {
		defines.emplace("RISCV_EXT_C", "1");
	}
	if constexpr (vector_extension) {
		defines.emplace("RISCV_EXT_VECTOR", std::to_string(vector_extension));
	}
	if constexpr (nanboxing) {
		defines.emplace("RISCV_NANBOXING", "1");
	}
	if (options.translate_trace) {
		// Adding this as a define will change the hash of the translation,
		// so it will be recompiled if the trace option is toggled.
		defines.emplace("RISCV_TRACING", "1");
	}
	if (options.translate_ignore_instruction_limit) {
		defines.emplace("RISCV_IGNORE_INSTRUCTION_LIMIT", "1");
	}
	if constexpr (encompassing_Nbit_arena != 0) {
		defines.emplace("RISCV_NBIT_UNBOUNDED", std::to_string(encompassing_Nbit_arena));
	}
	return defines;
}

template <int W>
int CPU<W>::load_translation(const MachineOptions<W>& options,
	std::string* filename, DecodedExecuteSegment<W>& exec) const
{
	// Binary translation using libtcc doesn't use files
	if constexpr (libtcc_enabled) {
		if (options.translate_enabled)
			return 1;
		else
			return -1;
	}

	// Disable translator by setting options.translate_enabled to false
	// or by setting max blocks to zero.
	if (0 == options.translate_blocks_max || (!options.translate_enabled && !options.translate_enable_embedded)) {
		if (options.verbose_loader) {
			printf("libriscv: Binary translation disabled\n");
		}
		exec.set_binary_translated(nullptr);
		return -1;
	}
	if (exec.is_binary_translated()) {
		throw MachineException(ILLEGAL_OPERATION, "Execute segment already binary translated");
	}

	// Checksum the execute segment + compiler flags
	TIME_POINT(t5);
	const std::string cflags = defines_to_string(create_defines_for(machine(), options));
	extern std::string compile_command(int arch, const std::string& cflags);
	uint32_t checksum = exec.crc32c_hash();
	if (UNLIKELY(checksum == 0)) {
		throw MachineException(INVALID_PROGRAM, "Invalid execute segment hash for translation");
	}
	// Also add the compiler flags to the checksum
	checksum = ~crc32c(~checksum, cflags.c_str(), cflags.size());
	exec.set_translation_hash(checksum);

	// Check if translation is registered
	if (options.translate_enable_embedded)
	{
		for (size_t i = 0; i < registered_embedded_translations<W>.count; i++)
		{
			auto& translation = registered_embedded_translations<W>.translations[i];
			if (translation.hash == checksum)
			{
				if (options.verbose_loader) {
					printf("Found embedded translation for hash %08X\n", checksum);
				}
				*translation.api_table = create_bintr_callback_table(*this);
				exec.reserve_mappings(translation.nmappings);
				for (unsigned i = 0; i < translation.nmappings; i++) {
					const auto& mapping = translation.mappings[i];
					exec.add_mapping(mapping.handler);
					auto& entry = decoder_entry_at(exec, mapping.addr);
					if (mapping.handler != nullptr) {
						entry.instr = i;
						entry.set_bytecode(CPU<W>::computed_index_for(RV32_INSTR_BLOCK_END));
					} else {
						entry.set_bytecode(0x0); /* Invalid opcode */
					}
				}
				return 0;
			}
		}
		if (options.verbose_loader) {
			printf("No embedded translation found for hash %08X\n", checksum);
		}
	}

	if (!options.translate_enabled)
		return -1;

	char filebuffer[256];
	int len = snprintf(filebuffer, sizeof(filebuffer),
		"%s%08X%s", options.translation_prefix.c_str(), checksum, options.translation_suffix.c_str());
	if (len <= 0)
		return -1;

	void* dylib = nullptr;
	if (options.translate_timing) {
		TIME_POINT(t6);
		printf(">> Execute segment hashing took %ld ns\n", nanodiff(t5, t6));
	}

	// Always check if there is an existing file
	if (access(filebuffer, R_OK) == 0) {
		TIME_POINT(t7);
		// Probably not needed, but on Windows there might be some issues
		// with the emulated dlopen() functionality. Let's serialize it.
		static std::mutex dlopen_mutex;
		std::lock_guard<std::mutex> lock(dlopen_mutex);
		dylib = dlopen(filebuffer, RTLD_LAZY);
		if (options.translate_timing) {
			TIME_POINT(t8);
			printf(">> dlopen took %ld ns\n", nanodiff(t7, t8));
		}
	}
	bool must_compile = dylib == nullptr;

#ifndef _MSC_VER
	// If cross compilation is enabled, we should check if all results exist
	for (auto& cc : options.cross_compile) {
		if (std::holds_alternative<MachineTranslationCrossOptions>(cc))
		{
			auto& mingw = std::get<MachineTranslationCrossOptions>(cc);
			const uint32_t hash = checksum;
			const std::string cross_filename = options.translation_filename(
				mingw.cross_prefix, hash, mingw.cross_suffix);
			if (access(cross_filename.c_str(), R_OK) != 0) {
				must_compile = true;
				break; // We must compile at least one of the cross-compiled binaries
			}
		} else if (std::holds_alternative<MachineTranslationEmbeddableCodeOptions>(cc)) {
			must_compile = true;
			break; // We must compile embeddable source code
		} else {
			throw MachineException(INVALID_PROGRAM, "Invalid cross-compile option");
		}
	}
#endif

	// We must compile ourselves
	if (dylib == nullptr) {
		if (filename) *filename = std::string(filebuffer);
		return 1;
	}

	this->activate_dylib(options, exec, dylib);

	if (options.translate_timing) {
		TIME_POINT(t10);
		printf(">> Total binary translation loading time %ld ns\n", nanodiff(t5, t10));
	}

	// If the mingw PE-dll is not found, we must also compile (despite activating the ELF)
	if (must_compile) {
		if (filename) *filename = std::string(filebuffer);
		return 1;
	}
	return 0;
}

static bool is_stopping_instruction(rv32i_instruction instr) {
	if (instr.opcode() == RV32I_JALR || instr.whole == RV32_INSTR_STOP
		|| (instr.opcode() == RV32I_SYSTEM && instr.Itype.funct3 == 0 && instr.Itype.imm == 261) // WFI
	) return true;

#ifdef RISCV_EXT_C
	if (instr.is_compressed()) {
		#define CI_CODE(x, y) ((x << 13) | (y))
		const rv32c_instruction ci { instr };
		if (ci.opcode() == CI_CODE(0b100, 0b10)) { // VARIOUS
			if (ci.CR.rd != 0 && ci.CR.rs2 == 0) {
				return true; // C.JR and C.JALR (aka. RET)
			}
		}
	}
#endif

	return false;
}

template <int W>
void CPU<W>::try_translate(const MachineOptions<W>& options,
	const std::string& filename,
	DecodedExecuteSegment<W>& exec, address_t basepc, address_t endbasepc) const
{
	// Run with VERBOSE=1 to see command and output
	const bool verbose = options.verbose_loader;
	const bool trace_instructions = options.translate_trace;

	// Check if compiling new translations is enabled
	if (!options.translate_invoke_compiler)
		return;

	address_t gp = 0;
	TIME_POINT(t0);
if constexpr (SCAN_FOR_GP) {
	// We assume that GP is initialized with AUIPC,
	// followed by OP_IMM (and maybe OP_IMM32)
	for (address_t pc = basepc; pc < endbasepc; ) {
		const rv32i_instruction instruction
			= read_instruction(exec.exec_data(), pc, endbasepc);
		if (instruction.opcode() == RV32I_AUIPC) {
			const auto auipc = instruction;
			if (auipc.Utype.rd == 3) { // GP
				const rv32i_instruction addi
					= read_instruction(exec.exec_data(), pc + 4, endbasepc);
				if (addi.opcode() == RV32I_OP_IMM && addi.Itype.funct3 == 0x0) {
					//printf("Found OP_IMM: ADDI  rd=%d, rs1=%d\n", addi.Itype.rd, addi.Itype.rs1);
					if (addi.Itype.rd == 3 && addi.Itype.rs1 == 3) { // GP
						gp = pc + auipc.Utype.upper_imm() + addi.Itype.signed_imm();
						break;
					}
				} else {
					gp = pc + auipc.Utype.upper_imm();
					break;
				}
			}
		} // opcode

		if constexpr (compressed_enabled)
			pc += instruction.length();
		else
			pc += 4;
	} // iterator
	if (options.translate_timing) {
		TIME_POINT(t1);
		printf(">> GP scan took %ld ns, GP=0x%lX\n", nanodiff(t0, t1), (long)gp);
	}
} // SCAN_FOR_GP

	// Code block and loop detection
	TIME_POINT(t2);
	static constexpr size_t ITS_TIME_TO_SPLIT = 1'250;
	size_t icounter = 0;
	std::unordered_set<address_type<W>> global_jump_locations;
	std::vector<TransInfo<W>> blocks;

	// Insert the ELF entry point as the first global jump location
	const auto elf_entry = machine().memory.start_address();
	if (elf_entry >= basepc && elf_entry < endbasepc)
		global_jump_locations.insert(elf_entry);

	for (address_t pc = basepc; pc < endbasepc && icounter < options.translate_instr_max; )
	{
		const auto block = pc;
		std::size_t block_insns = 0;

		for (; pc < endbasepc; ) {
			const rv32i_instruction instruction
				= read_instruction(exec.exec_data(), pc, endbasepc);
			if constexpr (compressed_enabled)
				pc += instruction.length();
			else
				pc += 4;
			block_insns++;

			// JALR and STOP are show-stoppers / code-block enders
			if (block_insns >= ITS_TIME_TO_SPLIT && is_stopping_instruction(instruction)) {
				break;
			}
		}

		auto block_end = pc;
		std::unordered_set<address_t> jump_locations;
		std::vector<rv32i_instruction> block_instructions;
		block_instructions.reserve(block_insns);

		// Find jump locations inside block
		for (pc = block; pc < block_end; ) {
			const rv32i_instruction instruction
				= read_instruction(exec.exec_data(), pc, endbasepc);
			const auto opcode = instruction.opcode();
			bool is_jal = false;
			bool is_branch = false;

			address_t location = 0;
			if (opcode == RV32I_JAL) {
				is_jal = true;
				const auto offset = instruction.Jtype.jump_offset();
				location = pc + offset;
			} else if (opcode == RV32I_BRANCH) {
				is_branch = true;
				const auto offset = instruction.Btype.signed_imm();
				location = pc + offset;
			}
#ifdef RISCV_EXT_C
			else if (instruction.is_compressed())
			{
				const rv32c_instruction ci { instruction };

				// Find branch and jump locations
				if (W == 4 && ci.opcode() == CI_CODE(0b001, 0b01)) { // C.JAL
					is_jal = true;
					const int32_t imm = ci.CJ.signed_imm();
					location = pc + imm;
				}
				else if (ci.opcode() == CI_CODE(0b101, 0b01)) { // C.JMP
					is_jal = true;
					const int32_t imm = ci.CJ.signed_imm();
					location = pc + imm;
				}
				else if (ci.opcode() == CI_CODE(0b110, 0b01)) { // C.BEQZ
					is_branch = true;
					const int32_t imm = ci.CB.signed_imm();
					location = pc + imm;
				}
				else if (ci.opcode() == CI_CODE(0b111, 0b01)) { // C.BNEZ
					is_branch = true;
					const int32_t imm = ci.CB.signed_imm();
					location = pc + imm;
				}
			}
#endif

			// detect far JAL, otherwise use as local jump
			if (is_jal) {
				// All JAL target addresses need to be recorded in order
				// to detect function calls
				global_jump_locations.insert(location);

				if (location >= block && location < block_end)
					jump_locations.insert(location);
			}
			// loop detection (negative branch offsets)
			else if (is_branch) {
				// only accept branches relative to current block
				if (location >= block && location < block_end)
					jump_locations.insert(location);
			}

			// Add instruction to block
			block_instructions.push_back(instruction);
			if constexpr (compressed_enabled)
				pc += instruction.length();
			else
				pc += 4;
		} // process block

		// Process block and add it for emission
		const size_t length = block_instructions.size();
		if (length > 0 && icounter + length < options.translate_instr_max)
		{
			if constexpr (VERBOSE_BLOCKS) {
				printf("Block found at %#lX -> %#lX. Length: %zu\n", long(block), long(block_end), length);
				for (auto loc : jump_locations)
					printf("-> Jump to %#lX\n", long(loc));
			}

			blocks.push_back({
				std::move(block_instructions),
				block, block_end,
				basepc, endbasepc,
				gp,
				trace_instructions,
				options.translate_ignore_instruction_limit,
				options.use_shared_execute_segments,
				std::move(jump_locations),
				nullptr, // blocks
				global_jump_locations,
				(uintptr_t)machine().memory.memory_arena_ptr_ref()
			});
			icounter += length;
			// we can't translate beyond this estimate, otherwise
			// the compiler will never finish code generation
			if (blocks.size() >= options.translate_blocks_max)
				break;
		}

		pc = block_end;
	}

	TIME_POINT(t3);
	if (options.translate_timing) {
		printf(">> Code block detection %ld ns\n", nanodiff(t2, t3));
	}

	// Code generation
	std::vector<TransMapping<W>> dlmappings;
	extern const std::string bintr_code;
	std::string code = bintr_code;

	for (auto& block : blocks)
	{
		block.blocks = &blocks;
		auto result = emit(code, block);

		for (auto& mapping : result) {
			dlmappings.push_back(std::move(mapping));
		}
	}
	// Append all instruction handler -> dl function mappings
	// to the footer used by shared libraries
	std::string footer;
	footer += "VISIBLE const uint32_t no_mappings = "
		+ std::to_string(dlmappings.size()) + ";\n";
	footer += R"V0G0N(
struct Mapping {
	addr_t addr;
	ReturnValues (*handler)(CPU*, uint64_t, uint64_t, addr_t);
};
VISIBLE const struct Mapping mappings[] = {
)V0G0N";
	for (const auto& mapping : dlmappings)
	{
		char buffer[128];
		snprintf(buffer, sizeof(buffer), 
			"{0x%lX, %s},\n",
			(long)mapping.addr, mapping.symbol.c_str());
		footer.append(buffer);
	}
	footer += "};\n";

	if (options.translate_timing) {
		TIME_POINT(t4);
		printf(">> Code generation took %ld ns\n", nanodiff(t3, t4));
	}

	if (verbose) {
		printf("Emitted %zu accelerated instructions and %zu functions. GP=0x%lX\n",
			icounter, dlmappings.size(), (long) gp);
	}
	// nothing to compile without mappings
	if (dlmappings.empty()) {
		if (verbose) {
			printf("Binary translator has nothing to compile! No mappings.\n");
		}
		return;
	}

	const auto defines = create_defines_for(machine(), options);
	void* dylib = nullptr;
	// Final shared library loadable code w/footer
	const std::string shared_library_code = code + footer;

	TIME_POINT(t9);
	if constexpr (libtcc_enabled) {
		extern void* libtcc_compile(const std::string&, int arch, const std::unordered_map<std::string, std::string>& defines, const std::string&);
		static std::mutex libtcc_mutex;
		// libtcc uses global state, so we need to serialize compilation
		std::lock_guard<std::mutex> lock(libtcc_mutex);
		dylib = libtcc_compile(shared_library_code, W, defines, options.libtcc1_location);
	} else {
		extern void* compile(const std::string&, int arch, const std::string& cflags, const std::string&);
		extern bool mingw_compile(const std::string&, int arch, const std::string& cflags, const std::string&, const MachineTranslationCrossOptions&);
		const std::string cflags = defines_to_string(defines);

		// If the binary translation has already been loaded, we can skip compilation
		if (exec.is_binary_translated()) {
			dylib = exec.binary_translation_so();
		} else {
			dylib = compile(shared_library_code, W, cflags, filename);
		}

		// Optionally produce cross-compiled binaries
		for (auto& cc : options.cross_compile)
		{
			if (std::holds_alternative<MachineTranslationCrossOptions>(cc))
			{
#ifndef _MSC_VER
				auto& mingw = std::get<MachineTranslationCrossOptions>(cc);
				const uint32_t hash = exec.translation_hash();
				const std::string cross_filename = options.translation_filename(
					mingw.cross_prefix, hash, mingw.cross_suffix);
				mingw_compile(shared_library_code, W, cflags, cross_filename, mingw);
#endif
			}
			else if (std::holds_alternative<MachineTranslationEmbeddableCodeOptions>(cc))
			{
				auto& embed = std::get<MachineTranslationEmbeddableCodeOptions>(cc);
				const uint32_t hash = exec.translation_hash();
				const std::string& embed_filename = options.translation_filename(
					embed.prefix, hash, embed.suffix);
				// Write the embeddable code to a file
				std::ofstream embed_file(embed_filename);
				embed_file << "#define EMBEDDABLE_CODE 1\n"; // Mark as embeddable variant
				for (auto& def : defines) {
					embed_file << "#define " << def.first << " " << def.second << "\n";
				}
				embed_file << code;
				// Construct a footer that self-registers the translation
				const std::string reg_func = "libriscv_register_translation" + std::to_string(W);
				embed_file << R"V0G0N(
				struct Mappings {
					addr_t addr;
					ReturnValues (*handler)(CPU*, uint64_t, uint64_t, addr_t);
				};
				extern "C" void libriscv_register_translation4(uint32_t hash, const Mappings* mappings, uint32_t nmappings, struct CallbackTable*);
				extern "C" void libriscv_register_translation8(uint32_t hash, const Mappings* mappings, uint32_t nmappings, struct CallbackTable*);
				static __attribute__((constructor)) void register_translation() {
					static const Mappings mappings[] = {
				)V0G0N";
				for (const auto& mapping : dlmappings)
				{
					char buffer[128];
					snprintf(buffer, sizeof(buffer), 
						"{0x%lX, %s},\n",
						(long)mapping.addr, mapping.symbol.c_str());
					embed_file << buffer;
				}
				embed_file << "    };\n";
				embed_file << "    " << reg_func << "(" << hash << ", mappings, " << dlmappings.size() << ", &api);\n";
				embed_file << "}\n";
			}
			else
			{
				throw MachineException(INVALID_PROGRAM, "Invalid/unrecognized cross-compile option");
			}
		}
	}

	if (options.translate_timing) {
		TIME_POINT(t10);
		printf(">> Code compilation took %.2f ms\n", nanodiff(t9, t10) / 1e6);
	}

	// Check compilation result
	if (dylib == nullptr) {
		return;
	}

	if (!exec.is_binary_translated()) {
		this->activate_dylib(options, exec, dylib);
	}

	if constexpr (!libtcc_enabled) {
		if (!options.translation_cache) {
			// Delete the program if the shared ELF is unwanted
			unlink(filename.c_str());
		}
	}
	if (options.translate_timing) {
		TIME_POINT(t12);
		printf(">> Binary translation totals %.2f ms\n", nanodiff(t0, t12) / 1e6);
	}
}

template <int W>
void CPU<W>::activate_dylib(const MachineOptions<W>& options, DecodedExecuteSegment<W>& exec, void* dylib) const
{
	TIME_POINT(t11);

	if (!initialize_translated_segment(exec, dylib))
	{
		if constexpr (!libtcc_enabled) {
			// only warn when translation is not already disabled
			if (!options.translate_enabled) {
				fprintf(stderr, "libriscv: Could not find dylib init function\n");
			}
		}
		if (dylib != nullptr)
			dylib_close(dylib);
		exec.set_binary_translated(nullptr);
		return;
	}

	// Map all the functions to instruction handlers
	const uint32_t* no_mappings = (const uint32_t *)dylib_lookup(dylib, "no_mappings");
	const auto* mappings = (const Mapping<W> *)dylib_lookup(dylib, "mappings");

	if (no_mappings == nullptr || mappings == nullptr || *no_mappings > 500000UL) {
		dylib_close(dylib);
		exec.set_binary_translated(nullptr);
		throw MachineException(INVALID_PROGRAM, "Invalid mappings in binary translation program");
	}

	// After this, we should automatically close the dylib on destruction
	exec.set_binary_translated(dylib);

	// Apply mappings to decoder cache
	const auto nmappings = *no_mappings;
	exec.reserve_mappings(nmappings);
	for (unsigned i = 0; i < nmappings; i++) {
		const auto addr = mappings[i].addr;
		if (exec.is_within(addr)) {
			exec.add_mapping(mappings[i].handler);
			auto& entry = decoder_entry_at(exec, addr);
			if (mappings[i].handler != nullptr) {
				entry.instr = i;
				entry.set_bytecode(CPU<W>::computed_index_for(RV32_INSTR_BLOCK_END));
			} else {
				entry.set_bytecode(0x0); /* Invalid opcode */
			}
		} else {
			if (options.verbose_loader) {
				fprintf(stderr, "libriscv: Translation mapping 0x%lX outside execute area 0x%lX-0x%lX\n",
					(long)addr, (long)exec.exec_begin(), (long)exec.exec_end());
			}
			exec.add_mapping([] (CPU<W>&, uint64_t, uint64_t, address_t) -> bintr_block_returns<W> {
				throw MachineException(INVALID_PROGRAM, "Translation mapping outside execute area");
			});
			//throw MachineException(INVALID_PROGRAM, "Translation mapping outside execute area", addr);
			// It might be technically possible to produce some valid instructions in the
			// padding areas, but it's not worth the effort to support this. Ignore.
			continue;
		}
	}

	if (options.translate_timing) {
		TIME_POINT(t12);
		printf(">> Binary translation activation %ld ns\n", nanodiff(t11, t12));
	}
}

template <int W>
CallbackTable<W> create_bintr_callback_table(const CPU<W>& cpu)
{
	return CallbackTable<W>{
		.mem_read = [] (CPU<W>& cpu, address_type<W> addr, unsigned size) -> address_type<W> {
			if constexpr (libtcc_enabled) {
				try {
					switch (size) {
					case 1: return cpu.machine().memory.template read<uint8_t>(addr);
					case 2: return cpu.machine().memory.template read<uint16_t>(addr);
					case 4: return cpu.machine().memory.template read<uint32_t>(addr);
					case 8: return cpu.machine().memory.template read<uint64_t>(addr);
					default: throw MachineException(ILLEGAL_OPERATION, "Invalid memory read size", size);
					}
				} catch (...) {
					cpu.set_current_exception(std::current_exception());
					cpu.machine().stop();
					return 0;
				}
			} else {
				switch (size) {
				case 1: return cpu.machine().memory.template read<uint8_t>(addr);
				case 2: return cpu.machine().memory.template read<uint16_t>(addr);
				case 4: return cpu.machine().memory.template read<uint32_t>(addr);
				case 8: return cpu.machine().memory.template read<uint64_t>(addr);
				default: throw MachineException(ILLEGAL_OPERATION, "Invalid memory read size", size);
				}
			}
		},
		.mem_write = [] (CPU<W>& cpu, address_type<W> addr, address_type<W> value, unsigned size) -> void {
			if constexpr (libtcc_enabled) {
				try {
					switch (size) {
					case 1: cpu.machine().memory.template write<uint8_t>(addr, value); break;
					case 2: cpu.machine().memory.template write<uint16_t>(addr, value); break;
					case 4: cpu.machine().memory.template write<uint32_t>(addr, value); break;
					case 8: cpu.machine().memory.template write<uint64_t>(addr, value); break;
					default: throw MachineException(ILLEGAL_OPERATION, "Invalid memory write size", size);
					}
				} catch (...) {
					cpu.set_current_exception(std::current_exception());
					cpu.machine().stop();
				}
			} else {
				switch (size) {
				case 1: cpu.machine().memory.template write<uint8_t>(addr, value); break;
				case 2: cpu.machine().memory.template write<uint16_t>(addr, value); break;
				case 4: cpu.machine().memory.template write<uint32_t>(addr, value); break;
				case 8: cpu.machine().memory.template write<uint64_t>(addr, value); break;
				default: throw MachineException(ILLEGAL_OPERATION, "Invalid memory write size", size);
				}
			}
		},
		.vec_load = [] (CPU<W>& cpu, int vd, address_type<W> addr) {
#ifdef RISCV_EXT_VECTOR
			auto& rvv = cpu.registers().rvv();
			rvv.get(vd) = cpu.machine().memory.template read<VectorLane> (addr);
#else
			(void)cpu; (void)addr; (void)vd;
#endif
		},
		.vec_store = [] (CPU<W>& cpu, address_type<W> addr, int vd) {
#ifdef RISCV_EXT_VECTOR
			auto& rvv = cpu.registers().rvv();
			cpu.machine().memory.template write<VectorLane> (addr, rvv.get(vd));
#else
			(void)cpu; (void)addr; (void)vd;
#endif
		},
		.syscalls = cpu.machine().syscall_handlers.data(),
		.system_call = [] (CPU<W>& cpu, int sysno) -> int {
			if constexpr (libtcc_enabled) {
				try {
					const auto current_pc = cpu.registers().pc;
					cpu.machine().system_call(sysno);
					return cpu.registers().pc != current_pc || cpu.machine().stopped();
				} catch (...) {
					cpu.set_current_exception(std::current_exception());
					cpu.machine().stop();
					return false;
				}
			} else {
				return false;
			}
		},
		.unknown_syscall = [] (CPU<W>& cpu, address_type<W> sysno) {
			cpu.machine().on_unhandled_syscall(cpu.machine(), sysno);
		},
		.system = [] (CPU<W>& cpu, uint32_t instr) {
			if constexpr (libtcc_enabled) {
				try {
					cpu.machine().system(rv32i_instruction{instr});
				} catch (...) {
					cpu.set_current_exception(std::current_exception());
					cpu.machine().stop();
				}
			} else {
				cpu.machine().system(rv32i_instruction{instr});
			}
		},
		.execute = [] (CPU<W>& cpu, uint32_t instr) -> unsigned {
			const rv32i_instruction rvi{instr};
			if constexpr (libtcc_enabled) {
				try {
					cpu.decode(rvi).handler(cpu, rvi);
					return 0;
				} catch (...) {
					cpu.set_current_exception(std::current_exception());
					return 1;
				}
			} else {
				auto* handler = cpu.decode(rvi).handler;
				handler(cpu, rvi);
				return DecoderData<W>::handler_index_for(handler);
			}
		},
		.execute_handler = [] (CPU<W>& cpu, unsigned index, uint32_t instr) -> unsigned {
			const rv32i_instruction rvi{instr};
			try {
				DecoderData<W>::get_handlers()[index](cpu, rvi);
				return 0;
			} catch (...) {
				cpu.set_current_exception(std::current_exception());
				return 1;
			}
		},
		.handlers = (void (**)(CPU<W>&, uint32_t)) DecoderData<W>::get_handlers(),
		.trigger_exception = [] (CPU<W>& cpu, address_type<W> pc, int e) {
			cpu.registers().pc = pc; // XXX: Set PC to the failing instruction (?)
			if constexpr (libtcc_enabled) {
				// If we're using libtcc, we can't throw C++ exceptions because
				// there's no unwinding support. But we can mark an exception
				// in the CPU state and return back to dispatch.
				try {
					cpu.trigger_exception(e);
				} catch (...) {
					cpu.set_current_exception(std::current_exception());
					// Trigger a slow-path in dispatch (which will check for exceptions)
					cpu.machine().stop();
				}
			} else {
				cpu.trigger_exception(e);
			}
		},
		.trace = [] (CPU<W>& cpu, const char* msg, address_type<W> addr, uint32_t instr) {
			(void)cpu;
			printf("f %s pc 0x%lX instr %08X\n", msg, (long)addr, instr);
		},
		.sqrtf32 = [] (float f) -> float {
			return std::sqrt(f);
		},
		.sqrtf64 = [] (double d) -> double {
			return std::sqrt(d);
		},
		.clz = [] (uint32_t x) -> int {
#ifdef RISCV_HAS_BITOPS
			return std::countl_zero(x);
#else
			return x ? __builtin_clz(x) : 32;
#endif
		},
		.clzl = [] (uint64_t x) -> int {
#ifdef RISCV_HAS_BITOPS
			return std::countl_zero(x);
#else
			return x ? __builtin_clzl(x) : 64;
#endif
		},
		.ctz = [] (uint32_t x) -> int {
#ifdef RISCV_HAS_BITOPS
			return std::countr_zero(x);
#else
			return x ? __builtin_ctz(x) : 0;
#endif
		},
		.ctzl = [] (uint64_t x) -> int {
#ifdef RISCV_HAS_BITOPS
			return std::countr_zero(x);
#else
			return x ? __builtin_ctzl(x) : 0;
#endif
		},
		.cpop = [] (uint32_t x) -> int {
#ifdef RISCV_HAS_BITOPS
			return std::popcount(x);
#else
			return __builtin_popcount(x);
#endif
		},
		.cpopl = [] (uint64_t x) -> int {
#ifdef RISCV_HAS_BITOPS
			return std::popcount(x);
#else
			return __builtin_popcountl(x);
#endif
		},
	};
}

template <int W>
bool CPU<W>::initialize_translated_segment(DecodedExecuteSegment<W>&, void* dylib) const
{
	// NOTE: At some point this must be able to duplicate the dylib
	// in order to be able to share execute segments across machines.

	auto* ptr = dylib_lookup(dylib, "init"); // init() function
	if (ptr == nullptr) {
		return false;
	}

	// Map the API callback table
	auto func = (binary_translation_init_func<W>) ptr;
	func(create_bintr_callback_table<W>(*this), machine().memory.memory_arena_ptr_ref());

	return true;
}

template <int W>
std::string MachineOptions<W>::translation_filename(const std::string& prefix, uint32_t hash, const std::string& suffix)
{
	char buffer[256];
	const int len = snprintf(buffer, sizeof(buffer), "%s%08X%s",
		prefix.c_str(), hash, suffix.c_str());
	return std::string(buffer, len);
}

#ifdef RISCV_32I
	template void CPU<4>::try_translate(const MachineOptions<4>&, const std::string&, DecodedExecuteSegment<4>&, address_t, address_t) const;
	template int CPU<4>::load_translation(const MachineOptions<4>&, std::string*, DecodedExecuteSegment<4>&) const;
	template bool CPU<4>::initialize_translated_segment(DecodedExecuteSegment<4>&, void* dylib) const;
	template void CPU<4>::activate_dylib(const MachineOptions<4>&, DecodedExecuteSegment<4>&, void*) const;
	template std::string MachineOptions<4>::translation_filename(const std::string&, uint32_t, const std::string&);
#endif
#ifdef RISCV_64I
	template void CPU<8>::try_translate(const MachineOptions<8>&, const std::string&, DecodedExecuteSegment<8>&, address_t, address_t) const;
	template int CPU<8>::load_translation(const MachineOptions<8>&, std::string*, DecodedExecuteSegment<8>&) const;
	template bool CPU<8>::initialize_translated_segment(DecodedExecuteSegment<8>&, void* dylib) const;
	template void CPU<8>::activate_dylib(const MachineOptions<8>&, DecodedExecuteSegment<8>&, void*) const;
	template std::string MachineOptions<8>::translation_filename(const std::string&, uint32_t, const std::string&);
#endif

	timespec time_now()
	{
		timespec t;
#ifdef YEP_IS_WINDOWS
		std::chrono::nanoseconds ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::high_resolution_clock::now().time_since_epoch());
		t.tv_sec  = ns.count() / 1000000000;
		t.tv_nsec = ns.count() % 1000000000;
#else
		clock_gettime(CLOCK_MONOTONIC, &t);
#endif
		return t;
	}
	long nanodiff(timespec start_time, timespec end_time)
	{
		return (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
	}
} // riscv

extern "C" {
#ifdef RISCV_32I
	void libriscv_register_translation4(uint32_t hash, const riscv::Mapping<4>* mappings, uint32_t nmappings, riscv::CallbackTable<4>* table_ptr)
	{
		riscv::register_translation<4>(hash, mappings, nmappings, table_ptr);
	}
#endif
#ifdef RISCV_64I
	void libriscv_register_translation8(uint32_t hash, const riscv::Mapping<8>* mappings, uint32_t nmappings, riscv::CallbackTable<8>* table_ptr)
	{
		riscv::register_translation<8>(hash, mappings, nmappings, table_ptr);
	}
#endif
}
