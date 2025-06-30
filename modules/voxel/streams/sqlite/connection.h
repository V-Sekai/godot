/**************************************************************************/
/*  connection.h                                                          */
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

#include "../../storage/voxel_buffer.h"
#include "../../util/string/std_string.h"
#include "../voxel_stream.h"
#include "block_location.h"

struct sqlite3;
struct sqlite3_stmt;

namespace zylann::voxel::sqlite {

// One connection to the database, with our prepared statements
class Connection {
public:
	static constexpr int VERSION_V0 = 0;
	static constexpr int VERSION_V1 = 1;
	static constexpr int VERSION_LATEST = VERSION_V1;

	struct Meta {
		int version = -1;
		int block_size_po2 = 0;
		BlockLocation::CoordinateFormat coordinate_format =
				// Default as of V0
				BlockLocation::FORMAT_INT64_X16_Y16_Z16_L16;

		struct Channel {
			VoxelBuffer::Depth depth;
			bool used = false;
		};

		FixedArray<Channel, VoxelBuffer::MAX_CHANNELS> channels;
	};

	enum BlockType { //
		VOXELS,
		INSTANCES
	};

	Connection();
	~Connection();

	bool open(const char *fpath, const BlockLocation::CoordinateFormat preferred_coordinate_format);
	void close();

	bool is_open() const {
		return _db != nullptr;
	}

	// Returns the file path from SQLite
	const char *get_file_path() const;

	// Return the file path that was used to open the connection.
	// You may use this one if you want determinism, as SQLite seems to globalize its path.
	const char *get_opened_file_path() const {
		return _opened_path.c_str();
	}

	bool begin_transaction();
	bool end_transaction();

	bool save_block(const BlockLocation loc, const Span<const uint8_t> block_data, const BlockType type);

	VoxelStream::ResultCode load_block(
			const BlockLocation loc,
			StdVector<uint8_t> &out_block_data,
			const BlockType type
	);

	bool load_all_blocks(
			void *callback_data,
			void (*process_block_func)(
					void *callback_data,
					BlockLocation location,
					Span<const uint8_t> voxel_data,
					Span<const uint8_t> instances_data
			)
	);

	bool load_all_block_keys(
			void *callback_data,
			void (*process_block_func)(void *callback_data, BlockLocation location)
	);

	const Meta &get_meta() const {
		return _meta;
	}

	void migrate_to_latest_version();

private:
	int load_version();
	Meta load_meta();
	void save_meta(Meta meta);
	bool migrate_to_next_version();
	bool migrate_from_v0_to_v1();

	StdString _opened_path;
	Meta _meta;
	sqlite3 *_db = nullptr;
	sqlite3_stmt *_load_version_statement = nullptr;
	sqlite3_stmt *_begin_statement = nullptr;
	sqlite3_stmt *_end_statement = nullptr;
	sqlite3_stmt *_update_voxel_block_statement = nullptr;
	sqlite3_stmt *_get_voxel_block_statement = nullptr;
	sqlite3_stmt *_update_instance_block_statement = nullptr;
	sqlite3_stmt *_get_instance_block_statement = nullptr;
	sqlite3_stmt *_load_meta_statement = nullptr;
	sqlite3_stmt *_save_meta_statement = nullptr;
	sqlite3_stmt *_load_channels_statement = nullptr;
	sqlite3_stmt *_save_channel_statement = nullptr;
	sqlite3_stmt *_load_all_blocks_statement = nullptr;
	sqlite3_stmt *_load_all_block_keys_statement = nullptr;
};

} // namespace zylann::voxel::sqlite
