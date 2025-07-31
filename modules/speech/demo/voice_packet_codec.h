/**************************************************************************/
/*  voice_packet_codec.h                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#ifndef VOICE_PACKET_CODEC_H
#define VOICE_PACKET_CODEC_H

#include "core/variant/variant.h"

class VoicePacketCodec {
public:
	// Binary encoding utilities ported from network_layer.gd
	static int bitand(int a, int b);
	static PackedByteArray encode_16_bit_value(int p_value);
	static int decode_16_bit_value(const PackedByteArray &p_buffer);
	static PackedByteArray encode_24_bit_value(int p_value);
	static int decode_24_bit_value(const PackedByteArray &p_buffer);
	
	// Voice packet encoding/decoding
	static PackedByteArray encode_voice_packet(int p_index, const PackedByteArray &p_voice_buffer);
	static Array decode_voice_packet(const PackedByteArray &p_voice_buffer);
};

#endif // VOICE_PACKET_CODEC_H
