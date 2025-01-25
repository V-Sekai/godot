/**************************************************************************/
/*  image_saver_dds.cpp                                                   */
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

#include "image_saver_dds.h"

#include "core/io/file_access.h"
#include "core/io/stream_peer.h"

// Reference: https://docs.microsoft.com/en-us/windows/win32/direct3ddds/dds-header

enum {
	DDS_MAGIC = 0x20534444,
	DDSD_PITCH = 0x00000008,
	DDSD_LINEARSIZE = 0x00080000,
	DDSD_MIPMAPCOUNT = 0x00020000,
	DDPF_ALPHAPIXELS = 0x00000001,
	DDPF_ALPHAONLY = 0x00000002,
	DDPF_FOURCC = 0x00000004,
	DDPF_RGB = 0x00000040,
	DDPF_RG_SNORM = 0x00080000,
	DDSC2_CUBEMAP = 0x200,
	DDSC2_VOLUME = 0x200000,
	DX10D_1D = 2,
	DX10D_2D = 3,
	DX10D_3D = 4,
};

// Reference: https://learn.microsoft.com/en-us/windows/win32/api/dxgiformat/ne-dxgiformat-dxgi_format
enum DXGIFormat {
	DXGI_R32G32B32A32_FLOAT = 2,
	DXGI_R32G32B32_FLOAT = 6,
	DXGI_R16G16B16A16_FLOAT = 10,
	DXGI_R32G32_FLOAT = 16,
	DXGI_R10G10B10A2_UNORM = 24,
	DXGI_R8G8B8A8_UNORM = 28,
	DXGI_R8G8B8A8_UNORM_SRGB = 29,
	DXGI_R16G16_FLOAT = 34,
	DXGI_R32_FLOAT = 41,
	DXGI_R8G8_UNORM = 49,
	DXGI_R16_FLOAT = 54,
	DXGI_R8_UNORM = 61,
	DXGI_A8_UNORM = 65,
	DXGI_R9G9B9E5 = 67,
	DXGI_BC1_UNORM = 71,
	DXGI_BC1_UNORM_SRGB = 72,
	DXGI_BC2_UNORM = 74,
	DXGI_BC2_UNORM_SRGB = 75,
	DXGI_BC3_UNORM = 77,
	DXGI_BC3_UNORM_SRGB = 78,
	DXGI_BC4_UNORM = 80,
	DXGI_BC5_UNORM = 83,
	DXGI_B5G6R5_UNORM = 85,
	DXGI_B5G5R5A1_UNORM = 86,
	DXGI_B8G8R8A8_UNORM = 87,
	DXGI_BC6H_UF16 = 95,
	DXGI_BC6H_SF16 = 96,
	DXGI_BC7_UNORM = 98,
	DXGI_BC7_UNORM_SRGB = 99,
	DXGI_B4G4R4A4_UNORM = 115
};

// The legacy bitmasked format names here represent the actual data layout in the files,
// while their official names are flipped (e.g. RGBA8 layout is officially called ABGR8).
enum DDSFormat {
	DDS_DXT1,
	DDS_DXT3,
	DDS_DXT5,
	DDS_ATI1,
	DDS_ATI2,
	DDS_BC6U,
	DDS_BC6S,
	DDS_BC7,
	DDS_R16F,
	DDS_RG16F,
	DDS_RGBA16F,
	DDS_R32F,
	DDS_RG32F,
	DDS_RGB32F,
	DDS_RGBA32F,
	DDS_RGB9E5,
	DDS_RGB8,
	DDS_RGBA8,
	DDS_BGR8,
	DDS_BGRA8,
	DDS_BGR5A1,
	DDS_BGR565,
	DDS_B2GR3,
	DDS_B2GR3A8,
	DDS_BGR10A2,
	DDS_RGB10A2,
	DDS_BGRA4,
	DDS_LUMINANCE,
	DDS_LUMINANCE_ALPHA,
	DDS_LUMINANCE_ALPHA_4,
	DDS_MAX
};

enum DDSType {
	DDST_2D = 1,
	DDST_CUBEMAP,
	DDST_3D,

	DDST_TYPE_MASK = 0x7F,
	DDST_ARRAY = 0x80,
};

struct DDSFormatInfo {
	const char *name = nullptr;
	bool compressed = false;
	uint32_t divisor = 0;
	uint32_t block_size = 0;
	Image::Format format = Image::Format::FORMAT_BPTC_RGBA;
};

static const DDSFormatInfo dds_format_info[DDS_MAX] = {
	{ "DXT1/BC1", true, 4, 8, Image::FORMAT_DXT1 },
	{ "DXT2/DXT3/BC2", true, 4, 16, Image::FORMAT_DXT3 },
	{ "DXT4/DXT5/BC3", true, 4, 16, Image::FORMAT_DXT5 },
	{ "ATI1/BC4", true, 4, 8, Image::FORMAT_RGTC_R },
	{ "ATI2/A2XY/BC5", true, 4, 16, Image::FORMAT_RGTC_RG },
	{ "BC6UF", true, 4, 16, Image::FORMAT_BPTC_RGBFU },
	{ "BC6SF", true, 4, 16, Image::FORMAT_BPTC_RGBF },
	{ "BC7", true, 4, 16, Image::FORMAT_BPTC_RGBA },
	{ "R16F", false, 1, 2, Image::FORMAT_RH },
	{ "RG16F", false, 1, 4, Image::FORMAT_RGH },
	{ "RGBA16F", false, 1, 8, Image::FORMAT_RGBAH },
	{ "R32F", false, 1, 4, Image::FORMAT_RF },
	{ "RG32F", false, 1, 8, Image::FORMAT_RGF },
	{ "RGB32F", false, 1, 12, Image::FORMAT_RGBF },
	{ "RGBA32F", false, 1, 16, Image::FORMAT_RGBAF },
	{ "RGB9E5", false, 1, 4, Image::FORMAT_RGBE9995 },
	{ "RGB8", false, 1, 3, Image::FORMAT_RGB8 },
	{ "RGBA8", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "BGR8", false, 1, 3, Image::FORMAT_RGB8 },
	{ "BGRA8", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "BGR5A1", false, 1, 2, Image::FORMAT_RGBA8 },
	{ "BGR565", false, 1, 2, Image::FORMAT_RGB8 },
	{ "B2GR3", false, 1, 1, Image::FORMAT_RGB8 },
	{ "B2GR3A8", false, 1, 2, Image::FORMAT_RGBA8 },
	{ "BGR10A2", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "RGB10A2", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "BGRA4", false, 1, 2, Image::FORMAT_RGBA8 },
	{ "GRAYSCALE", false, 1, 1, Image::FORMAT_L8 },
	{ "GRAYSCALE_ALPHA", false, 1, 2, Image::FORMAT_LA8 },
	{ "GRAYSCALE_ALPHA_4", false, 1, 1, Image::FORMAT_LA8 }
};

Error save_dds(const String &p_path, const Ref<Image> &p_img) {
	Vector<uint8_t> buffer = save_dds_buffer(p_img);

	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
	if (file.is_null()) {
		return ERR_CANT_CREATE;
	}

	file->store_buffer(buffer.ptr(), buffer.size());

	return OK;
}

DDSFormat _image_format_to_dds_format(Image::Format p_image_format) {
	switch (p_image_format) {
		case Image::FORMAT_RGBAF: {
			return DDS_RGBA32F;
		}
		case Image::FORMAT_RGBF: {
			return DDS_RGB32F;
		}
		case Image::FORMAT_RGBAH: {
			return DDS_RGBA16F;
		}
		case Image::FORMAT_RGF: {
			return DDS_RG32F;
		}
		case Image::FORMAT_RGBA8: {
			return DDS_RGBA8;
		}
		case Image::FORMAT_RGH: {
			return DDS_RG16F;
		}
		case Image::FORMAT_RF: {
			return DDS_R32F;
		}
		case Image::FORMAT_L8: {
			return DDS_LUMINANCE;
		}
		case Image::FORMAT_RH: {
			return DDS_R16F;
		}
		case Image::FORMAT_LA8: {
			return DDS_LUMINANCE_ALPHA;
		}
		case Image::FORMAT_RGBE9995: {
			return DDS_RGB9E5;
		}
		case Image::FORMAT_DXT1: {
			return DDS_DXT1;
		}
		case Image::FORMAT_DXT3: {
			return DDS_DXT3;
		}
		case Image::FORMAT_DXT5: {
			return DDS_DXT5;
		}
		case Image::FORMAT_RGTC_R: {
			return DDS_ATI1;
		}
		case Image::FORMAT_RGTC_RG: {
			return DDS_ATI2;
		}
		case Image::FORMAT_RGB8: {
			return DDS_RGB8;
		}
		case Image::FORMAT_BPTC_RGBFU: {
			return DDS_BC6U;
		}
		case Image::FORMAT_BPTC_RGBF: {
			return DDS_BC6S;
		}
		case Image::FORMAT_BPTC_RGBA: {
			return DDS_BC7;
		}
		default: {
			return DDS_MAX;
		}
	}
}

#define PF_FOURCC(s) ((uint32_t)(((s)[3] << 24U) | ((s)[2] << 16U) | ((s)[1] << 8U) | ((s)[0])))

enum DDSFourCC {
	DDFCC_DXT1 = PF_FOURCC("DXT1"),
	DDFCC_DXT2 = PF_FOURCC("DXT2"),
	DDFCC_DXT3 = PF_FOURCC("DXT3"),
	DDFCC_DXT4 = PF_FOURCC("DXT4"),
	DDFCC_DXT5 = PF_FOURCC("DXT5"),
	DDFCC_ATI1 = PF_FOURCC("ATI1"),
	DDFCC_BC4U = PF_FOURCC("BC4U"),
	DDFCC_ATI2 = PF_FOURCC("ATI2"),
	DDFCC_BC5U = PF_FOURCC("BC5U"),
	DDFCC_A2XY = PF_FOURCC("A2XY"),
	DDFCC_DX10 = PF_FOURCC("DX10"),
	DDFCC_R16F = 111,
	DDFCC_RG16F = 112,
	DDFCC_RGBA16F = 113,
	DDFCC_R32F = 114,
	DDFCC_RG32F = 115,
	DDFCC_RGBA32F = 116,
	DDFCC_R9G9B9E5 = 117,
	DDFCC_R8G8B8A8_UNORM = 118,
	DDFCC_B8G8R8A8_UNORM = 119,
	DDFCC_B5G5R5A1_UNORM = 120,
	DDFCC_B5G6R5_UNORM = 121,
	DDFCC_B4G4R4A4_UNORM = 122,
	DDFCC_A8_UNORM = 123,
	DDFCC_R8G8_UNORM = 124
};

uint32_t _image_format_to_fourcc_format(Image::Format p_format) {
	switch (p_format) {
		case Image::FORMAT_DXT1:
			return DDFCC_DXT1;
		case Image::FORMAT_DXT3:
			return DDFCC_DXT3;
		case Image::FORMAT_DXT5:
			return DDFCC_DXT5;
		case Image::FORMAT_RGTC_R:
			return DDFCC_ATI1;
		case Image::FORMAT_RGTC_RG:
			return DDFCC_ATI2;
		case Image::FORMAT_RGBAF:
			return DDFCC_RGBA32F;
		case Image::FORMAT_RGBF:
			return DDFCC_R32F;
		case Image::FORMAT_RGF:
			return DDFCC_RG32F;
		case Image::FORMAT_RF:
			return DDFCC_R32F;
		case Image::FORMAT_RGBAH:
			return DDFCC_RGBA16F;
		case Image::FORMAT_RGBH:
			return DDFCC_R16F;
		case Image::FORMAT_RGH:
			return DDFCC_RG16F;
		case Image::FORMAT_RH:
			return DDFCC_R16F;
		case Image::FORMAT_RGBE9995:
			return DDFCC_R9G9B9E5;
		case Image::FORMAT_RGBA8:
			return DDFCC_R8G8B8A8_UNORM;
		case Image::FORMAT_RGB8:
			return DDFCC_R8G8B8A8_UNORM;
		case Image::FORMAT_RG8:
			return DDFCC_R8G8_UNORM;
		case Image::FORMAT_R8:
			return DDFCC_A8_UNORM;
		case Image::FORMAT_L8:
			return DDFCC_A8_UNORM;
		case Image::FORMAT_LA8:
			return DDFCC_R8G8_UNORM;
		case Image::FORMAT_RGBA4444:
			return DDFCC_B4G4R4A4_UNORM;
		case Image::FORMAT_RGB565:
			return DDFCC_B5G6R5_UNORM;
		default:
			return 0;
	}
}

const int DDSD_CAPS = 0x1;
const int DDSD_HEIGHT = 0x2;
const int DDSD_WIDTH = 0x4;
const int DDSD_PIXELFORMAT = 0x1000;

Vector<uint8_t> save_dds_buffer(const Ref<Image> &p_img) {
	Ref<StreamPeerBuffer> stream_buffer;
	stream_buffer.instantiate();

	Ref<Image> image = p_img;

	stream_buffer->put_32(DDS_MAGIC);

	uint32_t header_size = 124;
	stream_buffer->put_32(header_size);

	uint32_t flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_PITCH | DDSD_LINEARSIZE;

	if (image->has_mipmaps()) {
		flags |= DDSD_MIPMAPCOUNT;
	}

	stream_buffer->put_32(flags);

	uint32_t height = image->get_height();
	stream_buffer->put_32(height);

	uint32_t width = image->get_width();
	stream_buffer->put_32(width);

	DDSFormat dds_format = _image_format_to_dds_format(image->get_format());
	const DDSFormatInfo &info = dds_format_info[dds_format];

	uint32_t depth = 1; // Default depth for 2D textures

	uint32_t pitch;
	if (info.compressed) {
		pitch = ((MAX(info.divisor, width) + info.divisor - 1) / info.divisor) * ((MAX(info.divisor, height) + info.divisor - 1) / info.divisor) * info.block_size;
	} else {
		pitch = width * info.block_size;
	}

	stream_buffer->put_32(pitch);

	stream_buffer->put_32(depth);

	uint32_t mipmaps = image->has_mipmaps() ? (image->get_mipmap_count() + 1) : 1;
	stream_buffer->put_32(mipmaps);

	uint32_t reserved = 0;
	for (int i = 0; i < 11; i++) {
		stream_buffer->put_32(reserved);
	}

	uint32_t pf_size = 32;
	stream_buffer->put_32(pf_size);

	uint32_t pf_flags = info.compressed ? DDPF_FOURCC : DDPF_RGB;
	stream_buffer->put_32(pf_flags);

	uint32_t fourcc = _image_format_to_fourcc_format(image->get_format());
	stream_buffer->put_32(fourcc);

	uint32_t rgb_bit_count = 32;
	stream_buffer->put_32(rgb_bit_count);

	uint32_t red_mask = 0x00FF0000;
	stream_buffer->put_32(red_mask);

	uint32_t green_mask = 0x0000FF00;
	stream_buffer->put_32(green_mask);

	uint32_t blue_mask = 0x000000FF;
	stream_buffer->put_32(blue_mask);

	uint32_t alpha_mask = 0xFF000000;
	stream_buffer->put_32(alpha_mask);

	uint32_t caps1 = DDSD_LINEARSIZE;
	stream_buffer->put_32(caps1);

	uint32_t caps2 = 0;
	stream_buffer->put_32(caps2);

	uint32_t caps3 = 0;
	stream_buffer->put_32(caps3);

	uint32_t caps4 = 0;
	stream_buffer->put_32(caps4);

	uint32_t reserved2 = 0;
	stream_buffer->put_32(reserved2);

	for (uint32_t i = 0; i < mipmaps; i++) {
		Ref<Image> mip_image = image->get_image_from_mipmap(i);
		Vector<uint8_t> data = mip_image->get_data();
		uint32_t mip_width = MAX(1u, width >> i);
		uint32_t mip_height = MAX(1u, height >> i);

		uint32_t expected_size = 0;
		if (info.compressed) {
			uint32_t blocks_x = (mip_width + info.divisor - 1) / info.divisor;
			uint32_t blocks_y = (mip_height + info.divisor - 1) / info.divisor;
			expected_size = blocks_x * blocks_y * info.block_size;
		} else {
			expected_size = mip_width * mip_height * info.block_size;
		}

		ERR_FAIL_COND_V_MSG(data.size() != expected_size, Vector<uint8_t>(),
				"Image data size mismatch for mipmap level " + itos(i) +
						". Expected size: " + itos(expected_size) + ", actual size: " + itos(data.size()) + ".");

		stream_buffer->put_data(data.ptr(), data.size());
	}

	return stream_buffer->get_data_array();
}
