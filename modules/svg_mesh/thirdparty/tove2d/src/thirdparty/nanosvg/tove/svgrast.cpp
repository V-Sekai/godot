/*
 * TÖVE - Animated vector graphics for LÖVE.
 * https://github.com/poke1024/Tove
 *
 * Portions of this code are taken from NanoSVG.
 * Portions of this code Copyright (c) 2019, Bernhard Liebl.
 *
 * All rights reserved.
 */

void tove_deleteRasterizer(NSVGrasterizer* r) {

	if (r->stencil.data) free(r->stencil.data);
	if (r->dither.data) free(r->dither.data);
}

void tove__scanlineBit(
	NSVGrasterizer* r,
	int x,
	int y,
	int count,
	float tx,
	float ty,
	float scale,
	NSVGcachedPaint* cache,
	TOVEclip *clip) {

	unsigned char* const row = &r->bitmap[y * r->stride];
	unsigned char* const cover = r->scanline;

	const int x1 = x + count;
	for (; x < x1; x++) {
		row[x / 8] |= (cover[x] > 0 ? 1 : 0) << (x % 8);
	}
}

inline void maskClip(
	NSVGrasterizer* r,
	TOVEclip* clip,
	int xmin,
	int y,
	int count) {

	unsigned char* const cover = r->scanline;
	const int xmax = xmin + count - 1;

	for (int i = 0; i < clip->count; i++) {
		unsigned char* stencil = &r->stencil.data[
			r->stencil.size * clip->index[i] + y * r->stencil.stride];

		for (int j = xmin; j <= xmax; j++) {
			if (((stencil[j / 8] >> (j % 8)) & 1) == 0) {
				cover[j] = 0;
			}
		}
	}
}

void tove__drawColorScanline(
	NSVGrasterizer* r,
	int xmin,
	int y,
	int count,
	float tx,
	float ty,
	float scale,
	NSVGcachedPaint* cache,
	TOVEclip* clip) {

	unsigned char* dst = &r->bitmap[y * r->stride] + xmin*4;
	unsigned char* cover = &r->scanline[xmin];
	maskClip(r, clip, xmin, y, count);

	int i, cr, cg, cb, ca;
	cr = cache->colors[0] & 0xff;
	cg = (cache->colors[0] >> 8) & 0xff;
	cb = (cache->colors[0] >> 16) & 0xff;
	ca = (cache->colors[0] >> 24) & 0xff;

	for (i = 0; i < count; i++) {
		int r,g,b;
		int a = nsvg__div255((int)cover[0] * ca);
		int ia = 255 - a;
		// Premultiply
		r = nsvg__div255(cr * a);
		g = nsvg__div255(cg * a);
		b = nsvg__div255(cb * a);

		// Blend over
		r += nsvg__div255(ia * (int)dst[0]);
		g += nsvg__div255(ia * (int)dst[1]);
		b += nsvg__div255(ia * (int)dst[2]);
		a += nsvg__div255(ia * (int)dst[3]);

		dst[0] = (unsigned char)r;
		dst[1] = (unsigned char)g;
		dst[2] = (unsigned char)b;
		dst[3] = (unsigned char)a;

		cover++;
		dst += 4;
	}
}

class LinearGradient {
	const float* const t;

public:
	inline LinearGradient(const NSVGcachedPaint* cache) : t(cache->xform) {
	}

	inline float operator()(float fx, float fy) const {
		return fx*t[1] + fy*t[3] + t[5];
	}
};

class RadialGradient {
	const float* const t;

public:
	inline RadialGradient(const NSVGcachedPaint* cache) : t(cache->xform) {
	}

	inline float operator()(float fx, float fy) const {
		const float gx = fx*t[0] + fy*t[2] + t[4];
		const float gy = fx*t[1] + fy*t[3] + t[5];
		return sqrtf(gx*gx + gy*gy);
	}
};

class FastGradientColors {
	const NSVGcachedPaint * const cache;

public:
	inline FastGradientColors(
		NSVGrasterizer* r,
		NSVGcachedPaint *cache,
		int x,
		int y,
		int count) : cache(cache) {
	}

	inline bool good() {
		return true;
	}

	inline uint32_t operator()(int x, float gy) const {
		return cache->colors[(int)nsvg__clampf(gy*255.0f, 0, 255.0f)];
	}
};

class BestGradientColors {
	typedef float dither_error_t;
	
	static constexpr int diffusion_matrix_width = 5;
	static constexpr int diffusion_matrix_height = 3;
	static constexpr int dither_components = 4;

	dither_error_t *diffusion[diffusion_matrix_height];

	const NSVGgradient * const gradient;
	const NSVGgradientStop *stop;
	const NSVGgradientStop *const stopN;

	inline void rotate(
		const NSVGrasterizer* r,
		NSVGcachedPaint* cache,
		const int x,
		const int y,
		const int count) {

		dither_error_t **rows = diffusion;
		dither_error_t * const data = static_cast<dither_error_t*>(r->dither.data);

		const unsigned int stride = r->dither.stride;

		int roll = y - cache->tove.ditherY;
		if (roll < 0) { // should never happen.
			roll = diffusion_matrix_height;
		} else if (roll > diffusion_matrix_height) {
			roll = diffusion_matrix_height;
		}

		cache->tove.ditherY = y;

		// rotate rows.
		for (int i = 0; i < diffusion_matrix_height; i++) {
			rows[i] = &data[((y + i) % diffusion_matrix_height) * stride];
		}

		const int span0 = x * dither_components;
		const int span1 = (x + count + diffusion_matrix_width - 1) * dither_components;

		// new rows.
		for (int i = 0; i < roll; i++) {
			dither_error_t * const row = rows[diffusion_matrix_height - 1 - i];
			for (int j = span0; j < span1; j++) {
				row[j] = 0.0f;
			}
		}

		// old rows that already have content.
		for (int i = 0; i < diffusion_matrix_height - roll; i++) {
			dither_error_t * const row = rows[i];

			const int previousLeftSpan = cache->tove.ditherSpan[0];
			for (int j = span0; j < previousLeftSpan; j++) {
				row[j] = 0.0f;
			}
			
			for (int j = cache->tove.ditherSpan[1]; j < span1; j++) {
				row[j] = 0.0f;
			}
		}

		cache->tove.ditherSpan[0] = span0;
		cache->tove.ditherSpan[1] = span1;

	}

public:
	inline BestGradientColors(
		NSVGrasterizer* r,
		NSVGcachedPaint *cache,
		int x,
		int y,
		int count) :

		gradient(cache->tove.paint->gradient),
		stop(gradient->stops),
		stopN(gradient->stops + gradient->nstops) {

		rotate(r, cache, x, y, count);
	}

	static inline bool enabled(const NSVGrasterizer* r) {
		return r->quality > 0;
	}

	static inline bool allocate(NSVGrasterizer* r, int w) {
		if (enabled(r)) {
			TOVEdither &dither = r->dither;
			dither.stride = (w + diffusion_matrix_width - 1) * dither_components;
			dither.data = (dither_error_t*)realloc(dither.data,
				dither.stride * diffusion_matrix_height * sizeof(dither_error_t));
			return dither.data != nullptr;
		} else {
			return true;
		}
	}

	static void init(
		NSVGcachedPaint* cache,
		NSVGpaint* paint,
		float opacity) {

		cache->tove.paint = paint;
		cache->tove.ditherY = -diffusion_matrix_height;

		NSVGgradientStop *stop = paint->gradient->stops;
		const int n = paint->gradient->nstops;

		for (int i = 0; i < n; i++) {
			stop->tove.color = nsvg__applyOpacity(stop->color, opacity);
			stop->tove.offset = nsvg__clampf(stop->offset, 0.0f, 1.0f);
			stop++;
		}
	}

	inline bool good() const {
		return stop + 1 < stopN;
	}

	inline uint32_t operator()(const int x, const float gy) {
		while (gy >= (stop + 1)->tove.offset && (stop + 2) < stopN) {
			stop++;
		}
		while (gy < stop->tove.offset && stop > gradient->stops) {
			stop--;
		}

		dither_error_t * const r0 = diffusion[0];
		dither_error_t * const r1 = diffusion[1];
		dither_error_t * const r2 = diffusion[2];

		const unsigned int c0 = stop->tove.color;
		const int cr0 = (c0) & 0xff;
		const int cg0 = (c0 >> 8) & 0xff;
		const int cb0 = (c0 >> 16) & 0xff;
		const int ca0 = (c0 >> 24) & 0xff;

		const unsigned int c1 = (stop + 1)->tove.color;
		const int cr1 = (c1) & 0xff;
		const int cg1 = (c1 >> 8) & 0xff;
		const int cb1 = (c1 >> 16) & 0xff;
		const int ca1 = (c1 >> 24) & 0xff;

		const float offset0 = stop->tove.offset;
		const float range = (stop + 1)->tove.offset - offset0;
		const float t = nsvg__clampf((gy - offset0) / range, 0.0f, 1.0f);
		const float s = 1.0f - t;

		const float f_cr = (cr0 * s + cr1 * t) + r0[x * dither_components + 0];
		const float f_cg = (cg0 * s + cg1 * t) + r0[x * dither_components + 1];
		const float f_cb = (cb0 * s + cb1 * t) + r0[x * dither_components + 2];
		const float f_ca = (ca0 * s + ca1 * t) + r0[x * dither_components + 3];

		const int cr = nsvg__clampf(f_cr + 0.5, 0.0f, 255.0f);
		const int cg = nsvg__clampf(f_cg + 0.5, 0.0f, 255.0f);
		const int cb = nsvg__clampf(f_cb + 0.5, 0.0f, 255.0f);
		const int ca = nsvg__clampf(f_ca + 0.5, 0.0f, 255.0f);

		// future feature:
		// apply restricted paletted on (cr, cg, cb) here.

		const float errors[] = {f_cr - cr, f_cg - cg, f_cb - cb, f_ca - ca};

		// Distribute errors using sierra dithering.
		// #pragma clang loop vectorize(enable)
		for (int j = 0; j < 4; j++) {
			const int offset = x * dither_components + j;
			const float error = errors[j];

			r0[offset + 1 * dither_components] += error * (5.0f / 32.0f);
			r0[offset + 2 * dither_components] += error * (3.0f / 32.0f);

			r1[offset + 0 * dither_components] += error * (2.0f / 32.0f);
			r1[offset + 1 * dither_components] += error * (4.0f / 32.0f);
			r1[offset + 2 * dither_components] += error * (5.0f / 32.0f);
			r1[offset + 3 * dither_components] += error * (4.0f / 32.0f);
			r1[offset + 4 * dither_components] += error * (2.0f / 32.0f);

			r2[offset + 1 * dither_components] += error * (2.0f / 32.0f);
			r2[offset + 2 * dither_components] += error * (3.0f / 32.0f);
			r2[offset + 3 * dither_components] += error * (2.0f / 32.0f);
		}

		return cr | (cg << 8) | (cb << 16) | (ca << 24);
	}
};

template<typename Gradient, typename Colors>
void drawGradientScanline(
	NSVGrasterizer* r,
	int x,
	int y,
	int count,
	float tx,
	float ty,
	float scale,
	NSVGcachedPaint* cache,
	TOVEclip* clip) {

	using tove::nsvg__div255;

	unsigned char* dst = &r->bitmap[y * r->stride] + x*4;
	unsigned char* cover = &r->scanline[x];
	maskClip(r, clip, x, y, count);

	// TODO: spread modes.
	// TODO: plenty of opportunities to optimize.
	float fx, fy, dx, gy;
	int i, cr, cg, cb, ca;
	unsigned int c;
	Gradient gradient(cache);
	Colors colors(r, cache, x, y, count);

	fx = ((float)x - tx) / scale;
	fy = ((float)y - ty) / scale;
	dx = 1.0f / scale;

	for (i = 0; i < count; i++) {
		int r,g,b,a,ia;
		c = colors(x, gradient(fx, fy));
		cr = (c) & 0xff;
		cg = (c >> 8) & 0xff;
		cb = (c >> 16) & 0xff;
		ca = (c >> 24) & 0xff;

		a = nsvg__div255((int)cover[0] * ca);
		ia = 255 - a;

		// Premultiply
		r = nsvg__div255(cr * a);
		g = nsvg__div255(cg * a);
		b = nsvg__div255(cb * a);

		// Blend over
		r += nsvg__div255(ia * (int)dst[0]);
		g += nsvg__div255(ia * (int)dst[1]);
		b += nsvg__div255(ia * (int)dst[2]);
		a += nsvg__div255(ia * (int)dst[3]);

		dst[0] = (unsigned char)r;
		dst[1] = (unsigned char)g;
		dst[2] = (unsigned char)b;
		dst[3] = (unsigned char)a;

		cover++;
		dst += 4;
		fx += dx;
		x += 1;
	}
}

TOVEscanlineFunction tove__initPaint(
	NSVGcachedPaint* cache,
	const NSVGrasterizer* r,
	NSVGpaint* paint,
	float opacity,
	bool &initCacheColors) {

	if (r && BestGradientColors::enabled(r)) {
		switch (cache->type) {
			case NSVG_PAINT_LINEAR_GRADIENT:
				BestGradientColors::init(cache, paint, opacity);
				initCacheColors = false;
				return drawGradientScanline<LinearGradient, BestGradientColors>;
			case NSVG_PAINT_RADIAL_GRADIENT:
				BestGradientColors::init(cache, paint, opacity);
				initCacheColors = false;
				return drawGradientScanline<RadialGradient, BestGradientColors>;
		}
	}

	initCacheColors = true;
	switch (cache->type) {
		case NSVG_PAINT_LINEAR_GRADIENT:
			return drawGradientScanline<LinearGradient, FastGradientColors>;
		case NSVG_PAINT_RADIAL_GRADIENT:
			return drawGradientScanline<RadialGradient, FastGradientColors>;
	}

	return tove__drawColorScanline;
}

bool tove__rasterize(
	NSVGrasterizer* r,
    NSVGimage* image,
    int w,
    int h,
	float tx,
    float ty,
    float scale)
{
	if (!BestGradientColors::allocate(r, w)) {
		return false;
	}

	TOVEclipPath* clipPath;
	int clipPathCount = 0;

	clipPath = image->clipPaths;
	if (clipPath == NULL) {
		return true;
	}

	while (clipPath != NULL) {
		clipPathCount++;
		clipPath = clipPath->next;
	}

	r->stencil.stride = w / 8 + (w % 8 != 0 ? 1 : 0);
	r->stencil.size = h * r->stencil.stride;
	r->stencil.data = (unsigned char*)realloc(
		r->stencil.data, r->stencil.size * clipPathCount);
	if (r->stencil.data == NULL) {
		return false;
	}
	memset(r->stencil.data, 0, r->stencil.size * clipPathCount);

	clipPath = image->clipPaths;
	while (clipPath != NULL) {
		nsvg__rasterizeShapes(r, clipPath->shapes, tx, ty, scale,
			&r->stencil.data[r->stencil.size * clipPath->index],
			w, h, r->stencil.stride, tove__scanlineBit);
		clipPath = clipPath->next;
	}

	return true;
}
