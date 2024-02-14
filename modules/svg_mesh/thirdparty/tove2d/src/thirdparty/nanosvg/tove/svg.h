/*
 * TÖVE - Animated vector graphics for LÖVE.
 * https://github.com/poke1024/Tove
 *
 * Portions taken from NanoSVG
 * Copyright (c) 2019, Bernhard Liebl
 *
 * Distributed under the MIT license. See LICENSE file for details.
 *
 * All rights reserved.
 */

struct NSVGrasterizer;
struct NSVGimage;
struct NSVGcachedPaint;
struct NSVGpaint;
struct NSVGshape;
struct NSVGparser;

typedef uint8_t TOVEclipPathIndex;

struct TOVEclip {
	TOVEclipPathIndex* index;	// Array of clip path indices (of related NSVGimage).
	TOVEclipPathIndex count;	// Number of clip paths in this set.
};

struct TOVEclipPath {
	char id[64];				// Unique id of this clip path (from SVG).
	TOVEclipPathIndex index;	// Unique internal index of this clip path.
	NSVGshape* shapes;			// Linked list of shapes in this clip path.
	struct TOVEclipPath* next;	// Pointer to next clip path or NULL.
};

#define TOVE_MAX_CLIP_PATHS 255 // also note TOVEclipPathIndex

struct TOVEgradientStop {
    unsigned int color;
    float offset;
};

TOVEclipPath* tove__createClipPath(const char* name, int index);
TOVEclipPath* tove__findClipPath(NSVGparser* p, const char* name);
void tove__deleteClipPaths(TOVEclipPath* path);
