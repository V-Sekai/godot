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

TOVEclipPath* tove__createClipPath(const char* name, int index)
{
	TOVEclipPath* clipPath = (TOVEclipPath*)malloc(sizeof(TOVEclipPath));
	if (clipPath == NULL) return NULL;
	memset(clipPath, 0, sizeof(TOVEclipPath));
	strncpy(clipPath->id, name, 63);
	clipPath->id[63] = '\0';
	clipPath->index = index;
	return clipPath;
}

TOVEclipPath* tove__findClipPath(NSVGparser* p, const char* name)
{
	int i = 0;
	TOVEclipPath** link;

	link = &p->image->clipPaths;
	while (*link != NULL) {
		if (strcmp((*link)->id, name) == 0) {
			break;
		}
		link = &(*link)->next;
		i++;
	}
	if (*link == NULL) {
		*link = tove__createClipPath(name, i);
	}
	return *link;
}

void tove__deleteClipPaths(TOVEclipPath* path)
{
	TOVEclipPath *pnext;
	while (path != NULL) {
		pnext = path->next;
		nsvg__deleteShapes(path->shapes);
		free(path);
		path = pnext;
	}
}
