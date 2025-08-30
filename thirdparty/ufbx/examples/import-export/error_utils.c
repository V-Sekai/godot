#include "scene_utils.h"
#include <stdio.h>

void print_error(const ufbx_error *error, const char *description)
{
    char buffer[1024];
    ufbx_format_error(buffer, sizeof(buffer), error);
    fprintf(stderr, "Error: %s\n%s\n", description, buffer);
}
