/**************************************************************************/
/*  godot_main_cnode.cpp                                                  */
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
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "godot_cnode.h"

#include <stdio.h>
#include <string.h>

// Forward declarations - these are defined in godot_cnode.cpp
extern "C" {
    extern godot_instance_t instances[MAX_INSTANCES];
    extern int listen_fd;
    
    int init_cnode(char *nodename, char *cookie);
    void main_loop(void);
}

/*
 * Main entry point for Godot CNode
 */
int main(int argc, char **argv) {
    char *nodename = const_cast<char*>("godot@127.0.0.1");
    char *cookie = const_cast<char*>("godotcookie");
    int i;
    
    /* Initialize instances array */
    memset(instances, 0, sizeof(instances));
    
    /* Parse command line arguments */
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-name") == 0 && i + 1 < argc) {
            nodename = argv[++i];
        } else if (strcmp(argv[i], "-setcookie") == 0 && i + 1 < argc) {
            cookie = argv[++i];
        }
    }
    
    printf("Godot CNode starting...\n");
    printf("  Node: %s\n", nodename);
    printf("  Cookie: %s\n", cookie);
    
    /* Initialize CNode */
    if (init_cnode(nodename, cookie) < 0) {
        fprintf(stderr, "Failed to initialize CNode\n");
        return 1;
    }
    
    printf("Godot CNode published and ready (listen_fd: %d)\n", listen_fd);
    
    /* Enter main loop */
    main_loop();
    
    /* Cleanup: destroy all instances */
    for (i = 0; i < MAX_INSTANCES; i++) {
        if (instances[i].instance != NULL) {
            libgodot_destroy_godot_instance(instances[i].instance);
        }
    }
    
    return 0;
}

