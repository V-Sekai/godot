/*
 * Godot CNode - Erlang/Elixir CNode interface for Godot Engine
 * 
 * This CNode allows Elixir/Erlang nodes to communicate with Godot
 * using the Erlang distribution protocol.
 */

extern "C" {
#include "ei.h"
#include "ei_connect.h"
}

#include "core/string/node_path.h"
#include "core/object/object.h"
#include "core/variant/variant.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "core/object/class_db.h"
#include "core/extension/libgodot.h"
#include "core/extension/godot_instance.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/main/node.h"

// Direct calls to GodotInstance methods (see core/extension/godot_instance.cpp)
// start() calls Main::setup2() and Main::start(), then initializes main loop
// iteration() calls DisplayServer::process_events() and Main::iteration()
// stop() finalizes main loop and sets started = false

/* Get SceneTree singleton */
static SceneTree *get_scene_tree() {
    return SceneTree::get_singleton();
}

/* Get root node from scene tree */
static Node *get_scene_tree_root(SceneTree *tree) {
    if (tree == NULL) return NULL;
    return tree->get_root();
}

/* Find node by path */
static Node *find_node_by_path(SceneTree *tree, const char *path_str) {
    if (tree == NULL || path_str == NULL) return NULL;
    Node *root = tree->get_root();
    if (root == NULL) return NULL;
    NodePath path(path_str);
    return root->get_node_or_null(path);
}

/* Get node name as string */
static const char *get_node_name(Node *node) {
    if (node == NULL) return NULL;
    StringName name = node->get_name();
    return name.operator String().utf8().get_data();
}

/* Get node by pointer ID (instance ID) */
static Node *get_node_by_id(int64_t node_id) {
    if (node_id == 0) return NULL;
    ObjectID obj_id = ObjectID((uint64_t)node_id);
    Object *obj = ObjectDB::get_instance(obj_id);
    if (obj == NULL) return NULL;
    return Object::cast_to<Node>(obj);
}

/* Convert BERT to Variant (decode from ei buffer) */
static Variant bert_to_variant(char *buf, int *index) {
    int type, arity;
    char atom[MAXATOMLEN];
    long long_val;
    double double_val;
    char string_buf[256];
    
    if (ei_decode_version(buf, index, NULL) < 0) {
        return Variant(); // Error
    }
    
    if (ei_get_type(buf, index, &type, &arity) < 0) {
        return Variant(); // Error
    }
    
    switch (type) {
        case ERL_ATOM_EXT:
            if (ei_decode_atom(buf, index, atom) == 0) {
                if (strcmp(atom, "nil") == 0) {
                    return Variant();
                } else if (strcmp(atom, "true") == 0) {
                    return Variant(true);
                } else if (strcmp(atom, "false") == 0) {
                    return Variant(false);
                }
            }
            return Variant(String(atom));
            
        case ERL_INTEGER_EXT:
            if (ei_decode_long(buf, index, &long_val) == 0) {
                return Variant((int64_t)long_val);
            }
            break;
            
        case ERL_FLOAT_EXT:
        case NEW_FLOAT_EXT:
            if (ei_decode_double(buf, index, &double_val) == 0) {
                return Variant(double_val);
            }
            break;
            
        case ERL_STRING_EXT:
            if (ei_decode_string(buf, index, string_buf) == 0) {
                return Variant(String::utf8(string_buf));
            }
            break;
            
        case ERL_LIST_EXT:
            if (arity == 0) {
                ei_decode_list_header(buf, index, &arity);
                return Variant(Array());
            } else {
                Array arr;
                ei_decode_list_header(buf, index, &arity);
                for (int i = 0; i < arity; i++) {
                    Variant elem = bert_to_variant(buf, index);
                    arr.push_back(elem);
                }
                // Check and skip the list tail (should be nil/empty list)
                int type, size;
                if (ei_get_type(buf, index, &type, &size) == 0 && type == ERL_NIL_EXT) {
                    ei_skip_term(buf, index);
                }
                return Variant(arr);
            }
            
        case ERL_SMALL_TUPLE_EXT:
        case ERL_LARGE_TUPLE_EXT:
            if (ei_decode_tuple_header(buf, index, &arity) == 0 && arity > 0) {
                if (ei_decode_atom(buf, index, atom) == 0) {
                    if (strcmp(atom, "vector2") == 0 && arity == 3) {
                        double x, y;
                        ei_decode_double(buf, index, &x);
                        ei_decode_double(buf, index, &y);
                        return Variant(Vector2(x, y));
                    } else if (strcmp(atom, "vector3") == 0 && arity == 4) {
                        double x, y, z;
                        ei_decode_double(buf, index, &x);
                        ei_decode_double(buf, index, &y);
                        ei_decode_double(buf, index, &z);
                        return Variant(Vector3(x, y, z));
                    } else if (strcmp(atom, "color") == 0 && arity == 5) {
                        double r, g, b, a;
                        ei_decode_double(buf, index, &r);
                        ei_decode_double(buf, index, &g);
                        ei_decode_double(buf, index, &b);
                        ei_decode_double(buf, index, &a);
                        return Variant(Color(r, g, b, a));
                    }
                }
            }
            break;
    }
    
    return Variant(); // Unsupported type
}

/* Convert Godot Variant to BERT format (encode to ei_x_buff) */
static void variant_to_bert(const Variant &var, ei_x_buff *x) {
    if (x == NULL) return;
    
    switch (var.get_type()) {
        case Variant::NIL:
            ei_x_encode_atom(x, "nil");
            break;
            
        case Variant::BOOL: {
            bool val = var;
            ei_x_encode_boolean(x, val);
            break;
        }
        
        case Variant::INT: {
            int64_t val = var;
            if (val >= INT32_MIN && val <= INT32_MAX) {
                ei_x_encode_long(x, (long)val);
            } else {
                // Use tuple for large integers: {bignum, sign, high, low}
                ei_x_encode_tuple_header(x, 4);
                ei_x_encode_atom(x, "bignum");
                ei_x_encode_long(x, val < 0 ? 1 : 0); // sign
                ei_x_encode_long(x, (long)(val >> 32)); // high
                ei_x_encode_long(x, (long)(val & 0xFFFFFFFF)); // low
            }
            break;
        }
        
        case Variant::FLOAT: {
            double val = var;
            ei_x_encode_double(x, val);
            break;
        }
        
        case Variant::STRING: {
            String str = var;
            CharString utf8 = str.utf8();
            ei_x_encode_string(x, utf8.get_data());
            break;
        }
        
        case Variant::VECTOR2: {
            Vector2 vec = var;
            ei_x_encode_tuple_header(x, 3);
            ei_x_encode_atom(x, "vector2");
            ei_x_encode_double(x, vec.x);
            ei_x_encode_double(x, vec.y);
            break;
        }
        
        case Variant::VECTOR2I: {
            Vector2i vec = var;
            ei_x_encode_tuple_header(x, 3);
            ei_x_encode_atom(x, "vector2i");
            ei_x_encode_long(x, vec.x);
            ei_x_encode_long(x, vec.y);
            break;
        }
        
        case Variant::VECTOR3: {
            Vector3 vec = var;
            ei_x_encode_tuple_header(x, 4);
            ei_x_encode_atom(x, "vector3");
            ei_x_encode_double(x, vec.x);
            ei_x_encode_double(x, vec.y);
            ei_x_encode_double(x, vec.z);
            break;
        }
        
        case Variant::VECTOR3I: {
            Vector3i vec = var;
            ei_x_encode_tuple_header(x, 4);
            ei_x_encode_atom(x, "vector3i");
            ei_x_encode_long(x, vec.x);
            ei_x_encode_long(x, vec.y);
            ei_x_encode_long(x, vec.z);
            break;
        }
        
        case Variant::VECTOR4: {
            Vector4 vec = var;
            ei_x_encode_tuple_header(x, 5);
            ei_x_encode_atom(x, "vector4");
            ei_x_encode_double(x, vec.x);
            ei_x_encode_double(x, vec.y);
            ei_x_encode_double(x, vec.z);
            ei_x_encode_double(x, vec.w);
            break;
        }
        
        case Variant::VECTOR4I: {
            Vector4i vec = var;
            ei_x_encode_tuple_header(x, 5);
            ei_x_encode_atom(x, "vector4i");
            ei_x_encode_long(x, vec.x);
            ei_x_encode_long(x, vec.y);
            ei_x_encode_long(x, vec.z);
            ei_x_encode_long(x, vec.w);
            break;
        }
        
        case Variant::COLOR: {
            Color col = var;
            ei_x_encode_tuple_header(x, 5);
            ei_x_encode_atom(x, "color");
            ei_x_encode_double(x, col.r);
            ei_x_encode_double(x, col.g);
            ei_x_encode_double(x, col.b);
            ei_x_encode_double(x, col.a);
            break;
        }
        
        case Variant::ARRAY: {
            Array arr = var;
            ei_x_encode_list_header(x, arr.size());
            for (int i = 0; i < arr.size(); i++) {
                variant_to_bert(arr[i], x);
            }
            ei_x_encode_empty_list(x);
            break;
        }
        
        case Variant::DICTIONARY: {
            Dictionary dict = var;
            Array keys = dict.keys();
            ei_x_encode_map_header(x, keys.size());
            for (int i = 0; i < keys.size(); i++) {
                Variant key = keys[i];
                Variant value = dict[key];
                variant_to_bert(key, x);
                variant_to_bert(value, x);
            }
            break;
        }
        
        case Variant::OBJECT: {
            Object *obj = var;
            if (obj == NULL) {
                ei_x_encode_atom(x, "nil");
            } else {
                // Encode object as tuple with type name and instance ID
                ei_x_encode_tuple_header(x, 3);
                ei_x_encode_atom(x, "object");
                String class_name = obj->get_class();
                CharString utf8 = class_name.utf8();
                ei_x_encode_string(x, utf8.get_data());
                ei_x_encode_long(x, (int64_t)obj->get_instance_id());
            }
            break;
        }
        
        default: {
            // For unsupported types, encode as tuple with type name
            ei_x_encode_tuple_header(x, 2);
            ei_x_encode_atom(x, "unsupported");
            String type_name = Variant::get_type_name(var.get_type());
            CharString utf8 = type_name.utf8();
            ei_x_encode_string(x, utf8.get_data());
            break;
        }
    }
}

/* CNode configuration */
#define MAXBUFLEN 8192
/* MAXATOMLEN is already defined in ei.h */
#define MAX_INSTANCES 16

/* Godot instance structure */
typedef struct {
    void *instance;
    int id;
    int started;
#ifdef GODOT_AVAILABLE
    SceneTree *scene_tree;
#endif
} godot_instance_t;

/* Global state */
// These are now extern for use by godot_main_cnode.cpp
ei_cnode ec;
extern "C" {
    int listen_fd = -1;
    godot_instance_t instances[MAX_INSTANCES];
}
int next_instance_id = 1;

/* Forward declarations */
static int process_message(char *buf, int *index, int fd);
static int handle_call(char *buf, int *index, int fd);
static int handle_cast(char *buf, int *index);
static void send_reply(ei_x_buff *x, int fd);

/*
 * Initialize the CNode
 */
extern "C" {
int init_cnode(char *nodename, char *cookie) {
    int res;
    int fd;
    
    /* Initialize ei library */
    res = ei_connect_init(&ec, nodename, cookie, 0);
    if (res < 0) {
        fprintf(stderr, "ei_connect_init failed: %d\n", res);
        return -1;
    }
    
    /* Publish the node and get listen file descriptor */
    fd = ei_publish(&ec, 0);
    if (fd < 0) {
        fprintf(stderr, "ei_publish failed: %d\n", fd);
        return -1;
    }
    
    listen_fd = fd;
    return 0;
}
} // extern "C"

/*
 * Process incoming message from Erlang/Elixir
 */
static int process_message(char *buf, int *index, int fd) {
    int arity;
    char atom[MAXATOMLEN];
    
    /* Decode the message */
    if (ei_decode_version(buf, index, NULL) < 0) {
        fprintf(stderr, "Error decoding version\n");
        return -1;
    }
    
    if (ei_decode_tuple_header(buf, index, &arity) < 0) {
        fprintf(stderr, "Error decoding tuple header\n");
        return -1;
    }
    
    /* Get the message type atom */
    if (ei_decode_atom(buf, index, atom) < 0) {
        fprintf(stderr, "Error decoding atom\n");
        return -1;
    }
    
    /* Handle different message types */
    if (strcmp(atom, "call") == 0) {
        return handle_call(buf, index, fd);
    } else if (strcmp(atom, "cast") == 0) {
        return handle_cast(buf, index);
    } else {
        fprintf(stderr, "Unknown message type: %s\n", atom);
        return -1;
    }
}

/*
 * Find instance by ID
 */
static godot_instance_t *find_instance(int id) {
    int i;
    for (i = 0; i < MAX_INSTANCES; i++) {
        if (instances[i].id == id && instances[i].instance != NULL) {
            return &instances[i];
        }
    }
    return NULL;
}

/*
 * Allocate a new instance slot
 */
static godot_instance_t *allocate_instance() {
    int i;
    for (i = 0; i < MAX_INSTANCES; i++) {
        if (instances[i].instance == NULL) {
            instances[i].id = next_instance_id++;
            instances[i].started = 0;
            return &instances[i];
        }
    }
    return NULL;
}

/*
 * Handle synchronous call from Erlang/Elixir
 */
static int handle_call(char *buf, int *index, int fd) {
    char atom[MAXATOMLEN];
    ei_x_buff reply;
    long instance_id;
    godot_instance_t *inst;
    
    /* Get the function name */
    if (ei_decode_atom(buf, index, atom) < 0) {
        fprintf(stderr, "Error decoding function atom\n");
        return -1;
    }
    
    /* Initialize reply buffer */
    ei_x_new(&reply);
    
    /* Handle different function calls */
    if (strcmp(atom, "ping") == 0) {
        ei_x_encode_tuple_header(&reply, 2);
        ei_x_encode_atom(&reply, "reply");
        ei_x_encode_atom(&reply, "pong");
    } else if (strcmp(atom, "godot_version") == 0) {
        ei_x_encode_tuple_header(&reply, 2);
        ei_x_encode_atom(&reply, "reply");
        ei_x_encode_string(&reply, "4.x");
    } else if (strcmp(atom, "create_instance") == 0) {
        /* Create a new Godot instance */
        inst = allocate_instance();
        if (inst == NULL) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "no_slots_available");
        } else {
            /* Create instance with minimal args */
            const char *argv[] = {"godot_cnode"};
            void *godot_inst = libgodot_create_godot_instance(1, const_cast<char**>(argv), NULL);
            if (godot_inst == NULL) {
                inst->instance = NULL;
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "creation_failed");
            } else {
                inst->instance = godot_inst;
#ifdef GODOT_AVAILABLE
                inst->scene_tree = NULL;  // Will be set when instance starts
#endif
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "reply");
                ei_x_encode_long(&reply, inst->id);
            }
        }
    } else if (strcmp(atom, "start_instance") == 0) {
        /* Start a Godot instance */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            inst = find_instance(instance_id);
            if (inst == NULL) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "instance_not_found");
            } else {
                if (inst->instance != NULL) {
                    GodotInstance *gi = (GodotInstance *)inst->instance;
                    if (gi->start()) {
                        inst->started = 1;
#ifdef GODOT_AVAILABLE
                        inst->scene_tree = get_scene_tree();
#endif
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "reply");
                        ei_x_encode_atom(&reply, "ok");
                    } else {
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "error");
                        ei_x_encode_string(&reply, "start_failed");
                    }
                } else {
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "instance_null");
                }
            }
        }
    } else if (strcmp(atom, "iteration") == 0) {
        /* Run one iteration of a Godot instance */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            inst = find_instance(instance_id);
            if (inst == NULL || !inst->started) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "instance_not_started");
            } else {
                int result = 0;
                if (inst->instance != NULL) {
                    GodotInstance *gi = (GodotInstance *)inst->instance;
                    result = gi->iteration() ? 1 : 0;
                }
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "reply");
                ei_x_encode_long(&reply, result);
            }
        }
    } else if (strcmp(atom, "stop_instance") == 0) {
        /* Stop a Godot instance */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            inst = find_instance(instance_id);
            if (inst == NULL) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "instance_not_found");
            } else {
                if (inst->instance != NULL) {
                    GodotInstance *gi = (GodotInstance *)inst->instance;
                    gi->stop();
                }
                inst->started = 0;
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "reply");
                ei_x_encode_atom(&reply, "ok");
            }
        }
    } else if (strcmp(atom, "destroy_instance") == 0) {
        /* Destroy a Godot instance */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            inst = find_instance(instance_id);
            if (inst == NULL) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "instance_not_found");
            } else {
                libgodot_destroy_godot_instance(inst->instance);
                inst->instance = NULL;
                inst->id = 0;
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "reply");
                ei_x_encode_atom(&reply, "ok");
            }
        }
    } else if (strcmp(atom, "get_scene_tree_root") == 0) {
        /* Get the root node of the scene tree */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            inst = find_instance(instance_id);
            if (inst == NULL || !inst->started) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "instance_not_started");
            } else {
                Node *root = get_scene_tree_root(inst->scene_tree);
                if (root == NULL) {
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "no_root");
                } else {
                    const char *root_name = get_node_name(root);
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "reply");
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_string(&reply, root_name ? root_name : "root");
                    ei_x_encode_long(&reply, (int64_t)root->get_instance_id());  // Return instance ID
                }
            }
        }
    } else if (strcmp(atom, "find_node") == 0) {
        /* Find a node by path */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            char path[256];
            if (ei_decode_string(buf, index, path) < 0) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "invalid_path");
            } else {
                inst = find_instance(instance_id);
                if (inst == NULL || !inst->started) {
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "instance_not_started");
                } else {
#ifdef GODOT_AVAILABLE
                    Node *node = find_node_by_path(inst->scene_tree, path);
                    if (node == NULL) {
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "error");
                        ei_x_encode_string(&reply, "node_not_found");
                    } else {
                        const char *node_name = get_node_name(node);
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "reply");
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_string(&reply, node_name ? node_name : "");
                        ei_x_encode_long(&reply, (int64_t)node->get_instance_id());  // Return instance ID
                    }
#else
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "godot_not_available");
#endif
                }
            }
        }
    } else if (strcmp(atom, "get_current_scene") == 0) {
        /* Get the current scene node */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            inst = find_instance(instance_id);
            if (inst == NULL || !inst->started) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "instance_not_started");
            } else {
#ifdef GODOT_AVAILABLE
                if (inst->scene_tree == NULL) {
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "no_scene_tree");
                } else {
                    Node *current_scene = inst->scene_tree->get_current_scene();
                    if (current_scene == NULL) {
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "reply");
                        ei_x_encode_atom(&reply, "nil");
                    } else {
                        const char *scene_name = get_node_name(current_scene);
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "reply");
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_string(&reply, scene_name ? scene_name : "");
                        ei_x_encode_long(&reply, (int64_t)current_scene->get_instance_id());  // Return instance ID
                    }
                }
#else
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "godot_not_available");
#endif
            }
        }
    } else if (strcmp(atom, "get_node_property") == 0) {
        /* Get a node property */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            long node_id;
            char prop_name[256];
            if (ei_decode_long(buf, index, &node_id) < 0 || ei_decode_string(buf, index, prop_name) < 0) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "invalid_parameters");
            } else {
                inst = find_instance(instance_id);
                if (inst == NULL || !inst->started) {
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "instance_not_started");
                } else {
#ifdef GODOT_AVAILABLE
                    Node *node = get_node_by_id(node_id);
                    if (node == NULL) {
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "error");
                        ei_x_encode_string(&reply, "node_not_found");
                    } else {
                        StringName prop(prop_name);
                        bool valid = false;
                        Variant value = node->get(prop, &valid);
                        if (!valid) {
                            ei_x_encode_tuple_header(&reply, 2);
                            ei_x_encode_atom(&reply, "error");
                            ei_x_encode_string(&reply, "property_not_found");
                        } else {
                            ei_x_encode_tuple_header(&reply, 2);
                            ei_x_encode_atom(&reply, "reply");
                            variant_to_bert(value, &reply);
                        }
                    }
#else
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "godot_not_available");
#endif
                }
            }
        }
    } else if (strcmp(atom, "set_node_property") == 0) {
        /* Set a node property */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            long node_id;
            char prop_name[256];
            if (ei_decode_long(buf, index, &node_id) < 0 || ei_decode_string(buf, index, prop_name) < 0) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "invalid_parameters");
            } else {
                inst = find_instance(instance_id);
                if (inst == NULL || !inst->started) {
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "instance_not_started");
                } else {
#ifdef GODOT_AVAILABLE
                    Node *node = get_node_by_id(node_id);
                    if (node == NULL) {
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "error");
                        ei_x_encode_string(&reply, "node_not_found");
                    } else {
                        Variant value = bert_to_variant(buf, index);
                        StringName prop(prop_name);
                        bool valid = false;
                        node->set(prop, value, &valid);
                        if (!valid) {
                            ei_x_encode_tuple_header(&reply, 2);
                            ei_x_encode_atom(&reply, "error");
                            ei_x_encode_string(&reply, "property_set_failed");
                        } else {
                            ei_x_encode_tuple_header(&reply, 2);
                            ei_x_encode_atom(&reply, "reply");
                            ei_x_encode_atom(&reply, "ok");
                        }
                    }
#else
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "godot_not_available");
#endif
                }
            }
        }
    } else if (strcmp(atom, "call_node_method") == 0) {
        /* Call a method on a node */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            long node_id;
            char method_name[256];
            int arg_count;
            if (ei_decode_long(buf, index, &node_id) < 0 || 
                ei_decode_string(buf, index, method_name) < 0 ||
                ei_decode_long(buf, index, (long*)&arg_count) < 0) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "invalid_parameters");
            } else {
                inst = find_instance(instance_id);
                if (inst == NULL || !inst->started) {
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "instance_not_started");
                } else {
#ifdef GODOT_AVAILABLE
                    Node *node = get_node_by_id(node_id);
                    if (node == NULL) {
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "error");
                        ei_x_encode_string(&reply, "node_not_found");
                    } else {
                        StringName method(method_name);
                        Variant *args = NULL;
                        if (arg_count > 0) {
                            args = (Variant *)malloc(sizeof(Variant) * arg_count);
                            for (int i = 0; i < arg_count; i++) {
                                args[i] = bert_to_variant(buf, index);
                            }
                        }
                        Callable::CallError ce;
                        Variant result = node->callp(method, (const Variant **)args, arg_count, ce);
                        if (args) free(args);
                        if (ce.error != Callable::CallError::CALL_OK) {
                            ei_x_encode_tuple_header(&reply, 2);
                            ei_x_encode_atom(&reply, "error");
                            char error_msg[256];
                            snprintf(error_msg, sizeof(error_msg), "call_error_%d", (int)ce.error);
                            ei_x_encode_string(&reply, error_msg);
                        } else {
                            ei_x_encode_tuple_header(&reply, 2);
                            ei_x_encode_atom(&reply, "reply");
                            variant_to_bert(result, &reply);
                        }
                    }
#else
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "godot_not_available");
#endif
                }
            }
        }
    } else if (strcmp(atom, "get_node_children") == 0) {
        /* Get children of a node */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            long node_id;
            if (ei_decode_long(buf, index, &node_id) < 0) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "invalid_node_id");
            } else {
                inst = find_instance(instance_id);
                if (inst == NULL || !inst->started) {
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "instance_not_started");
                } else {
#ifdef GODOT_AVAILABLE
                    Node *node = get_node_by_id(node_id);
                    if (node == NULL) {
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "error");
                        ei_x_encode_string(&reply, "node_not_found");
                    } else {
                        int child_count = node->get_child_count();
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "reply");
                        ei_x_encode_list_header(&reply, child_count);
                        for (int i = 0; i < child_count; i++) {
                            Node *child = node->get_child(i);
                            ei_x_encode_tuple_header(&reply, 2);
                            const char *child_name = get_node_name(child);
                            ei_x_encode_string(&reply, child_name ? child_name : "");
                            ei_x_encode_long(&reply, (int64_t)child->get_instance_id());
                        }
                        ei_x_encode_empty_list(&reply);
                    }
#else
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "godot_not_available");
#endif
                }
            }
        }
    } else if (strcmp(atom, "get_node_class") == 0) {
        /* Get the class name of a node */
        if (ei_decode_long(buf, index, &instance_id) < 0) {
            ei_x_encode_tuple_header(&reply, 2);
            ei_x_encode_atom(&reply, "error");
            ei_x_encode_string(&reply, "invalid_instance_id");
        } else {
            long node_id;
            if (ei_decode_long(buf, index, &node_id) < 0) {
                ei_x_encode_tuple_header(&reply, 2);
                ei_x_encode_atom(&reply, "error");
                ei_x_encode_string(&reply, "invalid_node_id");
            } else {
                inst = find_instance(instance_id);
                if (inst == NULL || !inst->started) {
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "instance_not_started");
                } else {
#ifdef GODOT_AVAILABLE
                    Node *node = get_node_by_id(node_id);
                    if (node == NULL) {
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "error");
                        ei_x_encode_string(&reply, "node_not_found");
                    } else {
                        String class_name = node->get_class();
                        CharString utf8 = class_name.utf8();
                        ei_x_encode_tuple_header(&reply, 2);
                        ei_x_encode_atom(&reply, "reply");
                        ei_x_encode_string(&reply, utf8.get_data());
                    }
#else
                    ei_x_encode_tuple_header(&reply, 2);
                    ei_x_encode_atom(&reply, "error");
                    ei_x_encode_string(&reply, "godot_not_available");
#endif
                }
            }
        }
    } else {
        /* Unknown function */
        ei_x_encode_tuple_header(&reply, 2);
        ei_x_encode_atom(&reply, "error");
        ei_x_encode_string(&reply, "unknown_function");
    }
    
    /* Send reply */
    send_reply(&reply, fd);
    ei_x_free(&reply);
    
    return 0;
}

/*
 * Handle asynchronous cast from Erlang/Elixir
 */
static int handle_cast(char *buf, int *index) {
    char atom[MAXATOMLEN];
    
    /* Get the function name */
    if (ei_decode_atom(buf, index, atom) < 0) {
        fprintf(stderr, "Error decoding function atom\n");
        return -1;
    }
    
    /* Handle different cast functions */
    if (strcmp(atom, "log") == 0) {
        char msg[256];
        if (ei_decode_string(buf, index, msg) == 0) {
            printf("[Godot CNode] %s\n", msg);
        }
    }
    
    return 0;
}

/*
 * Send reply to Erlang/Elixir
 * Note: For CNode communication, we need to track the 'from' pid
 * For now, we'll use ei_send_encoded with NULL pid (sends to connected node)
 */
static void send_reply(ei_x_buff *x, int fd) {
    /* Send encoded message to the connected node */
    if (ei_send_encoded(fd, NULL, x->buff, x->index) < 0) {
        fprintf(stderr, "Error sending reply\n");
    }
}

/*
 * Main loop - listen for messages from Erlang/Elixir
 */
extern "C" {
void main_loop() {
    ei_x_buff x;
    erlang_msg msg;
    int fd;
    int res;
    
    printf("Godot CNode: Entering main loop\n");
    
    ei_x_new(&x);
    
    while (1) {
        /* Accept connection from Erlang/Elixir node */
        fd = ei_accept(&ec, listen_fd, NULL);
        if (fd < 0) {
            fprintf(stderr, "ei_accept failed: %d\n", fd);
            break;
        }
        
        /* Receive message */
        res = ei_receive_msg(fd, &msg, &x);
        
        if (res == ERL_TICK) {
            /* Just a tick, continue */
            continue;
        } else if (res == ERL_ERROR) {
            fprintf(stderr, "Error receiving message: %d\n", res);
            close(fd);
            continue;
        }
        
        /* Process the message */
        if (process_message(x.buff, &x.index, fd) < 0) {
            fprintf(stderr, "Error processing message\n");
        }
        
        close(fd);
    }
    
    ei_x_free(&x);
}
} // extern "C"

// Main function moved to godot_main_cnode.cpp

