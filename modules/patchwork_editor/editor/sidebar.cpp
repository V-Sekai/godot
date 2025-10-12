#include "sidebar.h"

#include "core/os/os.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "scene/gui/option_button.cpp"

#include "patchwork_editor.h"

Sidebar::Sidebar(EditorPlugin *p_plugin, GodotProject *p_godot_project) : MarginContainer() {
    plugin = p_plugin;
    godot_project = p_godot_project;
    
    set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
    
    branch_picker = memnew(OptionButton);
    add_child(branch_picker);
    
    // Basic setup
    _ready();
}

Sidebar::~Sidebar() {
}

void Sidebar::_ready() {
    // Initialize UI elements
    // For now, empty
    print_line("Sidebar ready!");
}
