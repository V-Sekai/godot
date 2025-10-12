#ifndef SIDEBAR_H
#define SIDEBAR_H

#include "scene/gui/margin_container.h"

#include "core/os/os.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "scene/gui/item_list.h"
#include "scene/gui/line_edit.cpp"

class GodotProject;

class Sidebar : public MarginContainer {
    GDCLASS(Sidebar, MarginContainer);

private:
    EditorPlugin *plugin;
    GodotProject *godot_project;

    OptionButton *branch_picker;
    ItemList *history_list;
    Button *user_button;
    VBoxContainer *main_diff_container;
    Control *merge_preview_modal;

    // Other UI elements as onready vars
    Button *cancel_merge_button;
    Button *confirm_merge_button;
    Label *merge_preview_title;
    Label *merge_preview_source_label;
    Label *merge_preview_target_label;
    MarginContainer *merge_preview_diff_container;
    Button *sync_status_icon;
    Button *fork_button;
    Button *merge_button;

    Button *history_section_header;
    Control *history_section_body;
    Button *diff_section_header;
    Control *diff_section_body;
    Button *branch_picker_cover;
    Button *init_button;
    LineEdit *project_id_box;
    Button *load_existing_button;
    Button *copy_project_id_button;
    Button *clear_project_button;

    // Signals
    void _on_reload_ui_button_pressed();
    void _on_init_button_pressed();
    void _on_load_project_button_pressed();
    void _on_user_button_pressed(bool disable_cancel = false);
    void _on_fork_button_pressed();
    void _on_merge_button_pressed();
    void _on_cancel_merge_button_pressed();
    void _on_confirm_merge_button_pressed();
    void _on_clear_diff_button_pressed();
    void _on_branch_picker_item_selected(int index);
    void _on_history_list_item_selected(int index, bool at_pos, int mouse_button_index);
    void _on_sync_status_icon_pressed();
    void _on_history_section_header_pressed();
    void _on_diff_section_header_pressed();

    void update_ui(bool update_diff = false);
    void update_branch_picker();
    void update_history_list();
    void update_sync_status();
    void toggle_section(bool visible);

    bool dev_mode = true;

    // Called when this node enters the scene tree for the first time.
    void _ready();
    void _process(double delta);

public:
    Sidebar(EditorPlugin *p_plugin = nullptr, GodotProject *p_godot_project = nullptr);
    ~Sidebar();

    void init();
    void wait_for_checked_out_branch();
    bool check_and_prompt_for_user_name();
    void clear_project();
    void create_new_branch(bool disable_cancel = false);
    void checkout_branch(String branch_id);
    void create_merge_preview_branch();
    void cancel_merge_preview();
    void confirm_merge_preview();
    void move_inspector_to_merge_preview();
    void move_inspector_to_main();
};

#endif // SIDEBAR_H
