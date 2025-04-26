#include "hddl_document.h"
#include "core/io/file_access.h"
#include "core/variant/dictionary.h"

HDDLDocument::HDDLDocument() {}
HDDLDocument::~HDDLDocument() {}

Error HDDLDocument::append_from_string(const String &hddl_string, Object *&root_node_obj) {
	return Error::OK;
}

Error HDDLDocument::append_from_file(const String &path, Object *&root_node_obj) {
	String hddl_string;
	Error err = load_file_string(path, hddl_string);
	if (err != OK) {
		return err;
	}
	return append_from_string(hddl_string, root_node_obj);
}

Error HDDLDocument::write_to_string(const Variant &root_node_data, String &hddl_string) {
	hddl_string = "";
	return Error::OK;
}

Error HDDLDocument::write_to_filesystem(const Variant &root_node_data, const String &path) {
	String hddl_string;
	Error err = write_to_string(root_node_data, hddl_string);
	if (err != OK) {
		return err;
	}
	return save_string_to_file(path, hddl_string);
}

Error HDDLDocument::load_file_string(const String &path, String &str) {
	Error err;
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ, &err);
	if (err != OK) {
		return err;
	}
	str = file->get_as_text();
	return OK;
}

Error HDDLDocument::save_string_to_file(const String &path, const String &str) {
	Error err;
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE, &err);
	if (err != OK) {
		return err;
	}
	file->store_string(str);
	return OK;
}

Variant HDDLDocument::hddl_ast_to_variant(const HDDLNode *node, Dictionary &state) {
	return Variant();
}

HDDLDocument::HDDLNode *HDDLDocument::variant_to_hddl_node(const Variant &data, Dictionary &state) {
	return nullptr;
}

void HDDLDocument::delete_hddl_tree(HDDLNode *node) {
	if (!node) return;
	for (HDDLNode *child : node->children) {
		delete_hddl_tree(child);
	}
	delete node->precondition;
	delete node->effect;
	delete node;
}

String HDDLDocument::generate_predicate_string(const AtomNode *predicate, Dictionary &state) {
	return String();
}

String HDDLDocument::generate_action_string(const ActionNode *action, Dictionary &state) {
	return String();
}

String HDDLDocument::generate_condition_string(const ConditionNode *cond, Dictionary &state) {
	return String();
}

String HDDLDocument::generate_effect_string(const EffectNode *effect, Dictionary &state) {
	return String();
}

String HDDLDocument::generate_task_string(const TaskNode *task, Dictionary &state) {
	return String();
}
