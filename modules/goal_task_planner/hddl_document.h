#pragma once

#include "core/object/ref_counted.h"

class HDDLDocument : public RefCounted {
	GDCLASS(HDDLDocument, RefCounted);

private:
	struct ParameterNode;
	struct AtomNode;
	struct ConditionNode;
	struct EffectNode;
	struct TaskNode;
	struct ActionNode;
	struct MethodNode;
	struct NumericExpressionNode;
	struct DurationConstraintNode;
	struct HDDLNode;

	enum TemporalQualifier { TQ_NONE, TQ_AT_START, TQ_AT_END, TQ_OVER_ALL };

	struct ParameterNode {
		String name;
		String type;
	};

	struct AtomNode {
		String name;
		Vector<ParameterNode> parameters;
	};

	struct NumericExpressionNode {
		String expression_string;
	};

	struct ConditionNode {
		enum Type { AND, OR, NOT, ATOM, COMPARISON, EXISTS, FORALL, UNKNOWN } type = UNKNOWN;
		TemporalQualifier temporal_qualifier = TQ_NONE;
		Vector<ConditionNode *> children;
		AtomNode *atom = nullptr;
		String comparison_operator;
		NumericExpressionNode *left_operand = nullptr;
		NumericExpressionNode *right_operand = nullptr;
		Vector<ParameterNode> quantified_variables;
	};

	struct EffectNode {
		enum Type { AND, WHEN, ATOM, ASSIGN, INCREASE, DECREASE, FORALL, UNKNOWN } type = UNKNOWN;
		TemporalQualifier temporal_qualifier = TQ_NONE;
		Vector<EffectNode *> children;
		ConditionNode *condition = nullptr;
		AtomNode *atom = nullptr;
		bool is_delete_effect = false;
		AtomNode *fluent = nullptr;
		NumericExpressionNode *expression = nullptr;
		Vector<ParameterNode> quantified_variables;
	};

	struct TaskNode {
		String name;
		Vector<ParameterNode> parameters;
		String id;
	};

	struct DurationConstraintNode {
		String constraint_string;
	};

	struct ActionNode {
		String name;
		Vector<ParameterNode> parameters;
		DurationConstraintNode *duration = nullptr;
		ConditionNode *precondition = nullptr;
		EffectNode *effect = nullptr;
	};

	struct MethodNode {
		String name;
		String task_name;
		Vector<ParameterNode> parameters;
		ConditionNode *precondition = nullptr;
		enum Ordering { ORDERED, UNORDERED } ordering = ORDERED;
		Vector<TaskNode> subtasks;
	};

	struct HDDLNode {
		String type;
		String name;
		Vector<HDDLNode *> children;
		Vector<ParameterNode> parameters;
		ConditionNode *precondition = nullptr;
		EffectNode *effect = nullptr;
		ConditionNode *goal = nullptr;
		Vector<AtomNode> initial_state;
		Vector<ParameterNode> objects;
	};

	Error load_file_string(const String &path, String &str);
	Error save_string_to_file(const String &path, const String &str);
	Variant hddl_ast_to_variant(const HDDLNode *node, Dictionary &state);
	HDDLNode *variant_to_hddl_node(const Variant &data, Dictionary &state);
	void delete_hddl_tree(HDDLNode *node);

	String generate_predicate_string(const AtomNode *predicate, Dictionary &state);
	String generate_action_string(const ActionNode *action, Dictionary &state);
	String generate_condition_string(const ConditionNode *cond, Dictionary &state);
	String generate_effect_string(const EffectNode *effect, Dictionary &state);
	String generate_task_string(const TaskNode *task, Dictionary &state);

public:
	HDDLDocument();
	~HDDLDocument();

	virtual Error append_from_string(const String &hddl_string, Object *&root_node_obj);
	virtual Error append_from_file(const String &path, Object *&root_node_obj);

	virtual Error write_to_string(const Variant &root_node_data, String &hddl_string);
	virtual Error write_to_filesystem(const Variant &root_node_data, const String &path);
};
