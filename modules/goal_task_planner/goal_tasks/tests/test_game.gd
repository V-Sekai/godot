# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# test_vr_community.gd
# SPDX-License-Identifier: MIT

extends GutTest
"""
 A simplified example using the task_goal plugin framework for Godot
 demonstrating the planning of activities involved in running a VR social
 network like V-Sekai.
"""

var domain_name = "vr_community_management"
var the_domain = preload("res://addons/task_goal/core/domain.gd").new(domain_name)

var planner = preload("res://addons/task_goal/core/plan.gd").new()

@export var types = {
	"activity": [
		"conceptualize_platform",
		"gather_requirements",
		"create_project_roadmap",
		"assemble_development_team",
		"choose_technology_stack",
		"design_system_architecture",
		"prototype_interactive_elements",
		"develop_user_interface_design",
		"establish_security_protocols",
		"build_core_functionality",
		"code_networking_features",
		"construct_virtual_worlds",
		"animate_avatars_and_objects",
		"compose_background_music",
		"design_spatial_audio_effects",
		"implement_voice_chat",
		"code_text_chat_system",
		"create_emote_system",
		"integrate_third_party_services",
		"develop_content_moderation_tools",
		"set_up_user_account_system",
		"configure_database_systems",
		"optimize_performance",
		"conduct_unit_testing",
		"execute_integration_testing",
		"perform_system_testing",
		"ensure_compatibility_with_devices",
		"develop_data_backup_solutions",
		"plan_for_scalability",
		"deploy_beta_version",
		"train_support_staff",
		"release_updates_and_patches",
		"maintain_server_infrastructure",
		"monitor_user_activity_trends",
		"analyze_platform_metrics",
		"gather_user_suggestions",
		"improve_user_experience",
		"expand_virtual_environments",
		"enhance_avatar_customization",
		"offer_new_social_features",
		"promote_content_creator_tools",
		"increase_platform_accessibility",
		"collaborate_with_partners",
		"celebrate_platform_anniversary",
		"launch_full_platform",
		"optimize_cloud_services",
		"facilitate_live_streaming_capabilities",
		"introduce_advanced_AI_features",
		"develop_global_leaderboards",
		"enhance_community_interaction_tools",
		"improve_asset_library",
		"deploy_continuous_integration_system",
		"establish_brand_guidelines",
		"host_global_tournaments"
	]
}

@export var dependencies: Dictionary = {
	"conceptualize_platform": [],
	"gather_requirements": ["conceptualize_platform"],
	"create_project_roadmap": ["gather_requirements"],
	"assemble_development_team": ["create_project_roadmap"],
	"choose_technology_stack": ["assemble_development_team"],
	"design_system_architecture": ["choose_technology_stack"],
	"prototype_interactive_elements": ["design_system_architecture"],
	"develop_user_interface_design": ["prototype_interactive_elements"],
	"establish_security_protocols": ["design_system_architecture"],
	"build_core_functionality": ["develop_user_interface_design", "establish_security_protocols"],
	"code_networking_features": ["build_core_functionality"],
	"construct_virtual_worlds": ["code_networking_features"],
	"animate_avatars_and_objects": ["construct_virtual_worlds"],
	"compose_background_music": ["animate_avatars_and_objects"],
	"design_spatial_audio_effects": ["compose_background_music"],
	"implement_voice_chat": ["code_networking_features"],
	"code_text_chat_system": ["implement_voice_chat"],
	"create_emote_system": ["code_text_chat_system"],
	"integrate_third_party_services": ["build_core_functionality"],
	"develop_content_moderation_tools": ["integrate_third_party_services"],
	"set_up_user_account_system": ["develop_content_moderation_tools"],
	"configure_database_systems": ["set_up_user_account_system"],
	"optimize_performance": ["configure_database_systems", "build_core_functionality"],
	"conduct_unit_testing": ["optimize_performance"],
	"execute_integration_testing": ["conduct_unit_testing"],
	"perform_system_testing": ["execute_integration_testing"],
	"ensure_compatibility_with_devices": ["perform_system_testing"],
	"develop_data_backup_solutions": ["configure_database_systems"],
	"plan_for_scalability": ["develop_data_backup_solutions"],
	"deploy_beta_version": ["plan_for_scalability", "perform_system_testing"],
	"train_support_staff": ["deploy_beta_version"],
	"release_updates_and_patches": ["train_support_staff"],
	"maintain_server_infrastructure": ["release_updates_and_patches"],
	"monitor_user_activity_trends": ["maintain_server_infrastructure"],
	"analyze_platform_metrics": ["monitor_user_activity_trends"],
	"gather_user_suggestions": ["analyze_platform_metrics"],
	"improve_user_experience": ["gather_user_suggestions"],
	"expand_virtual_environments": ["improve_user_experience"],
	"enhance_avatar_customization": ["expand_virtual_environments"],
	"offer_new_social_features": ["enhance_avatar_customization"],
	"promote_content_creator_tools": ["offer_new_social_features"],
	"increase_platform_accessibility": ["promote_content_creator_tools"],
	"collaborate_with_partners": ["increase_platform_accessibility"],
	"celebrate_platform_anniversary": ["launch_full_platform"],
	"launch_full_platform": ["deploy_beta_version"],
	"optimize_cloud_services": ["configure_database_systems"],
	"facilitate_live_streaming_capabilities": ["build_core_functionality"],
	"introduce_advanced_AI_features": ["design_system_architecture"],
	"develop_global_leaderboards": ["code_networking_features"],
	"enhance_community_interaction_tools": ["monitor_user_activity_trends"],
	"improve_asset_library": ["build_core_functionality"],
	"deploy_continuous_integration_system": ["conduct_unit_testing"],
	"establish_brand_guidelines": ["conceptualize_platform"],
	"host_global_tournaments": ["develop_global_leaderboards"]
}


# Prototypical initial state, can include current progress, resources etc.
var state0: Dictionary = {
	# Example state properties
	"completed_activities": [],
	"pending_activities": types["activity"]  # Initial list from 'types'
}

# Helper functions:

func set_as_completed(state, activity_name):
	if activity_name in state.pending_activities:
		state.pending_activities.erase(activity_name)
		state.completed_activities.append(activity_name)
	return state


## Actions:

func complete_activity(state, activity) -> Variant:
	if activity in state.pending_activities:
		return set_as_completed(state, activity)
	return false


## Methods:

func schedule_activity(state: Dictionary, activity: String) -> Array:
	var actions = []

	if not activity in state["pending_activities"]:
		return []

	for predecessor in dependencies[activity]:
		if not predecessor in state["completed_activities"]:
			actions.push_back(["schedule_activity", predecessor])

	actions.append(["complete_activity", activity])

	return actions


func _ready() -> void:
	planner.domains.push_back(the_domain)
	planner.current_domain = the_domain
	planner.declare_actions([Callable(self, "complete_activity")])
	planner.declare_task_methods(
		"schedule_activity",
		[
			Callable(self, "schedule_activity"),
		]
	)

func test_vr_community_planning() -> void:
	planner.current_domain = the_domain

	# Example: Executing a single activity
	var state1 = state0.duplicate(true)
	var goals = [["schedule_activity", "launch_full_platform"]]
	var plan: Variant = planner.find_plan(state1, goals)

	# Assert that plan is not false and has a positive size
	assert_true( not plan is bool and plan.size() > 0)

	gut.p("Plan to finish:")
	if typeof(plan) == TYPE_ARRAY:
		for action in plan:
			gut.p("- %s" % str(action))
