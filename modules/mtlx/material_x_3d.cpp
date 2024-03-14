/**************************************************************************/
/*  material_x_3d.cpp                                                     */
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
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "material_x_3d.h"

#include "core/config/project_settings.h"
#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "modules/tinyexr/image_loader_tinyexr.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/material.h"
#include "scene/resources/visual_shader.h"

void apply_materialx_modifiers(mx::DocumentPtr doc, const DocumentModifiers &modifiers) {
	for (mx::ElementPtr elem : doc->traverseTree()) {
		if (modifiers.remapElements.count(elem->getCategory())) {
			elem->setCategory(modifiers.remapElements.at(elem->getCategory()));
		}
		if (modifiers.remapElements.count(elem->getName())) {
			elem->setName(modifiers.remapElements.at(elem->getName()));
		}
		mx::StringVec attrNames = elem->getAttributeNames();
		for (const std::string &attrName : attrNames) {
			if (modifiers.remapElements.count(elem->getAttribute(attrName))) {
				elem->setAttribute(
						attrName, modifiers.remapElements.at(elem->getAttribute(attrName)));
			}
		}
		if (elem->hasFilePrefix() && !modifiers.filePrefixTerminator.empty()) {
			std::string filePrefix = elem->getFilePrefix();
			if (!mx::stringEndsWith(filePrefix, modifiers.filePrefixTerminator)) {
				elem->setFilePrefix(filePrefix + modifiers.filePrefixTerminator);
			}
		}
		std::vector<mx::ElementPtr> children = elem->getChildren();
		for (mx::ElementPtr child : children) {
			if (modifiers.skipElements.count(child->getCategory()) ||
					modifiers.skipElements.count(child->getName())) {
				elem->removeChild(child->getName());
			}
		}
	}

	// Remap references to unimplemented shader nodedefs.
	for (mx::NodePtr materialNode : doc->getMaterialNodes()) {
		for (mx::NodePtr shader : getShaderNodes(materialNode)) {
			mx::NodeDefPtr nodeDef = shader->getNodeDef();
			if (nodeDef && !nodeDef->getImplementation()) {
				std::vector<mx::NodeDefPtr> altNodeDefs =
						doc->getMatchingNodeDefs(nodeDef->getNodeString());
				for (mx::NodeDefPtr altNodeDef : altNodeDefs) {
					if (altNodeDef->getImplementation()) {
						shader->setNodeDefString(altNodeDef->getName());
					}
				}
			}
		}
	}
}

/*
Variant get_value_as_material_x_variant(mx::InputPtr p_input) {
	if (!p_input) {
		return Variant();
	}
	mx::ValuePtr value = p_input->getValue();
	if (!value) {
		return Variant();
	}
	if (value->getTypeString() == "float") {
		return value->asA<float>();
	} else if (value->getTypeString() == "integer") {
		return value->asA<int>();
	} else if (value->getTypeString() == "boolean") {
		return value->asA<bool>();
	} else if (value->getTypeString() == "color3") {
		mx::Color3 color = value->asA<mx::Color3>();
		return Color(color[0], color[1], color[2]);
	} else if (value->getTypeString() == "color4") {
		mx::Color4 color = value->asA<mx::Color4>();
		return Color(color[0], color[1], color[2], color[3]);
	} else if (value->getTypeString() == "vector2") {
		mx::Vector2 vector_2 = value->asA<mx::Vector2>();
		return Vector2(vector_2[0], vector_2[1]);
	} else if (value->getTypeString() == "vector3") {
		mx::Vector3 vector_3 = value->asA<mx::Vector3>();
		return Vector3(vector_3[0], vector_3[1], vector_3[2]);
	} else if (value->getTypeString() == "vector4") {
		mx::Vector4 vector_4 = value->asA<mx::Vector4>();
		return Color(vector_4[0], vector_4[1], vector_4[2], vector_4[3]);
	} else if (value->getTypeString() == "matrix33") {
		// Matrix33 m = value->asA<Matrix33>();
		// TODO: fire 2022-03-11 add basis
	} else if (value->getTypeString() == "matrix44") {
		// Matrix44 m = value->asA<Matrix44>();
		// TODO: fire 2022-03-11 add transform
	}
	return Variant();
}
*/

Error load_mtlx_document(mx::DocumentPtr p_doc, String p_path) {
	mx::FilePath materialFilename = ProjectSettings::get_singleton()->globalize_path(p_path).utf8().get_data();
	std::vector<MaterialPtr> materials;
	mx::DocumentPtr dependLib = mx::createDocument();
	mx::StringSet skipLibraryFiles;
	mx::DocumentPtr stdLib;
	mx::StringSet xincludeFiles;

	mx::StringVec distanceUnitOptions;
	mx::LinearUnitConverterPtr distanceUnitConverter;

	mx::UnitConverterRegistryPtr unitRegistry =
			mx::UnitConverterRegistry::create();
	mx::FileSearchPath searchPath;
	try {
		stdLib = mx::createDocument();
		mx::FilePathVec libraryFolders;
		libraryFolders.push_back(ProjectSettings::get_singleton()->globalize_path(p_path.get_base_dir()).utf8().get_data());
		libraryFolders.push_back(ProjectSettings::get_singleton()->globalize_path("res://libraries").utf8().get_data());
		libraryFolders.push_back(ProjectSettings::get_singleton()->globalize_path("user://libraries").utf8().get_data());
		mx::StringSet xincludeFiles = mx::loadLibraries(libraryFolders, searchPath, stdLib);
		// Import libraries.
		if (xincludeFiles.empty()) {
			std::cerr << "Could not find standard data libraries on the given "
						 "search path: "
					  << searchPath.asString() << std::endl;
			return FAILED;
		}

		// // Initialize color management.
		// mx::DefaultColorManagementSystemPtr cms =
		// 		mx::DefaultColorManagementSystem::create(
		// 				context.getShaderGenerator().getTarget());
		// cms->loadLibrary(stdLib);
		// context.getShaderGenerator().setColorManagementSystem(cms);

		// // Initialize unit management.
		// mx::UnitSystemPtr unitSystem =
		// 		mx::UnitSystem::create(context.getShaderGenerator().getTarget());
		// unitSystem->loadLibrary(stdLib);
		// unitSystem->setUnitConverterRegistry(unitRegistry);
		// context.getShaderGenerator().setUnitSystem(unitSystem);
		// context.getOptions().targetDistanceUnit = "meter";

		mx::UnitTypeDefPtr distanceTypeDef = stdLib->getUnitTypeDef("distance");
		distanceUnitConverter = mx::LinearUnitConverter::create(distanceTypeDef);
		unitRegistry->addUnitConverter(distanceTypeDef, distanceUnitConverter);
		mx::UnitTypeDefPtr angleTypeDef = stdLib->getUnitTypeDef("angle");
		mx::LinearUnitConverterPtr angleConverter =
				mx::LinearUnitConverter::create(angleTypeDef);
		unitRegistry->addUnitConverter(angleTypeDef, angleConverter);

		auto unitScales = distanceUnitConverter->getUnitScale();
		distanceUnitOptions.resize(unitScales.size());
		for (auto unitScale : unitScales) {
			int location = distanceUnitConverter->getUnitAsInteger(unitScale.first);
			distanceUnitOptions[location] = unitScale.first;
		}

		// Clear user data on the generator.
		// context.clearUserData();
	} catch (std::exception &e) {
		std::cerr << "Failed to load standard data libraries: " << e.what()
				  << std::endl;
		return FAILED;
	}

	p_doc->importLibrary(stdLib);

	MaterialX::FilePath parentPath = materialFilename.getParentPath();
	searchPath.append(materialFilename.getParentPath());

	// Set up read options.
	mx::XmlReadOptions readOptions;
	readOptions.readXIncludeFunction = [](mx::DocumentPtr p_doc,
											   const mx::FilePath &materialFilename,
											   const mx::FileSearchPath &searchPath,
											   const mx::XmlReadOptions *newReadoptions) {
		mx::FilePath resolvedFilename = searchPath.find(materialFilename);
		if (resolvedFilename.exists()) {
			readFromXmlFile(p_doc, resolvedFilename, searchPath, newReadoptions);
		} else {
			std::cerr << "Include file not found: " << materialFilename.asString()
					  << std::endl;
		}
	};
	mx::readFromXmlFile(p_doc, materialFilename, searchPath, &readOptions);

	DocumentModifiers modifiers;
	// TODO: fire 2022-03-11 Does nothing yet.
	// Apply modifiers to the content document.
	apply_materialx_modifiers(p_doc, modifiers);

	// Validate the document.
	std::string message;
	if (!p_doc->validate(&message)) {
		print_line(vformat(String("Validation warnings for %s"), String(message.c_str())));
	}
	return OK;
}

Variant MTLXLoader::_load(const String &p_save_path, const String &p_original_path, bool p_use_sub_threads, int64_t p_cache_mode) const {
	String save_path = ProjectSettings::get_singleton()->globalize_path(p_save_path);
	String original_path = ProjectSettings::get_singleton()->globalize_path(p_original_path);
	mx::DocumentPtr doc = mx::createDocument();
	Error err;
	try {
		err = load_mtlx_document(doc, original_path);
	} catch (std::exception &e) {
		ERR_PRINT(String("Can't load Materialx materials. Error: ") + String(e.what()));
		return Ref<Resource>();
	}
	if (err != OK) {
		return Ref<Resource>();
	}

	std::string message;
	bool docValid = doc->validate(&message);
	ERR_FAIL_COND_V_MSG(!docValid, Ref<Resource>(), String("The MaterialX document is invalid: [") + String(doc->getSourceUri().c_str()) + "] " + String(message.c_str()));

	std::vector<mx::TypedElementPtr> renderable_materials = findRenderableElements(doc);
	Ref<ShaderMaterial> mat;
	mat.instantiate();
	Ref<VisualShader> shader;
	shader.instantiate();
	std::set<mx::NodePtr> processed_nodes;
	int id = 2;
	std::map<mx::NodePtr, int> node_ids; // TODO move to godot map.
	std::map<mx::NodePtr, std::map<std::string, int>> node_input_slot_maps; // TODO move to godot map.
	std::map<mx::NodePtr, std::map<std::string, int>> node_output_slot_maps; // TODO move to godot map.

	for (size_t i = 0; i < renderable_materials.size(); i++) {
		const mx::TypedElementPtr &element = renderable_materials[i];
		if (!element || !element->isA<mx::Node>()) {
			continue;
		}
		const mx::NodePtr &node = element->asA<mx::Node>();
		if (!node) {
			continue;
		}

		// Input slot map
		std::map<std::string, int> input_slot_map;
		int input_slot_number = 0;
		for (mx::InputPtr input : node->getInputs()) {
			input_slot_map[input->getName()] = input_slot_number++;
		}
		node_input_slot_maps[node] = input_slot_map;
		print_line(String("Input slot map for node: ") + node->getName().c_str());
		for (const auto &pair : input_slot_map) {
			print_line(String("Input: ") + String(pair.first.c_str()) + ", Slot: " + itos(pair.second));
		}

		// Output slot map
		std::map<std::string, int> output_slot_map;
		int output_slot_number = 0;
		for (mx::OutputPtr output : node->getOutputs()) {
			output_slot_map[output->getName()] = output_slot_number++;
		}
		node_output_slot_maps[node] = output_slot_map;
		print_line(String("Output slot map for node: ") + node->getName().c_str());
		for (const auto &pair : output_slot_map) {
			print_line(String("Output: ") + String(pair.first.c_str()) + ", Slot: " + itos(pair.second));
		}

		String shader_name = node->getCategory().c_str();
		std::vector<mx::NodePtr> shader_nodes = mx::getShaderNodes(node); // Convert to Godot Vector
		if (!shader_nodes.empty()) {
			shader_name = shader_nodes[0]->getCategory().c_str();
		}
		print_line(String("Shader Name: ") + shader_name);

		create_node(node, 0, shader, processed_nodes, id, node_ids);
	}

	for (size_t i = 0; i < renderable_materials.size(); i++) {
		const mx::TypedElementPtr &element = renderable_materials[i];
		if (!element || !element->isA<mx::Node>()) {
			continue;
		}
		const mx::NodePtr &node = element->asA<mx::Node>();
		if (!node) {
			continue;
		}
		connect_node(node, 0, shader, processed_nodes, id, node_ids);
	}
	mat->set_shader(shader);
	return mat;
}

void MTLXLoader::create_node(const mx::NodePtr &node, int depth, Ref<VisualShader> &shader, std::set<mx::NodePtr> &processed_nodes,
		int &id, std::map<mx::NodePtr, int> &node_ids) const {
	if (processed_nodes.find(node) != processed_nodes.end()) {
		return;
	}
	processed_nodes.insert(node);
	node_ids[node] = id;
	Ref<VisualShaderNodeExpression> expression_node;
	expression_node.instantiate();
	String expression_text = String(node->getName().c_str());
	print_line(String("MaterialX node " + expression_text));
	expression_node->set_expression(expression_text);
	shader->add_node(VisualShader::TYPE_FRAGMENT, expression_node, Vector2(depth * 200, -200), id++);
	int i = 0;
	for (mx::InputPtr input : node->getInputs()) {
		const std::string &input_name = input->getName();
		print_line(String("MaterialX input " + String(input_name.c_str())));
		mx::ValuePtr value = input->getValue();
		if (value) {
			std::string typeString = value->getTypeString();
			print_line(String("MaterialX input type: ") + typeString.c_str());
		}
		expression_node->add_input_port(i, Variant::NIL, input_name.c_str());
		i++;
	}
	i = 0;
	for (mx::OutputPtr output : node->getOutputs()) {
		const std::string &output_name = output->getName();
		print_line(String("MaterialX output " + String(output_name.c_str())));
		expression_node->add_output_port(i, Variant::NIL, output_name.c_str());
		i++;
	}
	for (mx::ElementPtr child_element : node->getChildren()) {
		mx::NodePtr child_node = child_element->asA<mx::Node>();
		if (child_node) {
			create_node(child_node, depth + 1, shader, processed_nodes, id, node_ids); // increment depth for each recursive call
		}
	}
}

void MTLXLoader::connect_node(const mx::NodePtr &node, int depth, Ref<VisualShader> &shader, std::set<mx::NodePtr> &processed_nodes,
		int &id, const std::map<mx::NodePtr, int> &node_ids) const {
	for (mx::InputPtr input : node->getInputs()) {
		if (input->getConnectedNode() != nullptr) {
			mx::NodePtr connected_node = input->getConnectedNode();
			if (node_ids.find(connected_node) == node_ids.end()) {
				ERR_PRINT("Error: Connected MaterialX Node '" + String(connected_node->getName().c_str()) + "' not found in the map.");
				continue;
			}
			if (node_ids.find(node) == node_ids.end()) {
				ERR_PRINT("Error: MaterialX Node '" + String(node->getName().c_str()) + "' not found in the map.");
				continue;
			}
			int source_id = node_ids.at(connected_node);
			int target_id = node_ids.at(node);
			shader->connect_nodes(VisualShader::TYPE_FRAGMENT, source_id, 0, target_id, 0);
		}
	}
}

int MTLXLoader::get_node_id(const mx::NodePtr &node, const std::map<mx::NodePtr, int> &node_ids) const {
	auto it = node_ids.find(node); // TODO: Remove auto.
	if (it != node_ids.end()) {
		return it->second;
	} else {
		// Handle error: Node ID not found
		ERR_PRINT("Node ID not found");
		return -1;
	}
}
