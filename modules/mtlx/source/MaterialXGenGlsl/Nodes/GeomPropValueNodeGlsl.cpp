/**************************************************************************/
/*  GeomPropValueNodeGlsl.cpp                                             */
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

//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/Nodes/GeomPropValueNodeGlsl.h>

#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr GeomPropValueNodeGlsl::create()
{
    return std::make_shared<GeomPropValueNodeGlsl>();
}

void GeomPropValueNodeGlsl::createVariables(const ShaderNode& node, GenContext&, Shader& shader) const
{
    const ShaderInput* geomPropInput = node.getInput(GEOMPROP);
    if (!geomPropInput || !geomPropInput->getValue())
    {
        throw ExceptionShaderGenError("No 'geomprop' parameter found on geompropvalue node '" + node.getName() + "'. Don't know what property to bind");
    }
    const string geomProp = geomPropInput->getValue()->getValueString();
    const ShaderOutput* output = node.getOutput();

    ShaderStage& vs = shader.getStage(Stage::VERTEX);
    ShaderStage& ps = shader.getStage(Stage::PIXEL);

    addStageInput(HW::VERTEX_INPUTS, output->getType(), HW::T_IN_GEOMPROP + "_" + geomProp, vs);
    addStageConnector(HW::VERTEX_DATA, output->getType(), HW::T_IN_GEOMPROP + "_" + geomProp, vs, ps);
}

void GeomPropValueNodeGlsl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    const GlslShaderGenerator& shadergen = static_cast<const GlslShaderGenerator&>(context.getShaderGenerator());

    const ShaderInput* geomPropInput = node.getInput(GEOMPROP);
    if (!geomPropInput)
    {
        throw ExceptionShaderGenError("No 'geomprop' parameter found on geompropvalue node '" + node.getName() + "'. Don't know what property to bind");
    }
    const string geomname = geomPropInput->getValue()->getValueString();
    const string variable = HW::T_IN_GEOMPROP + "_" + geomname;

    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        VariableBlock& vertexData = stage.getOutputBlock(HW::VERTEX_DATA);
        const string prefix = shadergen.getVertexDataPrefix(vertexData);
        ShaderPort* geomprop = vertexData[variable];
        if (!geomprop->isEmitted())
        {
            shadergen.emitLine(prefix + geomprop->getVariable() + " = " + HW::T_IN_GEOMPROP + "_" + geomname, stage);
            geomprop->setEmitted();
        }
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        VariableBlock& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        const string prefix = shadergen.getVertexDataPrefix(vertexData);
        ShaderPort* geomprop = vertexData[variable];
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(node.getOutput(), true, false, context, stage);
        shadergen.emitString(" = " + prefix + geomprop->getVariable(), stage);
        shadergen.emitLineEnd(stage);
    }
}

ShaderNodeImplPtr GeomPropValueNodeGlslAsUniform::create()
{
    return std::make_shared<GeomPropValueNodeGlslAsUniform>();
}

void GeomPropValueNodeGlslAsUniform::createVariables(const ShaderNode& node, GenContext&, Shader& shader) const
{
    const ShaderInput* geomPropInput = node.getInput(GEOMPROP);
    if (!geomPropInput || !geomPropInput->getValue())
    {
        throw ExceptionShaderGenError("No 'geomprop' parameter found on geompropvalue node '" + node.getName() + "'. Don't know what property to bind");
    }
    const string geomProp = geomPropInput->getValue()->getValueString();
    ShaderStage& ps = shader.getStage(Stage::PIXEL);
    ShaderPort* uniform = addStageUniform(HW::PRIVATE_UNIFORMS, node.getOutput()->getType(), HW::T_GEOMPROP + "_" + geomProp, ps);
    uniform->setPath(geomPropInput->getPath());
}

void GeomPropValueNodeGlslAsUniform::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        const ShaderInput* geomPropInput = node.getInput(GEOMPROP);
        if (!geomPropInput)
        {
            throw ExceptionShaderGenError("No 'geomprop' parameter found on geompropvalue node '" + node.getName() + "'. Don't know what property to bind");
        }
        const string attrName = geomPropInput->getValue()->getValueString();
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(node.getOutput(), true, false, context, stage);
        shadergen.emitString(" = " + HW::T_GEOMPROP + "_" + attrName, stage);
        shadergen.emitLineEnd(stage);
    }
}

MATERIALX_NAMESPACE_END
