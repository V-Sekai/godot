
// Copyright (c) 2020-2021, ARM Limited.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#include "tosa_serialization_handler.h"

#include <iostream>
using namespace tosa;

TosaSerializationTensor::TosaSerializationTensor(const flatbuffers::String* name,
                                                 const flatbuffers::Vector<int32_t>* shape,
                                                 DType dtype,
                                                 const flatbuffers::Vector<uint8_t>* data)
{
    _dtype = dtype;

    std::copy(shape->begin(), shape->end(), std::back_inserter(_shape));

    assert(name);
    _name = name->str();

    if (data)
    {
        std::copy(data->begin(), data->end(), std::back_inserter(_data));
    }
}

TosaSerializationTensor::TosaSerializationTensor(const std::string& name,
                                                 const std::vector<int32_t>& shape,
                                                 DType dtype,
                                                 const std::vector<uint8_t>& data)
{
    _dtype = dtype;
    _shape = shape;
    _name  = name;
    _data  = data;
}

TosaSerializationTensor::TosaSerializationTensor()
{
    _dtype = DType_UNKNOWN;
    _name  = "UNKNOWN";
}

TosaSerializationTensor::~TosaSerializationTensor()
{}

void TosaSerializationOperator::InitializeAttributeQinfo(Attribute attribute_type,
                                                         const TosaAttributeBase* attribute,
                                                         QuantInfo qinfo_type,
                                                         const TosaQuantInfoBase* qinfo)
{
    _attribute_type = attribute_type;

    switch (attribute_type)
    {
        case Attribute_NONE:
            _attribute = new TosaNoneAttribute();
            break;
#define DEF_ATTRIBUTE(NAME, ...)                                                                                       \
    case Attribute_##NAME##Attribute:                                                                                  \
        _attribute = new Tosa##NAME##Attribute(attribute);                                                             \
        break;
#include "attribute.def"
#undef DEF_ATTRIBUTE
        default:
            printf("TosaSerializationOperator::TosaSerializationOperator(): Attribute %s not implemented yet\n",
                   EnumNamesAttribute()[attribute_type]);
            assert(0);
    }

    _qinfo_type = qinfo_type;
    switch (qinfo_type)
    {
        case QuantInfo_NONE:
            _qinfo = new TosaNoneQuantInfo();
            break;
#define DEF_QUANTIZATION_INFO(NAME, ...)                                                                               \
    case QuantInfo_##NAME##QuantInfo:                                                                                  \
        _qinfo = new Tosa##NAME##QuantInfo(qinfo);                                                                     \
        break;
#include "quant_info.def"
#undef DEF_QUANTIZATION_INFO
        default:
            printf("TosaSerializationOperator::TosaSerializationOperator(): QuantInfo %s not implemented yet\n",
                   EnumNamesQuantInfo()[qinfo_type]);
            assert(0);
    }

    assert(_attribute && _qinfo);
}

TosaSerializationOperator::TosaSerializationOperator(Op op,
                                                     Attribute attribute_type,
                                                     const TosaAttributeBase* attribute,
                                                     QuantInfo qinfo_type,
                                                     const TosaQuantInfoBase* qinfo,
                                                     const std::vector<std::string>& input_tensor_names,
                                                     const std::vector<std::string>& output_tensor_names)
{
    _op                  = op;
    _input_tensor_names  = input_tensor_names;
    _output_tensor_names = output_tensor_names;

    InitializeAttributeQinfo(attribute_type, attribute, qinfo_type, qinfo);
}

TosaSerializationOperator::TosaSerializationOperator(Op op,
                                                     Attribute attribute_type,
                                                     const TosaAttributeBase* attribute,
                                                     QuantInfo qinfo_type,
                                                     const TosaQuantInfoBase* qinfo,
                                                     std::vector<std::string>&& input_tensor_names,
                                                     std::vector<std::string>&& output_tensor_names)
{
    _op                  = op;
    _input_tensor_names  = std::move(input_tensor_names);
    _output_tensor_names = std::move(output_tensor_names);

    InitializeAttributeQinfo(attribute_type, attribute, qinfo_type, qinfo);
}

TosaSerializationOperator::~TosaSerializationOperator()
{
    delete _attribute;
    delete _qinfo;
}

TosaSerializationBasicBlock::TosaSerializationBasicBlock(const std::string& name,
                                                         const std::vector<TosaSerializationOperator*>& operators,
                                                         const std::vector<TosaSerializationTensor*>& tensors,
                                                         const std::vector<std::string>& inputs,
                                                         const std::vector<std::string>& outputs)
{
    _name      = name;
    _operators = operators;
    _tensors   = tensors;
    _inputs    = inputs;
    _outputs   = outputs;
}

TosaSerializationBasicBlock::TosaSerializationBasicBlock(std::string&& name,
                                                         std::vector<TosaSerializationOperator*>&& operators,
                                                         std::vector<TosaSerializationTensor*>&& tensors,
                                                         std::vector<std::string>&& inputs,
                                                         std::vector<std::string>&& outputs)
{
    _name      = std::move(name);
    _operators = std::move(operators);
    _tensors   = std::move(tensors);
    _inputs    = std::move(inputs);
    _outputs   = std::move(outputs);
}

TosaSerializationBasicBlock::~TosaSerializationBasicBlock()
{
    // deallocate all operators
    for (auto op : GetOperators())
    {
        delete op;    // ~TosaSerializationOperator()
    }

    // deallocate all tensors
    for (auto ts : GetTensors())
    {
        delete ts;    // ~TosaSerializationTensor()
    }
}

TosaSerializationHandler::TosaSerializationHandler()
{
    _schemaLoaded = false;
    _version      = TosaVersion(TOSA_VERSION_MAJOR, TOSA_VERSION_MINOR, TOSA_VERSION_PATCH, TOSA_VERSION_DRAFT);
}

TosaSerializationHandler::~TosaSerializationHandler()
{
    Clear();    // deallocate all basic blocks
}

tosa_err_t TosaSerializationHandler::LoadFileSchema(const char* schema_filename)
{
    std::string schema;
    bool ok;

    ok = flatbuffers::LoadFile(schema_filename, false, &schema);
    if (!ok)
    {
        printf("Error loading schema file: %s\n", schema_filename);
        return TOSA_FILE_ERROR;
    }

    ok = _parser.Parse(schema.c_str());
    if (!ok)
    {
        printf("Error parsing ISA schema file: %s\n", schema_filename);
        return TOSA_FILE_ERROR;
    }
    _schemaLoaded = true;

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::LoadFileJson(const char* filename)
{
    std::string jsonfile;
    bool ok;
    tosa_err_t err;

    if (!_schemaLoaded)
    {
        return TOSA_SCHEMA_MISSING;
    }

    ok = flatbuffers::LoadFile(filename, false, &jsonfile);
    if (!ok)
    {
        printf("Error loading json file: %s\n", filename);
        return TOSA_FILE_ERROR;
    }

    ok = _parser.Parse(jsonfile.c_str());
    if (!ok)
    {
        printf("Error parsing json file: %s\n", filename);
        return TOSA_FILE_ERROR;
    }

    uint8_t* buf = _parser.builder_.GetBufferPointer();

    err = Deserialize(buf);
    if (err != TOSA_OK)
    {
        return err;
    }

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::SaveFileJson(const char* filename)
{
    std::string jsongen;
    tosa_err_t err;

    if (!_schemaLoaded)
    {
        return TOSA_SCHEMA_MISSING;
    }

    err = Serialize();
    if (err != TOSA_OK)
    {
        return err;
    }

    uint8_t* buf = _builder.GetBufferPointer();

    if (!GenerateText(_parser, buf, &jsongen))
    {
        printf("Couldn't serialize parsed data to JSON!\n");
        return TOSA_FILE_ERROR;
    }

    FILE* file = fopen(filename, "wb");

    if (!file)
    {
        printf("Couldn't open output file: %s\n", filename);
        return TOSA_FILE_ERROR;
    }

    if (fwrite(jsongen.c_str(), sizeof(char), jsongen.size(), file) != jsongen.size())
    {
        printf("Error writing to json output file: %s\n", filename);
        fclose(file);
        return TOSA_FILE_ERROR;
    }

    if (file)
        fclose(file);

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::LoadFileTosaFlatbuffer(const char* filename)
{
    std::string read_buffer;
    tosa_err_t err;
    uint8_t* buf;
    bool ok;

    ok = flatbuffers::LoadFile(filename, false, &read_buffer);
    if (!ok)
    {
        printf("Error loading flatbuffer file: %s\n", filename);
        return TOSA_FILE_ERROR;
    }

    buf = (uint8_t*)read_buffer.data();

    err = Deserialize(buf);
    if (err != TOSA_OK)
    {
        return err;
    }

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::SaveFileTosaFlatbuffer(const char* filename)
{
    tosa_err_t err;

    err = Serialize();
    if (err != TOSA_OK)
    {
        return err;
    }

    uint8_t* buf = _builder.GetBufferPointer();

    bool ok = flatbuffers::SaveFile(filename, (const char*)buf, _builder.GetSize(), false);
    if (!ok)
    {
        printf("Error saving floatbuffer file: %s\n", filename);
        return TOSA_FILE_ERROR;
    }

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::Clear()
{
    // deallocate all basic blocks
    for (auto bb : GetBlocks())
    {
        delete bb;
    }
    _blocks.clear();

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::Deserialize(const uint8_t* buf)
{
    if (!TosaGraphBufferHasIdentifier(buf))
    {
        printf("WARNING: TOSA file does not have TOSA file identifier\n");
    }
    auto fb_tosa_graph   = GetTosaGraph(buf);
    auto fb_tosa_version = fb_tosa_graph->version();
    auto fb_tosa_blocks  = fb_tosa_graph->blocks();

    std::vector<std::string> operator_inputs_container;
    std::vector<std::string> operator_outputs_container;

    std::vector<TosaSerializationOperator*> block_operators_container;
    std::vector<TosaSerializationTensor*> block_tensors_container;
    std::vector<std::string> block_inputs_container;
    std::vector<std::string> block_outputs_container;

    TosaAttributeBase* typed_attribute      = NULL;
    TosaQuantInfoBase* typed_qinfo          = NULL;
    TosaSerializationOperator* new_operator = NULL;
    TosaSerializationBasicBlock* new_block  = NULL;
    TosaSerializationTensor* new_tensor     = NULL;

    // erase container
    Clear();

    TosaVersion read_version(fb_tosa_version->_major(), fb_tosa_version->_minor(), fb_tosa_version->_patch(),
                             fb_tosa_version->_draft());

    TosaVersion::compat_t is_compat = read_version.is_compatible(GetVersion());
    switch (is_compat)
    {
        case TosaVersion::compat_t::COMPLETELY_COMPATIBLE:
            break;
        case TosaVersion::compat_t::PARTIALLY_COMPATIBLE:
            printf("WARNING: Read flatbuffer version %s is partially compatible with serializer version %s\n",
                   read_version.to_string().c_str(), GetVersion().to_string().c_str());
            break;
        case TosaVersion::compat_t::NOT_COMPATIBLE:
            printf("ERROR: Read flatbuffer version %s is not compatible with serializer version %s\n",
                   read_version.to_string().c_str(), GetVersion().to_string().c_str());
            return TOSA_VERSION_MISMATCH;
    }

    for (size_t i = 0; i < fb_tosa_blocks->size(); i++)
    {
        auto curr_block = fb_tosa_blocks->Get(i);

        auto block_name = curr_block->name()->str();

        auto fb_tosa_operators = curr_block->operators();
        block_operators_container.clear();
        for (size_t j = 0; j < fb_tosa_operators->size(); j++)
        {
            auto curr_operator = fb_tosa_operators->Get(j);

            auto operator_op         = curr_operator->op();
            auto attribute_type      = curr_operator->attribute_type();
            auto attribute           = curr_operator->attribute();
            auto operator_qinfo_type = curr_operator->quant_info_type();
            auto operator_qinfo      = curr_operator->quant_info();

            // input tensors
            auto operator_inputs = curr_operator->inputs();
            operator_inputs_container.clear();
            if (operator_inputs)
            {
                for (size_t k = 0; k < operator_inputs->size(); k++)
                {
                    auto curr_input = operator_inputs->Get(k);
                    operator_inputs_container.push_back(curr_input->str());
                }
            }

            // output tensors
            auto operator_outputs = curr_operator->outputs();
            operator_outputs_container.clear();
            if (operator_outputs)
            {
                for (size_t k = 0; k < operator_outputs->size(); k++)
                {
                    auto curr_output = operator_outputs->Get(k);
                    operator_outputs_container.push_back(curr_output->str());
                }
            }

            switch (attribute_type)
            {
                case Attribute_NONE:
                    typed_attribute = new TosaNoneAttribute();
                    break;
#define DEF_ATTRIBUTE(NAME, ...)                                                                                       \
    case Attribute_##NAME##Attribute:                                                                                  \
        typed_attribute = new Tosa##NAME##Attribute(attribute);                                                        \
        break;
#include "attribute.def"
#undef DEF_ATTRIBUTE
                default:
                    printf("TosaSerializationHandler::Deserialize(): Attribute %s not implemented yet\n",
                           EnumNamesAttribute()[attribute_type]);
                    return TOSA_INTERNAL_ERROR;
            }

            switch (operator_qinfo_type)
            {
                case QuantInfo_NONE:
                    typed_qinfo = new TosaNoneQuantInfo();
                    break;
#define DEF_QUANTIZATION_INFO(NAME, ...)                                                                               \
    case QuantInfo_##NAME##QuantInfo:                                                                                  \
        typed_qinfo = new Tosa##NAME##QuantInfo(operator_qinfo);                                                       \
        break;

#include "quant_info.def"
#undef DEF_QUANTIZATION_INFO
                default:
                    printf("TosaSerializationHandler::Deserialize(): QuantInfo %s not implemented yet\n",
                           EnumNamesQuantInfo()[operator_qinfo_type]);
                    return TOSA_INTERNAL_ERROR;
            }

            new_operator =
                new TosaSerializationOperator(operator_op, attribute_type, typed_attribute, operator_qinfo_type,
                                              typed_qinfo, operator_inputs_container, operator_outputs_container);
            if (new_operator)
            {
                block_operators_container.push_back(new_operator);
            }
            else
            {
                return TOSA_MEMORY_ERROR;
            }

            if (typed_attribute)
                delete typed_attribute;
            if (typed_qinfo)
                delete typed_qinfo;
        }

        auto fb_tosa_tensors = curr_block->tensors();
        block_tensors_container.clear();
        for (size_t j = 0; j < fb_tosa_tensors->size(); j++)
        {
            auto curr_tensor = fb_tosa_tensors->Get(j);

            auto tensor_name  = curr_tensor->name();
            auto tensor_shape = curr_tensor->shape();
            auto tensor_type  = curr_tensor->type();
            auto tensor_data  = curr_tensor->data();

            new_tensor = new TosaSerializationTensor(tensor_name, tensor_shape, tensor_type, tensor_data);
            if (new_tensor)
            {
                block_tensors_container.push_back(new_tensor);
            }
            else
            {
                return TOSA_MEMORY_ERROR;
            }
        }

        auto block_inputs  = curr_block->inputs();
        auto block_outputs = curr_block->outputs();

        block_inputs_container.clear();
        block_outputs_container.clear();

        for (size_t j = 0; j < block_inputs->size(); j++)
        {
            auto curr_block_input = block_inputs->Get(j);
            block_inputs_container.push_back(curr_block_input->str());
        }
        for (size_t j = 0; j < block_outputs->size(); j++)
        {
            auto curr_block_output = block_outputs->Get(j);
            block_outputs_container.push_back(curr_block_output->str());
        }

        new_block = new TosaSerializationBasicBlock(block_name, block_operators_container, block_tensors_container,
                                                    block_inputs_container, block_outputs_container);
        if (new_block)
        {
            this->GetBlocks().push_back(new_block);
        }
        else
        {
            return TOSA_MEMORY_ERROR;
        }
    }

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::Serialize()
{
    std::vector<flatbuffers::Offset<TosaBasicBlock>> fboffset_blocks;

    std::vector<flatbuffers::Offset<TosaOperator>> fboffset_block_operators;
    std::vector<flatbuffers::Offset<TosaTensor>> fboffset_block_tensors;
    std::vector<flatbuffers::Offset<flatbuffers::String>> fboffset_block_inputs;
    std::vector<flatbuffers::Offset<flatbuffers::String>> fboffset_block_outputs;

    std::vector<flatbuffers::Offset<flatbuffers::String>> fboffset_operator_inputs;
    std::vector<flatbuffers::Offset<flatbuffers::String>> fboffset_operator_outputs;

    // translate TosaFlatbufferOperator to flatbuffers::Offset<TosaOperator>
    for (auto block : GetBlocks())
    {
        fboffset_block_operators.clear();
        fboffset_block_tensors.clear();
        fboffset_block_inputs.clear();
        fboffset_block_outputs.clear();

        auto block_name = _builder.CreateString(block->GetName().c_str());

        for (auto tensor_str : block->GetInputs())
        {
            auto tensor_name = _builder.CreateString(tensor_str.c_str());
            fboffset_block_inputs.push_back(tensor_name);
        }

        for (auto tensor_str : block->GetOutputs())
        {
            auto tensor_name = _builder.CreateString(tensor_str.c_str());
            fboffset_block_outputs.push_back(tensor_name);
        }

        auto fb_block_inputs  = _builder.CreateVector(fboffset_block_inputs);
        auto fb_block_outputs = _builder.CreateVector(fboffset_block_outputs);

        for (auto op : block->GetOperators())
        {
            fboffset_operator_inputs.clear();
            fboffset_operator_outputs.clear();

            auto operator_op    = op->GetOp();
            auto attribute_type = op->GetAttributeType();

            for (auto tensor_str : op->GetInputTensorNames())
            {
                auto tensor_name = _builder.CreateString(tensor_str.c_str());
                fboffset_operator_inputs.push_back(tensor_name);
            }

            for (auto tensor_str : op->GetOutputTensorNames())
            {
                auto tensor_name = _builder.CreateString(tensor_str.c_str());
                fboffset_operator_outputs.push_back(tensor_name);
            }

            auto fb_operator_inputs  = _builder.CreateVector(fboffset_operator_inputs);
            auto fb_operator_outputs = _builder.CreateVector(fboffset_operator_outputs);

            flatbuffers::Offset<void> fb_attribute;
            switch (attribute_type)
            {
                case Attribute_NONE:
                    fb_attribute = 0;
                    break;

#define DEF_ARGS_S_STR(NAME, V) , _builder.CreateString(reinterpret_cast<Tosa##NAME*>(op->GetAttribute())->V().c_str())
#define DEF_ARGS_S_DEFAULT(NAME, V) , reinterpret_cast<Tosa##NAME*>(op->GetAttribute())->V()

#define DEF_ARGS_S_int32_t(NAME, V) DEF_ARGS_S_DEFAULT(NAME, V)
#define DEF_ARGS_S_float(NAME, V) DEF_ARGS_S_DEFAULT(NAME, V)
#define DEF_ARGS_S_bool(NAME, V) DEF_ARGS_S_DEFAULT(NAME, V)
#define DEF_ARGS_S_ResizeMode(NAME, V) DEF_ARGS_S_DEFAULT(NAME, V)
#define DEF_ARGS_S_string(NAME, V) DEF_ARGS_S_STR(NAME, V)

#define DEF_ARGS_S(NAME, T, V) DEF_ARGS_S_##T(NAME, V)
#define DEF_ARGS_V(NAME, T, V) , _builder.CreateVector<T>(reinterpret_cast<Tosa##NAME*>(op->GetAttribute())->V())

#define DEF_ARGS_1(NAME, T0, F0, V0) DEF_ARGS_##F0(NAME, T0, V0)
#define DEF_ARGS_2(NAME, T0, F0, V0, T1, F1, V1) DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1)
#define DEF_ARGS_3(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2)                                                           \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2)
#define DEF_ARGS_4(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3)                                               \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)
#define DEF_ARGS_5(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4)                                   \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4)
#define DEF_ARGS_6(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5)                       \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5)
#define DEF_ARGS_7(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5, T6, F6, V6)           \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5) DEF_ARGS_##F6(NAME, T6, V6)
#define DEF_ATTRIBUTE(NAME, NUM_ARGS, ...)                                                                             \
    case Attribute_##NAME##Attribute:                                                                                  \
        fb_attribute = Create##NAME##Attribute(_builder DEF_ARGS_##NUM_ARGS(NAME##Attribute, __VA_ARGS__)).Union();    \
        break;

#include "attribute.def"
#undef DEF_ATTRIBUTE
#undef DEF_ARGS_1
#undef DEF_ARGS_2
#undef DEF_ARGS_3
#undef DEF_ARGS_4
#undef DEF_ARGS_5
#undef DEF_ARGS_6
#undef DEF_ARGS_7
#undef DEF_ARGS_S
#undef DEF_ARGS_V
#undef DEF_ARGS_S_int32_t
#undef DEF_ARGS_S_float
#undef DEF_ARGS_S_bool
#undef DEF_ARGS_S_ResizeMode
#undef DEF_ARGS_S_string
#undef DEF_ARGS_S_STR
#undef DEF_ARGS_S_DEFAULT
                default:
                    printf("TosaSerializationHandler::Serialize(): Attribute %s not implemented yet\n",
                           EnumNamesAttribute()[attribute_type]);
                    return TOSA_INTERNAL_ERROR;
            }

            auto qinfo_type = op->GetQInfoType();
            flatbuffers::Offset<void> fb_operator_qinfo;
            switch (qinfo_type)
            {
                case QuantInfo_NONE:
                    fb_operator_qinfo = 0;
                    break;
#define DEF_ARGS_S(NAME, T, V) , reinterpret_cast<Tosa##NAME*>(op->GetQInfo())->V()
#define DEF_ARGS_V(NAME, T, V) , _builder.CreateVector<T>(reinterpret_cast<Tosa##NAME*>(op->GetQInfo())->V())

#define DEF_ARGS_1(NAME, T0, F0, V0) DEF_ARGS_##F0(NAME, T0, V0)
#define DEF_ARGS_2(NAME, T0, F0, V0, T1, F1, V1) DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1)
#define DEF_ARGS_3(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2)                                                           \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2)
#define DEF_ARGS_4(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3)                                               \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)
#define DEF_ARGS_5(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4)                                   \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4)
#define DEF_ARGS_6(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5)                       \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5)
#define DEF_ARGS_7(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5, T6, F6, V6)           \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5) DEF_ARGS_##F6(NAME, T6, V6)
#define DEF_ARGS_8(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5, T6, F6, V6, T7, F7,   \
                   V7)                                                                                                 \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5) DEF_ARGS_##F6(NAME, T6, V6)                            \
            DEF_ARGS_##F7(NAME, T7, V7)
#define DEF_ARGS_9(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5, T6, F6, V6, T7, F7,   \
                   V7, T8, F8, V8)                                                                                     \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5) DEF_ARGS_##F6(NAME, T6, V6)                            \
            DEF_ARGS_##F7(NAME, T7, V7) DEF_ARGS_##F8(NAME, T8, V8)
#define DEF_ARGS_10(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5, T6, F6, V6, T7, F7,  \
                    V7, T8, F8, V8, T9, F9, V9)                                                                        \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5) DEF_ARGS_##F6(NAME, T6, V6)                            \
            DEF_ARGS_##F7(NAME, T7, V7) DEF_ARGS_##F8(NAME, T8, V8) DEF_ARGS_##F9(NAME, T9, V9)
#define DEF_QUANTIZATION_INFO(NAME, NUM_ARGS, ...)                                                                     \
    case QuantInfo_##NAME##QuantInfo:                                                                                  \
        fb_operator_qinfo =                                                                                            \
            Create##NAME##QuantInfo(_builder DEF_ARGS_##NUM_ARGS(NAME##QuantInfo, __VA_ARGS__)).Union();               \
        break;

#include "quant_info.def"
#undef DEF_QUANTIZATION_INFO
#undef DEF_ARGS_1
#undef DEF_ARGS_2
#undef DEF_ARGS_3
#undef DEF_ARGS_4
#undef DEF_ARGS_5
#undef DEF_ARGS_6
#undef DEF_ARGS_7
#undef DEF_ARGS_8
#undef DEF_ARGS_9
#undef DEF_ARGS_10
#undef DEF_ARGS_S
#undef DEF_ARGS_V
                default:
                    printf("TosaSerializationHandler::Serialize(): Attribute %s not implemented yet\n",
                           EnumNamesAttribute()[attribute_type]);
                    return TOSA_INTERNAL_ERROR;
            }

            auto fboffset_operator =
                CreateTosaOperator(_builder, operator_op, attribute_type, fb_attribute, fb_operator_inputs,
                                   fb_operator_outputs, qinfo_type, fb_operator_qinfo);
            fboffset_block_operators.push_back(fboffset_operator);
        }

        auto fb_block_operators = _builder.CreateVector(fboffset_block_operators);

        for (auto tensor : block->GetTensors())
        {

            auto tensor_name  = _builder.CreateString(tensor->GetName().c_str());
            auto tensor_shape = _builder.CreateVector(tensor->GetShape());
            auto tensor_dtype = tensor->GetDtype();
            auto tensor_data  = _builder.CreateVector(tensor->GetData());

            auto fboffset_tensor = CreateTosaTensor(_builder, tensor_name, tensor_shape, tensor_dtype, tensor_data);
            fboffset_block_tensors.push_back(fboffset_tensor);
        }

        auto fb_block_tensors = _builder.CreateVector(fboffset_block_tensors);

        auto fboffset_block = CreateTosaBasicBlock(_builder, block_name, fb_block_operators, fb_block_tensors,
                                                   fb_block_inputs, fb_block_outputs);
        fboffset_blocks.push_back(fboffset_block);
    }

    auto fb_blocks = _builder.CreateVector(fboffset_blocks);

    auto fb_version =
        CreateVersion(_builder, TOSA_VERSION_MAJOR, TOSA_VERSION_MINOR, TOSA_VERSION_PATCH, TOSA_VERSION_DRAFT);

    auto fb_graph = CreateTosaGraph(_builder, fb_version, fb_blocks);
    _builder.Finish(fb_graph, TosaGraphIdentifier());

    return TOSA_OK;
}

void zero_pad(std::vector<uint8_t>& buf)
{
    while ((buf.size() % TENSOR_BUFFER_FORCE_ALIGNMENT) != 0)
    {
        buf.push_back(0);
    }
}

tosa_err_t TosaSerializationHandler::ConvertF32toU8(const std::vector<float>& in, std::vector<uint8_t>& out)
{
    out.clear();
    for (auto val : in)
    {
        uint32_t* val_u32 = reinterpret_cast<uint32_t*>(&val);
        out.push_back(*val_u32 & 0xFF);
        out.push_back((*val_u32 >> 8) & 0xFF);
        out.push_back((*val_u32 >> 16) & 0xFF);
        out.push_back((*val_u32 >> 24) & 0xFF);
    }
    zero_pad(out);
    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::ConvertI48toU8(const std::vector<int64_t>& in, std::vector<uint8_t>& out)
{
    out.clear();
    for (auto val : in)
    {
        uint64_t* val_u64 = reinterpret_cast<uint64_t*>(&val);
        out.push_back(*val_u64 & 0xFF);
        out.push_back((*val_u64 >> 8) & 0xFF);
        out.push_back((*val_u64 >> 16) & 0xFF);
        out.push_back((*val_u64 >> 24) & 0xFF);
        out.push_back((*val_u64 >> 32) & 0xFF);
        out.push_back((*val_u64 >> 40) & 0xFF);
    }
    zero_pad(out);
    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::ConvertI32toU8(const std::vector<int32_t>& in, std::vector<uint8_t>& out)
{
    out.clear();
    for (auto val : in)
    {
        uint32_t* val_u32 = reinterpret_cast<uint32_t*>(&val);
        out.push_back(*val_u32 & 0xFF);
        out.push_back((*val_u32 >> 8) & 0xFF);
        out.push_back((*val_u32 >> 16) & 0xFF);
        out.push_back((*val_u32 >> 24) & 0xFF);
    }
    zero_pad(out);
    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::ConvertI16toU8(const std::vector<int16_t>& in, std::vector<uint8_t>& out)
{
    out.clear();
    for (auto val : in)
    {
        uint16_t* val_u16 = reinterpret_cast<uint16_t*>(&val);
        out.push_back(*val_u16 & 0xFF);
        out.push_back((*val_u16 >> 8) & 0xFF);
    }
    zero_pad(out);
    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::ConvertI8toU8(const std::vector<int8_t>& in, std::vector<uint8_t>& out)
{
    out.clear();
    for (auto val : in)
    {
        uint8_t* val_u8 = reinterpret_cast<uint8_t*>(&val);
        out.push_back(*val_u8);
    }
    zero_pad(out);
    return TOSA_OK;
}

// Two int4 values are packed into one byte out.
// For given input value val_0 = in[2*i], and val_1 = in[2*i+1],
// they'll be packed as out[3:0] = val_0, and out[7:4] = val_1
tosa_err_t TosaSerializationHandler::ConvertI4toU8(const std::vector<int8_t>& in, std::vector<uint8_t>& out)
{
    out.clear();
    uint32_t in_size  = in.size();
    uint32_t out_size = (in_size % 2 == 0) ? (in_size / 2) : ((in_size + 1) / 2);
    for (int i = 0; i < out_size; i++)
    {
        int8_t val_0 = in[2 * i];
        int8_t val_1 = 0;
        if (2 * i + 1 < in_size)
        {
            val_1 = in[2 * i + 1];
        }
        // In TOSA spec, int4 ranges [-7, 7]
        if (val_0 < -7 || val_0 > 7 || val_1 < -7 || val_1 > 7)
        {
            printf("TosaSerializationHandler::ConvertI4toU8(): element in input array (%d or %d) exceeds int4 range.\n",
                   val_0, val_1);
            return TOSA_USER_ERROR;
        }
        int8_t val_packed = (val_0 & 0xF) | ((val_1 & 0xF) << 4);
        uint8_t val_u8    = static_cast<uint8_t>(val_packed);
        out.push_back(val_u8);
    }
    zero_pad(out);
    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::ConvertBooltoU8(const std::vector<bool>& in, std::vector<uint8_t>& out)
{
    out.clear();
    for (auto val : in)
    {
        uint8_t* val_u8 = reinterpret_cast<uint8_t*>(&val);
        out.push_back(*val_u8);
    }
    zero_pad(out);
    return TOSA_OK;
}

tosa_err_t
    TosaSerializationHandler::ConvertU8toF32(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<float>& out)
{
    out.clear();
    if (in.size() < out_size * sizeof(float))
    {
        printf("TosaSerializationHandler::ConvertU8toF32(): uint8 buffer size %ld must >= target size %ld\n", in.size(),
               out_size * sizeof(float));
        return TOSA_USER_ERROR;
    }
    for (int i = 0; i < out_size; i++)
    {
        uint32_t byte0   = in[i * sizeof(float)];
        uint32_t byte1   = in[i * sizeof(float) + 1];
        uint32_t byte2   = in[i * sizeof(float) + 2];
        uint32_t byte3   = in[i * sizeof(float) + 3];
        uint32_t val_u32 = byte0 + (byte1 << 8) + (byte2 << 16) + (byte3 << 24);
        float* val_fp32  = reinterpret_cast<float*>(&val_u32);
        out.push_back(*val_fp32);
    }
    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::ConvertU8toI48(const std::vector<uint8_t>& in,
                                                    uint32_t out_size,
                                                    std::vector<int64_t>& out)
{
    out.clear();
    if (in.size() < out_size * 6 /* sizeof(int48) */)
    {
        printf("TosaSerializationHandler::ConvertU8toI48(): uint8 buffer size %ld must >= target size %d\n", in.size(),
               out_size * 6);
        return TOSA_USER_ERROR;
    }
    for (int i = 0; i < out_size; i++)
    {
        uint64_t byte0   = in[i * 6];
        uint64_t byte1   = in[i * 6 + 1];
        uint64_t byte2   = in[i * 6 + 2];
        uint64_t byte3   = in[i * 6 + 3];
        uint64_t byte4   = in[i * 6 + 4];
        uint64_t byte5   = in[i * 6 + 5];
        bool sign        = ((byte5 >> 7) & 1) == 1 ? true : false;
        uint64_t val_u64 = byte0 + (byte1 << 8) + (byte2 << 16) + (byte3 << 24) + (byte4 << 32) + (byte5 << 40);
        if (sign)
        {
            uint64_t sext_mask = (0xFFFFUL << 48);
            val_u64 |= sext_mask;
        }
        int64_t* val_i64 = reinterpret_cast<int64_t*>(&val_u64);
        out.push_back(*val_i64);
    }
    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::ConvertU8toI32(const std::vector<uint8_t>& in,
                                                    uint32_t out_size,
                                                    std::vector<int32_t>& out)
{
    out.clear();
    if (in.size() < out_size * sizeof(int32_t))
    {
        printf("TosaSerializationHandler::ConvertU8toI32(): uint8 buffer size %ld must >= target size %ld\n", in.size(),
               out_size * sizeof(int32_t));
        return TOSA_USER_ERROR;
    }
    for (int i = 0; i < out_size; i++)
    {
        uint32_t byte0   = in[i * sizeof(int32_t)];
        uint32_t byte1   = in[i * sizeof(int32_t) + 1];
        uint32_t byte2   = in[i * sizeof(int32_t) + 2];
        uint32_t byte3   = in[i * sizeof(int32_t) + 3];
        uint32_t val_u32 = byte0 + (byte1 << 8) + (byte2 << 16) + (byte3 << 24);
        int32_t* val_i32 = reinterpret_cast<int32_t*>(&val_u32);
        out.push_back(*val_i32);
    }
    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::ConvertU8toI16(const std::vector<uint8_t>& in,
                                                    uint32_t out_size,
                                                    std::vector<int16_t>& out)
{
    out.clear();
    if (in.size() < out_size * sizeof(int16_t))
    {
        printf("TosaSerializationHandler::ConvertU8toI16(): uint8 buffer size %ld must >= target size %ld\n", in.size(),
               out_size * sizeof(int16_t));
        return TOSA_USER_ERROR;
    }
    for (int i = 0; i < out_size; i++)
    {
        uint16_t byte0   = in[i * sizeof(int16_t)];
        uint16_t byte1   = in[i * sizeof(int16_t) + 1];
        uint16_t val_u16 = byte0 + (byte1 << 8);
        int16_t* val_i16 = reinterpret_cast<int16_t*>(&val_u16);
        out.push_back(*val_i16);
    }
    return TOSA_OK;
}

tosa_err_t
    TosaSerializationHandler::ConvertU8toI8(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<int8_t>& out)
{
    out.clear();
    if (in.size() < out_size * sizeof(int8_t))
    {
        printf("TosaSerializationHandler::ConvertU8toI8(): uint8 buffer size %ld must >= target size %ld\n", in.size(),
               out_size * sizeof(int8_t));
        return TOSA_USER_ERROR;
    }
    for (int i = 0; i < out_size; i++)
    {
        uint8_t val_u8 = in[i];
        int8_t* val_i8 = reinterpret_cast<int8_t*>(&val_u8);
        out.push_back(*val_i8);
    }
    return TOSA_OK;
}

tosa_err_t
    TosaSerializationHandler::ConvertU8toI4(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<int8_t>& out)
{
    out.clear();
    if (out_size > in.size() * 2)
    {
        printf("TosaSerializationHandler::ConvertU8toI4(): output size %u must <= uint8 buffer size %ld x 2.\n",
               out_size, in.size());
        return TOSA_USER_ERROR;
    }
    for (int i = 0; i < in.size(); i++)
    {
        uint8_t val_u8        = in[i];
        uint8_t val_0_u4      = val_u8 & 0xF;
        uint8_t val_1_u4      = val_u8 >> 4;
        uint8_t val_0_u8_sext = (val_0_u4 & 0x08) ? (val_0_u4 | 0xF0) : val_0_u4;
        uint8_t val_1_u8_sext = (val_1_u4 & 0x08) ? (val_1_u4 | 0xF0) : val_1_u4;
        int8_t val_0          = static_cast<int8_t>(val_0_u8_sext);
        int8_t val_1          = static_cast<int8_t>(val_1_u8_sext);
        // In TOSA spec, int4 ranges [-7, 7]
        if (val_0 < -7 || val_0 > 7 || val_1 < -7 || val_1 > 7)
        {
            printf(
                "TosaSerializationHandler::ConvertU8toI4(): element in output array (%d or %d) exceeds int4 range.\n",
                val_0, val_1);
            return TOSA_USER_ERROR;
        }
        out.push_back(val_0);
        if (2 * i + 1 < out_size)
            out.push_back(val_1);
    }
    return TOSA_OK;
}

tosa_err_t
    TosaSerializationHandler::ConvertU8toBool(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<bool>& out)
{
    out.clear();
    if (in.size() < out_size * sizeof(bool))
    {
        printf("TosaSerializationHandler::ConvertU8toBool(): uint8 buffer size %ld must >= target size %ld\n",
               in.size(), out_size * sizeof(bool));
        return TOSA_USER_ERROR;
    }
    for (int i = 0; i < out_size; i++)
    {
        uint8_t val_u8 = in[i];
        bool* val_bool = reinterpret_cast<bool*>(&val_u8);
        out.push_back(*val_bool);
    }
    return TOSA_OK;
}
