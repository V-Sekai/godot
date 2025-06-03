
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

#ifndef _TOSA_SERIALIZATION_ATTRIBUTE_H
#define _TOSA_SERIALIZATION_ATTRIBUTE_H
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "tosa_generated.h"

using std::string;

namespace tosa
{

class TosaAttributeBase
{
public:
    virtual ~TosaAttributeBase()
    {}
};

class TosaNoneAttribute : public TosaAttributeBase
{
public:
    TosaNoneAttribute()
    {}
    TosaNoneAttribute(TosaNoneAttribute* p)
    {}
};

#define DEF_ARGS_VER0_S_STR(V) _##V = p->V()->str();
#define DEF_ARGS_VER0_S_DEFAULT(V) _##V = p->V();

#define DEF_ARGS_VER0_S_int32_t(V) DEF_ARGS_VER0_S_DEFAULT(V)
#define DEF_ARGS_VER0_S_float(V) DEF_ARGS_VER0_S_DEFAULT(V)
#define DEF_ARGS_VER0_S_bool(V) DEF_ARGS_VER0_S_DEFAULT(V)
#define DEF_ARGS_VER0_S_ResizeMode(V) DEF_ARGS_VER0_S_DEFAULT(V)
#define DEF_ARGS_VER0_S_string(V) DEF_ARGS_VER0_S_STR(V)

#define DEF_ARGS_VER0_S(T, V) DEF_ARGS_VER0_S_##T(V)
#define DEF_ARGS_VER0_V(T, V) _##V = std::vector<T>(p->V()->begin(), p->V()->end());

#define DEF_ARGS_VER1_S(T, V) const T& V
#define DEF_ARGS_VER1_V(T, V) const std::vector<T>& V
#define DEF_ARGS_VER2_S(T, V) _##V = V;
#define DEF_ARGS_VER2_V(T, V) _##V = V;
#define DEF_ARGS_VER3_S(T, V)                                                                                          \
    T V() const                                                                                                        \
    {                                                                                                                  \
        return _##V;                                                                                                   \
    }
#define DEF_ARGS_VER3_V(T, V)                                                                                          \
    std::vector<T> V() const                                                                                           \
    {                                                                                                                  \
        return _##V;                                                                                                   \
    }
#define DEF_ARGS_VER4_S(T, V) T _##V;
#define DEF_ARGS_VER4_V(T, V) std::vector<T> _##V;

// another level of preprocessor indirection to handle ", " as function's input argument
#define DEF_ARGS_VER1_TRUE(T, F, V) DEF_ARGS_VER1_##F(T, V)
#define DEF_ARGS_VER1_FALSE(T, F, V) , DEF_ARGS_VER1_##F(T, V)

#define DEF_ARGS_VER0(FIRST, T, F, V) DEF_ARGS_VER0_##F(T, V)
#define DEF_ARGS_VER1(FIRST, T, F, V) DEF_ARGS_VER1_##FIRST(T, F, V)
#define DEF_ARGS_VER2(FIRST, T, F, V) DEF_ARGS_VER2_##F(T, V)
#define DEF_ARGS_VER3(FIRST, T, F, V) DEF_ARGS_VER3_##F(T, V)
#define DEF_ARGS_VER4(FIRST, T, F, V) DEF_ARGS_VER4_##F(T, V)

#define DEF_ARGS_0(VER, ...)
#define DEF_ARGS_1(VER, T0, F0, V0) DEF_ARGS_##VER(TRUE, T0, F0, V0)
#define DEF_ARGS_2(VER, T0, F0, V0, T1, F1, V1) DEF_ARGS_##VER(TRUE, T0, F0, V0) DEF_ARGS_##VER(FALSE, T1, F1, V1)
#define DEF_ARGS_3(VER, T0, F0, V0, T1, F1, V1, T2, F2, V2)                                                            \
    DEF_ARGS_##VER(TRUE, T0, F0, V0) DEF_ARGS_##VER(FALSE, T1, F1, V1) DEF_ARGS_##VER(FALSE, T2, F2, V2)
#define DEF_ARGS_4(VER, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3)                                                \
    DEF_ARGS_##VER(TRUE, T0, F0, V0) DEF_ARGS_##VER(FALSE, T1, F1, V1) DEF_ARGS_##VER(FALSE, T2, F2, V2)               \
        DEF_ARGS_##VER(FALSE, T3, F3, V3)
#define DEF_ARGS_5(VER, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4)                                    \
    DEF_ARGS_##VER(TRUE, T0, F0, V0) DEF_ARGS_##VER(FALSE, T1, F1, V1) DEF_ARGS_##VER(FALSE, T2, F2, V2)               \
        DEF_ARGS_##VER(FALSE, T3, F3, V3) DEF_ARGS_##VER(FALSE, T4, F4, V4)

#define DEF_ARGS_6(VER, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5)                        \
    DEF_ARGS_##VER(TRUE, T0, F0, V0) DEF_ARGS_##VER(FALSE, T1, F1, V1) DEF_ARGS_##VER(FALSE, T2, F2, V2)               \
        DEF_ARGS_##VER(FALSE, T3, F3, V3) DEF_ARGS_##VER(FALSE, T4, F4, V4) DEF_ARGS_##VER(FALSE, T5, F5, V5)

#define DEF_ARGS_7(VER, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5, T6, F6, V6)            \
    DEF_ARGS_##VER(TRUE, T0, F0, V0) DEF_ARGS_##VER(FALSE, T1, F1, V1) DEF_ARGS_##VER(FALSE, T2, F2, V2)               \
        DEF_ARGS_##VER(FALSE, T3, F3, V3) DEF_ARGS_##VER(FALSE, T4, F4, V4) DEF_ARGS_##VER(FALSE, T5, F5, V5)          \
            DEF_ARGS_##VER(FALSE, T6, F6, V6)

#define DEF_VER0_VAR_DECL_PTR(NAME) const NAME* p = static_cast<const NAME*>(options);
#define DEF_VER0_VAR_0(NAME)
#define DEF_VER0_VAR_1(NAME) DEF_VER0_VAR_DECL_PTR(NAME)
#define DEF_VER0_VAR_2(NAME) DEF_VER0_VAR_DECL_PTR(NAME)
#define DEF_VER0_VAR_3(NAME) DEF_VER0_VAR_DECL_PTR(NAME)
#define DEF_VER0_VAR_4(NAME) DEF_VER0_VAR_DECL_PTR(NAME)
#define DEF_VER0_VAR_5(NAME) DEF_VER0_VAR_DECL_PTR(NAME)
#define DEF_VER0_VAR_6(NAME) DEF_VER0_VAR_DECL_PTR(NAME)
#define DEF_VER0_VAR_7(NAME) DEF_VER0_VAR_DECL_PTR(NAME)

#define DEF_ATTRIBUTE(NAME, NUM_ARGS, ...)                                                                             \
    class Tosa##NAME##Attribute : public TosaAttributeBase                                                             \
    {                                                                                                                  \
    public:                                                                                                            \
        Tosa##NAME##Attribute(const TosaAttributeBase* options)                                                        \
        {                                                                                                              \
            const Tosa##NAME##Attribute* p = static_cast<const Tosa##NAME##Attribute*>(options);                       \
            *this                          = *p;                                                                       \
        }                                                                                                              \
        Tosa##NAME##Attribute(const Tosa##NAME##Attribute* p)                                                          \
        {                                                                                                              \
            *this = *p;                                                                                                \
        }                                                                                                              \
        Tosa##NAME##Attribute(const void* options){ DEF_VER0_VAR_##NUM_ARGS(NAME##Attribute)                           \
                                                        DEF_ARGS_##NUM_ARGS(VER0, __VA_ARGS__) } Tosa##NAME            \
            ##Attribute(DEF_ARGS_##NUM_ARGS(VER1, __VA_ARGS__))                                                        \
        {                                                                                                              \
            DEF_ARGS_##NUM_ARGS(VER2, __VA_ARGS__)                                                                     \
        }                                                                                                              \
        virtual ~Tosa##NAME##Attribute()                                                                               \
        {}                                                                                                             \
        DEF_ARGS_##NUM_ARGS(VER3, __VA_ARGS__) private : DEF_ARGS_##NUM_ARGS(VER4, __VA_ARGS__)                        \
    };

#include "attribute.def"
#undef DEF_ATTRIBUTE
#undef DEF_ARGS_0
#undef DEF_ARGS_1
#undef DEF_ARGS_2
#undef DEF_ARGS_3
#undef DEF_ARGS_4
#undef DEF_ARGS_5
#undef DEF_ARGS_6
#undef DEF_ARGS_7
#undef DEF_ARGS_VER0
#undef DEF_ARGS_VER1
#undef DEF_ARGS_VER2
#undef DEF_ARGS_VER3
#undef DEF_ARGS_VER4
#undef DEF_ARGS_VER0_S_int32_t
#undef DEF_ARGS_VER0_S_float
#undef DEF_ARGS_VER0_S_bool
#undef DEF_ARGS_VER0_S_ResizeMode
#undef DEF_ARGS_VER0_S_string
#undef DEF_ARGS_VER0_S_STR
#undef DEF_ARGS_VER0_S_DEFAULT
#undef DEF_ARGS_VER1_TRUE
#undef DEF_ARGS_VER1_FALSE
#undef DEF_ARGS_VER0_S
#undef DEF_ARGS_VER0_V
#undef DEF_ARGS_VER1_S
#undef DEF_ARGS_VER1_V
#undef DEF_ARGS_VER2_S
#undef DEF_ARGS_VER2_V
#undef DEF_ARGS_VER3_S
#undef DEF_ARGS_VER3_V
#undef DEF_ARGS_VER4_S
#undef DEF_ARGS_VER4_V
#undef DEF_VER0_VAR_0
#undef DEF_VER0_VAR_1
#undef DEF_VER0_VAR_2
#undef DEF_VER0_VAR_3
#undef DEF_VER0_VAR_4
#undef DEF_VER0_VAR_5
#undef DEF_VER0_VAR_DECL_PTR

}    // namespace tosa

#endif    // _TOSA_SERIALIZATION_ATTRIBUTE_H
