
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

#ifndef _TOSA_NUMPY_UTILS_H
#define _TOSA_NUMPY_UTILS_H

#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

class NumpyUtilities
{
public:
    enum NPError
    {
        NO_ERROR = 0,
        FILE_NOT_FOUND,
        FILE_IO_ERROR,
        FILE_TYPE_MISMATCH,
        HEADER_PARSE_ERROR,
        BUFFER_SIZE_MISMATCH,
    };

    static NPError readFromNpyFile(const char* filename, const uint32_t elems, float* databuf);

    static NPError readFromNpyFile(const char* filename, const uint32_t elems, int32_t* databuf);

    static NPError readFromNpyFile(const char* filename, const uint32_t elems, int64_t* databuf);

    static NPError readFromNpyFile(const char* filename, const uint32_t elems, bool* databuf);

    static NPError writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const bool* databuf);

    static NPError writeToNpyFile(const char* filename, const uint32_t elems, const bool* databuf);

    static NPError writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const int32_t* databuf);

    static NPError writeToNpyFile(const char* filename, const uint32_t elems, const int32_t* databuf);

    static NPError writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const int64_t* databuf);

    static NPError writeToNpyFile(const char* filename, const uint32_t elems, const int64_t* databuf);

    static NPError writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const float* databuf);

    static NPError writeToNpyFile(const char* filename, const uint32_t elems, const float* databuf);

private:
    static NPError writeToNpyFileCommon(const char* filename,
                                        const char* dtype_str,
                                        const size_t elementsize,
                                        const std::vector<int32_t>& shape,
                                        const void* databuf,
                                        bool bool_translate);
    static NPError readFromNpyFileCommon(const char* filename,
                                         const char* dtype_str,
                                         const size_t elementsize,
                                         const uint32_t elems,
                                         void* databuf,
                                         bool bool_translate);
    static NPError checkNpyHeader(FILE* infile, const uint32_t elems, const char* dtype_str);
    static NPError writeNpyHeader(FILE* outfile, const std::vector<int32_t>& shape, const char* dtype_str);
};

#endif    // _TOSA_NUMPY_UTILS_H
