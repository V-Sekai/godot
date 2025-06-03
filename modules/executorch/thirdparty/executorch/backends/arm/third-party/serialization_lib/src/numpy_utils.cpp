
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

#include "numpy_utils.h"

// Magic NUMPY header
static const char NUMPY_HEADER_STR[] = "\x93NUMPY\x1\x0\x76\x0{";
static const int NUMPY_HEADER_SZ     = 128;
// Maximum shape dimensions supported
static const int NUMPY_MAX_DIMS_SUPPORTED = 10;

NumpyUtilities::NPError NumpyUtilities::readFromNpyFile(const char* filename, const uint32_t elems, bool* databuf)
{
    const char dtype_str[] = "'|b1'";
    return readFromNpyFileCommon(filename, dtype_str, 1, elems, databuf, true);
}

NumpyUtilities::NPError NumpyUtilities::readFromNpyFile(const char* filename, const uint32_t elems, int32_t* databuf)
{
    const char dtype_str[] = "'<i4'";
    return readFromNpyFileCommon(filename, dtype_str, sizeof(int32_t), elems, databuf, false);
}

NumpyUtilities::NPError NumpyUtilities::readFromNpyFile(const char* filename, const uint32_t elems, int64_t* databuf)
{
    const char dtype_str[] = "'<i8'";
    return readFromNpyFileCommon(filename, dtype_str, sizeof(int64_t), elems, databuf, false);
}

NumpyUtilities::NPError NumpyUtilities::readFromNpyFile(const char* filename, const uint32_t elems, float* databuf)
{
    const char dtype_str[] = "'<f4'";
    return readFromNpyFileCommon(filename, dtype_str, sizeof(float), elems, databuf, false);
}

NumpyUtilities::NPError NumpyUtilities::readFromNpyFileCommon(const char* filename,
                                                              const char* dtype_str,
                                                              const size_t elementsize,
                                                              const uint32_t elems,
                                                              void* databuf,
                                                              bool bool_translate)
{
    FILE* infile = nullptr;
    NPError rc   = NO_ERROR;

    assert(filename);
    assert(databuf);

    infile = fopen(filename, "rb");
    if (!infile)
    {
        return FILE_NOT_FOUND;
    }

    rc = checkNpyHeader(infile, elems, dtype_str);
    if (rc == NO_ERROR)
    {
        if (bool_translate)
        {
            // Read in the data from numpy byte array to native bool
            // array format
            bool* buf = reinterpret_cast<bool*>(databuf);
            for (uint32_t i = 0; i < elems; i++)
            {
                int val = fgetc(infile);

                if (val == EOF)
                {
                    rc = FILE_IO_ERROR;
                }

                buf[i] = val;
            }
        }
        else
        {
            // Now we are at the beginning of the data
            // Parse based on the datatype and number of dimensions
            if (fread(databuf, elementsize, elems, infile) != elems)
            {
                rc = FILE_IO_ERROR;
            }
        }
    }

    if (infile)
        fclose(infile);

    return rc;
}

NumpyUtilities::NPError NumpyUtilities::checkNpyHeader(FILE* infile, const uint32_t elems, const char* dtype_str)
{
    char buf[NUMPY_HEADER_SZ + 1];
    char* ptr         = nullptr;
    NPError rc        = NO_ERROR;
    bool foundFormat  = false;
    bool foundOrder   = false;
    bool foundShape   = false;
    bool fortranOrder = false;
    std::vector<int> shape;
    uint32_t totalElems = 1;
    char* outer_end     = NULL;

    assert(infile);
    assert(elems > 0);

    if (fread(buf, NUMPY_HEADER_SZ, 1, infile) != 1)
    {
        return HEADER_PARSE_ERROR;
    }

    if (memcmp(buf, NUMPY_HEADER_STR, sizeof(NUMPY_HEADER_STR) - 1))
    {
        return HEADER_PARSE_ERROR;
    }

    ptr = strtok_r(buf + sizeof(NUMPY_HEADER_STR) - 1, ":", &outer_end);

    // Read in the data type, order, and shape
    while (ptr && (!foundFormat || !foundOrder || !foundShape))
    {

        // End of string?
        if (!ptr)
            break;

        // Skip whitespace
        while (isspace(*ptr))
            ptr++;

        // Parse the dictionary field name
        if (!strcmp(ptr, "'descr'"))
        {
            ptr = strtok_r(NULL, ",", &outer_end);
            if (!ptr)
                break;

            while (isspace(*ptr))
                ptr++;

            if (strcmp(ptr, dtype_str))
            {
                return FILE_TYPE_MISMATCH;
            }

            foundFormat = true;
        }
        else if (!strcmp(ptr, "'fortran_order'"))
        {
            ptr = strtok_r(NULL, ",", &outer_end);
            if (!ptr)
                break;

            while (isspace(*ptr))
                ptr++;

            if (!strcmp(ptr, "False"))
            {
                fortranOrder = false;
            }
            else
            {
                return FILE_TYPE_MISMATCH;
            }

            foundOrder = true;
        }
        else if (!strcmp(ptr, "'shape'"))
        {

            ptr = strtok_r(NULL, "(", &outer_end);
            if (!ptr)
                break;
            ptr = strtok_r(NULL, ")", &outer_end);
            if (!ptr)
                break;

            while (isspace(*ptr))
                ptr++;

            // The shape contains N comma-separated integers. Read up to MAX_DIMS.
            char* end = NULL;

            ptr = strtok_r(ptr, ",", &end);
            for (int i = 0; i < NUMPY_MAX_DIMS_SUPPORTED; i++)
            {
                // Out of dimensions
                if (!ptr)
                    break;

                int dim = atoi(ptr);

                // Dimension is 0
                if (dim == 0)
                    break;

                shape.push_back(dim);
                totalElems *= dim;
                ptr = strtok_r(NULL, ",", &end);
            }

            foundShape = true;
        }
        else
        {
            return HEADER_PARSE_ERROR;
        }

        if (!ptr)
            break;

        ptr = strtok_r(NULL, ":", &outer_end);
    }

    if (!foundShape || !foundFormat || !foundOrder)
    {
        return HEADER_PARSE_ERROR;
    }

    // Validate header
    if (fortranOrder)
    {
        return FILE_TYPE_MISMATCH;
    }

    if (totalElems != elems)
    {
        return BUFFER_SIZE_MISMATCH;
    }

    // Go back to the begininng and read until the end of the header dictionary
    rewind(infile);
    int val;

    do
    {
        val = fgetc(infile);
    } while (val != EOF && val != '\n');

    return rc;
}

NumpyUtilities::NPError NumpyUtilities::writeToNpyFile(const char* filename, const uint32_t elems, const bool* databuf)
{
    std::vector<int32_t> shape = { (int32_t)elems };
    return writeToNpyFile(filename, shape, databuf);
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const bool* databuf)
{
    const char dtype_str[] = "'|b1'";
    return writeToNpyFileCommon(filename, dtype_str, 1, shape, databuf, true);    // bools written as size 1
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const uint32_t elems, const int32_t* databuf)
{
    std::vector<int32_t> shape = { (int32_t)elems };
    return writeToNpyFile(filename, shape, databuf);
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const int32_t* databuf)
{
    const char dtype_str[] = "'<i4'";
    return writeToNpyFileCommon(filename, dtype_str, sizeof(int32_t), shape, databuf, false);
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const uint32_t elems, const int64_t* databuf)
{
    std::vector<int32_t> shape = { (int32_t)elems };
    return writeToNpyFile(filename, shape, databuf);
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const int64_t* databuf)
{
    const char dtype_str[] = "'<i8'";
    return writeToNpyFileCommon(filename, dtype_str, sizeof(int64_t), shape, databuf, false);
}

NumpyUtilities::NPError NumpyUtilities::writeToNpyFile(const char* filename, const uint32_t elems, const float* databuf)
{
    std::vector<int32_t> shape = { (int32_t)elems };
    return writeToNpyFile(filename, shape, databuf);
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const float* databuf)
{
    const char dtype_str[] = "'<f4'";
    return writeToNpyFileCommon(filename, dtype_str, sizeof(float), shape, databuf, false);
}

NumpyUtilities::NPError NumpyUtilities::writeToNpyFileCommon(const char* filename,
                                                             const char* dtype_str,
                                                             const size_t elementsize,
                                                             const std::vector<int32_t>& shape,
                                                             const void* databuf,
                                                             bool bool_translate)
{
    FILE* outfile       = nullptr;
    NPError rc          = NO_ERROR;
    uint32_t totalElems = 1;

    assert(filename);
    assert(shape.size() >= 0);
    assert(databuf);

    outfile = fopen(filename, "wb");

    if (!outfile)
    {
        return FILE_NOT_FOUND;
    }

    for (uint32_t i = 0; i < shape.size(); i++)
    {
        totalElems *= shape[i];
    }

    rc = writeNpyHeader(outfile, shape, dtype_str);

    if (rc == NO_ERROR)
    {
        if (bool_translate)
        {
            // Numpy save format stores booleans as a byte array
            // with one byte per boolean.  This somewhat inefficiently
            // remaps from system bool[] to this format.
            const bool* buf = reinterpret_cast<const bool*>(databuf);
            for (uint32_t i = 0; i < totalElems; i++)
            {
                int val = buf[i] ? 1 : 0;
                if (fputc(val, outfile) == EOF)
                {
                    rc = FILE_IO_ERROR;
                }
            }
        }
        else
        {
            if (fwrite(databuf, elementsize, totalElems, outfile) != totalElems)
            {
                rc = FILE_IO_ERROR;
            }
        }
    }

    if (outfile)
        fclose(outfile);

    return rc;
}

NumpyUtilities::NPError
    NumpyUtilities::writeNpyHeader(FILE* outfile, const std::vector<int32_t>& shape, const char* dtype_str)
{
    NPError rc = NO_ERROR;
    uint32_t i;
    char header[NUMPY_HEADER_SZ + 1];
    int headerPos = 0;

    assert(outfile);
    assert(shape.size() >= 0);

    // Space-fill the header and end with a newline to start per numpy spec
    memset(header, 0x20, NUMPY_HEADER_SZ);
    header[NUMPY_HEADER_SZ - 1] = '\n';
    header[NUMPY_HEADER_SZ]     = 0;

    // Write out the hard-coded header.  We only support a 128-byte 1.0 header
    // for now, which should be sufficient for simple tensor types of any
    // reasonable rank.
    memcpy(header, NUMPY_HEADER_STR, sizeof(NUMPY_HEADER_STR) - 1);
    headerPos += sizeof(NUMPY_HEADER_STR) - 1;

    // Output the format dictionary
    // Hard-coded for I32 for now
    headerPos +=
        snprintf(header + headerPos, NUMPY_HEADER_SZ - headerPos, "'descr': %s, 'fortran_order': False, 'shape': (%d,",
                 dtype_str, shape.empty() ? 1 : shape[0]);

    // Remainder of shape array
    for (i = 1; i < shape.size(); i++)
    {
        headerPos += snprintf(header + headerPos, NUMPY_HEADER_SZ - headerPos, " %d,", shape[i]);
    }

    // Close off the dictionary
    headerPos += snprintf(header + headerPos, NUMPY_HEADER_SZ - headerPos, "), }");

    // snprintf leaves a NULL at the end. Replace with a space
    header[headerPos] = 0x20;

    if (fwrite(header, NUMPY_HEADER_SZ, 1, outfile) != 1)
    {
        rc = FILE_IO_ERROR;
    }

    return rc;
}
