// Copyright (c) 2021, ARM Limited.
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

#include <getopt.h>
#include <iostream>
#include <random>
#include <sstream>
#include <tosa_serialization_handler.h>

using namespace tosa;

void usage()
{
    std::cout << "Usage: serialization_npy_test -f <filename> -t <shape> -d <datatype> -s <seed>" << std::endl;
}

template <class T>
int test_int_type(std::vector<int32_t> shape, std::default_random_engine& gen, std::string& filename)
{
    size_t total_size = 1;
    std::uniform_int_distribution<T> gen_data(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    for (auto i : shape)
    {
        total_size *= i;
    }

    auto buffer = std::make_unique<T[]>(total_size);
    for (int i = 0; i < total_size; i++)
    {
        buffer[i] = gen_data(gen);
    }

    NumpyUtilities::NPError err = NumpyUtilities::writeToNpyFile(filename.c_str(), shape, buffer.get());
    if (err != NumpyUtilities::NO_ERROR)
    {
        std::cout << "Error writing file, code " << err << std::endl;
        return 1;
    }

    auto read_buffer = std::make_unique<T[]>(total_size);
    err              = NumpyUtilities::readFromNpyFile(filename.c_str(), total_size, read_buffer.get());
    if (err != NumpyUtilities::NO_ERROR)
    {
        std::cout << "Error reading file, code " << err << std::endl;
        return 1;
    }
    if (memcmp(buffer.get(), read_buffer.get(), total_size * sizeof(T)))
    {
        std::cout << "Miscompare" << std::endl;
        return 1;
    }
    return 0;
}

template <class T>
int test_float_type(std::vector<int32_t> shape, std::default_random_engine& gen, std::string& filename)
{
    size_t total_size = 1;
    std::uniform_real_distribution<T> gen_data(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    for (auto i : shape)
    {
        total_size *= i;
    }

    auto buffer = std::make_unique<T[]>(total_size);
    for (int i = 0; i < total_size; i++)
    {
        buffer[i] = gen_data(gen);
    }

    NumpyUtilities::NPError err = NumpyUtilities::writeToNpyFile(filename.c_str(), shape, buffer.get());
    if (err != NumpyUtilities::NO_ERROR)
    {
        std::cout << "Error writing file, code " << err << std::endl;
        return 1;
    }

    auto read_buffer = std::make_unique<T[]>(total_size);
    err              = NumpyUtilities::readFromNpyFile(filename.c_str(), total_size, read_buffer.get());
    if (err != NumpyUtilities::NO_ERROR)
    {
        std::cout << "Error reading file, code " << err << std::endl;
        return 1;
    }
    if (memcmp(buffer.get(), read_buffer.get(), total_size * sizeof(T)))
    {
        std::cout << "Miscompare" << std::endl;
        return 1;
    }
    return 0;
}

int test_bool_type(std::vector<int32_t> shape, std::default_random_engine& gen, std::string& filename)
{
    size_t total_size = 1;
    std::uniform_int_distribution<uint32_t> gen_data(0, 1);

    for (auto i : shape)
    {
        total_size *= i;
    }

    auto buffer = std::make_unique<bool[]>(total_size);
    for (int i = 0; i < total_size; i++)
    {
        buffer[i] = (gen_data(gen)) ? true : false;
    }

    NumpyUtilities::NPError err = NumpyUtilities::writeToNpyFile(filename.c_str(), shape, buffer.get());
    if (err != NumpyUtilities::NO_ERROR)
    {
        std::cout << "Error writing file, code " << err << std::endl;
        return 1;
    }

    auto read_buffer = std::make_unique<bool[]>(total_size);
    err              = NumpyUtilities::readFromNpyFile(filename.c_str(), total_size, read_buffer.get());
    if (err != NumpyUtilities::NO_ERROR)
    {
        std::cout << "Error reading file, code " << err << std::endl;
        return 1;
    }

    if (memcmp(buffer.get(), read_buffer.get(), total_size * sizeof(bool)))
    {
        std::cout << "Miscompare" << std::endl;
        return 1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    size_t total_size = 1;
    int32_t seed      = 1;
    std::string str_type;
    std::string str_shape;
    std::string filename = "npytest.npy";
    std::vector<int32_t> shape;
    bool verbose = false;
    int opt;
    while ((opt = getopt(argc, argv, "d:f:s:t:v")) != -1)
    {
        switch (opt)
        {
            case 'd':
                str_type = optarg;
                break;
            case 'f':
                filename = optarg;
                break;
            case 's':
                seed = strtol(optarg, nullptr, 0);
                break;
            case 't':
                str_shape = optarg;
                break;
            case 'v':
                verbose = true;
                break;
            default:
                std::cerr << "Invalid argument" << std::endl;
                break;
        }
    }
    if (str_shape == "")
    {
        usage();
        return 1;
    }

    // parse shape from argument
    std::stringstream ss(str_shape);
    while (ss.good())
    {
        std::string substr;
        size_t pos;
        std::getline(ss, substr, ',');
        if (substr == "")
            break;
        int val = stoi(substr, &pos, 0);
        assert(val);
        total_size *= val;
        shape.push_back(val);
    }

    std::default_random_engine gen(seed);

    // run with type from argument
    if (str_type == "int32")
    {
        return test_int_type<int32_t>(shape, gen, filename);
    }
    else if (str_type == "int64")
    {
        return test_int_type<int64_t>(shape, gen, filename);
    }
    else if (str_type == "float")
    {
        return test_float_type<float>(shape, gen, filename);
    }
    else if (str_type == "bool")
    {
        return test_bool_type(shape, gen, filename);
    }
    else
    {
        std::cout << "Unknown type " << str_type << std::endl;
        usage();
        return 1;
    }
}
