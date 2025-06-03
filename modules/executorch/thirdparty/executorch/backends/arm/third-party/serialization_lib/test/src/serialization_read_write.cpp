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

#include <iostream>
#include <tosa_serialization_handler.h>

using namespace tosa;

void usage()
{
    std::cerr << "Usage: <src> <dest>" << std::endl;
    std::cerr << "    <src>: source TOSA serialize filename" << std::endl;
    std::cerr << "    <dest>: destination TOSA serialized filename" << std::endl;
}

int main(int argc, char** argv)
{
    TosaSerializationHandler handler;
    if (argc != 3)
    {
        usage();
        return 1;
    }

    tosa_err_t err = handler.LoadFileTosaFlatbuffer(argv[1]);
    if (err != TOSA_OK)
    {
        std::cout << "error reading file " << argv[1] << " code " << err << std::endl;
        return 1;
    }

    err = handler.SaveFileTosaFlatbuffer(argv[2]);
    if (err != TOSA_OK)
    {
        std::cout << "error writing file " << argv[2] << " code " << err << std::endl;
        return 1;
    }
    return 0;
}
