#!/bin/bash

# Copyright (c) 2020-2021, ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Bail if any of the commands fail
set -e

if test -f "third_party/flatbuffers/flatc";
then
    echo "Found flatc, skip building..."
else
    echo "flatc not found, building now..."
    pushd third_party/flatbuffers/ > /dev/null
        cmake .
        make flatc -j8
    popd > /dev/null
fi

pushd include/ > /dev/null
    ../third_party/flatbuffers/flatc --cpp ../schema/tosa.fbs
popd > /dev/null
pushd python/ > /dev/null
    ../third_party/flatbuffers/flatc --python ../schema/tosa.fbs
popd > /dev/null

