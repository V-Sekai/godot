#!/usr/bin/env python3

# Copyright (c) 2021, ARM Limited.
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

""" Simple test script which tests numpy file read/write"""

import argparse
import random
import shlex
import subprocess
from datetime import datetime
from enum import IntEnum, unique
from pathlib import Path
from xunit.xunit import xunit_results, xunit_test


@unique
class TestResult(IntEnum):
    PASS = 0
    COMMAND_ERROR = 1
    MISMATCH = 2
    SKIPPED = 3


def parseArgs():
    baseDir = (Path(__file__).parent / "../..").resolve()
    buildDir = (baseDir / "build").resolve()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--cmd",
        default=str(buildDir / "serialization_npy_test"),
        help="Command to write/read test file",
    )
    parser.add_argument("-s", "--seed", default=1, help="Random number seed")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose", default=False
    )
    parser.add_argument(
        "--xunit-file", default="npy-result.xml", help="xunit result output file"
    )
    args = parser.parse_args()

    # check that required files exist
    if not Path(args.cmd).exists():
        print("command not found at location " + args.cmd)
        parser.print_help()
        exit(1)
    return args


def run_sh_command(full_cmd, verbose=False, capture_output=False):
    """Utility function to run an external command. Optionally return captured
    stdout/stderr"""

    # Quote the command line for printing
    full_cmd_esc = [shlex.quote(x) for x in full_cmd]

    if verbose:
        print("### Running {}".format(" ".join(full_cmd_esc)))

    if capture_output:
        rc = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if rc.returncode != 0:
            print(rc.stdout.decode("utf-8"))
            print(rc.stderr.decode("utf-8"))
            raise Exception(
                "Error running command: {}.\n{}".format(
                    " ".join(full_cmd_esc), rc.stderr.decode("utf-8")
                )
            )
        return (rc.stdout, rc.stderr)
    else:
        rc = subprocess.run(full_cmd)
    if rc.returncode != 0:
        raise Exception("Error running command: {}".format(" ".join(full_cmd_esc)))


def runTest(args, dtype, shape):
    start_time = datetime.now()
    result = TestResult.PASS
    message = ""

    target = Path(f"npytest-{random.randint(0,10000)}.npy")
    shape_str = ",".join(shape)
    # Remove any previous files
    if target.exists():
        target.unlink()

    try:
        cmd = [args.cmd, "-d", dtype, "-f", str(target), "-t", shape_str]
        run_sh_command(cmd, args.verbose)
        target.unlink()

    except Exception as e:
        message = str(e)
        result = TestResult.COMMAND_ERROR
    end_time = datetime.now()
    return result, message, end_time - start_time


def main():
    args = parseArgs()

    suitename = "basic_serialization"
    classname = "npy_test"

    xunit_result = xunit_results()
    xunit_suite = xunit_result.create_suite("basic_serialization")

    max_size = 128
    datatypes = ["int32", "int64", "float", "bool"]
    random.seed(args.seed)

    failed = 0
    count = 0
    for test in datatypes:
        count = count + 1
        shape = []
        for i in range(4):
            shape.append(str(random.randint(1, max_size)))
        (result, message, time_delta) = runTest(args, test, shape)
        xt = xunit_test(str(test), f"{suitename}.{classname}")
        xt.time = str(
            float(time_delta.seconds) + (float(time_delta.microseconds) * 1e-6)
        )
        if result == TestResult.PASS:
            pass
        else:
            xt.failed(message)
            failed = failed + 1
        xunit_suite.tests.append(xt)

    xunit_result.write_results(args.xunit_file)
    print(f"Total tests run: {count} failures: {failed}")


if __name__ == "__main__":
    exit(main())
