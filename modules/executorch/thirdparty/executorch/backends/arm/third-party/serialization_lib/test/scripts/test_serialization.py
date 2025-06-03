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

""" Simple test script which uses serialization_read_write to copy tosa files. It
uses flatc to convert to json for comparison since the binary files may
differ. """

import argparse
import filecmp
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
        "-t",
        "--testdir",
        dest="test",
        type=str,
        required=True,
        help="Directory of tosa files to verify",
    )
    parser.add_argument(
        "--flatc",
        default=str(buildDir / "third_party/flatbuffers/flatc"),
        help="location of flatc compiler",
    )
    parser.add_argument(
        "-s",
        "--schema",
        default=str(baseDir / "schema/tosa.fbs"),
        help="location of schema file",
    )
    parser.add_argument(
        "-c",
        "--cmd",
        default=str(buildDir / "serialization_read_write"),
        help="Command to read/write test file",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose", default=False
    )
    parser.add_argument(
        "--xunit-file", default="result.xml", help="xunit result output file"
    )
    args = parser.parse_args()

    # check that required files exist
    if not Path(args.flatc).exists():
        print("flatc not found at location " + args.flatc)
        parser.print_help()
        exit(1)
    if not Path(args.cmd).exists():
        print("command not found at location " + args.cmd)
        parser.print_help()
        exit(1)
    if not Path(args.schema).exists():
        print("schema not found at location " + args.schema)
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


def runTest(args, testfile):
    start_time = datetime.now()
    result = TestResult.PASS
    message = ""

    target = Path(f"serialization_script_output-{random.randint(0,10000)}.tosa")
    source_json = Path(testfile.stem + ".json")
    target_json = Path(target.stem + ".json")

    # Remove any previous files
    if target.exists():
        target.unlink()
    if source_json.exists():
        source_json.unlink()
    if target_json.exists():
        target_json.unlink()

    try:
        cmd = [args.cmd, str(testfile), str(target)]
        run_sh_command(cmd, args.verbose)
        # Create result json
        cmd = [args.flatc, "--json", "--raw-binary", args.schema, "--", str(target)]
        run_sh_command(cmd, args.verbose)
        # Create source json
        cmd = [args.flatc, "--json", "--raw-binary", args.schema, "--", str(testfile)]
        run_sh_command(cmd, args.verbose)
        if not filecmp.cmp(str(target_json), str(source_json), False):
            print("Failed to compare files on " + str(testfile))
            result = TestResult.MISMATCH
        # Cleanup generated files
        source_json.unlink()
        target_json.unlink()
        target.unlink()

    except Exception as e:
        message = str(e)
        result = TestResult.COMMAND_ERROR
    end_time = datetime.now()
    return result, message, end_time - start_time


def getTestFiles(dir):
    files = Path(dir).glob("**/*.tosa")
    return files


def main():
    args = parseArgs()
    testfiles = getTestFiles(args.test)

    suitename = "basic_serialization"
    classname = "copy_test"

    xunit_result = xunit_results()
    xunit_suite = xunit_result.create_suite("basic_serialization")

    failed = 0
    count = 0
    for test in testfiles:
        count = count + 1
        (result, message, time_delta) = runTest(args, test)
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
