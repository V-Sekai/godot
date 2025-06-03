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

import xml.etree.ElementTree as ET
from xml.dom import minidom


class xunit_results:
    def __init__(self):
        self.name = "testsuites"
        self.suites = []

    def create_suite(self, name):
        s = xunit_suite(name)
        self.suites.append(s)
        return s

    def write_results(self, filename):
        suites = ET.Element(self.name)
        tree = ET.ElementTree(suites)
        for s in self.suites:
            testsuite = ET.SubElement(
                suites, "testsuite", {"name": s.name, "errors": "0"}
            )
            tests = 0
            failures = 0
            skip = 0
            for t in s.tests:
                test = ET.SubElement(
                    testsuite,
                    "testcase",
                    {"name": t.name, "classname": t.classname, "time": t.time},
                )
                tests += 1
                if t.skip:
                    skip += 1
                    ET.SubElement(test, "skipped", {"type": "Skipped test"})
                if t.fail:
                    failures += 1
                    fail = ET.SubElement(test, "failure", {"type": "Test failed"})
                    fail.text = t.fail
                if t.sysout:
                    sysout = ET.SubElement(test, "system-out")
                    sysout.text = t.sysout
                if t.syserr:
                    syserr = ET.SubElement(test, "system-err")
                    syserr.text = t.syserr
            testsuite.attrib["tests"] = str(tests)
            testsuite.attrib["failures"] = str(failures)
            testsuite.attrib["skip"] = str(skip)
        xmlstr = minidom.parseString(ET.tostring(tree.getroot())).toprettyxml(
            indent="  "
        )
        with open(filename, "w") as f:
            f.write(xmlstr)


class xunit_suite:
    def __init__(self, name):
        self.name = name
        self.tests = []


# classname should be of the form suite.class/subclass/subclass2/... It appears
# you can have an unlimited number of subclasses in this manner


class xunit_test:
    def __init__(self, name, classname=None):
        self.name = name
        if classname:
            self.classname = classname
        else:
            self.classname = name
        self.time = "0.000"
        self.fail = None
        self.skip = False
        self.sysout = None
        self.syserr = None

    def failed(self, text):
        self.fail = text

    def skipped(self):
        self.skip = True


if __name__ == "__main__":
    r = xunit_results()
    s = r.create_suite("selftest")
    for i in range(0, 10):
        t = xunit_test("atest" + str(i), "selftest")
        if i == 3:
            t.failed("Unknown failure foo")
        if i == 7:
            t.skipped()
        s.tests.append(t)
    r.write_results("foo.xml")
