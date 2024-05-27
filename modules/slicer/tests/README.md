# Tests

Mostly as an experiment in implementation, Slicer includes a simple suite of unit tests. Configuration details can be found in the [SCSub file](./SCsub) found in this folder. Currently tests are only available on Unix environments, as they make use of the Godot headless server for initializing the Godot environment.

## Building
The tests can be built with the scons option `slicer_tests`, such as:

```bash
scons slicer_tests=yes
```

`slicer_tests` is also available as an alias if you wish to only target building the test binarry:

```bash
scons slicer_tests=yes slicer_tests
```

To speed up build times during development the `slicer_shared` option can be used to separate the building of the main test binary and the test files themselves. Both test and Slicer logic can be built as dynamic libraries using a command such as:

```bash
scons platform=osx slicer_shared=yes slicer_tests=yes slicer-test-shared slicer-shared
```

## Running
The testing binary will be built in Godot's `./bin/` folder with the naming format: `./bin/test-slicer.{os}.tools.{arch}`. Running this binary will start the test suite.

## Adding new tests
You should be able to add new test files by simply adding a file with the `test.cpp` prefix into this folder. the SCsub file will run a Glob to try to find all test files at build.
