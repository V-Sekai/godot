# WasGo Lifecycle Design

## Contributors

* MMMaellon
* Pushknat
* Rammalyala
* Luftccc
* iFire
* Lyuma
* Saracen

## Abstract

This document defines how web assembly code is handled by the godot
engine. The approach outlined here is based on mozilla's JS
implementation:
[_https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly_](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly)

## Classes

We will need to define a WasGo class that holds the Wasm instance, it's
associated function table. The mozilla implementation also requires a
special memory buffer to hold all the stack and heap data that the wasm
code requires, but the wasm runtime we're using handles memory for us.
Reference line 116 where they instantiate a wasm module by passing in
the wasm code, and the max size of the stack, heap, and error buffers:
[_https://github.com/bytecodealliance/wasm-micro-runtime/blob/main/samples/basic/src/main.c_](https://github.com/bytecodealliance/wasm-micro-runtime/blob/main/samples/basic/src/main.c)

The function table is a lookup table that connects wasm functions to
their godot counterparts. An example of a table can be seen starting at
line 61 in the basic example for the wasm runtime we're using:
[_https://github.com/bytecodealliance/wasm-micro-runtime/blob/main/samples/basic/src/main.c_](https://github.com/bytecodealliance/wasm-micro-runtime/blob/main/samples/basic/src/main.c)

The function table will be defined by the game settings within Godot. If
a user decides they want to expose every function, then we will populate
the table with as many Godot functions as we can. By default we should
only expose a whitelist of functions that excludes networking, folder
and file manipulation, and other dangerous parts of the godot engine.

## Lifecycle

The Official WebAssembly Github defines a WebAssembly C and C++ API that
is meant to be VM agnostic. We should follow their lifecycle. It's best
exemplified by their hello world example: [_wasm-c-api/hello.c at master
·
WebAssembly/wasm-c-api_](https://github.com/WebAssembly/wasm-c-api/blob/master/example/hello.c)

Basically the steps are:

1.  Start Up the Wasm Engine.

    a. We plan for the engine to be a Godot singleton, meaning only one
    copy of the engine will be running at any time and all wasm code
    gets passed through it.

2.  Load Binary

    a. We'll use the built-in Godot tools to ready the entire wasm
    module we're planning on running into a memory buffer. There's
    nothing special about the memory buffer, it's literally just an
    array of bytes.

3.  Compile

    a. Technically we can skip this part if we want to run the runtime
    as an interpreter because the runtime we're using supports that.

4.  Connect External Functions

    a. We define the function table an link up the appropriate
    callbacks
    b. If the wasm code is trying to call a function that we are not
    exposing we instead connect it to an error callback so we can
    handle it later or at least let the user know.

5.  Instantiate

    a. Finally we instantiate the compiled wasm code with the function
    table, so that it's ready to run.

6.  Extract the Main Function

    a. We need to find the main function of the wasm code. We expect
    that it'll be called main and it will accept arguments like any
    old main function.

7.  Call the Main Function

    a. Call the main function in a new thread so that if an error
    callback is ever triggered we can kill that thread and report
    the error

8.  Shut Down & Cleanup

## Safety Settings

Because we connect external functions at runtime in step 4 we can change
how much power we give to each program. We plan to allow for 3 levels of
power.

### Level 1

Completely Safe.

We whitelist only functions that we know cannot be abused for any
malicious purposes. Functions relating to things such as file io and
sending and receiving network data will throw errors in any program that
runs them.

### Level 2

Safe but with Wrappers.

The same core Godot functions that are whitelisted in Level 1 are also
available here in Level 2, but a custom set of classes and functions
will be added that will act as wrappers that expose more dangerous
functions in a safe way. For example, one could make a wrapper for
writing a file to disk that forces a confirmation dialog, or a wrapper
for networking that only allows sending and receiving data from a
whitelist of urls.

### Level 3

Completely Unsafe.

Here we expose every function possible including the custom classes and
wrappers found in Level 2.

## Inclusion Order

The functions in Level 1 are a subset of the functions exposed in Level
2 and the functions exposed in Level 2 are a subset of the functions
exposed in Level 3. The wrappers and custom classes in levels 2 and 3
will not overwrite an existing function. This is done so that any
function found in multiple levels will behave the same regardless of
which Level is being exposed. We want to ensure that a program that
works on a lower level never breaks on a higher level.
