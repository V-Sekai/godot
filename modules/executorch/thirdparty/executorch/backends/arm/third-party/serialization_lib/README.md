TOSA Serialization Library
==========================

# Introduction

The *TOSA Serialization library* provides methods to read and write serialized
TOSA graphs (<https://developer.mlplatform.org/w/tosa/>).  The library includes
a FlatBuffers schema and a C++ API for reading and writing TOSA graphs.

# Usage

The section below describes serialization_lib API usage. For more
details, please refer to `include/tosa_serialization_handler.h`.

## TosaSerializationHandler

This is the top-level class that contains the entire TOSA graph.  In
particular, it contains a vector of `TosaSerializationBasicBlock` objects,
and provides API for file IO, block access, and version checking.

    a. `LoadFileJson(filename)`:

        Load json-formatted file "filename" from disk, and initialize the
        internal graph structure.

        Requires the schema file to be loaded via `LoadFileSchema()`.

    b. `SaveFileJson(filename)`:

        Snapshots the internal graph structure and saves out JSON-formatted file
        `filename` to disk.
        Requires the schema file to be loaded via `LoadFileSchema()`.

    c. `LoadFileTosaFlatbuffer(filename)`:

        Load serialized flatbuffer file "filename" from disk, and initialize the
        internal graph structure.

    d. `SaveFileTosaFlatbuffer(filename)`:

        Snapshot the internal graph structure and saves out serialized
        flatbuffer file `filename` to disk.

    e. `GetTosaVersion()`:

        Return TOSA version implemented by the serialization library.

    f. `GetBlockByName()`:

        Return vector of `TosaSerializationBasicBlock`. Returns `nullptr` if
        nothing matches. A valid graph must have one `main` block as the first
        block being traversed.

    g. `GetMainBlock()`:

        Shortcut for accessing the `main` block. Equivalent to
        `GetBlockByName("main")`.

    h. `GetInputs()` / `GetOutputs()`:

        Shortcut for `main` block's input/output tensor name. Input tensors of
        the main block are usually treated as `tosa.PLACEHOLDER`. Output tensors
        are the output of the entire graph and should be evaluated when graph
        traversal has finished.

## TosaSerializationBasicBlock

This is the basic-block class. It contains vectors of
`TosaSerializationOperator` and `TosaSerializationTensor`. Once entering
a basic block, all of the operators within the block will be evaluated
in order.

Upon reaching a TOSA control flow operator (`tosa.WHILE` and
`tosa.COND_IF`), the status of current unfinished block will be saved, and
the blocks specified in control flow operator will be evaluated first. Once
the control flow blocks finishes its evaluation, the original unfinished
block status will be restored and evaluation continues.  This is more
analogous to a function call than a compiler basic block.

    a. `GetName()`:

        Return string of the basic block.

    b. `GetOperators()`:

        Return vector of `TosaSerializationOperator`

    c. `GetTensors()`:

        Return vector of `TosaSerializationTensor`

    d. `GetTensorByName(name)`:

        Return the `TosaSerializationTensor` with name `name`. Returns `nullptr`
        if nothing matches.

    e. `GetInputs()` / `GetOutputs()`:

        Return input/output tensor name of the basic block.

## TosaSerializationOperator

The operator class contains (1) what TOSA Op, (2) attribute (compile-time-
known input), (3) quantization information and (4) input/output tensor
names. The combination of (Op, attribute, quantization information) is
defined in include/operator.def.

    a. `GetOp()`:

        Return TOSA Op. Defined in schema `tosa.fbs`.

    b. `GetAttribute()` / `GetAttributeType()`:

        `GetAttribute()` returns the base object of attribute.
        `GetAttributeType()` returns which type of attribute the base object
        needs to be casted to.  Type of attribute is defined in `tosa.fbs` and
        `include/attribute.def`.

    c. `GetQuantInfo()` + `GetQuantInfoType()`:

        `GetQuantInfo()` returns the base object's quantization information.
        `GetQuantInfoType()` returns which type of quantization information the
        base object needs to be casted to. Type of quantization information is
        defined in `tosa.fbs` and `include/quant_info.def`.

    d. `GetInputTensorNames()` / `GetOutputTensorNames()`:

        Returns the input/output tensor name of the basic block.

    e. `GetInputTensors()` / `GetOutputTensors()`:

        Returns the input/output tensor of the basic block.

## TosaSerializationTensor

The tensor class contains (1) data type, (2) shape, (3) symbolic link to
numpy file, (4) format and (5) usage.

    a. `GetName()` / `SetName(name)`:

        `GetName()` returns the name of the tensor. `SetName()` sets the name
        of the tensor.

    b. `GetShape()`:

        Returns the shape of the tensor as `vector<int32_t>`.

    c. `GetDtype()` / `SetDtype(dtype)`:

        `GetDtype()` returns the data type of the tensor. `SetDtype()` sets the
        data type of the tensor. DType is defined in `tosa.fbs`.

    d. `GetNpyFilePtr()`:

        Return the numpy file pointer of this tensor if this is a constant
        tensor. Return `nullptr` if the tensor is not constant.

# License

The *TOSA Serialization Library* is licensed under Apache-2.0.
