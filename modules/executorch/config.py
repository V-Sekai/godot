def can_build(env, platform):
    """Check if the module can be built on the target platform."""
    return True


def configure(env):
    """Configure build environment for the ExecuTorch module."""
    pass


def get_doc_classes():
    """Return documentation classes."""
    return [
        "ExecuTorchRuntime",
        "ExecuTorchResource", 
        "ExecuTorchNode",
        "ExecuTorchLinearRegression",
        "ExecuTorchModule",
        "ExecuTorchMemoryManager",
        "ModelContextProtocolServer",
    ]


def get_doc_path():
    """Return documentation path."""
    return "doc_classes"
