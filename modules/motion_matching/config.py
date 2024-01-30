def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "CritDampSpring",
        "BonePositionVelocityMotionFeature",
        "RootVelocityMotionFeature",
        "PredictionMotionFeature",
        "MotionPlayer",
        "MotionFeature",
    ]


def get_doc_path():
    return "doc_classes"
