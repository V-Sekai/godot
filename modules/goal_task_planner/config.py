# config.py


def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "PlannerMultigoal",
        "PlannerDomain",
        "PlannerPlan",
    ]


def get_doc_path():
    return "doc_classes"
