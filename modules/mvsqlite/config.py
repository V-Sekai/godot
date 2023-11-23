# config.py


def can_build(env, platform):
    if platform == "ios" or platform == "web" or platform == "android":
        return False
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "MVSQLite",
        "MVSQLiteQuery",
    ]


def get_doc_path():
    return "doc_classes"
