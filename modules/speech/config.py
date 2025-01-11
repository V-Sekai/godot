def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "Speech",
        "SpeechDecoder",
        "SpeechProcessor",
        "PlaybackStats",
    ]


def get_doc_path():
    return "doc_classes"
