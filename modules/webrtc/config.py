def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "WebRTCPeerConnection",
        "WebRTCDataChannel",
        "WebRTCMultiplayerPeer",
        "WebRTCPeerConnectionExtension",
        "WebRTCDataChannelExtension",
        "WebRTCLibDataChannel",
        "WebRTCLibPeerConnection",
    ]


def get_doc_path():
    return "doc_classes"
