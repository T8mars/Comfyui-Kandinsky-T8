"""Checkpoint key normalization shared by the loader and offline converter."""

PREFIXES = (
    ("visual_transformer_blocks.", "visual_blocks."),
    ("audio_out_layer.", "audio_outLayer."),
)
INNER_NAMES = (
    (".timestep_embedder.linear_1.", ".in_layer."),
    (".timestep_embedder.linear_2.", ".out_layer."),
    (".video_dec_block.", ".videoT."),
    (".audio_dec_block.", ".audioT."),
    (".feed_forward.net.0.proj.", ".feed_forward.in_layer."),
    (".feed_forward.net.2.", ".feed_forward.out_layer."),
    (".attn.", ".self_attention."),
)


def native_key(key):
    for source, target in PREFIXES:
        if key.startswith(source):
            key = target + key[len(source):]
            break
    for source, target in INNER_NAMES:
        key = key.replace(source, target)
    return key
