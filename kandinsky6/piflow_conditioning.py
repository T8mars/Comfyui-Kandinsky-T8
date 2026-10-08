"""Validate conditioning features supported by the distilled DX sampler."""


def joint_conditioning(positive):
    if len(positive) != 1:
        raise ValueError("K6 PiFlow expects one joint video/audio conditioning, without regional or scheduled prompts.")
    context, metadata = positive[0]
    unsupported = [name for name in ("area", "mask", "hooks", "control", "gligen")
                   if metadata.get(name) is not None]
    for name, default in (("start_percent", 0.0), ("end_percent", 1.0),
                          ("clip_start_percent", 0.0), ("clip_end_percent", 1.0), ("strength", 1.0)):
        if metadata.get(name, default) != default:
            unsupported.append(name)
    if unsupported:
        raise ValueError("K6 PiFlow does not support regional, masked, scheduled or hooked conditioning: "
                         + ", ".join(unsupported))
    if "pooled_output" not in metadata:
        raise ValueError("K6 PiFlow requires CLIP-L pooled conditioning.")
    return context, metadata
