"""Execute the real sampler body with small CPU tensors and controlled rollout."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

root = Path(__file__).parents[1] / "kandinsky6"


def load(name):
    spec = importlib.util.spec_from_file_location("k6_sampling_test_" + name, root / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_null_audio_metadata_falls_back_before_device_cast():
    calls, observed = [], {}

    class Joint:
        shape = (1, 1, 2)

        def __init__(self, streams):
            self.streams = streams

        def to(self, *args, **kwargs):
            return self

        def unbind(self):
            return self.streams

    samples = Joint([torch.zeros(1, 16, 2, 2, 2), torch.zeros(1, 3, 40)])
    context, pooled = torch.randn(1, 2, 4), torch.randn(1, 3)
    model = SimpleNamespace(load_device=torch.device("cpu"), model_options={},
                            pre_run=lambda: calls.append("pre"), cleanup=lambda: calls.append("model"),
                            model=SimpleNamespace(get_dtype_inference=lambda: torch.float32,
                                                  memory_required=lambda shape: 0,
                                                  process_latent_in=lambda value: value,
                                                  process_latent_out=lambda value: value,
                                                  diffusion_model=object()))

    def rollout(dit, video, audio, context, pooled, **kwargs):
        observed.update(context=context, pooled=pooled, **kwargs)
        return video, audio

    namespace = {"torch": torch, "joint_conditioning": load("piflow_conditioning").joint_conditioning,
                 "validate_noise_mask": load("piflow_latent").validate_noise_mask,
                 "cleanup_sampling": load("runtime_cleanup").cleanup_sampling, "rollout": rollout,
                 "mm": SimpleNamespace(load_models_gpu=lambda *args, **kwargs: calls.append("load"),
                                       cast_to_device=lambda value, device, dtype: value.to(device=device, dtype=dtype),
                                       intermediate_device=lambda: torch.device("cpu"), intermediate_dtype=lambda: torch.float32),
                 "comfy": SimpleNamespace(sample=SimpleNamespace(prepare_noise=lambda *args: samples),
                                          nested_tensor=SimpleNamespace(NestedTensor=Joint),
                                          model_prefetch=SimpleNamespace(cleanup_prefetch_queues=lambda: calls.append("prefetch")))}
    tree = ast.parse((root / "sampling.py").read_text(encoding="utf-8"))
    function = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "sample_piflow")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(root / "sampling.py"), "exec"), namespace)
    result = namespace["sample_piflow"](model, {"samples": samples},
        [[context, {"pooled_output": pooled, "k6_audio_context": None, "k6_audio_pooled_output": None}]], 0, 1)
    assert observed["audio_context"] is context and observed["audio_pooled"] is pooled
    assert result["samples"].unbind() == samples.unbind()
    assert calls == ["load", "pre", "prefetch", "model"]
