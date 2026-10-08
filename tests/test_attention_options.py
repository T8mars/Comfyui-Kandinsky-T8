"""Exercise the real fused block with small recording attention backends."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("cross_gates", [False, True])
def test_all_attention_branches_receive_native_options(cross_gates):
    path = Path(__file__).parents[1] / "kandinsky6/ldm/model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    block = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FusedTransformerDecoderBlock")
    forward = next(node for node in block.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_apply_scale_shift_norm_fp32", "_apply_gate_sum_fp32")]
    namespace = {"torch": torch, "_modulation_fp32": lambda module, x: module(x),
                 "get_shift_scale_gate": lambda x: tuple(part.unsqueeze(1) for part in x.chunk(3, dim=-1))}
    exec(compile(ast.Module(body=helpers + [forward], type_ignores=[]), str(path), "exec"), namespace)
    calls = []
    options = {"optimized_attention_override": lambda label, x: calls.append(label) or (x + 1)}

    def attention(label):
        def apply(x, *args, transformer_options=None, **kwargs):
            assert transformer_options is options, label
            return transformer_options["optimized_attention_override"](label, x)
        return apply

    def modulation(width):
        return lambda x: torch.ones(x.shape[0], width)

    def branch(label, width):
        return SimpleNamespace(visual_modulation=modulation(width * 9),
                               self_attention_norm=torch.nn.Identity(), cross_attention_norm=torch.nn.Identity(),
                               feed_forward_norm=torch.nn.Identity(), feed_forward=torch.zeros_like,
                               self_attention=attention(label + " self"), cross_attention=attention(label + " text"))

    dim, dim_a = (4, 2) if cross_gates else (3, 3)
    model = SimpleNamespace(videoT=branch("video", dim), audioT=branch("audio", dim_a),
                            fix_modulation=True, cross_gates=cross_gates, ca_rope=True, model_dim=dim, model_dim_a=dim_a,
                            va_modulation=modulation(2 * dim + dim_a if cross_gates else 3 * dim),
                            av_modulation=modulation(2 * dim_a + dim if cross_gates else 3 * dim_a),
                            va_normalization=torch.nn.Identity(), av_normalization=torch.nn.Identity(),
                            va_cross_attention=attention("video/audio"), av_cross_attention=attention("audio/video"))
    video, audio = torch.zeros(2, 5, dim), torch.zeros(2, 7, dim_a)
    outputs = namespace["forward"](model, video, audio, video, audio,
                                    torch.zeros(2, 1), torch.zeros(2, 1), object(), object(), options)
    assert calls == ["video self", "video text", "audio self", "audio text", "video/audio", "audio/video"]
    assert outputs[0].shape == video.shape and outputs[1].shape == audio.shape
    assert all(torch.isfinite(output).all() for output in outputs)
