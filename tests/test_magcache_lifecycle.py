import importlib.util
from pathlib import Path
import sys

import torch

spec = importlib.util.spec_from_file_location("k6_magcache_lifecycle", Path(__file__).parents[1] / "kandinsky6/magcache.py")
magcache = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = magcache
spec.loader.exec_module(magcache)


def test_interrupted_first_step_cannot_reuse_old_generation_residuals():
    state = magcache.K6MagCacheState(num_steps=4, threshold=0.5, max_skip_steps=2, retention_ratio=0,
                                    mag_ratios={"t2va": [1.0] * 8, "i2va": [1.0] * 8})
    video, audio = torch.zeros(1, 3, 4), torch.zeros(1, 5, 2)

    def begin():
        return state.begin(profile="t2va", timestep=torch.tensor([1000.0]), cond_or_uncond=[0],
                           visual=video, audio=audio)

    first = begin()
    assert not first.skip
    state.record_computed(video, audio, video + 11, audio + 22, first)
    assert state._lanes[0].visual_residual is not None
    state.reset_generation()  # Native ON_CLEANUP and ON_PRE_RUN both invoke this.
    assert not state._lanes
    second = begin()
    assert not second.skip
    assert state._generation == 2


def test_reset_releases_partial_residuals_for_both_cfg_lanes():
    state = magcache.K6MagCacheState(num_steps=4, threshold=0.5, max_skip_steps=2, retention_ratio=0,
                                    mag_ratios={"t2va": [1.0] * 8, "i2va": [1.0] * 8})
    video, audio = torch.zeros(2, 3, 4), torch.zeros(2, 5, 2)
    decision = state.begin(profile="t2va", timestep=torch.tensor([1000.0]), cond_or_uncond=[0, 1],
                           visual=video, audio=audio)
    state.record_computed(video, audio, video + 1, audio + 1, decision)
    assert set(state._lanes) == {0, 1}
    state.reset_generation()
    assert state._profile is None and not state._lanes and not state._seen_lanes
