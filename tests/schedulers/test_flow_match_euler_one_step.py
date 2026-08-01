from types import SimpleNamespace

import pytest

from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler


def _scheduler(num_steps: int, schedule: str = "linear") -> FlowMatchEulerDiscreteScheduler:
    config = SimpleNamespace(
        num_inference_steps=num_steps,
        model_config=SimpleNamespace(),
        sigma_schedule=schedule,
    )
    return FlowMatchEulerDiscreteScheduler(config)


@pytest.mark.fast
def test_one_step_schedule_is_full_denoise():
    # Crashed with ZeroDivisionError before the fix (upstream #494). The 1-step schedule
    # must match what get_timesteps_and_sigmas and the set_mu path already produce.
    scheduler = _scheduler(1)
    assert scheduler.sigmas.tolist() == [1.0, 0.0]
    assert scheduler.timesteps.tolist() == [1000.0]


@pytest.mark.fast
@pytest.mark.parametrize("num_steps", [2, 4])
def test_multi_step_schedules_are_unchanged(num_steps):
    scheduler = _scheduler(num_steps)
    sigmas = scheduler.sigmas.tolist()
    assert len(sigmas) == num_steps + 1
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[-2] == pytest.approx(0.02)  # shift_terminal
    assert sigmas[-1] == 0.0
    assert sigmas == sorted(sigmas, reverse=True)


@pytest.mark.fast
def test_one_step_matches_sibling_paths():
    timesteps, sigmas = FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas(
        image_seq_len=1024,
        num_inference_steps=1,
    )
    scheduler = _scheduler(1)
    assert scheduler.sigmas.tolist() == sigmas.tolist()
    assert scheduler.timesteps.tolist() == timesteps.tolist()
