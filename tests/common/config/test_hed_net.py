import mlx.core as mx
import pytest

from mflux.models.hed.hed import _BLOCKS, _HEDNet


def _fake_state():
    # torch-layout (OIHW) tensors with the real channel dims but tiny random values, so the net wiring
    # can be exercised without downloading the 28MB checkpoint.
    dims = {"block1": (3, 64), "block2": (64, 128), "block3": (128, 256), "block4": (256, 512), "block5": (512, 512)}
    state = {"norm": mx.zeros((1, 3, 1, 1))}
    for name, n_conv in _BLOCKS:
        cin, cout = dims[name]
        for c in range(n_conv):
            i = cin if c == 0 else cout
            state[f"{name}.convs.{c}.weight"] = mx.zeros((cout, i, 3, 3))
            state[f"{name}.convs.{c}.bias"] = mx.zeros((cout,))
        state[f"{name}.projection.weight"] = mx.zeros((1, cout, 1, 1))
        state[f"{name}.projection.bias"] = mx.zeros((1,))
    return state


@pytest.mark.fast
def test_hed_net_emits_five_downsampled_single_channel_side_outputs():
    net = _HEDNet(_fake_state())
    projections = net(mx.zeros((1, 64, 64, 3)))
    mx.eval(projections)
    # one side output per block, each a single channel, halving in resolution after the first block
    assert len(projections) == 5
    expected = [64, 32, 16, 8, 4]
    for p, size in zip(projections, expected):
        assert p.shape == (1, size, size, 1)
