import math

import mlx.core as mx
import numpy as np

from mflux.models.common_models.qwen3_vl.qwen3_vl_attention import Qwen3VLAttention


def test_qwen3_vl_attention_native_gqa_matches_explicit_kv_repeat() -> None:
    mx.random.seed(17)
    attention = Qwen3VLAttention(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    hidden_states = mx.random.normal((1, 5, 16))

    output, cache = attention(hidden_states)

    query = attention.q_proj(hidden_states).reshape(1, 5, 4, 4)
    key = attention.k_proj(hidden_states).reshape(1, 5, 2, 4)
    value = attention.v_proj(hidden_states).reshape(1, 5, 2, 4)
    query = attention.q_norm(query).transpose(0, 2, 1, 3)
    key = attention.k_norm(key).transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    repeated_key = Qwen3VLAttention._repeat_kv(key, n_rep=2)
    repeated_value = Qwen3VLAttention._repeat_kv(value, n_rep=2)
    reference = mx.fast.scaled_dot_product_attention(
        query,
        repeated_key,
        repeated_value,
        scale=1 / math.sqrt(4),
    )
    reference = reference.transpose(0, 2, 1, 3).reshape(1, 5, 16)
    reference = attention.o_proj(reference)
    mx.eval(output, reference)

    np.testing.assert_allclose(np.asarray(output), np.asarray(reference), rtol=2e-5, atol=2e-5)
    cache_key, cache_value, cache_length = cache
    assert cache_key.shape == (1, 2, 5, 4)
    assert cache_value.shape == (1, 2, 5, 4)
    assert cache_length == 5
