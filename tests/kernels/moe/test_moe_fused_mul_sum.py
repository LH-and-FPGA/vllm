# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE fused mul-sum kernel

Run `pytest tests/kernels/moe/test_moe_fused_mul_sum.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum
from vllm.platforms import current_platform

NUM_REAL_TOKENS = 8
NUM_PADDED_TOKENS = 4
TOP_K = 4
HIDDEN_SIZE = 512
NUM_EXPERTS = 32
# The w2 GEMM never writes rows belonging to padded tokens, so whatever the
# workspace happened to hold is what the reduction would pick up.
STALE = -999.0


def _make_padded_case(dtype: torch.dtype, preceding_value: int):
    """Build a batch whose trailing rows are cudagraph padding.

    `expert_map` starts one element into a larger buffer, so the memory just
    before it holds `preceding_value`. A correct kernel never reads it.
    """
    num_tokens = NUM_REAL_TOKENS + NUM_PADDED_TOKENS

    inputs = torch.zeros(num_tokens, TOP_K, HIDDEN_SIZE, dtype=dtype, device="cuda")
    inputs[NUM_REAL_TOKENS:] = STALE

    topk_weights = torch.full((num_tokens, TOP_K), 0.5, dtype=dtype, device="cuda")

    topk_ids = torch.zeros(num_tokens, TOP_K, dtype=torch.int32, device="cuda")
    topk_ids[NUM_REAL_TOKENS:] = -1

    backing = torch.full(
        (NUM_EXPERTS + 1,), preceding_value, dtype=torch.int32, device="cuda"
    )
    backing[1:] = torch.arange(NUM_EXPERTS, dtype=torch.int32, device="cuda")

    outputs = torch.empty(num_tokens, HIDDEN_SIZE, dtype=dtype, device="cuda")
    return inputs, topk_weights, topk_ids, backing[1:], outputs


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("preceding_value", [-7, 0, 12345])
def test_padded_rows_contribute_nothing(dtype: torch.dtype, preceding_value: int):
    """Rows whose topk_ids are -1 must reduce to zero.

    Parametrizing the memory in front of `expert_map` pins the contract: the
    result cannot depend on bytes outside the tensor.
    """
    inputs, topk_weights, topk_ids, expert_map, outputs = _make_padded_case(
        dtype, preceding_value
    )

    moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
        outputs=outputs,
    )

    padded = outputs[NUM_REAL_TOKENS:]
    torch.testing.assert_close(padded, torch.zeros_like(padded), rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
def test_unpadded_rows_are_unaffected():
    """Masking the padded rows must not change the ordinary reduction."""
    dtype = torch.bfloat16
    inputs, topk_weights, topk_ids, expert_map, outputs = _make_padded_case(dtype, -7)

    inputs[:NUM_REAL_TOKENS] = 1.0
    topk_ids[:NUM_REAL_TOKENS] = torch.arange(
        TOP_K, dtype=torch.int32, device="cuda"
    ).expand(NUM_REAL_TOKENS, TOP_K)

    moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
        outputs=outputs,
    )

    expected = torch.full(
        (NUM_REAL_TOKENS, HIDDEN_SIZE), TOP_K * 0.5, dtype=dtype, device="cuda"
    )
    torch.testing.assert_close(outputs[:NUM_REAL_TOKENS], expected)
