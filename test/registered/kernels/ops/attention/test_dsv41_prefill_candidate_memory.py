"""Regression contract for compact DeepSeek-V4.1 prefill candidates."""

import torch

from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    PrefillCandidateBlocks,
    TritonPrefillInputs,
)
from sglang.srt.layers.attention.dsv4.triton_candidate_indexer import (
    TritonCandidateIndexer,
)
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def test_prefill_source_publishes_compact_blocks_across_score_tiles(monkeypatch):
    """Do not retain or concatenate one full-width boolean mask per score tile."""
    rows, length = 17, 16393  # Cross the 2048 * 8 candidate-window boundary.
    candidate_blocks, block_size = 2048, 8
    keys = torch.ones((length, 1), device=DEVICE)
    inputs = TritonPrefillInputs(
        q=torch.ones((rows, 1, 1), device=DEVICE),
        weights=torch.ones((rows, 1), device=DEVICE),
        compress_lens=torch.full((rows,), length, device=DEVICE, dtype=torch.int32),
        request_starts=torch.zeros(rows, device=DEVICE, dtype=torch.int32),
        lens_per_request=[length],
        rows_per_request=[rows],
        get_keys=lambda _: keys,
    )
    indexer = TritonCandidateIndexer(candidate_blocks, block_size, budget_bytes=1)
    score_tile_rows = []

    def dense_scores(q, keys, weights):
        score_tile_rows.append(q.shape[0])
        return torch.zeros((q.shape[0], keys.shape[0]), device=q.device)

    def forbidden_cat(*args, **kwargs):
        raise AssertionError("prefill candidates must not concatenate score-tile masks")

    monkeypatch.setattr(indexer, "_dense_scores", dense_scores)
    monkeypatch.setattr(torch, "cat", forbidden_cat)
    out = torch.empty((rows, 512), dtype=torch.int32, device=DEVICE)
    published = indexer.publish_prefill(inputs, out)

    assert score_tile_rows == [1] * rows
    assert isinstance(published, PrefillCandidateBlocks)
    blocks = published.request_blocks[0]
    assert blocks.shape == (rows, candidate_blocks)
    assert blocks.dtype == torch.int32
    assert blocks.numel() * blocks.element_size() == rows * candidate_blocks * 4

    # Exact metadata bound for the original 16K-row H200 prefill shape. This
    # is metadata only, not a claim about the end-to-end peak allocation.
    assert 16_384 * candidate_blocks * 4 == 128 << 20
