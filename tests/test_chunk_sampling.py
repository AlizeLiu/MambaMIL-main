"""
Unit tests for Hilbert contiguous chunk sampling.
Run: python tests/test_chunk_sampling.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from dataset.dataset_survival import hilbert_chunk_sample_indices


def test_no_sampling():
    """n <= max_seq_len should return full arange."""
    print("=== No Sampling Test ===")
    idx = hilbert_chunk_sample_indices(n=1000, max_seq_len=2500, chunk_size=50)
    assert idx.shape[0] == 1000, f"Expected 1000, got {idx.shape[0]}"
    assert torch.equal(idx, torch.arange(1000)), "Should be arange(1000)"
    print("  PASSED ✅\n")


def test_exact_length():
    """Output length should equal max_seq_len."""
    print("=== Exact Length Test ===")
    idx = hilbert_chunk_sample_indices(n=10000, max_seq_len=2500, chunk_size=50)
    assert idx.shape[0] == 2500, f"Expected 2500, got {idx.shape[0]}"
    print(f"  Length: {idx.shape[0]} ✅")
    print("  PASSED ✅\n")


def test_contiguous_chunks():
    """Each chunk of 50 tokens should be contiguous (diff=1 within chunk)."""
    print("=== Contiguous Chunk Test ===")
    idx = hilbert_chunk_sample_indices(n=10000, max_seq_len=2500, chunk_size=50)

    # Reshape into chunks
    num_chunks = 2500 // 50
    chunks = idx.reshape(num_chunks, 50)

    for i in range(num_chunks):
        chunk = chunks[i]
        diffs = chunk[1:] - chunk[:-1]
        assert torch.all(diffs == 1), f"Chunk {i} not contiguous: diffs={diffs.unique()}"

    print(f"  {num_chunks} chunks, all contiguous ✅")
    print("  PASSED ✅\n")


def test_ordered_chunks():
    """Chunk start positions should be in ascending order."""
    print("=== Ordered Chunks Test ===")
    idx = hilbert_chunk_sample_indices(n=10000, max_seq_len=2500, chunk_size=50)

    num_chunks = 2500 // 50
    chunks = idx.reshape(num_chunks, 50)

    for i in range(1, num_chunks):
        assert chunks[i][0] > chunks[i-1][-1], \
            f"Chunk {i} start {chunks[i][0]} <= chunk {i-1} end {chunks[i-1][-1]}"

    print(f"  {num_chunks} chunks, all ordered ✅")
    print("  PASSED ✅\n")


def test_train_randomness():
    """Training mode should produce different indices across calls."""
    print("=== Train Randomness Test ===")
    torch.manual_seed(42)
    idx1 = hilbert_chunk_sample_indices(n=10000, max_seq_len=2500, chunk_size=50, training=True)
    torch.manual_seed(123)
    idx2 = hilbert_chunk_sample_indices(n=10000, max_seq_len=2500, chunk_size=50, training=True)

    # At least one chunk should differ
    num_chunks = 2500 // 50
    chunks1 = idx1.reshape(num_chunks, 50)
    chunks2 = idx2.reshape(num_chunks, 50)

    any_different = False
    for i in range(num_chunks):
        if not torch.equal(chunks1[i], chunks2[i]):
            any_different = True
            break

    assert any_different, "Training mode should produce different results with different seeds"
    print("  Different seeds → different chunks ✅")
    print("  PASSED ✅\n")


def test_eval_deterministic():
    """Eval mode with center strategy should be deterministic."""
    print("=== Eval Deterministic Test ===")
    idx1 = hilbert_chunk_sample_indices(n=10000, max_seq_len=2500, chunk_size=50,
                                         training=False, eval_strategy='center')
    idx2 = hilbert_chunk_sample_indices(n=10000, max_seq_len=2500, chunk_size=50,
                                         training=False, eval_strategy='center')

    assert torch.equal(idx1, idx2), "Eval center mode should be deterministic"
    print("  Two calls → identical results ✅")
    print("  PASSED ✅\n")


def test_non_divisible():
    """Non-divisible max_seq_len should work (remainder chunk)."""
    print("=== Non-Divisible Test ===")

    # 2550 / 50 = 51 chunks (exact)
    idx = hilbert_chunk_sample_indices(n=10000, max_seq_len=2550, chunk_size=50)
    assert idx.shape[0] == 2550, f"Expected 2550, got {idx.shape[0]}"
    print(f"  max_seq_len=2550: {idx.shape[0]} tokens ✅")

    # 2530 / 50 = 50 full + 30 remainder
    idx = hilbert_chunk_sample_indices(n=10000, max_seq_len=2530, chunk_size=50)
    assert idx.shape[0] == 2530, f"Expected 2530, got {idx.shape[0]}"
    print(f"  max_seq_len=2530: {idx.shape[0]} tokens ✅")

    # 2510 / 50 = 50 full + 10 remainder
    idx = hilbert_chunk_sample_indices(n=10000, max_seq_len=2510, chunk_size=50)
    assert idx.shape[0] == 2510, f"Expected 2510, got {idx.shape[0]}"
    print(f"  max_seq_len=2510: {idx.shape[0]} tokens ✅")

    print("  PASSED ✅\n")


def test_integration_smoke():
    """Integration test: sample from a feature tensor."""
    print("=== Integration Smoke Test ===")
    x = torch.randn(10000, 1024)
    idx = hilbert_chunk_sample_indices(10000, 2500, 50, training=True)
    x_sampled = x[idx]

    assert x_sampled.shape == (2500, 1024), f"Expected (2500, 1024), got {x_sampled.shape}"

    # Check each chunk preserves original features
    num_chunks = 2500 // 50
    chunks_idx = idx.reshape(num_chunks, 50)
    chunks_feat = x_sampled.reshape(num_chunks, 50, 1024)

    for i in range(num_chunks):
        expected = x[chunks_idx[i]]
        assert torch.equal(chunks_feat[i], expected), f"Chunk {i} features mismatch"

    print(f"  {x.shape} -> {x_sampled.shape} ✅")
    print(f"  All chunks preserve original features ✅")
    print("  PASSED ✅\n")


def test_no_cuda_tensor():
    """Indices should always be CPU tensors."""
    print("=== No CUDA Tensor Test ===")
    idx = hilbert_chunk_sample_indices(n=10000, max_seq_len=2500, chunk_size=50, training=True)
    assert idx.device == torch.device('cpu'), f"Expected CPU, got {idx.device}"
    print(f"  Device: {idx.device} ✅")
    print("  PASSED ✅\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Hilbert Chunk Sampling Tests")
    print("=" * 50)
    print()

    test_no_sampling()
    test_exact_length()
    test_contiguous_chunks()
    test_ordered_chunks()
    test_train_randomness()
    test_eval_deterministic()
    test_non_divisible()
    test_integration_smoke()
    test_no_cuda_tensor()

    print("=" * 50)
    print("ALL TESTS PASSED ✅")
    print("=" * 50)
