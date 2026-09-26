"""CSA2 教学实现的关键行为测试。"""

import torch

from csa2_attention import CSA2Config, CSA2Layer, Compressor, SharedCSA2State


def test_compressor_prefill_shape_and_weights() -> None:
    torch.manual_seed(0)
    compressor = Compressor(CSA2Config())
    latent, trace = compressor.forward_prefill(torch.randn(2, 8, 16))
    assert latent.shape == (2, 4, 8)
    assert trace["weights"].shape == (2, 4, 2, 8)
    torch.testing.assert_close(trace["weights"].sum(dim=2), torch.ones(2, 4, 8))


def test_decode_only_emits_when_group_is_complete() -> None:
    torch.manual_seed(1)
    config = CSA2Config()
    compressor = Compressor(config)
    state_v = torch.zeros(1, config.compress_ratio, config.head_dim)
    state_g = torch.zeros_like(state_v)
    first, state_v, state_g = compressor.forward_decode(
        torch.randn(1, 1, config.dim), 0, state_v, state_g
    )
    second, _, _ = compressor.forward_decode(
        torch.randn(1, 1, config.dim), 1, state_v, state_g
    )
    assert first is None
    assert second is not None and second.shape == (1, 1, config.head_dim)


def test_full_reindex_reuse_share_the_expected_state() -> None:
    torch.manual_seed(2)
    config = CSA2Config()
    x = torch.randn(1, 8, config.dim)
    shared = SharedCSA2State()
    x, shared, full_trace = CSA2Layer(config, "full").forward_prefill(x, shared)
    main_pointer, key_pointer = shared.main_kv.data_ptr(), shared.index_k.data_ptr()
    full_indices = shared.topk_indices.clone()

    x, shared, reindex_trace = CSA2Layer(config, "reindex").forward_prefill(x, shared)
    reindex_indices = shared.topk_indices.clone()
    assert shared.main_kv.data_ptr() == main_pointer
    assert shared.index_k.data_ptr() == key_pointer
    assert "index_scores" in reindex_trace
    assert reindex_indices.shape == full_indices.shape == (1, 8, 2)

    x, shared, reuse_trace = CSA2Layer(config, "reuse").forward_prefill(x, shared)
    assert "index_scores" not in reuse_trace
    torch.testing.assert_close(reuse_trace["global_indices"], reindex_indices)
    assert x.shape == (1, 8, config.dim)


def test_compressed_kv_is_causally_visible() -> None:
    torch.manual_seed(3)
    config = CSA2Config()
    _, _, trace = CSA2Layer(config, "full").forward_prefill(
        torch.randn(1, 8, config.dim), SharedCSA2State()
    )
    indices = trace["global_indices"]
    assert indices[0, 0].tolist() == [-1, -1]
    # 有效索引排序后位于前面，剩余 Top-K 槽位用 -1 填充。
    assert indices[0, 1].tolist() == [8, -1]
    valid = indices[indices >= 0]
    assert valid.min() >= 8 and valid.max() <= 11
