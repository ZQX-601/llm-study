"""验证局部 SWA 教学实现的关键不变量。"""

import unittest

import torch

from swa_attention import LocalSWA, SWAConfig


class LocalSWATest(unittest.TestCase):
    """检查形状、因果窗口范围，以及 prefill/decode 的一致性。"""
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.config = SWAConfig()
        self.model = LocalSWA(self.config).eval()
        self.x = torch.randn(2, 4, self.config.dim)

    def test_prefill_shapes_and_window_indices(self) -> None:
        """每个 query 最多取得 W 个局部索引；空槽以 -1 表示。"""
        with torch.no_grad():
            output, trace = self.model(self.x, return_trace=True)
        self.assertEqual(tuple(output.shape), (2, 4, 16))
        self.assertEqual(tuple(trace["q"].shape), (2, 4, 4, 8))
        self.assertEqual(tuple(trace["kv"].shape), (2, 4, 8))
        self.assertEqual(tuple(trace["weights"].shape), (2, 4, 4, 2))
        self.assertEqual(trace["indices"][0].tolist(),
                         [[0, -1], [0, 1], [1, 2], [2, 3]])

    def test_one_layer_window_excludes_older_token(self) -> None:
        """W=2 且只有一层时，修改 A 不应影响 D 的 attention 输出。"""
        modified = self.x.clone()
        modified[:, 0] += 100
        with torch.no_grad():
            original = self.model(self.x)
            changed = self.model(modified)
        torch.testing.assert_close(original[:, 3], changed[:, 3])

    def test_decode_matches_full_prefill_last_token(self) -> None:
        """已填充的环形缓存应复现完整 prefill 最后位置的局部注意力。"""
        new = torch.randn(2, 1, self.config.dim)
        with torch.no_grad():
            self.model(self.x)
            decoded, trace = self.model(new, start_pos=4, return_trace=True)
            full = self.model(torch.cat((self.x, new), dim=1))
        torch.testing.assert_close(decoded, full[:, -1:], atol=1e-5, rtol=1e-5)
        self.assertEqual(tuple(trace["cache"].shape), (2, 2, 8))


if __name__ == "__main__":
    unittest.main()
