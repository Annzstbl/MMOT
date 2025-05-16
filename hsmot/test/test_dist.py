import pytest
from hsmot.util.dist import *
import torch
# 测试用例
@pytest.mark.parametrize("x1, x2, aligned, expected", [
    (
        torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.25],
                       [0.0, 0.0, 0.5, 0.5, 0.75]]),
        torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.5],
                       [0.1, 0.1, 0.5, 0.5, 0.2]]),
        True,
        torch.tensor([1.25, 0.65])
    ),
    (
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.5]]),
        False,
        torch.tensor([4.5])
    ),
    (
        torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.25],
                       [0.0, 0.0, 0.5, 0.5, 0.75]]),
        torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.5],
                       [0.1, 0.1, 0.5, 0.5, 0.2]]),
        False,
        torch.tensor([[1.25, 1.85],[1.25,0.65]])
    ),
])
def test_l1_dist_rotate(x1, x2, aligned, expected):
    result = l1_dist_rotate(x1, x2, aligned)
    assert torch.allclose(result, expected, atol=1e-5), f"结果 {result} 与期望值 {expected} 不匹配"