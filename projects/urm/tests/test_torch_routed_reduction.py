from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from urm.backends import TorchRoutedReductionBackend


def test_torch_backend_matches_explicit_fp32_reduction_and_gradients() -> None:
    generator = torch.Generator().manual_seed(7)
    indices = torch.tensor([[1, 1, 3], [0, 4, 2]], dtype=torch.int64)
    weights = torch.randn(2, 3, generator=generator, requires_grad=True)
    values = torch.randn(5, 7, generator=generator, requires_grad=True)
    reference_weights = weights.detach().clone().requires_grad_(True)
    reference_values = values.detach().clone().requires_grad_(True)

    result = TorchRoutedReductionBackend().execute(indices, weights, values)
    expected = (
        reference_values[indices]
        .float()
        .mul(reference_weights.float().unsqueeze(-1))
        .sum(dim=1)
        .to(reference_values.dtype)
    )
    torch.testing.assert_close(result.output, expected)

    gradient = torch.randn(expected.shape, generator=generator)
    result.output.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(weights.grad, reference_weights.grad)
    torch.testing.assert_close(values.grad, reference_values.grad)


def test_torch_backend_rejects_out_of_range_routes() -> None:
    indices = torch.tensor([[0, 2]], dtype=torch.int32)
    weights = torch.ones(1, 2)
    values = torch.ones(2, 4)

    with pytest.raises(ValueError, match="indices"):
        TorchRoutedReductionBackend().execute(indices, weights, values)
