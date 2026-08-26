from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Triton tests", allow_module_level=True)

from urm.backends import (
    TorchRoutedReductionBackend,
    TritonRoutedReductionBackend,
)


@pytest.mark.parametrize(
    ("queries", "sources", "route_width", "value_dim"),
    [(1, 7, 1, 13), (17, 31, 3, 70), (64, 128, 8, 128), (9, 97, 32, 257)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_triton_forward_matches_torch(
    queries: int,
    sources: int,
    route_width: int,
    value_dim: int,
    dtype: object,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(11)
    indices = torch.randint(
        sources,
        (queries, route_width),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    weights = torch.randn(
        queries, route_width, device="cuda", dtype=dtype, generator=generator
    )
    values = torch.randn(
        sources, value_dim, device="cuda", dtype=dtype, generator=generator
    )

    expected = (
        TorchRoutedReductionBackend()
        .execute(indices, weights, values, validate_indices=False)
        .output
    )
    actual = (
        TritonRoutedReductionBackend()
        .execute(indices, weights, values, validate_indices=False)
        .output
    )

    tolerance = 1e-4 if dtype is torch.float32 else 3e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_triton_backward_matches_torch_with_route_collisions() -> None:
    indices = torch.tensor([[1, 1, 4], [1, 3, 4]], device="cuda", dtype=torch.int64)
    weights = torch.randn(2, 3, device="cuda", dtype=torch.float32)
    values = torch.randn(5, 37, device="cuda", dtype=torch.float32)
    triton_weights = weights.detach().clone().requires_grad_(True)
    triton_values = values.detach().clone().requires_grad_(True)
    torch_weights = weights.detach().clone().requires_grad_(True)
    torch_values = values.detach().clone().requires_grad_(True)
    gradient = torch.randn(2, 37, device="cuda")

    triton_output = (
        TritonRoutedReductionBackend()
        .execute(indices, triton_weights, triton_values, validate_indices=False)
        .output
    )
    torch_output = (
        TorchRoutedReductionBackend()
        .execute(indices, torch_weights, torch_values, validate_indices=False)
        .output
    )
    triton_output.backward(gradient)
    torch_output.backward(gradient)

    torch.testing.assert_close(triton_weights.grad, torch_weights.grad)
    torch.testing.assert_close(
        triton_values.grad, torch_values.grad, atol=1e-5, rtol=1e-5
    )
