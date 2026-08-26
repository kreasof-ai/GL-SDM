"""Report whether the host can run the URM Triton backend."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_environment() -> dict[str, object]:
    torch_version = package_version("torch")
    triton_version = package_version("triton")
    report: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_version,
        "triton": triton_version,
        "cuda_available": False,
        "ready": False,
        "reasons": [],
    }
    reasons: list[str] = report["reasons"]  # type: ignore[assignment]
    if torch_version is None:
        reasons.append("PyTorch is not installed")
    if triton_version is None:
        reasons.append("Triton is not installed")
    if importlib.util.find_spec("torch") is not None:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            report.update(
                {
                    "gpu": properties.name,
                    "compute_capability": list(
                        torch.cuda.get_device_capability(device)
                    ),
                    "total_memory_bytes": properties.total_memory,
                }
            )
        else:
            reasons.append("CUDA is not available to PyTorch")
    report["ready"] = not reasons
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit non-zero unless PyTorch, Triton, and CUDA are available",
    )
    args = parser.parse_args()
    report = inspect_environment()
    print(json.dumps(report, indent=2))
    if args.require_ready and not report["ready"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
