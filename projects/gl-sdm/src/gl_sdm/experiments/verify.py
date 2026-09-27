"""Independent recurrence, backward, causality and cached continuation checks."""
import copy
import tempfile
from contextlib import ExitStack
from unittest.mock import patch
import torch
from gl_sdm.experiments.checkpoint import save, load
from gl_sdm.model import create_model


def compare(actual, expected, label, atol=0.03, rtol=0.03):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise FloatingPointError(f"{label}: non-finite tensor")
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol, msg=lambda detail: f"{label}\n{detail}")
    return (actual.float() - expected.float()).abs().max().item()


def compare_bf16_sdm(actual, expected, label):
    # Sparse top-k routing is discontinuous: BF16 WY vs sequential rounding can
    # change a later layer's route, so max-element relative error is unsuitable.
    # Use a bounded whole-tensor RMS error, alongside a strict FP32 check below.
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise FloatingPointError(f"{label}: non-finite tensor")
    delta = actual.float() - expected.float()
    relative = delta.norm() / expected.float().norm().clamp_min(1e-12)
    if relative > 0.05:
        raise AssertionError(f"{label}: BF16 relative L2 drift {relative.item():.6f} exceeds 5%")
    return delta.abs().max().item(), relative.item()


def check(cfg, device="cuda", length=65):
    if cfg["arch_type"] == "gl_sdm" and cfg.get("gl_memory_backend", "torch") == "urm":
        if cfg.get("gl_chunk_size", 1) > 1:
            from gl_sdm.memory.backends import urm, commit
            name = "routed_snapshot_read"
            with patch.object(urm, name, wraps=getattr(urm, name)) as reads, patch.object(commit, "commit", wraps=commit.commit) as commits:
                result = _check(cfg, device, length)
                if not reads.call_count or not commits.call_count:
                    raise AssertionError("chunk check must execute URM route/read and native commit")
                result["native_calls"] = {"urm_route_read": reads.call_count, "commit": commits.call_count}
                result["memory_backend"] = "urm"
                return result
        from gl_sdm.memory.backends import token, commit
        with ExitStack() as stack:
            observed = {name: stack.enter_context(patch.object(commit if name == "commit" else token, name,
                wraps=getattr(commit if name == "commit" else token, name)))
                        for name in ("route", "read", "propose", "commit")}
            result = _check(cfg, device, length)
            if any(spy.call_count == 0 for spy in observed.values()):
                raise AssertionError("GL-SDM verification did not execute every native memory operator")
            result["native_calls"] = {name: spy.call_count for name, spy in observed.items()}
            result["memory_backend"] = "urm"
            return result
    if cfg["arch_type"] != "sdm":
        return _check(cfg, device, length)
    from gl_sdm.baselines.upstream import sdm_layer, verified_import
    sdm_layer()
    sparse = verified_import("sdm", "lingua.sparse_delta_memory.cuda.sparse_ip_cuda")
    gather = verified_import("sdm", "lingua.sparse_delta_memory.cuda.warp_cooperative_gather_cuda")
    with patch.object(sparse, "sparse_ip_gated_fwd_sorted", wraps=sparse.sparse_ip_gated_fwd_sorted) as ip, patch.object(gather, "warp_cooperative_gather", wraps=gather.warp_cooperative_gather) as wc:
        result = _check(cfg, device, length)
        if ip.call_count == 0 or (cfg.get("dtype") == "bfloat16" and wc.call_count == 0):
            raise AssertionError("SDM verification did not execute the required CUDA extensions")
        result["cuda_calls"] = {"sparse_inner_product": ip.call_count, "warp_cooperative_gather": wc.call_count}
        return result


def _check(cfg, device="cuda", length=65):
    torch.manual_seed(1234)
    model = create_model(cfg).to(device)
    oracle = copy.deepcopy(model)
    x = torch.randint(cfg["vocab_size"], (2, length), device=device)
    targets = torch.randint(cfg["vocab_size"], x.shape, device=device)
    h, _, _ = model.hidden(x)
    logits = model.head(h)
    with oracle.reference():
        rh, _, _ = oracle.hidden(x)
        expected = oracle.head(rh)
    bf16_sdm = cfg["arch_type"] == "sdm" and cfg.get("dtype") == "bfloat16"
    bf16_chunk = cfg["arch_type"] == "gl_sdm" and cfg.get("gl_chunk_size", 1) > 1 and cfg.get("dtype") == "bfloat16"
    errors = {}
    if bf16_sdm:
        errors["logits_max_abs"], errors["logits_relative_l2"] = compare_bf16_sdm(logits, expected, "reference logits")
    else:
        errors["logits_max_abs"] = compare(logits, expected, "reference logits", atol=5e-4 if cfg.get("dtype") == "float32" else 0.03, rtol=0.003 if cfg.get("dtype") == "float32" else 0.03)
        errors["logits_relative_l2"] = ((logits.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg["vocab_size"]), targets.flatten())
    ref_loss = torch.nn.functional.cross_entropy(expected.reshape(-1, cfg["vocab_size"]), targets.flatten())
    loss.backward()
    ref_loss.backward()
    for (name, p), (rname, r) in zip(model.named_parameters(), oracle.named_parameters(), strict=True):
        assert name == rname
        if p.grad is None or r.grad is None:
            raise AssertionError(f"missing gradient for {name}")
        compare(p.grad, r.grad, f"reference gradient {name}", atol=1e-5 if cfg.get("dtype") == "float32" else (0.004 if bf16_sdm else 0.002), rtol=0.005 if cfg.get("dtype") == "float32" else 0.12)
    errors["gradient_max_abs"] = max((p.grad.float() - r.grad.float()).abs().max().item() for p, r in zip(model.parameters(), oracle.parameters()))
    model.eval()
    before_inference = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.inference_mode():
        full = model.head(model.hidden(x)[0])
        cache = model.new_cache(2)
        # Nonaligned, multi-token continuation exercises partition-safe padding
        # and offset causal masks, followed by actual single-token decode.
        first, cache = model.prefill(x[:, :17], cache)
        second, cache = model.prefill(x[:, 17:-2], cache)
        third, cache = model.decode(x[:, -2:-1], cache)
        fourth, cache = model.decode(x[:, -1:], cache)
        continued = torch.cat((first, second, third, fourth), 1)
        if bf16_sdm:
            errors["continuation_max_abs"], errors["continuation_relative_l2"] = compare_bf16_sdm(continued, full, "prefill/decode continuation")
        elif bf16_chunk:
            # Changing GEMM/SDPA batch shapes can change a BF16 route even with
            # the pure PyTorch backend. Bound this separately and independently
            # require strict FP32 continuation below; output/gradient reference
            # limits are unchanged.
            delta = continued.float() - full.float()
            relative = delta.norm() / full.float().norm().clamp_min(1e-12)
            if not torch.isfinite(relative) or relative > 0.01:
                raise AssertionError(f"BF16 chunk continuation relative L2 {relative.item():.6f} exceeds 1%")
            errors["continuation_max_abs"] = delta.abs().max().item()
            errors["continuation_relative_l2"] = relative.item()
        else:
            errors["continuation_max_abs"] = compare(continued, full, "prefill/decode continuation", atol=1e-4 if cfg.get("dtype") == "float32" else 0.05, rtol=0.003 if cfg.get("dtype") == "float32" else 0.05)
        fresh, _ = model.prefill(x)
        if bf16_sdm:
            errors["request_reset_max_abs"], errors["request_reset_relative_l2"] = compare_bf16_sdm(fresh, full, "request reset")
        else:
            compare(fresh, full, "request reset", atol=1e-4, rtol=0.003)
        mutated = x.clone()
        mutated[:, length // 2:] = torch.randint(cfg["vocab_size"], mutated[:, length // 2:].shape, device=device)
        causal = model.head(model.hidden(mutated)[0])
        if bf16_sdm:
            errors["causality_max_abs"], errors["causality_relative_l2"] = compare_bf16_sdm(causal[:, :length // 2], full[:, :length // 2], "causality")
        else:
            compare(causal[:, :length // 2], full[:, :length // 2], "causality", atol=1e-4 if cfg.get("dtype") == "float32" else 0.002, rtol=0.002)
        for name, p in model.named_parameters():
            compare(p, before_inference[name], f"inference mutated parameter {name}", atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            save(model, directory)
            reloaded, _ = load(directory, device)
            reloaded.eval()
            for (name, p), (rname, r) in zip(model.named_parameters(), reloaded.named_parameters(), strict=True):
                assert name == rname
                compare(p, r, f"strict checkpoint parameter {name}", atol=0, rtol=0)
            restored_logits = reloaded.head(reloaded.hidden(x)[0])
            if bf16_sdm:
                errors["reload_max_abs"], errors["reload_relative_l2"] = compare_bf16_sdm(restored_logits, full, "checkpoint reload logits")
            else:
                compare(restored_logits, full, "checkpoint reload logits", atol=1e-4 if cfg.get("dtype") == "float32" else 0.002, rtol=0.002)
    if bf16_sdm:
        fp32_cfg = {**cfg, "dtype": "float32"}
        errors["fp32_check"] = check(fp32_cfg, device, length)
        errors["bf16_note"] = "5% relative L2 limit; strict FP32 check independently validates routing/recurrence integration"
    if bf16_chunk:
        errors["fp32_check"] = check({**cfg, "dtype": "float32"}, device, length)
        errors["bf16_note"] = "output/gradient limits unchanged; BF16 uses the production SDPA surround and bounds split continuation at 1% relative L2; strict FP32 independently checks dense attention and continuation"
    return {"arch_type": cfg["arch_type"], "status": "passed", "batch": 2, "length": length, **errors}
