import pytest
import torch
from gl_sdm.model import create_model
from gl_sdm.experiments.verify import check
from gl_sdm.experiments.evaluate import _blocks_forward, _chunked_loss
from gl_sdm.experiments.metrics import utilization


def config(arch="transformer", dtype="float32"):
    return dict(arch_type=arch, vocab_size=128, hidden_size=64, head_dim=32,
                num_hidden_layers=2, dtype=dtype, sdm_slots=64, sdm_reads=4,
                sdm_writes=4, sdm_chunk=16)


def test_transformer_contract_reference_and_continuation():
    check(config(), "cpu")
    model = create_model(config()).eval()
    x = torch.randint(128, (2, 17))
    y = torch.randint(128, (2, 17))
    with torch.inference_mode():
        loss, reg, align = model(x, y)
        head_loss, count = _chunked_loss(model, _blocks_forward(model, x), y, chunk=5)
    assert count == y.numel()
    assert reg == align == 0
    assert head_loss == pytest.approx(loss.item(), rel=1e-6)
    metrics = utilization(model, 34, 1.0, 70e12)
    assert metrics["mfu_6nd_pct"] == 100 * 6 * metrics["active_params"] * 34 / 70e12


@pytest.mark.parametrize("arch", ["sdm", "gdn2"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="production kernels need CUDA")
def test_upstream_model_reference_and_continuation(arch):
    check(config(arch, "bfloat16"))


def test_unknown_arch_fails():
    with pytest.raises(ValueError, match="unknown arch"):
        create_model(config("unknown"))
