import json
import numpy as np
import pytest
import torch
from gl_sdm.data import data_generator, load_shard, available_steps
from gl_sdm.train import run, optimizers
from gl_sdm.checkpoint import load
from gl_sdm.inference import generate
from gl_sdm.evaluate import run_eval, needle_retrieval, _chunked_loss
from test_models import config


def shard(path, vocab=128):
    tokens = np.arange(1024, dtype=np.uint16) % vocab
    header = np.zeros(256, dtype=np.int32)
    header[:4] = [20240520, 1, len(tokens), 2]
    path.write_bytes(header.tobytes() + tokens.tobytes())


@pytest.mark.parametrize("architecture,device", [("transformer", "cpu"), ("gl_sdm", "cpu"),
    pytest.param("gl_sdm", "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_data_training_resume_eval_and_generation(tmp_path, architecture, device):
    path = tmp_path / "tokens.bin"
    shard(path)
    cfg = {**config(), "run_id": "test", "train_data": str(path), "val_data": str(path),
           "seq_len": 16, "batch_size": 32, "mbs": 1, "num_chunks": 1,
           "max_steps": 3, "val_tokens": 32, "val_freq": 1, "eval_lengths": [16, 32], "num_eval_docs": 2}
    if architecture == "gl_sdm":
        cfg.update(arch_type="gl_sdm", num_hidden_layers=1, gl_reasoning="adaptive", gl_max_steps=3,
                   gl_slots=16, gl_reads=2, gl_writes=2, auxiliary_loss_weight=0.001,
                   gl_memory_backend="urm" if device == "cuda" else "torch")
    generator = data_generator(str(path), 32, 16, device)
    x, y = next(generator)
    assert torch.equal(x.flatten()[1:], y.flatten()[:-1])
    assert available_steps(str(path), 32, 1) == 31
    events = []
    def emit(name, value):
        json.dumps(value, allow_nan=False)
        events.append((name, value))
    reference = run(cfg, device, tmp_path / "complete", emit)
    rows = next(value for name, value in events if name == "ABLATION_CURVE_JSON")
    assert rows[-1]["mfu"] is None if device == "cpu" else np.isfinite(rows[-1]["mfu"])
    assert rows[-1]["step_ms"] > 0 and rows[-1]["wall_s"] > 0
    # Interrupt immediately after step 1 checkpoint, then resume the exact
    # schedule/config and verify optimizer + RNG + data position restoration.
    class Interrupted(Exception):
        pass
    from unittest.mock import patch
    from gl_sdm import checkpoint
    original_save = checkpoint.save
    def stop_after_save(*args, **kwargs):
        original_save(*args, **kwargs)
        if args[3] == 1:
            raise Interrupted
    with patch("gl_sdm.checkpoint.save", stop_after_save), pytest.raises(Interrupted):
        run(cfg, device, tmp_path / "resume", emit)
    resumed = run(cfg, device, tmp_path / "resume", emit, resume=tmp_path / "resume")
    for p, r in zip(reference.parameters(), resumed.parameters()):
        torch.testing.assert_close(p, r, atol=0, rtol=0)
    restored, payload = load(tmp_path / "complete", device)
    assert payload["step"] == payload["data_batches"] == 3
    result = run_eval(restored, cfg, device)
    assert result["junk_tokens"] == {16: 32, 32: 64}
    assert all(np.isfinite(list(result["junk_perplexity"].values())))
    a = generate(restored, x[:1, :4], 5)
    b = generate(restored, x[:1, :4], 5)
    assert torch.equal(a, b) and a.shape[1] == 9


def test_optimizer_roles_cover_every_parameter_once():
    from gl_sdm.model import create_model
    model = create_model(config())
    opts = optimizers(model, {"optimizer": "atma_muon"}, "cpu")
    assigned = [p for o in opts for g in o.param_groups for p in g["params"]]
    assert len(assigned) == len(set(assigned)) == len(list(model.parameters()))


def test_bad_shard_and_nonfinite_eval_fail(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(bytes(1024))
    with pytest.raises(ValueError):
        load_shard(path)
    from gl_sdm.model import create_model
    model = create_model(config()).eval()
    model.proj.weight.data.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        _chunked_loss(model, torch.ones(1, 2, 64), torch.ones(1, 2, dtype=torch.long))


def test_needle_protocol_without_network():
    from gl_sdm.model import create_model
    class Tokenizer:
        eos_token_id = 0
        def encode(self, text):
            return [ord(c) % 128 for c in text]
    model = create_model(config()).eval()
    docs = [torch.arange(128)]
    needle, control = needle_retrieval(model, docs, [16, 32], 2, 3, "cpu", tokenizer=Tokenizer())
    assert all(row["trials"] == 2 for row in needle.values())
    assert np.isfinite(control)
