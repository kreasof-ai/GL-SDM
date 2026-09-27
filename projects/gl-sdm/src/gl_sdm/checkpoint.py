"""ATMA checkpoint directory and strict reload, plus resumable training state."""
import json
from pathlib import Path
import torch
from .model import create_model


def save(model, directory, optimizers=(), step=0, data_batches=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg
    state = {k.removeprefix("_orig_mod."): v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {"model": state, "step": step, "data_batches": data_batches,
               "optimizers": [o.state_dict() for o in optimizers], "rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(payload, directory / "weights.tmp")
    (directory / "weights.tmp").replace(directory / "weights.pt")
    for name in ("config.json", "run_config.json"):
        (directory / name).write_text(json.dumps(cfg, indent=2) + "\n")
    (directory / "tokenizer.json").write_text(json.dumps({"tokenizer_name": cfg.get("tokenizer_name", "gpt2")}) + "\n")


def load(directory, device="cuda"):
    directory = Path(directory)
    cfg = json.loads((directory / "config.json").read_text())
    model = create_model(cfg).to(device)
    payload = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload
