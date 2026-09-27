"""ATMA's clean/junk CE and induction needle protocol, with complete counts."""
import math
import random
import torch
import torch.nn.functional as F
from .data import data_generator


def _blocks_forward(model, inputs):
    x = model.embed(inputs)
    for block in model.blocks:
        x, _, _ = block(x)
    return x


def _chunked_loss(model, x, targets, chunk=512):
    total, count = 0.0, 0
    for start in range(0, x.shape[1], chunk):
        logits = model.head(x[:, start:start + chunk])
        target = targets[:, start:start + chunk].reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite evaluation loss")
        total += loss.item()
        count += target.numel()
    return total, count


def select_long_docs(dataset_id, text_key, split, min_tokens, num_docs, tokenizer_name="gpt2"):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    docs = []
    for row in load_dataset(dataset_id, split=split, streaming=True):
        ids = tok.encode(row[text_key], add_special_tokens=False)
        if len(ids) >= min_tokens + 1:
            docs.append(torch.tensor(ids[:min_tokens + 1], dtype=torch.long))
        if len(docs) == num_docs:
            break
    if len(docs) != num_docs:
        raise ValueError(f"only {len(docs)} of {num_docs} requested long documents available")
    return docs


@torch.inference_mode()
def clean_perplexity(model, docs, lengths, device, loss_chunk=512):
    out, counts = {}, {}
    for length in lengths:
        total, n = 0.0, 0
        for doc in docs:
            if doc.numel() < length + 1:
                raise ValueError("clean document is shorter than requested prefix")
            buf = doc[:length + 1].to(device)
            x = _blocks_forward(model, buf[:-1][None])
            loss, count = _chunked_loss(model, x, buf[1:][None], loss_chunk)
            total, n = total + loss, n + count
        if not n:
            raise ValueError("empty clean evaluation")
        out[length], counts[length] = total / n, n
    return out, counts


@torch.inference_mode()
def junk_perplexity(model, val_data, lengths, num_seqs, device, loss_chunk=512):
    out, counts = {}, {}
    for length in lengths:
        gen = data_generator(val_data, length, length, device)
        total, n = 0.0, 0
        for _ in range(num_seqs):
            inputs, targets = next(gen)
            x = _blocks_forward(model, inputs)
            loss, count = _chunked_loss(model, x, targets, loss_chunk)
            total, n = total + loss, n + count
        out[length], counts[length] = total / n, n
    return out, counts


def _value_logits(model, inputs, n_last):
    x = _blocks_forward(model, inputs)
    logits = model.head(x[:, -n_last:])[0]
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite needle logits")
    return logits


@torch.inference_mode()
def needle_retrieval(model, haystack, distances, num_trials, vlen, device, seed=1234, tokenizer=None):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    rng = random.Random(seed)
    ce, acc = {d: 0.0 for d in distances}, {d: 0.0 for d in distances}
    base = 0.0
    for trial in range(num_trials):
        hay = haystack[trial % len(haystack)].tolist()
        key = rng.randint(10**6, 10**7 - 1)
        cue = tokenizer.encode(f" The access code for record {key} is")
        val = tokenizer.encode("".join(f" {rng.randint(0, 9)}" for _ in range(vlen)))
        needle = cue + val
        target = torch.tensor(val, device=device)
        def score(seq):
            inputs = torch.tensor(seq[:-1], device=device)[None]
            return _value_logits(model, inputs, len(val))
        logits = score([tokenizer.eos_token_id] + hay[:min(distances) + len(needle)] + cue + val)
        base += F.cross_entropy(logits, target).item()
        for distance in distances:
            if len(hay) < distance:
                raise ValueError("haystack shorter than requested needle distance")
            logits = score([tokenizer.eos_token_id] + needle + hay[:distance] + cue + val)
            ce[distance] += F.cross_entropy(logits, target).item()
            acc[distance] += (logits.argmax(-1) == target).float().mean().item()
    return {d: {"ce": ce[d] / num_trials, "acc": 100 * acc[d] / num_trials, "trials": num_trials} for d in distances}, base / num_trials


@torch.inference_mode()
def run_eval(model, cfg, device, docs=None):
    was_training = model.training
    model.eval()
    try:
        lengths = cfg["eval_lengths"]
        result = {}
        result["junk_ppl"], result["junk_tokens"] = junk_perplexity(model, cfg["val_data"], lengths, cfg.get("num_eval_docs", 16), device)
        result["junk_perplexity"] = {k: math.exp(v) for k, v in result["junk_ppl"].items()}
        if cfg.get("clean_dataset") or docs is not None:
            distances = cfg["needle_distances"]
            need = max(max(lengths), max(distances) + 64)
            docs = docs if docs is not None else select_long_docs(cfg["clean_dataset"], "text", "train", need, cfg.get("num_eval_docs", 16), cfg.get("tokenizer_name", "gpt2"))
            result["clean_ppl"], result["clean_tokens"] = clean_perplexity(model, docs, lengths, device)
            result["clean_perplexity"] = {k: math.exp(v) for k, v in result["clean_ppl"].items()}
            result["needle"], result["needle_baseline"] = needle_retrieval(model, docs, distances, cfg.get("num_needle_trials", 16), cfg.get("needle_val_len", 5), device, cfg.get("eval_seed", 1234))
            result["num_clean_docs"] = len(docs)
        return result
    finally:
        model.train(was_training)
