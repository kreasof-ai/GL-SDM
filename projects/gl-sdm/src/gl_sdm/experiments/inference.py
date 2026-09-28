"""Request-local prefill/decode using each baseline's production state."""
import torch


@torch.inference_mode()
def generate(model, inputs, max_new_tokens=64, temperature=0, top_k=0, eos_token_id=None):
    if temperature < 0 or top_k < 0 or max_new_tokens < 0:
        raise ValueError("sampling arguments must be nonnegative")
    model.eval()
    output = [inputs]
    if max_new_tokens == 0:
        return inputs
    logits, cache = model.prefill(inputs)
    finished = torch.zeros(inputs.shape[0], dtype=torch.bool, device=inputs.device)
    for step in range(max_new_tokens):
        scores = logits[:, -1].float()
        if not torch.isfinite(scores).all():
            raise FloatingPointError("non-finite generation logits")
        if temperature == 0:
            token = scores.argmax(-1, keepdim=True)
        else:
            scores = scores / temperature
            if top_k:
                threshold = scores.topk(min(top_k, scores.shape[-1])).values[:, -1:]
                scores = scores.masked_fill(scores < threshold, -torch.inf)
            token = torch.multinomial(scores.softmax(-1), 1)
        if eos_token_id is not None:
            token[finished] = eos_token_id
            finished |= token[:, 0] == eos_token_id
        output.append(token)
        if finished.all() or step == max_new_tokens - 1:
            break
        logits, cache = model.decode(token, cache)
    return torch.cat(output, 1)
