"""Static fixed-depth training graph; optimizer and finite checks stay explicit."""
import torch
import weakref


class TrainingGraph:
    def __init__(self, model, inputs, targets, cfg):
        if inputs.device.type != "cuda" or cfg.get("arch_type") != "gl_sdm" or cfg.get("gl_reasoning") != "fixed":
            raise ValueError("gl_cuda_graph requires CUDA and fixed reasoning depth")
        if cfg.get("reg_mode", "baseline") != "baseline":
            raise ValueError("training graphs currently require baseline regularization")
        self.inputs, self.targets = inputs.clone(), targets.clone()
        self.model, self.cfg = weakref.proxy(model), cfg
        mbs = cfg["mbs"]
        if inputs.shape[0] % mbs:
            raise ValueError("sequences per token batch must be divisible by mbs")
        def compute():
            losses, depths = [], []
            for i in range(0, inputs.shape[0], mbs):
                loss, reg, align = model(self.inputs[i:i + mbs], self.targets[i:i + mbs])
                alpha = cfg.get("sigr_alpha", 0.0)
                objective = (1 - alpha) * loss + alpha * reg + cfg.get("auxiliary_loss_weight", 0.0) * align
                objective.backward()
                losses.append(loss.detach())
                depths.append(model.blocks[0].last_depth)
            return torch.stack(losses).sum(), torch.cat(depths).flatten()
        # Warm kernels/autograd on a side stream; no optimizer step or parameter
        # mutation. Capturing also does no optimizer step, so data is not consumed.
        stream = torch.cuda.Stream(device=inputs.device)
        stream.wait_stream(torch.cuda.current_stream(inputs.device))
        rng = torch.cuda.get_rng_state(inputs.device)
        cpu_rng = torch.get_rng_state()
        with torch.cuda.stream(stream):
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                compute()
        torch.cuda.current_stream(inputs.device).wait_stream(stream)
        model.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.loss, self.depth = compute()
        torch.cuda.set_rng_state(rng, inputs.device)
        torch.set_rng_state(cpu_rng)
        # Keep the gradient storage captured by backward alive across updates.
        self.gradients = [p.grad for p in model.parameters()]

    def update(self, opts, inputs, targets):
        if inputs.shape != self.inputs.shape or targets.shape != self.targets.shape:
            raise ValueError("training graph input shapes must remain fixed")
        self.inputs.copy_(inputs)
        self.targets.copy_(targets)
        for p, gradient in zip(self.model.parameters(), self.gradients, strict=True):
            p.grad = gradient
        self.graph.replay()
        if not torch.isfinite(self.loss):
            raise FloatingPointError("non-finite training loss")
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite training gradient; run aborted")
        for opt in opts:
            opt.step()
        self.model.last_training_depth = self.depth
        # The captured first microbatch overwrites gradients; setting .grad=None
        # here would disconnect the optimizer from the captured buffers.
        return self.loss.item() / targets.numel()
