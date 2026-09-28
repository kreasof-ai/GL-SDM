"""Compile pure tensor helpers across training and serving configurations."""
import torch


def compiled(function):
    implementation = torch.compile(function, fullgraph=True,
        options={"emulate_precision_casts": True})
    def invoke(*args):
        # Experiments deliberately vary dtype, dimensions, ACT/fixed mode,
        # gradient mode and serving shape. Keep a bounded collection of compiled
        # variants; fullgraph still raises rather than silently dropping compile.
        with torch._dynamo.config.patch(recompile_limit=64):
            return implementation(*args)
    return invoke
