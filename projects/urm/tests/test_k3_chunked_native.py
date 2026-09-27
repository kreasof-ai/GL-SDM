"""Compiler-selected native K3 scheduling, independent of model/front-end names.

The second admission client is a supplied-route cache: pre-update queries,
terminal-state read-only probes and linked update transactions. It has no
product-key router, uses nonsquare banks and unequal read/write widths.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from urm.compiler.common.diagnostics import CompilerError
from urm.compiler.pipeline import CompilationIntent, ScheduleParams, UrmCompiler, compile_graph
from urm.compiler.rewrite.proof import EquivalenceClass
from urm.compiler.rewrite.rules import PHYSICAL_REPARAMETERIZATIONS
from urm.compiler.select.anchors import (
    AnchorRegistry, NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME, TRUSTED_ANCHORS,
    make_sparse_state_mixer_selector, sparse_state_chunked_eligible, sparse_state_launch_schedule,
)
from urm.ir.program import (
    DType, SparseReadTiming, SparseStateExecutionMode, SparseStateOperation,
    sparse_state_mixer_program,
)

try:
    import torch
except ImportError:
    torch = None
gpu = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA required")


def _program(**changes):
    config = dict(name="supplied_route_cache", parallel=2, sequence=41,
                  slots_per_partition=67, value_dim=37, writes=3, reads=2,
                  dtype=DType.BFLOAT16, mode=SparseStateExecutionMode.TRAINING)
    config.update(changes)
    return sparse_state_mixer_program(**config)


def _compiler():
    anchor = next(a for a in TRUSTED_ANCHORS if a.name == NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME)
    registry = AnchorRegistry()
    registry.register(make_sparse_state_mixer_selector(anchor,
        support_probe=lambda _: SimpleNamespace(supported=True)))
    return UrmCompiler(anchors=registry)


def test_physical_contract_and_compiler_selection_are_name_independent():
    program = _program()
    compiled = _compiler().compile(program, intent=CompilationIntent.TRAINING)
    config = compiled.plan.steps[0].launch_config
    assert compiled.trace.anchors == (NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME,)
    assert config['schedule_family'] == 'guarded_chunked_decayed_delta'
    contract = PHYSICAL_REPARAMETERIZATIONS[config['reparameterization_rule']]
    assert contract.equivalence is EquivalenceClass.FLOATING_POINT
    assert contract.backward_covers(DType.BFLOAT16) and not contract.forward_only
    assert contract.preconditions and contract.preserved_effects
    renamed = replace(program, name='unrelated_client')
    renamed = renamed.replaced((replace(program.ops[0], name='cache_transition'),))
    other = _compiler().compile(renamed, intent=CompilationIntent.TRAINING)
    assert other.plan.steps[0].launch_config == config
    base = _compiler().compile(program, intent=CompilationIntent.TRAINING,
                              schedule_params=ScheduleParams(sparse_state_schedule='scan'))
    assert base.plan.steps[0].launch_config['schedule_family'] == 'partition_owned_ordered_token_scan'


@pytest.mark.parametrize('changes', [
    {'dtype': DType.FLOAT32}, {'mode': SparseStateExecutionMode.INFERENCE},
    {'slots_per_partition': 8192}, {'parallel': 4096, 'sequence': 2048},
    {'writes': 0, 'operation': SparseStateOperation.READ_ONLY,
     'read_timing': SparseReadTiming.CURRENT_STATE},
])
def test_typed_regimes_retain_base_provider(changes):
    spec = _program(**changes).ops[0].spec
    assert not sparse_state_chunked_eligible(spec)
    assert sparse_state_launch_schedule(spec)['schedule_family'] == 'partition_owned_ordered_token_scan'
    with pytest.raises(ValueError, match='envelope'):
        sparse_state_launch_schedule(spec, 'chunked')


def test_invalid_or_unqualified_schedule_hints_fail_before_execution():
    with pytest.raises(CompilerError):
        _compiler().compile(_program(), intent=CompilationIntent.TRAINING,
                            schedule_params=ScheduleParams(sparse_state_schedule='sdm'))
    with pytest.raises(CompilerError):
        _compiler().compile(_program(dtype=DType.FLOAT32), intent=CompilationIntent.TRAINING,
                            schedule_params=ScheduleParams(sparse_state_schedule='chunked'))
    assert not Path('architectures/sdm_chunked.py').exists()


def _operands(tokens=41, decay=-.1):
    torch.manual_seed(217)
    p, s, d, w, r = 2, 67, 37, 3, 2
    wi = torch.rand(p, tokens, s, device='cuda').argsort(-1)[..., :w].sort(-1).values.contiguous()
    ri = torch.rand(p, tokens, s, device='cuda').argsort(-1)[..., :r].sort(-1).values.contiguous()
    data = dict(memory=torch.randn(p, s, d, device='cuda', dtype=torch.bfloat16) * .05,
                read_weights=torch.randn(p, tokens, r, device='cuda', dtype=torch.bfloat16).softmax(-1),
                write_weights=torch.randn(p, tokens, w, device='cuda', dtype=torch.bfloat16).softmax(-1),
                values=torch.randn(p, tokens, d, device='cuda', dtype=torch.bfloat16) * .05,
                beta=torch.rand(p, tokens, 1, device='cuda', dtype=torch.bfloat16) * .5,
                log_decay=torch.full((p, tokens, 1), decay, device='cuda', dtype=torch.bfloat16))
    return dict(write_addresses=wi, read_addresses=ri, **data)


def _leaves(data):
    return {k: v.detach().clone().requires_grad_(v.is_floating_point()) for k, v in data.items()}


def _reference(data, spec):
    from urm.backends.torch.k3.sparse_state import sparse_delta_state
    return sparse_delta_state(**data, spec=spec)


def _assert_gradients(actual, expected, inputs, ref_inputs, weights):
    loss = sum((out.float() * weight.float()).sum() for out, weight in zip(actual, weights))
    ref_loss = sum((out.float() * weight.float()).sum() for out, weight in zip(expected, weights))
    keys = [k for k, v in inputs.items() if v.requires_grad]
    ag = torch.autograd.grad(loss, [inputs[k] for k in keys], allow_unused=True)
    eg = torch.autograd.grad(ref_loss, [ref_inputs[k] for k in keys], allow_unused=True)
    for key, a, b in zip(keys, ag, eg):
        a = torch.zeros_like(inputs[key]) if a is None else a
        b = torch.zeros_like(ref_inputs[key]) if b is None else b
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), key
        if b.float().norm() < 1e-8:
            assert a.float().norm() < 1e-8, key
        else:
            error = (a.float() - b.float()).norm() / b.float().norm()
            assert error < .04, (key, float(error))


@gpu
@pytest.mark.parametrize('timing', [SparseReadTiming.BEFORE_UPDATE, SparseReadTiming.AFTER_UPDATE])
@pytest.mark.parametrize('terminal_only', [False, True])
def test_public_native_state_outputs_and_all_cotangents(timing, terminal_only, monkeypatch):
    import urm.backends.triton.k3.sparse_state as native
    program = _program(read_timing=timing)
    plan = compile_graph(program, target='native', intent=CompilationIntent.TRAINING)
    inputs, reference = _leaves(_operands()), _leaves(_operands())
    snapshot = inputs['memory'].detach().clone()
    calls = []
    original = native.chunked_sparse_state_update
    def traced(*args, **kwargs):
        calls.append(kwargs['read_before_update'])
        return original(*args, **kwargs)
    monkeypatch.setattr(native, 'chunked_sparse_state_update', traced)
    result = plan.execute(**inputs)
    actual = result['readings'], result['updated_memory']
    expected = _reference(reference, program.ops[0].spec)
    assert calls == [timing is SparseReadTiming.BEFORE_UPDATE]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=.02, rtol=.03)
    weights = [torch.randn_like(a) for a in actual]
    if terminal_only:
        weights[0].zero_()
    _assert_gradients(actual, expected, inputs, reference, weights)
    torch.testing.assert_close(inputs['memory'].detach(), snapshot, atol=0, rtol=0)


@gpu
def test_supplied_route_cache_continuation_and_read_only_probe(monkeypatch):
    import urm.backends.triton.k3.sparse_state as native
    # Model recipes carry runtime P/T dimensions. A symbolic recipe's placeholder
    # dimensions must not hard-code the actual training schedule or state shape.
    program = _program(parallel=1, sequence=1, read_timing=SparseReadTiming.BEFORE_UPDATE)
    plan = compile_graph(program, target='native', intent=CompilationIntent.TRAINING)
    probe = compile_graph(_program(sequence=1, writes=0, operation=SparseStateOperation.READ_ONLY,
                                  read_timing=SparseReadTiming.CURRENT_STATE),
                          target='native', intent=CompilationIntent.TRAINING)
    calls = []
    original = native.chunked_sparse_state_update
    def traced(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(native, 'chunked_sparse_state_update', traced)
    inputs, reference = _leaves(_operands()), _leaves(_operands())
    first = plan.execute(**inputs)
    saved_snapshot = first['updated_memory'].detach().clone()
    expected_first = _reference(reference, program.ops[0].spec)
    second_inputs = {**inputs, 'memory': first['updated_memory']}
    ref_second_inputs = {**reference, 'memory': expected_first[1]}
    second = plan.execute(**second_inputs)
    expected_second = _reference(ref_second_inputs, program.ops[0].spec)
    read_indices, weights = inputs['read_addresses'][:, :1].contiguous(), inputs['read_weights'][:, :1].contiguous()
    query = probe.execute(memory=second['updated_memory'], read_addresses=read_indices, read_weights=weights)
    expected_query = (expected_second[1].gather(1, read_indices[:, 0, :, None].expand(2, 2, 37)).float()
                      * reference['read_weights'][:, 0, :, None].float()).sum(1).unsqueeze(1).bfloat16()
    torch.testing.assert_close(query['readings'], expected_query, atol=.02, rtol=.03)
    actual = query['readings'], second['updated_memory']
    expected = expected_query, expected_second[1]
    _assert_gradients(actual, expected, inputs, reference, [torch.randn_like(a) for a in actual])
    assert len(calls) == 2
    torch.testing.assert_close(first['updated_memory'].detach(), saved_snapshot, atol=0, rtol=0)


@gpu
@pytest.mark.parametrize('case', ['scan', 'growth', 'gate', 'strong_decay', 'short', 'workspace', 'cache'])
def test_declared_base_schedule_and_runtime_guards_are_honored(case, monkeypatch):
    import urm.backends.triton.k3.sparse_state as native
    inputs = _leaves(_operands(tokens=7 if case == 'short' else 41,
                               decay=-1000 if case == 'strong_decay' else -.1))
    if case == 'growth':
        inputs['log_decay'] = -inputs['log_decay']
    if case == 'gate':
        inputs['beta'] = inputs['beta'] + 1
    program = _program(sequence=inputs['values'].shape[1])
    plan = compile_graph(program, target='native', intent=CompilationIntent.TRAINING,
                         schedule_params=ScheduleParams(sparse_state_schedule='scan' if case == 'scan' else 'auto'))
    if case == 'workspace':
        monkeypatch.setattr(native, 'sparse_state_chunked_eligible', lambda _: False)
    def rejected(*args, **kwargs):
        if case == 'cache':
            raise torch._dynamo.exc.FailOnRecompileLimitHit('bounded compilation cache exhausted')
        pytest.fail('chunk schedule must not execute outside its declared regime')
    monkeypatch.setattr(native, 'chunked_sparse_state_update', rejected)
    out = plan.execute(**inputs)
    expected = _reference(inputs, program.ops[0].spec)
    for a, b in zip((out['readings'], out['updated_memory']), expected):
        torch.testing.assert_close(a, b, atol=.02, rtol=.03)


@gpu
def test_tampered_schedule_fails_before_provider_execution():
    from urm.runtime.bind import BoundGraphPlan, PlanBindingError
    plan = compile_graph(_program(), target='native', intent=CompilationIntent.TRAINING)
    step = plan.compilation.plan.steps[0]
    bad = replace(step, launch_config={**step.launch_config, 'chunk_size': 1024})
    compilation = replace(plan.compilation, plan=replace(plan.compilation.plan, steps=(bad,)))
    with pytest.raises(PlanBindingError, match='physical schedule'):
        BoundGraphPlan(compilation).execute(**_operands())


@gpu
def test_non_autograd_chunk_execution_preserves_in_place_state_and_preallocated_output():
    from urm.backends.triton.k3.sparse_state import TritonSparseStateMixerBackend
    from urm.runtime.certification import CertifiedSparseStateRoutes, SparseState
    inputs = _operands()
    spec = _program().ops[0].spec
    expected = _reference({**inputs, 'memory': inputs['memory'].clone()}, spec)
    routes = CertifiedSparseStateRoutes.certify(spec, inputs['read_addresses'], inputs['read_weights'],
        write_indices=inputs['write_addresses'], write_weights=inputs['write_weights'])
    backend = TritonSparseStateMixerBackend(spec)
    prepared = backend.prepare(routes, values=inputs['values'], beta=inputs['beta'],
                               log_decay=inputs['log_decay'])
    out = torch.empty_like(expected[0])
    pointer = inputs['memory'].data_ptr()
    with torch.no_grad():
        readings, state = backend.execute(SparseState(inputs['memory']), prepared, out=out)
    assert readings.data_ptr() == out.data_ptr()
    assert state.memory.data_ptr() == pointer and state.sequence_length == 41
    for a, b in zip((readings, state.memory), expected):
        torch.testing.assert_close(a, b, atol=.02, rtol=.03)
