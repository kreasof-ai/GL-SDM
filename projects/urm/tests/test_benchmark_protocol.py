"""Regression checks for measurement integrity and the external schedules."""
import copy
import math

import pytest
import torch

from train.harness import TrainConfig
from train.model import MixerSpec, URMDecoderLM
from train.optimizer import build_optimizers
from train.report import comparison_reason, render, verified


def test_compile_wrapper_preserves_optimizer_roles():
    spec = MixerSpec("identity", lambda *a: torch.nn.Linear(a[0], a[0]), None, False, False)
    model = URMDecoderLM(vocab_size=16, sequence_length=8, layers=1, width=8,
                         num_heads=1, head_dim=8, mixer=spec)
    eager = build_optimizers(model)
    compiled = build_optimizers(torch.compile(model))
    def roles(optimizers):
        return {id(p): (type(o), group['lr']) for o in optimizers
                for group in o.param_groups for p in group['params']}
    assert roles(eager) == roles(compiled)
    assert roles(compiled)[id(model.token.weight)][1] == 0.02
    assert roles(compiled)[id(model.blocks[0].mlp.up.weight)][1] == 0.005


def test_batched_muon_preserves_independent_updates_and_momentum():
    from train.optimizer import Muon, muon_update
    torch.manual_seed(3)
    params = [torch.nn.Parameter(torch.randn(8, 16)) for _ in range(3)]
    expected = [p.detach().clone() for p in params]
    moments = [torch.zeros_like(p) for p in params]
    optimizer = Muon(params, lr=.005, weight_decay=.01)
    for _ in range(2):
        grads = [torch.randn_like(p) for p in params]
        for p, grad in zip(params, grads):
            p.grad = grad.clone()
        for p, grad, momentum in zip(expected, grads, moments):
            update = muon_update(grad.clone(), momentum)
            p.mul_(1 - .005 * .01).add_(update, alpha=-.005)
        optimizer.step()
    for p, expected_p, momentum in zip(params, expected, moments):
        torch.testing.assert_close(p, expected_p, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(optimizer.state[p]['momentum'], momentum, atol=0, rtol=0)


@pytest.mark.parametrize("overrides", [{"microbatch_tokens": 0}, {"steps": 0},
                                      {"batch_tokens": 768, "microbatch_tokens": 256}])
def test_invalid_training_budgets_fail_early(overrides):
    with pytest.raises(ValueError):
        TrainConfig(mixer="dense_attention", **overrides)


def _record():
    return dict(mixer="example", measurement_version=2, steps=2, loss_trace=[2., 1.],
                final_loss=1., mfu=.2, checkpoint_aligned=True, wallclock_s=1.,
                throughput_tokens_s=100., peak_memory_gib=1., params=100,
                source_fingerprint="abc", config={
                    key: 1 for key in __import__('train.report', fromlist=['PROTOCOL_KEYS']).PROTOCOL_KEYS},
                environment={"device": "A10G"}, baseline_tier="production-kernel")


def test_invalid_or_mismatched_results_never_enter_comparison():
    good = _record()
    assert verified(good)
    assert comparison_reason(good, good) is None
    for key, value in [('final_loss', math.nan), ('checkpoint_aligned', False),
                       ('measurement_version', 1), ('loss_trace', [1., math.inf])]:
        bad = {**good, key: value}
        assert not verified(bad)
    for key, value in [('baseline_tier', 'reference-implementation'),
                       ('source_fingerprint', 'different'), ('microbatch_fallback', 1024)]:
        assert comparison_reason(good, {**good, key: value})
    changed = copy.deepcopy(good)
    changed['config']['microbatch_tokens'] = 1024
    assert 'microbatch_tokens' in comparison_reason(good, changed)
    changed['baseline_tier'] = 'reference-implementation'
    report = render({'example': good}, {'example': changed})
    comparison = report.split('## Production-kernel measurements')[1].split('### Production adapters')[0]
    assert '| example |' not in comparison


@pytest.mark.parametrize('target', ['reference', 'native'])
def test_block_solve_forward_and_gradients_match_independent_solver(target):
    if target == 'native' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from architectures.deltaformer import DeltaFormerLayer
    from architectures.triangular_schedule import solve_in_blocks
    device = 'cuda' if target == 'native' else 'cpu'
    torch.manual_seed(14)
    probs = (torch.randn(2, 2, 65, 65, device=device) * .01).requires_grad_()
    beta = torch.rand(2, 2, 65, device=device, requires_grad=True)
    value = torch.randn(2, 2, 65, 8, device=device, requires_grad=True)
    plan = DeltaFormerLayer(2, 8, target=target, intent='training')._solve
    actual = solve_in_blocks(plan, probs, beta, value, 'u')
    matrix = torch.eye(65, device=device) + beta[..., None] * probs.tril(-1)
    expected = torch.linalg.solve_triangular(matrix, value, upper=False, unitriangular=True)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    grad = torch.randn_like(value)
    a_grads = torch.autograd.grad(actual, (probs, beta, value), grad, retain_graph=True)
    e_grads = torch.autograd.grad(expected, (probs, beta, value), grad)
    for a, e in zip(a_grads, e_grads):
        torch.testing.assert_close(a, e, atol=5e-6, rtol=5e-5)


def test_strict_softmax_empty_row_has_finite_gradients():
    from architectures.deltaformer import strict_tril_softmax
    q = torch.randn(1, 2, 8, 8, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    p = strict_tril_softmax(q, k)
    p.square().sum().backward()
    assert torch.equal(p[..., 0, :], torch.zeros_like(p[..., 0, :]))
    assert q.grad.isfinite().all() and k.grad.isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_nonfinite_loss_is_a_failed_run():
    from train.harness import train
    from train.data import synthetic_generator
    class Nonfinite(torch.nn.Module):
        def forward(self, x):
            return x * float('nan')
    spec = MixerSpec('nonfinite', lambda *a: Nonfinite(), None, False, False)
    cfg = TrainConfig(mixer='nonfinite', vocab_size=16, layers=1, width=128,
                      num_heads=2, sequence_length=8, batch_tokens=16,
                      microbatch_tokens=16, steps=1, compile_model=False)
    with pytest.raises(FloatingPointError, match='non-finite training loss'):
        train(cfg, spec, synthetic_generator(16, 8, 16), device='cuda')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('row', ['comba', 'dplr', 'gdn2', 'gated_delta_product',
                               'rwkv7', 'mamba2', 'log_linear_attention', 'log_linear_mamba2', 'raven', 'tda'])
def test_matched_production_kernel_forward_and_input_gradient(row):
    from train.registry import get_mixer
    from train.upstream import UPSTREAM_BUILDERS
    torch.manual_seed(7)
    native = get_mixer(row).builder(128, 2, 64, 'training', 'native').cuda()
    upstream = UPSTREAM_BUILDERS[row]()(128, 2, 64, 'training', 'reference').cuda()
    upstream.load_state_dict(native.state_dict())
    calls = []
    for module in upstream.modules():
        if (module.__class__.__module__ == 'train.production'
                or module.__class__.__name__ == '_TDAUpstream') and hasattr(module, '_kernel'):
            original = module._kernel
            def record_call(*args, _original=original, **kwargs):
                calls.append(True)
                return _original(*args, **kwargs)
            module._kernel = record_call
    a = torch.randn(2, 65, 128, device='cuda', requires_grad=True)
    b = a.detach().clone().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        native_out, upstream_out = native(a), upstream(b)
    assert calls, "production kernel must execute; an unused adapter is not a baseline"
    torch.testing.assert_close(native_out.float(), upstream_out.float(), rtol=.03, atol=.02)
    cotangent = torch.randn_like(native_out)
    native_out.backward(cotangent)
    upstream_out.backward(cotangent)
    assert a.grad.isfinite().all() and b.grad.isfinite().all()
    torch.testing.assert_close(a.grad, b.grad, rtol=.06, atol=.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('row', ['pattention', 'tucker'])
def test_wide_external_schedule_preserves_output_and_gradients(row):
    if row == 'pattention':
        from architectures.pattention import PattentionLayer
        build = lambda target: PattentionLayer(256, 512, 8, target=target, intent='training')
        width = 256
    else:
        from architectures.tucker_attention import TuckerAttentionLayer
        build = lambda target: TuckerAttentionLayer(128, 2, (256, 2, 256), (256, 2, 256),
                                                    target=target, intent='training')
        width = 128
    torch.manual_seed(23)
    native, reference = build('native').cuda(), build('reference').cuda()
    reference.load_state_dict(native.state_dict())
    a = torch.randn(2, 8, width, device='cuda', requires_grad=True)
    b = a.detach().clone().requires_grad_()
    out, expected = native(a), reference(b)
    torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-3)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(a.grad, b.grad, atol=3e-4, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_attnres_indexed_spatial_layout_matches_dense_depth_reduction():
    from architectures.attnres import AttnResLayer
    torch.manual_seed(18)
    residuals = [torch.randn(2, 33, 768, device='cuda', requires_grad=True) for _ in range(3)]
    reference_residuals = [r.detach().clone().requires_grad_() for r in residuals]
    query = (torch.randn(768, device='cuda') * .1).requires_grad_()
    reference_query = query.detach().clone().requires_grad_()
    weight = torch.ones(768, device='cuda')
    native = AttnResLayer(3, target='native', intent='training')
    reference = AttnResLayer(3, target='reference', intent='training')
    out = native(query, residuals, weight, weight)
    expected = reference(reference_query, reference_residuals, weight, weight)
    torch.testing.assert_close(out, expected, atol=5e-6, rtol=5e-5)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(query.grad, reference_query.grad, atol=2e-4, rtol=2e-4)
    for a, b in zip(residuals, reference_residuals):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_pattention_softmax_composition_preserves_routing_scale_and_parameter_gradients():
    from architectures.pattention import PattentionLayer
    torch.manual_seed(31)
    native = PattentionLayer(256, 129, 11, target='native', intent='training').cuda()
    reference = PattentionLayer(256, 129, 11, target='reference', intent='training').cuda()
    reference.load_state_dict(native.state_dict())
    a = (torch.randn(2, 9, 256, device='cuda') * .1).requires_grad_()
    b = a.detach().clone().requires_grad_()
    selected = torch.tensor([0, 2, 3, 5, 7, 8, 10], device='cuda')
    out = native(a, selected, scale=2.5)
    expected = reference(b, selected, scale=2.5)
    torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-3)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(a.grad, b.grad, atol=3e-4, rtol=3e-3)
    for p, expected_p in zip(native.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, expected_p.grad, atol=1e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_tokenformer_autocast_full_block_output_and_gradient_parity():
    from architectures.pattention import TokenformerBlock
    torch.manual_seed(44)
    native = TokenformerBlock(128, 2, 64, 11, 7, target='native', intent='training').cuda()
    reference = TokenformerBlock(128, 2, 64, 11, 7, target='reference', intent='training').cuda()
    reference.load_state_dict(native.state_dict())
    a = torch.randn(2, 17, 128, device='cuda', requires_grad=True)
    b = a.detach().clone().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out, expected = native(a), reference(b)
    torch.testing.assert_close(out, expected, atol=.01, rtol=.05)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    assert a.grad.isfinite().all()
    torch.testing.assert_close(a.grad, b.grad, atol=.02, rtol=.08)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_path_long_block_schedule_forward_and_gradient_parity():
    from architectures.path_attention import PaTHAttentionLayer
    torch.manual_seed(55)
    operands = [torch.randn(2, 65, 2, 8, device='cuda') * .1 for _ in range(4)]
    operands[3] = torch.nn.functional.normalize(operands[3], dim=-1)
    operands += [torch.rand(2, 65, 2, device='cuda') * .5,
                 -torch.rand(2, 65, 2, device='cuda') * .05]
    operands = [x.requires_grad_() for x in operands]
    expected_operands = [x.detach().clone().requires_grad_() for x in operands]
    native = PaTHAttentionLayer(2, 8, target='native', intent='training')
    reference = PaTHAttentionLayer(2, 8, target='reference', intent='training')
    out = native(*operands, scale=8 ** -.5)
    expected = reference(*expected_operands, scale=8 ** -.5)
    torch.testing.assert_close(out, expected, atol=2e-5, rtol=2e-4)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    for actual, ref in zip(operands, expected_operands):
        assert actual.grad.isfinite().all()
        torch.testing.assert_close(actual.grad, ref.grad, atol=3e-5, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('empty_experts', [False, True])
def test_mom_shared_padding_matches_independent_packed_stream_gradients(empty_experts):
    from architectures.mom import MoMLayer
    torch.manual_seed(66)
    native = MoMLayer(64, 1, 64, 64, 4, 2, target='native', intent='training').cuda()
    reference = MoMLayer(64, 1, 64, 64, 4, 2, target='reference', intent='training').cuda()
    if empty_experts:
        with torch.no_grad():
            native.gate.weight.zero_()
    reference.load_state_dict(native.state_dict())
    a = torch.randn(2, 65, 64, device='cuda', requires_grad=True)
    b = a.detach().clone().requires_grad_()
    out = native(a)
    weights, indices = reference.gate(b).softmax(-1).topk(2, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    routed_weights = torch.zeros(2, 65, 4, device='cuda').scatter_add(2, indices, weights)
    expected = torch.zeros(130, 64, device='cuda')
    for expert in range(4):
        for batch in range(2):
            positions = (routed_weights[batch, :, expert] > 0).nonzero().flatten()
            if not positions.numel():
                continue
            values = reference.memories[expert](b[batch, positions][None])[0]
            values = values * routed_weights[batch, positions, expert, None]
            expected = expected.index_add(0, batch * 65 + positions, values)
    expected = expected.reshape_as(out)
    torch.testing.assert_close(out, expected, atol=2e-5, rtol=2e-4)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(a.grad, b.grad, atol=3e-5, rtol=3e-4)
    for actual, ref in zip(native.parameters(), reference.parameters()):
        if ref.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, ref.grad, atol=2e-4, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('layer_idx', [0, 1])
def test_samba_production_schedule_uses_matched_frontends_and_kernels(layer_idx):
    from train.registry import get_mixer
    from train.upstream import _samba_upstream_layer
    from train.production import Mamba2Production, _SDPAPlan
    torch.manual_seed(77)
    native = get_mixer('samba_attention').builder(layer_idx, 128, 2, 64, 'training', 'native').cuda()
    upstream = _samba_upstream_layer(layer_idx, 128, 2, 64, 'training', 'reference').cuda()
    upstream.load_state_dict(native.state_dict())
    owner = upstream if layer_idx == 0 else upstream._plan
    assert isinstance(owner, Mamba2Production if layer_idx == 0 else _SDPAPlan)
    calls, original = [], owner._kernel
    def record_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    owner._kernel = record_call
    a = torch.randn(2, 65, 128, device='cuda', requires_grad=True)
    b = a.detach().clone().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out, expected = native(a), upstream(b)
    assert calls
    torch.testing.assert_close(out.float(), expected.float(), atol=.02, rtol=.03)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    assert a.grad.isfinite().all() and b.grad.isfinite().all()
    # SSD and K2 round different bf16 intermediates. Bound both cancellation
    # near zero and the error across the full gradient, rather than only RMS.
    torch.testing.assert_close(a.grad, b.grad, atol=.05, rtol=.06)
    assert (a.grad - b.grad).norm() / b.grad.norm() < .01


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_raven_slot_duplication_preserves_all_operand_gradients():
    from train.production import RavenProduction
    from architectures.abc_gsa import _SlotAttentionBase
    torch.manual_seed(88)
    production = RavenProduction(128, 2, 64).cuda()
    reference = _SlotAttentionBase(2, 64, 64, 8, intent='training', target='reference').cuda()
    operands = [torch.randn(2, 65, 2, 64, device='cuda') * .1 for _ in range(3)]
    operands += [torch.rand(2, 65, 2, 8, device='cuda') * .1,
                 -torch.rand(2, 65, 2, 8, device='cuda') * .1]
    operands = [x.transpose(1, 2).contiguous().requires_grad_() for x in operands]
    copies = [x.detach().clone().requires_grad_() for x in operands]
    duplicated = [x.detach().clone().requires_grad_() for x in operands]
    duplicated_reference = _SlotAttentionBase(2, 64, 64, 16, intent='training', target='reference').cuda()
    out = production._mixer(*operands, .125)
    expected = reference._mixer(*copies, .125)
    expanded = duplicated_reference._mixer(
        *duplicated[:3], torch.cat((duplicated[3], duplicated[3]), -1),
        torch.cat((duplicated[4], duplicated[4]), -1), .125)
    torch.testing.assert_close(expanded, expected, atol=2e-7, rtol=2e-5)
    # The unmodified production kernel uses tensor-core approximations even
    # for fp32 inputs; verify the exact duplication law separately above.
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=5e-3)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    expanded.backward(grad)
    for actual, ref, doubled in zip(operands, copies, duplicated):
        assert actual.grad.isfinite().all()
        torch.testing.assert_close(actual.grad, ref.grad, atol=5e-4, rtol=5e-3)
        assert (actual.grad - ref.grad).norm() / ref.grad.norm() < .01
        torch.testing.assert_close(doubled.grad, ref.grad, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_tda_pinned_kernel_handles_strided_projection_inputs_and_gradients():
    from train.upstream import _TDAUpstream
    torch.manual_seed(99)
    layer = _TDAUpstream(128, 2, 64).cuda()
    source = [torch.randn(2, 65, 2, 64, device='cuda') * .3 for _ in range(3)]
    operands = [x.transpose(1, 2).requires_grad_() for x in source]
    copies = [x.detach().clone().requires_grad_() for x in operands]
    out = layer._call(*operands)
    q, k, v = copies
    positions = torch.arange(65, device='cuda')
    tau = (2 * (positions + 1).float().log() / 64).sqrt()
    scores = (q @ k.transpose(-1, -2) - tau[:, None]).relu().square()
    scores = scores.masked_fill(positions[None, :] > positions[:, None], 0)
    expected = scores @ v
    torch.testing.assert_close(out, expected, atol=2e-5, rtol=2e-3)
    grad = torch.randn(2, 65, 2, 64, device='cuda').transpose(1, 2)
    assert not grad.is_contiguous()
    out.backward(grad)
    expected.backward(grad)
    for actual, ref in zip(operands, copies):
        assert actual.grad.isfinite().all()
        torch.testing.assert_close(actual.grad, ref.grad, atol=2e-4, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_native_tda_differential_merge_preserves_distinct_path_gradients():
    from architectures.tda import TDALayer
    torch.manual_seed(111)
    operands = [torch.randn(2, 65, 2, 64, device='cuda').requires_grad_() for _ in range(5)]
    copies = [x.detach().clone().requires_grad_() for x in operands]
    lam = torch.tensor(.3, device='cuda', requires_grad=True)
    ref_lam = lam.detach().clone().requires_grad_()
    q1, k1, q2, k2, v = operands
    native = TDALayer(2, 64, differential=True, target='native', intent='training')
    out = native(q1, k1, v, query2=q2, key2=k2, lam=lam)
    positions = torch.arange(65, device='cuda')
    tau = (2 * (positions + 1).float().log() / 64).sqrt()
    def threshold(q, k, value):
        weights = ((q.transpose(1, 2) * .125) @ k.transpose(1, 2).transpose(-1, -2) - tau[:, None]).relu().square()
        weights = weights.masked_fill(positions[None, :] > positions[:, None], 0)
        return (weights @ value.transpose(1, 2)).transpose(1, 2)
    expected = threshold(copies[0], copies[1], copies[4]) - ref_lam * threshold(copies[2], copies[3], copies[4])
    torch.testing.assert_close(out, expected, atol=3e-5, rtol=3e-4)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    for actual, ref in zip(operands + [lam], copies + [ref_lam]):
        assert actual.grad is not None and actual.grad.isfinite().all()
        torch.testing.assert_close(actual.grad, ref.grad, atol=5e-4, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_native_differential_attention_preserves_projection_and_lambda_gradients():
    from architectures.differential_attention import DifferentialAttentionLayer
    from torch.nn.functional import scaled_dot_product_attention
    torch.manual_seed(112)
    native = DifferentialAttentionLayer(128, 2, 64, 0, target='native', intent='training').cuda()
    reference = DifferentialAttentionLayer(128, 2, 64, 0, target='reference', intent='training').cuda()
    reference.load_state_dict(native.state_dict())
    a = torch.randn(2, 65, 128, device='cuda', requires_grad=True)
    b = a.detach().clone().requires_grad_()
    out = native(a)
    q = reference.q_proj(b).reshape(2, 65, 2, 2, 64)
    k = reference.k_proj(b).reshape(2, 65, 2, 2, 64)
    v = reference.v_proj(b).reshape(2, 65, 2, 128).transpose(1, 2)
    paths = [scaled_dot_product_attention(q[:, :, :, i].transpose(1, 2),
              k[:, :, :, i].transpose(1, 2), v, is_causal=True) for i in range(2)]
    mixed = (paths[0] - reference._lambda_full() * paths[1]).transpose(1, 2)
    expected = reference.out_proj((reference.subln(mixed) * (1 - reference.lambda_init)).reshape(2, 65, 256))
    torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-3)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(a.grad, b.grad, atol=3e-4, rtol=3e-3)
    for actual, ref in zip(native.parameters(), reference.parameters()):
        assert actual.grad is not None and actual.grad.isfinite().all()
        torch.testing.assert_close(actual.grad, ref.grad, atol=5e-4, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('operand', ['bias', 'mask'])
def test_native_softmax_autograd_metadata_does_not_retain_operand_graph(operand):
    import gc
    import weakref
    from urm.backends.triton.k1.online_softmax import execute_online_softmax
    source = torch.randn(1, 1, 8, 8, device='cuda', requires_grad=True)
    extra = source * .1
    ref = weakref.ref(extra)
    q, k, v = [torch.randn(1, 8, 1, 16, device='cuda', requires_grad=True) for _ in range(3)]
    out = execute_online_softmax(q, k, v, scale=.25, causal=True,
                                 score_bias=extra if operand == 'bias' else None,
                                 attention_mask=extra if operand == 'mask' else None)
    # Autograd/compiler metadata can outlive the graph. It must not keep its
    # input tensors alive after the backward buffers have been released.
    metadata = out.grad_fn._forward_cls
    out.square().sum().backward()
    assert source.grad is not None and source.grad.isfinite().all()
    del out, extra, q, k, v
    gc.collect()
    assert metadata is not None and ref() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_native_matrix_autograd_metadata_does_not_retain_gate_graphs():
    import gc
    import weakref
    from urm.backends.triton.k2.matrix_scan import execute_matrix_state_recurrence
    source = torch.randn(1, 8, 1, device='cuda', requires_grad=True)
    decay, beta = -source.sigmoid() * .1, source.sigmoid()
    refs = [weakref.ref(decay), weakref.ref(beta)]
    q, k, v = [torch.randn(1, 8, 1, 16, device='cuda', requires_grad=True) * .1 for _ in range(3)]
    out, final = execute_matrix_state_recurrence(
        query=q, key=k, value=v, log_decay=decay, beta=beta, initial_state=None,
        scale=.25, decay_granularity='head', is_delta=True, read_before=False)
    metadata = out.grad_fn._forward_cls
    out.square().sum().backward()
    assert source.grad is not None and source.grad.isfinite().all()
    del out, final, decay, beta, q, k, v
    gc.collect()
    assert metadata is not None and all(ref() is None for ref in refs)
