"""Contract checks for the three-family unified mixer prototype."""

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from urm.compiler.unified_mixer import (
    DecayGranularity,
    FeatureMap,
    MixerBackend,
    MixerIntent,
    MixerKernelFamily,
    PolynomialBasis,
    ReadTiming,
    RecurrentLayout,
    StateTransition,
    StateEffect,
    StateUpdateRule,
    UnifiedMixerSpec,
    compile_frontend_mixer,
    compile_mixer,
)
from urm.frontend.mixer_recipes import (
    MIXER_RECIPE_NAMES,
    delta_rule_spec,
    diagonal_ssm_spec,
    linear_attention_spec,
    named_mixer_recipe,
    softmax_attention_spec,
    sparse_delta_spec,
)


def test_compiler_selects_one_of_the_three_serializable_families():
    specs = (
        softmax_attention_spec(),
        linear_attention_spec(),
        sparse_delta_spec(),
    )
    plans = [compile_mixer(spec, intent=MixerIntent.TRAINING) for spec in specs]

    assert [plan.spec.family for plan in plans] == [
        MixerKernelFamily.SOFTMAX,
        MixerKernelFamily.RECURRENCE,
        MixerKernelFamily.SPARSE_DELTA,
    ]
    assert all(plan.to_dict()["unified_gpu_fusion"] is False for plan in plans)
    assert all(plan.to_dict()["autograd"] is True for plan in plans)
    assert UnifiedMixerSpec(**plans[0].spec.to_dict()) == plans[0].spec


def test_backend_selection_is_explicit_and_family_checked():
    attention = compile_mixer(softmax_attention_spec(), backend=MixerBackend.LIBRARY)
    native_sparse = compile_mixer(sparse_delta_spec(), backend="native")
    assert attention.to_dict()["backend"] == "library"
    assert native_sparse.to_dict()["implementation"] == "urm_native_anchor"
    with pytest.raises(ValueError, match="K3 sparse delta"):
        compile_mixer(linear_attention_spec(), backend="native")
    fla_linear = compile_mixer(
        linear_attention_spec(), backend="library", dtype="bfloat16"
    )
    assert fla_linear.anchor == "fla_chunk_linear_attention_adapter"
    assert (
        compile_mixer(
            named_mixer_recipe("simple_gla"),
            backend="library",
            dtype="bfloat16",
        ).anchor
        == "fla_fused_recurrent_simple_gla_decode_adapter"
    )
    assert (
        compile_mixer(
            named_mixer_recipe("gla"), backend="library", dtype="bfloat16"
        ).anchor
        == "fla_fused_recurrent_gla_decode_adapter"
    )
    assert (
        compile_mixer(
            named_mixer_recipe("simple_gla"), backend="library", dtype="float32"
        ).anchor
        == "fla_chunk_simple_gla_adapter"
    )
    assert (
        compile_mixer(
            named_mixer_recipe("gla"), backend="library", dtype="float32"
        ).anchor
        == "fla_chunk_gla_adapter"
    )
    unsupported_library_spec = delta_rule_spec(
        "key_channel_delta", decay=DecayGranularity.KEY_CHANNEL
    )
    with pytest.raises(ValueError, match="K2 subsets only"):
        compile_mixer(unsupported_library_spec, backend="library", dtype="bfloat16")
    with pytest.raises(ValueError, match="compiled for float32"):
        compile_mixer(softmax_attention_spec()).execute(
            query=_torch().zeros(1, 1, 1, 1, dtype=_torch().float16),
            key=_torch().zeros(1, 1, 1, 1, dtype=_torch().float16),
            value=_torch().zeros(1, 1, 1, 1, dtype=_torch().float16),
        )


def test_atma_gated_delta_decode_is_a_pinned_forward_only_k2_anchor():
    from urm.compiler.diagnostics import CompilerError

    recipe = named_mixer_recipe("atma_gated_delta_decode_core")
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.INFERENCE,
        dtype="float32",
    )
    assert plan.anchor == "atma_gated_delta_decode_adapter"
    assert plan.to_dict()["named_coverage"]["architecture_ids"] == ["arch-026"]
    assert recipe.spec.feature_map is FeatureMap.L2_NORMALIZE
    with pytest.raises(CompilerError, match="forward-only"):
        compile_mixer(
            recipe,
            backend=MixerBackend.LIBRARY,
            intent=MixerIntent.TRAINING,
            dtype="float32",
        )
    with pytest.raises(ValueError, match="float32"):
        compile_mixer(recipe, backend=MixerBackend.LIBRARY, dtype="bfloat16")


def test_atma_selection_is_semantic_and_independent_of_recipe_name():
    torch = _torch()
    recipe = named_mixer_recipe("atma_gated_delta_decode_core")
    renamed = replace(recipe.spec, name="renamed_decode_operation")
    assert recipe.spec.semantic_signature() == renamed.semantic_signature()
    original_plan = compile_mixer(recipe.spec, backend=MixerBackend.LIBRARY)
    renamed_plan = compile_mixer(renamed, backend=MixerBackend.LIBRARY)
    assert original_plan.anchor == renamed_plan.anchor == "atma_gated_delta_decode_adapter"

    operands = {
        "query": torch.randn(1, 1, 2, 8),
        "key": torch.randn(1, 1, 2, 8),
        "value": torch.randn(1, 1, 2, 4),
        "gamma": torch.rand(1, 1, 2),
        "beta": torch.rand(1, 1, 2),
        "state_table": torch.randn(3, 2, 8, 4),
        "slots": torch.tensor([1], dtype=torch.int64),
    }
    original = compile_mixer(recipe.spec).execute(**operands)
    renamed_result = compile_mixer(renamed).execute(**operands)
    torch.testing.assert_close(renamed_result.output, original.output)
    torch.testing.assert_close(renamed_result.final_state, original.final_state)

    changed_semantics = (
        replace(recipe.spec, read_timing=ReadTiming.BEFORE_UPDATE),
        replace(recipe.spec, update_rule=StateUpdateRule.ADDITIVE),
        replace(recipe.spec, feature_map=FeatureMap.IDENTITY),
        replace(recipe.spec, read_scale=0.5),
    )
    for spec in changed_semantics:
        assert spec.semantic_signature() != recipe.spec.semantic_signature()
        with pytest.raises(ValueError, match="exact gated-delta decode semantics"):
            compile_mixer(spec, backend=MixerBackend.LIBRARY)


def test_k1_operation_name_does_not_change_math_or_backend_eligibility():
    torch = _torch()
    from urm.ir import K1Operation

    standard = named_mixer_recipe("mha").spec
    renamed_standard = replace(standard, name="wall_attention_core")
    assert renamed_standard.k1_operation is K1Operation.SOFTMAX
    assert standard.semantic_signature() == renamed_standard.semantic_signature()
    assert (
        compile_mixer(standard, backend=MixerBackend.LIBRARY).anchor
        == compile_mixer(renamed_standard, backend=MixerBackend.LIBRARY).anchor
    )
    native_plan = compile_mixer(standard, backend=MixerBackend.NATIVE)
    renamed_native_plan = compile_mixer(
        renamed_standard, backend=MixerBackend.NATIVE
    )
    assert native_plan.anchor == renamed_native_plan.anchor
    assert native_plan.backend_selection == renamed_native_plan.backend_selection
    selection = native_plan.to_dict()["backend_selection"]
    assert selection["selected_backend"] == "triton_online_softmax"
    assert selection["fallback_used"] is False
    assert selection["request"] == {
        "operation": "K1",
        "semantic_contract": "normalized_softmax_attention_v1",
        "device": "cuda",
        "dtype": "float32",
        "layout": "BTHD",
        "mode": "inference",
    }
    with pytest.raises(ValueError, match="unsupported layout: BHDT"):
        compile_mixer(standard, backend=MixerBackend.NATIVE, layout="BHDT")
    with pytest.raises(ValueError, match="device='cuda'"):
        compile_mixer(standard, backend=MixerBackend.NATIVE, device="cpu")
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 6)
    expected = compile_mixer(standard).execute(query=q, key=k, value=v).output
    actual = compile_mixer(renamed_standard).execute(query=q, key=k, value=v).output
    torch.testing.assert_close(actual, expected)

    polar = named_mixer_recipe("polar_attention_core").spec
    renamed_polar = replace(polar, name="mha")
    assert renamed_polar.k1_operation is K1Operation.POLAR
    assert not renamed_polar.is_normalized_softmax_attention()
    assert compile_mixer(renamed_polar, backend=MixerBackend.LIBRARY).anchor == (
        "atma_polar_triton_adapter"
    )
    with pytest.raises(ValueError, match="K1 normalized softmax"):
        compile_mixer(renamed_polar, backend=MixerBackend.NATIVE)
    operands = {
        "query": torch.randn(1, 2, 4, 8),
        "key": torch.randn(1, 2, 4, 8),
        "value": torch.randn(1, 2, 4, 8),
        "n_keys": torch.tensor([1, 2, 3, 4], dtype=torch.float32),
        "v_null": torch.randn(2, 8),
        "null_base": torch.randn(2),
        "null_slope_raw": torch.randn(2),
        "len_gain_raw": torch.randn(2),
        "mag_beta_raw": torch.randn(2),
    }
    original = compile_mixer(polar).execute(**operands)
    renamed_result = compile_mixer(renamed_polar).execute(**operands)
    torch.testing.assert_close(renamed_result.output, original.output)
    torch.testing.assert_close(renamed_result.auxiliary_output, original.auxiliary_output)


def test_native_sparse_k1_requires_the_explicit_route_mask():
    plan = compile_mixer(
        named_mixer_recipe("sparse_attention_core"), backend=MixerBackend.NATIVE
    )
    with pytest.raises(ValueError, match="requires a precomputed attention_mask"):
        plan.execute()


# Recipes whose K2 equations are not yet expressed as reusable semantic
# operations: their library/native dispatch still reads ``spec.name``, so
# renaming them changes the selected anchor or the dtype rejection. These are
# the remaining architecture-specific branches, explicitly marked unfinished.
# Each must be lowered into explicit reusable operations (see the compiler
# charter) before it can join the name-invariant set asserted below.
_NAME_DEPENDENT_RECIPES_UNFINISHED = frozenset(
    {
        "abc_core",
        "bdh_attention_core",
        "gru_core",
        "h3_ssm_fft_core",
        "hla_second_order_core",
        "hyena_fftconv_core",
        "m2rnn_core",
        "mamba3_siso_core",
        "mesa_net_core",
        "rnn_core",
        "rwkv7_transition_core",
        "titans_linear_memory_core",
        "ttt_linear_core",
    }
)


@pytest.mark.parametrize("recipe_name", sorted(MIXER_RECIPE_NAMES))
def test_recipe_anchor_selection_is_name_invariant(recipe_name):
    """Renaming a recipe must not change the selected anchor or its rejection.

    This is the name-invariance acceptance criterion: the executed mathematics
    is fixed by the semantic fields, never by ``spec.name``. Recipes still
    dispatched by name are enumerated in ``_NAME_DEPENDENT_RECIPES_UNFINISHED``
    and excluded here until their equations are lowered into reusable
    operations; any *new* name-dependent recipe fails this test.
    """
    if recipe_name in _NAME_DEPENDENT_RECIPES_UNFINISHED:
        pytest.skip(
            f"{recipe_name} is still dispatched by spec.name (unfinished); "
            "see _NAME_DEPENDENT_RECIPES_UNFINISHED"
        )
    spec = named_mixer_recipe(recipe_name).spec
    renamed = replace(spec, name="zz_renamed_operation")
    assert spec.semantic_signature() == renamed.semantic_signature()
    for backend in (MixerBackend.REFERENCE, MixerBackend.LIBRARY, MixerBackend.NATIVE):
        for dtype in ("float32", "bfloat16"):

            def _outcome(candidate):
                try:
                    return compile_mixer(candidate, backend=backend, dtype=dtype).anchor
                except Exception as error:  # noqa: BLE001 - rejection identity matters
                    return f"declined:{type(error).__name__}"

            assert _outcome(spec) == _outcome(renamed), (
                f"{recipe_name} changed its {backend}/{dtype} outcome when renamed; "
                "dispatch must depend on semantic fields, not spec.name"
            )


def test_name_dependent_recipes_are_explicitly_enumerated():
    """The unfinished name-dependent set must match reality exactly.

    If a recipe's equation is lowered into reusable operations, remove it from
    ``_NAME_DEPENDENT_RECIPES_UNFINISHED`` so the name-invariance test above
    starts enforcing it. If a new name-dependent recipe appears, this test
    fails until it is either fixed or explicitly marked unfinished.
    """
    actually_name_dependent = set()
    for name in MIXER_RECIPE_NAMES:
        spec = named_mixer_recipe(name).spec
        renamed = replace(spec, name="zz_renamed_operation")
        for backend in (MixerBackend.REFERENCE, MixerBackend.LIBRARY, MixerBackend.NATIVE):
            for dtype in ("float32", "bfloat16"):

                def _outcome(candidate):
                    try:
                        return compile_mixer(
                            candidate, backend=backend, dtype=dtype
                        ).anchor
                    except Exception as error:  # noqa: BLE001
                        return f"declined:{type(error).__name__}"

                if _outcome(spec) != _outcome(renamed):
                    actually_name_dependent.add(name)
    assert actually_name_dependent == set(_NAME_DEPENDENT_RECIPES_UNFINISHED)


def test_production_matrix_native_status_matches_compiler():
    """The frozen matrix's native-status claims must match the compiler.

    A workload marked ``candidate_exists`` must have a native path for a
    representative recipe; one marked ``native_generation_gap`` must not. This
    keeps the release envelope from overclaiming native generation coverage.
    """
    import json
    from pathlib import Path

    matrix = json.loads(
        (Path(__file__).parents[1] / "benchmarks" / "production-matrix.json").read_text()
    )
    # Representative recipe exercising each matrix workload's native path.
    representative = {
        "k1-mha": "mha",
        "k1-gqa": "gqa",
        "k1-masked-variant": "mha",
        "k2-diagonal-recurrence": "hgrn_ssm_core",
        "k2-gated-delta-recurrence": "gated_delta_net",
        "k3-sparse-state": "sparse_delta_memory",
    }
    for workload in matrix["workloads"]:
        recipe = representative[workload["id"]]
        spec = named_mixer_recipe(recipe).spec
        try:
            compile_mixer(spec, backend=MixerBackend.NATIVE, dtype="float32")
            native_ok = True
        except Exception:  # noqa: BLE001 - any decline means no native candidate
            native_ok = False
        if workload["native_status"] == "candidate_exists":
            assert native_ok, (
                f"{workload['id']} claims a native candidate but {recipe} has no "
                "native path"
            )
        else:
            assert not native_ok, (
                f"{workload['id']} is marked a native gap but {recipe} now compiles "
                "natively; promote it to candidate_exists"
            )


def test_representation_coverage_claims_match_compiler():
    """Pin the representation-coverage evidence to the live compiler.

    ``docs/validation/representation-coverage.md`` records, for each mandatory
    workload class, whether it is expressible and which reference/library/native
    anchors the compiler selects. This test re-derives those anchors so the
    documented coverage cannot drift from reality.
    """
    expected = {
        # (recipe, family): {backend: anchor substring or None for decline}
        "k1_mha": (
            softmax_attention_spec(),
            {
                MixerBackend.REFERENCE: "urm.unified.k1.softmax_reference.v1",
                MixerBackend.LIBRARY: "scaled_dot_product_attention",
                MixerBackend.NATIVE: "urm_native_k1_online_softmax_v1",
            },
        ),
        "k2_diagonal": (
            diagonal_ssm_spec("hgrn_ssm_core", hgrn=True),
            {
                MixerBackend.REFERENCE: "urm.unified.k2.state_reference.v1",
                MixerBackend.LIBRARY: "fla_fused_recurrent_hgrn_adapter",
                MixerBackend.NATIVE: "urm_native_diagonal_recurrence_v1",
            },
        ),
        "k2_gated_delta": (
            delta_rule_spec("gated_delta", decay=DecayGranularity.HEAD),
            {
                MixerBackend.REFERENCE: "urm.unified.k2.state_reference.v1",
                MixerBackend.LIBRARY: "fla_gated_delta_rule_adapter",
                MixerBackend.NATIVE: None,  # native generation gap, not representational
            },
        ),
        "k3_sparse": (
            sparse_delta_spec(),
            {
                MixerBackend.REFERENCE: "urm.unified.k3.sparse_delta_reference.v1",
                MixerBackend.LIBRARY: None,  # external adapter boundary, not in-process
                MixerBackend.NATIVE: "urm_native_sparse_state_mixer_v0",
            },
        ),
    }
    for label, (spec, backends) in expected.items():
        for backend, anchor_fragment in backends.items():
            dtypes = ("float32", "bfloat16") if backend is MixerBackend.LIBRARY else ("float32",)
            outcome = None
            for dt in dtypes:
                try:
                    outcome = compile_mixer(spec, backend=backend, intent="training", dtype=dt).anchor
                    break
                except Exception:  # noqa: BLE001 - decline identity is the signal
                    outcome = None
            if anchor_fragment is None:
                assert outcome is None, (
                    f"{label}/{backend.value} was expected to decline but selected {outcome}"
                )
            else:
                assert outcome is not None and anchor_fragment in outcome, (
                    f"{label}/{backend.value} expected anchor containing "
                    f"{anchor_fragment!r}, got {outcome!r}"
                )


@pytest.mark.parametrize("recipe_name", sorted(_NAME_DEPENDENT_RECIPES_UNFINISHED))
def test_native_backend_declines_name_dependent_recipes(recipe_name):
    """The native production path must never silently name-dispatch.

    Name-dependent recipes select an upstream comparator by ``spec.name``; that
    is legitimate for the LIBRARY/REFERENCE comparison boundary but must not
    reach the native generator. Until their equations are lowered into reusable
    semantic operations, the native backend must decline them explicitly rather
    than substitute an equation selected by name.
    """
    spec = named_mixer_recipe(recipe_name).spec
    for dtype in ("float32", "bfloat16"):
        with pytest.raises(Exception):
            compile_mixer(spec, backend=MixerBackend.NATIVE, dtype=dtype)


def test_missing_route_mask_diagnostic_does_not_require_torch():
    """The missing-mask check must run before importing any optional backend.

    Regression: ``execute`` imported PyTorch before validating operands, so a
    minimal-dependency environment reported "requires PyTorch" instead of the
    actionable missing-mask diagnostic.
    """
    import os

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"  # importing torch now raises ImportError
        "from urm.compiler.unified_mixer import MixerBackend, compile_mixer\n"
        "from urm.frontend.mixer_recipes import named_mixer_recipe\n"
        "plan = compile_mixer(\n"
        "    named_mixer_recipe('sparse_attention_core'), backend=MixerBackend.NATIVE\n"
        ")\n"
        "try:\n"
        "    plan.execute()\n"
        "except ValueError as error:\n"
        "    assert 'requires a precomputed attention_mask' in str(error), str(error)\n"
        "except Exception as error:\n"
        "    raise AssertionError(f'unexpected {type(error).__name__}: {error}')\n"
        "else:\n"
        "    raise AssertionError('execute() should have rejected the missing mask')\n"
    )
    result = subprocess.run(
        [subprocess.sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_atma_decode_dimension_contract_matches_both_pinned_value_tiles():
    from urm.compiler.unified_mixer import (
        _atma_decode_value_block,
        _validate_atma_decode_dimensions,
    )

    assert _validate_atma_decode_dimensions(1, 64, 64) == 32
    assert _validate_atma_decode_dimensions(256, 64, 64) == 64
    assert _atma_decode_value_block(1, 64) == 32
    assert _atma_decode_value_block(256, 64) == 64
    with pytest.raises(ValueError, match="power of two"):
        _validate_atma_decode_dimensions(1, 48, 64)
    with pytest.raises(ValueError, match="divisible by the pinned unmasked 32-wide"):
        _validate_atma_decode_dimensions(1, 64, 33)


def test_atma_decode_rejects_unsafe_dimensions_before_launch():
    torch = _torch()
    plan = compile_mixer(
        named_mixer_recipe("atma_gated_delta_decode_core"),
        backend=MixerBackend.LIBRARY,
    )
    for key_dim, value_dim, message in (
        (48, 64, "power of two"),
        (64, 33, "unmasked 32-wide"),
    ):
        with pytest.raises(ValueError, match=message):
            plan.execute(
                query=torch.zeros(1, 1, 1, key_dim),
                key=torch.zeros(1, 1, 1, key_dim),
                value=torch.zeros(1, 1, 1, value_dim),
                gamma=torch.ones(1, 1, 1),
                beta=torch.ones(1, 1, 1),
                state_table=torch.zeros(2, 1, key_dim, value_dim),
                slots=torch.tensor([0], dtype=torch.int64),
            )


def test_atma_pinned_decode_executes_both_unmasked_value_tile_branches():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("ATMA gated-delta decode requires CUDA")
    pytest.importorskip("triton")
    from urm.adapters.atma_gated_delta import atma_gated_delta_decode_step

    atma_gated_delta_decode_step()
    torch.manual_seed(1002)
    spec = named_mixer_recipe("atma_gated_delta_decode_core").spec
    reference_plan = compile_mixer(spec, intent=MixerIntent.INFERENCE)
    native_upstream_plan = compile_mixer(
        spec, backend=MixerBackend.LIBRARY, intent=MixerIntent.INFERENCE
    )
    for batch in (1, 256):
        heads, key_dim, value_dim, capacity = 1, 64, 64, batch
        inputs = {
            "query": torch.randn(batch, 1, heads, key_dim, device="cuda"),
            "key": torch.randn(batch, 1, heads, key_dim, device="cuda"),
            "value": torch.randn(batch, 1, heads, value_dim, device="cuda"),
            "gamma": torch.rand(batch, 1, heads, device="cuda"),
            "beta": torch.rand(batch, 1, heads, device="cuda"),
            "slots": torch.arange(batch, dtype=torch.int64, device="cuda"),
        }
        initial = torch.randn(
            capacity, heads, key_dim, value_dim, device="cuda", dtype=torch.float32
        ) * 0.01
        reference = reference_plan.execute(
            **inputs, state_table=initial.clone()
        )
        upstream = native_upstream_plan.execute(
            **inputs, state_table=initial.clone()
        )
        torch.testing.assert_close(upstream.output, reference.output, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(
            upstream.final_state, reference.final_state, atol=3e-6, rtol=3e-5
        )


@pytest.mark.parametrize("mask_kind", ["boolean", "additive", "score_bias"])
def test_k1_fully_masked_rows_have_zero_outputs_and_zero_gradients(mask_kind):
    torch = _torch()
    torch.manual_seed(811)
    query = torch.randn(1, 2, 1, 3, requires_grad=True)
    key = torch.randn(1, 2, 1, 3, requires_grad=True)
    value = torch.randn(1, 2, 1, 4, requires_grad=True)
    operands = {"query": query, "key": key, "value": value}
    if mask_kind == "boolean":
        mask = torch.tensor([[True, False], [False, False]]).view(1, 1, 2, 2)
        operands["attention_mask"] = mask
    elif mask_kind == "additive":
        mask = torch.tensor([[0.0, float("-inf")], [float("-inf"), float("-inf")]])
        operands["attention_mask"] = mask.view(1, 1, 2, 2)
    else:
        bias = torch.zeros(1, 1, 2, 2)
        bias[..., 0, 1] = float("-inf")
        bias[..., 1, :] = float("-inf")
        operands["score_bias"] = bias.requires_grad_()

    output = compile_mixer(
        softmax_attention_spec("empty_rows", causal=False, score_bias=True),
        intent=MixerIntent.TRAINING,
    ).execute(**operands).output
    torch.testing.assert_close(output[:, 1], torch.zeros_like(output[:, 1]))
    output.sum().backward()
    torch.testing.assert_close(query.grad[:, 1], torch.zeros_like(query.grad[:, 1]))
    torch.testing.assert_close(key.grad[:, 1], torch.zeros_like(key.grad[:, 1]))
    torch.testing.assert_close(value.grad[:, 1], torch.zeros_like(value.grad[:, 1]))
    if mask_kind == "score_bias":
        bias_grad = operands["score_bias"].grad
        torch.testing.assert_close(bias_grad[..., 1, :], torch.zeros_like(bias_grad[..., 1, :]))


def test_sparse_frontend_mask_requirement_survives_name_lowering():
    from urm.ir import (
        Domain,
        MixerSpec,
        Normalization,
        RoutingKind,
        SelectionGranularity,
        SelectionScope,
        SparseAttentionSpec,
        SparseIndexerKind,
    )

    frontend = MixerSpec(
        name="user_sparse_attention",
        query_domain=Domain.SEQUENCE,
        source_domain=Domain.SEQUENCE,
        routing=RoutingKind.BLOCK_SPARSE,
        normalization=Normalization.SOFTMAX,
        sparse_attention=SparseAttentionSpec(
            indexer=SparseIndexerKind.STATIC_MASK,
            granularity=SelectionGranularity.BLOCK,
            scope=SelectionScope.SHARED_ACROSS_HEADS,
            block_size=1,
        ),
    )
    plan = compile_frontend_mixer(frontend)
    assert plan.spec.name == "user_sparse_attention"
    assert plan.spec.requires_attention_mask
    assert compile_mixer(plan.spec).spec.requires_attention_mask
    operands = {
        "query": _torch().randn(1, 2, 1, 3),
        "key": _torch().randn(1, 2, 1, 3),
        "value": _torch().randn(1, 2, 1, 4),
    }
    with pytest.raises(ValueError, match="requires a precomputed attention_mask"):
        plan.execute(**operands)
    native = compile_frontend_mixer(frontend, backend=MixerBackend.NATIVE)
    assert native.anchor == "urm_native_k1_online_softmax_v1"
    with pytest.raises(ValueError, match="requires a precomputed attention_mask"):
        native.execute(**operands)


def test_frontend_sparse_mask_lowers_to_native_k1_when_cuda_is_available():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("native K1 requires CUDA")
    from urm.ir import (
        Domain,
        MixerSpec,
        Normalization,
        RoutingKind,
        SelectionGranularity,
        SelectionScope,
        SparseAttentionSpec,
        SparseIndexerKind,
    )

    frontend = MixerSpec(
        name="user_sparse_attention",
        query_domain=Domain.SEQUENCE,
        source_domain=Domain.SEQUENCE,
        routing=RoutingKind.BLOCK_SPARSE,
        normalization=Normalization.SOFTMAX,
        sparse_attention=SparseAttentionSpec(
            indexer=SparseIndexerKind.STATIC_MASK,
            granularity=SelectionGranularity.BLOCK,
            scope=SelectionScope.SHARED_ACROSS_HEADS,
            block_size=1,
        ),
    )
    spec = compile_frontend_mixer(frontend).spec
    native = compile_frontend_mixer(frontend, backend=MixerBackend.NATIVE)
    query = torch.randn(1, 3, 2, 8, device="cuda")
    key = torch.randn(1, 4, 1, 8, device="cuda")
    value = torch.randn(1, 4, 1, 6, device="cuda")
    route = torch.tensor(
        [[[[True, False, False, False], [True, True, False, False], [False, True, True, False]]]],
        device="cuda",
    )
    expected = compile_mixer(spec).execute(
        query=query, key=key, value=value, attention_mask=route
    ).output
    actual = native.execute(
        query=query, key=key, value=value, attention_mask=route
    ).output
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=8e-4)


def test_unified_mixer_runs_through_general_semantic_candidate_and_anchor_pipeline():
    specs_and_backends = (
        (softmax_attention_spec(), MixerBackend.REFERENCE),
        (softmax_attention_spec(), MixerBackend.LIBRARY),
        (linear_attention_spec(), MixerBackend.REFERENCE),
        (sparse_delta_spec(), MixerBackend.REFERENCE),
        (sparse_delta_spec(), MixerBackend.NATIVE),
    )
    for spec, backend in specs_and_backends:
        plan = compile_mixer(spec, backend=backend)
        compiler = plan.to_dict()["compiler_plan"]
        assert compiler["steps"][0]["anchor"] == plan.anchor
        assert plan.to_dict()["compiler_candidate"] == "base"
        assert (
            plan.to_dict()["compiler_result"]["candidates"][0]["candidate_id"] == "base"
        )

    training = compile_mixer(
        delta_rule_spec("gated_delta", decay=DecayGranularity.HEAD),
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert training.anchor == "fla_gated_delta_rule_adapter"
    assert training.to_dict()["compiler_plan"]["steps"][0]["anchor"] == training.anchor
    fla_linear = compile_mixer(
        linear_attention_spec(), backend=MixerBackend.LIBRARY, dtype="bfloat16"
    )
    assert (
        fla_linear.to_dict()["compiler_plan"]["steps"][0]["anchor"]
        == "fla_chunk_linear_attention_adapter"
    )

    with pytest.raises(ValueError, match="requires float16 or bfloat16"):
        compile_mixer(
            delta_rule_spec("gated_delta", decay=DecayGranularity.HEAD),
            intent=MixerIntent.TRAINING,
            backend=MixerBackend.LIBRARY,
        )


def test_named_recipes_expose_kernel_scope_and_unfinished_layer_stages():
    recipes = (
        named_mixer_recipe("mha"),
        named_mixer_recipe("gla"),
        named_mixer_recipe("mamba1_ssm_core"),
        named_mixer_recipe("sparse-delta-memory"),
    )
    assert [recipe.spec.family for recipe in recipes] == [
        MixerKernelFamily.SOFTMAX,
        MixerKernelFamily.RECURRENCE,
        MixerKernelFamily.RECURRENCE,
        MixerKernelFamily.SPARSE_DELTA,
    ]
    assert all(recipe.component_scope for recipe in recipes)
    assert all(recipe.required_external_stages for recipe in recipes)
    assert named_mixer_recipe("simple_gla").spec.update_rule is StateUpdateRule.ADDITIVE
    assert named_mixer_recipe("gla").spec.update_rule is StateUpdateRule.ADDITIVE
    assert (
        named_mixer_recipe("gated_delta_net").spec.update_rule is StateUpdateRule.DELTA
    )
    serialized = compile_mixer(recipes[1]).to_dict()["named_coverage"]
    assert serialized["architecture_ids"] == ["arch-019"]
    assert "gate" in serialized["required_external_stages"][0]
    assert compile_mixer(recipes[1]).to_dict()["semantic_spec"]["name"] == "gla"
    with pytest.raises(ValueError, match="no unified mixer recipe"):
        named_mixer_recipe("mamba3")


def test_existing_frontend_specs_lower_or_decline_explicitly():
    from urm.presets import (
        DENSE_ATTENTION,
        GATED_DELTANET,
        GL_SDM_TRANSACTION,
        MAMBA,
        MAMBA3,
        SPARSE_DELTA_MEMORY,
        TOP2_MOE,
    )

    assert (
        compile_frontend_mixer(DENSE_ATTENTION).spec.family is MixerKernelFamily.SOFTMAX
    )
    assert (
        compile_frontend_mixer(MAMBA).spec.recurrent_layout is RecurrentLayout.DIAGONAL
    )
    assert (
        compile_frontend_mixer(GATED_DELTANET).spec.update_rule is StateUpdateRule.DELTA
    )
    assert (
        compile_frontend_mixer(SPARSE_DELTA_MEMORY).spec.family
        is MixerKernelFamily.SPARSE_DELTA
    )
    with pytest.raises(ValueError, match="recurrent algorithm"):
        compile_frontend_mixer(MAMBA3)
    with pytest.raises(ValueError, match="ordered collisions"):
        compile_frontend_mixer(GL_SDM_TRANSACTION)
    with pytest.raises(ValueError, match="expert routing"):
        compile_frontend_mixer(TOP2_MOE)


def test_semantics_reject_false_family_combinations():
    with pytest.raises(ValueError, match="K1 accepts"):
        UnifiedMixerSpec(
            "bad_attention",
            MixerKernelFamily.SOFTMAX,
            update_rule=StateUpdateRule.DELTA,
        )
    with pytest.raises(ValueError, match="diagonal SSM requires"):
        UnifiedMixerSpec(
            "bad_ssm",
            MixerKernelFamily.RECURRENCE,
            recurrent_layout=RecurrentLayout.DIAGONAL,
            update_rule=StateUpdateRule.DELTA,
            decay=DecayGranularity.ELEMENTWISE,
        )


def _torch():
    torch = pytest.importorskip("torch")
    return torch


def _coverage_recipe_operands(torch, spec):
    def leaf(*shape):
        return torch.randn(*shape, requires_grad=True)

    if spec.name == "atma_gated_delta_decode_core":
        return {
            "query": leaf(1, 1, 1, 3),
            "key": leaf(1, 1, 1, 3),
            "value": leaf(1, 1, 1, 2),
            "beta": torch.sigmoid(leaf(1, 1, 1)),
            "gamma": torch.sigmoid(leaf(1, 1, 1)),
            "state_table": leaf(3, 1, 3, 2) * 0.01,
            "slots": torch.tensor([1], dtype=torch.int64),
        }

    if spec.name == "h3_ssm_fft_core":
        return {
            "query": leaf(1, 4, 2, 1),
            "key": leaf(1, 4, 2, 1),
            "value": leaf(1, 4, 2, 1),
            "ssm_kernel": leaf(2, 4),
            "ssm_k_kernel": leaf(2, 4),
            "ssm_k_direct": leaf(2),
            "skip": leaf(2),
        }
    if spec.name == "hyena_fftconv_core":
        return {
            "query": leaf(1, 4, 3),
            "kernel": leaf(3, 4),
            "direct": leaf(3),
        }
    if spec.name == "hla_second_order_core":
        return {
            "query": leaf(1, 4, 1, 3),
            "key": leaf(1, 4, 1, 3),
            "value": leaf(1, 4, 1, 2),
        }

    if spec.family is MixerKernelFamily.SOFTMAX:
        if spec.name in {"polar_attention_core", "foveal_sparse_polar_attention_core"}:
            sequence = 4 if spec.name == "polar_attention_core" else 32
            heads, dim = 1, 3 if sequence == 4 else 8
            operands = {
                "query": leaf(1, heads, sequence, dim),
                "key": leaf(1, heads, sequence, dim),
                "value": leaf(1, heads, sequence, dim),
                "n_keys": torch.arange(1, sequence + 1, dtype=torch.float32),
                "v_null": leaf(heads, dim),
                "null_base": leaf(heads),
                "null_slope_raw": leaf(heads),
                "len_gain_raw": leaf(heads),
                "mag_beta_raw": leaf(heads),
            }
            if spec.name == "foveal_sparse_polar_attention_core":
                operands["page_indices"] = torch.tensor(
                    [[[0, 0], [0, 0]]], dtype=torch.int32
                )
                operands["page_counts"] = torch.tensor([[0, 1]], dtype=torch.int32)
                operands["page_size"] = 16
                operands["local_window"] = 16
            return operands
        if spec.name == "cat_attention_core":
            return {
                "query": leaf(1, 2, 2, 3),
                "key": leaf(1, 2, 2, 3),
                "value": leaf(1, 2, 2, 4),
                "attention_mask": torch.tensor(
                    [[True, False], [True, True]], dtype=torch.bool
                ).view(1, 1, 2, 2),
            }
        if spec.name == "differential_attention_core":
            return {
                "query_a": leaf(1, 2, 2, 3),
                "query_b": leaf(1, 2, 2, 3),
                "key_a": leaf(1, 2, 2, 3),
                "key_b": leaf(1, 2, 2, 3),
                "value": leaf(1, 2, 2, 4),
                "lambda_weight": torch.tensor(0.35, requires_grad=True),
            }
        if spec.name == "tda_attention_core":
            return {
                "query_a": leaf(1, 2, 2, 3),
                "query_b": leaf(1, 2, 2, 3),
                "key_a": leaf(1, 2, 2, 3),
                "key_b": leaf(1, 2, 2, 3),
                "value": leaf(1, 2, 2, 3),
                "beta": torch.tensor(0.2, requires_grad=True),
                "lambda_weight": torch.tensor(0.5, requires_grad=True),
            }
        if spec.name == "tucker_attention_core":
            return {
                "query": leaf(1, 3, 4),
                "key": leaf(1, 3, 4),
                "value": leaf(1, 3, 2),
                "B_pre": leaf(2, 4, 4),
            }
        if spec.name == "longformer_attention_core":
            return {
                "query": leaf(1, 8, 2, 3),
                "key": leaf(1, 8, 2, 3),
                "value": leaf(1, 8, 2, 4),
                "attention_window": 2,
            }
        if spec.name == "kata_attention_core":
            return {
                "query": leaf(1, 4, 2, 8),
                "key": leaf(1, 4, 2, 8),
                "value": leaf(1, 4, 2, 6),
                "num_groups": 2,
            }
        if spec.name == "fwpkm_memory_read_core":
            return {
                "query": torch.ones(1, 1, 1, 1, requires_grad=True),
                "key": leaf(1, 4, 1, 1),
                "value": leaf(1, 4, 1, 3),
            }
        operands = {
            "query": leaf(1, 2, 2, 3),
            "key": leaf(1, 2, 1, 3),
            "value": leaf(1, 2, 1, 4),
        }
        if spec.deltaformer_attention:
            operands["query"] = leaf(1, 2, 2, 3)
            operands["key"] = leaf(1, 2, 2, 3)
            operands["value"] = leaf(1, 2, 2, 3)
            operands["beta"] = torch.sigmoid(leaf(1, 2, 2))
        elif spec.name == "dsa_attention_core":
            operands["key"] = leaf(1, 2, 2, 3)
            operands["value"] = leaf(1, 2, 2, 4)
            operands["attention_mask"] = torch.eye(2, dtype=torch.bool).view(1, 1, 2, 2)
        elif spec.name == "nsa_selected_attention_core":
            operands["query"] = leaf(1, 2, 16, 3)
            operands["key"] = leaf(1, 2, 1, 3)
            operands["value"] = leaf(1, 2, 1, 4)
            operands["attention_mask"] = torch.eye(2, dtype=torch.bool).view(1, 1, 2, 2)
        elif spec.name == "moba_selected_attention_core":
            operands["attention_mask"] = torch.tril(
                torch.ones(2, 2, dtype=torch.bool)
            ).view(1, 1, 2, 2)
        elif spec.name == "mla_attention_core":
            pass
        elif spec.path_attention:
            operands["w"] = leaf(1, 2, 1, 3)
            operands["beta"] = torch.sigmoid(leaf(1, 2, 1))
            operands["g"] = leaf(1, 2, 2) * 0.01
        elif spec.name == "parallax_attention_core":
            operands["query"] = leaf(1, 2, 2, 3)
            operands["r"] = leaf(1, 2, 2, 3)
            operands["key"] = leaf(1, 2, 1, 3)
            operands["value"] = leaf(1, 2, 1, 3)
        elif spec.name == "wall_attention_core":
            operands["query"] = leaf(1, 2, 2, 3)
            operands["key"] = leaf(1, 2, 1, 3)
            operands["value"] = leaf(1, 2, 1, 4)
            operands["g"] = -leaf(1, 2, 2, 3).abs() * 0.01
        if spec.requires_attention_mask and "attention_mask" not in operands:
            operands["attention_mask"] = torch.ones(
                1,
                1,
                operands["query"].shape[1],
                operands["key"].shape[1],
                dtype=torch.bool,
            )
        if spec.accepts_score_bias:
            operands["score_bias"] = leaf(1, 1, 2, 2)
        return operands

    if spec.name == "rnn_core":
        return {
            "query": leaf(1, 2, 1, 3),
            "weight": leaf(1, 3, 3) * 0.1,
            "initial_state": leaf(1, 1, 3) * 0.1,
        }
    if spec.name == "gru_core":
        return {
            "query": leaf(1, 2, 1, 3),
            "weight": leaf(1, 3, 3) * 0.1,
            "forget_input": leaf(1, 2, 1, 3),
            "forget_weight": leaf(1, 3, 3) * 0.1,
            "reset_input": leaf(1, 2, 1, 3),
            "reset_weight": leaf(1, 3, 3) * 0.1,
            "initial_state": leaf(1, 1, 3) * 0.1,
        }
    if spec.name == "m2rnn_core":
        return {
            "query": leaf(1, 2, 1, 3),
            "key": leaf(1, 2, 1, 3),
            "value": leaf(1, 2, 1, 2),
            "weight": leaf(1, 2, 2) * 0.1,
            "forget_input": torch.sigmoid(leaf(1, 2, 1)),
            "initial_state": leaf(1, 1, 3, 2) * 0.1,
        }

    if spec.name == "mamba3_siso_core":
        query = leaf(1, 2, 1, 4).to(torch.bfloat16).detach().requires_grad_()
        key = leaf(1, 2, 1, 4).to(torch.bfloat16).detach().requires_grad_()
        value = leaf(1, 2, 1, 4).to(torch.bfloat16).detach().requires_grad_()
        return {
            "query": query,
            "key": key,
            "value": value,
            "adt": -leaf(1, 1, 2).abs() * 0.1,
            "dt": leaf(1, 1, 2).abs() * 0.1 + 0.01,
            "trap": torch.sigmoid(leaf(1, 1, 2))
            .to(torch.bfloat16)
            .detach()
            .requires_grad_(),
            "query_bias": leaf(1, 4).to(torch.bfloat16).detach().requires_grad_(),
            "key_bias": leaf(1, 4).to(torch.bfloat16).detach().requires_grad_(),
            "angles": leaf(1, 2, 1, 1),
        }

    if spec.name == "mesa_net_core":
        query = leaf(1, 2, 1, 3).to(torch.bfloat16).detach().requires_grad_()
        key = leaf(1, 2, 1, 3).to(torch.bfloat16).detach().requires_grad_()
        value = leaf(1, 2, 1, 3).to(torch.bfloat16).detach().requires_grad_()
        return {
            "query": query,
            "key": key,
            "value": value,
            "log_decay": (-leaf(1, 2, 1).abs() * 0.02).requires_grad_(),
            "beta": torch.sigmoid(leaf(1, 2, 1)),
            "lamb": torch.nn.functional.softplus(leaf(1, 3)) + 1.0,
        }

    if spec.recurrent_layout is RecurrentLayout.DIAGONAL:
        if spec.diagonal_hgrn:
            return {
                "x": leaf(1, 2, 3),
                "log_decay": -leaf(1, 2, 3).abs() * 0.1,
                "initial_state": leaf(1, 3),
            }
        operands = {
            "x": leaf(1, 2, 3),
            "input_gate": leaf(1, 2, 4),
            "read_gate": leaf(1, 2, 4),
            "log_decay": -leaf(1, 2, 4).abs() * 0.1,
            "initial_state": leaf(1, 3, 4),
            "skip": leaf(3),
        }
        if spec.step_size_discretization:
            operands["step_size"] = leaf(1, 2, 3).sigmoid()
        return operands

    if spec.rwkv6_memory:
        return {
            "query": leaf(1, 2, 1, 3),
            "key": leaf(1, 2, 1, 3),
            "value": leaf(1, 2, 1, 4),
            "log_decay": -leaf(1, 2, 1, 3).abs() * 0.01,
            "bonus": leaf(1, 3),
            "initial_state": leaf(1, 1, 3, 4),
        }

    if spec.momentum_delta:
        return {
            "query": leaf(1, 2, 1, 3),
            "key": leaf(1, 2, 1, 3),
            "value": leaf(1, 2, 1, 4),
            "p": leaf(1, 2, 1, 3),
            "log_alpha": -leaf(1, 2, 1).abs() * 0.05,
            "log_mu": -leaf(1, 2, 1).abs() * 0.05,
            "beta": torch.sigmoid(leaf(1, 2, 1)),
            "eta": torch.sigmoid(leaf(1, 2, 1)),
            "initial_state": leaf(1, 1, 3, 4) * 0.05,
            "initial_normalizer_state": leaf(1, 1, 3, 4) * 0.05,
        }

    if spec.gated_oja:
        return {
            "query": leaf(1, 2, 1, 3),
            "key": leaf(1, 2, 1, 3),
            "value": leaf(1, 2, 1, 4),
            "gv": -leaf(1, 2, 1, 4).abs() * 0.05,
            "beta": torch.sigmoid(leaf(1, 2, 1)),
            "initial_state": leaf(1, 1, 3, 4) * 0.05,
        }

    if spec.comba_rule:
        return {
            "query": leaf(1, 2, 1, 3),
            "key": leaf(1, 2, 1, 3),
            "value": leaf(1, 2, 1, 4),
            "p": leaf(1, 2, 1, 3),
            "g": -leaf(1, 2, 1).abs() * 0.05,
            "beta": torch.sigmoid(leaf(1, 2, 1)),
            "initial_state": leaf(1, 1, 3, 4) * 0.05,
        }

    if spec.preconditioned_gated_delta:
        return {
            "query": leaf(1, 2, 1, 3),
            "key": leaf(1, 2, 1, 3),
            "value": leaf(1, 2, 1, 4),
            "g_atk": -leaf(1, 2, 1).abs() * 0.05,
            "g": -leaf(1, 2, 1).abs() * 0.05,
            "beta_atk": torch.sigmoid(leaf(1, 2, 1)),
            "beta": torch.sigmoid(leaf(1, 2, 1)),
            "initial_state": leaf(1, 1, 3, 4) * 0.05,
            "initial_A_state": leaf(1, 1, 3) * 0.01,
        }

    if spec.preconditioned_kda:
        return {
            "query": leaf(1, 2, 1, 3),
            "key": leaf(1, 2, 1, 3),
            "value": leaf(1, 2, 1, 4),
            "g": -leaf(1, 2, 1, 3).abs() * 0.05,
            "g_atk": -leaf(1, 2, 1).abs() * 0.05,
            "beta_atk": torch.sigmoid(leaf(1, 2, 1)),
            "beta": torch.sigmoid(leaf(1, 2, 1)),
            "initial_state": leaf(1, 1, 3, 4) * 0.01,
            "initial_A_state": leaf(1, 1, 3) * 0.01,
        }

    if spec.slot_attention:
        operands = {
            "query": leaf(1, 2, 2, 3),
            "key": leaf(1, 2, 1, 3),
            "value": leaf(1, 2, 1, 4),
            "initial_key_state": leaf(1, 1, 3, 2) * 0.05,
            "initial_value_state": leaf(1, 1, 2, 4) * 0.05,
        }
        if spec.name == "abc_core":
            operands["slot_logits"] = leaf(1, 2, 1, 2)
        else:
            operands["slot_weights"] = torch.sigmoid(leaf(1, 2, 1, 2))
            operands["log_decay"] = -leaf(1, 2, 1, 2).abs() * 0.05
        return operands

    if spec.family is MixerKernelFamily.RECURRENCE:
        if spec.name == "bdh_attention_core":
            query = leaf(1, 4, 1, 4)
            return {
                "query": query,
                "key": query,
                "value": leaf(1, 4, 1, 3),
            }
        if spec.name == "ttt_linear_core":
            return {
                "query": leaf(1, 32, 1, 8) * 0.1,
                "key": torch.nn.functional.normalize(leaf(1, 32, 1, 8), dim=-1),
                "value": leaf(1, 32, 1, 8) * 0.1,
                "w": torch.ones(1, 8, requires_grad=True),
                "b": leaf(1, 8) * 0.01,
                "eta": leaf(1, 32, 1, 1) * 0.005,
                "initial_state": leaf(1, 1, 8, 8) * 0.01,
                "initial_state_bias": leaf(1, 1, 1, 8) * 0.01,
            }
        if spec.name == "titans_linear_memory_core":
            return {
                "query": leaf(1, 32, 1, 8) * 0.1,
                "key": torch.nn.functional.normalize(leaf(1, 32, 1, 8), dim=-1),
                "value": leaf(1, 32, 1, 8) * 0.1,
                "w": torch.ones(1, 8, requires_grad=True),
                "b": leaf(1, 8) * 0.01,
                "theta": torch.rand(1, 32, 1, 1, requires_grad=True) * 0.1 + 0.05,
                "alpha": torch.rand(1, 32, 1, 1, requires_grad=True) * 0.1 + 0.05,
                "eta": torch.rand(1, 32, 1, 1, requires_grad=True) * 0.05 + 0.9,
                "initial_state": leaf(1, 1, 8, 8) * 0.01,
            }
        if spec.rwkv4_memory:
            rwkv_state = torch.stack(
                (
                    leaf(1, 4) * 0.1,
                    torch.rand(1, 4) + 0.5,
                    leaf(1, 4) * 0.1,
                ),
                dim=1,
            ).unsqueeze(2)
            return {
                "w": -leaf(4).abs() - 1.0,
                "u": leaf(4) * 0.1,
                "k": leaf(1, 2, 4) * 0.1,
                "v": leaf(1, 2, 4) * 0.1,
                "state": rwkv_state.requires_grad_(),
            }
        if spec.log_linear_attention:
            return {
                "query": leaf(1, 64, 1, 64) * 0.1,
                "key": leaf(1, 64, 1, 64) * 0.1,
                "value": leaf(1, 64, 1, 16) * 0.1,
                "log_decay": -leaf(1, 64, 1).abs() * 0.03,
                "level_scales": torch.rand(1, 64, 1, 7, requires_grad=True) * 0.2,
            }
        if spec.generalized_delta_iplr or spec.generalized_delta_dplr:
            operands = {
                "query": leaf(1, 2, 1, 3),
                "key": leaf(1, 2, 1, 3),
                "value": leaf(1, 2, 1, 2),
                "transition_alpha": leaf(1, 2, 1, 3) * 0.05,
                "transition_beta": leaf(1, 2, 1, 3) * 0.05,
                "initial_state": leaf(1, 1, 3, 2),
            }
            if spec.generalized_delta_dplr:
                operands["log_decay"] = -leaf(1, 2, 1, 3).abs() * 0.1
            return operands
        if spec.kda_delta:
            return {
                "query": leaf(1, 2, 1, 3),
                "key": leaf(1, 2, 1, 3),
                "value": leaf(1, 2, 1, 2),
                "log_decay": -leaf(1, 2, 1, 3).abs() * 0.1,
                "beta": torch.sigmoid(leaf(1, 2, 1)),
                "initial_state": leaf(1, 1, 3, 2),
            }
        if spec.polynomial_basis is not PolynomialBasis.NONE:
            return {
                "query": leaf(1, 2, 1, 3),
                "key": leaf(1, 2, 1, 3),
                "value": leaf(1, 2, 1, 2),
            }
        if spec.gdn2_ssm:
            return {
                "query": leaf(1, 2, 1, 3),
                "key": leaf(1, 2, 1, 3),
                "value": leaf(1, 2, 1, 2),
                "log_decay": -leaf(1, 2, 1, 3).abs() * 0.1,
                "erase_gate": torch.sigmoid(leaf(1, 2, 1, 3)),
                "write_gate": torch.sigmoid(leaf(1, 2, 1, 2)),
                "initial_state": leaf(1, 1, 3, 2),
            }
        if spec.mamba2_ssm:
            return {
                "x": leaf(1, 2, 1, 2),
                "dt": leaf(1, 2, 1).abs() * 0.1 + 0.01,
                "A": -leaf(1).abs() - 0.1,
                "B": leaf(1, 2, 1, 3),
                "C": leaf(1, 2, 1, 3),
                "initial_states": leaf(1, 1, 2, 3),
            }
        operands = {
            "query": leaf(1, 2, 1, 3),
            "key": leaf(1, 2, 1, 3),
            "value": leaf(1, 2, 1, 2),
            "initial_state": leaf(1, 1, 3, 2),
        }
        if spec.state_v_first:
            operands["initial_state"] = leaf(1, 1, 2, 3)
        if spec.update_rule is StateUpdateRule.DELTA:
            ranks = 2 if spec.name == "gated_delta_product_core" else 1
            operands["beta"] = torch.sigmoid(leaf(1, 2, ranks, 1))
            if ranks > 1:
                operands["update_keys"] = leaf(1, 2, ranks, 1, 3)
                operands["update_values"] = leaf(1, 2, ranks, 1, 2)
        if spec.decay is DecayGranularity.HEAD:
            if spec.static_head_decay:
                operands["log_decay"] = torch.zeros(1, requires_grad=False)
            else:
                operands["log_decay"] = -leaf(1, 2, 1).abs() * 0.1
        elif spec.decay is DecayGranularity.KEY_CHANNEL:
            operands["log_decay"] = -leaf(1, 2, 1, 3).abs() * 0.1
        if spec.transition is StateTransition.FACTORED_MATRIX:
            left = torch.eye(3).view(1, 1, 1, 3, 3).expand(1, 2, 1, 3, 3)
            right = torch.eye(2).view(1, 1, 1, 2, 2).expand(1, 2, 1, 2, 2)
            operands["left_transition"] = left.clone().requires_grad_()
            operands["right_transition"] = right.clone().requires_grad_()
        return operands

    return {
        "memory": leaf(1, 4, 2),
        "read_indices": torch.tensor([[[0], [1]]], dtype=torch.int64),
        "read_weights": torch.ones(1, 2, 1, requires_grad=True),
        "write_indices": torch.tensor([[[2], [3]]], dtype=torch.int64),
        "write_weights": torch.ones(1, 2, 1, requires_grad=True),
        "values": leaf(1, 2, 2),
        "beta": torch.full((1, 2, 1), 0.5, requires_grad=True),
        "log_decay": torch.zeros(1, 2, 1, requires_grad=True),
    }


@pytest.mark.parametrize("recipe_name", MIXER_RECIPE_NAMES)
def test_every_named_coverage_recipe_runs_forward_and_backward(recipe_name):
    torch = _torch()
    recipe = named_mixer_recipe(recipe_name)
    plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        dtype=(
            "bfloat16"
            if recipe_name in {"mesa_net_core", "mamba3_siso_core"}
            else "float32"
        ),
    )
    result = plan.execute(**_coverage_recipe_operands(torch, recipe.spec))
    loss = result.output.float().square().mean()
    if result.final_state is not None:
        state = getattr(result.final_state, "ht", result.final_state)
        if isinstance(state, tuple):
            loss = loss + sum(item.float().square().mean() for item in state)
        else:
            loss = loss + state.float().square().mean()
    if result.final_normalizer_state is not None:
        loss = loss + result.final_normalizer_state.float().square().mean()
    if result.auxiliary_output is not None:
        loss = loss + result.auxiliary_output.float().square().mean()
    loss.backward()
    assert result.output.numel() > 0
    assert result.metadata["anchor"] == plan.anchor


def test_titans_linear_memory_core_matches_pinned_upstream_outputs_state_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned FLA Titans comparison")
    source = pytest.importorskip("fla.ops.titans.naive")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7819)
    batch, sequence, heads, dim = 1, 32, 1, 8
    query = (
        torch.randn(batch, sequence, heads, dim, device="cuda", generator=generator)
        * 0.1
    )
    key = torch.nn.functional.normalize(
        torch.randn(batch, sequence, heads, dim, device="cuda", generator=generator),
        dim=-1,
    )
    value = (
        torch.randn(batch, sequence, heads, dim, device="cuda", generator=generator)
        * 0.1
    )
    operands = {
        "query": query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
        "w": (
            torch.ones(heads, dim, device="cuda")
            + torch.randn(heads, dim, device="cuda", generator=generator) * 0.01
        ).requires_grad_(),
        "b": (
            torch.randn(heads, dim, device="cuda", generator=generator) * 0.01
        ).requires_grad_(),
        "theta": (
            torch.rand(batch, sequence, heads, 1, device="cuda", generator=generator)
            * 0.05
            + 0.05
        ).requires_grad_(),
        "alpha": (
            torch.rand(batch, sequence, heads, 1, device="cuda", generator=generator)
            * 0.05
            + 0.05
        ).requires_grad_(),
        "eta": (
            torch.rand(batch, sequence, heads, 1, device="cuda", generator=generator)
            * 0.05
            + 0.9
        ).requires_grad_(),
        "initial_state": (
            torch.randn(batch, heads, dim, dim, device="cuda", generator=generator)
            * 0.01
        ).requires_grad_(),
    }
    reference_plan = compile_mixer(
        named_mixer_recipe("titans_linear_memory_core"),
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    reference = reference_plan.execute(**operands)
    upstream_output, upstream_state = source.chunk_titans_linear_ref(
        operands["query"],
        operands["key"],
        operands["value"],
        operands["w"],
        operands["b"],
        operands["theta"],
        operands["alpha"],
        operands["eta"],
        chunk_size=16,
        initial_state=operands["initial_state"],
        output_final_state=True,
        use_chunk=True,
    )
    torch.testing.assert_close(reference.output, upstream_output, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(
        reference.final_state, upstream_state, atol=3e-5, rtol=3e-4
    )
    reference_loss = (
        reference.output.square().mean() + reference.final_state.square().mean()
    )
    upstream_loss = upstream_output.square().mean() + upstream_state.square().mean()
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    upstream_grads = torch.autograd.grad(upstream_loss, tuple(operands.values()))
    for name, actual, expected in zip(
        operands, reference_grads, upstream_grads, strict=True
    ):
        torch.testing.assert_close(
            actual,
            expected,
            atol=2e-4,
            rtol=1e-2,
            msg=lambda message: f"{name}: {message}",
        )

    library_plan = compile_mixer(
        named_mixer_recipe("titans_linear_memory_core"),
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    library = library_plan.execute(
        **{
            name: value.detach().clone().requires_grad_()
            for name, value in operands.items()
        }
    )
    torch.testing.assert_close(library.output, upstream_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        library.final_state, upstream_state, atol=1e-6, rtol=1e-6
    )


def test_ttt_linear_core_matches_pinned_upstream_outputs_states_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned FLA TTT-Linear comparison")
    source = pytest.importorskip("fla.ops.ttt.chunk")
    equation_source = pytest.importorskip("fla.ops.ttt.naive")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7823)
    batch, sequence, heads, dim = 1, 32, 2, 16
    dtype = torch.bfloat16
    query = (
        torch.randn(
            batch, sequence, heads, dim, device="cuda", dtype=dtype, generator=generator
        )
        * 0.1
    )
    key = torch.nn.functional.normalize(
        torch.randn(batch, sequence, heads, dim, device="cuda", generator=generator),
        dim=-1,
    ).to(dtype)
    value = (
        torch.randn(
            batch, sequence, heads, dim, device="cuda", dtype=dtype, generator=generator
        )
        * 0.1
    )
    operands = {
        "query": query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
        "w": (
            torch.ones(heads, dim, device="cuda", dtype=dtype)
            + torch.randn(heads, dim, device="cuda", dtype=dtype, generator=generator)
            * 0.01
        ).requires_grad_(),
        "b": (
            torch.randn(heads, dim, device="cuda", dtype=dtype, generator=generator)
            * 0.01
        ).requires_grad_(),
        "eta": (
            torch.randn(
                batch,
                sequence,
                heads,
                1,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * 0.005
        ).requires_grad_(),
        "initial_state": (
            torch.randn(batch, heads, dim, dim, device="cuda", generator=generator)
            * 0.01
        ).requires_grad_(),
        "initial_state_bias": (
            torch.randn(batch, heads, 1, dim, device="cuda", generator=generator) * 0.01
        ).requires_grad_(),
    }
    plan = compile_mixer(
        named_mixer_recipe("ttt_linear_core"),
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )
    reference = plan.execute(**operands)
    equation_output, equation_state, equation_bias_state = (
        equation_source.chunk_ttt_linear_ref(
            operands["query"],
            operands["key"],
            operands["value"],
            operands["w"],
            operands["b"],
            operands["eta"],
            mini_batch_size=16,
            initial_state=operands["initial_state"],
            initial_state_bias=operands["initial_state_bias"],
            output_final_state=True,
        )
    )
    torch.testing.assert_close(
        reference.output.float(), equation_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        reference.final_state, equation_state, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        reference.final_normalizer_state, equation_bias_state, atol=1e-2, rtol=1e-2
    )
    reference_loss = (
        reference.output.float().square().mean()
        + reference.final_state.square().mean()
        + reference.final_normalizer_state.square().mean()
    )
    upstream_loss = (
        equation_output.float().square().mean()
        + equation_state.square().mean()
        + equation_bias_state.square().mean()
    )
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    upstream_grads = torch.autograd.grad(upstream_loss, tuple(operands.values()))
    for name, actual, expected in zip(
        operands, reference_grads, upstream_grads, strict=True
    ):
        torch.testing.assert_close(
            actual,
            expected,
            atol=2e-2,
            rtol=2e-2,
            msg=lambda message: f"{name}: {message}",
        )

    library_plan = compile_mixer(
        named_mixer_recipe("ttt_linear_core"),
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )
    library_inputs = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in operands.items()
    }
    library = library_plan.execute(**library_inputs)
    kernel_inputs = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in operands.items()
    }
    kernel_output, kernel_state, kernel_bias_state = source.chunk_ttt_linear(
        kernel_inputs["query"],
        kernel_inputs["key"],
        kernel_inputs["value"],
        kernel_inputs["w"],
        kernel_inputs["b"],
        kernel_inputs["eta"],
        chunk_size=16,
        initial_state=kernel_inputs["initial_state"],
        initial_state_bias=kernel_inputs["initial_state_bias"],
        output_final_state=True,
    )
    torch.testing.assert_close(library.output, kernel_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(library.final_state, kernel_state, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        library.final_normalizer_state, kernel_bias_state, atol=1e-6, rtol=1e-6
    )
    library_loss = (
        library.output.float().square().mean()
        + library.final_state.square().mean()
        + library.final_normalizer_state.square().mean()
    )
    kernel_loss = (
        kernel_output.float().square().mean()
        + kernel_state.square().mean()
        + kernel_bias_state.square().mean()
    )
    library_grads = torch.autograd.grad(library_loss, tuple(library_inputs.values()))
    kernel_grads = torch.autograd.grad(kernel_loss, tuple(kernel_inputs.values()))
    for name, actual, expected in zip(
        operands, library_grads, kernel_grads, strict=True
    ):
        torch.testing.assert_close(
            actual,
            expected,
            atol=1e-6,
            rtol=1e-6,
            msg=lambda message: f"{name}: {message}",
        )


@pytest.mark.parametrize("recipe_name", ["rnn_core", "gru_core", "m2rnn_core"])
def test_xma_nonlinear_recurrences_match_pinned_equations_and_triton_gradients(
    recipe_name,
):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned XMA recurrent comparison")
    xma = pytest.importorskip("xma")
    from xma import KernelBackend
    from urm.compiler.unified_mixer import (
        MixerBackend,
        MixerIntent,
        compile_mixer,
    )
    from urm.frontend.mixer_recipes import named_mixer_recipe

    source_root = __import__("pathlib").Path(xma.__file__).resolve().parents[1]
    revision = (
        __import__("subprocess")
        .check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True)
        .strip()
    )
    if revision != "384ed0a7bd82ced1f40609603dd541cac5416844":
        pytest.skip("the exact pinned XMA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(
        {"rnn_core": 1142, "gru_core": 1143, "m2rnn_core": 1144}[recipe_name]
    )
    spec = named_mixer_recipe(recipe_name).spec
    operands = {
        name: tensor.detach().to(device="cuda", dtype=torch.float32).requires_grad_()
        for name, tensor in _coverage_recipe_operands(torch, spec).items()
    }
    # Use small states so the nonlinear recurrence stays away from saturation.
    for value in operands.values():
        value.data.copy_(
            torch.randn(value.shape, device="cuda", generator=generator) * 0.1
        )
    if recipe_name == "m2rnn_core":
        operands["forget_input"].data.copy_(
            torch.sigmoid(
                torch.randn(
                    operands["forget_input"].shape, device="cuda", generator=generator
                )
            )
        )

    if recipe_name == "rnn_core":
        from xma.layers.rnn import rnn as upstream

        def call(values, backend):
            return upstream(
                values["query"],
                values["weight"],
                input_state=values["initial_state"],
                kernel_backend=backend,
            )
    elif recipe_name == "gru_core":
        from xma.layers.gru import gru as upstream

        def call(values, backend):
            return upstream(
                values["query"],
                values["weight"],
                values["forget_input"],
                values["forget_weight"],
                values["reset_input"],
                values["reset_weight"],
                input_state=values["initial_state"],
                kernel_backend=backend,
            )
    else:
        from xma.layers.m2rnn import m2rnn as upstream

        def call(values, backend):
            return upstream(
                values["query"],
                values["key"],
                values["value"],
                values["weight"],
                values["forget_input"],
                input_state=values["initial_state"],
                kernel_backend=backend,
            )

    reference_plan = compile_mixer(
        named_mixer_recipe(recipe_name), intent=MixerIntent.TRAINING, dtype="float32"
    )
    reference_inputs = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in operands.items()
    }
    reference = reference_plan.execute(**reference_inputs)
    upstream_inputs = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in operands.items()
    }
    upstream_reference = call(upstream_inputs, KernelBackend.torch)
    torch.testing.assert_close(
        reference.output, upstream_reference[0], atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        reference.final_state, upstream_reference[1], atol=2e-6, rtol=2e-6
    )
    reference_loss = (
        reference.output.square().mean() + reference.final_state.square().mean()
    )
    upstream_reference_loss = (
        upstream_reference[0].square().mean() + upstream_reference[1].square().mean()
    )
    reference_grads = torch.autograd.grad(
        reference_loss, tuple(reference_inputs.values())
    )
    upstream_reference_grads = torch.autograd.grad(
        upstream_reference_loss, tuple(upstream_inputs.values())
    )
    for name, actual, expected in zip(
        operands, reference_grads, upstream_reference_grads, strict=True
    ):
        torch.testing.assert_close(
            actual,
            expected,
            atol=3e-6,
            rtol=3e-6,
            msg=lambda message: f"{name}: {message}",
        )

    library_plan = compile_mixer(
        named_mixer_recipe(recipe_name),
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    library_inputs = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in operands.items()
    }
    library = library_plan.execute(**library_inputs)
    upstream_inputs = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in operands.items()
    }
    upstream_result = call(upstream_inputs, KernelBackend.triton)
    torch.testing.assert_close(library.output, upstream_result[0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        library.final_state, upstream_result[1], atol=2e-6, rtol=2e-6
    )
    library_loss = library.output.square().mean() + library.final_state.square().mean()
    upstream_loss = (
        upstream_result[0].square().mean() + upstream_result[1].square().mean()
    )
    library_grads = torch.autograd.grad(library_loss, tuple(library_inputs.values()))
    upstream_grads = torch.autograd.grad(upstream_loss, tuple(upstream_inputs.values()))
    for name, actual, expected in zip(
        operands, library_grads, upstream_grads, strict=True
    ):
        torch.testing.assert_close(
            actual,
            expected,
            atol=3e-6,
            rtol=3e-6,
            msg=lambda message: f"{name}: {message}",
        )


@pytest.mark.parametrize(
    "recipe_name", ["polar_attention_core", "foveal_sparse_polar_attention_core"]
)
def test_atma_polar_k1_kernels_match_materialized_reference_and_gradients(recipe_name):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned ATMA Polar comparison")
    source = pytest.importorskip("kernel.polar_triton")
    atma = __import__("model.blocks", fromlist=["polar_reduce"])
    from urm.compiler.unified_mixer import (
        MixerBackend,
        MixerIntent,
        compile_mixer,
    )
    from urm.frontend.mixer_recipes import named_mixer_recipe

    source_root = __import__("pathlib").Path(source.__file__).resolve().parents[1]
    revision = (
        __import__("subprocess")
        .check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True)
        .strip()
    )
    if revision != "28bb3de8afbe7c0b00115e0fbff36afc9ad49c11":
        pytest.skip("the exact pinned ATMA source checkout is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(
        1830 if recipe_name == "polar_attention_core" else 1831
    )
    batch, heads, sequence, dim = 1, 2, 64, 16

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=generator) * 0.2

    operands = {
        "query": rand(batch, heads, sequence, dim).requires_grad_(),
        "key": rand(batch, heads, sequence, dim).requires_grad_(),
        "value": rand(batch, heads, sequence, dim).requires_grad_(),
        "n_keys": torch.arange(1, sequence + 1, device="cuda", dtype=torch.float32),
        "v_null": rand(heads, dim).requires_grad_(),
        "null_base": (rand(heads) + 2.0).requires_grad_(),
        "null_slope_raw": (rand(heads) + 0.5).requires_grad_(),
        "len_gain_raw": (rand(heads) - 1.0).requires_grad_(),
        "mag_beta_raw": (rand(heads) - 1.5).requires_grad_(),
    }
    kwargs = {}
    if recipe_name == "foveal_sparse_polar_attention_core":
        page_size = local_window = 16
        page_indices = torch.zeros(
            (batch, sequence // page_size, 2), device="cuda", dtype=torch.int32
        )
        page_counts = torch.tensor([[0, 0, 1, 2]], device="cuda", dtype=torch.int32)
        page_indices[0, 2, 0] = 0
        page_indices[0, 3, :2] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        operands.update(page_indices=page_indices, page_counts=page_counts)
        kwargs.update(page_size=page_size, local_window=local_window)
    positions = torch.arange(sequence, device="cuda")
    allowed = positions[None, :] <= positions[:, None]
    if kwargs:
        allowed = allowed & (positions[None, :] > positions[:, None] - local_window)
        allowed = allowed.unsqueeze(0).expand(batch, -1, -1).clone()
        for query_page in range(sequence // page_size):
            start, stop = query_page * page_size, (query_page + 1) * page_size
            for route_slot in range(int(page_counts[0, query_page])):
                key_page = int(page_indices[0, query_page, route_slot])
                key_start, key_stop = key_page * page_size, (key_page + 1) * page_size
                allowed[0, start:stop, key_start:key_stop] = True
    else:
        allowed = allowed.unsqueeze(0)

    def oracle(values):
        scores = torch.matmul(
            values["query"].float(), values["key"].float().transpose(-1, -2)
        ) / (dim**0.5)
        scores = scores.masked_fill(~allowed[:, None], -torch.inf)
        direction, magnitude = atma.polar_reduce(
            scores,
            values["value"],
            values["n_keys"],
            v_null=values["v_null"],
            null_base=values["null_base"],
            null_slope_raw=values["null_slope_raw"],
            len_gain_raw=values["len_gain_raw"],
            mag_beta_raw=values["mag_beta_raw"],
        )
        return direction, magnitude

    def direct(values):
        if recipe_name == "polar_attention_core":
            return source.polar_attention(
                values["query"],
                values["key"],
                values["value"],
                values["n_keys"],
                v_null=values["v_null"],
                null_base=values["null_base"],
                null_slope_raw=values["null_slope_raw"],
                len_gain_raw=values["len_gain_raw"],
                mag_beta_raw=values["mag_beta_raw"],
            )
        return source.polar_attention_sparse(
            values["query"],
            values["key"],
            values["value"],
            values["page_indices"],
            values["page_counts"],
            page_size=page_size,
            local_window=local_window,
            v_null=values["v_null"],
            null_base=values["null_base"],
            null_slope_raw=values["null_slope_raw"],
            len_gain_raw=values["len_gain_raw"],
            mag_beta_raw=values["mag_beta_raw"],
        )

    recipe = named_mixer_recipe(recipe_name)
    reference_plan = compile_mixer(recipe, intent=MixerIntent.TRAINING, dtype="float32")
    reference_inputs = {
        name: tensor.detach().clone().requires_grad_()
        if tensor.is_floating_point() and tensor.requires_grad
        else tensor.detach().clone()
        for name, tensor in operands.items()
    }
    reference = reference_plan.execute(**reference_inputs)
    oracle_inputs = {
        name: tensor.detach().clone().requires_grad_()
        if tensor.is_floating_point() and tensor.requires_grad
        else tensor.detach().clone()
        for name, tensor in operands.items()
    }
    expected = oracle(oracle_inputs)
    torch.testing.assert_close(reference.output, expected[0], atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(
        reference.auxiliary_output, expected[1], atol=2e-4, rtol=2e-4
    )

    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    library_inputs = {
        name: tensor.detach().clone().requires_grad_()
        if tensor.is_floating_point() and tensor.requires_grad
        else tensor.detach().clone()
        for name, tensor in operands.items()
    }
    library = library_plan.execute(**library_inputs, **kwargs)
    direct_inputs = {
        name: tensor.detach().clone().requires_grad_()
        if tensor.is_floating_point() and tensor.requires_grad
        else tensor.detach().clone()
        for name, tensor in operands.items()
    }
    upstream = direct(direct_inputs)
    torch.testing.assert_close(library.output, upstream[0], atol=0, rtol=0)
    torch.testing.assert_close(library.auxiliary_output, upstream[1], atol=0, rtol=0)
    direction_weight, magnitude_weight = (
        rand(*library.output.shape),
        rand(*library.auxiliary_output.shape),
    )
    losses = (
        (reference.output * direction_weight).sum()
        + (reference.auxiliary_output * magnitude_weight).sum(),
        (expected[0] * direction_weight).sum() + (expected[1] * magnitude_weight).sum(),
        (library.output * direction_weight).sum()
        + (library.auxiliary_output * magnitude_weight).sum(),
        (upstream[0] * direction_weight).sum() + (upstream[1] * magnitude_weight).sum(),
    )
    reference_grads = torch.autograd.grad(
        losses[0],
        tuple(
            reference_inputs[name]
            for name in operands
            if reference_inputs[name].requires_grad
        ),
    )
    oracle_grads = torch.autograd.grad(
        losses[1],
        tuple(
            oracle_inputs[name]
            for name in operands
            if oracle_inputs[name].requires_grad
        ),
    )
    library_grads = torch.autograd.grad(
        losses[2],
        tuple(
            library_inputs[name]
            for name in operands
            if library_inputs[name].requires_grad
        ),
    )
    upstream_grads = torch.autograd.grad(
        losses[3],
        tuple(
            direct_inputs[name]
            for name in operands
            if direct_inputs[name].requires_grad
        ),
    )
    grad_names = [name for name in operands if reference_inputs[name].requires_grad]
    for name, actual, expected_grad in zip(
        grad_names, reference_grads, oracle_grads, strict=True
    ):
        torch.testing.assert_close(
            actual,
            expected_grad,
            atol=3e-3,
            rtol=3e-3,
            msg=lambda m: f"equation {name}: {m}",
        )
    for name, actual, expected_grad in zip(
        grad_names, library_grads, upstream_grads, strict=True
    ):
        torch.testing.assert_close(
            actual, expected_grad, atol=0, rtol=0, msg=lambda m: f"adapter {name}: {m}"
        )


def test_k1_attention_handles_gqa_causality_bias_and_backward():
    torch = _torch()
    torch.manual_seed(11)
    q = torch.randn(2, 4, 4, 3, requires_grad=True)
    k = torch.randn(2, 6, 2, 3, requires_grad=True)
    v = torch.randn(2, 6, 2, 5, requires_grad=True)
    bias_source = torch.randn(2, 1, 4, 6, requires_grad=True)
    bias = bias_source * 0.05
    result = compile_mixer(
        softmax_attention_spec("gqa", score_bias=True),
        intent=MixerIntent.TRAINING,
    ).execute(query=q, key=k, value=v, score_bias=bias)

    assert result.output.shape == (2, 4, 4, 5)
    result.output.square().mean().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    assert bias_source.grad is not None


def test_k1_sdpa_library_anchor_matches_reference():
    torch = _torch()
    torch.manual_seed(111)
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 5, 1, 4)
    value = torch.randn(1, 5, 1, 6)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[:, 0] = False
    spec = softmax_attention_spec("gqa")
    reference = compile_mixer(spec).execute(
        query=query, key=key, value=value, attention_mask=mask
    )
    library = compile_mixer(spec, backend=MixerBackend.LIBRARY).execute(
        query=query, key=key, value=value, attention_mask=mask
    )
    torch.testing.assert_close(library.output, reference.output, atol=2e-5, rtol=2e-5)
    assert library.metadata["execution"] == "trusted_library_anchor"
    assert library.metadata["enable_gqa"] is True


def test_k1_sdpa_preserves_fused_causal_prefill_and_cached_decode_alignment():
    torch = _torch()
    torch.manual_seed(112)
    prefill_q = torch.randn(1, 5, 4, 8)
    prefill_k = torch.randn(1, 5, 2, 8)
    prefill_v = torch.randn(1, 5, 2, 6)
    spec = softmax_attention_spec("gqa_prefill")
    prefill = compile_mixer(spec, backend=MixerBackend.LIBRARY).execute(
        query=prefill_q, key=prefill_k, value=prefill_v
    )
    assert prefill.metadata["causal_strategy"] == "sdpa_is_causal"

    decode_q = torch.randn(1, 1, 4, 8)
    decode_k = torch.randn(1, 7, 2, 8)
    decode_v = torch.randn(1, 7, 2, 6)
    decode_reference = compile_mixer(spec).execute(
        query=decode_q, key=decode_k, value=decode_v
    )
    decode_library = compile_mixer(spec, backend=MixerBackend.LIBRARY).execute(
        query=decode_q, key=decode_k, value=decode_v
    )
    torch.testing.assert_close(
        decode_library.output, decode_reference.output, atol=2e-5, rtol=2e-5
    )
    assert decode_library.metadata["causal_strategy"] == "single_query_cached_decode"
    assert decode_library.metadata["enable_gqa"] is True


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_k1_sdpa_training_matches_reference_outputs_and_gradients(dtype_name):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("SDPA mixed-precision backward parity requires CUDA")
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(113)
    operands = {
        "query": torch.randn(1, 16, 4, 32, device="cuda", dtype=dtype),
        "key": torch.randn(1, 16, 2, 32, device="cuda", dtype=dtype),
        "value": torch.randn(1, 16, 2, 32, device="cuda", dtype=dtype),
    }
    operands = {
        name: tensor.detach().requires_grad_() for name, tensor in operands.items()
    }
    spec = softmax_attention_spec("sdpa_training")
    reference = compile_mixer(spec, intent="training", dtype=dtype_name).execute(
        **operands
    )
    reference_grads = torch.autograd.grad(
        reference.output.float().square().mean(), tuple(operands.values())
    )
    library = compile_mixer(
        spec,
        intent="training",
        backend=MixerBackend.LIBRARY,
        dtype=dtype_name,
    ).execute(**operands)
    library_grads = torch.autograd.grad(
        library.output.float().square().mean(), tuple(operands.values())
    )
    torch.testing.assert_close(
        library.output.float(), reference.output.float(), atol=5e-2, rtol=5e-2
    )
    for actual, expected in zip(library_grads, reference_grads):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=8e-2, rtol=8e-2
        )


def test_dsa_attention_core_matches_pinned_upstream_with_precomputed_routes():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned DSA attention comparison")
    source = pytest.importorskip("fla.ops.dsa.naive")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7714)
    batch, sequence, heads, key_dim, value_dim, topk = 1, 64, 2, 32, 16, 8
    dtype = torch.bfloat16
    query = torch.randn(
        batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator
    )
    key = torch.randn(
        batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator
    )
    value = torch.randn(
        batch,
        sequence,
        heads,
        value_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    positions = torch.arange(sequence, device="cuda")
    indices = (
        torch.rand(batch, sequence, topk, device="cuda", generator=generator)
        * (positions[None, :, None] + 1)
    ).long()
    attention_mask = torch.zeros(
        batch, sequence, sequence, dtype=torch.bool, device="cuda"
    )
    attention_mask.scatter_(2, indices, True)
    operands = {
        "query": query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
        "attention_mask": attention_mask[:, None],
    }
    recipe = named_mixer_recipe("dsa_attention_core")
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    ).execute(**operands)
    upstream_output = source.naive_dsa(
        operands["query"],
        operands["key"],
        operands["value"],
        q_idx=None,
        k_idx=None,
        indices=indices,
        topk=topk,
        scale=key_dim**-0.5,
    )
    torch.testing.assert_close(reference.output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(library.output, upstream_output, atol=2e-2, rtol=2e-2)
    reference_grads = torch.autograd.grad(
        reference.output.float().square().mean(),
        tuple(operands[name] for name in ("query", "key", "value")),
    )
    upstream_grads = torch.autograd.grad(
        upstream_output.float().square().mean(),
        tuple(operands[name] for name in ("query", "key", "value")),
    )
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )


def test_nsa_selected_attention_core_matches_pinned_upstream_routes():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned NSA comparison")
    source = pytest.importorskip("fla.ops.nsa.parallel")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7715)
    batch, sequence, query_heads, key_dim, value_dim = 1, 128, 16, 32, 32
    block_size, block_topk = 32, 2
    dtype = torch.bfloat16
    query = torch.randn(
        batch,
        sequence,
        query_heads,
        key_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    key = torch.randn(
        batch, sequence, 1, key_dim, device="cuda", dtype=dtype, generator=generator
    )
    value = torch.randn(
        batch, sequence, 1, value_dim, device="cuda", dtype=dtype, generator=generator
    )
    positions = torch.arange(sequence, device="cuda")
    current_block = (positions // block_size).view(1, sequence, 1, 1)
    block_indices = torch.cat(
        (
            current_block.expand(batch, -1, -1, -1),
            (current_block - 1).expand(batch, -1, -1, -1),
        ),
        dim=-1,
    )
    block_counts = (positions // block_size + 1).clamp(max=block_topk)
    block_counts = block_counts.view(1, sequence, 1).expand(batch, -1, -1)
    block_tokens = block_indices[:, :, 0, :, None] * block_size + torch.arange(
        block_size, device="cuda"
    )
    valid_tokens = (
        (block_tokens <= positions[None, :, None, None])
        & (block_tokens < sequence)
        & (torch.arange(block_topk, device="cuda")[None, None, :] < block_counts)[
            ..., None
        ]
    )
    key_positions = torch.arange(sequence, device="cuda").view(1, 1, 1, 1, sequence)
    attention_mask = (
        (key_positions == block_tokens[..., None]) & valid_tokens[..., None]
    ).any(dim=(2, 3))
    operands = {
        "query": query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
        "attention_mask": attention_mask[:, None],
    }
    gate = torch.ones(batch, sequence, query_heads, device="cuda", dtype=dtype)
    recipe = named_mixer_recipe("nsa_selected_attention_core")
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    ).execute(**operands)
    upstream_output = source.parallel_nsa(
        operands["query"],
        operands["key"],
        operands["value"],
        g_slc=gate,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        window_size=0,
        scale=key_dim**-0.5,
    )
    torch.testing.assert_close(reference.output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(library.output, upstream_output, atol=2e-2, rtol=2e-2)
    differentiated = tuple(operands[name] for name in ("query", "key", "value"))
    reference_grads = torch.autograd.grad(
        reference.output.float().square().mean(), differentiated
    )
    upstream_grads = torch.autograd.grad(
        upstream_output.float().square().mean(), differentiated
    )
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )


def test_fox_attention_core_matches_pinned_upstream_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned FoX attention comparison")
    source = pytest.importorskip("fla.ops.forgetting_attn.parallel")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7716)
    batch, sequence, heads, key_dim, value_dim = 1, 64, 2, 32, 32
    dtype = torch.bfloat16
    query = torch.randn(
        batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator
    )
    key = torch.randn(
        batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator
    )
    value = torch.randn(
        batch,
        sequence,
        heads,
        value_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    log_decay = (
        -torch.rand(
            batch, sequence, heads, device="cuda", dtype=dtype, generator=generator
        )
        * 0.1
    )
    operands = {
        "query": query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
        "log_decay": log_decay.requires_grad_(),
    }

    def score_bias_from_gate(gate):
        cumulative_decay = gate.float().cumsum(dim=1).transpose(1, 2)
        return cumulative_decay.unsqueeze(-1) - cumulative_decay.unsqueeze(-2)

    equation = compile_mixer(
        named_mixer_recipe("fox"), intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(
        **{name: operands[name] for name in ("query", "key", "value")},
        score_bias=score_bias_from_gate(operands["log_decay"]),
    )
    library = compile_mixer(
        named_mixer_recipe("fox"),
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    ).execute(**operands)
    upstream_output = source.parallel_forgetting_attn(
        operands["query"],
        operands["key"],
        operands["value"],
        operands["log_decay"],
        scale=key_dim**-0.5,
    )
    torch.testing.assert_close(equation.output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(library.output, upstream_output, atol=2e-2, rtol=2e-2)
    differentiated = tuple(operands.values())
    equation_grads = torch.autograd.grad(
        equation.output.float().square().mean(), differentiated
    )
    library_grads = torch.autograd.grad(
        library.output.float().square().mean(), differentiated
    )
    upstream_grads = torch.autograd.grad(
        upstream_output.float().square().mean(), differentiated
    )
    for actual, expected in zip(equation_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )


def test_parallax_attention_core_matches_pinned_upstream_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned Parallax attention comparison")
    source = pytest.importorskip("fla.ops.parallax.parallel")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7717)
    batch, sequence, heads, key_dim = 1, 64, 2, 32
    dtype = torch.bfloat16
    query = torch.randn(
        batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator
    )
    secondary_query = (
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.1
    )
    key = torch.randn(
        batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator
    )
    value = torch.randn(
        batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator
    )
    operands = {
        "query": query.requires_grad_(),
        "r": secondary_query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
    }
    equation = compile_mixer(
        named_mixer_recipe("parallax_attention_core"),
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    ).execute(**operands)
    library_plan = compile_mixer(
        named_mixer_recipe("parallax_attention_core"),
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "fla_parallel_parallax_adapter"
    library = library_plan.execute(**operands)
    upstream_output = source.parallel_parallax(
        operands["query"],
        operands["r"],
        operands["key"],
        operands["value"],
        scale=key_dim**-0.5,
    )
    torch.testing.assert_close(equation.output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(library.output, upstream_output, atol=2e-2, rtol=2e-2)
    differentiated = tuple(operands.values())
    equation_grads = torch.autograd.grad(
        equation.output.float().square().mean(), differentiated
    )
    library_grads = torch.autograd.grad(
        library.output.float().square().mean(), differentiated
    )
    upstream_grads = torch.autograd.grad(
        upstream_output.float().square().mean(), differentiated
    )
    for actual, expected in zip(equation_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )


def test_wall_attention_core_matches_pinned_upstream_outputs_and_gradients(monkeypatch):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned Wall attention comparison")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    source = pytest.importorskip("fla.ops.wall_attn.parallel")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7718)
    batch, sequence, query_heads, kv_heads, key_dim, value_dim = 1, 48, 2, 1, 32, 16
    query = torch.randn(
        batch,
        sequence,
        query_heads,
        key_dim,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    key = torch.randn(
        batch,
        sequence,
        kv_heads,
        key_dim,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    value = torch.randn(
        batch,
        sequence,
        kv_heads,
        value_dim,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    gate = (
        -torch.rand(
            batch,
            sequence,
            query_heads,
            key_dim,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.03
    )
    operands = {
        "query": query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
        "g": gate.requires_grad_(),
    }
    equation = compile_mixer(
        named_mixer_recipe("wall_attention_core"),
        intent=MixerIntent.TRAINING,
        dtype="float32",
    ).execute(**operands)
    library_plan = compile_mixer(
        named_mixer_recipe("wall_attention_core"),
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="float32",
    )
    assert library_plan.anchor == "fla_parallel_wall_attention_adapter"
    library = library_plan.execute(**operands)
    upstream_output = source.parallel_wall_attn(
        operands["query"],
        operands["key"],
        operands["value"],
        operands["g"],
        scale=key_dim**-0.5,
    )
    torch.testing.assert_close(equation.output, upstream_output, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(library.output, upstream_output, atol=5e-3, rtol=5e-3)
    differentiated = tuple(operands.values())
    equation_grads = torch.autograd.grad(
        equation.output.square().mean(), differentiated
    )
    library_grads = torch.autograd.grad(library.output.square().mean(), differentiated)
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(), differentiated
    )
    for actual, expected in zip(equation_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def _moba_selected_attention_mask(query, key, chunk_size: int, topk: int):
    torch = _torch()
    batch, sequence, heads, _ = query.shape
    if batch != 1 or sequence % chunk_size:
        raise ValueError("the MoBA fixture requires one evenly chunked sequence")
    num_chunks = sequence // chunk_size
    target_chunks = num_chunks - 1
    block_keys = key[0].view(num_chunks, chunk_size, heads, -1)[:-1].mean(dim=1).float()
    gate = torch.einsum("nhd,thd->nht", block_keys, query[0].float())
    positions = torch.arange(sequence, device=query.device)
    block_ends = (torch.arange(target_chunks, device=query.device) + 1) * chunk_size
    gate.masked_fill_(
        positions[None, None, :] < block_ends[:, None, None], -float("inf")
    )
    selected_count = min(topk - 1, target_chunks)
    selected_indices = torch.topk(
        gate, k=selected_count, dim=0, largest=True, sorted=False
    ).indices
    finite_routes = ~torch.isinf(gate)
    selected = torch.zeros_like(finite_routes).scatter_(0, selected_indices, True)
    selected &= finite_routes
    q_positions = positions[:, None]
    k_positions = positions[None, :]
    local_causal = (q_positions // chunk_size == k_positions // chunk_size) & (
        k_positions <= q_positions
    )
    mask = (
        local_causal.view(1, 1, sequence, sequence).expand(batch, heads, -1, -1).clone()
    )
    for chunk_idx in range(target_chunks):
        start = chunk_idx * chunk_size
        stop = start + chunk_size
        mask[:, :, :, start:stop] |= selected[chunk_idx].view(1, heads, sequence, 1)
    return mask


def test_moba_selected_attention_core_matches_pinned_upstream_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned MoBA attention comparison")
    source = pytest.importorskip("fla.ops.moba.parallel")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7719)
    batch, sequence, heads, key_dim = 1, 128, 2, 32
    chunk_size, topk = 32, 3
    dtype = torch.bfloat16
    query = torch.randn(
        batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator
    )
    key = torch.randn(
        batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator
    )
    value = torch.randn(
        batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator
    )
    operands = {
        "query": query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
        "attention_mask": _moba_selected_attention_mask(query, key, chunk_size, topk),
    }
    cu_seqlens = torch.tensor([0, sequence], device="cuda", dtype=torch.int32)
    recipe = named_mixer_recipe("moba_selected_attention_core")
    equation = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "fla_parallel_moba_adapter"
    try:
        library = library_plan.execute(
            **operands,
            cu_seqlens=cu_seqlens,
            max_seqlen=sequence,
            chunk_size=chunk_size,
            topk=topk,
        )
        upstream_output = source.parallel_moba(
            operands["query"],
            operands["key"],
            operands["value"],
            cu_seqlens,
            max_seqlen=sequence,
            chunk_size=chunk_size,
            topk=topk,
        )
    except RuntimeError as error:
        if (
            "locally narrowed FlashAttention comparator only includes BF16 causal D=32"
            in str(error)
        ):
            pytest.skip(
                "the installed FlashAttention build excludes MoBA's noncausal block call"
            )
        raise
    torch.testing.assert_close(equation.output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(library.output, upstream_output, atol=2e-2, rtol=2e-2)
    differentiated = tuple(operands[name] for name in ("query", "key", "value"))
    equation_grads = torch.autograd.grad(
        equation.output.float().square().mean(), differentiated
    )
    library_grads = torch.autograd.grad(
        library.output.float().square().mean(), differentiated
    )
    upstream_grads = torch.autograd.grad(
        upstream_output.float().square().mean(), differentiated
    )
    for actual, expected in zip(equation_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )


def test_bdh_attention_core_matches_pinned_upstream_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned BDH attention comparison")
    source = pytest.importorskip("bdh")
    from pathlib import Path
    import subprocess

    from urm.adapters.bdh import EXPECTED_BDH_REVISION, bdh_source_identity

    identity = bdh_source_identity()
    source_root = Path(source.__file__).resolve().parent
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        revision != EXPECTED_BDH_REVISION
        or identity["revision"] != EXPECTED_BDH_REVISION
    ):
        pytest.skip("the exact pinned BDH source checkout is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(63063)
    batch, sequence, heads, dim, value_dim = 1, 32, 2, 16, 8
    base_query = torch.randn(
        batch, sequence, heads, dim, device="cuda", generator=generator
    )
    base_value = torch.randn(
        batch, sequence, heads, value_dim, device="cuda", generator=generator
    )

    source_module = source.Attention(
        source.BDHConfig(
            n_embd=heads * dim,
            n_head=heads,
            mlp_internal_dim_multiplier=1,
            dropout=0.0,
        )
    ).to("cuda")

    def upstream(values):
        query = values["query"].transpose(1, 2)
        value = values["value"].transpose(1, 2)
        return source_module(query, query, value).transpose(1, 2)

    def clone_inputs():
        query = base_query.detach().clone()
        value = base_value.detach().clone()
        query.requires_grad_()
        value.requires_grad_()
        return {"query": query, "key": query, "value": value}

    recipe = named_mixer_recipe("bdh_attention_core")
    reference_plan = compile_mixer(recipe, intent=MixerIntent.TRAINING, dtype="float32")
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    assert reference_plan.spec.family is MixerKernelFamily.RECURRENCE
    assert library_plan.anchor == "bdh_attention_adapter"

    reference_inputs = clone_inputs()
    reference = reference_plan.execute(**reference_inputs)
    reference_loss = reference.output.float().square().mean()
    reference_grads = torch.autograd.grad(
        reference_loss, (reference_inputs["query"], reference_inputs["value"])
    )

    upstream_inputs = clone_inputs()
    upstream_output = upstream(upstream_inputs)
    upstream_grads = torch.autograd.grad(
        upstream_output.float().square().mean(),
        (upstream_inputs["query"], upstream_inputs["value"]),
    )

    library_inputs = clone_inputs()
    library = library_plan.execute(**library_inputs)
    library_grads = torch.autograd.grad(
        library.output.float().square().mean(),
        (library_inputs["query"], library_inputs["value"]),
    )
    torch.testing.assert_close(reference.output, upstream_output, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(library.output, upstream_output, atol=0.0, rtol=0.0)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_mom_selected_memory_core_matches_pinned_fla_outputs_state_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned MoM per-memory comparison")
    source = pytest.importorskip("fla.ops.gated_delta_rule")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(51051)
    batch, sequence, heads, key_dim, value_dim = 2, 32, 2, 16, 8
    dtype = torch.bfloat16
    base = {
        "query": torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.1,
        "key": torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.1,
        "value": torch.randn(
            batch,
            sequence,
            heads,
            value_dim,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.1,
        "beta": torch.sigmoid(
            torch.randn(batch, sequence, heads, device="cuda", generator=generator)
        ).to(dtype),
        "log_decay": -torch.rand(
            batch, sequence, heads, device="cuda", generator=generator
        )
        * 0.03,
    }

    def clone_inputs():
        return {
            name: value.detach().clone().requires_grad_()
            for name, value in base.items()
        }

    def upstream(values):
        return source.chunk_gated_delta_rule(
            values["query"],
            values["key"],
            values["value"],
            values["log_decay"],
            values["beta"],
            scale=1.0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            state_v_first=True,
        )

    recipe = named_mixer_recipe("mom_selected_memory_core")
    reference_plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    )
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )
    assert reference_plan.spec.feature_map is FeatureMap.L2_NORMALIZE
    assert reference_plan.spec.state_v_first is True

    reference_inputs, upstream_inputs, library_inputs = (
        clone_inputs(),
        clone_inputs(),
        clone_inputs(),
    )
    reference = reference_plan.execute(**reference_inputs)
    upstream_output, upstream_state = upstream(upstream_inputs)
    library = library_plan.execute(**library_inputs)
    torch.testing.assert_close(reference.output, upstream_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        reference.final_state, upstream_state, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(library.output, upstream_output, atol=0.0, rtol=0.0)
    torch.testing.assert_close(library.final_state, upstream_state, atol=0.0, rtol=0.0)

    def gradients(output, values):
        return torch.autograd.grad(
            output.float().square().mean(), tuple(values[name] for name in base)
        )

    reference_grads = gradients(reference.output, reference_inputs)
    upstream_grads = gradients(upstream_output, upstream_inputs)
    library_grads = gradients(library.output, library_inputs)
    for name, actual, expected in zip(
        base, reference_grads, upstream_grads, strict=True
    ):
        torch.testing.assert_close(
            actual,
            expected,
            atol=2e-2,
            rtol=2e-2,
            msg=lambda message: f"{name}: {message}",
        )
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_cat_attention_core_matches_pinned_fla_flex_attention_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned CAT attention comparison")
    source = pytest.importorskip("fla.models.cat.modeling_cat")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(66066)
    chunk_size = 8
    block_size = 2 + chunk_size
    sequence = 2 * block_size + 2
    batch, heads, dim = 1, 2, 16
    positions = torch.arange(sequence, device="cuda")
    mask_mod = source.get_cat_mask_mod(block_size)
    attention_mask = mask_mod(None, None, positions[:, None], positions[None, :]).view(
        1, 1, sequence, sequence
    )
    block_mask = source.create_block_mask_compiled(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=sequence,
        KV_LEN=sequence,
    )
    dtype = torch.bfloat16
    base = {
        "query": torch.randn(
            batch, sequence, heads, dim, device="cuda", dtype=dtype, generator=generator
        )
        * 0.1,
        "key": torch.randn(
            batch, sequence, heads, dim, device="cuda", dtype=dtype, generator=generator
        )
        * 0.1,
        "value": torch.randn(
            batch, sequence, heads, dim, device="cuda", dtype=dtype, generator=generator
        )
        * 0.1,
        "attention_mask": attention_mask,
    }

    def clone_inputs():
        return {
            name: value.detach().clone().requires_grad_()
            if name != "attention_mask"
            else value
            for name, value in base.items()
        }

    def upstream(values):
        return source.flex_attention_compiled(
            values["query"].transpose(1, 2),
            values["key"].transpose(1, 2),
            values["value"].transpose(1, 2),
            block_mask=block_mask,
        ).transpose(1, 2)

    recipe = named_mixer_recipe("cat_attention_core")
    reference_plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    )
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )
    reference_inputs, upstream_inputs, library_inputs = (
        clone_inputs(),
        clone_inputs(),
        clone_inputs(),
    )
    reference = reference_plan.execute(**reference_inputs)
    upstream_output = upstream(upstream_inputs)
    library = library_plan.execute(**library_inputs)
    torch.testing.assert_close(reference.output, upstream_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(library.output, upstream_output, atol=1e-2, rtol=1e-2)

    def gradients(output, values):
        return torch.autograd.grad(
            output.float().square().mean(),
            (values["query"], values["key"], values["value"]),
        )

    reference_grads = gradients(reference.output, reference_inputs)
    upstream_grads = gradients(upstream_output, upstream_inputs)
    library_grads = gradients(library.output, library_inputs)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_tda_attention_core_matches_pinned_triton_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned TDA comparison")
    source = pytest.importorskip("triton_threshold_attention")
    from urm.adapters.tda import tda_source_identity

    if tda_source_identity()["revision"] != "cd8ddc9d5b43a1dcf86f9cfda302edb5cc108da2":
        pytest.skip("the exact pinned TDA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(68068)
    batch, sequence, heads, dim = 1, 64, 2, 32
    base = {
        name: torch.randn(
            batch,
            sequence,
            heads,
            dim,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.1)
        .requires_grad_()
        for name in ("query_a", "query_b", "key_a", "key_b", "value")
    }
    base["beta"] = torch.tensor(0.7, device="cuda")
    base["lambda_weight"] = torch.tensor(0.35, device="cuda")

    def clone_inputs():
        return {
            name: value.detach().clone().requires_grad_(value.requires_grad)
            for name, value in base.items()
        }

    def upstream(values):
        return source.differential_threshold_rela_triton(
            *(
                values[name].transpose(1, 2).contiguous()
                for name in ("query_a", "query_b", "key_a", "key_b", "value")
            ),
            values["beta"],
            values["lambda_weight"],
            relu_power=2.0,
            normalize=True,
        ).transpose(1, 2)

    recipe = named_mixer_recipe("tda_attention_core")
    reference_plan = compile_mixer(recipe, intent=MixerIntent.TRAINING, dtype="float32")
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    reference_inputs, upstream_inputs, library_inputs = (
        clone_inputs(),
        clone_inputs(),
        clone_inputs(),
    )
    reference = reference_plan.execute(**reference_inputs)
    upstream_output = upstream(upstream_inputs)
    library = library_plan.execute(**library_inputs)
    torch.testing.assert_close(reference.output, upstream_output, atol=2e-4, rtol=1e-2)
    torch.testing.assert_close(library.output, upstream_output, atol=0.0, rtol=0.0)

    def gradients(output, values):
        return torch.autograd.grad(
            output.square().mean(),
            tuple(
                values[name]
                for name in ("query_a", "query_b", "key_a", "key_b", "value")
            ),
        )

    reference_grads = gradients(reference.output, reference_inputs)
    upstream_grads = gradients(upstream_output, upstream_inputs)
    library_grads = gradients(library.output, library_inputs)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-2)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_differential_attention_core_matches_pinned_microsoft_v1_outputs_and_gradients():
    import importlib
    import subprocess
    from pathlib import Path

    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned Differential Transformer comparison")
    source = pytest.importorskip("multihead_diffattn")
    module_path = Path(source.__file__).resolve()
    repository = next(
        parent for parent in module_path.parents if (parent / ".git").exists()
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "50224e387211f15ac6a3b2685730b9a0c850f145":
        pytest.skip(
            "the exact pinned Microsoft Differential Transformer source is unavailable"
        )

    class SliceProjection(torch.nn.Module):
        def __init__(self, start, width):
            super().__init__()
            self.start, self.width = start, width

        def forward(self, value):
            return value[..., self.start : self.start + self.width]

    batch, sequence, heads, dim = 1, 64, 2, 16
    width = 2 * heads * dim
    source.apply_rotary_emb = lambda value, *args, **kwargs: value
    source_layer = (
        source.MultiheadDiffAttn(embed_dim=width, depth=2, num_heads=heads)
        .to("cuda")
        .eval()
    )
    source_layer.q_proj = SliceProjection(0, width)
    source_layer.k_proj = SliceProjection(width, width)
    source_layer.v_proj = SliceProjection(2 * width, width)
    source_layer.subln = torch.nn.Identity()
    source_layer.out_proj = torch.nn.Identity()
    generator = torch.Generator(device="cuda").manual_seed(67067)
    base = {
        name: torch.randn(
            batch,
            sequence,
            width,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        ).mul_(0.1)
        for name in ("query_raw", "key_raw", "value_raw")
    }
    lambda_1 = torch.exp(
        torch.sum(source_layer.lambda_q1 * source_layer.lambda_k1).float()
    )
    lambda_2 = torch.exp(
        torch.sum(source_layer.lambda_q2 * source_layer.lambda_k2).float()
    )
    lambda_weight = lambda_1 - lambda_2 + source_layer.lambda_init

    def clone_raw():
        return {
            name: value.detach().clone().requires_grad_()
            for name, value in base.items()
        }

    def branch_operands(raw):
        query = raw["query_raw"].view(batch, sequence, heads, 2, dim)
        key = raw["key_raw"].view(batch, sequence, heads, 2, dim)
        return {
            "query_a": query[..., 0, :],
            "query_b": query[..., 1, :],
            "key_a": key[..., 0, :],
            "key_b": key[..., 1, :],
            "value": raw["value_raw"].view(batch, sequence, heads, 2 * dim),
            "lambda_weight": lambda_weight.detach(),
        }

    def upstream(raw):
        joined = torch.cat((raw["query_raw"], raw["key_raw"], raw["value_raw"]), dim=-1)
        output = source_layer(joined, (None, None))
        return output / (1.0 - source_layer.lambda_init)

    recipe = named_mixer_recipe("differential_attention_core")
    reference_plan = compile_mixer(recipe, intent=MixerIntent.TRAINING, dtype="float32")
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    upstream_raw, reference_raw, library_raw = clone_raw(), clone_raw(), clone_raw()
    upstream_output = upstream(upstream_raw)
    reference = reference_plan.execute(**branch_operands(reference_raw)).output.flatten(
        2
    )
    library = library_plan.execute(**branch_operands(library_raw)).output.flatten(2)
    torch.testing.assert_close(reference, upstream_output, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(library, upstream_output, atol=2e-6, rtol=2e-5)

    upstream_gradients = torch.autograd.grad(
        upstream_output.square().mean(), tuple(upstream_raw.values())
    )

    def plan_gradients(output, raw):
        return torch.autograd.grad(output.square().mean(), tuple(raw.values()))

    reference_gradients = plan_gradients(reference, reference_raw)
    library_gradients = plan_gradients(library, library_raw)
    # Split source gradients back into the three [Q,K,V] inputs. The equation
    # plan differentiates the corresponding head/interleave views directly.
    for label, actual, expected in zip(
        ("query", "key", "value"),
        reference_gradients,
        upstream_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual, expected, atol=2e-6, rtol=2e-5, msg=lambda m: f"{label}: {m}"
        )
    for label, actual, expected in zip(
        ("query", "key", "value"),
        library_gradients,
        upstream_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual, expected, atol=2e-6, rtol=2e-5, msg=lambda m: f"{label}: {m}"
        )


def test_tpa_attention_core_matches_pinned_t6_layer_outputs_and_gradients():
    import subprocess
    from pathlib import Path

    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned TPA comparison")
    source = pytest.importorskip("model.T6")
    source_file = Path(source.__file__).resolve()
    repository = next(
        parent for parent in source_file.parents if (parent / ".git").exists()
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != "c276c80d5ad807881dedb4707d8d3c20b4e97ec6" or dirty:
        pytest.skip("the exact clean pinned TPA source checkout is unavailable")

    layer = (
        source.CausalSelfAttention(
            source.GPTConfig(n_embd=64, n_head=2, head_dim=32, rank=4, q_rank=4)
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        layer.c_proj.weight.copy_(torch.eye(64, device="cuda"))
    batch, sequence, width = 1, 64, 64
    generator = torch.Generator(device="cuda").manual_seed(69069)
    base = torch.randn(
        batch, sequence, width, device="cuda", dtype=torch.float32, generator=generator
    ).mul_(0.1)
    recipe = named_mixer_recipe("tpa_attention_core")
    reference_plan = compile_mixer(recipe, intent=MixerIntent.TRAINING, dtype="float32")
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )

    def upstream(x):
        return layer(x)

    def planned(x, plan):
        query, key, value = layer.c_qkv(x)
        result = plan.execute(query=query, key=key, value=value).output
        return layer.c_proj(result.contiguous().view(batch, sequence, width))

    x_upstream = base.detach().clone().requires_grad_()
    x_reference = base.detach().clone().requires_grad_()
    x_library = base.detach().clone().requires_grad_()
    upstream_output = upstream(x_upstream)
    reference_output = planned(x_reference, reference_plan)
    library_output = planned(x_library, library_plan)
    torch.testing.assert_close(reference_output, upstream_output, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(library_output, upstream_output, atol=0.0, rtol=0.0)

    parameters = tuple(layer.parameters())
    reference_grads = torch.autograd.grad(
        reference_output.square().mean(), (x_reference, *parameters), retain_graph=True
    )
    library_grads = torch.autograd.grad(
        library_output.square().mean(), (x_library, *parameters), retain_graph=True
    )
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(), (x_upstream, *parameters)
    )
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_tucker_attention_core_matches_pinned_triton_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned Tucker Attention comparison")
    source = pytest.importorskip("src.attn.triton.tucker_attn")
    from urm.adapters.tucker import tucker_source_identity

    if (
        tucker_source_identity()["revision"]
        != "c3e3d3cec991f4303b824c7fb7cbb95e3748d5c7"
    ):
        pytest.skip("the exact pinned Tucker source checkout is unavailable")
    batch, sequence, heads, query_rank, key_rank, value_rank = 1, 64, 4, 16, 16, 16
    generator = torch.Generator(device="cuda").manual_seed(70070)
    base = tuple(
        (
            torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
            * 0.1
        )
        for shape in (
            (batch, sequence, query_rank),
            (batch, sequence, key_rank),
            (batch, sequence, value_rank),
            (heads, query_rank, key_rank),
        )
    )
    source_layer = source.FlashAttentionTucker(causal=False, attn_autotune=False).cuda()
    recipe = named_mixer_recipe("tucker_attention_core")
    reference_plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    )
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )

    def clones():
        return tuple(item.detach().clone().requires_grad_() for item in base)

    upstream_inputs = clones()
    reference_inputs = clones()
    library_inputs = clones()
    upstream_output = source_layer(*upstream_inputs, sm_scale=key_rank**-0.5).transpose(
        1, 2
    )
    reference_output = reference_plan.execute(
        query=reference_inputs[0],
        key=reference_inputs[1],
        value=reference_inputs[2],
        B_pre=reference_inputs[3],
    ).output
    library_output = library_plan.execute(
        query=library_inputs[0],
        key=library_inputs[1],
        value=library_inputs[2],
        B_pre=library_inputs[3],
    ).output
    torch.testing.assert_close(reference_output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(library_output, upstream_output, atol=0.0, rtol=0.0)
    upstream_grads = torch.autograd.grad(
        upstream_output.float().square().mean(), upstream_inputs
    )
    reference_grads = torch.autograd.grad(
        reference_output.float().square().mean(), reference_inputs
    )
    library_grads = torch.autograd.grad(
        library_output.float().square().mean(), library_inputs
    )
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_longformer_attention_core_matches_pinned_sliding_chunks_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned Longformer comparison")
    from urm.adapters.longformer import (
        longformer_attention_adapter,
        longformer_source_identity,
    )

    try:
        identity = longformer_source_identity()
    except ModuleNotFoundError as error:
        pytest.skip(str(error))
    assert identity["revision"] == "caefee668e39cacdece7dd603a0bebf24df6d8ca"

    batch, sequence, heads, dim, window = 1, 128, 2, 16, 16
    generator = torch.Generator(device="cuda").manual_seed(71071)
    base = tuple(
        torch.randn(
            batch,
            sequence,
            heads,
            width,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        ).mul_(0.1)
        for width in (dim, dim, 12)
    )

    def clones():
        return tuple(item.detach().clone().requires_grad_() for item in base)

    def upstream(values):
        return longformer_attention_adapter(*values, attention_window=window)[0]

    recipe = named_mixer_recipe("longformer_attention_core")
    reference_plan = compile_mixer(recipe, intent=MixerIntent.TRAINING, dtype="float32")
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    upstream_inputs, reference_inputs, library_inputs = clones(), clones(), clones()
    args = lambda items: {
        "query": items[0],
        "key": items[1],
        "value": items[2],
        "attention_window": window,
    }
    upstream_output = upstream(upstream_inputs)
    reference_output = reference_plan.execute(**args(reference_inputs)).output
    library_output = library_plan.execute(**args(library_inputs)).output
    torch.testing.assert_close(reference_output, upstream_output, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(library_output, upstream_output, atol=0.0, rtol=0.0)
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(), upstream_inputs
    )
    reference_grads = torch.autograd.grad(
        reference_output.square().mean(), reference_inputs
    )
    library_grads = torch.autograd.grad(library_output.square().mean(), library_inputs)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_kata_attention_core_matches_pinned_triton_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned KATA comparison")
    source = pytest.importorskip("kata.parallel_kata_attn")
    from urm.adapters.kata import kata_source_identity

    identity = kata_source_identity()
    assert identity["revision"] == "f93fe75750be6400a0068749794985d70666926d"
    batch, sequence, heads, dim, value_dim, groups = 1, 128, 2, 32, 32, 2
    generator = torch.Generator(device="cuda").manual_seed(73073)
    base = tuple(
        torch.randn(
            batch,
            sequence,
            heads,
            width,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.1)
        for width in (dim, dim, value_dim)
    )
    recipe = named_mixer_recipe("kata_attention_core")
    reference_plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    )
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )

    def clones():
        return tuple(value.detach().clone().requires_grad_() for value in base)

    upstream_inputs, reference_inputs, library_inputs = clones(), clones(), clones()
    upstream_output = source.parallel_kata_attn(
        *upstream_inputs, num_groups=groups, use_triton_bwd=True
    )
    reference_output = reference_plan.execute(
        query=reference_inputs[0],
        key=reference_inputs[1],
        value=reference_inputs[2],
        num_groups=groups,
    ).output
    library_output = library_plan.execute(
        query=library_inputs[0],
        key=library_inputs[1],
        value=library_inputs[2],
        num_groups=groups,
    ).output
    torch.testing.assert_close(reference_output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(library_output, upstream_output, atol=0.0, rtol=0.0)
    upstream_grads = torch.autograd.grad(
        upstream_output.float().square().mean(), upstream_inputs
    )
    reference_grads = torch.autograd.grad(
        reference_output.float().square().mean(), reference_inputs
    )
    library_grads = torch.autograd.grad(
        library_output.float().square().mean(), library_inputs
    )
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_conformer_attention_core_matches_pinned_espnet_sdpa_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned Conformer comparison")
    source = pytest.importorskip(
        "espnet2.legacy.nets.pytorch_backend.transformer.attention"
    )
    source_path = Path(source.__file__).resolve()
    repository = next(
        parent for parent in source_path.parents if (parent / ".git").exists()
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    assert revision == "2950325ea62c8052f448aaf11affdabe169ec8ab"
    assert not dirty

    batch, sequence, heads, dim = 1, 32, 2, 16
    width = heads * dim
    generator = torch.Generator(device="cuda").manual_seed(75075)
    base = torch.randn(batch, sequence, width, device="cuda", generator=generator).mul_(
        0.1
    )
    layer = (
        source.MultiHeadedAttention(
            n_head=heads,
            n_feat=width,
            dropout_rate=0.0,
            qk_norm=False,
            use_flash_attn=False,
            causal=False,
            use_sdpa=True,
        )
        .cuda()
        .eval()
    )
    recipe = named_mixer_recipe("conformer_attention_core")
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )

    def clones():
        return base.detach().clone().requires_grad_()

    upstream_x, compiled_x = clones(), clones()
    upstream_output = layer(upstream_x, upstream_x, upstream_x, mask=None)
    q, k, v = layer.forward_qkv(compiled_x, compiled_x, compiled_x)
    output = plan.execute(
        query=q.transpose(1, 2).contiguous(),
        key=k.transpose(1, 2).contiguous(),
        value=v.transpose(1, 2).contiguous(),
    ).output
    compiled_output = layer.linear_out(output.reshape(batch, sequence, width))
    torch.testing.assert_close(compiled_output, upstream_output, atol=0.0, rtol=0.0)
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(), (upstream_x, *layer.parameters())
    )
    compiled_grads = torch.autograd.grad(
        compiled_output.square().mean(), (compiled_x, *layer.parameters())
    )
    for actual, expected in zip(compiled_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_hopfield_attention_core_matches_pinned_single_update_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned Hopfield comparison")
    source = pytest.importorskip("hflayers.activation")
    source_path = Path(source.__file__).resolve()
    repository = next(
        parent for parent in source_path.parents if (parent / ".git").exists()
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    assert revision == "f56f929c95b77a070ae675ea4f56b6d54d36e730"
    assert not dirty

    batch, sequence, heads, dim = 1, 32, 2, 16
    width = heads * dim
    generator = torch.Generator(device="cuda").manual_seed(78078)
    base = torch.randn(batch, sequence, width, device="cuda", generator=generator).mul_(
        0.1
    )
    layer = (
        source.HopfieldCore(embed_dim=width, num_heads=heads, dropout=0.0, bias=True)
        .cuda()
        .eval()
    )
    recipe = named_mixer_recipe("hopfield_attention_core")
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )

    def clones():
        return base.detach().clone().requires_grad_()

    upstream_x, compiled_x = clones(), clones()
    seq_first = upstream_x.transpose(0, 1)
    upstream_output = (
        layer(
            seq_first,
            seq_first,
            seq_first,
            need_weights=False,
            scaling=1.0,
            update_steps_max=0,
        )[0]
        .transpose(0, 1)
        .contiguous()
    )
    q, k, v = torch.nn.functional.linear(
        compiled_x, layer.in_proj_weight, layer.in_proj_bias
    ).chunk(3, dim=-1)
    q, k, v = (item.reshape(batch, sequence, heads, dim) for item in (q, k, v))
    output = plan.execute(query=q, key=k, value=v).output
    compiled_output = layer.out_proj(output.reshape(batch, sequence, width))
    torch.testing.assert_close(compiled_output, upstream_output, atol=1e-6, rtol=1e-6)
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(), (upstream_x, *layer.parameters())
    )
    compiled_grads = torch.autograd.grad(
        compiled_output.square().mean(), (compiled_x, *layer.parameters())
    )
    for actual, expected in zip(compiled_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("score_type", ("dot_product", "idw"))
def test_fwpkm_memory_read_core_matches_pinned_source_outputs_and_gradients(score_type):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned FwPKM comparison")
    source = pytest.importorskip("src.models.fwpkm.fwpkm")
    source_path = Path(source.__file__).resolve()
    repository = next(
        parent for parent in source_path.parents if (parent / ".git").exists()
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    assert revision == "b1c8e234b523d70245fa197eed4b80a985c413a8"
    assert not dirty

    generator = torch.Generator(device="cuda").manual_seed(56056)
    batch, sequence, heads, key_dim, value_dim, topk, subsize = 2, 16, 2, 16, 16, 4, 32
    layer = source.FastWeightProductKeyMemory(
        mem_k_dim=key_dim,
        mem_v_dim=value_dim,
        mem_heads=heads,
        mem_topk=topk,
        mem_n_subkeys=subsize,
        qk_score_type=score_type,
        score_nonlinear="softmax",
        score_temperature=1.0,
        addr_loss=None,
    ).cuda()
    layer.reset_parameters()
    base_query = torch.randn(
        batch, sequence, heads * key_dim, device="cuda", generator=generator
    ).mul_(0.1)
    recipe = named_mixer_recipe("fwpkm_memory_read_core")
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )

    upstream_query = base_query.detach().clone().requires_grad_()
    upstream_keys = layer.keys.detach().clone().requires_grad_()
    upstream_values = layer.values.detach().clone().requires_grad_()
    upstream_output = layer.retrieve_values(
        upstream_query,
        {"keys": upstream_keys, "values": upstream_values},
    )["retireved_values"]

    compiled_query = base_query.detach().clone().requires_grad_()
    compiled_keys = layer.keys.detach().clone().requires_grad_()
    compiled_values = layer.values.detach().clone().requires_grad_()
    scores, indices, *_ = layer.get_indices(
        compiled_query.reshape(batch * sequence, heads, key_dim), compiled_keys
    )
    selected_values = compiled_values.index_select(0, indices.reshape(-1)).reshape(
        batch * sequence, heads * topk, 1, value_dim
    )
    score_logits = scores.reshape(batch * sequence, heads * topk, 1, 1)
    unit_query = torch.ones(
        batch * sequence, 1, 1, 1, device="cuda", dtype=score_logits.dtype
    )
    compiled_output = plan.execute(
        query=unit_query, key=score_logits, value=selected_values
    ).output.reshape(batch * sequence, value_dim)

    torch.testing.assert_close(compiled_output, upstream_output, atol=1e-6, rtol=1e-5)
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(),
        (upstream_query, upstream_keys, upstream_values),
    )
    compiled_grads = torch.autograd.grad(
        compiled_output.square().mean(),
        (compiled_query, compiled_keys, compiled_values),
    )
    for actual, expected in zip(compiled_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_samba_attention_core_matches_pinned_nope_attention_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned Samba comparison")
    from urm.adapters.samba import load_samba_attention, samba_source_root

    source_root = samba_source_root()
    if source_root is None:
        pytest.skip("the pinned Samba checkout is unavailable")
    source_class, identity = load_samba_attention(source_root)
    assert identity["revision"] == "617c7a0f8c71f1b7cb6180b86f9543d146f5c66f"

    batch, sequence, heads, dim = 1, 32, 2, 16
    width = heads * dim
    config = type(
        "NopeSambaConfig",
        (),
        {
            "full_per_layer": 1_000_000,
            "head_size": dim,
            "n_head": heads,
            "n_query_groups": heads,
            "bias": False,
            "sc_attn": False,
            "nope": True,
            "local_window": 2048,
        },
    )()
    layer = source_class(config, layer_idx=1, n_embd=width).cuda().eval()
    generator = torch.Generator(device="cuda").manual_seed(53053)
    base = torch.randn(batch, sequence, width, device="cuda", generator=generator).mul_(
        0.1
    )
    recipe = named_mixer_recipe("samba_attention_core")
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    upstream_x = base.detach().clone().requires_grad_()
    compiled_x = base.detach().clone().requires_grad_()
    upstream_output = layer(
        upstream_x, rope=(None, None), max_seq_length=sequence, mask=None
    )[0]
    qkv = layer.attn(compiled_x)
    q_per_kv = layer.n_head // layer.n_query_groups
    qkv = qkv.view(batch, sequence, layer.n_query_groups, q_per_kv + 2, dim)
    query, key, value = qkv.split((q_per_kv, 1, 1), dim=-2)
    query = query.reshape(batch, sequence, heads, dim)
    key = key.reshape(batch, sequence, heads, dim)
    value = value.reshape(batch, sequence, heads, dim)
    attention = plan.execute(query=query, key=key, value=value).output
    compiled_output = layer.proj(attention.reshape(batch, sequence, width))

    torch.testing.assert_close(compiled_output, upstream_output, atol=0.0, rtol=0.0)
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(), (upstream_x, *layer.parameters())
    )
    compiled_grads = torch.autograd.grad(
        compiled_output.square().mean(), (compiled_x, *layer.parameters())
    )
    for actual, expected in zip(compiled_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_h3_ssm_fft_core_matches_pinned_source_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned H3 comparison")
    source = pytest.importorskip("src.models.ssm.h3")
    source_path = Path(source.__file__).resolve()
    repository = next(
        parent for parent in source_path.parents if (parent / ".git").exists()
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    assert revision == "5c4d06b5795405170387c80998b58d76179a8a1a"
    assert not dirty

    batch, sequence, width = 1, 32, 4
    layer = (
        source.H3(
            d_model=width,
            d_state=4,
            l_max=sequence,
            head_dim=1,
            use_fast_fftconv=False,
        )
        .cuda()
        .eval()
    )
    generator = torch.Generator(device="cuda").manual_seed(76076)
    base = torch.randn(batch, sequence, width, device="cuda", generator=generator).mul_(
        0.1
    )
    recipe = named_mixer_recipe("h3_ssm_fft_core")
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )

    upstream_x = base.detach().clone().requires_grad_()
    compiled_x = base.detach().clone().requires_grad_()
    upstream_output = layer(upstream_x)

    qkv_input = compiled_x.reshape(batch * sequence, width).transpose(0, 1)
    q, k, v = (
        weight @ qkv_input + bias.unsqueeze(-1)
        for weight, bias in (
            (layer.q_proj.weight, layer.q_proj.bias),
            (layer.k_proj.weight, layer.k_proj.bias),
            (layer.v_proj.weight, layer.v_proj.bias),
        )
    )
    q, k, v = (
        item.reshape(width, batch, sequence).permute(1, 2, 0).unsqueeze(-1)
        for item in (q, k, v)
    )
    ssm_kernel = layer.kernel(L=sequence, state=None, rate=1.0)[0].squeeze(0)
    ssm_k_kernel = layer.ssm_k_kernel(L=sequence, state=None, rate=1.0)[0].squeeze(0)
    result = plan.execute(
        query=q,
        key=k,
        value=v,
        ssm_kernel=ssm_kernel,
        ssm_k_kernel=ssm_k_kernel,
        ssm_k_direct=layer.ssm_k_D,
        skip=layer.D,
    )
    compiled_output = layer.output_linear(result.output.squeeze(-1))

    torch.testing.assert_close(compiled_output, upstream_output, atol=1e-6, rtol=1e-5)
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(), (upstream_x, *layer.parameters())
    )
    compiled_grads = torch.autograd.grad(
        compiled_output.square().mean(), (compiled_x, *layer.parameters())
    )
    for actual, expected in zip(compiled_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_hyena_fftconv_core_matches_pinned_source_outputs_and_gradients():
    torch = _torch()
    source = pytest.importorskip("standalone_hyena")
    source_path = Path(source.__file__).resolve()
    repository = next(
        parent for parent in source_path.parents if (parent / ".git").exists()
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    assert revision == "02220c69d247e5473616cd053a443ad99fd2559b"
    assert not dirty
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned Hyena comparison")

    batch, sequence, width = 1, 16, 8
    layer = (
        source.HyenaOperator(
            d_model=width,
            l_max=sequence,
            order=2,
            filter_order=8,
            dropout=0.0,
            filter_dropout=0.0,
        )
        .cuda()
        .eval()
    )
    recipe = named_mixer_recipe("hyena_fftconv_core")
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    base = torch.randn(batch, sequence, width, device="cuda")
    upstream_x = base.detach().clone().requires_grad_()
    compiled_x = base.detach().clone().requires_grad_()
    upstream_output = layer(upstream_x)

    projected = layer.in_proj(compiled_x).transpose(1, 2)
    filtered = layer.short_filter(projected)[..., :sequence]
    x0, x1, value = filtered.split(width, dim=1)
    raw_kernel = layer.filter_fn.filter(sequence)[0]
    mixer = plan.execute(
        query=(value * x1).transpose(1, 2),
        kernel=raw_kernel.transpose(0, 1).contiguous(),
        direct=layer.filter_fn.bias,
    ).output.transpose(1, 2)
    compiled_output = layer.out_proj((mixer * x0).transpose(1, 2))

    torch.testing.assert_close(compiled_output, upstream_output, atol=2e-6, rtol=2e-5)
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(), (upstream_x, *layer.parameters())
    )
    compiled_grads = torch.autograd.grad(
        compiled_output.square().mean(), (compiled_x, *layer.parameters())
    )
    for actual, expected in zip(compiled_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_hla_second_order_core_matches_pinned_paper_equation_and_gradients():
    import hashlib

    torch = _torch()
    repository = Path("/tmp/urm-comparator-pins/hla")
    if not (repository / "HLA.pdf").is_file():
        pytest.skip("the pinned HLA paper checkout is unavailable")
    assert (
        subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        == "484fef2bb40d4ed58f7656e545cf5ef64c40c962"
    )
    paper_path = repository / "HLA.pdf"
    assert hashlib.sha256(paper_path.read_bytes()).hexdigest() == (
        "574242e6b6694e1ef87440f588cbe364517b76bcd9e771b2fb93b0d38feb822b"
    )

    torch.manual_seed(74074)
    batch, sequence, heads, key_dim, value_dim = 2, 12, 2, 5, 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = tuple(
        torch.randn(batch, sequence, heads, width, device=device).mul_(0.1)
        for width in (key_dim, key_dim, value_dim)
    )
    plan = compile_mixer(
        named_mixer_recipe("hla_second_order_core"),
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    expected_inputs = tuple(value.detach().clone().requires_grad_() for value in base)
    actual_inputs = tuple(value.detach().clone().requires_grad_() for value in base)

    def dense_paper_equation(query, key, value):
        scores = torch.matmul(
            query.transpose(1, 2), key.transpose(1, 2).transpose(-1, -2)
        )
        causal = torch.ones(sequence, sequence, dtype=torch.bool, device=device).tril()
        masked_affinity = scores.masked_fill(~causal, 0.0)
        second_order = torch.matmul(masked_affinity, masked_affinity.transpose(-1, -2))
        second_order = second_order.masked_fill(~causal, 0.0)
        output = torch.matmul(second_order, value.transpose(1, 2))
        return output.transpose(1, 2)

    expected = dense_paper_equation(*expected_inputs)
    actual = plan.execute(
        query=actual_inputs[0], key=actual_inputs[1], value=actual_inputs[2]
    ).output
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    expected_grads = torch.autograd.grad(expected.square().mean(), expected_inputs)
    actual_grads = torch.autograd.grad(actual.square().mean(), actual_inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize(
    "mode,context", [("all", None), ("local", 32), ("strided", 8), ("fixed", 64)]
)
def test_sparse_transformer_dense_source_mask_modes_match_k1(mode, context):
    torch = _torch()
    tensorflow = pytest.importorskip("tensorflow")
    from urm.adapters.sparse_transformer import fixed_mode_mask, load_dense_attention

    if not Path("/tmp/urm-comparator-pins/sparse_transformer/attention.py").is_file():
        pytest.skip("the pinned sparse-attention checkout is unavailable")
    source, identity = load_dense_attention()
    assert identity["revision"] == "c53f3bdbf6225be0582f0357072e82b13c69be7d"
    if torch.cuda.is_available():
        devices = tensorflow.config.list_physical_devices("GPU")
        if devices:
            try:
                tensorflow.config.experimental.set_memory_growth(devices[0], True)
            except RuntimeError:
                pass

    batch, sequence, heads, head_dim = 1, 128, 2, 16
    generator = torch.Generator(device="cpu").manual_seed(72072)
    base = tuple(
        torch.randn(batch, sequence, heads * head_dim, generator=generator).mul_(0.1)
        for _ in range(3)
    )
    source_inputs = tuple(tensorflow.Variable(value.numpy()) for value in base)
    if mode == "fixed":
        source_mask = fixed_mode_mask(
            source,
            n_ctx=sequence,
            heads=heads,
            block_size=32,
            local_attn_ctx=context,
            num_verts=2,
            vertsize=1,
        )

        def source_fixed_equation(q, k, v):
            q, k, v = (source.split_heads(tensor, heads) for tensor in (q, k, v))
            scores = tensorflow.matmul(q, k, transpose_b=True) * (head_dim**-0.5)
            mask_tensor = tensorflow.convert_to_tensor(
                source_mask[None], dtype=tensorflow.float32
            )
            weights = scores * mask_tensor + -1e9 * (1.0 - mask_tensor)
            weights = tensorflow.nn.softmax(weights)
            return source.merge_heads(tensorflow.matmul(weights, v))

        source_call = lambda: source_fixed_equation(*source_inputs)
        torch_mask_numpy = source_mask[None]
    else:
        source_call = lambda: source.attention_impl(
            *source_inputs, heads=heads, attn_mode=mode, local_attn_ctx=context
        )
        torch_mask_numpy = source.get_attn_mask(sequence, mode, context).numpy()
    with tensorflow.GradientTape() as tape:
        source_output = source_call()
        source_loss = tensorflow.reduce_mean(tensorflow.square(source_output))
    source_grads = tape.gradient(source_loss, source_inputs)

    mask = torch.from_numpy(torch_mask_numpy).to(dtype=torch.bool)
    recipe = named_mixer_recipe("sparse_attention_core")
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    actual_inputs = tuple(
        value.reshape(batch, sequence, heads, head_dim)
        .detach()
        .clone()
        .requires_grad_()
        for value in base
    )
    actual = plan.execute(
        query=actual_inputs[0],
        key=actual_inputs[1],
        value=actual_inputs[2],
        attention_mask=mask,
    ).output.reshape(batch, sequence, heads * head_dim)
    actual_grads = torch.autograd.grad(actual.square().mean(), actual_inputs)

    torch.testing.assert_close(
        actual, torch.as_tensor(source_output.numpy()), atol=2e-4, rtol=3e-4
    )
    for actual_grad, source_grad in zip(actual_grads, source_grads, strict=True):
        torch.testing.assert_close(
            actual_grad.reshape(batch, sequence, heads * head_dim),
            torch.as_tensor(source_grad.numpy()),
            atol=3e-4,
            rtol=5e-4,
        )


def test_path_attention_core_matches_pinned_upstream_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned PaTH attention comparison")
    source = pytest.importorskip("fla.ops.path_attn.parallel")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7730)
    batch, sequence, query_heads, key_heads, key_dim = 1, 128, 4, 2, 32
    query = (
        torch.randn(
            batch,
            sequence,
            query_heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    key = (
        torch.randn(
            batch,
            sequence,
            key_heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    value = (torch.randn_like(key) * 0.1).requires_grad_()
    weight = torch.nn.functional.normalize(
        torch.randn(
            batch, sequence, key_heads, key_dim, device="cuda", generator=generator
        ),
        dim=-1,
    ).requires_grad_()
    beta = (
        torch.rand(batch, sequence, key_heads, device="cuda", generator=generator) * 2
    ).requires_grad_()
    gate = torch.nn.functional.logsigmoid(
        torch.randn(batch, sequence, query_heads, device="cuda", generator=generator)
    ).requires_grad_()
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "w": weight,
        "beta": beta,
        "g": gate,
    }
    recipe = named_mixer_recipe("path_attention_core")
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "fla_parallel_path_attention_adapter"
    library = library_plan.execute(**operands)
    upstream_output, _ = source.parallel_path_attn(
        query,
        key,
        value,
        weight,
        beta,
        gate,
        scale=key_dim**-0.5,
    )
    torch.testing.assert_close(reference.output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(library.output, upstream_output, atol=0, rtol=0)
    differentiated = tuple(operands.values())
    reference_grads = torch.autograd.grad(
        reference.output.float().square().mean(), differentiated
    )
    library_grads = torch.autograd.grad(
        library.output.float().square().mean(), differentiated
    )
    upstream_grads = torch.autograd.grad(
        upstream_output.float().square().mean(), differentiated
    )
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=3e-2, rtol=3e-2
        )
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)


def test_k1_sdpa_can_require_the_flash_kernel_without_silent_fallback():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("Flash SDPA requires CUDA")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    query = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    plan = compile_mixer(
        softmax_attention_spec("flash_sdpa"),
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            output = plan.execute(query=query, key=key, value=value).output
        torch.cuda.synchronize()
    except RuntimeError as error:
        pytest.skip(f"Flash SDPA is unsupported for this GPU/build: {error}")
    assert output.isfinite().all()


def test_k2_normalized_linear_recurrence_matches_manual_steps_and_gradients():
    torch = _torch()
    torch.manual_seed(12)
    q = torch.randn(1, 3, 2, 4, requires_grad=True)
    k = torch.randn(1, 3, 2, 4, requires_grad=True)
    v = torch.randn(1, 3, 2, 3, requires_grad=True)
    plan = compile_mixer(
        linear_attention_spec(feature_map=FeatureMap.RELU),
        intent=MixerIntent.TRAINING,
    )
    result = plan.execute(query=q, key=k, value=v)

    state = torch.zeros(1, 2, 4, 3)
    normalizer = torch.zeros(1, 2, 4)
    expected = []
    for token in range(3):
        kt = torch.relu(k[:, token])
        qt = torch.relu(q[:, token])
        state = state + torch.einsum("bhk,bhv->bhkv", kt, v[:, token])
        normalizer = normalizer + kt
        numerator = torch.einsum("bhk,bhkv->bhv", qt, state)
        denominator = torch.einsum("bhk,bhk->bh", qt, normalizer)
        expected.append(numerator / denominator.clamp_min(1e-6).unsqueeze(-1))
    torch.testing.assert_close(
        result.output, torch.stack(expected, dim=2).transpose(1, 2)
    )
    result.output.sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    assert result.final_state.shape == (1, 2, 4, 3)
    assert result.final_normalizer_state.shape == (1, 2, 4)


def test_k2_gated_delta_carries_state_and_gate_gradients():
    torch = _torch()
    torch.manual_seed(13)
    q = torch.randn(1, 4, 2, 3, requires_grad=True)
    k = torch.randn(1, 4, 2, 3, requires_grad=True)
    v = torch.randn(1, 4, 2, 5, requires_grad=True)
    beta = torch.sigmoid(torch.randn(1, 4, 2)).requires_grad_()
    log_decay = (-torch.rand(1, 4, 2) * 0.2).requires_grad_()
    initial_state = torch.randn(1, 2, 3, 5, requires_grad=True)
    plan = compile_mixer(
        delta_rule_spec("gated_delta", decay=DecayGranularity.HEAD),
        intent=MixerIntent.TRAINING,
    )
    result = plan.execute(
        query=q,
        key=k,
        value=v,
        beta=beta,
        log_decay=log_decay,
        initial_state=initial_state,
    )
    (result.output.square().mean() + result.final_state.square().mean()).backward()
    for tensor in (q, k, v, beta, log_decay, initial_state):
        assert tensor.grad is not None


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_k2_fla_library_anchor_matches_reference_when_installed(dtype_name):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("FLA gated-delta anchor requires CUDA")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("version_compatible") is not True:
        pytest.skip("the exact FLA gated-delta comparator is unavailable")
    torch.manual_seed(131)
    dtype = getattr(torch, dtype_name)
    query = torch.randn(1, 4, 2, 8, device="cuda", dtype=dtype, requires_grad=True)
    key = (
        torch.nn.functional.normalize(
            torch.randn(1, 4, 2, 8, device="cuda", dtype=dtype), dim=-1
        )
        .detach()
        .requires_grad_()
    )
    value = torch.randn(1, 4, 2, 5, device="cuda", dtype=dtype, requires_grad=True)
    beta = torch.rand(1, 4, 2, device="cuda", requires_grad=True)
    log_decay = (-torch.rand(1, 4, 2, device="cuda") * 0.2).requires_grad_()
    initial = torch.randn(1, 2, 8, 5, device="cuda", requires_grad=True)
    spec = delta_rule_spec("gated_delta", decay=DecayGranularity.HEAD)
    operands = dict(
        query=query,
        key=key,
        value=value,
        beta=beta,
        log_decay=log_decay,
        initial_state=initial,
    )
    from urm.compiler.unified_mixer import _execute_matrix_recurrence

    reference = _execute_matrix_recurrence(spec, torch, **dict(operands))
    reference_loss = (
        reference.output.float().square().mean()
        + reference.final_state.float().square().mean()
    )
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    library = compile_mixer(
        spec,
        intent="training",
        backend=MixerBackend.LIBRARY,
        dtype=dtype_name,
    ).execute(**operands)
    torch.testing.assert_close(
        library.output.float(), reference.output.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        library.final_state.float(), reference.final_state.float(), atol=6e-2, rtol=2e-2
    )
    library_loss = (
        library.output.float().square().mean()
        + library.final_state.float().square().mean()
    )
    library_grads = torch.autograd.grad(library_loss, tuple(operands.values()))
    for actual, expected in zip(library_grads, reference_grads):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=8e-2, rtol=8e-2
        )


@pytest.mark.parametrize("normalized", [False, True])
def test_k2_fla_linear_attention_matches_reference_and_backward(normalized):
    torch = _torch()
    from urm.compiler.unified_mixer import _execute_matrix_recurrence

    if not torch.cuda.is_available():
        pytest.skip("FLA linear-attention anchor requires CUDA")
    from urm.adapters.gated_delta_rule import fla_version

    if fla_version().get("version_compatible") is not True:
        pytest.skip("the exact FLA library comparator is unavailable")
    torch.manual_seed(133)
    query = torch.randn(1, 12, 2, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.nn.functional.normalize(torch.randn_like(query), dim=-1)
    # The pinned low-precision FLA chunk backward requires even K and V.
    value = torch.randn(1, 12, 2, 6, device="cuda", dtype=torch.bfloat16)
    query, key, value = (
        tensor.detach().requires_grad_() for tensor in (query, key, value)
    )
    spec = linear_attention_spec(
        "fla_linear", feature_map=FeatureMap.ELU_PLUS_ONE, normalized=normalized
    )
    operands = {"query": query, "key": key, "value": value}
    if normalized:
        operands["initial_state"] = torch.randn(
            1, 2, 8, 6, device="cuda", requires_grad=True
        )
        operands["initial_normalizer_state"] = torch.zeros(
            1, 2, 8, device="cuda", requires_grad=True
        )
    else:
        operands["initial_state"] = torch.randn(
            1, 2, 8, 6, device="cuda", requires_grad=True
        )
    reference = _execute_matrix_recurrence(spec, torch, **dict(operands))
    reference_loss = (
        reference.output.float().square().mean()
        + reference.final_state.float().square().mean()
    )
    if reference.final_normalizer_state is not None:
        reference_loss = (
            reference_loss + reference.final_normalizer_state.square().mean()
        )
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    library = compile_mixer(
        spec,
        intent="training",
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    ).execute(**operands)
    library_loss = (
        library.output.float().square().mean()
        + library.final_state.float().square().mean()
    )
    if library.final_normalizer_state is not None:
        library_loss = library_loss + library.final_normalizer_state.square().mean()
    library_grads = torch.autograd.grad(library_loss, tuple(operands.values()))
    torch.testing.assert_close(
        library.output.float(), reference.output.float(), atol=8e-2, rtol=8e-2
    )
    for actual, expected in zip(library_grads, reference_grads):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=1e-1, rtol=1e-1
        )


@pytest.mark.parametrize("odd_axis", ["key", "value"])
def test_fla_linear_low_precision_backward_rejects_odd_dimensions(odd_axis):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("FLA linear-attention anchor requires CUDA")
    pytest.importorskip("fla")
    key_dim = 7 if odd_axis == "key" else 8
    value_dim = 5 if odd_axis == "value" else 6
    query = torch.randn(1, 4, 2, key_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn(1, 4, 2, value_dim, device="cuda", dtype=torch.bfloat16)
    plan = compile_mixer(
        linear_attention_spec("fla_linear", feature_map=FeatureMap.ELU_PLUS_ONE),
        intent="training",
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    with pytest.raises(ValueError, match="requires even key and value dimensions"):
        plan.execute(query=query, key=key, value=value)


@pytest.mark.parametrize(
    "recipe_name,supported", [("simple_gla", False), ("gla", True)]
)
def test_k2_fla_gated_additive_training_dtype_contract(recipe_name, supported):
    from urm.compiler.diagnostics import CompilerError

    if supported:
        plan = compile_mixer(
            named_mixer_recipe(recipe_name),
            intent="training",
            backend=MixerBackend.LIBRARY,
            dtype="bfloat16",
        )
        assert plan.anchor == "fla_chunk_gla_adapter"
    else:
        with pytest.raises(CompilerError, match="forward-only execution"):
            compile_mixer(
                named_mixer_recipe(recipe_name),
                intent="training",
                backend=MixerBackend.LIBRARY,
                dtype="bfloat16",
            )


@pytest.mark.parametrize("recipe_name", ["simple_gla", "gla"])
def test_k2_fla_gated_additive_chunk_prefill_matches_reference_and_backward(
    recipe_name,
):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("FLA gated-additive chunk anchors require CUDA")
    from urm.adapters.gated_delta_rule import fla_version

    if fla_version().get("version_compatible") is not True:
        pytest.skip("the exact FLA comparator pin is unavailable")

    torch.manual_seed(182)
    spec = named_mixer_recipe(recipe_name).spec
    batch, sequence, heads, key_dim, value_dim = 1, 12, 2, 8, 6
    query = torch.randn(
        batch, sequence, heads, key_dim, device="cuda", requires_grad=True
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(
        batch, sequence, heads, value_dim, device="cuda", requires_grad=True
    )
    decay_shape = (
        (batch, sequence, heads)
        if spec.decay is DecayGranularity.HEAD
        else (batch, sequence, heads, key_dim)
    )
    log_decay = (-torch.rand(*decay_shape, device="cuda") * 0.2).requires_grad_()
    initial_state = torch.randn(
        batch,
        heads,
        key_dim,
        value_dim,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "log_decay": log_decay,
        "initial_state": initial_state,
    }
    from urm.compiler.unified_mixer import _execute_matrix_recurrence

    reference = _execute_matrix_recurrence(spec, torch, **dict(operands))
    reference_loss = (
        reference.output.square().mean() + reference.final_state.square().mean()
    )
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    library = compile_mixer(
        spec,
        intent="training",
        backend=MixerBackend.LIBRARY,
        dtype="float32",
    ).execute(**operands)
    torch.testing.assert_close(library.output, reference.output, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(
        library.final_state, reference.final_state, atol=4e-2, rtol=4e-2
    )
    library_loss = library.output.square().mean() + library.final_state.square().mean()
    library_grads = torch.autograd.grad(library_loss, tuple(operands.values()))
    for actual, expected in zip(library_grads, reference_grads):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("recipe_name", ["simple_gla", "gla"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_k2_fla_gated_additive_recurrent_decode_matches_reference(
    recipe_name, dtype_name
):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("FLA gated-additive anchors require CUDA")
    from urm.adapters.gated_delta_rule import fla_version

    if fla_version().get("version_compatible") is not True:
        pytest.skip("the exact FLA comparator pin is unavailable")

    torch.manual_seed(191)
    spec = named_mixer_recipe(recipe_name).spec
    dtype = getattr(torch, dtype_name)
    query = torch.randn(1, 1, 2, 8, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(1, 1, 2, 6, device="cuda", dtype=dtype)
    gate_shape = (1, 1, 2) if spec.decay is DecayGranularity.HEAD else (1, 1, 2, 8)
    log_decay = -torch.rand(*gate_shape, device="cuda", dtype=dtype) * 0.2
    initial_state = torch.randn(1, 2, 8, 6, device="cuda", dtype=torch.float32)
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "log_decay": log_decay,
        "initial_state": initial_state,
    }
    reference = compile_mixer(spec, dtype=dtype_name).execute(**operands)
    library = compile_mixer(
        spec, backend=MixerBackend.LIBRARY, dtype=dtype_name
    ).execute(**operands)
    torch.testing.assert_close(
        library.output.float(), reference.output.float(), atol=8e-2, rtol=8e-2
    )
    torch.testing.assert_close(
        library.final_state.float(), reference.final_state.float(), atol=8e-2, rtol=8e-2
    )
    assert library.metadata["execution_mode"] == "decode"
    assert library.metadata["backward_supported"] is False


def test_k2_fla_delta_rule_matches_reference_and_backward():
    torch = _torch()
    from urm.compiler.unified_mixer import _execute_matrix_recurrence

    if not torch.cuda.is_available():
        pytest.skip("FLA delta-rule anchor requires CUDA")
    from urm.adapters.gated_delta_rule import fla_version

    if fla_version().get("version_compatible") is not True:
        pytest.skip("the exact FLA library comparator is unavailable")
    torch.manual_seed(134)
    query = torch.randn(1, 12, 2, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.nn.functional.normalize(torch.randn_like(query), dim=-1)
    value = torch.randn(1, 12, 2, 5, device="cuda", dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, 12, 2, device="cuda", dtype=torch.bfloat16))
    initial_state = torch.randn(1, 2, 8, 5, device="cuda", requires_grad=True)
    operands = {
        name: tensor.detach().requires_grad_()
        for name, tensor in {
            "query": query,
            "key": key,
            "value": value,
            "beta": beta,
        }.items()
    }
    operands["initial_state"] = initial_state
    spec = delta_rule_spec("fla_delta")
    reference = _execute_matrix_recurrence(spec, torch, **dict(operands))
    reference_loss = (
        reference.output.float().square().mean()
        + reference.final_state.float().square().mean()
    )
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    library = compile_mixer(
        spec,
        intent="training",
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    ).execute(**operands)
    library_loss = (
        library.output.float().square().mean()
        + library.final_state.float().square().mean()
    )
    library_grads = torch.autograd.grad(library_loss, tuple(operands.values()))
    torch.testing.assert_close(
        library.output.float(), reference.output.float(), atol=8e-2, rtol=8e-2
    )
    for actual, expected in zip(library_grads, reference_grads):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=1e-1, rtol=1e-1
        )


def test_k2_fla_delta_rule_selects_forward_only_one_token_decode():
    torch = _torch()
    from urm.compiler.unified_mixer import _execute_matrix_recurrence

    if not torch.cuda.is_available():
        pytest.skip("FLA delta-rule anchor requires CUDA")
    from urm.adapters.gated_delta_rule import fla_version

    if fla_version().get("version_compatible") is not True:
        pytest.skip("the exact FLA library comparator is unavailable")
    torch.manual_seed(135)
    query = torch.randn(1, 1, 2, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.nn.functional.normalize(torch.randn_like(query), dim=-1)
    value = torch.randn(1, 1, 2, 5, device="cuda", dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, 1, 2, device="cuda", dtype=torch.bfloat16))
    initial = torch.randn(1, 2, 8, 5, device="cuda")
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "initial_state": initial,
    }
    spec = delta_rule_spec("fla_delta_decode")
    reference = _execute_matrix_recurrence(spec, torch, **dict(operands))
    decode = compile_mixer(
        spec,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    ).execute(**operands)
    torch.testing.assert_close(
        decode.output.float(), reference.output.float(), atol=8e-2, rtol=8e-2
    )
    torch.testing.assert_close(
        decode.final_state.float(), reference.final_state.float(), atol=8e-2, rtol=8e-2
    )
    assert decode.metadata["execution_mode"] == "decode"
    assert decode.metadata["backward_supported"] is False


def test_k2_fla_library_selects_forward_only_decode_for_one_token():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("FLA gated-delta anchor requires CUDA")
    from urm.adapters.gated_delta_rule import fla_version

    if fla_version().get("version_compatible") is not True:
        pytest.skip("the exact FLA gated-delta comparator is unavailable")
    torch.manual_seed(132)
    query = torch.randn(1, 1, 2, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.nn.functional.normalize(
        torch.randn(1, 1, 2, 8, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    value = torch.randn(1, 1, 2, 5, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(1, 1, 2, device="cuda")
    log_decay = -torch.rand(1, 1, 2, device="cuda") * 0.2
    initial = torch.randn(1, 2, 8, 5, device="cuda")
    spec = delta_rule_spec("gated_delta_decode", decay=DecayGranularity.HEAD)
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "log_decay": log_decay,
        "initial_state": initial,
    }
    reference = compile_mixer(spec, dtype="bfloat16").execute(**operands)
    decode = compile_mixer(
        spec,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    ).execute(**operands)
    torch.testing.assert_close(
        decode.output.float(), reference.output.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        decode.final_state.float(), reference.final_state.float(), atol=6e-2, rtol=2e-2
    )
    assert decode.metadata["execution_mode"] == "decode"
    assert decode.metadata["backward_supported"] is False


def test_k2_factored_transition_and_ordered_rank_updates_are_executable():
    torch = _torch()
    torch.manual_seed(14)
    q = torch.randn(1, 2, 1, 2)
    k = torch.randn(1, 2, 1, 2)
    v = torch.randn(1, 2, 1, 3)
    update_keys = torch.randn(1, 2, 2, 1, 2)
    update_values = torch.randn(1, 2, 2, 1, 3)
    left = torch.eye(2).reshape(1, 1, 1, 2, 2).expand(1, 2, 1, 2, 2)
    right = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(1, 2, 1, 3, 3)
    spec = UnifiedMixerSpec(
        "factored_delta",
        MixerKernelFamily.RECURRENCE,
        update_rule=StateUpdateRule.DELTA,
        transition=StateTransition.FACTORED_MATRIX,
    )
    result = compile_mixer(spec).execute(
        query=q,
        key=k,
        value=v,
        update_keys=update_keys,
        update_values=update_values,
        beta=torch.ones(1, 2, 2, 1),
        left_transition=left,
        right_transition=right,
    )
    assert result.output.shape == (1, 2, 1, 3)
    assert result.metadata["ordered_updates_per_token"] == 2


def test_k2_diagonal_ssm_returns_state_and_gradients():
    torch = _torch()
    torch.manual_seed(15)
    x = torch.randn(2, 3, 4, requires_grad=True)
    input_gate = torch.randn(2, 3, 5, requires_grad=True)
    read_gate = torch.randn(2, 3, 5, requires_grad=True)
    log_decay = (-torch.rand(2, 3, 5) * 0.1).requires_grad_()
    initial_state = torch.randn(2, 4, 5, requires_grad=True)
    result = compile_mixer(diagonal_ssm_spec(), intent="training").execute(
        x=x,
        input_gate=input_gate,
        read_gate=read_gate,
        log_decay=log_decay,
        initial_state=initial_state,
        skip=torch.arange(1, 5, dtype=torch.float32),
    )
    (result.output.sum() + result.final_state.sum()).backward()
    assert result.output.shape == (2, 3, 4)
    for tensor in (x, input_gate, read_gate, log_decay, initial_state):
        assert tensor.grad is not None


def test_native_k2_diagonal_step_discretization_matches_reference_and_backward():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("native diagonal SSM requires CUDA")
    pytest.importorskip("triton")
    torch.manual_seed(99)
    device = "cuda"
    batch, channels, sequence, state_width = 1, 4, 7, 5
    base = {
        "u": torch.randn(batch, channels, sequence, device=device) * 0.1,
        "delta": torch.rand(batch, channels, sequence, device=device) * 0.05 + 0.01,
        "A": -torch.rand(channels, state_width, device=device) - 0.1,
        "B": torch.randn(batch, state_width, sequence, device=device) * 0.1,
        "C": torch.randn(batch, state_width, sequence, device=device) * 0.1,
        "D": torch.randn(channels, device=device) * 0.1,
    }
    spec = diagonal_ssm_spec("mamba_step_discretization", step_size_discretization=True)
    reference_plan = compile_mixer(spec, intent=MixerIntent.TRAINING)
    native_plan = compile_mixer(
        spec, intent=MixerIntent.TRAINING, backend=MixerBackend.NATIVE
    )
    results = []
    inputs = []
    for plan in (reference_plan, native_plan):
        leaves = {
            name: value.detach().clone().requires_grad_()
            for name, value in base.items()
        }
        inputs.append(leaves)
        u, delta, A, B, C, D = (
            leaves[name] for name in ("u", "delta", "A", "B", "C", "D")
        )
        result = plan.execute(
            x=u.transpose(1, 2),
            input_gate=B.permute(0, 2, 1),
            read_gate=C.permute(0, 2, 1),
            log_decay=A[None, None, :, :].expand(
                batch, sequence, channels, state_width
            ),
            step_size=delta.transpose(1, 2),
            skip=D,
        )
        loss = result.output.square().sum() + result.final_state.square().sum()
        loss.backward()
        results.append(result)
    torch.testing.assert_close(
        results[1].output, results[0].output, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        results[1].final_state, results[0].final_state, atol=2e-6, rtol=2e-5
    )
    for name in base:
        torch.testing.assert_close(
            inputs[1][name].grad, inputs[0][name].grad, atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize("layout", ["transposed", "expanded"])
def test_native_diagonal_initial_state_strides_preserve_outputs_and_gradients(layout):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("native diagonal SSM requires CUDA")
    pytest.importorskip("triton")
    torch.manual_seed(997)
    batch, sequence, channels, state_width = 2, 4, 3, 5
    base_inputs = {
        "x": torch.randn(batch, sequence, channels, device="cuda"),
        "input_gate": torch.randn(batch, sequence, state_width, device="cuda"),
        "read_gate": torch.randn(batch, sequence, state_width, device="cuda"),
        "log_decay": -torch.rand(batch, sequence, state_width, device="cuda") * 0.1,
    }
    base_state = torch.randn(
        (batch, state_width, channels)
        if layout == "transposed"
        else (1, channels, state_width),
        device="cuda",
    )
    plans = (
        compile_mixer(diagonal_ssm_spec(), intent=MixerIntent.TRAINING),
        compile_mixer(
            diagonal_ssm_spec(),
            intent=MixerIntent.TRAINING,
            backend=MixerBackend.NATIVE,
        ),
    )
    results = []
    leaves = []
    for plan in plans:
        inputs = {
            name: value.detach().clone().requires_grad_()
            for name, value in base_inputs.items()
        }
        if layout == "transposed":
            state_source = base_state.detach().clone().requires_grad_()
            initial_state = state_source.transpose(1, 2)
        else:
            state_source = base_state.detach().clone().requires_grad_()
            initial_state = state_source.expand(batch, channels, state_width)
        result = plan.execute(
            **inputs,
            initial_state=initial_state,
            skip=torch.tensor(0.25, device="cuda"),
        )
        (result.output.square().sum() + result.final_state.square().sum()).backward()
        results.append(result)
        leaves.append((inputs, state_source))

    torch.testing.assert_close(
        results[1].output, results[0].output, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        results[1].final_state, results[0].final_state, atol=2e-6, rtol=2e-5
    )
    for name in base_inputs:
        torch.testing.assert_close(
            leaves[1][0][name].grad,
            leaves[0][0][name].grad,
            atol=3e-6,
            rtol=3e-5,
            msg=lambda message: f"{layout} state, {name}: {message}",
        )
    torch.testing.assert_close(
        leaves[1][1].grad,
        leaves[0][1].grad,
        atol=3e-6,
        rtol=3e-5,
        msg=f"{layout} initial-state gradient differs",
    )


def test_mamba2_ssm_core_reference_matches_pinned_upstream():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned Mamba-2 source comparison")
    source = pytest.importorskip("mamba_ssm.ops.triton.ssd_combined")
    torch.manual_seed(7742)
    operands = {
        "x": torch.randn(1, 64, 2, 4, device="cuda", dtype=torch.float32),
        "dt": torch.rand(1, 64, 2, device="cuda", dtype=torch.float32) * 0.15 + 0.01,
        "A": -torch.rand(2, device="cuda", dtype=torch.float32) - 0.1,
        "B": torch.randn(1, 64, 1, 8, device="cuda", dtype=torch.float32) * 0.1,
        "C": torch.randn(1, 64, 1, 8, device="cuda", dtype=torch.float32) * 0.1,
        "initial_states": torch.randn(1, 2, 4, 8, device="cuda", dtype=torch.float32),
    }
    operands = {name: tensor.requires_grad_() for name, tensor in operands.items()}
    plan = compile_mixer(
        named_mixer_recipe("mamba2_ssm_core"),
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    reference = plan.execute(**operands)
    direct_output, direct_state = source.mamba_chunk_scan_combined(
        operands["x"],
        operands["dt"],
        operands["A"],
        operands["B"],
        operands["C"],
        chunk_size=16,
        initial_states=operands["initial_states"],
        return_final_states=True,
    )
    torch.testing.assert_close(reference.output, direct_output, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        reference.final_state, direct_state, atol=2e-4, rtol=2e-4
    )
    reference_loss = (
        reference.output.square().mean() + reference.final_state.square().mean()
    )
    direct_loss = direct_output.square().mean() + direct_state.square().mean()
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    direct_grads = torch.autograd.grad(direct_loss, tuple(operands.values()))
    for actual, expected in zip(reference_grads, direct_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)


def test_log_linear_core_matches_pinned_upstream_outputs_state_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned LogLinear source comparison")
    source = pytest.importorskip("fla.ops.log_linear_attn")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7703)
    batch, sequence, heads, key_dim, value_dim, levels = 1, 70, 2, 64, 16, 8
    operands = {
        "query": torch.randn(
            batch, sequence, 1, key_dim, device="cuda", generator=generator
        )
        * 0.1,
        "key": torch.randn(
            batch, sequence, 1, key_dim, device="cuda", generator=generator
        )
        * 0.1,
        "value": torch.randn(
            batch, sequence, heads, value_dim, device="cuda", generator=generator
        )
        * 0.1,
        "log_decay": -torch.rand(
            batch, sequence, heads, device="cuda", generator=generator
        )
        * 0.03,
        "level_scales": torch.rand(
            batch, sequence, heads, levels, device="cuda", generator=generator
        )
        * 0.2,
    }
    operands = {name: tensor.requires_grad_() for name, tensor in operands.items()}
    plan = compile_mixer(
        named_mixer_recipe("log_linear_attention_core"),
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    reference = plan.execute(**operands)
    upstream_output, upstream_state = source.chunk_log_linear_attn(
        operands["query"],
        operands["key"],
        operands["value"],
        operands["log_decay"],
        operands["level_scales"],
        output_final_state=True,
    )
    torch.testing.assert_close(reference.output, upstream_output, atol=1e-4, rtol=1e-3)
    for name in upstream_state.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(reference.final_state, name),
            getattr(upstream_state, name),
            atol=3e-4,
            rtol=2e-3,
        )
    reference_loss = (
        reference.output.square().mean() + reference.final_state.ht.square().mean()
    )
    upstream_loss = upstream_output.square().mean() + upstream_state.ht.square().mean()
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    upstream_grads = torch.autograd.grad(upstream_loss, tuple(operands.values()))
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-3)


def test_gdn2_core_reference_matches_pinned_upstream():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned FLA GDN-2 comparison")
    source = pytest.importorskip("fla.ops.gdn2.chunk")
    torch.manual_seed(7727)
    batch, sequence, heads, key_dim, value_dim = 1, 64, 2, 16, 16
    operands = {
        "query": torch.randn(batch, sequence, heads, key_dim, device="cuda") * 0.1,
        "key": torch.randn(batch, sequence, heads, key_dim, device="cuda") * 0.1,
        "value": torch.randn(batch, sequence, heads, value_dim, device="cuda") * 0.1,
        "log_decay": -torch.rand(batch, sequence, heads, key_dim, device="cuda") * 0.1,
        "erase_gate": torch.rand(batch, sequence, heads, key_dim, device="cuda"),
        "write_gate": torch.rand(batch, sequence, heads, value_dim, device="cuda"),
        "initial_state": torch.randn(batch, heads, key_dim, value_dim, device="cuda")
        * 0.1,
    }
    operands = {name: tensor.requires_grad_() for name, tensor in operands.items()}
    plan = compile_mixer(named_mixer_recipe("gdn2_core"), intent=MixerIntent.TRAINING)
    reference = plan.execute(**operands)
    direct_output, direct_state = source.chunk_gdn2(
        operands["query"],
        operands["key"],
        operands["value"],
        operands["log_decay"],
        operands["erase_gate"],
        operands["write_gate"],
        scale=key_dim**-0.5,
        initial_state=operands["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    torch.testing.assert_close(reference.output, direct_output, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        reference.final_state, direct_state, atol=2e-3, rtol=2e-3
    )
    reference_loss = (
        reference.output.square().mean() + reference.final_state.square().mean()
    )
    direct_loss = direct_output.square().mean() + direct_state.square().mean()
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    direct_grads = torch.autograd.grad(direct_loss, tuple(operands.values()))
    for actual, expected in zip(reference_grads, direct_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


def test_kda_core_matches_pinned_upstream_and_library_anchor():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned FLA KDA source comparison")
    source = pytest.importorskip("fla.ops.kda")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(8808)
    batch, sequence, heads, key_dim, value_dim = 1, 64, 2, 64, 16
    operands = {
        "query": torch.randn(
            batch, sequence, heads, key_dim, device="cuda", generator=generator
        )
        * 0.1,
        "key": torch.randn(
            batch, sequence, heads, key_dim, device="cuda", generator=generator
        )
        * 0.1,
        "value": torch.randn(
            batch, sequence, heads, value_dim, device="cuda", generator=generator
        )
        * 0.1,
        "log_decay": -torch.rand(
            batch, sequence, heads, key_dim, device="cuda", generator=generator
        )
        * 0.1,
        "beta": torch.sigmoid(
            torch.randn(batch, sequence, heads, device="cuda", generator=generator)
        ),
        "initial_state": torch.randn(
            batch, heads, key_dim, value_dim, device="cuda", generator=generator
        )
        * 0.1,
    }
    operands = {name: tensor.requires_grad_() for name, tensor in operands.items()}
    recipe = named_mixer_recipe("kda_core")
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="float32"
    ).execute(**operands)
    library = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="float32",
    ).execute(**operands)
    upstream_output, upstream_state = source.chunk_kda(
        operands["query"],
        operands["key"],
        operands["value"],
        operands["log_decay"],
        operands["beta"],
        scale=key_dim**-0.5,
        initial_state=operands["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        state_v_first=False,
    )
    torch.testing.assert_close(reference.output, upstream_output, atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(
        reference.final_state, upstream_state, atol=3e-4, rtol=2e-3
    )
    torch.testing.assert_close(library.output, upstream_output, atol=0, rtol=0)
    torch.testing.assert_close(library.final_state, upstream_state, atol=0, rtol=0)
    reference_loss = (
        reference.output.square().mean() + reference.final_state.square().mean()
    )
    upstream_loss = upstream_output.square().mean() + upstream_state.square().mean()
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    upstream_grads = torch.autograd.grad(upstream_loss, tuple(operands.values()))
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-3)


def test_gated_delta_product_core_matches_pinned_upstream_and_library_anchor():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned FLA Gated DeltaProduct comparison")
    source = pytest.importorskip("fla.ops.gated_delta_product")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7731)
    batch, sequence, ranks, heads, key_dim, value_dim = 1, 16, 2, 2, 16, 8
    query = torch.randn(
        batch,
        sequence,
        heads,
        key_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn(
        query.shape, device="cuda", dtype=query.dtype, generator=generator
    )
    value = torch.randn(
        batch,
        sequence,
        heads,
        value_dim,
        device="cuda",
        dtype=query.dtype,
        generator=generator,
    )
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "log_decay": -torch.rand(
            batch, sequence, heads, device="cuda", generator=generator
        )
        * 0.1,
        "beta": torch.sigmoid(
            torch.randn(
                batch,
                sequence,
                ranks,
                heads,
                device="cuda",
                dtype=query.dtype,
                generator=generator,
            )
        ),
        "update_keys": torch.nn.functional.normalize(
            torch.randn(
                batch,
                sequence,
                ranks,
                heads,
                key_dim,
                device="cuda",
                dtype=query.dtype,
                generator=generator,
            ),
            dim=-1,
        ),
        "update_values": torch.randn(
            batch,
            sequence,
            ranks,
            heads,
            value_dim,
            device="cuda",
            dtype=query.dtype,
            generator=generator,
        )
        * 0.1,
        "initial_state": torch.randn(
            batch, heads, key_dim, value_dim, device="cuda", generator=generator
        )
        * 0.1,
    }
    operands = {name: tensor.requires_grad_() for name, tensor in operands.items()}
    recipe = named_mixer_recipe("gated_delta_product_core")
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    ).execute(**operands)
    upstream_output, upstream_state = source.chunk_gated_delta_product(
        operands["query"],
        operands["update_keys"].reshape(batch, sequence * ranks, heads, key_dim),
        operands["update_values"].reshape(batch, sequence * ranks, heads, value_dim),
        operands["log_decay"],
        operands["beta"].reshape(batch, sequence * ranks, heads),
        num_householder=ranks,
        scale=key_dim**-0.5,
        initial_state=operands["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    torch.testing.assert_close(reference.output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        reference.final_state, upstream_state, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(library.output, upstream_output, atol=0, rtol=0)
    torch.testing.assert_close(library.final_state, upstream_state, atol=0, rtol=0)
    reference_loss = (
        reference.output.float().square().mean() + reference.final_state.square().mean()
    )
    upstream_loss = (
        upstream_output.float().square().mean() + upstream_state.square().mean()
    )
    differentiated = tuple(
        tensor for name, tensor in operands.items() if name not in {"key", "value"}
    )
    reference_grads = torch.autograd.grad(reference_loss, differentiated)
    upstream_grads = torch.autograd.grad(upstream_loss, differentiated)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=3e-2
        )


def test_rwkv4_memory_core_matches_pinned_upstream_and_library_anchor():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned RWKV-4 comparison")
    source = pytest.importorskip("fla.ops.rwkv4")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7733)
    batch, sequence, channels = 1, 64, 128
    w = (
        -2.0 + torch.randn(channels, device="cuda", generator=generator) * 0.05
    ).requires_grad_()
    u = (
        torch.randn(channels, device="cuda", generator=generator) * 0.1
    ).requires_grad_()
    key = (
        torch.randn(batch, sequence, channels, device="cuda", generator=generator) * 0.1
    ).requires_grad_()
    value = (
        torch.randn(batch, sequence, channels, device="cuda", generator=generator) * 0.1
    ).requires_grad_()
    state = (
        torch.stack(
            (
                torch.randn(batch, channels, device="cuda", generator=generator) * 0.1,
                torch.rand(batch, channels, device="cuda", generator=generator) + 0.5,
                torch.randn(batch, channels, device="cuda", generator=generator) * 0.1,
            ),
            dim=1,
        )
        .unsqueeze(2)
        .requires_grad_()
    )
    operands = {"w": w, "u": u, "k": key, "v": value, "state": state}
    recipe = named_mixer_recipe("rwkv4_memory_core")
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="float32"
    ).execute(**operands)
    library = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="float32",
    ).execute(**operands)
    upstream_output, upstream_state = source.fused_recurrent_rwkv4(
        w, u, key, value, state
    )
    torch.testing.assert_close(reference.output, upstream_output, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        reference.final_state, upstream_state, atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(library.output, upstream_output, atol=0, rtol=0)
    torch.testing.assert_close(library.final_state, upstream_state, atol=0, rtol=0)
    reference_loss = (
        reference.output.square().mean() + reference.final_state.square().mean()
    )
    upstream_loss = upstream_output.square().mean() + upstream_state.square().mean()
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    upstream_grads = torch.autograd.grad(upstream_loss, tuple(operands.values()))
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)


def test_rwkv6_memory_core_matches_pinned_upstream_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned RWKV-6 comparison")
    source = pytest.importorskip("fla.ops.rwkv6")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    import os

    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    generator = torch.Generator(device="cuda").manual_seed(7761)
    batch, sequence, heads, key_dim, value_dim = 1, 64, 2, 16, 16
    query = (
        torch.randn(batch, sequence, heads, key_dim, device="cuda", generator=generator)
        * 0.1
    ).requires_grad_()
    key = (torch.randn_like(query) * 0.1).requires_grad_()
    value = (
        torch.randn(
            batch, sequence, heads, value_dim, device="cuda", generator=generator
        )
        * 0.1
    ).requires_grad_()
    log_decay = (-torch.rand_like(query) * 0.05).requires_grad_()
    bonus = (
        torch.randn(heads, key_dim, device="cuda", generator=generator) * 0.1
    ).requires_grad_()
    initial_state = (
        torch.randn(
            batch, heads, key_dim, value_dim, device="cuda", generator=generator
        )
        * 0.05
    ).requires_grad_()
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "log_decay": log_decay,
        "bonus": bonus,
        "initial_state": initial_state,
    }
    recipe = named_mixer_recipe("rwkv6_memory_core")
    equation = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="float32"
    ).execute(**operands)
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="float32",
    )
    assert library_plan.anchor == "fla_fused_recurrent_rwkv6_adapter"
    library = library_plan.execute(**operands)
    upstream_output, upstream_state = source.fused_recurrent_rwkv6(
        query,
        key,
        value,
        log_decay,
        bonus,
        scale=key_dim**-0.5,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.testing.assert_close(equation.output, upstream_output, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        equation.final_state, upstream_state, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(library.output, upstream_output, atol=0, rtol=0)
    torch.testing.assert_close(library.final_state, upstream_state, atol=0, rtol=0)
    differentiated = tuple(operands.values())
    equation_grads = torch.autograd.grad(
        equation.output.square().mean(), differentiated
    )
    library_grads = torch.autograd.grad(library.output.square().mean(), differentiated)
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(), differentiated
    )
    for actual, expected in zip(equation_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_momentum_delta_core_matches_pinned_upstream_outputs_states_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned Momentum DeltaNet comparison")
    source = pytest.importorskip("fla.ops.momentum_delta_rule.chunk")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    torch.manual_seed(7762)
    batch, sequence, heads, key_dim, value_dim = 1, 64, 2, 16, 16
    query = (
        torch.randn(
            batch, sequence, heads, key_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.1
    ).requires_grad_()
    key = (torch.randn_like(query) * 0.1).requires_grad_()
    p = (torch.randn_like(query) * 0.1).requires_grad_()
    value = (
        torch.randn(
            batch, sequence, heads, value_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.1
    ).requires_grad_()
    log_alpha = (
        (-torch.rand(batch, sequence, heads, device="cuda") * 0.03)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    log_mu = (-torch.rand_like(log_alpha) * 0.03).requires_grad_()
    beta = torch.sigmoid(torch.randn_like(log_alpha)).requires_grad_()
    eta = torch.sigmoid(torch.randn_like(log_alpha)).requires_grad_()
    initial_state = (
        torch.randn(
            batch, heads, key_dim, value_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.01
    ).requires_grad_()
    initial_normalizer_state = (torch.randn_like(initial_state) * 0.01).requires_grad_()
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "p": p,
        "log_alpha": log_alpha,
        "log_mu": log_mu,
        "beta": beta,
        "eta": eta,
        "initial_state": initial_state,
        "initial_normalizer_state": initial_normalizer_state,
    }
    recipe = named_mixer_recipe("momentum_delta_core")
    equation = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "fla_chunk_momentum_delta_rule_adapter"
    library = library_plan.execute(**operands)
    upstream_output, upstream_state = source.chunk_momentum_delta_rule(
        query,
        key,
        value,
        log_alpha,
        log_mu,
        p=p,
        beta=beta,
        eta=eta,
        scale=key_dim**-0.5,
        initial_state=torch.stack((initial_state, initial_normalizer_state)),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_p_times_alpha=False,
        chunk_size=64,
    )
    torch.testing.assert_close(equation.output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        equation.final_state, upstream_state[0], atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        equation.final_normalizer_state, upstream_state[1], atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(library.output, upstream_output, atol=0, rtol=0)
    torch.testing.assert_close(library.final_state, upstream_state[0], atol=0, rtol=0)
    torch.testing.assert_close(
        library.final_normalizer_state, upstream_state[1], atol=0, rtol=0
    )
    differentiated = tuple(operands.values())
    equation_loss = (
        equation.output.float().square().mean()
        + equation.final_state.float().square().mean()
        + equation.final_normalizer_state.float().square().mean()
    )
    library_loss = (
        library.output.float().square().mean()
        + library.final_state.float().square().mean()
        + library.final_normalizer_state.float().square().mean()
    )
    upstream_loss = (
        upstream_output.float().square().mean()
        + upstream_state[0].float().square().mean()
        + upstream_state[1].float().square().mean()
    )
    equation_grads = torch.autograd.grad(equation_loss, differentiated)
    library_grads = torch.autograd.grad(library_loss, differentiated)
    upstream_grads = torch.autograd.grad(upstream_loss, differentiated)
    for actual, expected in zip(equation_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_gated_oja_core_matches_pinned_upstream_outputs_states_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned gated Oja comparison")
    source = pytest.importorskip("fla.ops.gated_oja_rule")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7733)
    batch, sequence, heads, key_dim, value_dim = 1, 64, 2, 16, 16
    query = (
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    key = (torch.randn_like(query) * 0.1).requires_grad_()
    value = (
        torch.randn(
            batch,
            sequence,
            heads,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    gate = (
        -torch.rand(
            batch, sequence, heads, value_dim, device="cuda", generator=generator
        )
        * 0.03
    ).requires_grad_()
    beta = torch.sigmoid(
        torch.randn(batch, sequence, heads, device="cuda", generator=generator)
    ).requires_grad_()
    initial_state = (
        torch.randn(
            batch,
            heads,
            key_dim,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).requires_grad_()
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "gv": gate,
        "beta": beta,
        "initial_state": initial_state,
    }
    recipe = named_mixer_recipe("gated_oja_core")
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "fla_chunk_gated_oja_adapter"
    library = library_plan.execute(**operands)
    upstream_output, upstream_state = source.chunk_gated_oja_rule(
        query,
        key,
        value,
        gate,
        beta,
        scale=key_dim**-0.5,
        initial_state=initial_state,
        output_final_state=True,
        use_q_l2norm=False,
        use_k_l2norm=False,
        chunk_size=64,
    )
    torch.testing.assert_close(reference.output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        reference.final_state, upstream_state, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(library.output, upstream_output, atol=0, rtol=0)
    torch.testing.assert_close(library.final_state, upstream_state, atol=0, rtol=0)
    differentiated = tuple(operands.values())
    reference_loss = (
        reference.output.float().square().mean()
        + reference.final_state.float().square().mean()
    )
    library_loss = (
        library.output.float().square().mean()
        + library.final_state.float().square().mean()
    )
    upstream_loss = (
        upstream_output.float().square().mean() + upstream_state.float().square().mean()
    )
    reference_grads = torch.autograd.grad(reference_loss, differentiated)
    library_grads = torch.autograd.grad(library_loss, differentiated)
    upstream_grads = torch.autograd.grad(upstream_loss, differentiated)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_comba_core_matches_pinned_upstream_outputs_states_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned COMBA comparison")
    source = pytest.importorskip("fla.ops.comba")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7737)
    batch, sequence, heads, key_dim, value_dim = 1, 64, 2, 16, 16
    query = (
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    key = (torch.randn_like(query) * 0.1).requires_grad_()
    prediction_key = (torch.randn_like(query) * 0.1).requires_grad_()
    value = (
        torch.randn(
            batch,
            sequence,
            heads,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    log_decay = (
        -torch.rand(batch, sequence, heads, device="cuda", generator=generator) * 0.03
    ).requires_grad_()
    beta = torch.sigmoid(
        torch.randn(batch, sequence, heads, device="cuda", generator=generator)
    ).requires_grad_()
    initial_state = (
        torch.randn(
            batch,
            heads,
            key_dim,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).requires_grad_()
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "p": prediction_key,
        "g": log_decay,
        "beta": beta,
        "initial_state": initial_state,
    }
    recipe = named_mixer_recipe("comba_core")
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "fla_chunk_comba_adapter"
    library = library_plan.execute(**operands)
    upstream_output, upstream_state = source.chunk_comba(
        query,
        key,
        value,
        prediction_key,
        log_decay,
        beta=beta,
        scale=key_dim**-0.5,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    torch.testing.assert_close(reference.output, upstream_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        reference.final_state, upstream_state, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(library.output, upstream_output, atol=0, rtol=0)
    torch.testing.assert_close(library.final_state, upstream_state, atol=0, rtol=0)
    differentiated = tuple(operands.values())
    reference_loss = (
        reference.output.float().square().mean()
        + reference.final_state.float().square().mean()
    )
    library_loss = (
        library.output.float().square().mean()
        + library.final_state.float().square().mean()
    )
    upstream_loss = (
        upstream_output.float().square().mean() + upstream_state.float().square().mean()
    )
    reference_grads = torch.autograd.grad(reference_loss, differentiated)
    library_grads = torch.autograd.grad(library_loss, differentiated)
    upstream_grads = torch.autograd.grad(upstream_loss, differentiated)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_pgdn_core_matches_pinned_upstream_outputs_states_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned PGDN comparison")
    source = pytest.importorskip("fla.ops.precond_gated_delta_rule.chunk")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7741)
    batch, sequence, heads, key_dim, value_dim = 1, 64, 2, 16, 16
    query = (
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    key = (torch.randn_like(query) * 0.1).requires_grad_()
    value = (
        torch.randn(
            batch,
            sequence,
            heads,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    g_atk = (
        -torch.rand(batch, sequence, heads, device="cuda", generator=generator) * 0.03
    ).requires_grad_()
    gate = (
        -torch.rand(batch, sequence, heads, device="cuda", generator=generator) * 0.03
    ).requires_grad_()
    beta_atk = torch.sigmoid(
        torch.randn(batch, sequence, heads, device="cuda", generator=generator)
    ).requires_grad_()
    beta = torch.sigmoid(
        torch.randn(batch, sequence, heads, device="cuda", generator=generator)
    ).requires_grad_()
    initial_state = (
        torch.randn(
            batch,
            heads,
            key_dim,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).requires_grad_()
    initial_A_state = (
        torch.rand(batch, heads, key_dim, device="cuda", generator=generator) * 0.01
    ).requires_grad_()
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "g_atk": g_atk,
        "g": gate,
        "beta_atk": beta_atk,
        "beta": beta,
        "initial_state": initial_state,
        "initial_A_state": initial_A_state,
    }
    recipe = named_mixer_recipe("pgdn_core")
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "fla_chunk_precond_gated_delta_adapter"
    library = library_plan.execute(**operands)
    upstream_output, upstream_state, upstream_A = source.chunk_precond_gated_delta_rule(
        query,
        key,
        value,
        g_atk,
        gate,
        beta_atk,
        beta,
        scale=key_dim**-0.5,
        initial_state=initial_state,
        initial_A_state=initial_A_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        x=1.5,
        eps=1e-6,
        log_atk_scale=None,
    )
    for actual, expected in (
        (reference.output, upstream_output),
        (reference.final_state, upstream_state),
        (reference.final_normalizer_state, upstream_A),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual, expected in (
        (library.output, upstream_output),
        (library.final_state, upstream_state),
        (library.final_normalizer_state, upstream_A),
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    differentiated = tuple(operands.values())
    reference_loss = (
        reference.output.float().square().mean()
        + reference.final_state.float().square().mean()
        + reference.final_normalizer_state.square().mean()
    )
    library_loss = (
        library.output.float().square().mean()
        + library.final_state.float().square().mean()
        + library.final_normalizer_state.square().mean()
    )
    upstream_loss = (
        upstream_output.float().square().mean()
        + upstream_state.float().square().mean()
        + upstream_A.square().mean()
    )
    reference_grads = torch.autograd.grad(reference_loss, differentiated)
    library_grads = torch.autograd.grad(library_loss, differentiated)
    upstream_grads = torch.autograd.grad(upstream_loss, differentiated)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_pkda_core_matches_pinned_upstream_outputs_states_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned PKDA comparison")
    source = pytest.importorskip("fla.ops.precond_kda.chunk")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7743)
    batch, sequence, heads, key_dim, value_dim = 1, 64, 2, 16, 16
    query = (
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    key = (torch.randn_like(query) * 0.1).requires_grad_()
    value = (
        torch.randn(
            batch,
            sequence,
            heads,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    gate = (
        -torch.rand(batch, sequence, heads, key_dim, device="cuda", generator=generator)
        * 0.03
    ).requires_grad_()
    g_atk = (
        -torch.rand(batch, sequence, heads, device="cuda", generator=generator) * 0.03
    ).requires_grad_()
    beta_atk = torch.sigmoid(
        torch.randn(batch, sequence, heads, device="cuda", generator=generator)
    ).requires_grad_()
    beta = torch.sigmoid(
        torch.randn(batch, sequence, heads, device="cuda", generator=generator)
    ).requires_grad_()
    initial_state = (
        torch.randn(
            batch, heads, key_dim, value_dim, device="cuda", generator=generator
        )
        * 0.01
    ).requires_grad_()
    initial_A_state = (
        torch.rand(batch, heads, key_dim, device="cuda", generator=generator) * 0.01
    ).requires_grad_()
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "g": gate,
        "g_atk": g_atk,
        "beta_atk": beta_atk,
        "beta": beta,
        "initial_state": initial_state,
        "initial_A_state": initial_A_state,
    }
    recipe = named_mixer_recipe("pkda_core")
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "fla_chunk_precond_kda_adapter"
    library = library_plan.execute(**operands)
    upstream_output, upstream_state, upstream_A = source.chunk_precond_kda(
        query,
        key,
        value,
        gate,
        g_atk,
        beta_atk,
        beta,
        scale=key_dim**-0.5,
        initial_state=initial_state,
        initial_A_state=initial_A_state,
        output_final_state=True,
        use_gate_in_kernel=False,
        safe_gate=False,
        x=1.5,
        eps=1e-6,
        log_atk_scale=None,
    )
    for actual, expected in (
        (reference.output, upstream_output),
        (reference.final_state, upstream_state),
        (reference.final_normalizer_state, upstream_A),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual, expected in (
        (library.output, upstream_output),
        (library.final_state, upstream_state),
        (library.final_normalizer_state, upstream_A),
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    differentiated = tuple(operands.values())
    reference_loss = (
        reference.output.float().square().mean()
        + reference.final_state.square().mean()
        + reference.final_normalizer_state.float().square().mean()
    )
    library_loss = (
        library.output.float().square().mean()
        + library.final_state.square().mean()
        + library.final_normalizer_state.float().square().mean()
    )
    upstream_loss = (
        upstream_output.float().square().mean()
        + upstream_state.square().mean()
        + upstream_A.float().square().mean()
    )
    reference_grads = torch.autograd.grad(reference_loss, differentiated)
    library_grads = torch.autograd.grad(library_loss, differentiated)
    upstream_grads = torch.autograd.grad(upstream_loss, differentiated)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_deltaformer_attention_core_matches_pinned_upstream_outputs_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned DeltaFormer comparison")
    source = pytest.importorskip("fla.ops.deltaformer")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7813)
    batch, sequence, heads, dim = 1, 64, 2, 32
    query = (
        torch.randn(
            batch,
            sequence,
            heads,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    key = (torch.randn_like(query) * 0.1).requires_grad_()
    value = (torch.randn_like(query) * 0.1).requires_grad_()
    beta = torch.sigmoid(
        torch.randn(
            batch,
            sequence,
            heads,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
    ).requires_grad_()
    operands = {"query": query, "key": key, "value": value, "beta": beta}
    recipe = named_mixer_recipe("deltaformer_attention_core")
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "fla_parallel_deltaformer_adapter"
    library = library_plan.execute(**operands)
    upstream = source.deltaformer_attn(query, key, value, beta, C=32)
    torch.testing.assert_close(reference.output, upstream, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(library.output, upstream, atol=0, rtol=0)

    differentiated = tuple(operands.values())
    reference_grads = torch.autograd.grad(
        reference.output.float().square().mean(), differentiated
    )
    library_grads = torch.autograd.grad(
        library.output.float().square().mean(), differentiated
    )
    upstream_grads = torch.autograd.grad(
        upstream.float().square().mean(), differentiated
    )
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_rodimus_gla_core_matches_pinned_upstream_outputs_states_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned Rodimus GLA comparison")
    source = pytest.importorskip("fla.ops.gla")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7836)
    batch, sequence, heads, key_dim, value_dim = 1, 64, 1, 64, 128
    query = (
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    key = torch.nn.functional.normalize(
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        ),
        dim=-1,
    ).to(torch.bfloat16)
    value = (
        torch.randn(
            batch,
            sequence,
            heads,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    log_decay = (
        -torch.rand(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.03
    )
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "log_decay": log_decay,
    }
    recipe = named_mixer_recipe("rodimus_gla_core")
    reference_plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    )
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "fla_chunk_gla_adapter"

    def run(function):
        leaves = {
            name: value.detach().clone().requires_grad_()
            for name, value in operands.items()
        }
        result = function(leaves)
        gradients = torch.autograd.grad(
            result[0].float().square().mean(), tuple(leaves.values())
        )
        return result, gradients

    reference, reference_grads = run(
        lambda values: (lambda result: (result.output, result.final_state))(
            reference_plan.execute(**values)
        )
    )
    library, library_grads = run(
        lambda values: (lambda result: (result.output, result.final_state))(
            library_plan.execute(**values)
        )
    )
    upstream, upstream_grads = run(
        lambda values: source.chunk_gla(
            values["query"],
            values["key"],
            values["value"],
            values["log_decay"],
            scale=key_dim**-0.5,
            output_final_state=True,
            state_v_first=True,
        )
    )
    for actual, expected in zip(reference, upstream, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual, expected in zip(library, upstream, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mesa_net_core_matches_pinned_upstream_outputs_states_and_gradients():
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned MesaNet comparison")
    source = pytest.importorskip("fla.ops.mesa_net.chunk")
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7738)
    batch, sequence, heads, key_dim = 1, 64, 2, 16
    query = (
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    key = torch.nn.functional.normalize(
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        ),
        dim=-1,
    ).to(torch.bfloat16)
    value = (
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    base_operands = {
        "query": query,
        "key": key,
        "value": value,
        "log_decay": -torch.rand(
            batch, sequence, heads, device="cuda", generator=generator
        )
        * 0.03,
        "beta": torch.rand(batch, sequence, heads, device="cuda", generator=generator)
        * 0.4
        + 0.2,
        "lamb": torch.nn.functional.softplus(
            torch.randn(heads, key_dim, device="cuda", generator=generator)
        )
        + 1.0,
    }
    recipe = named_mixer_recipe("mesa_net_core")
    reference_plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    )
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "fla_chunk_mesa_net_adapter"

    def run(function):
        leaves = {
            name: tensor.detach().clone().requires_grad_()
            for name, tensor in base_operands.items()
        }
        result = function(leaves)
        gradients = torch.autograd.grad(
            result[0].float().square().mean(), tuple(leaves.values())
        )
        return result, gradients

    reference, reference_grads = run(
        lambda values: (lambda result: (result.output, result.final_state))(
            reference_plan.execute(**values)
        )
    )
    library, library_grads = run(
        lambda values: (lambda result: (result.output, result.final_state))(
            library_plan.execute(**values)
        )
    )
    upstream, upstream_grads = run(
        lambda values: (lambda output, h_kk, h_kv: (output, (h_kk, h_kv)))(
            *source.chunk_mesa_net(
                values["query"],
                values["key"],
                values["value"],
                values["log_decay"],
                values["beta"],
                values["lamb"],
                output_final_state=True,
                max_CG_iteration=30,
                use_qk_l2norm_in_kernel=False,
            )
        )
    )
    torch.testing.assert_close(reference[0], upstream[0], atol=3e-3, rtol=3e-3)
    for actual, expected in zip(reference[1], upstream[1], strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(library[0], upstream[0], atol=0, rtol=0)
    for actual, expected in zip(library[1], upstream[1], strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=3e-2
        )


def test_mamba3_siso_core_matches_pinned_upstream_outputs_states_and_gradients():
    import inspect
    import subprocess
    from pathlib import Path

    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned Mamba-3 comparison")
    source = pytest.importorskip("mamba_ssm.ops.triton.mamba3.mamba3_siso_combined")
    import mamba_ssm

    source_file = Path(inspect.getfile(mamba_ssm)).resolve()
    repository = next(
        (parent for parent in source_file.parents if (parent / ".git").exists()),
        None,
    )
    if repository is None:
        pytest.skip("the exact pinned Mamba source checkout is unavailable")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "e9594ce1c732d97440f0332fdc43170a2294dbfa":
        pytest.skip("the exact pinned Mamba source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7745)
    batch, sequence, heads, key_dim, value_dim, angle_dim = 1, 64, 2, 16, 16, 4
    query = torch.nn.functional.normalize(
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ),
        dim=-1,
    )
    key = torch.nn.functional.normalize(
        torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ),
        dim=-1,
    )
    value = (
        torch.randn(
            batch,
            sequence,
            heads,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    dt = (
        torch.rand(batch, heads, sequence, device="cuda", generator=generator) * 0.08
        + 0.01
    )
    adt = (
        -torch.rand(batch, heads, sequence, device="cuda", generator=generator)
        * 4.0
        * dt
    )
    operands = {
        "query": query,
        "key": key,
        "value": value,
        "adt": adt,
        "dt": dt,
        "trap": torch.rand(
            batch,
            heads,
            sequence,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ),
        "query_bias": torch.randn(
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05,
        "key_bias": torch.randn(
            heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05,
        "angles": torch.randn(
            batch,
            sequence,
            heads,
            angle_dim,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.1,
    }
    recipe = named_mixer_recipe("mamba3_siso_core")
    reference_plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    )
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == "mamba3_siso_combined_adapter"

    def run(function):
        leaves = {
            name: tensor.detach().clone().requires_grad_()
            for name, tensor in operands.items()
        }
        output, states = function(leaves)
        gradients = torch.autograd.grad(
            output.float().square().mean(), tuple(leaves.values())
        )
        return (output, states), gradients

    reference, reference_grads = run(
        lambda values: (lambda result: (result.output, result.final_state))(
            reference_plan.execute(**values)
        )
    )
    library, library_grads = run(
        lambda values: (lambda result: (result.output, result.final_state))(
            library_plan.execute(**values)
        )
    )
    upstream, upstream_grads = run(
        lambda values: (
            lambda output, angle, ssm, key_state, value_state: (
                output,
                (angle, ssm, key_state, value_state),
            )
        )(
            *source.mamba3_siso_combined(
                values["query"],
                values["key"],
                values["value"],
                values["adt"],
                values["dt"],
                values["trap"],
                values["query_bias"],
                values["key_bias"],
                values["angles"],
                chunk_size=64,
                return_final_states=True,
            )
        )
    )
    torch.testing.assert_close(reference[0], upstream[0], atol=2e-2, rtol=2e-2)
    for actual, expected in zip(reference[1], upstream[1], strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )
    torch.testing.assert_close(library[0], upstream[0], atol=0, rtol=0)
    for actual, expected in zip(library[1], upstream[1], strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=5e-2, rtol=5e-2
        )


@pytest.mark.parametrize("recipe_name", ["abc_core", "gsa_core"])
def test_slot_attention_cores_match_pinned_upstream_outputs_states_and_gradients(
    recipe_name,
):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned ABC/GSA comparison")
    source = pytest.importorskip(
        "fla.ops.abc.chunk" if recipe_name == "abc_core" else "fla.ops.gsa.chunk"
    )
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(
        7747 if recipe_name == "abc_core" else 7749
    )
    batch, sequence, key_heads, query_heads, key_dim, value_dim, slots = (
        1,
        128,
        2,
        2,
        32,
        32,
        16,
    )
    query = (
        torch.randn(
            batch,
            sequence,
            query_heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    key = (
        torch.randn(
            batch,
            sequence,
            key_heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    value = (
        torch.randn(
            batch,
            sequence,
            key_heads,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    ).requires_grad_()
    initial_key_state = (
        torch.randn(
            batch,
            key_heads,
            key_dim,
            slots,
            device="cuda",
            generator=generator,
        )
        * 0.01
    ).requires_grad_()
    initial_value_state = (
        torch.randn(
            batch,
            key_heads,
            slots,
            value_dim,
            device="cuda",
            generator=generator,
        )
        * 0.01
    ).requires_grad_()
    if recipe_name == "abc_core":
        slot_logits = (
            torch.randn(
                batch,
                sequence,
                key_heads,
                slots,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            * 0.1
        ).requires_grad_()
        operands = {
            "query": query,
            "key": key,
            "value": value,
            "slot_logits": slot_logits,
            "initial_key_state": initial_key_state,
            "initial_value_state": initial_value_state,
        }
        upstream_output, upstream_states = source.chunk_abc(
            query,
            key,
            value,
            slot_logits,
            initial_state=(initial_key_state, initial_value_state),
            output_final_state=True,
        )
    else:
        slot_weights = torch.sigmoid(
            torch.randn(
                batch,
                sequence,
                key_heads,
                slots,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
        ).requires_grad_()
        log_decay = (
            torch.nn.functional.logsigmoid(
                torch.randn(
                    batch,
                    sequence,
                    key_heads,
                    slots,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
            )
            * 0.05
        ).requires_grad_()
        operands = {
            "query": query,
            "key": key,
            "value": value,
            "slot_weights": slot_weights,
            "log_decay": log_decay,
            "initial_key_state": initial_key_state,
            "initial_value_state": initial_value_state,
        }
        upstream_output, upstream_states = source.chunk_gsa(
            query,
            key,
            value,
            slot_weights,
            log_decay,
            scale=key_dim**-0.5,
            initial_state=(initial_key_state, initial_value_state),
            output_final_state=True,
            checkpoint_level=0,
        )

    recipe = named_mixer_recipe(recipe_name)
    reference = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    ).execute(**operands)
    library_plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16",
    )
    assert library_plan.anchor == (
        "fla_chunk_abc_adapter"
        if recipe_name == "abc_core"
        else "fla_chunk_gsa_adapter"
    )
    library = library_plan.execute(**operands)
    for actual, expected in (
        (reference.output, upstream_output),
        *zip(reference.final_state, upstream_states, strict=True),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual, expected in (
        (library.output, upstream_output),
        *zip(library.final_state, upstream_states, strict=True),
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    gradient_names = tuple(
        name
        for name in operands
        if name not in {"initial_key_state", "initial_value_state"}
    )
    differentiated = tuple(operands[name] for name in gradient_names)
    reference_grads = torch.autograd.grad(
        reference.output.float().square().mean(), differentiated, allow_unused=True
    )
    library_grads = torch.autograd.grad(
        library.output.float().square().mean(), differentiated, allow_unused=True
    )
    upstream_grads = torch.autograd.grad(
        upstream_output.float().square().mean(), differentiated, allow_unused=True
    )
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        if actual is None and expected is None:
            continue
        if actual is None or expected is None:
            raise AssertionError("reference/upstream gradient presence differs")
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual, expected in zip(library_grads, upstream_grads, strict=True):
        if actual is None and expected is None:
            continue
        if actual is None or expected is None:
            raise AssertionError("adapter/upstream gradient presence differs")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("recipe_name", "source_module", "source_callable"),
    [
        (
            "generalized_delta_iplr_core",
            "fla.ops.generalized_delta_rule.iplr",
            "fused_recurrent_iplr_delta_rule",
        ),
        (
            "generalized_delta_dplr_core",
            "fla.ops.generalized_delta_rule.dplr",
            "chunk_dplr_delta_rule",
        ),
        (
            "rwkv7_transition_core",
            "fla.ops.rwkv7",
            "chunk_rwkv7",
        ),
    ],
)
def test_generalized_delta_transition_cores_match_pinned_upstream(
    recipe_name, source_module, source_callable
):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned generalized-delta comparison")
    source = pytest.importorskip(source_module)
    from urm.adapters.gated_delta_rule import fla_version

    if (
        fla_version().get("source_revision")
        != "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    ):
        pytest.skip("the exact pinned FLA source checkout is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(7732)
    batch, sequence, heads, key_dim, value_dim = 1, 16, 2, 16, 16
    dtype = (
        torch.bfloat16
        if recipe_name in {"generalized_delta_dplr_core", "rwkv7_transition_core"}
        else torch.float32
    )
    operands = {
        "query": torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.1,
        "key": torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.1,
        "value": torch.randn(
            batch,
            sequence,
            heads,
            value_dim,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.1,
        "transition_alpha": torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.03,
        "transition_beta": torch.randn(
            batch,
            sequence,
            heads,
            key_dim,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.03,
        "initial_state": torch.randn(
            batch, heads, key_dim, value_dim, device="cuda", generator=generator
        )
        * 0.1,
    }
    if recipe_name == "generalized_delta_dplr_core":
        operands["log_decay"] = (
            -torch.rand(
                batch,
                sequence,
                heads,
                key_dim,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * 0.05
        )
    elif recipe_name == "rwkv7_transition_core":
        operands["log_decay"] = (
            -torch.rand(
                batch,
                sequence,
                heads,
                key_dim,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * 0.1
        )
    operands = {name: tensor.requires_grad_() for name, tensor in operands.items()}
    recipe = named_mixer_recipe(recipe_name)
    reference = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16" if dtype is torch.bfloat16 else "float32",
    ).execute(**operands)
    library = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.LIBRARY,
        dtype="bfloat16" if dtype is torch.bfloat16 else "float32",
    ).execute(**operands)
    upstream_fn = getattr(source, source_callable)
    if recipe_name == "generalized_delta_iplr_core":
        upstream_output, upstream_state = upstream_fn(
            operands["query"],
            operands["key"],
            operands["value"],
            operands["transition_alpha"],
            operands["transition_beta"],
            scale=key_dim**-0.5,
            initial_state=operands["initial_state"],
            output_final_state=True,
        )
    elif recipe_name == "generalized_delta_dplr_core":
        upstream_output, upstream_state = upstream_fn(
            operands["query"],
            operands["key"],
            operands["value"],
            operands["transition_alpha"],
            operands["transition_beta"],
            operands["log_decay"],
            scale=key_dim**-0.5,
            initial_state=operands["initial_state"],
            output_final_state=True,
            chunk_size=16,
        )
    else:
        upstream_output, upstream_state = upstream_fn(
            r=operands["query"],
            w=operands["log_decay"],
            k=operands["key"],
            v=operands["value"],
            a=operands["transition_alpha"],
            b=operands["transition_beta"],
            scale=1.0,
            initial_state=operands["initial_state"],
            output_final_state=True,
            safe_gate=True,
            lower_bound=-0.6065306597126334,
            chunk_size=64,
        )
    tolerance = (
        1e-2
        if recipe_name == "rwkv7_transition_core"
        else 2e-3
        if dtype is torch.bfloat16
        else 2e-4
    )
    torch.testing.assert_close(
        reference.output, upstream_output, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        reference.final_state, upstream_state, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(library.output, upstream_output, atol=0, rtol=0)
    torch.testing.assert_close(library.final_state, upstream_state, atol=0, rtol=0)
    reference_loss = (
        reference.output.square().mean() + reference.final_state.square().mean()
    )
    upstream_loss = upstream_output.square().mean() + upstream_state.square().mean()
    reference_grads = torch.autograd.grad(reference_loss, tuple(operands.values()))
    upstream_grads = torch.autograd.grad(upstream_loss, tuple(operands.values()))
    for actual, expected in zip(reference_grads, upstream_grads, strict=True):
        grad_tolerance = 2e-2 if recipe_name == "rwkv7_transition_core" else 2e-3
        torch.testing.assert_close(
            actual, expected, atol=grad_tolerance, rtol=grad_tolerance
        )


@pytest.mark.parametrize(
    "recipe_name", ["based_attention_core", "rebased_attention_core"]
)
def test_polynomial_attention_recurrence_matches_pinned_upstream(recipe_name):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned FLA polynomial-attention comparison")
    torch.manual_seed(7720)
    query = torch.randn(1, 32, 2, 8, device="cuda", requires_grad=True)
    key = torch.randn(1, 32, 2, 8, device="cuda", requires_grad=True)
    value = torch.randn(1, 32, 2, 8, device="cuda", requires_grad=True)
    operands = {"query": query, "key": key, "value": value}
    reference = compile_mixer(
        named_mixer_recipe(recipe_name), intent=MixerIntent.TRAINING
    ).execute(**operands)
    if recipe_name == "based_attention_core":
        source = pytest.importorskip("fla.ops.based.fused_chunk")
        direct = source.fused_chunk_based(
            query, key, value, scale=8**-0.5, use_norm=True
        )
    else:
        source = pytest.importorskip("fla.ops.rebased.parallel")
        direct = source.parallel_rebased(
            query,
            key,
            value,
            eps=1e-6,
            use_scale=True,
            use_normalize=True,
        )
    torch.testing.assert_close(reference.output, direct, atol=5e-4, rtol=5e-4)
    reference_gradients = torch.autograd.grad(
        reference.output.square().mean(), (query, key, value)
    )
    direct_gradients = torch.autograd.grad(direct.square().mean(), (query, key, value))
    for actual, expected in zip(reference_gradients, direct_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_k3_sparse_delta_preserves_token_order_and_gradients():
    torch = _torch()
    torch.manual_seed(16)
    memory = torch.randn(1, 4, 2, requires_grad=True)
    read_indices = torch.tensor([[[0], [0], [2]]])
    read_weights = torch.ones(1, 3, 1, requires_grad=True)
    write_indices = torch.tensor([[[0], [0], [3]]])
    write_weights = torch.ones(1, 3, 1, requires_grad=True)
    values = torch.randn(1, 3, 2, requires_grad=True)
    beta = torch.full((1, 3, 1), 0.5, requires_grad=True)
    log_decay = torch.zeros(1, 3, 1, requires_grad=True)
    result = compile_mixer(
        sparse_delta_spec(read_timing=ReadTiming.BEFORE_UPDATE),
        intent="training",
    ).execute(
        memory=memory,
        read_indices=read_indices,
        read_weights=read_weights,
        write_indices=write_indices,
        write_weights=write_weights,
        values=values,
        beta=beta,
        log_decay=log_decay,
    )
    (result.output.square().sum() + result.final_state.square().sum()).backward()
    assert result.output.shape == (1, 3, 2)
    for tensor in (memory, read_weights, write_weights, values, beta, log_decay):
        assert tensor.grad is not None


def test_k3_rejects_within_token_collisions():
    torch = _torch()
    plan = compile_mixer(sparse_delta_spec())
    with pytest.raises(ValueError, match="must be unique"):
        plan.execute(
            memory=torch.zeros(1, 4, 2),
            read_indices=torch.tensor([[[0]]]),
            read_weights=torch.ones(1, 1, 1),
            write_indices=torch.tensor([[[2, 2]]]),
            write_weights=torch.ones(1, 1, 2) / 2,
            values=torch.zeros(1, 1, 2),
            beta=torch.ones(1, 1, 1),
            log_decay=torch.zeros(1, 1, 1),
        )


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "read_timing", [ReadTiming.BEFORE_UPDATE, ReadTiming.AFTER_UPDATE]
)
def test_k3_native_overlapping_routes_match_reference_vjp(dtype_name, read_timing):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("native K3 requires CUDA")

    spec = sparse_delta_spec(read_timing=read_timing)
    from urm.backends.triton.sparse_state.backend import TritonSparseStateMixerBackend
    from urm.compiler.semantic import (
        DType,
        SparseReadTiming,
        SparseStateExecutionMode,
        SparseStateMixerSpec,
        SparseStateOperation,
    )

    dtype = getattr(torch, dtype_name)
    sparse_spec = SparseStateMixerSpec(
        parallel=1,
        sequence=4,
        slots_per_partition=64,
        value_dim=8,
        writes=2,
        reads=2,
        dtype=DType(dtype_name),
        operation=SparseStateOperation.UPDATE,
        read_timing=(
            SparseReadTiming.BEFORE_UPDATE
            if read_timing is ReadTiming.BEFORE_UPDATE
            else SparseReadTiming.AFTER_UPDATE
        ),
        mode=SparseStateExecutionMode.TRAINING,
    )
    status = TritonSparseStateMixerBackend.support_status(sparse_spec)
    if not status.supported:
        pytest.skip(f"native K3 is unavailable on this CUDA target: {status}")

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(44091)
    shared_routes = torch.tensor(
        [[[3, 7], [3, 7], [7, 11], [3, 11]]],
        dtype=torch.int32,
        device=device,
    )

    def make_inputs():
        return {
            "memory": (
                0.05
                * torch.randn(1, 64, 8, device=device, dtype=dtype, generator=generator)
            ).requires_grad_(),
            "read_indices": shared_routes,
            "read_weights": torch.softmax(
                torch.randn(
                    1, 4, 2, device=device, dtype=torch.float32, generator=generator
                ),
                dim=-1,
            )
            .to(dtype)
            .contiguous()
            .requires_grad_(),
            "write_indices": shared_routes,
            "write_weights": torch.softmax(
                torch.randn(
                    1, 4, 2, device=device, dtype=torch.float32, generator=generator
                ),
                dim=-1,
            )
            .to(dtype)
            .contiguous()
            .requires_grad_(),
            "values": (
                0.05
                * torch.randn(1, 4, 8, device=device, dtype=dtype, generator=generator)
            ).requires_grad_(),
            "beta": torch.rand(
                1, 4, 1, device=device, dtype=dtype, generator=generator
            ).requires_grad_(),
            "log_decay": (
                -0.1
                * torch.rand(1, 4, 1, device=device, dtype=dtype, generator=generator)
            ).requires_grad_(),
        }

    reference_inputs = make_inputs()
    native_inputs = {
        name: value.detach().clone().requires_grad_(value.is_floating_point())
        if name not in {"read_indices", "write_indices"}
        else value
        for name, value in reference_inputs.items()
    }
    reference = compile_mixer(
        spec,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.REFERENCE,
        dtype=dtype_name,
    ).execute(**reference_inputs)
    native = compile_mixer(
        spec,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.NATIVE,
        dtype=dtype_name,
    ).execute(**native_inputs)
    output_cotangent = torch.randn(
        reference.output.shape, device=device, dtype=torch.float32, generator=generator
    )
    state_cotangent = torch.randn(
        reference.final_state.shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    def loss(result):
        return (result.output.float() * output_cotangent).mean() + (
            result.final_state.float() * state_cotangent
        ).mean()

    loss(reference).backward()
    loss(native).backward()
    value_atol, value_rtol = (2e-2, 2e-2) if dtype is torch.bfloat16 else (2e-4, 2e-4)
    gradient_atol, gradient_rtol = (
        (3e-2, 3e-2) if dtype is torch.bfloat16 else (3e-5, 3e-4)
    )
    torch.testing.assert_close(
        native.output, reference.output, atol=value_atol, rtol=value_rtol
    )
    torch.testing.assert_close(
        native.final_state, reference.final_state, atol=value_atol, rtol=value_rtol
    )
    for name in (
        "memory",
        "read_weights",
        "write_weights",
        "values",
        "beta",
        "log_decay",
    ):
        torch.testing.assert_close(
            native_inputs[name].grad,
            reference_inputs[name].grad,
            atol=gradient_atol,
            rtol=gradient_rtol,
            msg=f"K3 {name} VJP differs for overlapping routes",
        )


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_k3_native_anchor_executes_when_cuda_contract_is_supported(dtype_name):
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("native K3 requires CUDA")
    from urm.backends.triton.sparse_state.backend import TritonSparseStateMixerBackend
    from urm.compiler.semantic import (
        DType,
        SparseReadTiming,
        SparseStateExecutionMode,
        SparseStateMixerSpec,
        SparseStateOperation,
    )

    native_spec = SparseStateMixerSpec(
        parallel=1,
        sequence=2,
        slots_per_partition=64,
        value_dim=8,
        writes=1,
        reads=1,
        dtype=DType(dtype_name),
        operation=SparseStateOperation.UPDATE,
        read_timing=SparseReadTiming.AFTER_UPDATE,
        mode=SparseStateExecutionMode.TRAINING,
    )
    status = TritonSparseStateMixerBackend.support_status(native_spec)
    if not status.supported:
        pytest.skip(f"native K3 is unavailable on this CUDA target: {status}")
    device = "cuda"
    dtype = getattr(torch, dtype_name)
    memory = torch.zeros(1, 64, 8, device=device, dtype=dtype, requires_grad=True)
    read_weights = torch.ones(1, 2, 1, device=device, dtype=dtype, requires_grad=True)
    write_weights = torch.ones(1, 2, 1, device=device, dtype=dtype, requires_grad=True)
    values = torch.ones(1, 2, 8, device=device, dtype=dtype, requires_grad=True)
    beta = torch.ones(1, 2, 1, device=device, dtype=dtype, requires_grad=True)
    log_decay = torch.zeros(1, 2, 1, device=device, dtype=dtype, requires_grad=True)
    result = compile_mixer(
        sparse_delta_spec(),
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.NATIVE,
        dtype=dtype_name,
    ).execute(
        memory=memory,
        read_indices=torch.tensor([[[0], [1]]], dtype=torch.int32, device=device),
        read_weights=read_weights,
        write_indices=torch.tensor([[[0], [1]]], dtype=torch.int32, device=device),
        write_weights=write_weights,
        values=values,
        beta=beta,
        log_decay=log_decay,
    )
    assert result.output.shape == (1, 2, 8)
    torch.testing.assert_close(result.output, torch.ones_like(result.output))
    torch.testing.assert_close(
        result.final_state[0, :2], torch.ones_like(result.final_state[0, :2])
    )
    (result.output.square().mean() + result.final_state.square().mean()).backward()
    for tensor in (memory, read_weights, write_weights, values, beta, log_decay):
        assert tensor.grad is not None
    assert result.metadata["execution"] == "urm_native_triton"
    assert result.metadata["urm_compiler_verified"] is True
    assert result.metadata["runtime_compiler_binding"] == "cached_semantic_shape"
    assert result.metadata["runtime_binding_cache_size"] >= 1
