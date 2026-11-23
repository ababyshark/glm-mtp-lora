#!/usr/bin/env python
"""V10 recipe/trainer unit tests (tiny CPU) — freeze contract, recipe loss,
grad isolation, CP round-trip."""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10 = os.path.dirname(HERE)
for p in (V10, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_v10_build import TINY_VC, cpu_env, _seed_params  # noqa: E402,F401


def _npu_ok():
    """torch_npu PATCHES torch.optim's device dispatch (utils/_optim.py) —
    any AdamW.step() touches npu.current_device() and dies on a broken-ACL
    host (dev box 507008, in a forked-subprocess npu init). torch_npu absent
    = pure CPU, fine. is_available() swallows the init error gracefully."""
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return True  # torch_npu not installed -> clean CPU path


_NPU_OK = _npu_ok()


def _mcore_mtp_uses_mtp_model_layer() -> bool:
    """Version probe — the MTP-dependent recipe/trainer tests target mcore
    0.18.2 (MultiTokenPredictionLayerSubmodules(mtp_model_layer=...)); older
    mcores (0.12.x) name it transformer_layer. Skip cleanly there."""
    try:
        import inspect

        from megatron.core.transformer.multi_token_prediction import (
            MultiTokenPredictionLayerSubmodules)
        return "mtp_model_layer" in inspect.signature(
            MultiTokenPredictionLayerSubmodules.__init__).parameters
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _mcore_mtp_uses_mtp_model_layer(),
    reason="mcore lacks MultiTokenPredictionLayerSubmodules(mtp_model_layer=...) "
           "(0.18.2 API); run on the .38 venv_v7")


def _tiny_mtp(cpu_env):
    from v10_model import build_v10_model
    model, cfg = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=True)
    _seed_params(model, seed=909)
    return model, cfg


def test_freeze_contract_lora_plus_mtp(cpu_env):
    """Contract 2026-11-20: LoRA adapters on the base decoder + the full mtp
    subtree are trainable; base weights / output_layer / embedding stay
    frozen (merged at export)."""
    from v10_trainer import build_trainable
    model, _ = _tiny_mtp(cpu_env)
    handled, n_frozen = build_trainable(model)
    req = {n for n, p in model.named_parameters() if p.requires_grad}
    assert req and all(n.startswith("mtp.") or ".lora_" in n for n in req), \
        f"whitelist violation: {sorted(req)[:6]}"
    names = dict(model.named_parameters())
    lora = [n for n in names if ".lora_" in n]
    assert lora, "no lora params injected"
    assert any("self_attention.linear_q_proj.lora_A" in n for n in lora)
    # routed experts excluded (docstring contract)
    assert not any(".mlp.experts." in n for n in lora), \
        f"expert lora leaked: {[n for n in lora if '.mlp.experts.' in n][:3]}"
    # frozen boundaries
    P = names
    assert not P["output_layer.weight"].requires_grad
    assert not P["embedding.word_embeddings.weight"].requires_grad
    assert not P["decoder.layers.0.self_attention.linear_q_proj.weight"].requires_grad
    # zero-init B => step-0 behavior EXACTLY the filled model
    for n in lora:
        if n.endswith(".lora_B"):
            assert (P[n].data == 0).all(), f"B not zero-initialized: {n}"


def test_recipe_loss_finite_and_grad_isolation(cpu_env):
    from v10_recipe import serving_chain_recipe_loss
    from v10_trainer import build_trainable as freeze_base
    model, cfg = _tiny_mtp(cpu_env)
    freeze_base(model)
    B, S = 1, 16
    ids = torch.randint(1, TINY_VC["vocab_size"], (B, S))
    labels = torch.roll(ids, -1, dims=1)
    causal_keep = torch.ones(S, S, dtype=torch.bool).tril() \
        .unsqueeze(0).unsqueeze(0)
    emb = model.embedding(input_ids=ids, position_ids=None)
    with torch.no_grad():
        dec = model.decoder(hidden_states=emb, attention_mask=causal_keep)
    h0 = dec[0] if isinstance(dec, tuple) else dec
    loss = serving_chain_recipe_loss(model, ids, labels, h0)
    assert torch.isfinite(loss), "recipe loss non-finite"
    loss.backward()
    g = {n for n, p in model.named_parameters() if p.grad is not None}
    assert g and all(n.startswith("mtp.") for n in g), \
        f"grad escaped the mtp subtree: {sorted(g)[:5]}"
    # decomp attributes exist (GLM53_LOSS_DECOMP consumers)
    assert hasattr(model, "_last_kl")


def test_recipe_kl_agree_stream(cpu_env):
    """GLM53_MTP_DISTILL_WEIGHT>0 builds the verifier stream; agree requires
    it (branch-1 :1765 invariant)."""
    from v10_recipe import serving_chain_recipe_loss
    from v10_trainer import build_trainable as freeze_base
    os.environ["GLM53_MTP_DISTILL_WEIGHT"] = "2.33"
    os.environ["GLM53_AGREE_CE_WEIGHT"] = "0.5"
    os.environ["GLM53_ACC_MARGIN_WEIGHT"] = "0.3"
    try:
        model, _ = _tiny_mtp(cpu_env)
        freeze_base(model)
        B, S = 1, 16
        ids = torch.randint(1, TINY_VC["vocab_size"], (B, S))
        labels = torch.roll(ids, -1, dims=1)
        causal_keep = torch.ones(S, S, dtype=torch.bool).tril() \
            .unsqueeze(0).unsqueeze(0)
        emb = model.embedding(input_ids=ids, position_ids=None)
        with torch.no_grad():
            dec = model.decoder(hidden_states=emb, attention_mask=causal_keep)
        h0 = dec[0] if isinstance(dec, tuple) else dec
        loss = serving_chain_recipe_loss(model, ids, labels, h0)
        assert torch.isfinite(loss)
        loss.backward()
        assert model._last_kl is not None and model._last_agree is not None
    finally:
        for k in ("GLM53_MTP_DISTILL_WEIGHT", "GLM53_AGREE_CE_WEIGHT",
                  "GLM53_ACC_MARGIN_WEIGHT"):
            os.environ.pop(k, None)


def test_recipe_agree_requires_distill(cpu_env):
    """aw>0 with dw<=0 must raise (silent skip would change recipe semantics)."""
    from v10_recipe import serving_chain_recipe_loss
    from v10_trainer import build_trainable as freeze_base
    os.environ["GLM53_AGREE_CE_WEIGHT"] = "0.5"
    os.environ.pop("GLM53_MTP_DISTILL_WEIGHT", None)
    try:
        model, _ = _tiny_mtp(cpu_env)
        freeze_base(model)
        ids = torch.randint(1, TINY_VC["vocab_size"], (1, 8))
        labels = torch.roll(ids, -1, dims=1)
        with torch.no_grad():
            dec = model.decoder(hidden_states=model.embedding(
                input_ids=ids, position_ids=None),
                attention_mask=torch.ones(8, 8, dtype=torch.bool).tril()
                .unsqueeze(0).unsqueeze(0))
        h0 = dec[0] if isinstance(dec, tuple) else dec
        with pytest.raises(RuntimeError):
            serving_chain_recipe_loss(model, ids, labels, h0)
    finally:
        os.environ.pop("GLM53_AGREE_CE_WEIGHT", None)


@pytest.mark.skipif(not _NPU_OK, reason='torch_npu present but ACL broken (AdamW dispatch patch); run on a healthy-NPU or pure-CPU host')
def test_cp_roundtrip(cpu_env):
    from v10_trainer import build_trainable as freeze_base, load_cp, save_cp, _opt_param_names
    model, _ = _tiny_mtp(cpu_env)
    freeze_base(model)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    # perturb + step so the optimizer state is non-trivial
    for p in params:
        p.grad = torch.randn_like(p) * 0.01
    opt.step()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        save_cp(model, opt, 3, td)
        model2, _ = build_none(cpu_env)
        freeze_base(model2)
        params2 = [p for n, p in model2.named_parameters() if p.requires_grad]
        opt2 = torch.optim.AdamW(params2, lr=1e-3)
        step = load_cp(model2, opt2, os.path.join(td, "cp3_rank0.pt"))
        assert step == 3
        pa = dict(model.named_parameters())
        pb = dict(model2.named_parameters())
        for n, p in pa.items():
            if p.requires_grad:
                assert torch.equal(p.data, pb[n].data), f"CP param mismatch {n}"


def build_none(cpu_env):
    from v10_model import build_v10_model
    return build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
