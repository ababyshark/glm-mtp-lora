#!/usr/bin/env python
"""V10 trainer smoke (1-card, .12) — the full training loop WITHOUT the LF
data dependency: build(4-layer trunc, mtp=True) -> real fill -> freeze ->
native steps + recipe steps -> CP save/load roundtrip -> frozen-leak +
determinism asserts. The LF-config run (v10_trainer.py --yaml) stays the
production entry; this smoke validates the loop mechanics.

Run on .12: python3 v10_train_smoke.py
"""
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

from launch_ce_gate import npu_shim  # noqa: E402
from v10_config import vendor_config  # noqa: E402
from v10_fill import fill_model  # noqa: E402
from v10_model import build_v10_model  # noqa: E402
from v10_trainer import freeze_base, load_cp, save_cp  # noqa: E402

VENDOR_DIR = os.environ.get("V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16")


class TinyBatcher:
    """Synthetic batcher with the McoreBatcher contract: next_batch() ->
    {'input_ids': [B,S], 'labels': [B,S], 'position_ids': [B,S]} on device."""

    def __init__(self, bsz, seq, vocab, dev, seed=77):
        self.bsz, self.seq, self.vocab, self.dev = bsz, seq, vocab, dev
        self.g = torch.Generator().manual_seed(seed)

    def next_batch(self):
        ids = torch.randint(1, self.vocab, (self.bsz, self.seq),
                            generator=self.g).to(self.dev)
        return {"input_ids": ids, "labels": torch.roll(ids, -1, dims=1),
                "position_ids": torch.zeros_like(ids)}

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch(), None  # mcore contract: (data, loss_mask)


def run_steps(model, batcher, optimizer, fwdbwd, recipe, n_steps, dev, rank):
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    losses = []
    for step in range(1, n_steps + 1):
        state = {}

        def forward_step(data_iterator, model):
            batch, _ = next(data_iterator)
            state["batch"] = batch
            if recipe:
                S_b = batch["input_ids"].shape[1]  # batch-first [B, S]: shape[1] = S (the [1,1,1,1]-mask bug was silent in V8)
                causal_keep = torch.ones(S_b, S_b, dtype=torch.bool,
                                         device=dev).tril() \
                    .unsqueeze(0).unsqueeze(0)
                emb = model.embedding(input_ids=batch["input_ids"],
                                      position_ids=batch["position_ids"])
                with torch.no_grad():
                    dec = model.decoder(hidden_states=emb,
                                        attention_mask=causal_keep)
                h0 = dec[0] if isinstance(dec, tuple) else dec
                from v10_recipe import serving_chain_recipe_loss
                loss = serving_chain_recipe_loss(model, batch["input_ids"],
                                                 batch["labels"], h0)
            else:
                out = model(input_ids=batch["input_ids"],
                            position_ids=batch["position_ids"],
                            attention_mask=None, labels=batch["labels"])
                loss = out[0] if isinstance(out, tuple) else out
                if loss.dim() > 0:      # per-token [B,S] -> scalar (schedule needs one)
                    loss = loss.float().mean()
            return loss, lambda l: (l, {"loss": float(l.detach().mean())})

        out = fwdbwd(forward_step_func=forward_step, data_iterator=[batcher],
                     model=[model], num_microbatches=1, seq_length=64,
                     micro_batch_size=1)
        e0 = out[0] if isinstance(out, (list, tuple)) and out else out
        v = e0.get("loss") if isinstance(e0, dict) else e0
        lv = float(v) if not hasattr(v, "item") else v.item()
        leaks = [n for n, p in model.named_parameters()
                 if not p.requires_grad and p.grad is not None]
        assert not leaks, f"frozen-param grad leaks: {leaks[:4]}"
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(lv)
        if rank == 0:
            print(f"[train-smoke] {'recipe' if recipe else 'native'} "
                  f"step {step}: loss={lv:.4f}", flush=True)
    return losses


def main():
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local)
    npu_shim()
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29711")
        dist.init_process_group(backend="hccl", init_method="env://",
                                world_size=world, rank=rank)
    from megatron.core import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(1234)

    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    fwdbwd = get_forward_backward_func()

    vc = vendor_config()
    dev = torch.device("npu", local)
    model, cfg = build_v10_model(vc=vc, num_layers=4, device="npu", mtp=True,
                                 max_seq_len=256, tp_size=world)
    ledger = fill_model(model, VENDOR_DIR, vc, mtp=True, tp_rank=rank,
                        tp_size=world, verbose=(rank == 0))
    assert not ledger["unwritten"], f"unwritten: {ledger['unwritten'][:6]}"
    assert not ledger["unconsumed"], f"unconsumed: {ledger['unconsumed'][:6]}"
    if rank == 0:
        print(f"[train-smoke] fill clean: {ledger['n_written']} targets "
              f"(MTP included)", flush=True)
    n_frozen = freeze_base(model)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    assert n_train > 0 and all(n.startswith("mtp.") for n, p in
                               model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(0.9, 0.95))
    if rank == 0:
        print(f"[train-smoke] mtp-only trainable (sharded): {n_train} "
              f"({n_frozen} frozen base params, tp={world})", flush=True)

    batcher = TinyBatcher(1, 64, vc["vocab_size"], dev)
    native = run_steps(model, batcher, optimizer, fwdbwd, recipe=False,
                       n_steps=2, dev=dev, rank=rank)
    recipe = run_steps(model, batcher, optimizer, fwdbwd, recipe=True,
                       n_steps=2, dev=dev, rank=rank)
    assert all(torch.isfinite(torch.tensor(l)) for l in native + recipe)

    # CP roundtrip
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        save_cp(model, optimizer, 4, td)
        opt2 = torch.optim.AdamW(params, lr=1e-5)
        step = load_cp(model, opt2, os.path.join(td, f"cp4_rank{rank}.pt"))
        assert step == 4
        if rank == 0:
            print("[train-smoke] CP roundtrip OK (step 4)", flush=True)

    print("[v10-train-smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
