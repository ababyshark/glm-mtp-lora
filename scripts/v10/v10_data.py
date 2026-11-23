#!/usr/bin/env python
"""LF data pipeline → mcore batches (iron rules rule 0: LF owns the data,
megatron owns the training core; the glue is THIS module only).

VERBATIM from the V8 line's scripts/v8/v8_data.py (V10 self-containment: the
V8 dir is not vendored here) — imports/uses unchanged, provenance only.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# LF is a git tree (iron rules rule 0: LlamaFactory, not pip) — self-contained
# path injection (the run_joint_lf PYTHONPATH contract, iron rules §3.1)
LF_SRC = os.environ.get("LF_SRC", "/mnt/share_space/l00840580/LlamaFactory/src")
if LF_SRC not in sys.path:
    sys.path.insert(0, LF_SRC)
# the branch-1 validated combo (LF git tree + transformers 5.14.1) rides on
# LF's version-check opt-out — the V7.0 judgment training runs exactly this
os.environ.setdefault("DISABLE_VERSION_CHECK", "1")


def load_lf_data(cfg_path, max_samples=None):
    """LF data pipeline → (tokenizer, template, tokenized train_dataset).

    cfg_path: the V8 yaml — ONLY the LF data keys are read (dataset,
    dataset_dir, template, cutoff_len, preprocessing_num_workers)."""
    import yaml
    cfg = yaml.safe_load(open(cfg_path))

    from llamafactory.hparams import DataArguments, ModelArguments
    model_args = ModelArguments(model_name_or_path=cfg["model_name_or_path"],
                                trust_remote_code=True)
    data_args = DataArguments(
        dataset=cfg["dataset"],
        dataset_dir=cfg["dataset_dir"],
        template=cfg["template"],
        cutoff_len=cfg["cutoff_len"],
        preprocessing_num_workers=cfg.get("preprocessing_num_workers", 8),
    )
    from transformers import Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(output_dir="/tmp/v8_data")

    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.model.loader import load_tokenizer
    tok_mod = load_tokenizer(model_args)
    tokenizer = tok_mod["tokenizer"] if isinstance(tok_mod, dict) else tok_mod.tokenizer
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args,
                                 training_args, "sft", tokenizer)
    train_ds = dataset_module["train_dataset"]
    if max_samples is not None:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
    return tokenizer, template, train_ds


class McoreBatcher:
    """Tokenized LF samples → mcore seq-first batches [S, B] on device.

    B samples per microbatch (pad/truncate to cutoff_len); labels = input_ids
    (the model shifts internally — Phase 2 native full-LM MTP loss)."""

    def __init__(self, dataset, batch_size, seq_len, device, pad_id=0):
        self.ds = dataset
        self.B = batch_size
        self.S = seq_len
        self.device = device
        self.pad_id = pad_id
        self._order = None
        self._pos = 0
        self.epoch = 0

    def _reshuffle(self):
        g = torch.Generator().manual_seed(42 + self.epoch)
        self._order = torch.randperm(len(self.ds), generator=g).tolist()
        self._pos = 0
        self.epoch += 1

    def next_batch(self):
        if self._order is None or self._pos + self.B > len(self.ds):
            self._reshuffle()
        idxs = self._order[self._pos:self._pos + self.B]
        self._pos += self.B
        ids_list = []
        for i in idxs:
            row = self.ds[int(i)]
            ids = row["input_ids"] if "input_ids" in row else row
            ids_list.append(torch.tensor(ids[:self.S], dtype=torch.long))
        B = len(ids_list)
        S = max(t.shape[0] for t in ids_list)
        # LAYOUT TRUTH (parity + embedding-source converged): GPTModel.forward
        # and model.embedding take BATCH-FIRST [B, S] input_ids; the embedding
        # transposes to [S, B, H] for the seq-first decoder, and the logits
        # come back [B, S, V]. (The V7 parity fed [1, 64] = [B,S] contextual —
        # argmax 73% against HF is only possible with true 64-token contexts.)
        input_ids = torch.full((B, S), self.pad_id, dtype=torch.long)
        for b, t in enumerate(ids_list):
            input_ids[b, :t.shape[0]] = t
        input_ids = input_ids.to(self.device)
        labels = input_ids.clone()
        position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, S).contiguous()
        return {"input_ids": input_ids, "labels": labels, "position_ids": position_ids}

    # mcore schedule contract: next(data_iterator) → (data, loss_mask)
    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch(), None
